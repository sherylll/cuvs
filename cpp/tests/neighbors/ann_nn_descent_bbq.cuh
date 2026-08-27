/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ann_nn_descent.cuh"

#include <cuvs/preprocessing/quantize/bbq.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <optional>
#include <sstream>
#include <utility>
#include <vector>

#include <raft/core/logger.hpp>

namespace cuvs::neighbors::nn_descent {
// Host-side Lucene OptimizedScalarQuantizer
namespace cpu_bbq {

using cuvs::preprocessing::quantize::bbq::bbq_code_layout;

constexpr float kMinimumMseGrid[8][2] = {{-0.798f, 0.798f},
                                         {-1.493f, 1.493f},
                                         {-2.051f, 2.051f},
                                         {-2.514f, 2.514f},
                                         {-2.916f, 2.916f},
                                         {-3.278f, 3.278f},
                                         {-3.611f, 3.611f},
                                         {-3.922f, 3.922f}};

constexpr float kDefaultLambda = 0.1f;
constexpr int kDefaultIters    = 5;

inline long round(double x) { return static_cast<long>(std::floor(x + 0.5)); }

inline double clamp(double x, double a, double b) { return std::min(std::max(x, a), b); }

inline double loss(
  const std::vector<float>& vector, const float interval[2], int points, float norm2, float lambda)
{
  const double a        = interval[0];
  const double b        = interval[1];
  const double step     = (b - a) / (points - 1.0);
  const double step_inv = 1.0 / step;
  double xe             = 0.0;
  double e              = 0.0;
  for (double xi : vector) {
    const double xiq = a + step * static_cast<double>(round((clamp(xi, a, b) - a) * step_inv));
    xe += xi * (xi - xiq);
    e += (xi - xiq) * (xi - xiq);
  }
  return (1.0 - lambda) * xe * xe / norm2 + lambda * e;
}

inline void optimize_intervals(float interval[2],
                               const std::vector<float>& vector,
                               float norm2,
                               int points,
                               float lambda = kDefaultLambda,
                               int iters    = kDefaultIters)
{
  double initial_loss = loss(vector, interval, points, norm2, lambda);
  const float scale   = (1.0f - lambda) / norm2;
  if (!std::isfinite(scale)) { return; }
  for (int i = 0; i < iters; ++i) {
    const float a        = interval[0];
    const float b        = interval[1];
    const float step_inv = (points - 1.0f) / (b - a);
    double daa = 0.0, dab = 0.0, dbb = 0.0, dax = 0.0, dbx = 0.0;
    for (float xi : vector) {
      const float k =
        static_cast<float>(round(static_cast<float>((clamp(xi, a, b) - a) * step_inv)));
      const float s = k / (points - 1);
      daa += (1.0 - s) * (1.0 - s);
      dab += (1.0 - s) * s;
      dbb += s * s;
      dax += xi * (1.0 - s);
      dbx += xi * s;
    }
    const double m0  = scale * dax * dax + lambda * daa;
    const double m1  = scale * dax * dbx + lambda * dab;
    const double m2  = scale * dbx * dbx + lambda * dbb;
    const double det = m0 * m2 - m1 * m1;
    if (det == 0) { return; }
    const float a_opt = static_cast<float>((m2 * dax - m1 * dbx) / det);
    const float b_opt = static_cast<float>((m0 * dbx - m1 * dax) / det);
    if (std::abs(interval[0] - a_opt) < 1e-8 && std::abs(interval[1] - b_opt) < 1e-8) { return; }
    float new_interval[2] = {a_opt, b_opt};
    const double new_loss = loss(vector, new_interval, points, norm2, lambda);
    if (new_loss > initial_loss) { return; }
    interval[0]  = a_opt;
    interval[1]  = b_opt;
    initial_loss = new_loss;
  }
}

struct row_result {
  float lower_interval;
  float upper_interval;
  float additional_correction;
  int32_t quantized_component_sum;
};

inline row_result scalar_quantize(std::vector<float>& vector,
                                  std::vector<uint8_t>& destination,
                                  uint8_t bits,
                                  const float* centroid,
                                  bool euclidean)
{
  const int n        = static_cast<int>(vector.size());
  const int points   = 1 << bits;
  double vec_mean    = 0.0;
  double vec_var     = 0.0;
  float norm2        = 0.0f;
  float centroid_dot = 0.0f;
  float min          = FLT_MAX;
  float max          = -FLT_MAX;
  for (int i = 0; i < n; ++i) {
    if (!euclidean) { centroid_dot += vector[i] * centroid[i]; }
    vector[i] = vector[i] - centroid[i];
    min       = std::min(min, vector[i]);
    max       = std::max(max, vector[i]);
    norm2 += vector[i] * vector[i];
    const double delta = vector[i] - vec_mean;
    vec_mean += delta / (i + 1);
    vec_var += delta * (vector[i] - vec_mean);
  }
  vec_var /= n;
  const double vec_std = std::sqrt(vec_var);

  float interval[2];
  interval[0] =
    static_cast<float>(clamp(kMinimumMseGrid[bits - 1][0] * vec_std + vec_mean, min, max));
  interval[1] =
    static_cast<float>(clamp(kMinimumMseGrid[bits - 1][1] * vec_std + vec_mean, min, max));
  optimize_intervals(interval, vector, norm2, points);

  const float n_steps = static_cast<float>((1 << bits) - 1);
  const float a       = interval[0];
  const float b       = interval[1];
  const float step    = (b - a) / n_steps;
  int sum_query       = 0;
  for (int h = 0; h < n; ++h) {
    const float xi       = static_cast<float>(clamp(vector[h], a, b));
    const int assignment = static_cast<int>(round((xi - a) / step));
    sum_query += assignment;
    destination[h] = static_cast<uint8_t>(assignment);
  }
  return row_result{interval[0], interval[1], euclidean ? norm2 : centroid_dot, sum_query};
}

inline size_t encoded_row_length(size_t dim, uint32_t bits, bbq_code_layout layout)
{
  switch (layout) {
    case bbq_code_layout::single_bit: return (dim * bits + 7) / 8;
    case bbq_code_layout::dibit: return bits * ((dim + 7) / 8);
    case bbq_code_layout::packed_nibble: return (dim + 1) / 2;
    case bbq_code_layout::seven_bit: return dim;
    case bbq_code_layout::unsigned_byte: return dim;
    case bbq_code_layout::transpose_half_byte: return 4 * ((dim + 7) / 8);
  }
  return 0;
}

// Packs one-byte-per-component codes into single_bit / dibit / packed_nibble /
// transpose_half_byte (or leaves unpacked). Matches Lucene packAsBinary,
// packNibbles, transposeDibit, transposeHalfByte.
inline std::vector<uint8_t> pack_codes(const std::vector<uint8_t>& unpacked,
                                       size_t n_rows,
                                       size_t dim,
                                       uint32_t bits,
                                       bbq_code_layout layout)
{
  const size_t row_length = encoded_row_length(dim, bits, layout);
  if (layout == bbq_code_layout::unsigned_byte || layout == bbq_code_layout::seven_bit) {
    return unpacked;
  }

  std::vector<uint8_t> packed(n_rows * row_length, 0);
  for (size_t row = 0; row < n_rows; ++row) {
    auto* output      = packed.data() + row * row_length;
    const auto* input = unpacked.data() + row * dim;
    if (layout == bbq_code_layout::packed_nibble) {
      // Lucene OffHeapScalarQuantizedVectorValues.packNibbles
      const size_t half = dim / 2;
      for (size_t i = 0; i < half; ++i) {
        output[i] = static_cast<uint8_t>((input[i] << 4) | (input[half + i] & 0x0f));
      }
      continue;
    }
    for (size_t d = 0; d < dim; ++d) {
      const uint8_t code = input[d];
      if (layout == bbq_code_layout::single_bit) {
        for (uint32_t bit = 0; bit < bits; ++bit) {
          const size_t position = d * bits + bit;
          output[position / 8] |=
            static_cast<uint8_t>(((code >> (bits - 1 - bit)) & 1u) << (7 - position % 8));
        }
      } else {
        // dibit / transpose_half_byte bit-planes (LSB plane first)
        const size_t stripe = (dim + 7) / 8;
        for (uint32_t bit = 0; bit < bits; ++bit) {
          output[bit * stripe + d / 8] |= static_cast<uint8_t>(((code >> bit) & 1u) << (7 - d % 8));
        }
      }
    }
  }
  return packed;
}

inline cuvs::neighbors::host_bbq_dataset<int64_t>::owning_storage_type quantize(
  const std::vector<float>& data,
  int64_t n_rows,
  int64_t dim,
  uint8_t bits,
  cuvs::distance::DistanceType metric,
  bbq_code_layout layout = bbq_code_layout::unsigned_byte)
{
  const bool euclidean = metric == cuvs::distance::DistanceType::L2Expanded ||
                         metric == cuvs::distance::DistanceType::L2SqrtExpanded;

  auto centroid          = raft::make_host_vector<float, int64_t>(dim);
  float centroid_norm_sq = 0.0f;
  std::fill_n(centroid.data_handle(), static_cast<size_t>(dim), 0.0f);
  for (int64_t i = 0; i < n_rows; ++i) {
    for (int64_t d = 0; d < dim; ++d) {
      centroid(d) += data[i * dim + d];
    }
  }
  for (int64_t d = 0; d < dim; ++d) {
    centroid(d) /= static_cast<float>(n_rows);
    centroid_norm_sq += centroid(d) * centroid(d);
  }

  std::vector<uint8_t> unpacked(static_cast<size_t>(n_rows * dim));
  auto lower_intervals          = raft::make_host_vector<float, int64_t>(n_rows);
  auto upper_intervals          = raft::make_host_vector<float, int64_t>(n_rows);
  auto additional_corrections   = raft::make_host_vector<float, int64_t>(n_rows);
  auto quantized_component_sums = raft::make_host_vector<int32_t, int64_t>(n_rows);

  for (int64_t i = 0; i < n_rows; ++i) {
    std::vector<float> row(data.begin() + i * dim, data.begin() + (i + 1) * dim);
    std::vector<uint8_t> codes(dim);
    const auto result = scalar_quantize(row, codes, bits, centroid.data_handle(), euclidean);
    std::copy(codes.begin(), codes.end(), unpacked.begin() + i * dim);
    lower_intervals(i)          = result.lower_interval;
    upper_intervals(i)          = result.upper_interval;
    additional_corrections(i)   = result.additional_correction;
    quantized_component_sums(i) = result.quantized_component_sum;
  }

  auto packed =
    pack_codes(unpacked, static_cast<size_t>(n_rows), static_cast<size_t>(dim), bits, layout);
  auto codes = raft::make_host_matrix<uint8_t, int64_t>(
    n_rows, static_cast<int64_t>(encoded_row_length(dim, bits, layout)));
  std::copy(packed.begin(), packed.end(), codes.data_handle());

  return cuvs::neighbors::host_bbq_dataset<int64_t>::owning_storage_type{
    std::move(codes),
    std::move(lower_intervals),
    std::move(upper_intervals),
    std::move(additional_corrections),
    std::move(quantized_component_sums),
    std::move(centroid),
    static_cast<uint32_t>(bits),
    layout,
    metric,
    centroid_norm_sq};
}

}  // namespace cpu_bbq

template <typename IdxT>
auto copy_bbq_owning_storage_host_to_device(
  raft::resources const& res,
  typename cuvs::neighbors::host_bbq_dataset<IdxT>::owning_storage_type const& host_storage) ->
  typename cuvs::neighbors::device_bbq_dataset<IdxT>::owning_storage_type
{
  auto stream = raft::resource::get_cuda_stream(res);
  auto codes  = raft::make_device_matrix<uint8_t, IdxT>(
    res, host_storage.codes.extent(0), host_storage.codes.extent(1));
  auto lower_intervals =
    raft::make_device_vector<float, IdxT>(res, host_storage.lower_intervals.extent(0));
  auto upper_intervals =
    raft::make_device_vector<float, IdxT>(res, host_storage.upper_intervals.extent(0));
  auto additional_corrections =
    raft::make_device_vector<float, IdxT>(res, host_storage.additional_corrections.extent(0));
  auto quantized_component_sums =
    raft::make_device_vector<int32_t, IdxT>(res, host_storage.quantized_component_sums.extent(0));
  auto centroid = raft::make_device_vector<float, IdxT>(res, host_storage.centroid.extent(0));

  raft::copy(codes.data_handle(), host_storage.codes.data_handle(), codes.size(), stream);
  raft::copy(lower_intervals.data_handle(),
             host_storage.lower_intervals.data_handle(),
             lower_intervals.size(),
             stream);
  raft::copy(upper_intervals.data_handle(),
             host_storage.upper_intervals.data_handle(),
             upper_intervals.size(),
             stream);
  raft::copy(additional_corrections.data_handle(),
             host_storage.additional_corrections.data_handle(),
             additional_corrections.size(),
             stream);
  raft::copy(quantized_component_sums.data_handle(),
             host_storage.quantized_component_sums.data_handle(),
             quantized_component_sums.size(),
             stream);
  raft::copy(centroid.data_handle(), host_storage.centroid.data_handle(), centroid.size(), stream);

  return {std::move(codes),
          std::move(lower_intervals),
          std::move(upper_intervals),
          std::move(additional_corrections),
          std::move(quantized_component_sums),
          std::move(centroid),
          host_storage.bits,
          host_storage.layout,
          host_storage.metric,
          host_storage.centroid_norm_sq};
}

template <typename IdxT>
auto make_device_bbq_dataset(raft::resources const& res,
                             cuvs::neighbors::host_bbq_dataset<IdxT> const& host)
  -> cuvs::neighbors::device_bbq_dataset<IdxT>
{
  RAFT_EXPECTS(host.quantizers.size() != 0, "host BBQ dataset has no storage");
  cuvs::neighbors::device_bbq_dataset<IdxT> device{
    copy_bbq_owning_storage_host_to_device<IdxT>(res, host.quantizers[0])};
  for (std::size_t i = 1; i < host.quantizers.size(); ++i) {
    device.add_quantizer(copy_bbq_owning_storage_host_to_device<IdxT>(res, host.quantizers[i]));
  }
  return device;
}

// CUDA-event elapsed time around @p fn on @p stream. Because the stop event is
// recorded only after @p fn returns, stream-idle gaps from host work inside NN-Descent
// are included — so this is effectively a wall-clock bracket of the build.
template <typename Fn>
float time_cuda_ms(rmm::cuda_stream_view stream, Fn&& fn)
{
  cudaEvent_t start{};
  cudaEvent_t stop{};
  RAFT_CUDA_TRY(cudaEventCreate(&start));
  RAFT_CUDA_TRY(cudaEventCreate(&stop));
  RAFT_CUDA_TRY(cudaEventRecord(start, stream));
  std::forward<Fn>(fn)();
  RAFT_CUDA_TRY(cudaEventRecord(stop, stream));
  RAFT_CUDA_TRY(cudaEventSynchronize(stop));
  float ms = 0.0f;
  RAFT_CUDA_TRY(cudaEventElapsedTime(&ms, start, stop));
  RAFT_CUDA_TRY(cudaEventDestroy(start));
  RAFT_CUDA_TRY(cudaEventDestroy(stop));
  return ms;
}

struct AnnNNDescentBbqInputs : AnnNNDescentInputs {
  uint8_t bits;
  cuvs::preprocessing::quantize::bbq::bbq_code_layout layout;
  std::optional<uint8_t> second_dataset_bits;
};

inline ::std::ostream& operator<<(::std::ostream& os, const AnnNNDescentBbqInputs& p)
{
  os << "dataset shape=" << p.n_rows << "x" << p.dim << ", graph_degree=" << p.graph_degree
     << ", metric="
     << cuvs::neighbors::print_metric{static_cast<cuvs::distance::DistanceType>((int)p.metric)}
     << (p.host_dataset ? ", host" : ", device") << ", bits=" << static_cast<int>(p.bits)
     << ", layout=" << static_cast<int>(p.layout) << ", second_dataset_bits="
     << (p.second_dataset_bits.has_value() ? static_cast<int>(p.second_dataset_bits.value()) : 0)
     << std::endl;
  return os;
}

class AnnNNDescentBbqTest : public ::testing::TestWithParam<AnnNNDescentBbqInputs> {
 public:
  AnnNNDescentBbqTest()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<AnnNNDescentBbqInputs>::GetParam()),
      database(raft::make_device_matrix<float, int64_t>(handle_, ps.n_rows, ps.dim))
  {
  }

 protected:
  void testNNDescent()
  {
    if (ps.second_dataset_bits.has_value()) {
      if (ps.bits > 4 ||
          (ps.bits == 4 &&
           ps.layout == cuvs::preprocessing::quantize::bbq::bbq_code_layout::packed_nibble) ||
          ps.bits == ps.second_dataset_bits.value() || ps.bits == 1) {
        GTEST_SKIP() << "Second dataset is N/A: bits=" << static_cast<int>(ps.bits)
                     << ", layout=" << static_cast<int>(ps.layout)
                     << " and second bits=" << static_cast<int>(ps.second_dataset_bits.value());
      }
    }
    size_t queries_size = ps.n_rows * ps.graph_degree;
    std::vector<uint32_t> indices_NNDescent(queries_size);
    std::vector<float> distances_NNDescent(queries_size);
    std::vector<uint32_t> indices_naive(queries_size);
    std::vector<float> distances_naive(queries_size);

    {
      rmm::device_uvector<float> distances_naive_dev(queries_size, stream_);
      rmm::device_uvector<uint32_t> indices_naive_dev(queries_size, stream_);
      naive_knn<float, float, uint32_t>(handle_,
                                        distances_naive_dev.data(),
                                        indices_naive_dev.data(),
                                        database.data_handle(),
                                        database.data_handle(),
                                        ps.n_rows,
                                        ps.n_rows,
                                        ps.dim,
                                        ps.graph_degree,
                                        ps.metric);
      raft::update_host(indices_naive.data(), indices_naive_dev.data(), queries_size, stream_);
      raft::update_host(distances_naive.data(), distances_naive_dev.data(), queries_size, stream_);
      raft::resource::sync_stream(handle_);
    }

    {
      std::vector<float> host_data(static_cast<size_t>(ps.n_rows) * ps.dim);
      raft::update_host(host_data.data(), database.data_handle(), host_data.size(), stream_);
      raft::resource::sync_stream(handle_);

      auto bbq_host_storage =
        cpu_bbq::quantize(host_data, ps.n_rows, ps.dim, ps.bits, ps.metric, ps.layout);
      auto bbq_host = cuvs::neighbors::host_bbq_dataset<int64_t>{std::move(bbq_host_storage)};

      if (ps.second_dataset_bits.has_value()) {
        auto second_layout           = ps.second_dataset_bits.value() == 1
                                         ? cuvs::preprocessing::quantize::bbq::bbq_code_layout::single_bit
                                         : cuvs::preprocessing::quantize::bbq::bbq_code_layout::dibit;
        auto bbq_host_second_storage = cpu_bbq::quantize(
          host_data, ps.n_rows, ps.dim, ps.second_dataset_bits.value(), ps.metric, second_layout);
        bbq_host.add_quantizer(std::move(bbq_host_second_storage));
      }
      auto owning_dataset = make_device_bbq_dataset(handle_, bbq_host);
      auto dataset        = owning_dataset.as_dataset_view();
      nn_descent::index_params index_params;
      index_params.metric                    = ps.metric;
      index_params.graph_degree              = ps.graph_degree;
      index_params.intermediate_graph_degree = 2 * ps.graph_degree;
      index_params.max_iterations            = 100;
      index_params.return_distances          = true;

      // Dense float baseline on the same data / params, timed with the same CUDA events.
      const float dense_ms = time_cuda_ms(stream_, [&] {
        auto database_view = raft::make_const_mdspan(database.view());
        auto dense_index   = nn_descent::build(handle_, index_params, database_view);
        (void)dense_index;
      });

      std::optional<index<uint32_t>> index;
      const float bbq_ms = time_cuda_ms(
        stream_, [&] { index.emplace(nn_descent::build(handle_, index_params, dataset)); });

      std::ostringstream metric_name;
      metric_name << print_metric{ps.metric};
      RAFT_LOG_INFO(
        "NN-Descent build timing: bbq(%u-bit,layout=%d, second_bits=%d) dense=%.3f ms, bbq=%.3f "
        "ms, speedup=%.2fx "
        "(n_rows=%d, dim=%d, graph_degree=%d, metric=%s)",
        static_cast<unsigned>(ps.bits),
        static_cast<int>(ps.layout),
        static_cast<int>(ps.second_dataset_bits.has_value() ? ps.second_dataset_bits.value() : 0),
        dense_ms,
        bbq_ms,
        dense_ms / std::max(bbq_ms, 1e-3f),
        ps.n_rows,
        ps.dim,
        ps.graph_degree,
        metric_name.str().c_str());

      raft::copy(indices_NNDescent.data(), index->graph().data_handle(), queries_size, stream_);
      ASSERT_TRUE(index->distances().has_value());
      raft::copy(distances_NNDescent.data(),
                 index->distances().value().data_handle(),
                 queries_size,
                 stream_);
      raft::resource::sync_stream(handle_);
    }

    EXPECT_TRUE(eval_neighbours(indices_naive,
                                indices_NNDescent,
                                distances_naive,
                                distances_NNDescent,
                                ps.n_rows,
                                ps.graph_degree,
                                0.001,
                                ps.min_recall));
  }

  void SetUp() override
  {
    raft::random::RngState r(1234ULL);
    raft::random::normal(handle_, r, database.data_handle(), ps.n_rows * ps.dim, 0.1f, 2.0f);
    raft::resource::sync_stream(handle_);
  }

  void TearDown() override { raft::resource::sync_stream(handle_); }

 private:
  raft::resources handle_;
  rmm::cuda_stream_view stream_;
  AnnNNDescentBbqInputs ps;
  raft::device_matrix<float, int64_t> database;
};

// Estimated recall based on bruteforce (InnerProduct): 1: 0.23, 2: 0.52, 4: 0.85, 7: 0.98, 8: 0.99.
const std::vector<AnnNNDescentBbqInputs> bbq_inputs = [] {
  using cuvs::preprocessing::quantize::bbq::bbq_code_layout;
  const std::vector<std::tuple<uint8_t, double, bbq_code_layout, std::optional<uint8_t>>>
    bits_specifications{// bits, min_recall, layout
                        {1, 0.15, bbq_code_layout::single_bit, std::optional<uint8_t>{}},
                        {2, 0.50, bbq_code_layout::dibit, std::optional<uint8_t>{}},
                        {2, 0.27, bbq_code_layout::dibit, std::optional<uint8_t>{1}},
                        {4, 0.80, bbq_code_layout::packed_nibble, std::optional<uint8_t>{}},
                        {4, 0.80, bbq_code_layout::transpose_half_byte, std::optional<uint8_t>{}},
                        {4, 0.35, bbq_code_layout::transpose_half_byte, std::optional<uint8_t>{1}},
                        {4, 0.65, bbq_code_layout::transpose_half_byte, std::optional<uint8_t>{2}},
                        {7, 0.93, bbq_code_layout::seven_bit, std::optional<uint8_t>{}},
                        {8, 0.95, bbq_code_layout::unsigned_byte, std::optional<uint8_t>{}}};
  std::vector<AnnNNDescentBbqInputs> out;
  for (const auto& [bits, min_recall, layout, second_bits] : bits_specifications) {
    const auto batch = raft::util::itertools::product<AnnNNDescentBbqInputs>(
      {10000},
      {64, 256},  // dim
      {32},       // graph_degree
      {cuvs::distance::DistanceType::L2Expanded,
       cuvs::distance::DistanceType::L2SqrtExpanded,
       cuvs::distance::DistanceType::InnerProduct,
       cuvs::distance::DistanceType::CosineExpanded},
      {false},  // host_dataset
      {min_recall},
      {bits},
      {layout},
      {second_bits});
    out.insert(out.end(), batch.begin(), batch.end());
  }
  return out;
}();

}  // namespace cuvs::neighbors::nn_descent
