/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ann_nn_descent.cuh"

#include <cuvs/preprocessing/quantize/bbq.hpp>

#include <raft/core/device_mdarray.hpp>
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
#include <vector>

#include <raft/core/logger.hpp>

namespace cuvs::neighbors::nn_descent {
// Host-side Lucene OptimizedScalarQuantizer
namespace cpu_bbq {

using cuvs::preprocessing::quantize::bbq::code_layout;

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
                                  const std::vector<float>& centroid,
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

struct host_dataset {
  std::vector<uint8_t> codes;
  std::vector<float> lower_intervals;
  std::vector<float> upper_intervals;
  std::vector<float> additional_corrections;
  std::vector<int32_t> quantized_component_sums;
  std::vector<float> centroid;
  float centroid_norm_sq{0.0f};
};

inline size_t encoded_row_length(size_t dim, uint32_t bits, code_layout layout)
{
  switch (layout) {
    case code_layout::single_bit: return (dim * bits + 7) / 8;
    case code_layout::dibit: return bits * ((dim + 7) / 8);
    case code_layout::packed_nibble: return (dim + 1) / 2;
    case code_layout::seven_bit: return dim;
    case code_layout::unsigned_byte: return dim;
    case code_layout::transpose_half_byte: return 4 * ((dim + 7) / 8);
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
                                       code_layout layout)
{
  const size_t row_length = encoded_row_length(dim, bits, layout);
  if (layout == code_layout::unsigned_byte || layout == code_layout::seven_bit) { return unpacked; }

  std::vector<uint8_t> packed(n_rows * row_length, 0);
  for (size_t row = 0; row < n_rows; ++row) {
    auto* output      = packed.data() + row * row_length;
    const auto* input = unpacked.data() + row * dim;
    if (layout == code_layout::packed_nibble) {
      // Lucene OffHeapScalarQuantizedVectorValues.packNibbles
      const size_t half = dim / 2;
      for (size_t i = 0; i < half; ++i) {
        output[i] = static_cast<uint8_t>((input[i] << 4) | (input[half + i] & 0x0f));
      }
      continue;
    }
    for (size_t d = 0; d < dim; ++d) {
      const uint8_t code = input[d];
      if (layout == code_layout::single_bit) {
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

inline host_dataset quantize(const std::vector<float>& data,
                             int64_t n_rows,
                             int64_t dim,
                             uint8_t bits,
                             cuvs::distance::DistanceType metric,
                             code_layout layout = code_layout::unsigned_byte)
{
  const bool euclidean = metric == cuvs::distance::DistanceType::L2Expanded ||
                         metric == cuvs::distance::DistanceType::L2SqrtExpanded;

  host_dataset out;
  out.centroid.assign(dim, 0.0f);
  for (int64_t i = 0; i < n_rows; ++i) {
    for (int64_t d = 0; d < dim; ++d) {
      out.centroid[d] += data[i * dim + d];
    }
  }
  for (int64_t d = 0; d < dim; ++d) {
    out.centroid[d] /= static_cast<float>(n_rows);
    out.centroid_norm_sq += out.centroid[d] * out.centroid[d];
  }

  std::vector<uint8_t> unpacked(static_cast<size_t>(n_rows * dim));
  out.lower_intervals.resize(n_rows);
  out.upper_intervals.resize(n_rows);
  out.additional_corrections.resize(n_rows);
  out.quantized_component_sums.resize(n_rows);

  for (int64_t i = 0; i < n_rows; ++i) {
    std::vector<float> row(data.begin() + i * dim, data.begin() + (i + 1) * dim);
    std::vector<uint8_t> codes(dim);
    const auto result = scalar_quantize(row, codes, bits, out.centroid, euclidean);
    std::copy(codes.begin(), codes.end(), unpacked.begin() + i * dim);
    out.lower_intervals[i]          = result.lower_interval;
    out.upper_intervals[i]          = result.upper_interval;
    out.additional_corrections[i]   = result.additional_correction;
    out.quantized_component_sums[i] = result.quantized_component_sum;
  }
  out.codes =
    pack_codes(unpacked, static_cast<size_t>(n_rows), static_cast<size_t>(dim), bits, layout);
  return out;
}

}  // namespace cpu_bbq

struct AnnNNDescentBbqInputs : AnnNNDescentInputs {
  uint8_t bits;
  cuvs::preprocessing::quantize::bbq::code_layout layout;
};

inline ::std::ostream& operator<<(::std::ostream& os, const AnnNNDescentBbqInputs& p)
{
  os << "dataset shape=" << p.n_rows << "x" << p.dim << ", graph_degree=" << p.graph_degree
     << ", metric="
     << cuvs::neighbors::print_metric{static_cast<cuvs::distance::DistanceType>((int)p.metric)}
     << (p.host_dataset ? ", host" : ", device") << ", bits=" << static_cast<int>(p.bits)
     << ", layout=" << static_cast<int>(p.layout) << std::endl;
  return os;
}

class AnnNNDescentBbqTest : public ::testing::TestWithParam<AnnNNDescentBbqInputs> {
 public:
  AnnNNDescentBbqTest()
    : stream_(raft::resource::get_cuda_stream(handle_)),
      ps(::testing::TestWithParam<AnnNNDescentBbqInputs>::GetParam()),
      database(0, stream_)
  {
  }

 protected:
  void testNNDescent()
  {
    raft::resource::set_workspace_to_pool_resource(handle_, 10 * 1024 * 1024 * 1024ull);
    if (ps.metric != cuvs::distance::DistanceType::L2Expanded &&
        ps.metric != cuvs::distance::DistanceType::L2SqrtExpanded &&
        ps.metric != cuvs::distance::DistanceType::InnerProduct &&
        ps.metric != cuvs::distance::DistanceType::CosineExpanded) {
      GTEST_SKIP();
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
                                        database.data(),
                                        database.data(),
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
      raft::update_host(host_data.data(), database.data(), host_data.size(), stream_);
      raft::resource::sync_stream(handle_);

      auto bbq_host =
        cpu_bbq::quantize(host_data, ps.n_rows, ps.dim, ps.bits, ps.metric, ps.layout);

      const auto row_length =
        static_cast<int64_t>(cpu_bbq::encoded_row_length(ps.dim, ps.bits, ps.layout));
      auto codes_d = raft::make_device_matrix<uint8_t, int64_t>(handle_, ps.n_rows, row_length);
      auto lower_d = raft::make_device_vector<float, int64_t>(handle_, ps.n_rows);
      auto upper_d = raft::make_device_vector<float, int64_t>(handle_, ps.n_rows);
      auto corrections_d = raft::make_device_vector<float, int64_t>(handle_, ps.n_rows);
      auto sums_d        = raft::make_device_vector<int32_t, int64_t>(handle_, ps.n_rows);
      auto centroid_d    = raft::make_device_vector<float, int64_t>(handle_, ps.dim);

      raft::update_device(
        codes_d.data_handle(), bbq_host.codes.data(), bbq_host.codes.size(), stream_);
      raft::update_device(lower_d.data_handle(),
                          bbq_host.lower_intervals.data(),
                          bbq_host.lower_intervals.size(),
                          stream_);
      raft::update_device(upper_d.data_handle(),
                          bbq_host.upper_intervals.data(),
                          bbq_host.upper_intervals.size(),
                          stream_);
      raft::update_device(corrections_d.data_handle(),
                          bbq_host.additional_corrections.data(),
                          bbq_host.additional_corrections.size(),
                          stream_);
      raft::update_device(sums_d.data_handle(),
                          bbq_host.quantized_component_sums.data(),
                          bbq_host.quantized_component_sums.size(),
                          stream_);
      raft::update_device(
        centroid_d.data_handle(), bbq_host.centroid.data(), bbq_host.centroid.size(), stream_);
      raft::resource::sync_stream(handle_);

      cuvs::preprocessing::quantize::bbq::bbq_dataset_view dataset{
        raft::make_const_mdspan(codes_d.view()),
        raft::make_const_mdspan(lower_d.view()),
        raft::make_const_mdspan(upper_d.view()),
        raft::make_const_mdspan(corrections_d.view()),
        raft::make_const_mdspan(sums_d.view()),
        raft::make_const_mdspan(centroid_d.view()),
        static_cast<size_t>(ps.dim),
        ps.bits,
        ps.layout,
        ps.metric,
        bbq_host.centroid_norm_sq};

      nn_descent::index_params index_params;
      index_params.metric                    = ps.metric;
      index_params.graph_degree              = ps.graph_degree;
      index_params.intermediate_graph_degree = 2 * ps.graph_degree;
      index_params.max_iterations            = 100;
      index_params.return_distances          = true;

      auto index = nn_descent::build(handle_, index_params, dataset);

      raft::copy(indices_NNDescent.data(), index.graph().data_handle(), queries_size, stream_);
      ASSERT_TRUE(index.distances().has_value());
      raft::copy(
        distances_NNDescent.data(), index.distances().value().data_handle(), queries_size, stream_);
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
    database.resize(static_cast<size_t>(ps.n_rows) * ps.dim, stream_);
    raft::random::RngState r(1234ULL);
    raft::random::normal(handle_, r, database.data(), ps.n_rows * ps.dim, 0.1f, 2.0f);
    raft::resource::sync_stream(handle_);
  }

  void TearDown() override
  {
    raft::resource::sync_stream(handle_);
    database.resize(0, stream_);
  }

 private:
  raft::resources handle_;
  rmm::cuda_stream_view stream_;
  AnnNNDescentBbqInputs ps;
  rmm::device_uvector<float> database;
};

// Estimated recall based on bruteforce (InnerProduct): 1: 0.23, 2: 0.52, 4: 0.85, 7: 0.98, 8: 0.99.
const std::vector<AnnNNDescentBbqInputs> bbq_inputs = [] {
  using cuvs::preprocessing::quantize::bbq::code_layout;
  const std::vector<std::tuple<uint8_t, double, code_layout>> bits_specifications{
    // bits, min_recall, layout
    {1, 0.15, code_layout::single_bit},
    {2, 0.50, code_layout::dibit},
    {4, 0.80, code_layout::packed_nibble},
    {4, 0.80, code_layout::transpose_half_byte},
    {7, 0.93, code_layout::seven_bit},
    {8, 0.95, code_layout::unsigned_byte}};
  std::vector<AnnNNDescentBbqInputs> out;
  for (const auto& [bits, min_recall, layout] : bits_specifications) {
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
      {layout});
    out.insert(out.end(), batch.begin(), batch.end());
  }
  return out;
}();

}  // namespace cuvs::neighbors::nn_descent
