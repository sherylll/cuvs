/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/export.hpp>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/preprocessing/quantize/bbq.hpp>

#include <raft/core/device_mdspan.hpp>

#include <cstddef>
#include <cstdint>

namespace CUVS_EXPORT cuvs {
namespace preprocessing::quantize::bbq {

#ifdef __CUDACC__

_RAFT_HOST_DEVICE constexpr uint32_t get_encoded_row_length(const bbq_code_layout layout,
                                                            const uint32_t bits,
                                                            const uint32_t dim)
{
  switch (layout) {
    case bbq_code_layout::single_bit: return (dim * bits + 7) / 8;
    case bbq_code_layout::dibit: return bits * ((dim + 7) / 8);
    case bbq_code_layout::packed_nibble: return (dim + 1) / 2;
    case bbq_code_layout::transpose_half_byte: return 4 * ((dim + 7) / 8);
    case bbq_code_layout::seven_bit: return dim;
    case bbq_code_layout::unsigned_byte: return dim;
  }
  return 0;
}

template <typename DataT, typename IdxT, typename Accessor>
_RAFT_HOST_DEVICE constexpr uint32_t get_encoded_row_length(
  const bbq_quantizer_view<DataT, IdxT, Accessor>& dataset)
{
  return get_encoded_row_length(dataset.layout, dataset.bits, dataset.dim());
}

#if 0  // No callers outside this header.
__device__ __forceinline__ uint32_t get_code(
  const uint8_t* row, size_t d, const bbq_code_layout layout, const uint32_t bits, const size_t dim)
{
  const uint32_t mask = (uint32_t{1} << bits) - 1;
  if (layout == bbq_code_layout::seven_bit || layout == bbq_code_layout::unsigned_byte) {
    return row[d] & mask;
  } else if (layout == bbq_code_layout::packed_nibble) {
    // Lucene unpackNibbles: high nibble = dims [0, half), low = [half, dim)
    const size_t half = dim / 2;
    if (d < half) {
      return (row[d] >> 4) & 0x0Fu;
    } else {
      return row[d - half] & 0x0Fu;
    }
  } else if (layout == bbq_code_layout::dibit || layout == bbq_code_layout::transpose_half_byte) {
    const size_t stripe_size = (dim + 7) / 8;
    uint32_t code            = 0;
    for (uint32_t bit = 0; bit < bits; ++bit) {
      const uint8_t byte = row[bit * stripe_size + d / 8];
      code |= static_cast<uint32_t>((byte >> (7 - d % 8)) & 1u) << bit;
    }
    return code;
  } else {  // code_layout::single_bit
    const size_t position = d * bits;
    return static_cast<uint32_t>((row[position / 8] >> (7 - position % 8)) & 1u);
  }
}
#endif

__device__ __forceinline__ uint32_t code_inner_product_binary(const uint8_t* row_a,
                                                              const uint8_t* row_b,
                                                              size_t n_bytes,
                                                              uint32_t result = 0)
{
  size_t i       = 0;
  for (; i + 16 <= n_bytes; i += 16) {
    uint4 a, b;
    memcpy(&a, row_a + i, 16);
    memcpy(&b, row_b + i, 16);
    result += __popc(a.x & b.x) + __popc(a.y & b.y) + __popc(a.z & b.z) + __popc(a.w & b.w);
  }
  for (; i + 4 <= n_bytes; i += 4) {
    uint32_t a, b;
    memcpy(&a, row_a + i, 4);
    memcpy(&b, row_b + i, 4);
    result += __popc(a & b);
  }
  for (; i < n_bytes; ++i) {
    result += __popc(static_cast<unsigned>(row_a[i] & row_b[i]));
  }
  return result;
}

__device__ __forceinline__ uint32_t code_inner_product_dibit_symmetric(const uint8_t* a,
                                                                       const uint8_t* b,
                                                                       size_t n_bytes,
                                                                       uint32_t result = 0)
{
  const size_t stripe_size = n_bytes / 2;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      result += code_inner_product_binary(a + i * stripe_size, b + j * stripe_size, stripe_size)
                << (i + j);
  return result;
}

__device__ __forceinline__ uint32_t code_inner_product_int4_transposeHalfByte_symmetric(
  const uint8_t* row_a, const uint8_t* row_b, size_t n_bytes, uint32_t result = 0)
{
  const size_t stripe_size = n_bytes / 4;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      result +=
        code_inner_product_binary(row_a + i * stripe_size, row_b + j * stripe_size, stripe_size)
        << (i + j);
    }
  }
  return result;
}

/** Symmetric for packNibbles (Lucene int4DotProductBothPacked). */
__device__ __forceinline__ uint32_t code_inner_product_int4_packed_nibble_symmetric(
  const uint8_t* row_a, const uint8_t* row_b, size_t n_bytes, uint32_t total = 0)
{
  constexpr uint32_t nibble_mask = 0x0F0F0F0Fu;
  size_t i                       = 0;
#pragma unroll 4
  for (; i + 4 <= n_bytes; i += 4) {
    const auto a = *reinterpret_cast<const uint32_t*>(row_a + i);
    const auto b = *reinterpret_cast<const uint32_t*>(row_b + i);
    total        = __dp4a(a & nibble_mask, b & nibble_mask, total);
    total        = __dp4a((a >> 4) & nibble_mask, (b >> 4) & nibble_mask, total);
  }
  for (; i < n_bytes; ++i) {
    const unsigned a = row_a[i];
    const unsigned b = row_b[i];
    total += (a & 0x0Fu) * (b & 0x0Fu);
    total += ((a >> 4) & 0x0Fu) * ((b >> 4) & 0x0Fu);
  }
  return total;
}

/** One-byte-per-code dot product, optionally masking unused high bits. */
__device__ __forceinline__ uint32_t code_inner_product_unsigned_byte(const uint8_t* row_a,
                                                                     const uint8_t* row_b,
                                                                     size_t n_bytes,
                                                                     uint32_t result   = 0,
                                                                     uint8_t code_mask = 0xFFu)
{
  const uint32_t word_mask = uint32_t{code_mask} * 0x01010101u;
  size_t i       = 0;
#pragma unroll 4
  for (; i + 4 <= n_bytes; i += 4) {
    const auto a = *reinterpret_cast<const uint32_t*>(row_a + i) & word_mask;
    const auto b = *reinterpret_cast<const uint32_t*>(row_b + i) & word_mask;
    result       = __dp4a(a, b, result);
  }
  for (; i < n_bytes; ++i) {
    result +=
      static_cast<uint32_t>(row_a[i] & code_mask) * static_cast<uint32_t>(row_b[i] & code_mask);
  }
  return result;
}

/**
 * Integer inner product between two encoded rows.
 *
 * The uint32_t result bounds every BBQ layout to 66,050 dimensions: the worst case is
 * `unsigned_byte`, where `dim * 255 * 255` must not exceed UINT32_MAX.
 */
__device__ __forceinline__ uint32_t code_inner_product(const uint8_t* row_a,
                                                       const uint8_t* row_b,
                                                       const bbq_code_layout layout,
                                                       const uint32_t bits,
                                                       const size_t n_bytes,
                                                       uint32_t result = 0)
{
  switch (layout) {
    case bbq_code_layout::single_bit:
      return code_inner_product_binary(row_a, row_b, n_bytes, result);
    case bbq_code_layout::dibit:
      return code_inner_product_dibit_symmetric(row_a, row_b, n_bytes, result);
    case bbq_code_layout::packed_nibble:
      return code_inner_product_int4_packed_nibble_symmetric(row_a, row_b, n_bytes, result);
    case bbq_code_layout::transpose_half_byte:
      return code_inner_product_int4_transposeHalfByte_symmetric(row_a, row_b, n_bytes, result);
    case bbq_code_layout::unsigned_byte:
      return code_inner_product_unsigned_byte(row_a, row_b, n_bytes, result);
    case bbq_code_layout::seven_bit:
    default:
      return code_inner_product_unsigned_byte(
        row_a, row_b, n_bytes, result, static_cast<uint8_t>((uint32_t{1} << bits) - 1));
  }
}

template <typename DataT, typename IdxT, typename Accessor>
__device__ __forceinline__ uint32_t
code_inner_product(const uint8_t* row_a,
                   const uint8_t* row_b,
                   const bbq_quantizer_view<DataT, IdxT, Accessor>& dataset)
{
  return code_inner_product(
    row_a, row_b, dataset.layout, dataset.bits, get_encoded_row_length(dataset));
}

/** Centered dot product of two rows. */
template <typename DataT, typename IdxT, typename Accessor>
__device__ __forceinline__ float centered_dot(
  const bbq_quantizer_view<DataT, IdxT, Accessor>& dataset,
  float code_ip,
  int64_t row_a,
  int64_t row_b)
{
  const float lower_a = dataset.lower_intervals(row_a);
  const float lower_b = dataset.lower_intervals(row_b);
  const float scale   = 1.0f / static_cast<float>((uint32_t{1} << dataset.bits) - 1);
  const float delta_a = (dataset.upper_intervals(row_a) - lower_a) * scale;
  const float delta_b = (dataset.upper_intervals(row_b) - lower_b) * scale;
  const float sum_a   = static_cast<float>(dataset.quantized_component_sums(row_a));
  const float sum_b   = static_cast<float>(dataset.quantized_component_sums(row_b));

  return static_cast<float>(dataset.dim()) * lower_a * lower_b + lower_b * delta_a * sum_a +
         lower_a * delta_b * sum_b + delta_a * delta_b * code_ip;
}

/** Centered dot product. */
template <typename DataT, typename IdxT, typename Accessor>
__device__ __forceinline__ float centered_dot(
  const bbq_quantizer_view<DataT, IdxT, Accessor>& dataset, int64_t row_a, int64_t row_b)
{
  const uint8_t* codes_a = dataset.codes.data_handle() + row_a * get_encoded_row_length(dataset);
  const uint8_t* codes_b = dataset.codes.data_handle() + row_b * get_encoded_row_length(dataset);
  return centered_dot(
    dataset, static_cast<float>(code_inner_product(codes_a, codes_b, dataset)), row_a, row_b);
}

// Dot product overload when the centered dot product is already computed
template <typename DataT, typename IdxT, typename Accessor>
__device__ __forceinline__ float dot_product(
  const bbq_quantizer_view<DataT, IdxT, Accessor>& dataset,
  float centered_dot_value,
  int64_t row_a,
  int64_t row_b)
{
  return centered_dot_value + dataset.additional_corrections(row_a) +
         dataset.additional_corrections(row_b) - dataset.centroid_norm_sq;
}
/** Dot product. */
template <typename DataT, typename IdxT, typename Accessor>
__device__ __forceinline__ float dot_product(
  const bbq_quantizer_view<DataT, IdxT, Accessor>& dataset, int64_t row_a, int64_t row_b)
{
  return dot_product(dataset, centered_dot(dataset, row_a, row_b), row_a, row_b);
}

/** Squared L2 distance overload when the centered dot product is already computed */
template <typename DataT, typename IdxT, typename Accessor>
__device__ __forceinline__ float l2_distance(
  const bbq_quantizer_view<DataT, IdxT, Accessor>& dataset,
  float centered_dot_value,
  int64_t row_a,
  int64_t row_b)
{
  const float distance = dataset.additional_corrections(row_a) +
                         dataset.additional_corrections(row_b) - 2.0f * centered_dot_value;
  return distance < 0.0f ? 0.0f : distance;
}
/** Squared L2 distance. */
#if 0  // No callers outside this header.
template <typename DataT, typename IdxT, typename Accessor>
__device__ __forceinline__ float l2_distance(
  const bbq_quantizer_view<DataT, IdxT, Accessor>& dataset, int64_t row_a, int64_t row_b)
{
  return l2_distance(dataset, centered_dot(dataset, row_a, row_b), row_a, row_b);
}
#endif

template <typename DataT, typename IdxT, typename Accessor>
__device__ __forceinline__ float cosine_distance(
  const bbq_quantizer_view<DataT, IdxT, Accessor>& dataset,
  float centered_dot_value,
  int64_t row_a,
  int64_t row_b,
  float norm_product)
{
  const auto dot = dot_product(dataset, centered_dot_value, row_a, row_b);
  return norm_product > 0.0f ? 1.0f - dot / sqrtf(norm_product) : 0.0f;
}

// norm_product = norm_a * norm_b
#if 0  // No callers outside this header.
template <typename DataT, typename IdxT, typename Accessor>
__device__ __forceinline__ float cosine_distance(
  const bbq_quantizer_view<DataT, IdxT, Accessor>& dataset,
  int64_t row_a,
  int64_t row_b,
  float norm_product)
{
  return cosine_distance(dataset, centered_dot(dataset, row_a, row_b), row_a, row_b, norm_product);
}
#endif

/** Squared norm of one original-space row. */
template <typename DataT, typename IdxT, typename Accessor>
__device__ __forceinline__ float row_norm(const bbq_quantizer_view<DataT, IdxT, Accessor>& dataset,
                                          int64_t row)
{
  const float norm = dot_product(dataset, row, row);
  return norm < 0.0f ? 0.0f : norm;
}

template <typename DataT, typename IdxT, typename Accessor>
struct bbq_row_norm_op {
  const bbq_quantizer_view<DataT, IdxT, Accessor> quantizer;

  __device__ auto operator()(size_t row) const -> float
  {
    return row_norm(quantizer, static_cast<int64_t>(row));
  }
};

#if 0  // No callers outside this header.
// Lucene int4BitDotProductImpl
__device__ __forceinline__ int64_t
code_inner_product_asymmetric_1_vs_4(const uint8_t* codes_document,
                                     const uint8_t* codes_query,
                                     size_t n_bytes,
                                     const size_t stripe_size)
{
  int64_t result = 0;
  for (int i = 0; i < 4; ++i) {
    result += code_inner_product_binary(codes_document, codes_query + i * stripe_size, stripe_size)
              << i;
  }
  return result;
}

// Lucene int4DibitDotProductImpl
__device__ __forceinline__ int64_t code_inner_product_asymmetric_2_vs_4(
  const uint8_t* codes_document, const uint8_t* codes_query, size_t n_bytes)
{
  int stripe_size = n_bytes / 4;
  auto ret0 =
    code_inner_product_asymmetric_1_vs_4(codes_document, codes_query, n_bytes, stripe_size);
  auto ret1 = code_inner_product_asymmetric_1_vs_4(
    codes_document + stripe_size, codes_query, n_bytes, stripe_size);
  return ret0 + (ret1 << 1);
}

__device__ __forceinline__ int64_t code_inner_product_asymmetric_1_vs_2(
  const uint8_t* codes_document, const uint8_t* codes_query, size_t n_bytes)
{
  int stripe_size = n_bytes / 2;
  auto res0       = code_inner_product_binary(codes_document, codes_query, stripe_size);
  auto res1 = code_inner_product_binary(codes_document, codes_query + stripe_size, stripe_size);
  return res0 + (res1 << 1);
}

__device__ __forceinline__ int64_t
code_inner_product_asymmetric(const uint8_t* codes_document,
                              const uint8_t* codes_query,
                              const bbq_code_layout layout_dataset,
                              const bbq_code_layout layout_query,
                              const size_t n_bytes_query)
{
  if (layout_dataset == bbq_code_layout::single_bit &&
      layout_query == bbq_code_layout::transpose_half_byte) {
    return code_inner_product_asymmetric_1_vs_4(
      codes_document, codes_query, n_bytes_query, n_bytes_query / 4);
  } else if (layout_dataset == bbq_code_layout::single_bit &&
             layout_query == bbq_code_layout::dibit) {
    return code_inner_product_asymmetric_1_vs_2(codes_document, codes_query, n_bytes_query);
  } else if (layout_dataset == bbq_code_layout::dibit &&
             layout_query == bbq_code_layout::transpose_half_byte) {
    return code_inner_product_asymmetric_2_vs_4(codes_document, codes_query, n_bytes_query);
  } else {
    return -1;  // Unsupported layouts
  }
}
#endif

/** Centered dot product of two rows. */
template <typename DataT, typename IdxT, typename Accessor>
__device__ __forceinline__ float centered_dot(
  const bbq_quantizer_view<DataT, IdxT, Accessor>& dataset_document,
  const bbq_quantizer_view<DataT, IdxT, Accessor>& dataset_query,
  float code_ip,
  int64_t row_document,
  int64_t row_query)
{
  const float lower_doc = dataset_document.lower_intervals(row_document);
  const float lower_q   = dataset_query.lower_intervals(row_query);
  const float scale_document =
    1.0f / static_cast<float>((uint32_t{1} << dataset_document.bits) - 1);
  const float scale_query = 1.0f / static_cast<float>((uint32_t{1} << dataset_query.bits) - 1);
  const float delta_doc =
    (dataset_document.upper_intervals(row_document) - lower_doc) * scale_document;
  const float delta_q = (dataset_query.upper_intervals(row_query) - lower_q) * scale_query;
  const float sum_doc = static_cast<float>(dataset_document.quantized_component_sums(row_document));
  const float sum_q   = static_cast<float>(dataset_query.quantized_component_sums(row_query));

  auto dim = static_cast<float>(dataset_document.dim());
  return dim * lower_doc * lower_q + lower_q * delta_doc * sum_doc + lower_doc * delta_q * sum_q +
         delta_doc * delta_q * code_ip;
}

/** Centered dot product. */
#if 0  // No callers outside this header.
template <typename DataT, typename IdxT, typename Accessor>
__device__ __forceinline__ float centered_dot(
  const bbq_quantizer_view<DataT, IdxT, Accessor>& dataset_document,
  const bbq_quantizer_view<DataT, IdxT, Accessor>& dataset_query,
  int64_t row_document,
  int64_t row_query)
{
  const uint8_t* codes_doc =
    dataset_document.codes.data_handle() + row_document * get_encoded_row_length(dataset_document);
  const uint8_t* codes_q =
    dataset_query.codes.data_handle() + row_query * get_encoded_row_length(dataset_query);
  auto n_bytes = get_encoded_row_length(dataset_query);
  return centered_dot(
    dataset_document,
    dataset_query,
    static_cast<float>(code_inner_product_asymmetric(
      codes_doc, codes_q, dataset_document.layout, dataset_query.layout, n_bytes)),
    row_document,
    row_query);
}
#endif

// Dot product overload when the centered dot product is already computed
template <typename DataT, typename IdxT, typename Accessor>
__device__ __forceinline__ float dot_product(
  const bbq_quantizer_view<DataT, IdxT, Accessor>& dataset_document,
  const bbq_quantizer_view<DataT, IdxT, Accessor>& dataset_query,
  float centered_dot_value,
  int64_t row_document,
  int64_t row_query)
{
  return centered_dot_value + dataset_document.additional_corrections(row_document) +
         dataset_query.additional_corrections(row_query) - dataset_document.centroid_norm_sq;
}
/** Dot product. */
#if 0  // No callers outside this header.
template <typename DataT, typename IdxT, typename Accessor>
__device__ __forceinline__ float dot_product(
  const bbq_quantizer_view<DataT, IdxT, Accessor>& dataset_doc,
  const bbq_quantizer_view<DataT, IdxT, Accessor>& dataset_query,
  int64_t row_document,
  int64_t row_query)
{
  return dot_product(dataset_doc,
                     dataset_query,
                     centered_dot(dataset_doc, dataset_query, row_document, row_query),
                     row_document,
                     row_query);
}
#endif

/** Squared L2 distance overload when the centered dot product is already computed */
template <typename DataT, typename IdxT, typename Accessor>
__device__ __forceinline__ float l2_distance(
  const bbq_quantizer_view<DataT, IdxT, Accessor>& dataset_doc,
  const bbq_quantizer_view<DataT, IdxT, Accessor>& dataset_query,
  float centered_dot_value,
  int64_t row_document,
  int64_t row_query)
{
  const float distance = dataset_doc.additional_corrections(row_document) +
                         dataset_query.additional_corrections(row_query) -
                         2.0f * centered_dot_value;
  return distance < 0.0f ? 0.0f : distance;
}
/** Squared L2 distance. */
#if 0  // No callers outside this header.
template <typename DataT, typename IdxT, typename Accessor>
__device__ __forceinline__ float l2_distance(
  const bbq_quantizer_view<DataT, IdxT, Accessor>& dataset_doc,
  const bbq_quantizer_view<DataT, IdxT, Accessor>& dataset_query,
  int64_t row_document,
  int64_t row_query)
{
  return l2_distance(dataset_doc,
                     dataset_query,
                     centered_dot(dataset_doc, dataset_query, row_document, row_query),
                     row_document,
                     row_query);
}
#endif

template <typename DataT, typename IdxT, typename Accessor>
__device__ __forceinline__ float cosine_distance(
  const bbq_quantizer_view<DataT, IdxT, Accessor>& dataset_doc,
  const bbq_quantizer_view<DataT, IdxT, Accessor>& dataset_query,
  float centered_dot_value,
  int64_t row_document,
  int64_t row_query,
  float norm_product)
{
  const auto dot =
    dot_product(dataset_doc, dataset_query, centered_dot_value, row_document, row_query);
  return norm_product > 0.0f ? 1.0f - dot / sqrtf(norm_product) : 0.0f;
}

// norm_product = norm_a * norm_b
#if 0  // No callers outside this header.
template <typename DataT, typename IdxT, typename Accessor>
__device__ __forceinline__ float cosine_distance(
  const bbq_quantizer_view<DataT, IdxT, Accessor>& dataset_doc,
  const bbq_quantizer_view<DataT, IdxT, Accessor>& dataset_query,
  int64_t row_document,
  int64_t row_query,
  float norm_product)
{
  return cosine_distance(dataset_doc,
                         dataset_query,
                         centered_dot(dataset_doc, dataset_query, row_document, row_query),
                         row_document,
                         row_query,
                         norm_product);
}
#endif

#endif  // __CUDACC__

}  // namespace preprocessing::quantize::bbq
}  // namespace CUVS_EXPORT cuvs
