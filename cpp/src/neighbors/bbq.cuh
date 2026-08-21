/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/export.hpp>
#include <cuvs/distance/distance.hpp>
#include <cuvs/preprocessing/quantize/bbq.hpp>

#include <raft/core/device_mdspan.hpp>

#include <cstddef>
#include <cstdint>

namespace CUVS_EXPORT cuvs {
namespace preprocessing::quantize::bbq {

#ifdef __CUDACC__

_RAFT_HOST_DEVICE constexpr uint32_t get_encoded_row_length(const bbq::code_layout layout,
                                                            const uint32_t bits,
                                                            const uint32_t dim)
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

_RAFT_HOST_DEVICE constexpr uint32_t get_encoded_row_length(
  const cuvs::neighbors::device_bbq_dataset_storage_view<int64_t>& dataset)
{
  return get_encoded_row_length(dataset.layout, dataset.bits, dataset.dim());
}

__device__ __forceinline__ uint32_t get_code(const uint8_t* row,
                                             size_t d,
                                             const bbq::code_layout layout,
                                             const uint32_t bits,
                                             const size_t dim)
{
  const uint32_t mask = (uint32_t{1} << bits) - 1;
  if (layout == code_layout::seven_bit || layout == code_layout::unsigned_byte) {
    return row[d] & mask;
  } else if (layout == code_layout::packed_nibble) {
    // Lucene unpackNibbles: high nibble = dims [0, half), low = [half, dim)
    const size_t half = dim / 2;
    if (d < half) {
      return (row[d] >> 4) & 0x0Fu;
    } else {
      return row[d - half] & 0x0Fu;
    }
  } else if (layout == code_layout::dibit || layout == code_layout::transpose_half_byte) {
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

__device__ __forceinline__ int64_t code_inner_product_binary(const uint8_t* row_a,
                                                             const uint8_t* row_b,
                                                             size_t n_bytes)
{
  int64_t result = 0;
  size_t i       = 0;
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

__device__ __forceinline__ int64_t code_inner_product_dibit_symmetric(const uint8_t* a,
                                                                      const uint8_t* b,
                                                                      size_t n_bytes)
{
  const size_t stripe_size = n_bytes / 2;
  int64_t r                = 0;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      r += code_inner_product_binary(a + i * stripe_size, b + j * stripe_size, stripe_size)
           << (i + j);
  return r;
}

__device__ __forceinline__ int64_t code_inner_product_int4_transposeHalfByte_symmetric(
  const uint8_t* row_a, const uint8_t* row_b, size_t n_bytes)
{
  const size_t stripe_size = n_bytes / 4;
  int64_t result           = 0;
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
__device__ __forceinline__ int64_t code_inner_product_int4_packed_nibble_symmetric(
  const uint8_t* row_a, const uint8_t* row_b, size_t n_bytes)
{
  int64_t total = 0;
  for (size_t i = 0; i < n_bytes; ++i) {
    const unsigned a = row_a[i];
    const unsigned b = row_b[i];
    total += (a & 0x0Fu) * (b & 0x0Fu);
    total += ((a >> 4) & 0x0Fu) * ((b >> 4) & 0x0Fu);
  }
  return total;
}

/** Unsigned byte dot product vectorized. */
__device__ __forceinline__ int64_t code_inner_product_unsigned_byte(const uint8_t* row_a,
                                                                    const uint8_t* row_b,
                                                                    size_t n_bytes)
{
  int64_t result = 0;
  size_t i       = 0;
  for (; i + 4 <= n_bytes; i += 4) {
    uint32_t a, b;
    memcpy(&a, row_a + i, sizeof(uint32_t));
    memcpy(&b, row_b + i, sizeof(uint32_t));
    result += __dp4a(a, b, 0u);
  }
  for (; i < n_bytes; ++i) {
    result += static_cast<int64_t>(row_a[i]) * static_cast<int64_t>(row_b[i]);
  }
  return result;
}

__device__ __forceinline__ int64_t code_inner_product(const uint8_t* row_a,
                                                      const uint8_t* row_b,
                                                      const bbq::code_layout layout,
                                                      const uint32_t bits,
                                                      const size_t dim,
                                                      const size_t n_bytes)
{
  switch (layout) {
    case code_layout::single_bit: return code_inner_product_binary(row_a, row_b, n_bytes);
    case code_layout::dibit: return code_inner_product_dibit_symmetric(row_a, row_b, n_bytes);
    case code_layout::packed_nibble:
      return code_inner_product_int4_packed_nibble_symmetric(row_a, row_b, n_bytes);
    case code_layout::transpose_half_byte:
      return code_inner_product_int4_transposeHalfByte_symmetric(row_a, row_b, n_bytes);
    case code_layout::unsigned_byte: return code_inner_product_unsigned_byte(row_a, row_b, n_bytes);
    case code_layout::seven_bit:  // unpacked seven_bit: one code per byte
    default:
      int64_t result      = 0;
      const uint32_t mask = (uint32_t{1} << bits) - 1;
      for (size_t d = 0; d < n_bytes; ++d) {
        result += static_cast<int64_t>(row_a[d] & mask) * static_cast<int64_t>(row_b[d] & mask);
      }
      return result;
  }
}
/** Integer inner product between two encoded rows. */
__device__ __forceinline__ int64_t
code_inner_product(const uint8_t* row_a,
                   const uint8_t* row_b,
                   const cuvs::neighbors::device_bbq_dataset_storage_view<int64_t>& dataset)
{
  return code_inner_product(
    row_a, row_b, dataset.layout, dataset.bits, dataset.dim(), get_encoded_row_length(dataset));
}

/** Centered dot product of two rows. */
__device__ __forceinline__ float centered_dot(
  const cuvs::neighbors::device_bbq_dataset_storage_view<int64_t>& dataset,
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
__device__ __forceinline__ float centered_dot(
  const cuvs::neighbors::device_bbq_dataset_storage_view<int64_t>& dataset,
  int64_t row_a,
  int64_t row_b)
{
  const uint8_t* codes_a = dataset.codes.data_handle() + row_a * get_encoded_row_length(dataset);
  const uint8_t* codes_b = dataset.codes.data_handle() + row_b * get_encoded_row_length(dataset);
  return centered_dot(
    dataset, static_cast<float>(code_inner_product(codes_a, codes_b, dataset)), row_a, row_b);
}

// Dot product overload when the centered dot product is already computed
__device__ __forceinline__ float dot_product(
  const cuvs::neighbors::device_bbq_dataset_storage_view<int64_t>& dataset,
  float centered_dot_value,
  int64_t row_a,
  int64_t row_b)
{
  return centered_dot_value + dataset.additional_corrections(row_a) +
         dataset.additional_corrections(row_b) - dataset.centroid_norm_sq;
}
/** Dot product. */
__device__ __forceinline__ float dot_product(
  const cuvs::neighbors::device_bbq_dataset_storage_view<int64_t>& dataset,
  int64_t row_a,
  int64_t row_b)
{
  return dot_product(dataset, centered_dot(dataset, row_a, row_b), row_a, row_b);
}

/** Squared L2 distance overload when the centered dot product is already computed */
__device__ __forceinline__ float l2_distance(
  const cuvs::neighbors::device_bbq_dataset_storage_view<int64_t>& dataset,
  float centered_dot_value,
  int64_t row_a,
  int64_t row_b)
{
  const float distance = dataset.additional_corrections(row_a) +
                         dataset.additional_corrections(row_b) - 2.0f * centered_dot_value;
  return distance < 0.0f ? 0.0f : distance;
}
/** Squared L2 distance. */
__device__ __forceinline__ float l2_distance(
  const cuvs::neighbors::device_bbq_dataset_storage_view<int64_t>& dataset,
  int64_t row_a,
  int64_t row_b)
{
  return l2_distance(dataset, centered_dot(dataset, row_a, row_b), row_a, row_b);
}

__device__ __forceinline__ float cosine_distance(
  const cuvs::neighbors::device_bbq_dataset_storage_view<int64_t>& dataset,
  float centered_dot_value,
  int64_t row_a,
  int64_t row_b,
  float norm_product)
{
  const auto dot = dot_product(dataset, centered_dot_value, row_a, row_b);
  return norm_product > 0.0f ? 1.0f - dot / sqrtf(norm_product) : 0.0f;
}

// norm_product = norm_a * norm_b
__device__ __forceinline__ float cosine_distance(
  const cuvs::neighbors::device_bbq_dataset_storage_view<int64_t>& dataset,
  int64_t row_a,
  int64_t row_b,
  float norm_product)
{
  return cosine_distance(dataset, centered_dot(dataset, row_a, row_b), row_a, row_b, norm_product);
}

/** Squared norm of one original-space row. */
__device__ __forceinline__ float row_norm(
  const cuvs::neighbors::device_bbq_dataset_storage_view<int64_t>& dataset, int64_t row)
{
  const float norm = dot_product(dataset, row, row);
  return norm < 0.0f ? 0.0f : norm;
}

#endif  // __CUDACC__

}  // namespace preprocessing::quantize::bbq
}  // namespace CUVS_EXPORT cuvs
