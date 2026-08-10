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

/** Return component @p d from one encoded row. */
__device__ __forceinline__ uint32_t get_code(const uint8_t* row,
                                             size_t d,
                                             const bbq_dataset_view& dataset)
{
  const uint32_t mask = (uint32_t{1} << dataset.bits) - 1;
  // TODO put dataset.layout as template
  if (dataset.layout == code_layout::seven_bit || dataset.layout == code_layout::unsigned_byte) {
    return row[d] & mask;
  } else if (dataset.layout == code_layout::packed_nibble) {
    // Lucene unpackNibbles: high nibble = dims [0, half), low = [half, dim)
    const size_t half = dataset.dim / 2;
    if (d < half) {
      return (row[d] >> 4) & 0x0Fu;
    } else {
      return row[d - half] & 0x0Fu;
    }
  } else if (dataset.layout == code_layout::dibit ||
             dataset.layout == code_layout::transpose_half_byte) {
    const size_t stripe_size = (dataset.dim + 7) / 8;
    uint32_t code            = 0;
    for (uint32_t bit = 0; bit < dataset.bits; ++bit) {
      const uint8_t byte = row[bit * stripe_size + d / 8];
      code |= static_cast<uint32_t>((byte >> (7 - d % 8)) & 1u) << bit;
    }
    return code;
  } else {  // code_layout::single_bit
    const size_t position = d * dataset.bits;
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
  // If dim is not a multiple of 8, clear unused high bits in the last byte
  // before ANDing, or mask the last popc: bits_in_last = dim % 8.
  return result;
}

/** Symmetric I_xy for transposeHalfByte (4 bit-planes): sum_i sum_j popcount(A_i & B_j) << (i+j).
 */
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

/** Symmetric I_xy for packNibbles (Lucene int4DotProductBothPacked). */
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

/** Integer inner product between two encoded rows. */
__device__ __forceinline__ int64_t code_inner_product(const uint8_t* row_a,
                                                      const uint8_t* row_b,
                                                      const bbq_dataset_view& dataset)
{
  int64_t result = 0;
  if (dataset.bits == 1 && dataset.layout == code_layout::single_bit) {
    return code_inner_product_binary(row_a, row_b, dataset.encoded_row_length());
  } else if (dataset.bits == 4 && dataset.layout == code_layout::packed_nibble) {
    return code_inner_product_int4_packed_nibble_symmetric(
      row_a, row_b, dataset.encoded_row_length());
  } else if (dataset.bits == 4 && dataset.layout == code_layout::transpose_half_byte) {
    return code_inner_product_int4_transposeHalfByte_symmetric(
      row_a, row_b, dataset.encoded_row_length());
  }
  for (size_t d = 0; d < dataset.dim; ++d) {
    result += static_cast<int64_t>(get_code(row_a, d, dataset) * get_code(row_b, d, dataset));
  }
  return result;
}

/** Dot product of two decoded centroid-centered rows. */
__device__ __forceinline__ float centered_dot(const bbq_dataset_view& dataset,
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

  const uint8_t* codes_a = dataset.codes.data_handle() + row_a * dataset.encoded_row_length();
  const uint8_t* codes_b = dataset.codes.data_handle() + row_b * dataset.encoded_row_length();
  const float code_ip    = static_cast<float>(code_inner_product(codes_a, codes_b, dataset));

  return static_cast<float>(dataset.dim) * lower_a * lower_b + lower_b * delta_a * sum_a +
         lower_a * delta_b * sum_b + delta_a * delta_b * code_ip;
}

/** Dot product. */
__device__ __forceinline__ float dot_product(const bbq_dataset_view& dataset,
                                             int64_t row_a,
                                             int64_t row_b)
{
  return centered_dot(dataset, row_a, row_b) + dataset.additional_corrections(row_a) +
         dataset.additional_corrections(row_b) - dataset.centroid_norm_sq;
}

/** Squared L2 distance. */
__device__ __forceinline__ float l2_distance(const bbq_dataset_view& dataset,
                                             int64_t row_a,
                                             int64_t row_b)
{
  const float distance = dataset.additional_corrections(row_a) +
                         dataset.additional_corrections(row_b) -
                         2.0f * centered_dot(dataset, row_a, row_b);
  return distance < 0.0f ? 0.0f : distance;
}

/** Squared norm of one original-space row. */
__device__ __forceinline__ float row_norm(const bbq_dataset_view& dataset, int64_t row)
{
  const float norm = dot_product(dataset, row, row);
  return norm < 0.0f ? 0.0f : norm;
}

#endif  // __CUDACC__

}  // namespace preprocessing::quantize::bbq
}  // namespace CUVS_EXPORT cuvs
