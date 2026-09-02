/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/export.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace CUVS_EXPORT cuvs {
namespace preprocessing::quantize::bbq {

#ifdef __CUDACC__

/** Two binary inner products with a shared right operand. */
template <size_t n_bytes>
__device__ inline void code_inner_product_binary_2x1(const uint8_t* row_a0,
                                                     const uint8_t* row_a1,
                                                     const uint8_t* row_b,
                                                     uint32_t& total0,
                                                     uint32_t& total1)
{
  static_assert(n_bytes % sizeof(uint32_t) == 0);
#pragma unroll 4
  for (size_t i = 0; i < n_bytes; i += sizeof(uint32_t)) {
    const auto a0 = *reinterpret_cast<const uint32_t*>(row_a0 + i);
    const auto a1 = *reinterpret_cast<const uint32_t*>(row_a1 + i);
    const auto b  = *reinterpret_cast<const uint32_t*>(row_b + i);
    total0 += __popc(a0 & b);
    total1 += __popc(a1 & b);
  }
}

/** Two asymmetric inner products over statically known document and query plane counts. */
template <int document_bits, int query_bits, size_t document_row_bytes, size_t query_row_bytes>
__device__ inline void code_inner_product_asymmetric_2x1(const uint8_t* row_a0,
                                                         const uint8_t* row_a1,
                                                         const uint8_t* row_b,
                                                         uint32_t& total0,
                                                         uint32_t& total1)
{
  constexpr size_t document_plane_stride = document_row_bytes / document_bits;
  constexpr size_t query_plane_stride    = query_row_bytes / query_bits;
  static_assert(query_plane_stride % sizeof(uint32_t) == 0);
#pragma unroll
  for (int p_query = 0; p_query < query_bits; ++p_query) {
#pragma unroll
    for (int p_document = 0; p_document < document_bits; ++p_document) {
      uint32_t partial0 = 0;
      uint32_t partial1 = 0;
      code_inner_product_binary_2x1<query_plane_stride>(row_a0 + p_document * document_plane_stride,
                                                        row_a1 + p_document * document_plane_stride,
                                                        row_b + p_query * query_plane_stride,
                                                        partial0,
                                                        partial1);
      total0 += partial0 << (p_document + p_query);
      total1 += partial1 << (p_document + p_query);
    }
  }
}

/** Two two-bit transposed inner products with a shared right operand. */
template <size_t n_bytes>
__device__ inline void code_inner_product_dibit_symmetric_2x1(const uint8_t* row_a0,
                                                              const uint8_t* row_a1,
                                                              const uint8_t* row_b,
                                                              uint32_t& total0,
                                                              uint32_t& total1)
{
  static_assert(n_bytes % (2 * sizeof(uint32_t)) == 0);
  const size_t stripe_size = n_bytes / 2;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      uint32_t partial0 = 0;
      uint32_t partial1 = 0;
      code_inner_product_binary_2x1<n_bytes / 2>(row_a0 + i * stripe_size,
                                                 row_a1 + i * stripe_size,
                                                 row_b + j * stripe_size,
                                                 partial0,
                                                 partial1);
      total0 += partial0 << (i + j);
      total1 += partial1 << (i + j);
    }
  }
}

/** Two four-bit transposed inner products with a shared right operand. */
template <size_t n_bytes>
__device__ inline void code_inner_product_int4_transposeHalfByte_symmetric_2x1(
  const uint8_t* row_a0,
  const uint8_t* row_a1,
  const uint8_t* row_b,
  uint32_t& total0,
  uint32_t& total1)
{
  static_assert(n_bytes % (4 * sizeof(uint32_t)) == 0);
  const size_t stripe_size = n_bytes / 4;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      uint32_t partial0 = 0;
      uint32_t partial1 = 0;
      code_inner_product_binary_2x1<n_bytes / 4>(row_a0 + i * stripe_size,
                                                 row_a1 + i * stripe_size,
                                                 row_b + j * stripe_size,
                                                 partial0,
                                                 partial1);
      total0 += partial0 << (i + j);
      total1 += partial1 << (i + j);
    }
  }
}

/** Two packed-nibble inner products with a shared right operand. */
template <size_t n_bytes>
__device__ inline void code_inner_product_int4_packed_nibble_symmetric_2x1(const uint8_t* row_a0,
                                                                           const uint8_t* row_a1,
                                                                           const uint8_t* row_b,
                                                                           uint32_t& total0,
                                                                           uint32_t& total1)
{
  static_assert(n_bytes % sizeof(uint32_t) == 0);
  constexpr uint32_t nibble_mask = 0x0F0F0F0Fu;
#pragma unroll 4
  for (size_t i = 0; i < n_bytes; i += sizeof(uint32_t)) {
    const auto a0     = *reinterpret_cast<const uint32_t*>(row_a0 + i);
    const auto a1     = *reinterpret_cast<const uint32_t*>(row_a1 + i);
    const auto b      = *reinterpret_cast<const uint32_t*>(row_b + i);
    const auto b_low  = b & nibble_mask;
    const auto b_high = (b >> 4) & nibble_mask;
    total0            = __dp4a(a0 & nibble_mask, b_low, total0);
    total0            = __dp4a((a0 >> 4) & nibble_mask, b_high, total0);
    total1            = __dp4a(a1 & nibble_mask, b_low, total1);
    total1            = __dp4a((a1 >> 4) & nibble_mask, b_high, total1);
  }
}

/** Two one-byte-per-code inner products with a shared right operand. */
template <size_t n_bytes>
__device__ inline void code_inner_product_unsigned_byte_2x1(const uint8_t* row_a0,
                                                            const uint8_t* row_a1,
                                                            const uint8_t* row_b,
                                                            uint32_t& total0,
                                                            uint32_t& total1,
                                                            uint8_t code_mask = 0xFFu)
{
  static_assert(n_bytes % sizeof(uint32_t) == 0);
  const uint32_t word_mask = uint32_t{code_mask} * 0x01010101u;
#pragma unroll 4
  for (size_t i = 0; i < n_bytes; i += sizeof(uint32_t)) {
    const auto a0 = *reinterpret_cast<const uint32_t*>(row_a0 + i) & word_mask;
    const auto a1 = *reinterpret_cast<const uint32_t*>(row_a1 + i) & word_mask;
    const auto b  = *reinterpret_cast<const uint32_t*>(row_b + i) & word_mask;
    total0        = __dp4a(a0, b, total0);
    total1        = __dp4a(a1, b, total1);
  }
}

#endif

}  // namespace preprocessing::quantize::bbq
}  // namespace CUVS_EXPORT cuvs
