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

/**
 * Two cross-plane inner products over statically known document and query plane counts.
 * Not asymmetric-specific: a symmetric pair is just document_planes == query_planes, which is how
 * the 1x1 and 2t x 2t self-joins are computed. Supersedes the hand-written 2x2 and 4x4
 * specialisations this replaced.
 */
template <int document_planes, int query_planes, size_t document_row_bytes, size_t query_row_bytes>
__device__ inline void code_inner_product_planes_2x1(const uint8_t* row_a0,
                                                     const uint8_t* row_a1,
                                                     const uint8_t* row_b,
                                                     uint32_t& total0,
                                                     uint32_t& total1)
{
  constexpr size_t document_plane_stride = document_row_bytes / document_planes;
  constexpr size_t query_plane_stride    = query_row_bytes / query_planes;
  // Both operands are stepped by their own plane stride and then read as uint32_t, so both
  // strides -- not just the query's -- must be 4-byte aligned.
  static_assert(query_plane_stride % sizeof(uint32_t) == 0);
  static_assert(document_plane_stride % sizeof(uint32_t) == 0);
#pragma unroll
  for (int p_query = 0; p_query < query_planes; ++p_query) {
#pragma unroll
    for (int p_document = 0; p_document < document_planes; ++p_document) {
      const uint8_t* a0 = row_a0 + p_document * document_plane_stride;
      const uint8_t* a1 = row_a1 + p_document * document_plane_stride;
      const uint8_t* b  = row_b + p_query * query_plane_stride;
      uint32_t partial0 = 0;
      uint32_t partial1 = 0;
#pragma unroll 4
      for (size_t i = 0; i < query_plane_stride; i += sizeof(uint32_t)) {
        const auto wa0 = *reinterpret_cast<const uint32_t*>(a0 + i);
        const auto wa1 = *reinterpret_cast<const uint32_t*>(a1 + i);
        const auto wb  = *reinterpret_cast<const uint32_t*>(b + i);
        partial0 += __popc(wa0 & wb);
        partial1 += __popc(wa1 & wb);
      }
      total0 += partial0 << (p_document + p_query);
      total1 += partial1 << (p_document + p_query);
    }
  }
}

/** Two packed 4-bit inner products with a shared right operand. */
template <size_t n_bytes>
__device__ inline void code_inner_product_packed_4b_symmetric_2x1(const uint8_t* row_a0,
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
__device__ inline void code_inner_product_packed_8b_2x1(const uint8_t* row_a0,
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
