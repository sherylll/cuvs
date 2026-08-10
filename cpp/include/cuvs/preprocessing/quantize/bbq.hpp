/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/export.hpp>
#include <cuvs/distance/distance.hpp>

#include <raft/core/device_mdspan.hpp>

#include <cstddef>
#include <cstdint>

namespace CUVS_EXPORT cuvs {
namespace preprocessing {
namespace quantize {
namespace bbq {

/**
 * @defgroup bbq Better Binary Quantization utilities
 * @{
 */

/**
 * Storage layout of the quantized component codes in each dataset row.
 *
 * `single_bit` Binary-packed bitstream (packAsBinary). Used for bits == 1.
 * `dibit` matches Lucene's transposeDibit/Used for bits == 2.
 transposeHalfByte layouts.
 * `unsigned_byte` one uint8_t per component. Used for bits == 7 and 8.
 */
enum class code_layout {
  single_bit,    /** Each dimension is quantized to a single bit and packed into bytes. Reflects
                  * OptimizedScalarQuantizer.packAsBinary.    During query time, the query vector is
                  * quantized to 4 bits per dimension. */
  dibit,         /** Each dimension is quantized to 2 bits (dibit) and packed into bytes. Reflects
                  * OptimizedScalarQuantizer.transposeDibit.         During query time, the query vector is
                  * quantized to 4 bits per dimension. */
  packed_nibble, /** Each dimension is quantized to 4 bits, two values are packed into each output
                  * byte. Reflects OffHeapScalarQuantizedVectorValues.packNibbles. During query
                  * time, the query vector is quantized to 4 bits per dimension. */
  seven_bit,     /** Each dimension is quantized to 7 bits and treated as a signed value. */
  unsigned_byte, /** Each dimension is quantized to 8 bits and treated as an unsigned value. */
  transpose_half_byte, /** Each dimension is quantized to 4 bits, optimized for bitwise operations.
                        * Reflects OptimizedScalarQuantizer.transposeHalfByte. the first bit of
                        * every dimension is in the first set dimensions bits, or (dimensions/8)
                        * bytes. The second, third, and fourth bits are in the second, third, and
                        * fourth set of dimensions bits, respectively. */
};

/**
 * View of a BBQ/Optimized-Scalar-Quantized dataset in device memory.
 *
 * For row `i`, the decoded centered component is
 *
 *   lower_intervals[i] +
 *     code * (upper_intervals[i] - lower_intervals[i]) / (2^bits - 1).
 *
 * `additional_corrections` contains the exact centered squared norm for
 * L2Expanded, or dot(original_vector, centroid) for cosine/inner product.
 */
struct bbq_dataset_view {
  raft::device_matrix_view<const uint8_t, int64_t> codes;
  raft::device_vector_view<const float, int64_t> lower_intervals;
  raft::device_vector_view<const float, int64_t> upper_intervals;
  raft::device_vector_view<const float, int64_t> additional_corrections;
  raft::device_vector_view<const int32_t, int64_t> quantized_component_sums;
  raft::device_vector_view<const float, int64_t> centroid;

  size_t dim;
  uint32_t bits;
  code_layout layout;
  cuvs::distance::DistanceType metric;
  float centroid_norm_sq;

  [[nodiscard]] constexpr size_t n_rows() const noexcept { return codes.extent(0); }

  [[nodiscard]] constexpr uint32_t levels() const noexcept { return uint32_t{1} << bits; }

  [[nodiscard]] constexpr size_t encoded_row_length() const noexcept
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
};

/** @} */  // end of bbq group

}  // namespace bbq
}  // namespace quantize
}  // namespace preprocessing
}  // namespace CUVS_EXPORT cuvs
