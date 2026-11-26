/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "prune.cuh"

#include <raft/core/resource/device_properties.hpp>
#include <raft/util/integer_utils.hpp>

#include <cub/warp/warp_scan.cuh>

#include <bit>

namespace cuvs::distance::detail::pruned_nn {

// Shared pair queue format
using pair_id                    = uint32_t;
constexpr uint32_t kPairsPerWarp = 16;
static_assert(std::has_single_bit(kPairsPerWarp) && kPairsPerWarp <= raft::warp_size());
static_assert(std::has_single_bit(kCentroidBatchSize));
constexpr uint32_t kPairCentroidBits = std::countr_zero(kCentroidBatchSize);
constexpr pair_id kPaddingPair       = UINT32_MAX;
// Reserve the all-ones ID for padding.
constexpr uint32_t kMaxPairRows = kPaddingPair >> kPairCentroidBits;

__device__ __forceinline__ uint32_t pair_row(pair_id pair) { return pair >> kPairCentroidBits; }

__device__ __forceinline__ uint32_t pair_centroid(pair_id pair)
{
  return pair & (kCentroidBatchSize - 1);
}

inline auto pair_queue_capacity(int64_t rows) -> uint32_t
{
  return static_cast<uint32_t>(std::max<int64_t>(
    raft::warp_size(), (rows * kCentroidBatchSize / 5) / raft::warp_size() * raft::warp_size()));
}

// Pair builder
constexpr int64_t kBuildRowsPerBlock  = 4;
constexpr uint32_t kBitmapWordBits    = 32;
constexpr uint32_t kBitmapWordsPerRow = kCentroidBatchSize / kBitmapWordBits;
static_assert(kCentroidBatchSize == kBitmapWordBits * raft::warp_size());

RAFT_KERNEL build_bitmap_pairs_kernel(int64_t n_rows,
                                      int bitmap_words,
                                      const uint32_t* bitmap,
                                      uint32_t capacity,
                                      pair_id* pairs,
                                      const packed_winner* winners,
                                      float* bound,
                                      uint32_t* pair_count)
{
  constexpr int kWarpSize = raft::warp_size();

  using BitmapScan = cub::WarpScan<uint32_t, kWarpSize>;
  // The power-of-two bitmap scan uses shuffles and empty CUB storage.
  __shared__ BitmapScan::TempStorage bitmap_scan[kBuildRowsPerBlock];
  __shared__ uint32_t row_counts[kBuildRowsPerBlock];
  __shared__ uint32_t queue_offsets[kBuildRowsPerBlock];

  const int warp           = threadIdx.x / kWarpSize;
  const int lane           = raft::laneId();
  const uint32_t first_row = blockIdx.x * kBuildRowsPerBlock;
  const uint32_t row       = first_row + warp;

  // Each lane owns one word. The final centroid batch may leave later words unwritten.
  uint32_t word = row < n_rows && lane < bitmap_words ? bitmap[row * kBitmapWordsPerRow + lane] : 0;
  uint32_t word_offset, row_count;
  BitmapScan(bitmap_scan[warp]).ExclusiveSum(__popc(word), word_offset, row_count);
  if (lane == 0) { row_counts[warp] = row_count; }

  __syncthreads();

  // Keep the pruning bound fixed while the tail kernel updates winners.
  if (threadIdx.x < kBuildRowsPerBlock && row_counts[lane] != 0) {
    bound[first_row + lane] = winner_distance(winners[first_row + lane]);
  }

  // Reserve one span per block and give each row its part of that span.
  if (threadIdx.x == 0) {
    uint32_t row_offsets[kBuildRowsPerBlock];
    uint32_t block_count = 0;
#pragma unroll
    for (int i = 0; i < kBuildRowsPerBlock; ++i) {
      row_offsets[i] = block_count;
      block_count += row_counts[i];
    }
    const uint32_t reserved = block_count != 0 ? atomicAdd(pair_count, block_count) : 0;
#pragma unroll
    for (int i = 0; i < kBuildRowsPerBlock; ++i) {
      queue_offsets[i] = reserved + row_offsets[i];
    }
  }
  __syncthreads();

  const uint32_t queue_offset = queue_offsets[warp];
  if (queue_offset >= capacity) { return; }
  const uint32_t retained = min(row_count, capacity - queue_offset);
  for (uint32_t index = word_offset; word != 0 && index < retained; ++index) {
    const uint32_t centroid     = lane * kBitmapWordBits + (__ffs(word) - 1);
    pairs[queue_offset + index] = (row << kPairCentroidBits) | centroid;
    word &= word - 1;
  }
}

/** Finalize the capped queue with invalid pair IDs in its last warp. */
RAFT_KERNEL pad_pairs_kernel(uint32_t capacity, pair_id* pairs, uint32_t* pair_count)
{
  const uint32_t total  = min(*pair_count, capacity);
  const uint32_t padded = (total + kPairsPerWarp - 1) & ~(kPairsPerWarp - 1);
  if (threadIdx.x < padded - total) {
    const uint32_t index = total + threadIdx.x;
    pairs[index]         = kPaddingPair;
  }
  // All threads must read the original count before lane zero overwrites it.
  __syncthreads();
  if (threadIdx.x == 0) { *pair_count = padded; }
}

// Pair pruning
constexpr uint32_t kPruneWarps    = 4;
constexpr uint32_t kPairThreads   = kPruneWarps * raft::warp_size();
constexpr uint32_t kPairsPerBlock = kPruneWarps * kPairsPerWarp;
constexpr int kPruneBlocksPerSm   = 16;

template <typename T>
constexpr uint32_t kPairPruneTileSize = std::is_same_v<T, float> ? 128 : kDimTileSize;

/** Each warp compacts its live pairs locally and submits full distances to each row's winner. */
template <typename T, bool Aligned, uint32_t PartialDim>
RAFT_KERNEL pair_tail_kernel(const T* rows,
                             const T* centroids,
                             uint32_t dim,
                             const pair_id* pairs,
                             const uint32_t* pair_count,
                             const float* bound,
                             uint32_t centroid_offset,
                             packed_winner* winners)
{
  constexpr uint32_t kTileDims     = kPairPruneTileSize<T>;
  constexpr uint32_t kFirstTileDim = PartialDim + kTileDims;
  static_assert(PartialDim % kTileDims == 0);
  __shared__ pair_id kept_pair[kPruneWarps][kPairsPerWarp];
  __shared__ float kept_distance[kPruneWarps][kPairsPerWarp];
  __shared__ float kept_bound[kPruneWarps][kPairsPerWarp];
  const uint32_t total      = *pair_count;
  const uint32_t lane       = threadIdx.x % raft::warp_size();
  const uint32_t local_warp = threadIdx.x / raft::warp_size();
  const uint32_t warp       = (blockIdx.x * blockDim.x + threadIdx.x) / raft::warp_size();
  const uint32_t stride     = gridDim.x * kPairsPerBlock;
  for (uint32_t first = warp * kPairsPerWarp; first < total; first += stride) {
    pair_id pair  = lane < kPairsPerWarp ? pairs[first + lane] : kPaddingPair;
    uint32_t live = __popc(__ballot_sync(raft::warp_full_mask(), pair != kPaddingPair));
    // Keep the first tile full-width without loading through a padding ID.
    if (live < kPairsPerWarp) {
      const pair_id first_pair = raft::shfl(pair, 0);
      if (lane >= live) { pair = first_pair; }
    }
    float best                 = lane < live ? bound[pair_row(pair)] : 0.f;
    const auto accumulate_tile = [&](pair_id ids, uint32_t d, float sum) {
      const uint32_t r = pair_row(ids);
      const uint32_t c = pair_centroid(ids);
      T row_tile[kTileDims / raft::warp_size()];
      T centroid_tile[kTileDims / raft::warp_size()];
      load_tile<kTileDims, Aligned>(rows + static_cast<int64_t>(r) * dim + d, dim - d, row_tile);
      load_tile<kTileDims, Aligned>(
        centroids + static_cast<int64_t>(c) * dim + d, dim - d, centroid_tile);
#pragma unroll
      for (uint32_t j = 0; j < kTileDims / raft::warp_size(); ++j) {
        const float diff = static_cast<float>(row_tile[j]) - static_cast<float>(centroid_tile[j]);
        sum              = fmaf(diff, diff, sum);
      }
      return sum;
    };

    float first_partial[kPairsPerWarp] = {};
    // Keep the partial sums in registers and reduce once at the first bound check.
#pragma unroll 1
    for (uint32_t d = 0; d < kFirstTileDim; d += kTileDims) {
#pragma unroll
      for (uint32_t q = 0; q < kPairsPerWarp; ++q) {
        first_partial[q] = accumulate_tile(raft::shfl(pair, q), d, first_partial[q]);
      }
    }
    using FirstReduce = cub::WarpReduceBatched<float, kPairsPerWarp>;
    typename FirstReduce::TempStorage first_storage;
    float distance = FirstReduce(first_storage).Sum(first_partial);

    uint32_t d = kFirstTileDim;
    for (;;) {
      const bool last   = d >= dim;
      const float ratio = last ? 1.f : adsampling_ratio(d, dim, kAdsamplingEps0);
      const bool keep   = lane < live && distance < best * ratio;
      if (last) {
        if (keep) {
          atomicMin(winners + pair_row(pair),
                    pack_winner(distance, centroid_offset + pair_centroid(pair)));
        }
        break;
      }
      const uint32_t mask = __ballot_sync(raft::warp_full_mask(), keep);
      if (keep) {
        const uint32_t position             = __popc(mask & ((1u << lane) - 1));
        kept_pair[local_warp][position]     = pair;
        kept_distance[local_warp][position] = distance;
        kept_bound[local_warp][position]    = best;
      }
      __syncwarp();
      live = __popc(mask);
      if (lane < live) {
        pair     = kept_pair[local_warp][lane];
        distance = kept_distance[local_warp][lane];
        best     = kept_bound[local_warp][lane];
      }
      __syncwarp();
      if (live <= 1) { break; }
      const auto process_batch = [&](auto tile, uint32_t i) {
        constexpr uint32_t kTile = decltype(tile)::value;
        float partial[kTile];
#pragma unroll
        for (uint32_t q = 0; q < kTile; ++q) {
          partial[q] = accumulate_tile(raft::shfl(pair, i + q), d, 0.f);
        }
        if constexpr (kTile == 1) {
          const float tail = raft::warpReduce(partial[0]);
          if (lane == i) { distance += tail; }
        } else {
          using Reduce = cub::WarpReduceBatched<float, kTile>;
          typename Reduce::TempStorage storage;
          const uint32_t chunk_lane = (lane - i) % raft::warp_size();
          const float tail          = raft::shfl(Reduce(storage).Sum(partial), chunk_lane);
          if (lane >= i && lane < i + kTile) { distance += tail; }
        }
      };
      uint32_t i                      = 0;
      const auto process_live_batches = [&](auto tile) {
        constexpr uint32_t kTile = decltype(tile)::value;
        for (; i + kTile <= live; i += kTile) {
          process_batch(tile, i);
        }
      };
      process_live_batches(std::integral_constant<uint32_t, 8>{});
      process_live_batches(std::integral_constant<uint32_t, 4>{});
      process_live_batches(std::integral_constant<uint32_t, 1>{});
      d += kTileDims;
    }
    if (live == 1 && d < dim) {
      // The remaining pair is in lane zero. Its distance no longer needs a batched reduction.
      const pair_id ids      = raft::shfl(pair, 0);
      const auto* row_values = rows + static_cast<int64_t>(pair_row(ids)) * dim;
      const auto* centroid   = centroids + static_cast<int64_t>(pair_centroid(ids)) * dim;
      distance               = raft::shfl(distance, 0);
      best                   = raft::shfl(best, 0);
      for (; d < dim; d += kTileDims) {
        T row_tile[kTileDims / raft::warp_size()];
        T centroid_tile[kTileDims / raft::warp_size()];
        load_tile<kTileDims, Aligned>(row_values + d, dim - d, row_tile);
        load_tile<kTileDims, Aligned>(centroid + d, dim - d, centroid_tile);
        float partial = 0.f;
#pragma unroll
        for (uint32_t j = 0; j < kTileDims / raft::warp_size(); ++j) {
          const float diff = static_cast<float>(row_tile[j]) - static_cast<float>(centroid_tile[j]);
          partial          = fmaf(diff, diff, partial);
        }
        distance += raft::warpReduce(partial);
        const bool last   = d + kTileDims >= dim;
        const float ratio = last ? 1.f : adsampling_ratio(d + kTileDims, dim, kAdsamplingEps0);
        const bool keep   = distance < best * ratio;
        if (last || !keep) {
          if (lane == 0 && keep) {
            atomicMin(winners + pair_row(ids),
                      pack_winner(distance, centroid_offset + pair_centroid(ids)));
          }
          break;
        }
      }
    }
  }
}

/** Publish the final labels and distances after the last centroid batch. */
RAFT_KERNEL unpack_winners_kernel(int64_t n_rows,
                                  const packed_winner* winners,
                                  uint32_t* labels,
                                  float* dist)
{
  const int64_t row = blockIdx.x * int64_t{blockDim.x} + threadIdx.x;
  if (row < n_rows) {
    const packed_winner winner = winners[row];
    labels[row]                = static_cast<uint32_t>(winner);
    dist[row]                  = winner_distance(winner);
  }
}

/** Build and pad the bitmap-derived queue, then progressively prune and update row winners. */
template <uint32_t PartialDim, typename T>
void prune_pairs(const raft::resources& handle,
                 const T* rows,
                 int64_t n_rows,
                 const T* centroids,
                 uint32_t centroid_offset,
                 int64_t dim,
                 const uint32_t* bitmap,
                 int bitmap_words,
                 pair_id* pairs,
                 uint32_t* pair_count,
                 float* bound,
                 packed_winner* winners)
{
  const auto capacity = pair_queue_capacity(n_rows);
  const bool aligned  = (dim - PartialDim) % kPairPruneTileSize<T> == 0;
  {
    common::nvtx::range<common::nvtx::domain::cuvs> range("pruned_1nn::build_pairs");
    RAFT_CUDA_TRY(cudaMemsetAsync(
      pair_count, 0, sizeof(uint32_t), raft::resource::get_cuda_stream(handle).get()));
    raft::launch_kernel(handle,
                        raft::ceildiv(n_rows, kBuildRowsPerBlock),
                        kBuildRowsPerBlock * raft::warp_size(),
                        build_bitmap_pairs_kernel,
                        n_rows,
                        bitmap_words,
                        bitmap,
                        capacity,
                        pairs,
                        winners,
                        bound,
                        pair_count);
    raft::launch_kernel(handle, 1, kPairsPerWarp, pad_pairs_kernel, capacity, pairs, pair_count);
  }
  {
    common::nvtx::range<common::nvtx::domain::cuvs> range("pruned_1nn::pair_tail");
    const auto kernel =
      aligned ? pair_tail_kernel<T, true, PartialDim> : pair_tail_kernel<T, false, PartialDim>;
    const auto blocks = std::min<uint32_t>(
      raft::ceildiv(capacity, kPairsPerBlock),
      kPruneBlocksPerSm * raft::resource::get_device_properties(handle).multiProcessorCount);
    raft::launch_kernel(handle,
                        blocks,
                        kPairThreads,
                        kernel,
                        rows,
                        centroids,
                        static_cast<uint32_t>(dim),
                        pairs,
                        pair_count,
                        bound,
                        centroid_offset,
                        winners);
  }
}

}  // namespace cuvs::distance::detail::pruned_nn
