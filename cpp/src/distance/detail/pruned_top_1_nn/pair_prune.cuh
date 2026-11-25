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

constexpr uint32_t kWarps             = 4;
constexpr uint32_t kPairThreads       = kWarps * raft::warp_size();
constexpr uint32_t kPairsPerWarp      = 16;
constexpr uint32_t kPairsPerBlock     = kWarps * kPairsPerWarp;
constexpr int64_t kBuildRowsPerBlock  = 4;
constexpr uint32_t kBitmapWordBits    = 32;
constexpr uint32_t kBitmapWordsPerRow = kCentroidBatchSize / kBitmapWordBits;
static_assert(kCentroidBatchSize == kBitmapWordBits * raft::warp_size());

template <typename T>
constexpr uint32_t kPairDimTileSize = std::is_same_v<T, float> ? 128 : kDimTileSize;

using pair_id     = uint32_t;
using queue_entry = uint64_t;
static_assert(std::has_single_bit(kCentroidBatchSize));
constexpr uint32_t kPairCentroidBits = std::countr_zero(kCentroidBatchSize);
constexpr pair_id kPaddingPair       = UINT32_MAX;
constexpr queue_entry kPaddingEntry = (queue_entry{kPaddingPair} << kSurvivorIdShift) | 0x7f800000u;
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

template <typename T, bool Aligned>
RAFT_KERNEL build_bitmap_pairs_kernel(int64_t n_rows,
                                      int dim,
                                      int bitmap_words,
                                      const uint32_t* bitmap,
                                      const T* rows,
                                      const T* centroids,
                                      uint32_t capacity,
                                      queue_entry* pairs,
                                      const packed_winner* winners,
                                      float* bound,
                                      uint32_t* pair_count)
{
  constexpr int kWarpSize    = raft::warp_size();
  constexpr int kDimsPerLane = kPartialDim / kWarpSize;

  using BitmapScan = cub::WarpScan<uint32_t, kWarpSize>;
  using RowScan    = cub::WarpScan<uint32_t, kBuildRowsPerBlock>;
  // Both power-of-two scans use shuffles and empty CUB storage.
  __shared__ BitmapScan::TempStorage bitmap_scan[kBuildRowsPerBlock];
  __shared__ RowScan::TempStorage row_scan;
  __shared__ uint32_t row_counts[kBuildRowsPerBlock];
  __shared__ uint32_t queue_offsets[kBuildRowsPerBlock];
  __shared__ uint16_t survivor_centroids[kBuildRowsPerBlock][kCentroidBatchSize];

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

  // Reserve one span per block and give each row its part of that span.
  if (threadIdx.x < kBuildRowsPerBlock) {
    const uint32_t count = row_counts[lane];
    // Keep the pruning bound fixed while the tail kernel updates winners.
    if (count != 0) { bound[first_row + lane] = winner_distance(winners[first_row + lane]); }
    uint32_t row_offset, block_count;
    RowScan(row_scan).ExclusiveSum(count, row_offset, block_count);
    const uint32_t reserved =
      lane == 0 && block_count != 0 ? atomicAdd(pair_count, block_count) : 0;
    constexpr uint32_t kRowMask = (1u << kBuildRowsPerBlock) - 1;
    queue_offsets[lane] = raft::shfl(reserved, 0, kBuildRowsPerBlock, kRowMask) + row_offset;
  }
  __syncthreads();

  const uint32_t queue_offset = queue_offsets[warp];
  if (queue_offset >= capacity || row_count == 0) { return; }
  const uint32_t retained = min(row_count, capacity - queue_offset);

  // Enumerate retained centroids in bitmap order for this row's warp.
  for (uint32_t index = word_offset; word != 0 && index < retained; ++index) {
    survivor_centroids[warp][index] = lane * kBitmapWordBits + (__ffs(word) - 1);
    word &= word - 1;
  }
  raft::warpFence();

  T row_tile[kDimsPerLane];
  load_tile<kPartialDim, Aligned>(rows + int64_t{row} * dim, kPartialDim, row_tile);
  const auto partial_distance = [&](uint32_t index) {
    const uint32_t centroid = survivor_centroids[warp][index];
    T centroid_tile[kDimsPerLane];
    load_tile<kPartialDim, Aligned>(
      centroids + int64_t{centroid} * dim, kPartialDim, centroid_tile);
    float distance = 0.f;
#pragma unroll
    for (int d = 0; d < kDimsPerLane; ++d) {
      const float delta = static_cast<float>(row_tile[d]) - static_cast<float>(centroid_tile[d]);
      distance          = fmaf(delta, delta, distance);
    }
    return distance;
  };
  const auto store_pair = [&](uint32_t index, float distance) {
    const pair_id id            = (row << kPairCentroidBits) | survivor_centroids[warp][index];
    pairs[queue_offset + index] = (queue_entry{id} << kSurvivorIdShift) | __float_as_uint(distance);
  };

  constexpr uint32_t kBatchSize = 8;
  using BatchReduce             = cub::WarpReduceBatched<float, kBatchSize>;
  typename BatchReduce::TempStorage storage;
  uint32_t index = 0;
  for (; index + kBatchSize <= retained; index += kBatchSize) {
    float partial[kBatchSize];
#pragma unroll
    for (uint32_t j = 0; j < kBatchSize; ++j) {
      partial[j] = partial_distance(index + j);
    }
    const float distance = BatchReduce(storage).Sum(partial);
    if (lane < kBatchSize) { store_pair(index + lane, distance); }
  }
  for (; index < retained; ++index) {
    const float distance = raft::warpReduce(partial_distance(index));
    if (lane == 0) { store_pair(index, distance); }
  }
}

/** Finalize the capped queue with invalid pair IDs in its last warp. */
RAFT_KERNEL pad_pairs_kernel(uint32_t capacity, queue_entry* pairs, uint32_t* pair_count)
{
  const uint32_t total  = min(*pair_count, capacity);
  const uint32_t padded = (total + raft::warp_size() - 1) & ~(uint32_t{raft::warp_size()} - 1);
  if (threadIdx.x < padded - total) {
    const uint32_t index = total + threadIdx.x;
    pairs[index]         = kPaddingEntry;
  }
  if (threadIdx.x == 0) { *pair_count = padded; }
}

/** Each warp compacts its live pairs locally and submits full distances to each row's winner. */
template <typename T, bool Aligned>
RAFT_KERNEL pair_tail_kernel(const T* rows,
                             const T* centroids,
                             uint32_t dim,
                             const queue_entry* pairs,
                             const uint32_t* pair_count,
                             const float* bound,
                             uint32_t centroid_offset,
                             packed_winner* winners)
{
  constexpr uint32_t kTileDims = kPairDimTileSize<T>;
  __shared__ pair_id kept_pair[kWarps][kPairsPerWarp];
  __shared__ float kept_distance[kWarps][kPairsPerWarp];
  __shared__ float kept_bound[kWarps][kPairsPerWarp];
  const uint32_t total      = *pair_count;
  const uint32_t lane       = threadIdx.x % raft::warp_size();
  const uint32_t local_warp = threadIdx.x / raft::warp_size();
  const uint32_t warp       = (blockIdx.x * blockDim.x + threadIdx.x) / raft::warp_size();
  const uint32_t stride     = gridDim.x * kPairsPerBlock;
  for (uint32_t first = warp * kPairsPerWarp; first < total; first += stride) {
    const queue_entry entry = lane < kPairsPerWarp ? pairs[first + lane] : kPaddingEntry;
    pair_id pair            = static_cast<pair_id>(entry >> kSurvivorIdShift);
    uint32_t live           = __popc(__ballot_sync(raft::warp_full_mask(), pair != kPaddingPair));
    float distance          = __uint_as_float(static_cast<uint32_t>(entry));
    float best              = lane < live ? bound[pair_row(pair)] : 0.f;
    uint32_t d              = kPartialDim;
    for (; live > 1 && d < dim; d += kTileDims) {
      uint32_t i = 0;
      for_each_halving<16>([&](auto tile) {
        constexpr uint32_t kTile = decltype(tile)::value;
        for (; i + kTile <= live; i += kTile) {
          float partial[kTile];
#pragma unroll
          for (uint32_t q = 0; q < kTile; ++q) {
            const pair_id ids = raft::shfl(pair, i + q);
            const uint32_t r  = pair_row(ids);
            const uint32_t c  = pair_centroid(ids);
            T row_tile[kTileDims / raft::warp_size()];
            T centroid_tile[kTileDims / raft::warp_size()];
            load_tile<kTileDims, Aligned>(
              rows + static_cast<int64_t>(r) * dim + d, dim - d, row_tile);
            load_tile<kTileDims, Aligned>(
              centroids + static_cast<int64_t>(c) * dim + d, dim - d, centroid_tile);
            float sum = 0.f;
#pragma unroll
            for (uint32_t j = 0; j < kTileDims / raft::warp_size(); ++j) {
              const float diff =
                static_cast<float>(row_tile[j]) - static_cast<float>(centroid_tile[j]);
              sum = fmaf(diff, diff, sum);
            }
            partial[q] = sum;
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
        }
      });
      const bool last   = d + kTileDims >= dim;
      const float ratio = last ? 1.f : adsampling_ratio(d + kTileDims, dim, kAdsamplingEps0);
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
    }
    if (live == 1) {
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
template <typename T>
void prune_pairs(const raft::resources& handle,
                 const T* rows,
                 int64_t n_rows,
                 const T* centroids,
                 uint32_t centroid_offset,
                 int64_t dim,
                 const uint32_t* bitmap,
                 int bitmap_words,
                 queue_entry* pairs,
                 uint32_t* pair_count,
                 float* bound,
                 packed_winner* winners)
{
  const auto capacity = pair_queue_capacity(n_rows);
  const bool aligned  = dim % kPairDimTileSize<T> == 0;
  const auto build_kernel =
    aligned ? build_bitmap_pairs_kernel<T, true> : build_bitmap_pairs_kernel<T, false>;
  {
    common::nvtx::range<common::nvtx::domain::cuvs> range("pruned_1nn::build_pairs");
    RAFT_CUDA_TRY(cudaMemsetAsync(
      pair_count, 0, sizeof(uint32_t), raft::resource::get_cuda_stream(handle).get()));
    raft::launch_kernel(handle,
                        raft::ceildiv(n_rows, kBuildRowsPerBlock),
                        kBuildRowsPerBlock * raft::warp_size(),
                        build_kernel,
                        n_rows,
                        static_cast<int>(dim),
                        bitmap_words,
                        bitmap,
                        rows,
                        centroids,
                        capacity,
                        pairs,
                        winners,
                        bound,
                        pair_count);
    raft::launch_kernel(
      handle, 1, raft::warp_size(), pad_pairs_kernel, capacity, pairs, pair_count);
  }
  {
    common::nvtx::range<common::nvtx::domain::cuvs> range("pruned_1nn::pair_tail");
    const auto kernel          = aligned ? pair_tail_kernel<T, true> : pair_tail_kernel<T, false>;
    constexpr int kBlocksPerSm = 16;
    const auto blocks          = std::min<uint32_t>(
      raft::ceildiv(capacity, kPairsPerBlock),
      kBlocksPerSm * raft::resource::get_device_properties(handle).multiProcessorCount);
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
