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

constexpr uint32_t kWarps            = 4;
constexpr uint32_t kPairThreads      = kWarps * raft::warp_size();
constexpr uint32_t kPairsPerWarp     = 16;
constexpr uint32_t kPairsPerBlock    = kWarps * kPairsPerWarp;
constexpr int64_t kBuildRowsPerBlock = 4;

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

RAFT_KERNEL build_pairs_kernel(int64_t n_rows,
                               const int32_t* counts,
                               const int64_t* scratch,
                               uint32_t capacity,
                               queue_entry* pairs,
                               const packed_winner* winners,
                               float* bound,
                               uint32_t* pair_count)
{
  using RowScan = cub::WarpScan<uint32_t, kBuildRowsPerBlock>;
  __shared__ RowScan::TempStorage scan_storage;
  __shared__ uint32_t offsets[kBuildRowsPerBlock + 1];
  const int64_t first = blockIdx.x * kBuildRowsPerBlock;
  const uint32_t warp = threadIdx.x / raft::warp_size();
  const uint32_t lane = threadIdx.x % raft::warp_size();
  if (threadIdx.x < kBuildRowsPerBlock) {
    // Reserve once per block. The scan assigns adjacent spans to the row warps.
    constexpr uint32_t kScanMask = (1u << kBuildRowsPerBlock) - 1;
    const uint32_t size = first + lane < n_rows ? static_cast<uint32_t>(counts[first + lane]) : 0;
    // Prune reads this snapshot while it updates the packed winners.
    if (size != 0) { bound[first + lane] = winner_distance(winners[first + lane]); }
    uint32_t prefix, total;
    RowScan(scan_storage).ExclusiveSum(size, prefix, total);
    const uint32_t reserved = lane == 0 && total != 0 ? atomicAdd(pair_count, total) : 0;
    const uint32_t base     = raft::shfl(reserved, 0, kBuildRowsPerBlock, kScanMask);
    offsets[lane]           = base + prefix;
    if (lane == 0) { offsets[kBuildRowsPerBlock] = base + total; }
  }
  __syncthreads();
  const uint32_t base = offsets[warp];
  const uint32_t size = offsets[warp + 1] - base;
  if (size == 0) { return; }
  const int64_t row = first + warp;
  if (base >= capacity) { return; }
  const uint32_t retained = min(size, capacity - base);
  const auto* survivors   = scratch + row * kCentroidBatchSize;
  // The scratch entry already has the centroid and partial distance in the right bits.
  const queue_entry row_bits = static_cast<queue_entry>(row)
                               << (kSurvivorIdShift + kPairCentroidBits);
  // Peel a single entry at either end so two consecutive stores can use a 16-byte vector.
  const uint32_t head = base & 1u;
  const uint32_t tail = (retained - head) & 1u;
  if (lane < head + tail) {
    const uint32_t q = lane < head ? 0 : retained - 1;
    pairs[base + q]  = row_bits | static_cast<queue_entry>(survivors[q]);
  }
  for (uint32_t q = head + 2 * lane; q + 1 < retained; q += 2 * raft::warp_size()) {
    const queue_entry first  = row_bits | static_cast<queue_entry>(survivors[q]);
    const queue_entry second = row_bits | static_cast<queue_entry>(survivors[q + 1]);
    *reinterpret_cast<ulonglong2*>(pairs + base + q) = make_ulonglong2(first, second);
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

/**
 * Build and pad the queue, then progressively prune pairs and update each row's winner.
 *
 * Nsight Systems on RTX PRO 6000 (SM120), OpenAI FP32 with 4.99M rows of dimension 1,536,
 * 50% training, 8,192 centroids, 10 iterations, and a 512 MiB workspace budget.
 * Queue build and padding took 125.98 ms (117.52 ms in build_pairs_kernel and 8.46 ms in
 * pad_pairs_kernel). Pruning took 692.28 ms in pair_tail_kernel.
 * Each kernel ran 8,784 times. GPU durations are summed across two overlapping streams.
 */
template <typename T>
void prune_pairs(const raft::resources& handle,
                 const T* rows,
                 int64_t n_rows,
                 const T* centroids,
                 uint32_t centroid_offset,
                 int64_t dim,
                 const int64_t* scratch,
                 const int32_t* count,
                 queue_entry* pairs,
                 uint32_t* pair_count,
                 float* bound,
                 packed_winner* winners)
{
  const auto capacity = pair_queue_capacity(n_rows);
  {
    common::nvtx::range<common::nvtx::domain::cuvs> range("pruned_1nn::build_pairs");
    RAFT_CUDA_TRY(cudaMemsetAsync(
      pair_count, 0, sizeof(uint32_t), raft::resource::get_cuda_stream(handle).get()));
    raft::launch_kernel(handle,
                        raft::ceildiv(n_rows, kBuildRowsPerBlock),
                        kBuildRowsPerBlock * raft::warp_size(),
                        build_pairs_kernel,
                        n_rows,
                        count,
                        scratch,
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
    const auto kernel =
      dim % kPairDimTileSize<T> == 0 ? pair_tail_kernel<T, true> : pair_tail_kernel<T, false>;
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
