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

using pair_id = uint32_t;
static_assert(std::has_single_bit(kCentroidBatchSize));
constexpr uint32_t kPairCentroidBits = std::countr_zero(kCentroidBatchSize);
constexpr pair_id kPaddingPair       = UINT32_MAX;
// Reserve the all-ones ID for padding.
constexpr uint32_t kMaxPairRows = kPaddingPair >> kPairCentroidBits;

__device__ __forceinline__ pair_id pack_pair(uint32_t row, uint32_t centroid)
{
  return (row << kPairCentroidBits) | centroid;
}

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
                               pair_id* pairs,
                               float* distances,
                               uint32_t* row_offsets,
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
  if (lane == 0) { row_offsets[row] = base; }
  if (base >= capacity) { return; }
  const uint32_t retained = min(size, capacity - base);
  const auto* survivors   = scratch + row * kCentroidBatchSize;
  const auto row_id       = static_cast<uint32_t>(row);
  // An even queue index aligns both the two-ID and two-distance stores.
  const uint32_t head = base & 1u;
  const uint32_t tail = (retained - head) & 1u;
  if (lane < head + tail) {
    const uint32_t q    = lane < head ? 0 : retained - 1;
    const int64_t entry = survivors[q];
    pairs[base + q]     = pack_pair(row_id, static_cast<uint32_t>(entry >> kSurvivorIdShift));
    distances[base + q] = __int_as_float(static_cast<int32_t>(entry));
  }
  for (uint32_t q = head + 2 * lane; q + 1 < retained; q += 2 * raft::warp_size()) {
    const int64_t first  = survivors[q];
    const int64_t second = survivors[q + 1];
    const uint32_t index = base + q;
    *reinterpret_cast<uint2*>(pairs + index) =
      make_uint2(pack_pair(row_id, static_cast<uint32_t>(first >> kSurvivorIdShift)),
                 pack_pair(row_id, static_cast<uint32_t>(second >> kSurvivorIdShift)));
    *reinterpret_cast<float2*>(distances + index) = make_float2(
      __int_as_float(static_cast<int32_t>(first)), __int_as_float(static_cast<int32_t>(second)));
  }
}

/** Finalize the capped queue with infinite-distance sentinels in its last warp. */
RAFT_KERNEL pad_pairs_kernel(uint32_t capacity,
                             pair_id* pairs,
                             float* distances,
                             uint32_t* pair_count)
{
  const uint32_t total  = min(*pair_count, capacity);
  const uint32_t padded = (total + raft::warp_size() - 1) & ~(uint32_t{raft::warp_size()} - 1);
  if (threadIdx.x < padded - total) {
    const uint32_t index = total + threadIdx.x;
    pairs[index]         = kPaddingPair;
    distances[index]     = CUDART_INF_F;
  }
  if (threadIdx.x == 0) { *pair_count = padded; }
}

/** Each warp compacts its live pairs locally. The queue positions stay fixed for row reduction. */
template <typename T, bool Aligned>
RAFT_KERNEL pair_tail_kernel(const T* rows,
                             const T* centroids,
                             uint32_t dim,
                             const pair_id* pairs,
                             const uint32_t* pair_count,
                             float* distances,
                             const float* bound)
{
  constexpr uint32_t kTileDims = kPairDimTileSize<T>;
  __shared__ uint32_t kept_slot[kWarps][kPairsPerWarp];
  __shared__ pair_id kept_pair[kWarps][kPairsPerWarp];
  __shared__ float kept_distance[kWarps][kPairsPerWarp];
  __shared__ float kept_bound[kWarps][kPairsPerWarp];
  const uint32_t total      = *pair_count;
  const uint32_t lane       = threadIdx.x % raft::warp_size();
  const uint32_t local_warp = threadIdx.x / raft::warp_size();
  const uint32_t warp       = (blockIdx.x * blockDim.x + threadIdx.x) / raft::warp_size();
  const uint32_t stride     = gridDim.x * kPairsPerBlock;
  for (uint32_t first = warp * kPairsPerWarp; first < total; first += stride) {
    uint32_t slot  = first + lane;
    pair_id pair   = lane < kPairsPerWarp ? pairs[slot] : kPaddingPair;
    uint32_t live  = __popc(__ballot_sync(raft::warp_full_mask(), pair != kPaddingPair));
    float distance = lane < kPairsPerWarp ? distances[slot] : CUDART_INF_F;
    float best     = lane < live ? bound[pair_row(pair)] : 0.f;
    uint32_t d     = kPartialDim;
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
      if (lane < live && (last || !keep)) { distances[slot] = keep ? distance : CUDART_INF_F; }
      if (last) { break; }
      const uint32_t mask = __ballot_sync(raft::warp_full_mask(), keep);
      if (keep) {
        const uint32_t position             = __popc(mask & ((1u << lane) - 1));
        kept_slot[local_warp][position]     = slot;
        kept_pair[local_warp][position]     = pair;
        kept_distance[local_warp][position] = distance;
        kept_bound[local_warp][position]    = best;
      }
      __syncwarp();
      live = __popc(mask);
      if (lane < live) {
        slot     = kept_slot[local_warp][lane];
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
          if (lane == 0) { distances[slot] = keep ? distance : CUDART_INF_F; }
          break;
        }
      }
    }
  }
}

/** Reduce each row's span, clipped to the pairs actually admitted to the queue. */
RAFT_KERNEL reduce_pairs_kernel(const pair_id* __restrict__ pairs,
                                const float* __restrict__ distances,
                                const uint32_t* __restrict__ row_offsets,
                                const int32_t* __restrict__ count,
                                int64_t n_rows,
                                uint32_t capacity,
                                uint32_t centroid_offset,
                                uint32_t* __restrict__ labels,
                                float* __restrict__ dist)
{
  const int64_t row   = (blockIdx.x * int64_t{blockDim.x} + threadIdx.x) / raft::warp_size();
  const uint32_t lane = threadIdx.x % raft::warp_size();
  if (row >= n_rows) { return; }
  const uint32_t size = static_cast<uint32_t>(count[row]);
  if (size == 0) { return; }
  const uint32_t base = row_offsets[row];
  if (base >= capacity) { return; }
  const uint32_t retained = min(size, capacity - base);
  // Prune already rejected candidates that did not improve the row's bound.
  if (retained == 1) {
    if (lane == 0) {
      const float distance = distances[base];
      if (distance < CUDART_INF_F) {
        labels[row] = centroid_offset + pair_centroid(pairs[base]);
        dist[row]   = distance;
      }
    }
    return;
  }
  float best          = CUDART_INF_F;
  uint32_t best_index = UINT32_MAX;
  for (uint32_t q = lane; q < retained; q += raft::warp_size()) {
    const float distance = distances[base + q];
    if (distance < best) {
      best       = distance;
      best_index = base + q;
    }
  }
  const auto nearest =
    raft::warpReduce(raft::KeyValuePair<uint32_t, float>{best_index, best}, raft::argmin_op{});
  if (lane == 0 && nearest.key != UINT32_MAX) {
    labels[row] = centroid_offset + pair_centroid(pairs[nearest.key]);
    dist[row]   = nearest.value;
  }
}

/**
 * Build a bounded queue, progressively prune it, and reduce the remaining distances per row.
 * Once the cap is reached, excess candidates are discarded in block reservation order.
 *
 * Nsight Systems, OpenAI FP32 on RTX PRO 6000 Blackwell (SM120): 4,990,000 vectors of dimension
 * 1,536, 50% training sample, 8,192 centroids, 10 iterations, two streams, 512 MiB workspace.
 * Full batches have 20,480 rows and 1,024 centroids. Prune uses 128-dimension tiles,
 * 16 pairs per warp, and four warps per block.
 *
 * Kernel                              Summed GPU time
 * build_pairs_kernel                  108.58 ms
 * pad_pairs_kernel                      7.54 ms
 * pair_tail_kernel                    629.64 ms
 * reduce_pairs_kernel                  89.12 ms
 *
 * Each kernel ran 8,784 times. The fused first filter, outside this function, took 1,179.56 ms.
 * Kernel durations are summed across overlapping streams. Fit took 1.83 s profiled and
 * 1.80 s unprofiled.
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
                 pair_id* pairs,
                 float* distances,
                 uint32_t* row_offsets,
                 uint32_t* pair_count,
                 uint32_t* labels,
                 float* dist)
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
                        distances,
                        row_offsets,
                        pair_count);
    raft::launch_kernel(
      handle, 1, raft::warp_size(), pad_pairs_kernel, capacity, pairs, distances, pair_count);
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
                        distances,
                        dist);
  }
  {
    common::nvtx::range<common::nvtx::domain::cuvs> range("pruned_1nn::pair_reduce");
    constexpr int64_t kRowsPerBlock = 4;
    raft::launch_kernel(handle,
                        raft::ceildiv(n_rows, kRowsPerBlock),
                        kRowsPerBlock * raft::warp_size(),
                        reduce_pairs_kernel,
                        pairs,
                        distances,
                        row_offsets,
                        count,
                        n_rows,
                        capacity,
                        centroid_offset,
                        labels,
                        dist);
  }
}

}  // namespace cuvs::distance::detail::pruned_nn
