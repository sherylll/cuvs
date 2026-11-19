/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "prune.cuh"

#include <raft/core/resource/device_properties.hpp>
#include <raft/util/integer_utils.hpp>

namespace cuvs::distance::detail::pruned_nn {

/** Queue entries index existing survivor slots, which already contain centroid IDs and distances.
 */
inline auto pair_queue_capacity(int64_t rows) -> uint32_t
{
  return static_cast<uint32_t>(std::max<int64_t>(
    raft::warp_size(), (rows * kCentroidBatchSize / 10) / raft::warp_size() * raft::warp_size()));
}

RAFT_KERNEL build_pairs_kernel(int64_t n_rows,
                               const int32_t* count,
                               int64_t* packed,
                               uint32_t capacity,
                               uint32_t* pairs,
                               uint32_t* pair_count)
{
  const int64_t row   = (blockIdx.x * int64_t{blockDim.x} + threadIdx.x) / raft::warp_size();
  const uint32_t lane = threadIdx.x % raft::warp_size();
  if (row >= n_rows) { return; }
  const uint32_t size = static_cast<uint32_t>(count[row]);
  if (size == 0) { return; }
  uint32_t base = lane == 0 ? atomicAdd(pair_count, size) : 0;
  base          = raft::shfl(base, 0);
  for (uint32_t q = lane; q < size; q += raft::warp_size()) {
    const auto slot = static_cast<uint32_t>(row * kCentroidBatchSize + q);
    if (base + q < capacity) {
      pairs[base + q] = slot;
    } else {
      const auto centroid = static_cast<uint32_t>(packed[slot] >> kSurvivorIdShift);
      packed[slot]        = pack_survivor(centroid, __int_as_float(0x7f800000));
    }
  }
}

/** Finalize the capped queue with infinite-distance sentinels in its last warp. */
RAFT_KERNEL pad_pairs_kernel(uint32_t capacity, uint32_t* pairs, uint32_t* pair_count)
{
  const uint32_t total  = min(*pair_count, capacity);
  const uint32_t padded = (total + raft::warp_size() - 1) & ~(uint32_t{raft::warp_size()} - 1);
  if (threadIdx.x < padded - total) { pairs[total + threadIdx.x] = UINT32_MAX; }
  if (threadIdx.x == 0) { *pair_count = padded; }
}

/** Each warp processes and compacts 32 pairs, re-filtering after each dimension tile. */
template <typename T, bool Aligned>
RAFT_KERNEL pair_tail_kernel(const T* rows,
                             const T* centroids,
                             uint32_t dim,
                             const uint32_t* pairs,
                             const uint32_t* pair_count,
                             int64_t* packed,
                             const float* bound)
{
  constexpr uint32_t kPairs = 32;
  constexpr uint32_t kWarps = 3;
  __shared__ uint32_t kept_slot[kWarps][raft::warp_size()];
  __shared__ uint32_t kept_centroid[kWarps][raft::warp_size()];
  __shared__ float kept_distance[kWarps][raft::warp_size()];
  __shared__ float kept_bound[kWarps][raft::warp_size()];
  const uint32_t total      = *pair_count;
  const uint32_t lane       = threadIdx.x % raft::warp_size();
  const uint32_t local_warp = threadIdx.x / raft::warp_size();
  const uint32_t warp       = (blockIdx.x * blockDim.x + threadIdx.x) / raft::warp_size();
  const uint32_t stride     = gridDim.x * blockDim.x;
  for (uint32_t first = warp * raft::warp_size(); first < total; first += stride) {
    uint32_t live       = raft::warp_size();
    uint32_t slot       = pairs[first + lane];
    const bool valid    = slot != UINT32_MAX;
    const int64_t entry = valid ? packed[slot] : 0;
    uint32_t centroid   = static_cast<uint32_t>(entry >> kSurvivorIdShift);
    uint32_t row        = valid ? slot / kCentroidBatchSize : 0;
    float distance =
      valid ? __int_as_float(static_cast<int32_t>(entry)) : __int_as_float(0x7f800000);
    float best = valid ? bound[row] : 0.f;
    for (uint32_t d = kPartialDim; live && d < dim; d += kDimTileSize) {
      float partial[kPairs] = {};
#pragma unroll
      for (uint32_t i = 0; i < kPairs; ++i) {
        if (i < live) {
          const uint32_t r = raft::shfl(row, i);
          const uint32_t c = raft::shfl(centroid, i);
          T row_tile[kDimTileSize / raft::warp_size()];
          T centroid_tile[kDimTileSize / raft::warp_size()];
          load_tile<kDimTileSize, Aligned>(
            rows + static_cast<int64_t>(r) * dim + d, dim - d, row_tile);
          load_tile<kDimTileSize, Aligned>(
            centroids + static_cast<int64_t>(c) * dim + d, dim - d, centroid_tile);
#pragma unroll
          for (uint32_t j = 0; j < kDimTileSize / raft::warp_size(); ++j) {
            const float diff =
              static_cast<float>(row_tile[j]) - static_cast<float>(centroid_tile[j]);
            partial[i] = fmaf(diff, diff, partial[i]);
          }
        }
      }
      using reduce = cub::WarpReduceBatched<float, kPairs>;
      typename reduce::TempStorage storage;
      const float tail = reduce(storage).Sum(partial);
      if (lane < live) { distance += tail; }
      const bool last   = d + kDimTileSize >= dim;
      const float ratio = last ? 1.f : adsampling_ratio(d + kDimTileSize, dim, kAdsamplingEps0);
      const bool keep   = lane < live && distance < best * ratio;
      if (lane < live && slot != UINT32_MAX && (last || !keep)) {
        packed[slot] = pack_survivor(centroid, keep ? distance : __int_as_float(0x7f800000));
      }
      if (last) { break; }
      const uint32_t mask = __ballot_sync(raft::warp_full_mask(), keep);
      if (keep) {
        const uint32_t position             = __popc(mask & ((1u << lane) - 1));
        kept_slot[local_warp][position]     = slot;
        kept_centroid[local_warp][position] = centroid;
        kept_distance[local_warp][position] = distance;
        kept_bound[local_warp][position]    = best;
      }
      __syncwarp();
      live = __popc(mask);
      if (lane < live) {
        slot     = kept_slot[local_warp][lane];
        centroid = kept_centroid[local_warp][lane];
        distance = kept_distance[local_warp][lane];
        best     = kept_bound[local_warp][lane];
        row      = slot / kCentroidBatchSize;
      }
      __syncwarp();
    }
  }
}

/** Reduce completed pair distances from their original row-oriented scratch slots. */
RAFT_KERNEL reduce_pairs_kernel(const int64_t* __restrict__ packed,
                                const int32_t* __restrict__ count,
                                uint32_t centroid_offset,
                                uint32_t* __restrict__ labels,
                                float* __restrict__ dist)
{
  const int64_t row   = blockIdx.x;
  const uint32_t lane = threadIdx.x;
  float best          = dist[row];
  uint32_t best_index = UINT32_MAX;
  const int64_t* list = packed + row * kCentroidBatchSize;
  for (uint32_t q = lane; q < static_cast<uint32_t>(count[row]); q += raft::warp_size()) {
    const int64_t entry  = list[q];
    const float distance = __int_as_float(static_cast<int32_t>(entry));
    if (distance < best) {
      best       = distance;
      best_index = centroid_offset + static_cast<uint32_t>(entry >> kSurvivorIdShift);
    }
  }
  const auto nearest =
    raft::warpReduce(raft::KeyValuePair<uint32_t, float>{best_index, best}, raft::argmin_op{});
  if (lane == 0 && nearest.key != UINT32_MAX) {
    labels[row] = nearest.key;
    dist[row]   = nearest.value;
  }
}

template <typename T>
void prune_pairs(const raft::resources& handle,
                 const T* rows,
                 int64_t n_rows,
                 const T* centroids,
                 uint32_t centroid_offset,
                 int64_t dim,
                 int64_t* packed,
                 const int32_t* count,
                 uint32_t* pairs,
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
                        raft::ceildiv(n_rows * raft::warp_size(), int64_t{128}),
                        128,
                        build_pairs_kernel,
                        n_rows,
                        count,
                        packed,
                        capacity,
                        pairs,
                        pair_count);
    raft::launch_kernel(
      handle, 1, raft::warp_size(), pad_pairs_kernel, capacity, pairs, pair_count);
  }
  {
    common::nvtx::range<common::nvtx::domain::cuvs> range("pruned_1nn::pair_tail");
    const auto kernel =
      dim % kDimTileSize == 0 ? pair_tail_kernel<T, true> : pair_tail_kernel<T, false>;
    const auto blocks =
      std::min<uint32_t>(raft::ceildiv(capacity, uint32_t{96}),
                         8 * raft::resource::get_device_properties(handle).multiProcessorCount);
    raft::launch_kernel(handle,
                        blocks,
                        96,
                        kernel,
                        rows,
                        centroids,
                        static_cast<uint32_t>(dim),
                        pairs,
                        pair_count,
                        packed,
                        dist);
  }
  {
    common::nvtx::range<common::nvtx::domain::cuvs> range("pruned_1nn::pair_reduce");
    raft::launch_kernel(handle,
                        n_rows,
                        raft::warp_size(),
                        reduce_pairs_kernel,
                        packed,
                        count,
                        centroid_offset,
                        labels,
                        dist);
  }
}

}  // namespace cuvs::distance::detail::pruned_nn
