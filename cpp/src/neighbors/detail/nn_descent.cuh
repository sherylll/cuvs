/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ann_utils.cuh"
#include "neighbors_device_intrinsics.cuh"
#include "nn_descent_gnnd.hpp"

#include "../../core/nvtx.hpp"
#include "../../core/omp_wrapper.hpp"
#include "../bbq.cuh"
#include "../bbq_optimized.cuh"
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/nn_descent.hpp>
#include <cuvs/preprocessing/quantize/bbq.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/pinned_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/matrix/init.cuh>
#include <raft/matrix/slice.cuh>
#include <raft/util/arch.cuh>  // raft::util::arch::SM_*
#include <raft/util/cuda_dev_essentials.cuh>
#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/pow2_utils.cuh>

#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>

#include <mma.h>

#include <cstdlib>
#include <limits>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <type_traits>

namespace cuvs::neighbors::nn_descent::detail {

template <typename DataT, typename IdxT>
using bbq_device_quantizer_view = cuvs::preprocessing::quantize::bbq::
  bbq_quantizer_view<DataT, IdxT, cuvs::neighbors::detail::device_view_accessor<const DataT>>;
using bbq_layout = cuvs::preprocessing::quantize::bbq::bbq_code_layout;

template <typename Index_t>
struct ResultItem;

template <>
class ResultItem<int> {
 private:
  using Index_t = int;
  Index_t id_;
  DistData_t dist_;

 public:
  __host__ __device__ ResultItem()
    : id_(std::numeric_limits<Index_t>::max()), dist_(std::numeric_limits<DistData_t>::max()) {};
  __host__ __device__ ResultItem(const Index_t id_with_flag, const DistData_t dist)
    : id_(id_with_flag), dist_(dist) {};
  __host__ __device__ bool is_new() const { return id_ >= 0; }
  __host__ __device__ Index_t& id_with_flag() { return id_; }
  __host__ __device__ Index_t id() const
  {
    if (is_new()) return id_;
    return -id_ - 1;
  }
  __host__ __device__ DistData_t& dist() { return dist_; }

  __host__ __device__ void mark_old()
  {
    if (id_ >= 0) id_ = -id_ - 1;
  }

  __host__ __device__ bool operator<(const ResultItem<Index_t>& other) const
  {
    if (dist_ == other.dist_) return id() < other.id();
    return dist_ < other.dist_;
  }
  __host__ __device__ bool operator==(const ResultItem<Index_t>& other) const
  {
    return id() == other.id();
  }
  __host__ __device__ bool operator>=(const ResultItem<Index_t>& other) const
  {
    return !(*this < other);
  }
  __host__ __device__ bool operator<=(const ResultItem<Index_t>& other) const
  {
    return (*this == other) || (*this < other);
  }
  __host__ __device__ bool operator>(const ResultItem<Index_t>& other) const
  {
    return !(*this <= other);
  }
  __host__ __device__ bool operator!=(const ResultItem<Index_t>& other) const
  {
    return !(*this == other);
  }
};

using align32 = raft::Pow2<32>;

template <typename T>
int get_batch_size(const int it_now, const T nrow, const int batch_size)
{
  int it_total = raft::ceildiv(nrow, batch_size);
  return (it_now == it_total - 1) ? nrow - it_now * batch_size : batch_size;
}

// for avoiding bank conflict
template <typename T>
constexpr __host__ __device__ __forceinline__ int skew_dim(int ndim)
{
  // all "4"s are for alignment
  if constexpr (std::is_same<T, float>::value) {
    ndim = raft::ceildiv(ndim, 4) * 4;
    return ndim + (ndim % 32 == 0) * 4;
  }
}

template <typename T>
struct dtype_traits;

template <>
struct dtype_traits<float> {
  static constexpr int APAD           = 4;
  static constexpr int BPAD           = 4;
  static constexpr int TILE_COL_WIDTH = 32;
  static __device__ __forceinline__ float to_float(float v) { return v; }
};

template <>
struct dtype_traits<__half> {
  static constexpr int APAD           = 8;
  static constexpr int BPAD           = 8;
  static constexpr int TILE_COL_WIDTH = 64;
  static __device__ __forceinline__ float to_float(__half v) { return __half2float(v); }
};

template <typename T>
concept Byte = std::is_same_v<T, uint8_t> or std::is_same_v<T, int8_t>;
template <Byte T>
struct dtype_traits<T> {
  static constexpr int APAD           = 4;
  static constexpr int BPAD           = 4;
  static constexpr int TILE_COL_WIDTH = 128;
  static __device__ __forceinline__ float to_float(T v) { return static_cast<float>(v); }
};

template <typename T>
__device__ __forceinline__ ResultItem<T> xor_swap(ResultItem<T> x, int mask, int dir)
{
  ResultItem<T> y;
  y.dist() = __shfl_xor_sync(raft::warp_full_mask(), x.dist(), mask, raft::warp_size());
  y.id_with_flag() =
    __shfl_xor_sync(raft::warp_full_mask(), x.id_with_flag(), mask, raft::warp_size());
  return x < y == dir ? y : x;
}

__device__ __forceinline__ int xor_swap(int x, int mask, int dir)
{
  int y = __shfl_xor_sync(raft::warp_full_mask(), x, mask, raft::warp_size());
  return x < y == dir ? y : x;
}

// TODO: Move to RAFT utils https://github.com/rapidsai/raft/issues/1827
__device__ __forceinline__ uint bfe(uint lane_id, uint pos)
{
  uint res;
  asm("bfe.u32 %0,%1,%2,%3;" : "=r"(res) : "r"(lane_id), "r"(pos), "r"(1));
  return res;
}

template <typename T>
__device__ __forceinline__ void warp_bitonic_sort(T* element_ptr, const int lane_id)
{
  static_assert(raft::warp_size() == 32);
  auto& element = *element_ptr;
  element       = xor_swap(element, 0x01, bfe(lane_id, 1) ^ bfe(lane_id, 0));
  element       = xor_swap(element, 0x02, bfe(lane_id, 2) ^ bfe(lane_id, 1));
  element       = xor_swap(element, 0x01, bfe(lane_id, 2) ^ bfe(lane_id, 0));
  element       = xor_swap(element, 0x04, bfe(lane_id, 3) ^ bfe(lane_id, 2));
  element       = xor_swap(element, 0x02, bfe(lane_id, 3) ^ bfe(lane_id, 1));
  element       = xor_swap(element, 0x01, bfe(lane_id, 3) ^ bfe(lane_id, 0));
  element       = xor_swap(element, 0x08, bfe(lane_id, 4) ^ bfe(lane_id, 3));
  element       = xor_swap(element, 0x04, bfe(lane_id, 4) ^ bfe(lane_id, 2));
  element       = xor_swap(element, 0x02, bfe(lane_id, 4) ^ bfe(lane_id, 1));
  element       = xor_swap(element, 0x01, bfe(lane_id, 4) ^ bfe(lane_id, 0));
  element       = xor_swap(element, 0x10, bfe(lane_id, 4));
  element       = xor_swap(element, 0x08, bfe(lane_id, 3));
  element       = xor_swap(element, 0x04, bfe(lane_id, 2));
  element       = xor_swap(element, 0x02, bfe(lane_id, 1));
  element       = xor_swap(element, 0x01, bfe(lane_id, 0));
  return;
}

constexpr int NUM_SAMPLES = 32;
// For now, the max. number of samples is 32, so the sample cache size is fixed
// to 64 (32 * 2).
constexpr int MAX_NUM_BI_SAMPLES        = 64;
constexpr int SKEWED_MAX_NUM_BI_SAMPLES = skew_dim<float>(MAX_NUM_BI_SAMPLES);
constexpr int BLOCK_SIZE                = 512;
constexpr int WMMA_M                    = 16;
constexpr int WMMA_N                    = 16;
constexpr int WMMA_K                    = 16;

template <typename Data_t>
__device__ __forceinline__ void load_vec(Data_t* vec_buffer,
                                         const Data_t* d_vec,
                                         const int load_dims,
                                         const int padding_dims,
                                         const int lane_id)
{
  if constexpr (std::is_same_v<Data_t, float> or std::is_same_v<Data_t, uint8_t> or
                std::is_same_v<Data_t, int8_t>) {
    constexpr int num_load_elems_per_warp = raft::warp_size();
    for (int step = 0; step < raft::ceildiv(padding_dims, num_load_elems_per_warp); step++) {
      int idx = step * num_load_elems_per_warp + lane_id;
      if (idx < load_dims) {
        vec_buffer[idx] = d_vec[idx];
      } else if (idx < padding_dims) {
        vec_buffer[idx] = 0.0f;
      }
    }
  }
  if constexpr (std::is_same_v<Data_t, __half>) {
    if ((size_t)d_vec % sizeof(float2) == 0 && (size_t)vec_buffer % sizeof(float2) == 0 &&
        load_dims % 4 == 0 && padding_dims % 4 == 0) {
      constexpr int num_load_elems_per_warp = raft::warp_size() * 4;
#pragma unroll
      for (int step = 0; step < raft::ceildiv(padding_dims, num_load_elems_per_warp); step++) {
        int idx_in_vec = step * num_load_elems_per_warp + lane_id * 4;
        if (idx_in_vec + 4 <= load_dims) {
          *(float2*)(vec_buffer + idx_in_vec) = *(float2*)(d_vec + idx_in_vec);
        } else if (idx_in_vec + 4 <= padding_dims) {
          *(float2*)(vec_buffer + idx_in_vec) = float2({0.0f, 0.0f});
        }
      }
    } else {
      constexpr int num_load_elems_per_warp = raft::warp_size();
      for (int step = 0; step < raft::ceildiv(padding_dims, num_load_elems_per_warp); step++) {
        int idx = step * num_load_elems_per_warp + lane_id;
        if (idx < load_dims) {
          vec_buffer[idx] = d_vec[idx];
        } else if (idx < padding_dims) {
          vec_buffer[idx] = 0.0f;
        }
      }
    }
  }
}

/** Converting load: loads Data_t from global memory into __half shared memory buffer. */
template <typename Data_t>
  requires(!std::is_same_v<Data_t, __half>)
__device__ __forceinline__ void load_vec(__half* vec_buffer,
                                         const Data_t* d_vec,
                                         const int load_dims,
                                         const int padding_dims,
                                         const int lane_id)
{
  constexpr int num_load_elems_per_warp = raft::warp_size();
  const __half half_0                   = __float2half(0.0f);
  for (int step = 0; step < raft::ceildiv(padding_dims, num_load_elems_per_warp); step++) {
    int idx = step * num_load_elems_per_warp + lane_id;
    if (idx < load_dims) {
      vec_buffer[idx] = d_vec[idx];
    } else if (idx < padding_dims) {
      vec_buffer[idx] = half_0;
    }
  }
}

template <int n_planes>
__device__ inline void load_vec_bbq_simt(uint32_t* vec_buffer,
                                         const uint32_t* d_vec,
                                         int plane_extent,
                                         int num_load,
                                         int plane_tile,
                                         int lane_id)
{
  // Loads only [0, num_load) per plane. Callers are responsible for zeroing the padding
  // [num_load, plane_tile) on the last tile (see the step == n_tiles - 1 branch at the
  // call site) -- this keeps the load loop branch-free and minimizes live registers.
  for (int idx = lane_id; idx < num_load; idx += raft::warp_size()) {
#pragma unroll
    for (int p = 0; p < n_planes; ++p) {
      vec_buffer[p * plane_tile + idx] = d_vec[p * plane_extent + idx];
    }
  }
}

// Zero the per-plane padding [num_load, plane_tile) so the dot product's full-tile read sees
// zeros beyond the real data. Called at the load site only when step == n_tiles - 1.
template <int n_planes>
__device__ inline void zero_pad_bbq_simt(uint32_t* vec_buffer,
                                         int num_load,
                                         int plane_tile,
                                         int lane_id)
{
  for (int idx = num_load + lane_id; idx < plane_tile; idx += raft::warp_size()) {
#pragma unroll
    for (int p = 0; p < n_planes; ++p) {
      vec_buffer[p * plane_tile + idx] = 0;
    }
  }
}

/** One warp per block. Computes squared L2 norm for each row. */
template <typename Data_t>
RAFT_KERNEL compute_l2_norms_kernel(const Data_t* data, int dim, DistData_t* l2_norms)
{
  extern __shared__ char buffer[];
  __shared__ float l2_norm;
  Data_t* s_vec  = (Data_t*)buffer;
  size_t list_id = blockIdx.x;
  int lane_id    = threadIdx.x % raft::warp_size();

  load_vec(s_vec, data + static_cast<size_t>(blockIdx.x) * dim, dim, dim, lane_id);
  if (threadIdx.x == 0) { l2_norm = 0; }
  __syncthreads();

  for (int step = 0; step < raft::ceildiv(dim, raft::warp_size()); step++) {
    int idx         = step * raft::warp_size() + lane_id;
    float part_dist = 0;
    if (idx < dim) {
      part_dist = static_cast<float>(s_vec[idx]);
      part_dist = part_dist * part_dist;
    }
    __syncwarp();
    for (int offset = raft::warp_size() >> 1; offset >= 1; offset >>= 1) {
      part_dist += __shfl_down_sync(raft::warp_full_mask(), part_dist, offset);
    }
    if (lane_id == 0) { l2_norm += part_dist; }
    __syncwarp();
  }

  if (lane_id == 0) { l2_norms[list_id] = l2_norm; }
}

template <typename Src_t, typename Dst_t>
RAFT_KERNEL convert_copy_kernel(const Src_t* src, Dst_t* dst, size_t n)
{
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) { dst[idx] = static_cast<Dst_t>(src[idx]); }
}

template <typename Index_t>
RAFT_KERNEL add_rev_edges_kernel(const Index_t* graph,
                                 Index_t* rev_graph,
                                 int num_samples,
                                 int2* list_sizes)
{
  size_t list_id = blockIdx.x;
  int2 list_size = list_sizes[list_id];

  for (int idx = threadIdx.x; idx < list_size.x; idx += blockDim.x) {
    // each node has same number (num_samples) of forward and reverse edges
    Index_t rev_list_id = graph[list_id * num_samples + idx];
    if (rev_list_id == std::numeric_limits<Index_t>::max()) {
      // sentinel value
      continue;
    }

    // there are already num_samples forward edges
    int idx_in_rev_list = atomicAdd(&list_sizes[rev_list_id].y, 1);
    if (idx_in_rev_list >= num_samples) {
      atomicExch(&list_sizes[rev_list_id].y, num_samples);
    } else {
      rev_graph[rev_list_id * num_samples + idx_in_rev_list] = list_id;
    }
  }
}

template <typename Index_t, typename ID_t = InternalID_t<Index_t>>
__device__ void insert_to_global_graph(ResultItem<Index_t> elem,
                                       size_t list_id,
                                       ID_t* graph,
                                       DistData_t* dists,
                                       int node_degree,
                                       int* locks)
{
  int tx                 = threadIdx.x;
  int lane_id            = tx % raft::warp_size();
  size_t global_idx_base = list_id * node_degree;
  if (elem.id() == list_id) return;

  const int num_segments = raft::ceildiv(node_degree, raft::warp_size());

  int loop_flag = 0;
  do {
    int segment_id = elem.id() % num_segments;
    if (lane_id == 0) {
      loop_flag = atomicCAS(&locks[list_id * num_segments + segment_id], 0, 1) == 0;
    }

    loop_flag = __shfl_sync(raft::warp_full_mask(), loop_flag, 0);

    if (loop_flag == 1) {
      ResultItem<Index_t> knn_list_frag;
      int local_idx     = segment_id * raft::warp_size() + lane_id;
      size_t global_idx = global_idx_base + local_idx;
      if (local_idx < node_degree) {
        knn_list_frag.id_with_flag() = graph[global_idx].id_with_flag();
        knn_list_frag.dist()         = dists[global_idx];
      }

      int pos_to_insert = -1;
      ResultItem<Index_t> prev_elem;

      prev_elem.id_with_flag() =
        __shfl_up_sync(raft::warp_full_mask(), knn_list_frag.id_with_flag(), 1);
      prev_elem.dist() = __shfl_up_sync(raft::warp_full_mask(), knn_list_frag.dist(), 1);

      if (lane_id == 0) {
        prev_elem = ResultItem<Index_t>{std::numeric_limits<Index_t>::min(),
                                        std::numeric_limits<DistData_t>::lowest()};
      }
      if (elem > prev_elem && elem < knn_list_frag) {
        pos_to_insert = segment_id * raft::warp_size() + lane_id;
      } else if (elem == prev_elem || elem == knn_list_frag) {
        pos_to_insert = -2;
      }
      uint mask = __ballot_sync(raft::warp_full_mask(), pos_to_insert >= 0);
      if (mask) {
        uint set_lane_id = __fns(mask, 0, 1);
        pos_to_insert    = __shfl_sync(raft::warp_full_mask(), pos_to_insert, set_lane_id);
      }

      if (pos_to_insert >= 0) {
        int local_idx = segment_id * raft::warp_size() + lane_id;
        if (local_idx > pos_to_insert) {
          local_idx++;
        } else if (local_idx == pos_to_insert) {
          graph[global_idx_base + local_idx].id_with_flag() = elem.id_with_flag();
          dists[global_idx_base + local_idx]                = elem.dist();
          local_idx++;
        }
        size_t global_pos = global_idx_base + local_idx;
        if (local_idx < (segment_id + 1) * raft::warp_size() && local_idx < node_degree) {
          graph[global_pos].id_with_flag() = knn_list_frag.id_with_flag();
          dists[global_pos]                = knn_list_frag.dist();
        }
      }
      __threadfence();
      if (loop_flag && lane_id == 0) { atomicExch(&locks[list_id * num_segments + segment_id], 0); }
    }
  } while (!loop_flag);
}

template <typename Index_t>
__device__ ResultItem<Index_t> get_min_item(const Index_t id,
                                            const int idx_in_list,
                                            const Index_t* neighbs,
                                            const DistData_t* distances,
                                            const bool find_in_row = true,
                                            const int stride       = SKEWED_MAX_NUM_BI_SAMPLES)
{
  int lane_id = threadIdx.x % raft::warp_size();

  static_assert(MAX_NUM_BI_SAMPLES == 64);
  int idx[MAX_NUM_BI_SAMPLES / raft::warp_size()];
  float dist[MAX_NUM_BI_SAMPLES / raft::warp_size()] = {std::numeric_limits<DistData_t>::max(),
                                                        std::numeric_limits<DistData_t>::max()};
  idx[0]                                             = lane_id;
  idx[1]                                             = raft::warp_size() + lane_id;

  if (neighbs[idx[0]] != id) {
    dist[0] = find_in_row ? distances[idx_in_list * stride + lane_id]
                          : distances[idx_in_list + lane_id * stride];
  }

  if (neighbs[idx[1]] != id) {
    dist[1] = find_in_row ? distances[idx_in_list * stride + raft::warp_size() + lane_id]
                          : distances[idx_in_list + (raft::warp_size() + lane_id) * stride];
  }

  if (dist[1] < dist[0]) {
    dist[0] = dist[1];
    idx[0]  = idx[1];
  }
  __syncwarp();
  for (int offset = raft::warp_size() >> 1; offset >= 1; offset >>= 1) {
    float other_idx  = __shfl_down_sync(raft::warp_full_mask(), idx[0], offset);
    float other_dist = __shfl_down_sync(raft::warp_full_mask(), dist[0], offset);
    if (other_dist < dist[0]) {
      dist[0] = other_dist;
      idx[0]  = other_idx;
    }
  }

  ResultItem<Index_t> result;
  result.dist()         = __shfl_sync(raft::warp_full_mask(), dist[0], 0);
  result.id_with_flag() = neighbs[__shfl_sync(raft::warp_full_mask(), idx[0], 0)];
  return result;
}

template <typename T>
__device__ __forceinline__ void remove_duplicates(
  T* list_a, int list_a_size, T* list_b, int list_b_size, int& unique_counter, int execute_warp_id)
{
  static_assert(raft::warp_size() == 32);
  if (!(threadIdx.x >= execute_warp_id * raft::warp_size() &&
        threadIdx.x < execute_warp_id * raft::warp_size() + raft::warp_size())) {
    return;
  }
  int lane_id = threadIdx.x % raft::warp_size();
  T elem      = std::numeric_limits<T>::max();
  if (lane_id < list_a_size) { elem = list_a[lane_id]; }
  warp_bitonic_sort(&elem, lane_id);

  if (elem != std::numeric_limits<T>::max()) { list_a[lane_id] = elem; }

  T elem_b = std::numeric_limits<T>::max();

  if (lane_id < list_b_size) { elem_b = list_b[lane_id]; }
  __syncwarp();

  int idx_l    = 0;
  int idx_r    = list_a_size;
  bool existed = false;
  while (idx_l < idx_r) {
    int idx  = (idx_l + idx_r) / 2;
    int elem = list_a[idx];
    if (elem == elem_b) {
      existed = true;
      break;
    }
    if (elem_b > elem) {
      idx_l = idx + 1;
    } else {
      idx_r = idx;
    }
  }
  if (!existed && elem_b != std::numeric_limits<T>::max()) {
    int idx                   = atomicAdd(&unique_counter, 1);
    list_a[list_a_size + idx] = elem_b;
  }
}

template <typename Index_t, typename Data_t, typename DistEpilogue_t>
__device__ __forceinline__ void calculate_metric(float* s_distances,
                                                 Index_t* row_neighbors,
                                                 int list_row_size,
                                                 Index_t* col_neighbors,
                                                 int list_col_size,
                                                 const Data_t* data,
                                                 const int data_dim,
                                                 DistData_t* l2_norms,
                                                 cuvs::distance::DistanceType metric,
                                                 DistEpilogue_t dist_epilogue)
{
  // if we have a distance epilogue, distances need to be fully calculated instead of postprocessing
  // them.
  bool can_postprocess_dist = std::is_same_v<DistEpilogue_t, raft::identity_op>;

  for (int i = threadIdx.x; i < MAX_NUM_BI_SAMPLES * SKEWED_MAX_NUM_BI_SAMPLES; i += blockDim.x) {
    int row_id = i / SKEWED_MAX_NUM_BI_SAMPLES;
    int col_id = i % SKEWED_MAX_NUM_BI_SAMPLES;

    if (row_id < list_row_size && col_id < list_col_size) {
      if (metric == cuvs::distance::DistanceType::InnerProduct && can_postprocess_dist) {
        s_distances[i] = -s_distances[i];
      } else if (metric == cuvs::distance::DistanceType::CosineExpanded) {
        float norm_product = l2_norms[row_neighbors[row_id]] * l2_norms[col_neighbors[col_id]];
        s_distances[i] =
          (norm_product > 0.0f) ? (1.0f - s_distances[i] / sqrtf(norm_product)) : 0.0f;
      } else if (metric == cuvs::distance::DistanceType::BitwiseHamming) {
        s_distances[i] = 0.0;
        int n1         = row_neighbors[row_id];
        int n2         = col_neighbors[col_id];
        // TODO: https://github.com/nvidia/cuvs/issues/1127
        const uint8_t* data_n1 = reinterpret_cast<const uint8_t*>(data) + n1 * data_dim;
        const uint8_t* data_n2 = reinterpret_cast<const uint8_t*>(data) + n2 * data_dim;
        for (int d = 0; d < data_dim; d++) {
          s_distances[i] += __popc(static_cast<uint32_t>(data_n1[d] ^ data_n2[d]) & 0xff);
        }
      } else if (metric == cuvs::distance::DistanceType::L2Expanded ||
                 metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
        s_distances[i] =
          l2_norms[row_neighbors[row_id]] + l2_norms[col_neighbors[col_id]] - 2.0 * s_distances[i];
        // for fp32 vs fp16 precision differences resulting in negative distances when distance
        // should be 0 related issue: https://github.com/nvidia/cuvs/issues/991
        s_distances[i] = s_distances[i] < 0.0f ? 0.0f : s_distances[i];
        if (!can_postprocess_dist && metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
          s_distances[i] = sqrtf(s_distances[i]);
        }
      }
      s_distances[i] = dist_epilogue(s_distances[i], row_neighbors[row_id], col_neighbors[col_id]);
    } else {
      s_distances[i] = std::numeric_limits<float>::max();
    }
  }
}

struct DistAccumulator {
  cuvs::distance::DistanceType metric;
  __device__ __forceinline__ float operator()(float a, float b) const
  {
    if (metric == cuvs::distance::DistanceType::L1) { return raft::abs(a - b); }
    // dot product: reused by IP, cosine, and L2 (postprocessed in calculate_metric)
    return a * b;
  }
};

// launch_bounds here denote BLOCK_SIZE = 512 and MIN_BLOCKS_PER_SM = 4
// Per
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications,
// MAX_RESIDENT_THREAD_PER_SM = BLOCK_SIZE * BLOCKS_PER_SM = 2048
// For architectures 750 and 860 (890), the values for MAX_RESIDENT_THREAD_PER_SM
// is 1024 and 1536 respectively, which means the bounds don't work anymore
// SIMT kernel: scalar element-wise distance computation.
// Used for fp32 data (all metrics) and L1 distance computation for all dtypes (which cannot use
// tensor cores).
template <typename Data_t,
          typename Index_t,
          typename ID_t = InternalID_t<Index_t>,
          typename DistEpilogue_t>
RAFT_KERNEL
#ifdef __CUDA_ARCH__
// Use minBlocksPerMultiprocessor = 4 on specific arches
#if (__CUDA_ARCH__) == 700 || (__CUDA_ARCH__) == 800 || (__CUDA_ARCH__) == 900 || \
  (__CUDA_ARCH__) == 1000
__launch_bounds__(BLOCK_SIZE, 4)
#else
__launch_bounds__(BLOCK_SIZE)
#endif
#endif
  local_join_kernel_simt(const Index_t* graph_new,
                         const Index_t* rev_graph_new,
                         const int2* sizes_new,
                         const Index_t* graph_old,
                         const Index_t* rev_graph_old,
                         const int2* sizes_old,
                         const int width,
                         const Data_t* data,
                         const int data_dim,
                         ID_t* graph,
                         DistData_t* dists,
                         int graph_width,
                         int* locks,
                         DistData_t* l2_norms,
                         cuvs::distance::DistanceType metric,
                         DistEpilogue_t dist_epilogue)
{
#if (__CUDA_ARCH__ >= 700)
  __shared__ int s_list[MAX_NUM_BI_SAMPLES * 2];

  constexpr int APAD           = dtype_traits<Data_t>::APAD;
  constexpr int BPAD           = dtype_traits<Data_t>::BPAD;
  constexpr int TILE_COL_WIDTH = dtype_traits<Data_t>::TILE_COL_WIDTH;
  __shared__ Data_t s_nv[MAX_NUM_BI_SAMPLES][TILE_COL_WIDTH + APAD];
  __shared__ Data_t s_ov[MAX_NUM_BI_SAMPLES][TILE_COL_WIDTH + BPAD];
  __shared__ float s_distances[MAX_NUM_BI_SAMPLES * SKEWED_MAX_NUM_BI_SAMPLES];

  // s_distances: MAX_NUM_BI_SAMPLES x SKEWED_MAX_NUM_BI_SAMPLES, reuse the space of s_ov
  int* s_unique_counter = (int*)&s_ov[0][0];

  if (threadIdx.x == 0) {
    s_unique_counter[0] = 0;
    s_unique_counter[1] = 0;
  }

  Index_t* new_neighbors = s_list;
  Index_t* old_neighbors = s_list + MAX_NUM_BI_SAMPLES;

  size_t list_id      = blockIdx.x;
  int2 list_new_size2 = sizes_new[list_id];
  int list_new_size   = list_new_size2.x + list_new_size2.y;
  int2 list_old_size2 = sizes_old[list_id];
  int list_old_size   = list_old_size2.x + list_old_size2.y;

  if (!list_new_size) return;
  int tx = threadIdx.x;

  if (tx < list_new_size2.x) {
    new_neighbors[tx] = graph_new[list_id * width + tx];
  } else if (tx >= list_new_size2.x && tx < list_new_size) {
    new_neighbors[tx] = rev_graph_new[list_id * width + tx - list_new_size2.x];
  }

  if (tx < list_old_size2.x) {
    old_neighbors[tx] = graph_old[list_id * width + tx];
  } else if (tx >= list_old_size2.x && tx < list_old_size) {
    old_neighbors[tx] = rev_graph_old[list_id * width + tx - list_old_size2.x];
  }

  __syncthreads();

  remove_duplicates(new_neighbors,
                    list_new_size2.x,
                    new_neighbors + list_new_size2.x,
                    list_new_size2.y,
                    s_unique_counter[0],
                    0);

  remove_duplicates(old_neighbors,
                    list_old_size2.x,
                    old_neighbors + list_old_size2.x,
                    list_old_size2.y,
                    s_unique_counter[1],
                    1);
  __syncthreads();
  list_new_size = list_new_size2.x + s_unique_counter[0];
  list_old_size = list_old_size2.x + s_unique_counter[1];

  int warp_id             = threadIdx.x / raft::warp_size();
  int lane_id             = threadIdx.x % raft::warp_size();
  constexpr int num_warps = BLOCK_SIZE / raft::warp_size();

  DistAccumulator dist_acc(metric);

  int tid = threadIdx.x;
  for (int i = tid; i < MAX_NUM_BI_SAMPLES * SKEWED_MAX_NUM_BI_SAMPLES; i += blockDim.x)
    s_distances[i] = 0.0f;

  __syncthreads();

  for (int step = 0; step < raft::ceildiv(data_dim, TILE_COL_WIDTH); step++) {
    int num_load_elems = (step == raft::ceildiv(data_dim, TILE_COL_WIDTH) - 1)
                           ? data_dim - step * TILE_COL_WIDTH
                           : TILE_COL_WIDTH;
#pragma unroll
    for (int i = 0; i < MAX_NUM_BI_SAMPLES / num_warps; i++) {
      int idx = i * num_warps + warp_id;
      if (idx < list_new_size) {
        size_t neighbor_id = new_neighbors[idx];
        size_t idx_in_data = neighbor_id * data_dim;
        // loaded to shared memory while keeping the original dtype
        load_vec(s_nv[idx],
                 data + idx_in_data + step * TILE_COL_WIDTH,
                 num_load_elems,
                 TILE_COL_WIDTH,
                 lane_id);
      }
    }
    __syncthreads();

    // this is much faster than a warp-collaborative multiplication because MAX_NUM_BI_SAMPLES is
    // fixed and small (64)
    for (int i = threadIdx.x; i < MAX_NUM_BI_SAMPLES * SKEWED_MAX_NUM_BI_SAMPLES; i += blockDim.x) {
      int tmp_row = i / SKEWED_MAX_NUM_BI_SAMPLES;
      int tmp_col = i % SKEWED_MAX_NUM_BI_SAMPLES;
      if (tmp_row < list_new_size && tmp_col < list_new_size) {
        float acc = 0.0f;
        for (int d = 0; d < num_load_elems; d++) {
          // converted to float for distance computation
          float a = dtype_traits<Data_t>::to_float(s_nv[tmp_row][d]);
          float b = dtype_traits<Data_t>::to_float(s_nv[tmp_col][d]);
          acc += dist_acc(a, b);
        }
        s_distances[i] += acc;
      }
    }
    __syncthreads();
  }
  __syncthreads();

  calculate_metric(s_distances,
                   new_neighbors,
                   list_new_size,
                   new_neighbors,
                   list_new_size,
                   data,
                   data_dim,
                   l2_norms,
                   metric,
                   dist_epilogue);

  __syncthreads();

  for (int step = 0; step < raft::ceildiv(list_new_size, num_warps); step++) {
    int idx_in_list = step * num_warps + tx / raft::warp_size();
    if (idx_in_list >= list_new_size) continue;
    auto min_elem = get_min_item(s_list[idx_in_list], idx_in_list, new_neighbors, s_distances);
    if (min_elem.id() < gridDim.x) {
      insert_to_global_graph(min_elem, s_list[idx_in_list], graph, dists, graph_width, locks);
    }
  }

  if (!list_old_size) return;

  __syncthreads();

  for (int i = tid; i < MAX_NUM_BI_SAMPLES * SKEWED_MAX_NUM_BI_SAMPLES; i += blockDim.x)
    s_distances[i] = 0.0f;

  __syncthreads();

  for (int step = 0; step < raft::ceildiv(data_dim, TILE_COL_WIDTH); step++) {
    int num_load_elems = (step == raft::ceildiv(data_dim, TILE_COL_WIDTH) - 1)
                           ? data_dim - step * TILE_COL_WIDTH
                           : TILE_COL_WIDTH;
    if (TILE_COL_WIDTH < data_dim) {
#pragma unroll
      for (int i = 0; i < MAX_NUM_BI_SAMPLES / num_warps; i++) {
        int idx = i * num_warps + warp_id;
        if (idx < list_new_size) {
          size_t neighbor_id = new_neighbors[idx];
          size_t idx_in_data = neighbor_id * data_dim;
          load_vec(s_nv[idx],
                   data + idx_in_data + step * TILE_COL_WIDTH,
                   num_load_elems,
                   TILE_COL_WIDTH,
                   lane_id);
        }
      }
    }
#pragma unroll
    for (int i = 0; i < MAX_NUM_BI_SAMPLES / num_warps; i++) {
      int idx = i * num_warps + warp_id;
      if (idx < list_old_size) {
        size_t neighbor_id = old_neighbors[idx];
        size_t idx_in_data = neighbor_id * data_dim;
        load_vec(s_ov[idx],
                 data + idx_in_data + step * TILE_COL_WIDTH,
                 num_load_elems,
                 TILE_COL_WIDTH,
                 lane_id);
      }
    }
    __syncthreads();

    // this is much faster than a warp-collaborative multiplication because MAX_NUM_BI_SAMPLES is
    // fixed and small (64)
    for (int i = threadIdx.x; i < MAX_NUM_BI_SAMPLES * SKEWED_MAX_NUM_BI_SAMPLES; i += blockDim.x) {
      int tmp_row = i / SKEWED_MAX_NUM_BI_SAMPLES;
      int tmp_col = i % SKEWED_MAX_NUM_BI_SAMPLES;
      if (tmp_row < list_new_size && tmp_col < list_old_size) {
        float acc = 0.0f;
        for (int d = 0; d < num_load_elems; d++) {
          float a = dtype_traits<Data_t>::to_float(s_nv[tmp_row][d]);
          float b = dtype_traits<Data_t>::to_float(s_ov[tmp_col][d]);
          acc += dist_acc(a, b);
        }
        s_distances[i] += acc;
      }
    }
    __syncthreads();
  }
  __syncthreads();

  calculate_metric(s_distances,
                   new_neighbors,
                   list_new_size,
                   old_neighbors,
                   list_old_size,
                   data,
                   data_dim,
                   l2_norms,
                   metric,
                   dist_epilogue);

  __syncthreads();

  for (int step = 0; step < raft::ceildiv(MAX_NUM_BI_SAMPLES * 2, num_warps); step++) {
    int idx_in_list = step * num_warps + tx / raft::warp_size();
    if (idx_in_list >= list_new_size && idx_in_list < MAX_NUM_BI_SAMPLES) continue;
    if (idx_in_list >= MAX_NUM_BI_SAMPLES + list_old_size && idx_in_list < MAX_NUM_BI_SAMPLES * 2)
      continue;
    ResultItem<Index_t> min_elem{std::numeric_limits<Index_t>::max(),
                                 std::numeric_limits<DistData_t>::max()};
    if (idx_in_list < MAX_NUM_BI_SAMPLES) {
      auto temp_min_item =
        get_min_item(s_list[idx_in_list], idx_in_list, old_neighbors, s_distances);
      if (temp_min_item.dist() < min_elem.dist()) { min_elem = temp_min_item; }
    } else {
      auto temp_min_item = get_min_item(
        s_list[idx_in_list], idx_in_list - MAX_NUM_BI_SAMPLES, new_neighbors, s_distances, false);
      if (temp_min_item.dist() < min_elem.dist()) { min_elem = temp_min_item; }
    }

    if (min_elem.id() < gridDim.x) {
      insert_to_global_graph(min_elem, s_list[idx_in_list], graph, dists, graph_width, locks);
    }
  }
#endif
}

// Fused replacement for calculate_metric_bbq_symmetric/calculate_metric_bbq_asymmetric +
// get_min_item, used by every BBQ local-join kernel. s_distances_u32 already holds the raw dot
// products, fully written and visible after the __syncthreads() that ends the K-loop, so no
// separate full-grid pass is needed to convert them to float distances before searching for the
// min: the conversion happens inline, in registers, right before the warp min-reduction. This
// removes both the separate calculate_metric_* sweep and the __syncthreads() between it and the
// search, and drops out-of-range entries from touching SMEM at all (the bounds check is done
// from registers).
//
// It is a *trade*, not a free win, and the balance differs per phase. Phase 1 sweeps rows only,
// so each (row, col) entry is converted exactly once either way -- pure traffic/barrier savings.
// Phase 2 sweeps rows (find_in_row=true) and columns (find_in_row=false) over the same matrix:
// the old code converted each entry once in calculate_metric_* and then read it twice, whereas
// this converts it once per pass, i.e. 2x the metric arithmetic in exchange for the traffic.
// That is why the win tracks how SMEM-bound a kernel is -- large for the wmma kernels (~11%,
// whose store bank conflicts dominate), moderate for the symmetric SIMT kernel (~5-7%, which
// also sheds 8 registers), and roughly break-even for the asymmetric SIMT kernel.
//
// One function covers both symmetric and asymmetric: the two-quantizer centered_dot/l2_distance/
// dot_product/cosine_distance overloads used below are exact generalizations of their
// single-quantizer counterparts -- passing quantizer_document == quantizer_query (and
// l2_norms_document == l2_norms_query) reproduces the symmetric computation bit-for-bit, verified
// directly against bbq.cuh's definitions. The plain (non-fused) get_min_item is untouched and
// still used by local_join_kernel_simt, the scalar local_join_kernel_bbq_simt, and
// local_join_kernel_wmma.
//
// Row/col semantics mirror get_min_item: `id`/`idx_in_list` name one fixed axis, `neighbs` the
// other (varying) axis. The s_distances_u32 matrix itself is always [row=document][col=query]
// regardless of find_in_row -- find_in_row only picks which axis is fixed vs. swept, so which of
// (id, neighbs[idx]) is the document vs. the query flips with it too (moot when document ==
// query, i.e. the symmetric case).
template <typename DataT, typename Index_t, typename DistEpilogue_t>
__device__ __forceinline__ ResultItem<Index_t> get_min_item_fused(
  const Index_t id,
  const int idx_in_list,
  const Index_t* neighbs,
  const int neighbs_size,
  const uint32_t* s_distances_u32,
  const bbq_device_quantizer_view<DataT, int64_t> quantizer_document,
  const bbq_device_quantizer_view<DataT, int64_t> quantizer_query,
  DistData_t* l2_norms_document,
  DistData_t* l2_norms_query,
  cuvs::distance::DistanceType metric,
  DistEpilogue_t dist_epilogue,
  const bool find_in_row = true,
  const int stride       = SKEWED_MAX_NUM_BI_SAMPLES)
{
  const int lane_id               = threadIdx.x % raft::warp_size();
  const bool can_postprocess_dist = std::is_same_v<DistEpilogue_t, raft::identity_op>;

  static_assert(MAX_NUM_BI_SAMPLES == 64);
  int idx[MAX_NUM_BI_SAMPLES / raft::warp_size()];
  idx[0] = lane_id;
  idx[1] = raft::warp_size() + lane_id;

  auto compute_dist = [&](int k) -> float {
    if (idx[k] >= neighbs_size || neighbs[idx[k]] == id) {
      return std::numeric_limits<DistData_t>::max();
    }
    const uint32_t raw         = find_in_row ? s_distances_u32[idx_in_list * stride + idx[k]]
                                             : s_distances_u32[idx_in_list + idx[k] * stride];
    const Index_t row_document = find_in_row ? id : neighbs[idx[k]];
    const Index_t row_query    = find_in_row ? neighbs[idx[k]] : id;
    const float centered       = cuvs::preprocessing::quantize::bbq::centered_dot(
      quantizer_document, quantizer_query, static_cast<float>(raw), row_document, row_query);
    float d;
    if (metric == cuvs::distance::DistanceType::L2Expanded ||
        metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
      d = cuvs::preprocessing::quantize::bbq::l2_distance(
        quantizer_document, quantizer_query, centered, row_document, row_query);
      if (!can_postprocess_dist && metric == cuvs::distance::DistanceType::L2SqrtExpanded) {
        d = sqrtf(d);
      }
    } else if (metric == cuvs::distance::DistanceType::InnerProduct) {
      d = -cuvs::preprocessing::quantize::bbq::dot_product(
        quantizer_document, quantizer_query, centered, row_document, row_query);
    } else {  // CosineExpanded
      const float norm_product = l2_norms_document[row_document] * l2_norms_query[row_query];
      d                        = cuvs::preprocessing::quantize::bbq::cosine_distance(
        quantizer_document, quantizer_query, centered, row_document, row_query, norm_product);
    }
    return dist_epilogue(d, row_document, row_query);
  };

  float dist[MAX_NUM_BI_SAMPLES / raft::warp_size()];
  dist[0] = compute_dist(0);
  dist[1] = compute_dist(1);

  if (dist[1] < dist[0]) {
    dist[0] = dist[1];
    idx[0]  = idx[1];
  }
  __syncwarp();
  for (int offset = raft::warp_size() >> 1; offset >= 1; offset >>= 1) {
    float other_idx  = __shfl_down_sync(raft::warp_full_mask(), idx[0], offset);
    float other_dist = __shfl_down_sync(raft::warp_full_mask(), dist[0], offset);
    if (other_dist < dist[0]) {
      dist[0] = other_dist;
      idx[0]  = other_idx;
    }
  }

  ResultItem<Index_t> result;
  result.dist()         = __shfl_sync(raft::warp_full_mask(), dist[0], 0);
  result.id_with_flag() = neighbs[__shfl_sync(raft::warp_full_mask(), idx[0], 0)];
  return result;
}

// Selects the SIMT inner product for a (document, query) layout pair. The transposed layouts are
// bit-sliced, so their dot product is a document_planes x query_planes cross-plane popc with a
// per-plane shift -- code_inner_product_planes_2x1 covers every such pair, including the
// symmetric ones (document_planes == query_planes). The densely-packed layouts have no planes to
// cross and use dp4a instead; those forms only exist for a self-join, since dp4a needs both
// operands in the same packing. packed_4b / packed_8b are currently unreachable from
// GNND::local_join (packed_4b goes to the wmma kernel, packed_8b is not dispatched), but are kept
// so either can be re-enabled as a SIMT reference point without resurrecting deleted code.
template <bbq_layout DocumentLayout,
          bbq_layout QueryLayout,
          bool SelfJoin,
          int DocumentPlanes,
          int QueryPlanes,
          size_t DocumentRowBytes,
          size_t QueryRowBytes>
__device__ __forceinline__ void bbq_code_inner_product_2x1(const uint8_t* row_a0,
                                                           const uint8_t* row_a1,
                                                           const uint8_t* row_b,
                                                           uint32_t& total0,
                                                           uint32_t& total1)
{
  namespace bbq = cuvs::preprocessing::quantize::bbq;
  if constexpr (SelfJoin && DocumentLayout == bbq_layout::packed_4b) {
    bbq::code_inner_product_packed_4b_symmetric_2x1<DocumentRowBytes>(
      row_a0, row_a1, row_b, total0, total1);
  } else if constexpr (SelfJoin && (DocumentLayout == bbq_layout::packed_8b ||
                                    DocumentLayout == bbq_layout::packed_7b)) {
    // packed_7b is packed_8b with the top bit masked off, matching code_inner_product's
    // (1 << bits) - 1 mask for the same two layouts.
    constexpr uint8_t code_mask = DocumentLayout == bbq_layout::packed_7b ? 0x7Fu : 0xFFu;
    bbq::code_inner_product_packed_8b_2x1<DocumentRowBytes>(
      row_a0, row_a1, row_b, total0, total1, code_mask);
  } else {
    bbq::
      code_inner_product_planes_2x1<DocumentPlanes, QueryPlanes, DocumentRowBytes, QueryRowBytes>(
        row_a0, row_a1, row_b, total0, total1);
  }
}

template <bbq_layout DocumentLayout,
          bbq_layout QueryLayout,
          bool SelfJoin,
          typename DataT,
          typename Index_t,
          typename ID_t = InternalID_t<Index_t>,
          typename DistEpilogue_t>
RAFT_KERNEL __launch_bounds__(BLOCK_SIZE)
  local_join_kernel_bbq_simt(const Index_t* graph_new,
                             const Index_t* rev_graph_new,
                             const int2* sizes_new,
                             const Index_t* graph_old,
                             const Index_t* rev_graph_old,
                             const int2* sizes_old,
                             const int width,
                             bbq_device_quantizer_view<DataT, int64_t> dataset_document,
                             bbq_device_quantizer_view<DataT, int64_t> dataset_query,
                             ID_t* graph,
                             DistData_t* dists,
                             int graph_width,
                             int* locks,
                             DistData_t* l2_norms_document,
                             DistData_t* l2_norms_query,
                             cuvs::distance::DistanceType metric,
                             DistEpilogue_t dist_epilogue)
{
  // Plane count per layout: the transposed layouts are bit-sliced into `N` planes, the packed
  // ones are a single dense plane. This is the only thing the tiling needs from the layout.
  constexpr auto planes_of = [](bbq_layout l) {
    return l == bbq_layout::transposed_2b ? 2 : l == bbq_layout::transposed_4b ? 4 : 1;
  };
  constexpr int document_planes = planes_of(DocumentLayout);
  constexpr int query_planes    = planes_of(QueryLayout);
  static_assert(!SelfJoin || DocumentLayout == QueryLayout,
                "a self-join must use the same layout on both operands");

  // Both operands are tiled at the same per-plane tile so each step covers the same dimension
  // range on both sides. QUERY_ROW_BYTES fixes the query row width; the document row width then
  // follows as QUERY_ROW_BYTES / (query_planes / document_planes), i.e. query_plane_tile scaled
  // by the document's own plane count. Worked out per supported pair:
  //
  //   pair     query_plane_tile   doc_row_bytes    doc_stride   query_stride
  //   ------   ----------------   --------------   ----------   ------------
  //   1 x 1    128                128 * 1 = 128    128          128
  //   2t x 2t  64                 64  * 2 = 128    64           64
  //   1 + 2t   64                 64  * 1 = 64     64           64
  //   1 + 4t   32                 32  * 1 = 32     32           32
  //   2t + 4t  32                 32  * 2 = 64     32           32
  //
  constexpr int QUERY_ROW_BYTES = 128;
  constexpr int BBQ_PAD         = alignof(uint32_t);
  // The document buffer is normally only the A operand (two rows broadcast across a warp), so it
  // needs no bank-conflict pad. Under SelfJoin it doubles as the B operand in phase 1 (32
  // consecutive columns at one byte offset), so it needs the same skew the query buffer gets.
  constexpr int DOC_PAD = SelfJoin ? BBQ_PAD : 0;
  static_assert((QUERY_ROW_BYTES + BBQ_PAD) % alignof(uint32_t) == 0);

  __shared__ int s_list[MAX_NUM_BI_SAMPLES * 2];
  __shared__ __align__(alignof(uint32_t)) uint8_t
    s_doc_vec[MAX_NUM_BI_SAMPLES][QUERY_ROW_BYTES / query_planes * document_planes + DOC_PAD];
  __shared__ __align__(alignof(uint32_t))
    uint8_t s_query_vec[MAX_NUM_BI_SAMPLES][QUERY_ROW_BYTES + BBQ_PAD];
  __shared__ uint32_t s_distances_u32[MAX_NUM_BI_SAMPLES * SKEWED_MAX_NUM_BI_SAMPLES];
  __shared__ int s_unique_counter[2];

  if (threadIdx.x == 0) {
    s_unique_counter[0] = 0;
    s_unique_counter[1] = 0;
  }

  Index_t* new_neighbors = s_list;
  Index_t* old_neighbors = s_list + MAX_NUM_BI_SAMPLES;
  const size_t list_id   = blockIdx.x;
  const int2 new_size2   = sizes_new[list_id];
  const int2 old_size2   = sizes_old[list_id];
  int new_size           = new_size2.x + new_size2.y;
  int old_size           = old_size2.x + old_size2.y;
  const int tx           = threadIdx.x;

  if (!new_size) return;
  if (tx < new_size2.x) {
    new_neighbors[tx] = graph_new[list_id * width + tx];
  } else if (tx < new_size) {
    new_neighbors[tx] = rev_graph_new[list_id * width + tx - new_size2.x];
  }
  if (tx < old_size2.x) {
    old_neighbors[tx] = graph_old[list_id * width + tx];
  } else if (tx < old_size) {
    old_neighbors[tx] = rev_graph_old[list_id * width + tx - old_size2.x];
  }
  __syncthreads();

  remove_duplicates(
    new_neighbors, new_size2.x, new_neighbors + new_size2.x, new_size2.y, s_unique_counter[0], 0);
  remove_duplicates(
    old_neighbors, old_size2.x, old_neighbors + old_size2.x, old_size2.y, s_unique_counter[1], 1);
  __syncthreads();
  new_size = new_size2.x + s_unique_counter[0];
  old_size = old_size2.x + s_unique_counter[1];

  const int warp_id       = threadIdx.x / raft::warp_size();
  const int lane_id       = threadIdx.x % raft::warp_size();
  constexpr int num_warps = BLOCK_SIZE / raft::warp_size();
  const int encoded_row_length_document =
    cuvs::preprocessing::quantize::bbq::get_encoded_row_length(dataset_document);

  // Each plane gets an equal slice of the row in shared memory, so the cached bytes always form a
  // valid encoded chunk.
  // Bytes per plane = encoded row length / plane count. Do NOT assume ceildiv(dim, 8): that is
  // bytes-per-plane only for the bit-plane layouts (packed_1b, transposed_2b, transposed_4b),
  // where it happens to equal encoded/planes for all three. The dense byte layouts
  // (packed_7b/packed_8b) are `dim` bytes in one plane, and would read 1/8 of each row.
  const int plane_bytes =
    static_cast<int>(cuvs::preprocessing::quantize::bbq::get_encoded_row_length(dataset_document)) /
    document_planes;
  assert(plane_bytes == static_cast<int>(cuvs::preprocessing::quantize::bbq::get_encoded_row_length(
                          dataset_query)) /
                          query_planes);
  constexpr int query_plane_tile = QUERY_ROW_BYTES / query_planes;
  constexpr int plane_tile       = query_plane_tile;
  constexpr int doc_row_bytes    = query_plane_tile * document_planes;
  static_assert(plane_tile % 4 == 0, "plane_tile must be 4-byte aligned for uint32 loads");
  // Row strides too: rows are indexed as base + idx * stride and then read as uint32_t, so a
  // stride that is not a multiple of 4 misaligns every odd row. Derived from QUERY_ROW_BYTES, so
  // this is what catches an ill-chosen QUERY_ROW_BYTES rather than letting it fault at runtime.
  static_assert((doc_row_bytes + DOC_PAD) % alignof(uint32_t) == 0,
                "document row stride must be 4-byte aligned for uint32 loads");
  static_assert(
    doc_row_bytes % document_planes == 0 && doc_row_bytes / document_planes == query_plane_tile,
    "document plane stride must match query plane stride");
  // plane_bytes is the per-plane stride in bytes; plane_extent is the same in uint32 elements,
  // computed once so call sites don't re-derive it. Alignment (plane_bytes % 4 == 0, i.e.
  // dataset dim % 32 == 0) is enforced by the launcher.
  const int plane_extent             = plane_bytes / 4;
  constexpr int plane_tile_u32       = plane_tile / 4;
  constexpr int query_plane_tile_u32 = query_plane_tile / 4;
  const int n_tiles                  = raft::ceildiv(plane_bytes, plane_tile);

  for (int i = tx; i < MAX_NUM_BI_SAMPLES * SKEWED_MAX_NUM_BI_SAMPLES; i += blockDim.x) {
    s_distances_u32[i] = 0;
  }
  for (int step = 0; step < n_tiles; ++step) {
    const bool last_tile   = (step == n_tiles - 1);
    const int num_load     = last_tile ? plane_bytes - step * plane_tile : plane_tile;
    const int num_load_u32 = num_load / 4;
    const size_t base      = static_cast<size_t>(step) * plane_tile;
    for (int i = 0; i < MAX_NUM_BI_SAMPLES / num_warps; ++i) {
      const int idx = i * num_warps + warp_id;
      if (idx < new_size) {
        auto* s_doc = reinterpret_cast<uint32_t*>(s_doc_vec[idx]);
        load_vec_bbq_simt<document_planes>(
          s_doc,
          reinterpret_cast<const uint32_t*>(&dataset_document.codes(new_neighbors[idx], base)),
          plane_extent,
          num_load_u32,
          plane_tile_u32,
          lane_id);
        if (last_tile) {
          zero_pad_bbq_simt<document_planes>(s_doc, num_load_u32, plane_tile_u32, lane_id);
        }
      }
    }
    __syncthreads();

    // Query and document tiles cover the same dimension range per step (both tile at
    // query_plane_tile), so load the query tile once and run the dot product directly -- no
    // per-step query sub-tile loop. Under SelfJoin phase 1 is new x new on a single quantizer,
    // so the document buffer already holds exactly what the query buffer would: skip the load
    // and point the B operand at s_doc_vec. This is the SIMT twin of the wmma kernel's phase-1
    // buffer alias, and halves this phase's staging traffic.
    if constexpr (!SelfJoin) {
      for (int i = 0; i < MAX_NUM_BI_SAMPLES / num_warps; ++i) {
        const int idx = i * num_warps + warp_id;
        if (idx < new_size) {
          auto* s_q = reinterpret_cast<uint32_t*>(s_query_vec[idx]);
          load_vec_bbq_simt<query_planes>(
            s_q,
            reinterpret_cast<const uint32_t*>(&dataset_query.codes(new_neighbors[idx], base)),
            plane_extent,
            num_load_u32,
            query_plane_tile_u32,
            lane_id);
          if (last_tile) {
            zero_pad_bbq_simt<query_planes>(s_q, num_load_u32, query_plane_tile_u32, lane_id);
          }
        }
      }
      __syncthreads();
    }

    // Pitch columns by MAX_NUM_BI_SAMPLES (multiple of warp size) so a warp never straddles
    // row-pair boundaries. SKEWED is only for the distance matrix layout.
    constexpr int num_row_pairs = MAX_NUM_BI_SAMPLES / 2;
    for (int pair_idx = tx; pair_idx < num_row_pairs * MAX_NUM_BI_SAMPLES; pair_idx += blockDim.x) {
      const int row0 = (pair_idx / MAX_NUM_BI_SAMPLES) * 2;
      const int col  = pair_idx % MAX_NUM_BI_SAMPLES;
      if (col < new_size) {
        const int distance0 = row0 * SKEWED_MAX_NUM_BI_SAMPLES + col;
        uint32_t total0     = 0;
        uint32_t total1     = 0;
        // Phase 1 B operand: s_doc_vec under SelfJoin (see the staging note above).
        if constexpr (SelfJoin) {
          bbq_code_inner_product_2x1<DocumentLayout,
                                     QueryLayout,
                                     SelfJoin,
                                     document_planes,
                                     query_planes,
                                     doc_row_bytes,
                                     QUERY_ROW_BYTES>(
            s_doc_vec[row0], s_doc_vec[row0 + 1], s_doc_vec[col], total0, total1);
        } else {
          bbq_code_inner_product_2x1<DocumentLayout,
                                     QueryLayout,
                                     SelfJoin,
                                     document_planes,
                                     query_planes,
                                     doc_row_bytes,
                                     QUERY_ROW_BYTES>(
            s_doc_vec[row0], s_doc_vec[row0 + 1], s_query_vec[col], total0, total1);
        }
        s_distances_u32[distance0] += total0;
        if (row0 + 1 < new_size) {
          s_distances_u32[distance0 + SKEWED_MAX_NUM_BI_SAMPLES] += total1;
        }
      }
    }
    __syncthreads();
  }

  for (int step = 0; step < raft::ceildiv(new_size, num_warps); ++step) {
    const int idx_in_list = step * num_warps + tx / raft::warp_size();
    if (idx_in_list >= new_size) continue;
    auto min_elem = get_min_item_fused(s_list[idx_in_list],
                                       idx_in_list,
                                       new_neighbors,
                                       new_size,
                                       s_distances_u32,
                                       dataset_document,
                                       dataset_query,
                                       l2_norms_document,
                                       l2_norms_query,
                                       metric,
                                       dist_epilogue);
    if (min_elem.id() < gridDim.x) {
      insert_to_global_graph(min_elem, s_list[idx_in_list], graph, dists, graph_width, locks);
    }
  }

  if (!old_size) return;
  __syncthreads();

  for (int i = tx; i < MAX_NUM_BI_SAMPLES * SKEWED_MAX_NUM_BI_SAMPLES; i += blockDim.x) {
    s_distances_u32[i] = 0;
  }

  for (int step = 0; step < n_tiles; ++step) {
    const bool last_tile   = (step == n_tiles - 1);
    const int num_load     = last_tile ? plane_bytes - step * plane_tile : plane_tile;
    const int num_load_u32 = num_load / 4;
    const size_t base      = static_cast<size_t>(step) * plane_tile;
    if (n_tiles > 1) {
      for (int i = 0; i < MAX_NUM_BI_SAMPLES / num_warps; ++i) {
        const int idx = i * num_warps + warp_id;
        if (idx < new_size) {
          auto* s_doc = reinterpret_cast<uint32_t*>(s_doc_vec[idx]);
          load_vec_bbq_simt<document_planes>(
            s_doc,
            reinterpret_cast<const uint32_t*>(&dataset_document.codes(new_neighbors[idx], base)),
            plane_extent,
            num_load_u32,
            plane_tile_u32,
            lane_id);
          if (last_tile) {
            zero_pad_bbq_simt<document_planes>(s_doc, num_load_u32, plane_tile_u32, lane_id);
          }
        }
      }
      __syncthreads();
    }
    for (int i = 0; i < MAX_NUM_BI_SAMPLES / num_warps; ++i) {
      const int idx = i * num_warps + warp_id;
      if (idx < old_size) {
        auto* s_q = reinterpret_cast<uint32_t*>(s_query_vec[idx]);
        load_vec_bbq_simt<query_planes>(
          s_q,
          reinterpret_cast<const uint32_t*>(&dataset_query.codes(old_neighbors[idx], base)),
          plane_extent,
          num_load_u32,
          query_plane_tile_u32,
          lane_id);
        if (last_tile) {
          zero_pad_bbq_simt<query_planes>(s_q, num_load_u32, query_plane_tile_u32, lane_id);
        }
      }
    }
    __syncthreads();

    // Pitch columns by MAX_NUM_BI_SAMPLES (multiple of warp size) so a warp never straddles
    // row-pair boundaries. SKEWED is only for the distance matrix layout.
    constexpr int num_row_pairs = MAX_NUM_BI_SAMPLES / 2;
    for (int pair_idx = tx; pair_idx < num_row_pairs * MAX_NUM_BI_SAMPLES; pair_idx += blockDim.x) {
      const int row0 = (pair_idx / MAX_NUM_BI_SAMPLES) * 2;
      const int col  = pair_idx % MAX_NUM_BI_SAMPLES;
      if (col < old_size) {
        const int distance0 = row0 * SKEWED_MAX_NUM_BI_SAMPLES + col;
        uint32_t total0     = 0;
        uint32_t total1     = 0;
        bbq_code_inner_product_2x1<DocumentLayout,
                                   QueryLayout,
                                   SelfJoin,
                                   document_planes,
                                   query_planes,
                                   doc_row_bytes,
                                   QUERY_ROW_BYTES>(
          s_doc_vec[row0], s_doc_vec[row0 + 1], s_query_vec[col], total0, total1);
        s_distances_u32[distance0] += total0;
        if (row0 + 1 < new_size) {
          s_distances_u32[distance0 + SKEWED_MAX_NUM_BI_SAMPLES] += total1;
        }
      }
    }
    __syncthreads();
  }

  for (int step = 0; step < raft::ceildiv(MAX_NUM_BI_SAMPLES * 2, num_warps); ++step) {
    const int idx_in_list = step * num_warps + tx / raft::warp_size();
    if (idx_in_list >= new_size && idx_in_list < MAX_NUM_BI_SAMPLES) continue;
    if (idx_in_list >= MAX_NUM_BI_SAMPLES + old_size && idx_in_list < MAX_NUM_BI_SAMPLES * 2) {
      continue;
    }

    ResultItem<Index_t> min_elem{std::numeric_limits<Index_t>::max(),
                                 std::numeric_limits<DistData_t>::max()};
    if (idx_in_list < MAX_NUM_BI_SAMPLES) {
      auto temp_min_item = get_min_item_fused(s_list[idx_in_list],
                                              idx_in_list,
                                              old_neighbors,
                                              old_size,
                                              s_distances_u32,
                                              dataset_document,
                                              dataset_query,
                                              l2_norms_document,
                                              l2_norms_query,
                                              metric,
                                              dist_epilogue);
      if (temp_min_item.dist() < min_elem.dist()) { min_elem = temp_min_item; }
    } else {
      auto temp_min_item = get_min_item_fused(s_list[idx_in_list],
                                              idx_in_list - MAX_NUM_BI_SAMPLES,
                                              new_neighbors,
                                              new_size,
                                              s_distances_u32,
                                              dataset_document,
                                              dataset_query,
                                              l2_norms_document,
                                              l2_norms_query,
                                              metric,
                                              dist_epilogue,
                                              false);
      if (temp_min_item.dist() < min_elem.dist()) { min_elem = temp_min_item; }
    }
    if (min_elem.id() < gridDim.x) {
      insert_to_global_graph(min_elem, s_list[idx_in_list], graph, dists, graph_width, locks);
    }
  }
}

// Promotes one native word of dense packed_2b codes (2 bits/value, 4 values/byte; byte k =
// (v[4k]<<6)|(v[4k+1]<<4)|(v[4k+2]<<2)|v[4k+3]) into two 4-bit-width, packed_4b-style
// output words, so the result can feed a u4 MMA fragment directly. native_word's byte i (LSB
// first, i.e. lowest address) becomes output bytes 2i and 2i+1 of (out_lo, out_hi).
//
// Branch-free SWAR: each byte's top/bottom nibble (v0v1 / v2v3, 2 bits each) is "spread" into a
// full nibble-per-value byte lane-wise across all 4 bytes at once (spread(x) turns a nibble
// v_hi:v_lo into a byte (v_hi<<4)|v_lo -- exact for the 2-bit range used here), then the two
// spread words are interleaved into the final byte order with __byte_perm. Equivalence with the
// straightforward per-byte-extraction version verified exhaustively over random 32-bit inputs.
__device__ __forceinline__ void promote_packed_2b_word_to_4b(uint32_t native_word,
                                                             uint32_t& out_lo,
                                                             uint32_t& out_hi)
{
  const uint32_t tn_word = (native_word >> 4) & 0x0F0F0F0Fu;  // byte i = (v0<<2)|v1
  const uint32_t bn_word = native_word & 0x0F0F0F0Fu;         // byte i = (v2<<2)|v3
  const auto spread      = [](uint32_t w) { return ((w & 0x0C0C0C0Cu) << 2) | (w & 0x03030303u); };
  const uint32_t spread_tn = spread(tn_word);                            // byte i = (v0<<4)|v1
  const uint32_t spread_bn = spread(bn_word);                            // byte i = (v2<<4)|v3
  out_lo                   = __byte_perm(spread_tn, spread_bn, 0x5140);  // [TN0,BN0,TN1,BN1]
  out_hi                   = __byte_perm(spread_tn, spread_bn, 0x7362);  // [TN2,BN2,TN3,BN3]
}

// Promotes one native word of dense packed_1b codes (1 bit/value, 8 values/byte, MSB-first:
// the value at position 8*byte+i sits at bit (7-i)) into four 4-bit-width, packed_4b-style
// output words.
//
// Branch-free SWAR: extract the 4 (2-bit) fields of each byte lane-wise across all 4 native
// bytes at once (field[j]'s byte i = field j of native byte i -- same cross-lane-safe
// shift+mask trick as promote_packed_2b_word_to_4b), spread each field 0-3 into a
// nibble value lane-wise, then transpose the 4 resulting field-words into the 4 per-native-byte
// output words with chained __byte_perm pairs (16 bits at a time, since one __byte_perm call
// only reaches 2 of the 4 field-words). Equivalence with the straightforward
// per-byte-extraction version verified exhaustively over random 32-bit inputs.
__device__ __forceinline__ void promote_packed_1b_word_to_4b(uint32_t native_word, uint32_t out[4])
{
  uint32_t spread[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint32_t field = (native_word >> (6 - 2 * j)) & 0x03030303u;
    spread[j]            = ((field & 0x02020202u) << 3) | (field & 0x01010101u);
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    // sel picks: out byte0 = spread[.]'s byte i (source index i), out byte1 = spread[.]'s byte
    // i (source index 4+i, i.e. the second byte_perm operand); byte2/3 are don't-care (masked
    // off below).
    const uint32_t sel = 0x00000040u + i * 0x00000011u;
    const uint32_t lo  = __byte_perm(spread[0], spread[1], sel);
    const uint32_t hi  = __byte_perm(spread[2], spread[3], sel);
    out[i]             = (lo & 0xFFFFu) | ((hi & 0xFFFFu) << 16);
  }
}

// Promotion to the 4-bit-width (packed_4b-style) layout a u4 MMA fragment consumes.
// Deliberately generic over the layout and over which operand it feeds: a u4 fragment needs 4
// bits per value, so a layout storing `bits` bits per value expands one native word into 4/bits
// promoted words. packed_4b is the identity case (expansion 1, a plain word copy), so the
// symmetric kernel's "no promotion" path is just the degenerate case of the same helper rather
// than a separate code path -- and a promoted *query* (1+2, 2+2) works exactly like a promoted
// document.
template <bbq_layout Layout>
constexpr int bbq_promoted_words()
{
  static_assert(Layout == bbq_layout::packed_1b || Layout == bbq_layout::packed_2b ||
                  Layout == bbq_layout::packed_4b,
                "int4 MMA path supports packed_1b (1b), packed_2b (2b), packed_4b (4b)");
  if constexpr (Layout == bbq_layout::packed_1b) {
    return 4;
  } else if constexpr (Layout == bbq_layout::packed_2b) {
    return 2;
  } else {
    return 1;
  }
}

template <bbq_layout Layout>
__device__ __forceinline__ void promote_word_to_4b(uint32_t native_word, uint32_t* out)
{
  if constexpr (Layout == bbq_layout::packed_1b) {
    promote_packed_1b_word_to_4b(native_word, out);
  } else if constexpr (Layout == bbq_layout::packed_2b) {
    promote_packed_2b_word_to_4b(native_word, out[0], out[1]);
  } else {
    out[0] = native_word;  // packed_4b: already 4-bit-width
  }
}

// Stages one K-tile of `count` rows from `quantizer` into a 4-bit-width SMEM buffer, promoting
// each native word via promote_word_to_4b<Layout>. One warp per row (rows strided by
// num_warps), lanes strided across the row's native words; the tail past the row's end is zeroed
// so the MMA sees defined data. `native_row_bytes` is the layout's own encoded row length, and
// one native tile is RowStride-independent: BBQ_ROW_BYTES / expansion native bytes promote to
// exactly BBQ_ROW_BYTES promoted bytes, so a single n_tiles drives every operand regardless of
// how compact each one's on-disk format is.
template <bbq_layout Layout, int TileBytes, int RowStride, typename DataT, typename Index_t>
__device__ __forceinline__ void stage_promoted_tile(
  uint8_t (*dst)[RowStride],
  const bbq_device_quantizer_view<DataT, int64_t>& quantizer,
  const Index_t* neighbors,
  const int count,
  const int step,
  const int native_row_bytes,
  const int warp_id,
  const int lane_id)
{
  constexpr int expansion       = bbq_promoted_words<Layout>();
  constexpr int native_tile     = TileBytes / expansion;
  constexpr int native_tile_u32 = native_tile / 4;
  constexpr int num_warps       = BLOCK_SIZE / raft::warp_size();

  const int native_base = step * native_tile;
  const int remaining   = native_row_bytes - native_base;
  const int num_load_u32 =
    (remaining < native_tile ? (remaining > 0 ? remaining : 0) : native_tile) / 4;

  for (int i = 0; i < MAX_NUM_BI_SAMPLES / num_warps; ++i) {
    const int idx = i * num_warps + warp_id;
    if (idx >= count) continue;
    auto* s_u32 = reinterpret_cast<uint32_t*>(dst[idx]);
    const uint32_t* src =
      reinterpret_cast<const uint32_t*>(&quantizer.codes(neighbors[idx], native_base));
    for (int w = lane_id; w < num_load_u32; w += raft::warp_size()) {
      uint32_t out[expansion];
      promote_word_to_4b<Layout>(src[w], out);
#pragma unroll
      for (int e = 0; e < expansion; ++e) {
        s_u32[expansion * w + e] = out[e];
      }
    }
    // No-op unless this is the last tile and the row doesn't fill it.
    for (int w = num_load_u32 + lane_id; w < native_tile_u32; w += raft::warp_size()) {
#pragma unroll
      for (int e = 0; e < expansion; ++e) {
        s_u32[expansion * w + e] = 0;
      }
    }
  }
}

// int4 tensor-core BBQ local join, covering both the symmetric (one quantizer, self-join) and
// asymmetric (two quantizers) cases. Modeled on local_join_kernel_wmma: nvcuda::wmma fragments
// (u4 x u4 -> s32, shape m8n8k32) replace the popc/dp4a inner product; the accumulator lives in
// registers across the whole K reduction and is stored to s_distances_u32 once per (row-tile,
// col-tile), not accumulated into shared memory every step like the scalar kernel.
//
// The two cases differ in exactly three places, all compile-time:
//   * DocumentLayout / QueryLayout -- each operand is promoted to 4-bit width by
//     stage_promoted_tile, and packed_4b is the identity case, so "no promotion" is not a
//     separate code path. Any promotable pair works (4x4, 2x4, 1x4, and 1+2 / 2+2 / 1+1 by
//     dispatch alone).
//   * SelfJoin -- when both operands are the same quantizer, phase 1 (new x new) stages one
//     buffer and feeds it to both fragments, halving that phase's SMEM writes. Asymmetric can
//     never do this: document and query are two distinct quantized representations of the same
//     rows, so aliasing them would compute doc.doc, a different (worse) estimator.
//   * The metric side needs no branch at all -- get_min_item_fused's two-quantizer overloads are
//     exact generalizations of the single-quantizer ones, so symmetric just passes the same
//     quantizer and l2_norms twice.
//
// Warp tiling: num_warps = BLOCK_SIZE/32 warps arranged as a WARPS_PER_DIM x WARPS_PER_DIM square
// grid (WARPS_PER_DIM=4 so 4x4=16=num_warps), each warp owning a (MAX_NUM_BI_SAMPLES/WARPS_PER_DIM)
// region of the MAX_NUM_BI_SAMPLES x MAX_NUM_BI_SAMPLES output matrix, same as
// local_join_kernel_wmma's WMMA_M=N=16 warp assignment. Since int4 MMA tiles are MMA_M x MMA_N
// (8x8, the only shape nvcuda::wmma exposes for u4), each warp covers its region via a
// SUB_PER_DIM x SUB_PER_DIM grid of native tiles instead of a single call -- SUB_PER_DIM is a
// forced consequence of (MAX_NUM_BI_SAMPLES/MMA_M) / WARPS_PER_DIM, not an arbitrary choice.
template <bbq_layout DocumentLayout,
          bbq_layout QueryLayout,
          bool SelfJoin,
          typename DataT,
          typename Index_t,
          typename ID_t = InternalID_t<Index_t>,
          typename DistEpilogue_t>
RAFT_KERNEL
#ifdef __CUDA_ARCH__
#if (__CUDA_ARCH__) == 700 || (__CUDA_ARCH__) == 800 || (__CUDA_ARCH__) == 900 || \
  (__CUDA_ARCH__) == 1000
__launch_bounds__(BLOCK_SIZE, 4)
#else
__launch_bounds__(BLOCK_SIZE)
#endif
#endif
  local_join_kernel_bbq_wmma(const Index_t* graph_new,
                             const Index_t* rev_graph_new,
                             const int2* sizes_new,
                             const Index_t* graph_old,
                             const Index_t* rev_graph_old,
                             const int2* sizes_old,
                             const int width,
                             const bbq_device_quantizer_view<DataT, int64_t> dataset_document,
                             const bbq_device_quantizer_view<DataT, int64_t> dataset_query,
                             ID_t* graph,
                             DistData_t* dists,
                             int graph_width,
                             int* locks,
                             DistData_t* l2_norms_document,
                             DistData_t* l2_norms_query,
                             cuvs::distance::DistanceType metric,
                             DistEpilogue_t dist_epilogue)
{
// int4 sub-byte MMA (nvcuda::wmma experimental::precision::u4) exists from Turing (sm_75) up to
// and including Hopper (sm_90); Blackwell (sm_100+) dropped it. Guarding both ends here means an
// unsupported arch fails to find a kernel body at compile time rather than silently building a
// no-op, so no host-side runtime arch check is needed.
#if (__CUDA_ARCH__ >= 750 && __CUDA_ARCH__ <= 900)
  using namespace nvcuda;
  constexpr int MMA_M = 8;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 32;
  // num_warps = BLOCK_SIZE/32 = 16, arranged as a square WARPS_PER_DIM x WARPS_PER_DIM grid since
  // 4*4=16 matches exactly; the static_assert is what actually enforces this holds for the
  // current BLOCK_SIZE, WARPS_PER_DIM itself isn't derived (no trivial constexpr integer sqrt).
  constexpr int WARPS_PER_DIM = 4;
  static_assert(WARPS_PER_DIM * WARPS_PER_DIM == BLOCK_SIZE / raft::warp_size(),
                "warp grid must be square and match num_warps = BLOCK_SIZE/32");
  // Each warp owns a WARP_TILE x WARP_TILE region of the MAX_NUM_BI_SAMPLES x MAX_NUM_BI_SAMPLES
  // output matrix. TILES_PER_DIM is how many native MMA_M x MMA_N tiles span one output dimension;
  // SUB_PER_DIM (native tiles per warp per dim) is a forced consequence of TILES_PER_DIM /
  // WARPS_PER_DIM, not an arbitrary choice -- it's 2 here only because 8/4=2 for these particular
  // MAX_NUM_BI_SAMPLES/MMA_M/WARPS_PER_DIM values.
  static_assert(MAX_NUM_BI_SAMPLES % MMA_M == 0 && MMA_M == MMA_N,
                "MAX_NUM_BI_SAMPLES must divide evenly into square MMA_MxMMA_N tiles");
  constexpr int TILES_PER_DIM = MAX_NUM_BI_SAMPLES / MMA_M;
  static_assert(TILES_PER_DIM % WARPS_PER_DIM == 0,
                "warps must evenly tile the native MMA tiles in each output dimension");
  constexpr int SUB_PER_DIM = TILES_PER_DIM / WARPS_PER_DIM;
  constexpr int WARP_TILE   = SUB_PER_DIM * MMA_M;

  // Promoted (4-bit-width) staging tile, 128 B/row. Row stride is BBQ_ROW_BYTES + MMA_PAD, not
  // just BBQ_ROW_BYTES: sub-byte IMMA loads need at least 16-byte row alignment, and MMA_PAD must
  // be a multiple of 16 to preserve that -- but BBQ_ROW_BYTES=128 alone is *also* exactly 32
  // shared-memory banks (4 B/bank), so every row would land on the same bank offset and any
  // multi-row access load_matrix_sync does internally would conflict. MMA_PAD=16 breaks that
  // exact-32-bank alignment (144 B/row is not a multiple of 128 B) while staying a multiple of 16
  // for the IMMA alignment requirement.
  constexpr int BBQ_ROW_BYTES = 128;
  constexpr int MMA_PAD       = 16;
  static_assert(MMA_PAD % 16 == 0, "row padding must preserve 16-byte IMMA row alignment");
  constexpr int ELEMS_PER_TILE   = BBQ_ROW_BYTES * 2;  // 2 u4 elements/byte
  constexpr int K_STEPS_PER_TILE = ELEMS_PER_TILE / MMA_K;
  constexpr int ROW_STRIDE_U4    = (BBQ_ROW_BYTES + MMA_PAD) * 2;  // row-to-row stride, u4 elements
  // v1.3 tried decoupling this from SKEWED_MAX_NUM_BI_SAMPLES (get_min_item_fused takes stride as
  // a parameter for exactly this) with a custom MMA_STORE_STRIDE=72: store bank conflicts dropped
  // ~4.2x, but overall cycles/duration were flat, so it wasn't earning its complexity -- reverted
  // back to the shared constant.
  constexpr int MMA_STORE_STRIDE = SKEWED_MAX_NUM_BI_SAMPLES;

  // s_row_vec is the A operand: always the `new` list, document quantizer. s_col_vec is the B
  // operand: the `new` list in phase 1 and the `old` list in phase 2, query quantizer. Under
  // SelfJoin, phase 1 leaves s_col_vec untouched and both fragments read s_row_vec.
  __shared__ int s_list[MAX_NUM_BI_SAMPLES * 2];
  __shared__ __align__(16) uint8_t s_row_vec[MAX_NUM_BI_SAMPLES][BBQ_ROW_BYTES + MMA_PAD];
  __shared__ __align__(16) uint8_t s_col_vec[MAX_NUM_BI_SAMPLES][BBQ_ROW_BYTES + MMA_PAD];
  __shared__ uint32_t s_distances_u32[MAX_NUM_BI_SAMPLES * MMA_STORE_STRIDE];
  __shared__ int s_unique_counter[2];

  if (threadIdx.x == 0) {
    s_unique_counter[0] = 0;
    s_unique_counter[1] = 0;
  }

  Index_t* new_neighbors = s_list;
  Index_t* old_neighbors = s_list + MAX_NUM_BI_SAMPLES;
  const size_t list_id   = blockIdx.x;
  const int2 new_size2   = sizes_new[list_id];
  const int2 old_size2   = sizes_old[list_id];
  int new_size           = new_size2.x + new_size2.y;
  int old_size           = old_size2.x + old_size2.y;
  const int tx           = threadIdx.x;

  if (!new_size) return;
  if (tx < new_size2.x) {
    new_neighbors[tx] = graph_new[list_id * width + tx];
  } else if (tx < new_size) {
    new_neighbors[tx] = rev_graph_new[list_id * width + tx - new_size2.x];
  }
  if (tx < old_size2.x) {
    old_neighbors[tx] = graph_old[list_id * width + tx];
  } else if (tx < old_size) {
    old_neighbors[tx] = rev_graph_old[list_id * width + tx - old_size2.x];
  }
  __syncthreads();

  remove_duplicates(
    new_neighbors, new_size2.x, new_neighbors + new_size2.x, new_size2.y, s_unique_counter[0], 0);
  remove_duplicates(
    old_neighbors, old_size2.x, old_neighbors + old_size2.x, old_size2.y, s_unique_counter[1], 1);
  __syncthreads();
  new_size = new_size2.x + s_unique_counter[0];
  old_size = old_size2.x + s_unique_counter[1];

  const int warp_id       = threadIdx.x / raft::warp_size();
  const int lane_id       = threadIdx.x % raft::warp_size();
  constexpr int num_warps = BLOCK_SIZE / raft::warp_size();
  // n_tiles is driven by the *promoted* (4-bit-width) row length, which both operands share at a
  // given dim; each side's own native row length only sets how many native bytes
  // stage_promoted_tile reads per tile (dim % 32 == 0 guarantees no rounding either way).
  const int doc_row_bytes =
    static_cast<int>(cuvs::preprocessing::quantize::bbq::get_encoded_row_length(dataset_document));
  const int query_row_bytes =
    static_cast<int>(cuvs::preprocessing::quantize::bbq::get_encoded_row_length(dataset_query));
  const int promoted_row_bytes = static_cast<int>((dataset_query.dim() + 1) / 2);
  const int n_tiles            = raft::ceildiv(promoted_row_bytes, BBQ_ROW_BYTES);

  const int warp_id_y = warp_id / WARPS_PER_DIM;
  const int warp_id_x = warp_id % WARPS_PER_DIM;

  // One phase: accumulate s_row_vec (rows, `new` list) against col_buf over the whole K range,
  // then store the accumulators to s_distances_u32. col_buf is s_row_vec itself when phase 1 is a
  // self-join, otherwise s_col_vec. col_neighbors/col_size select which list the B operand stages.
  // alias_tag is an integral_constant, not a bool, so the staging skip and the b_frag source
  // selection below are both compile-time -- no runtime branch inside the kk loop.
  auto run_phase = [&](const Index_t* col_neighbors, int col_size, auto alias_tag) {
    constexpr bool alias_col = decltype(alias_tag)::value;
    wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, int> c_frag[SUB_PER_DIM][SUB_PER_DIM];
#pragma unroll
    for (int msub = 0; msub < SUB_PER_DIM; ++msub) {
#pragma unroll
      for (int nsub = 0; nsub < SUB_PER_DIM; ++nsub) {
        wmma::fill_fragment(c_frag[msub][nsub], 0);
      }
    }

    for (int step = 0; step < n_tiles; ++step) {
      stage_promoted_tile<DocumentLayout, BBQ_ROW_BYTES>(s_row_vec,
                                                         dataset_document,
                                                         new_neighbors,
                                                         new_size,
                                                         step,
                                                         doc_row_bytes,
                                                         warp_id,
                                                         lane_id);
      if constexpr (!alias_col) {
        stage_promoted_tile<QueryLayout, BBQ_ROW_BYTES>(s_col_vec,
                                                        dataset_query,
                                                        col_neighbors,
                                                        col_size,
                                                        step,
                                                        query_row_bytes,
                                                        warp_id,
                                                        lane_id);
      }
      __syncthreads();

      // a_frag depends only on (msub, kk); b_frag depends only on (nsub, kk) -- load each once
      // per kk and reuse across the other sub-tile index, instead of reloading redundantly inside
      // a full msub x nsub x kk cross product.
      // Deliberately not #pragma unroll'd: full unrolling here keeps more fragment live ranges
      // simultaneous, driving register pressure up (64/thread, tied with SMEM for the occupancy
      // cap) -- letting the compiler pick reduces that at the cost of some intra-warp ILP.
      for (int kk = 0; kk < K_STEPS_PER_TILE; ++kk) {
        wmma::fragment<wmma::matrix_a,
                       MMA_M,
                       MMA_N,
                       MMA_K,
                       wmma::experimental::precision::u4,
                       wmma::row_major>
          a_frag[SUB_PER_DIM];
        wmma::fragment<wmma::matrix_b,
                       MMA_M,
                       MMA_N,
                       MMA_K,
                       wmma::experimental::precision::u4,
                       wmma::col_major>
          b_frag[SUB_PER_DIM];
        const auto* col_buf = alias_col ? s_row_vec : s_col_vec;  // compile-time select
#pragma unroll
        for (int msub = 0; msub < SUB_PER_DIM; ++msub) {
          const int row0 = warp_id_y * WARP_TILE + msub * MMA_M;
          wmma::load_matrix_sync(a_frag[msub], s_row_vec[row0] + kk * (MMA_K / 2), ROW_STRIDE_U4);
        }
#pragma unroll
        for (int nsub = 0; nsub < SUB_PER_DIM; ++nsub) {
          const int col0 = warp_id_x * WARP_TILE + nsub * MMA_N;
          wmma::load_matrix_sync(b_frag[nsub], col_buf[col0] + kk * (MMA_K / 2), ROW_STRIDE_U4);
        }
#pragma unroll
        for (int msub = 0; msub < SUB_PER_DIM; ++msub) {
#pragma unroll
          for (int nsub = 0; nsub < SUB_PER_DIM; ++nsub) {
            wmma::mma_sync(c_frag[msub][nsub], a_frag[msub], b_frag[nsub], c_frag[msub][nsub]);
          }
        }
      }
      __syncthreads();
    }

#pragma unroll
    for (int msub = 0; msub < SUB_PER_DIM; ++msub) {
      const int row0 = warp_id_y * WARP_TILE + msub * MMA_M;
#pragma unroll
      for (int nsub = 0; nsub < SUB_PER_DIM; ++nsub) {
        const int col0 = warp_id_x * WARP_TILE + nsub * MMA_N;
        wmma::store_matrix_sync(
          reinterpret_cast<int*>(s_distances_u32) + row0 * MMA_STORE_STRIDE + col0,
          c_frag[msub][nsub],
          MMA_STORE_STRIDE,
          wmma::mem_row_major);
      }
    }
    __syncthreads();
  };

  // list_idx indexes s_list (which id this row belongs to); idx_in_list is the row's position in
  // the distance matrix. They differ in phase 2's second branch, where the id comes from the
  // old-neighbour half of s_list but the matrix position is relative to the new half.
  auto min_over =
    [&](int list_idx, int idx_in_list, const Index_t* neighbs, int neighbs_size, bool find_in_row) {
      return get_min_item_fused(s_list[list_idx],
                                idx_in_list,
                                neighbs,
                                neighbs_size,
                                s_distances_u32,
                                dataset_document,
                                dataset_query,
                                l2_norms_document,
                                l2_norms_query,
                                metric,
                                dist_epilogue,
                                find_in_row,
                                MMA_STORE_STRIDE);
    };

  // ---- Phase 1: new x new ----
  run_phase(new_neighbors, new_size, std::integral_constant<bool, SelfJoin>{});

  for (int step = 0; step < raft::ceildiv(new_size, num_warps); ++step) {
    const int idx_in_list = step * num_warps + tx / raft::warp_size();
    if (idx_in_list >= new_size) continue;
    auto min_elem = min_over(idx_in_list, idx_in_list, new_neighbors, new_size, true);
    if (min_elem.id() < gridDim.x) {
      insert_to_global_graph(min_elem, s_list[idx_in_list], graph, dists, graph_width, locks);
    }
  }

  if (!old_size) return;
  __syncthreads();

  // ---- Phase 2: new x old ----
  run_phase(old_neighbors, old_size, std::false_type{});

  for (int step = 0; step < raft::ceildiv(MAX_NUM_BI_SAMPLES * 2, num_warps); ++step) {
    const int idx_in_list = step * num_warps + tx / raft::warp_size();
    if (idx_in_list >= new_size && idx_in_list < MAX_NUM_BI_SAMPLES) continue;
    if (idx_in_list >= MAX_NUM_BI_SAMPLES + old_size && idx_in_list < MAX_NUM_BI_SAMPLES * 2) {
      continue;
    }

    ResultItem<Index_t> min_elem{std::numeric_limits<Index_t>::max(),
                                 std::numeric_limits<DistData_t>::max()};
    auto temp_min_item =
      idx_in_list < MAX_NUM_BI_SAMPLES
        ? min_over(idx_in_list, idx_in_list, old_neighbors, old_size, true)
        : min_over(idx_in_list, idx_in_list - MAX_NUM_BI_SAMPLES, new_neighbors, new_size, false);
    if (temp_min_item.dist() < min_elem.dist()) { min_elem = temp_min_item; }
    if (min_elem.id() < gridDim.x) {
      insert_to_global_graph(min_elem, s_list[idx_in_list], graph, dists, graph_width, locks);
    }
  }
#endif  // 750 <= __CUDA_ARCH__ <= 900
}

// launch_bounds here denote BLOCK_SIZE = 512 and MIN_BLOCKS_PER_SM = 4
// Per
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications,
// MAX_RESIDENT_THREAD_PER_SM = BLOCK_SIZE * BLOCKS_PER_SM = 2048
// For architectures 750 and 860 (890), the values for MAX_RESIDENT_THREAD_PER_SM
// is 1024 and 1536 respectively, which means the bounds don't work anymore
// Used for fp32 data downcast to fp16, and all types using non-L1 distance metric.
template <typename Data_t,
          typename Index_t,
          typename ID_t = InternalID_t<Index_t>,
          typename DistEpilogue_t>
RAFT_KERNEL
#ifdef __CUDA_ARCH__
// Use minBlocksPerMultiprocessor = 4 on specific arches
#if (__CUDA_ARCH__) == 700 || (__CUDA_ARCH__) == 800 || (__CUDA_ARCH__) == 900 || \
  (__CUDA_ARCH__) == 1000
__launch_bounds__(BLOCK_SIZE, 4)
#else
__launch_bounds__(BLOCK_SIZE)
#endif
#endif
  local_join_kernel_wmma(const Index_t* graph_new,
                         const Index_t* rev_graph_new,
                         const int2* sizes_new,
                         const Index_t* graph_old,
                         const Index_t* rev_graph_old,
                         const int2* sizes_old,
                         const int width,
                         const Data_t* data,
                         const int data_dim,
                         ID_t* graph,
                         DistData_t* dists,
                         int graph_width,
                         int* locks,
                         DistData_t* l2_norms,
                         cuvs::distance::DistanceType metric,
                         DistEpilogue_t dist_epilogue)
{
#if (__CUDA_ARCH__ >= 700)
  using namespace nvcuda;
  __shared__ int s_list[MAX_NUM_BI_SAMPLES * 2];

  constexpr int APAD           = 8;
  constexpr int BPAD           = 8;
  constexpr int TILE_COL_WIDTH = 128;
  __shared__ __half s_nv[MAX_NUM_BI_SAMPLES][TILE_COL_WIDTH + APAD];  // New vectors
  __shared__ __half s_ov[MAX_NUM_BI_SAMPLES][TILE_COL_WIDTH + BPAD];  // Old vectors
  static_assert(sizeof(float) * MAX_NUM_BI_SAMPLES * SKEWED_MAX_NUM_BI_SAMPLES <=
                sizeof(__half) * MAX_NUM_BI_SAMPLES * (TILE_COL_WIDTH + BPAD));
  // s_distances: MAX_NUM_BI_SAMPLES x SKEWED_MAX_NUM_BI_SAMPLES, reuse the space of s_ov
  float* s_distances    = (float*)&s_ov[0][0];
  int* s_unique_counter = (int*)&s_ov[0][0];

  if (threadIdx.x == 0) {
    s_unique_counter[0] = 0;
    s_unique_counter[1] = 0;
  }

  Index_t* new_neighbors = s_list;
  Index_t* old_neighbors = s_list + MAX_NUM_BI_SAMPLES;

  size_t list_id      = blockIdx.x;
  int2 list_new_size2 = sizes_new[list_id];
  int list_new_size   = list_new_size2.x + list_new_size2.y;
  int2 list_old_size2 = sizes_old[list_id];
  int list_old_size   = list_old_size2.x + list_old_size2.y;

  if (!list_new_size) return;
  int tx = threadIdx.x;

  if (tx < list_new_size2.x) {
    new_neighbors[tx] = graph_new[list_id * width + tx];
  } else if (tx >= list_new_size2.x && tx < list_new_size) {
    new_neighbors[tx] = rev_graph_new[list_id * width + tx - list_new_size2.x];
  }

  if (tx < list_old_size2.x) {
    old_neighbors[tx] = graph_old[list_id * width + tx];
  } else if (tx >= list_old_size2.x && tx < list_old_size) {
    old_neighbors[tx] = rev_graph_old[list_id * width + tx - list_old_size2.x];
  }

  __syncthreads();

  remove_duplicates(new_neighbors,
                    list_new_size2.x,
                    new_neighbors + list_new_size2.x,
                    list_new_size2.y,
                    s_unique_counter[0],
                    0);

  remove_duplicates(old_neighbors,
                    list_old_size2.x,
                    old_neighbors + list_old_size2.x,
                    list_old_size2.y,
                    s_unique_counter[1],
                    1);
  __syncthreads();
  list_new_size = list_new_size2.x + s_unique_counter[0];
  list_old_size = list_old_size2.x + s_unique_counter[1];

  int warp_id             = threadIdx.x / raft::warp_size();
  int lane_id             = threadIdx.x % raft::warp_size();
  constexpr int num_warps = BLOCK_SIZE / raft::warp_size();

  int warp_id_y = warp_id / 4;
  int warp_id_x = warp_id % 4;

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  if (metric != cuvs::distance::DistanceType::BitwiseHamming) {
    wmma::fill_fragment(c_frag, 0.0);

    for (int step = 0; step < raft::ceildiv(data_dim, TILE_COL_WIDTH); step++) {
      int num_load_elems = (step == raft::ceildiv(data_dim, TILE_COL_WIDTH) - 1)
                             ? data_dim - step * TILE_COL_WIDTH
                             : TILE_COL_WIDTH;
#pragma unroll
      for (int i = 0; i < MAX_NUM_BI_SAMPLES / num_warps; i++) {
        int idx = i * num_warps + warp_id;
        if (idx < list_new_size) {
          size_t neighbor_id = new_neighbors[idx];
          size_t idx_in_data = neighbor_id * data_dim;
          // converted to fp16 on-the-fly while loading
          load_vec(s_nv[idx],
                   data + idx_in_data + step * TILE_COL_WIDTH,
                   num_load_elems,
                   TILE_COL_WIDTH,
                   lane_id);
        }
      }
      __syncthreads();

      for (int i = 0; i < TILE_COL_WIDTH / WMMA_K; i++) {
        wmma::load_matrix_sync(
          a_frag, s_nv[warp_id_y * WMMA_M] + i * WMMA_K, TILE_COL_WIDTH + APAD);
        wmma::load_matrix_sync(
          b_frag, s_nv[warp_id_x * WMMA_N] + i * WMMA_K, TILE_COL_WIDTH + BPAD);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
      }
    }

    wmma::store_matrix_sync(
      s_distances + warp_id_y * WMMA_M * SKEWED_MAX_NUM_BI_SAMPLES + warp_id_x * WMMA_N,
      c_frag,
      SKEWED_MAX_NUM_BI_SAMPLES,
      wmma::mem_row_major);
  }
  __syncthreads();

  calculate_metric(s_distances,
                   new_neighbors,
                   list_new_size,
                   new_neighbors,
                   list_new_size,
                   data,
                   data_dim,
                   l2_norms,
                   metric,
                   dist_epilogue);
  __syncthreads();

  for (int step = 0; step < raft::ceildiv(list_new_size, num_warps); step++) {
    int idx_in_list = step * num_warps + tx / raft::warp_size();
    if (idx_in_list >= list_new_size) continue;
    auto min_elem = get_min_item(s_list[idx_in_list], idx_in_list, new_neighbors, s_distances);
    if (min_elem.id() < gridDim.x) {
      insert_to_global_graph(min_elem, s_list[idx_in_list], graph, dists, graph_width, locks);
    }
  }

  if (!list_old_size) return;

  __syncthreads();

  if (metric != cuvs::distance::DistanceType::BitwiseHamming) {
    wmma::fill_fragment(c_frag, 0.0);
    for (int step = 0; step < raft::ceildiv(data_dim, TILE_COL_WIDTH); step++) {
      int num_load_elems = (step == raft::ceildiv(data_dim, TILE_COL_WIDTH) - 1)
                             ? data_dim - step * TILE_COL_WIDTH
                             : TILE_COL_WIDTH;
      if (TILE_COL_WIDTH < data_dim) {
#pragma unroll
        for (int i = 0; i < MAX_NUM_BI_SAMPLES / num_warps; i++) {
          int idx = i * num_warps + warp_id;
          if (idx < list_new_size) {
            size_t neighbor_id = new_neighbors[idx];
            size_t idx_in_data = neighbor_id * data_dim;
            load_vec(s_nv[idx],
                     data + idx_in_data + step * TILE_COL_WIDTH,
                     num_load_elems,
                     TILE_COL_WIDTH,
                     lane_id);
          }
        }
      }
#pragma unroll
      for (int i = 0; i < MAX_NUM_BI_SAMPLES / num_warps; i++) {
        int idx = i * num_warps + warp_id;
        if (idx < list_old_size) {
          size_t neighbor_id = old_neighbors[idx];
          size_t idx_in_data = neighbor_id * data_dim;
          load_vec(s_ov[idx],
                   data + idx_in_data + step * TILE_COL_WIDTH,
                   num_load_elems,
                   TILE_COL_WIDTH,
                   lane_id);
        }
      }
      __syncthreads();

      for (int i = 0; i < TILE_COL_WIDTH / WMMA_K; i++) {
        wmma::load_matrix_sync(
          a_frag, s_nv[warp_id_y * WMMA_M] + i * WMMA_K, TILE_COL_WIDTH + APAD);
        wmma::load_matrix_sync(
          b_frag, s_ov[warp_id_x * WMMA_N] + i * WMMA_K, TILE_COL_WIDTH + BPAD);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
      }
    }

    wmma::store_matrix_sync(
      s_distances + warp_id_y * WMMA_M * SKEWED_MAX_NUM_BI_SAMPLES + warp_id_x * WMMA_N,
      c_frag,
      SKEWED_MAX_NUM_BI_SAMPLES,
      wmma::mem_row_major);
    __syncthreads();
  }

  calculate_metric(s_distances,
                   new_neighbors,
                   list_new_size,
                   old_neighbors,
                   list_old_size,
                   data,
                   data_dim,
                   l2_norms,
                   metric,
                   dist_epilogue);

  __syncthreads();

  for (int step = 0; step < raft::ceildiv(MAX_NUM_BI_SAMPLES * 2, num_warps); step++) {
    int idx_in_list = step * num_warps + tx / raft::warp_size();
    if (idx_in_list >= list_new_size && idx_in_list < MAX_NUM_BI_SAMPLES) continue;
    if (idx_in_list >= MAX_NUM_BI_SAMPLES + list_old_size && idx_in_list < MAX_NUM_BI_SAMPLES * 2)
      continue;
    ResultItem<Index_t> min_elem{std::numeric_limits<Index_t>::max(),
                                 std::numeric_limits<DistData_t>::max()};
    if (idx_in_list < MAX_NUM_BI_SAMPLES) {
      auto temp_min_item =
        get_min_item(s_list[idx_in_list], idx_in_list, old_neighbors, s_distances);
      if (temp_min_item.dist() < min_elem.dist()) { min_elem = temp_min_item; }
    } else {
      auto temp_min_item = get_min_item(
        s_list[idx_in_list], idx_in_list - MAX_NUM_BI_SAMPLES, new_neighbors, s_distances, false);
      if (temp_min_item.dist() < min_elem.dist()) { min_elem = temp_min_item; }
    }

    if (min_elem.id() < gridDim.x) {
      insert_to_global_graph(min_elem, s_list[idx_in_list], graph, dists, graph_width, locks);
    }
  }
#endif
}

namespace {
template <typename Index_t>
int insert_to_ordered_list(InternalID_t<Index_t>* list,
                           DistData_t* dist_list,
                           const int width,
                           const InternalID_t<Index_t> neighb_id,
                           const DistData_t dist)
{
  if (dist > dist_list[width - 1]) { return width; }

  int idx_insert      = width;
  bool position_found = false;
  for (int i = 0; i < width; i++) {
    if (list[i].id() == neighb_id.id()) { return width; }
    if (!position_found && dist_list[i] > dist) {
      idx_insert     = i;
      position_found = true;
    }
  }
  if (idx_insert == width) return idx_insert;

  memmove(list + idx_insert + 1, list + idx_insert, sizeof(*list) * (width - idx_insert - 1));
  memmove(dist_list + idx_insert + 1,
          dist_list + idx_insert,
          sizeof(*dist_list) * (width - idx_insert - 1));

  list[idx_insert]      = neighb_id;
  dist_list[idx_insert] = dist;
  return idx_insert;
};

}  // namespace

template <typename Index_t>
GnndGraph<Index_t>::GnndGraph(raft::resources const& res,
                              const size_t nrow,
                              const size_t node_degree,
                              const size_t internal_node_degree,
                              const size_t num_samples)
  : res(res),
    nrow(nrow),
    node_degree(node_degree),
    num_samples(num_samples),
    bloom_filter(nrow, internal_node_degree / segment_size, 3),
    h_dists{raft::make_host_matrix<DistData_t, size_t, raft::row_major>(nrow, node_degree)},
    h_graph_new{raft::make_pinned_matrix<Index_t, size_t, raft::row_major>(res, nrow, num_samples)},
    h_list_sizes_new{raft::make_pinned_vector<int2, size_t>(res, nrow)},
    h_graph_old{raft::make_pinned_matrix<Index_t, size_t, raft::row_major>(res, nrow, num_samples)},
    h_list_sizes_old{raft::make_pinned_vector<int2, size_t>(res, nrow)}
{
  // node_degree must be a multiple of segment_size;
  RAFT_EXPECTS(node_degree % segment_size == 0,
               "node_degree (%u) %% segment_size (%u) == 0",
               static_cast<uint32_t>(node_degree),
               static_cast<uint32_t>(segment_size));
  RAFT_EXPECTS(internal_node_degree % segment_size == 0,
               "internal_node_degree (%u) %% segment_size (%u) == 0",
               static_cast<uint32_t>(internal_node_degree),
               static_cast<uint32_t>(segment_size));

  num_segments = node_degree / segment_size;
  // To save the CPU memory, graph should be allocated by external function
  h_graph = nullptr;
}

// This is the only operation on the CPU that cannot be overlapped.
// So it should be as fast as possible.
template <typename Index_t>
void GnndGraph<Index_t>::sample_graph_new(InternalID_t<Index_t>* new_neighbors, const size_t width)
{
  std::fill_n(h_graph_new.data_handle(), nrow * num_samples, std::numeric_limits<Index_t>::max());
#pragma omp parallel for
  for (size_t i = 0; i < nrow; i++) {
    auto list_new                       = h_graph_new.data_handle() + i * num_samples;
    h_list_sizes_new.data_handle()[i].x = 0;
    h_list_sizes_new.data_handle()[i].y = 0;

    for (size_t j = 0; j < width; j++) {
      auto new_neighb_id = new_neighbors[i * width + j].id();
      if ((size_t)new_neighb_id >= nrow) break;
      if (bloom_filter.check(i, new_neighb_id)) { continue; }
      bloom_filter.add(i, new_neighb_id);
      new_neighbors[i * width + j].mark_old();
      list_new[h_list_sizes_new.data_handle()[i].x++] = new_neighb_id;
      if (h_list_sizes_new.data_handle()[i].x == num_samples) break;
    }
  }
}

// Initialize the graph with random neighbors and apply the segmentation rule. Split the neighbor
// list into num_segments segments. A neighbor with index v is placed into segment (v %
// num_segments). The details are in Sec 4.3 in H Wang et.al. "Fast k-NN Graph Construction by GPU
// based NN-Descent".
template <typename Index_t>
void GnndGraph<Index_t>::init_random_graph()
{
  const auto extended_nrows =
    raft::round_up_safe(static_cast<uint32_t>(nrow), static_cast<uint32_t>(num_segments));
  for (uint32_t seg_id = 0; seg_id < static_cast<uint32_t>(num_segments); seg_id++) {
    const auto actual_segment_size =
      std::min(static_cast<uint64_t>(segment_size), node_degree - seg_id * segment_size);

    uint64_t stride = nrow / segment_size;
    while (std::gcd(extended_nrows, stride) != 1 || std::gcd(actual_segment_size, stride) != 1) {
      stride++;
    }

#pragma omp parallel for
    for (uint64_t i = 0; i < nrow; i++) {
      // Generate a starting index. The node ((i + 1) % nrow) will be included in the neighbor list
      // of node i. This rule guarantees the connectivity of the graph.
      uint64_t id;
      if ((i + 1) % num_segments == seg_id) {
        id = i + 1;
        if (id >= nrow) { id = seg_id; }
      } else {
        id = (i + 1) * num_segments + seg_id;
      }

      for (uint32_t j = 0; j < actual_segment_size; j++) {
        for (uint32_t steps = 0; (id >= nrow || id == i) && steps < extended_nrows; steps++) {
          id = (id + stride * num_segments) % extended_nrows;
        }

        const auto store_index = i * node_degree + seg_id * segment_size + j;
        h_graph[store_index].id_with_flag() =
          (id >= nrow || id == i) ? std::numeric_limits<Index_t>::max() : id;
        h_dists.data_handle()[store_index] = std::numeric_limits<DistData_t>::max();

        id = (id + num_segments * stride) % nrow;
      }
    }
  }
}

template <typename Index_t>
void GnndGraph<Index_t>::sample_graph(bool sample_new)
{
  std::fill_n(h_graph_old.data_handle(), nrow * num_samples, std::numeric_limits<Index_t>::max());
  if (sample_new) {
    std::fill_n(h_graph_new.data_handle(), nrow * num_samples, std::numeric_limits<Index_t>::max());
  }

#pragma omp parallel for
  for (size_t i = 0; i < nrow; i++) {
    h_list_sizes_old.data_handle()[i].x = 0;
    h_list_sizes_old.data_handle()[i].y = 0;
    h_list_sizes_new.data_handle()[i].x = 0;
    h_list_sizes_new.data_handle()[i].y = 0;

    auto list     = h_graph + i * node_degree;
    auto list_old = h_graph_old.data_handle() + i * num_samples;
    auto list_new = h_graph_new.data_handle() + i * num_samples;
    for (int j = 0; j < segment_size; j++) {
      for (int k = 0; k < num_segments; k++) {
        auto neighbor = list[k * segment_size + j];
        if ((size_t)neighbor.id() >= nrow) continue;
        if (!neighbor.is_new()) {
          if (h_list_sizes_old.data_handle()[i].x < num_samples) {
            list_old[h_list_sizes_old.data_handle()[i].x++] = neighbor.id();
          }
        } else if (sample_new) {
          if (h_list_sizes_new.data_handle()[i].x < num_samples) {
            list[k * segment_size + j].mark_old();
            list_new[h_list_sizes_new.data_handle()[i].x++] = neighbor.id();
          }
        }
        if (h_list_sizes_old.data_handle()[i].x == num_samples &&
            h_list_sizes_new.data_handle()[i].x == num_samples) {
          break;
        }
      }
      if (h_list_sizes_old.data_handle()[i].x == num_samples &&
          h_list_sizes_new.data_handle()[i].x == num_samples) {
        break;
      }
    }
  }
}

template <typename Index_t>
void GnndGraph<Index_t>::update_graph(const InternalID_t<Index_t>* new_neighbors,
                                      const DistData_t* new_dists,
                                      const size_t width,
                                      std::atomic<int64_t>& update_counter)
{
#pragma omp parallel for
  for (size_t i = 0; i < nrow; i++) {
    for (size_t j = 0; j < width; j++) {
      auto new_neighb_id = new_neighbors[i * width + j];
      auto new_dist      = new_dists[i * width + j];
      if (new_dist == std::numeric_limits<DistData_t>::max()) break;
      if ((size_t)new_neighb_id.id() == i) continue;
      int seg_idx    = new_neighb_id.id() % num_segments;
      auto list      = h_graph + i * node_degree + seg_idx * segment_size;
      auto dist_list = h_dists.data_handle() + i * node_degree + seg_idx * segment_size;
      int insert_pos =
        insert_to_ordered_list(list, dist_list, segment_size, new_neighb_id, new_dist);
      if (i % counter_interval == 0 && insert_pos != segment_size) { update_counter++; }
    }
  }
}

template <typename Index_t>
void GnndGraph<Index_t>::sort_lists()
{
#pragma omp parallel for
  for (size_t i = 0; i < nrow; i++) {
    std::vector<std::pair<DistData_t, Index_t>> new_list;
    for (size_t j = 0; j < node_degree; j++) {
      new_list.emplace_back(h_dists.data_handle()[i * node_degree + j],
                            h_graph[i * node_degree + j].id());
    }
    std::sort(new_list.begin(), new_list.end());
    for (size_t j = 0; j < node_degree; j++) {
      h_graph[i * node_degree + j].id_with_flag() = new_list[j].second;
      h_dists.data_handle()[i * node_degree + j]  = new_list[j].first;
    }
  }
}

template <typename Index_t>
void GnndGraph<Index_t>::clear()
{
  bloom_filter.clear();
}

template <typename Index_t>
GnndGraph<Index_t>::~GnndGraph()
{
}

template <typename Data_t, typename Index_t>
GNND<Data_t, Index_t>::GNND(raft::resources const& res, const BuildConfig& build_config)
  : res(res),
    build_config_(build_config),
    graph_(res,
           build_config.max_dataset_size,
           align32::roundUp(build_config.node_degree),
           align32::roundUp(build_config.internal_node_degree ? build_config.internal_node_degree
                                                              : build_config.node_degree),
           NUM_SAMPLES),
    nrow_(build_config.max_dataset_size),
    ndim_(build_config.dataset_dim),
    l2_norms_{raft::make_device_vector<DistData_t, size_t>(res, 0)},
    graph_buffer_{
      raft::make_device_matrix<ID_t, size_t, raft::row_major>(res, nrow_, DEGREE_ON_DEVICE)},
    dists_buffer_{
      raft::make_device_matrix<DistData_t, size_t, raft::row_major>(res, nrow_, DEGREE_ON_DEVICE)},
    graph_host_buffer_{
      raft::make_pinned_matrix<ID_t, size_t, raft::row_major>(res, nrow_, DEGREE_ON_DEVICE)},
    dists_host_buffer_{
      raft::make_pinned_matrix<DistData_t, size_t, raft::row_major>(res, nrow_, DEGREE_ON_DEVICE)},
    d_locks_{raft::make_device_vector<int, size_t>(res, nrow_)},
    h_rev_graph_new_{
      raft::make_pinned_matrix<Index_t, size_t, raft::row_major>(res, nrow_, NUM_SAMPLES)},
    h_graph_old_(
      raft::make_pinned_matrix<Index_t, size_t, raft::row_major>(res, nrow_, NUM_SAMPLES)),
    h_rev_graph_old_{
      raft::make_pinned_matrix<Index_t, size_t, raft::row_major>(res, nrow_, NUM_SAMPLES)},
    d_list_sizes_new_{raft::make_device_vector<int2, size_t>(res, nrow_)},
    d_list_sizes_old_{raft::make_device_vector<int2, size_t>(res, nrow_)}
{
  static_assert(NUM_SAMPLES <= 32);

  raft::matrix::fill(res, dists_buffer_.view(), std::numeric_limits<float>::max());
  auto graph_buffer_view = raft::make_device_matrix_view<Index_t, int64_t>(
    reinterpret_cast<Index_t*>(graph_buffer_.data_handle()), nrow_, DEGREE_ON_DEVICE);
  raft::matrix::fill(res, graph_buffer_view, std::numeric_limits<Index_t>::max());
  raft::matrix::fill(res, d_locks_.view(), 0);

  if (build_config.metric == cuvs::distance::DistanceType::L2Expanded ||
      build_config.metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
      build_config.metric == cuvs::distance::DistanceType::CosineExpanded) {
    // for device memory efficiency, we do not allocate a separate array for the data
    // to normalize the data when using CosineExpanded metric. Instead, we use the l2_norms_ vector
    // and compute inside the calculate_metric kernel.
    l2_norms_ = raft::make_device_vector<DistData_t, size_t>(res, nrow_);
  }
};

template <typename Data_t, typename Index_t>
void GNND<Data_t, Index_t>::reset(raft::resources const& res)
{
  raft::matrix::fill(res, dists_buffer_.view(), std::numeric_limits<float>::max());
  auto graph_buffer_view = raft::make_device_matrix_view<Index_t, int64_t>(
    reinterpret_cast<Index_t*>(graph_buffer_.data_handle()), nrow_, DEGREE_ON_DEVICE);
  raft::matrix::fill(res, graph_buffer_view, std::numeric_limits<Index_t>::max());
  raft::matrix::fill(res, d_locks_.view(), 0);
}

template <typename Data_t, typename Index_t>
void GNND<Data_t, Index_t>::add_reverse_edges(Index_t* graph_ptr,
                                              Index_t* h_rev_graph_ptr,
                                              Index_t* d_rev_graph_ptr,
                                              int2* list_sizes,
                                              cudaStream_t stream)
{
  raft::matrix::fill(
    res,
    raft::make_device_matrix_view<Index_t, int64_t>(d_rev_graph_ptr, nrow_, DEGREE_ON_DEVICE),
    std::numeric_limits<Index_t>::max());
  add_rev_edges_kernel<<<nrow_, raft::warp_size(), 0, stream>>>(
    graph_ptr, d_rev_graph_ptr, NUM_SAMPLES, list_sizes);
  raft::copy(res,
             raft::make_host_vector_view(h_rev_graph_ptr, nrow_ * NUM_SAMPLES),
             raft::make_device_vector_view(d_rev_graph_ptr, nrow_ * NUM_SAMPLES));
}

template <typename Data_t, typename Index_t>
template <typename DistEpilogue_t>
void GNND<Data_t, Index_t>::local_join(cudaStream_t stream, DistEpilogue_t dist_epilogue)
{
  raft::matrix::fill(res, dists_buffer_.view(), std::numeric_limits<float>::max());

  // Kernel dispatch logic, based on the effective distance-computation dtype (which depends on
  // the input dtype and dist_comp_dtype):
  //   fp32 dist (only fp32 input, dist_comp_dtype == FP32 or AUTO with dim <= 16) -> SIMT: scalar
  //     element-wise distance computation in fp32.
  //   fp16 dist (everything else: fp16/int8/uint8 input, or fp32 input with dist_comp_dtype ==
  //     FP16 or AUTO with dim > 16) -> WMMA (tensor-core accelerated dot product). Non-fp16
  //     dtypes are converted to fp16 on-the-fly while loading into shared memory; for fp32 host
  //     input this conversion happens earlier at copy-in time (see d_data_half_).
  //   L1 distance for any input -> SIMT (L1 needs element-wise ops, can't use tensor cores).
  using DCT = cuvs::neighbors::nn_descent::DIST_COMP_DTYPE;
  bool use_fp16_dist =
    std::is_same_v<input_t, float> && (build_config_.dist_comp_dtype == DCT::FP16 ||
                                       (build_config_.dist_comp_dtype == DCT::AUTO && ndim_ > 16));
  bool use_simt = (std::is_same_v<input_t, float> && !use_fp16_dist) ||
                  build_config_.metric == cuvs::distance::DistanceType::L1;

  auto launch_kernel = [&](auto* typed_ptr) {
    if (use_simt) {
      local_join_kernel_simt<<<nrow_, BLOCK_SIZE, 0, stream>>>(graph_.h_graph_new.data_handle(),
                                                               h_rev_graph_new_.data_handle(),
                                                               d_list_sizes_new_.data_handle(),
                                                               h_graph_old_.data_handle(),
                                                               h_rev_graph_old_.data_handle(),
                                                               d_list_sizes_old_.data_handle(),
                                                               NUM_SAMPLES,
                                                               typed_ptr,
                                                               ndim_,
                                                               graph_buffer_.data_handle(),
                                                               dists_buffer_.data_handle(),
                                                               DEGREE_ON_DEVICE,
                                                               d_locks_.data_handle(),
                                                               l2_norms_.data_handle(),
                                                               build_config_.metric,
                                                               dist_epilogue);
    } else {
      local_join_kernel_wmma<<<nrow_, BLOCK_SIZE, 0, stream>>>(graph_.h_graph_new.data_handle(),
                                                               h_rev_graph_new_.data_handle(),
                                                               d_list_sizes_new_.data_handle(),
                                                               h_graph_old_.data_handle(),
                                                               h_rev_graph_old_.data_handle(),
                                                               d_list_sizes_old_.data_handle(),
                                                               NUM_SAMPLES,
                                                               typed_ptr,
                                                               ndim_,
                                                               graph_buffer_.data_handle(),
                                                               dists_buffer_.data_handle(),
                                                               DEGREE_ON_DEVICE,
                                                               d_locks_.data_handle(),
                                                               l2_norms_.data_handle(),
                                                               build_config_.metric,
                                                               dist_epilogue);
    }
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  };

  if (d_data_half_.has_value()) {
    // Host fp32 input was downcast to a device-side fp16 buffer because distances are computed in
    // fp16 (dist_comp_dtype == FP16, or AUTO with dim > 16).
    launch_kernel(static_cast<const half*>(d_data_ptr_));
  } else {
    // Data stored as input_t: device data used directly, or host data copied as-is.
    launch_kernel(static_cast<const input_t*>(d_data_ptr_));
  }
}

template <typename Data_t, typename Index_t>
template <typename DistEpilogue_t>
void GNND<Data_t, Index_t>::local_join(cudaStream_t stream,
                                       cuvs::neighbors::device_bbq_dataset_view<int64_t> dataset,
                                       DistEpilogue_t dist_epilogue)
{
  namespace bbq = cuvs::preprocessing::quantize::bbq;
  using L       = bbq_layout;
  raft::matrix::fill(res, dists_buffer_.view(), std::numeric_limits<float>::max());

  // Both kernels take the same (document, query) pair, so there is no symmetric/asymmetric split
  // here: a single quantizer just means the same one on both operands, which is exactly what
  // SelfJoin encodes. Picking the two quantizers is all that differs.
  const bool self_join = dataset.quantizers.size() == 1;
  const bool has_1b    = dataset.has_bit_and_layout(1, L::packed_1b);
  const bool has_2b    = dataset.has_bit_and_layout(2, L::packed_2b);
  const bool has_4b    = dataset.has_bit_and_layout(4, L::packed_4b);
  const bool has_2bt   = dataset.has_bit_and_layout(2, L::transposed_2b);
  const bool has_4bt   = dataset.has_bit_and_layout(4, L::transposed_4b);

  // Asymmetric: a packed_4b query selects the tensor-core path, a transposed query the SIMT one.
  const bool tc_pair   = has_4b && (has_2b || has_1b);
  const bool simt_pair = (has_4bt && (has_1b || has_2bt)) || (has_2bt && has_1b);
  RAFT_EXPECTS(self_join || tc_pair || simt_pair,
               "Unsupported BBQ layout pair for asymmetric local join. Supported: "
               "packed_2b/packed_1b x packed_4b (tensor core); packed_1b x transposed_2b, "
               "packed_1b x transposed_4b, transposed_2b x transposed_4b (SIMT).");
  auto quantizer_query    = self_join
                              ? dataset.quantizers[0]
                              : (tc_pair ? dataset.get_quantizer(4, L::packed_4b)
                                         : (has_4bt ? dataset.get_quantizer(4, L::transposed_4b)
                                                    : dataset.get_quantizer(2, L::transposed_2b)));
  auto quantizer_document = self_join
                              ? dataset.quantizers[0]
                              : (has_1b ? dataset.get_quantizer(1, L::packed_1b)
                                        : (tc_pair ? dataset.get_quantizer(2, L::packed_2b)
                                                   : dataset.get_quantizer(2, L::transposed_2b)));

  // load_vec_bbq_simt / stage_promoted_tile cast code buffers to uint32_t*, so every plane stride
  // must be 4-byte aligned.
  {
    const auto len     = bbq::get_encoded_row_length(quantizer_query);
    const int n_planes = quantizer_query.layout == L::transposed_2b   ? 2
                         : quantizer_query.layout == L::transposed_4b ? 4
                                                                      : 1;
    RAFT_EXPECTS(len % (4u * static_cast<uint32_t>(n_planes)) == 0,
                 "BBQ local join requires the encoded row length to be a multiple of 4*n_planes "
                 "for 32-bit aligned plane loads, got %u with n_planes = %d",
                 len,
                 n_planes);
    RAFT_EXPECTS(quantizer_document.dim() % 32 == 0,
                 "BBQ local join requires dataset dim to be a multiple of 32, got %lld",
                 static_cast<long long>(quantizer_document.dim()));
  }

  auto l2_norms_query_owned         = std::optional<raft::device_vector<DistData_t, size_t>>();
  DistData_t* l2_norms_document_ptr = l2_norms_.data_handle();
  DistData_t* l2_norms_query_ptr    = l2_norms_.data_handle();
  if (build_config_.metric == cuvs::distance::DistanceType::CosineExpanded) {
    raft::linalg::map_offset(res, l2_norms_.view(), bbq::bbq_row_norm_op{quantizer_document});
    if (!self_join) {
      l2_norms_query_owned =
        std::make_optional(raft::make_device_vector<DistData_t, size_t>(res, nrow_));
      l2_norms_query_ptr = l2_norms_query_owned.value().data_handle();
      raft::linalg::map_offset(
        res, l2_norms_query_owned.value().view(), bbq::bbq_row_norm_op{quantizer_query});
    }
  }

  // One launch site for both kernels: they take identical arguments, and the query's layout picks
  // the path -- packed_4b is the only layout the int4 tensor-core kernel is dispatched for.
  auto launch = [&](auto document_layout, auto query_layout, auto self_join_tag) {
    constexpr auto D = decltype(document_layout)::value;
    constexpr auto Q = decltype(query_layout)::value;
    constexpr bool S = decltype(self_join_tag)::value;
#define CUVS_BBQ_LOCAL_JOIN_ARGS                                                                 \
  graph_.h_graph_new.data_handle(), h_rev_graph_new_.data_handle(),                              \
    d_list_sizes_new_.data_handle(), h_graph_old_.data_handle(), h_rev_graph_old_.data_handle(), \
    d_list_sizes_old_.data_handle(), NUM_SAMPLES, quantizer_document, quantizer_query,           \
    graph_buffer_.data_handle(), dists_buffer_.data_handle(), DEGREE_ON_DEVICE,                  \
    d_locks_.data_handle(), l2_norms_document_ptr, l2_norms_query_ptr, build_config_.metric,     \
    dist_epilogue
    if constexpr (Q == L::packed_4b) {
      local_join_kernel_bbq_wmma<D, Q, S>
        <<<nrow_, BLOCK_SIZE, 0, stream>>>(CUVS_BBQ_LOCAL_JOIN_ARGS);
    } else {
      local_join_kernel_bbq_simt<D, Q, S>
        <<<nrow_, BLOCK_SIZE, 0, stream>>>(CUVS_BBQ_LOCAL_JOIN_ARGS);
    }
#undef CUVS_BBQ_LOCAL_JOIN_ARGS
  };
  const L d = quantizer_document.layout;
  const L q = quantizer_query.layout;
  if (self_join) {
    switch (d) {
      case L::packed_1b:
        launch(std::integral_constant<L, L::packed_1b>{},
               std::integral_constant<L, L::packed_1b>{},
               std::true_type{});
        break;
      case L::transposed_2b:
        launch(std::integral_constant<L, L::transposed_2b>{},
               std::integral_constant<L, L::transposed_2b>{},
               std::true_type{});
        break;
      case L::packed_4b:
        launch(std::integral_constant<L, L::packed_4b>{},
               std::integral_constant<L, L::packed_4b>{},
               std::true_type{});
        break;
      case L::packed_7b:
        launch(std::integral_constant<L, L::packed_7b>{},
               std::integral_constant<L, L::packed_7b>{},
               std::true_type{});
        break;
      case L::packed_8b:
        launch(std::integral_constant<L, L::packed_8b>{},
               std::integral_constant<L, L::packed_8b>{},
               std::true_type{});
        break;
      default: RAFT_FAIL("Unsupported BBQ layout for symmetric local join on this branch.");
    }
  } else if (d == L::packed_1b && q == L::packed_4b) {
    launch(std::integral_constant<L, L::packed_1b>{},
           std::integral_constant<L, L::packed_4b>{},
           std::false_type{});
  } else if (d == L::packed_2b && q == L::packed_4b) {
    launch(std::integral_constant<L, L::packed_2b>{},
           std::integral_constant<L, L::packed_4b>{},
           std::false_type{});
  } else if (d == L::packed_1b && q == L::transposed_2b) {
    launch(std::integral_constant<L, L::packed_1b>{},
           std::integral_constant<L, L::transposed_2b>{},
           std::false_type{});
  } else if (d == L::packed_1b && q == L::transposed_4b) {
    launch(std::integral_constant<L, L::packed_1b>{},
           std::integral_constant<L, L::transposed_4b>{},
           std::false_type{});
  } else if (d == L::transposed_2b && q == L::transposed_4b) {
    launch(std::integral_constant<L, L::transposed_2b>{},
           std::integral_constant<L, L::transposed_4b>{},
           std::false_type{});
  } else {
    RAFT_FAIL("Unsupported BBQ layout pair for asymmetric local join.");
  }
}

template <typename Data_t, typename Index_t>
template <typename DistEpilogue_t>
void GNND<Data_t, Index_t>::build(Data_t* data,
                                  const Index_t nrow,
                                  Index_t* output_graph,
                                  bool return_distances,
                                  DistData_t* output_distances,
                                  DistEpilogue_t dist_epilogue)
{
  using input_t = typename std::remove_const<Data_t>::type;

  if (build_config_.metric == distance::DistanceType::BitwiseHamming &&
      !(std::is_same_v<input_t, uint8_t> || std::is_same_v<input_t, int8_t>)) {
    RAFT_FAIL(
      "Data type needs to be int8 or uint8 for NN Descent to run with BitwiseHamming distance.");
  }

  cudaStream_t stream = raft::resource::get_cuda_stream(res);
  nrow_               = nrow;
  graph_.nrow         = nrow;
  graph_.bloom_filter.set_nrow(nrow);
  update_counter_ = 0;
  graph_.h_graph  = (InternalID_t<Index_t>*)output_graph;

  d_data_ptr_ = nullptr;

  cudaPointerAttributes data_ptr_attr;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&data_ptr_attr, data));
  bool data_on_device = (data_ptr_attr.type == cudaMemoryTypeDevice);

  bool needs_l2_norms = build_config_.metric == cuvs::distance::DistanceType::L2Expanded ||
                        build_config_.metric == cuvs::distance::DistanceType::L2SqrtExpanded ||
                        build_config_.metric == cuvs::distance::DistanceType::CosineExpanded;

  // For fp32 host input, downcast to a device-side fp16 buffer when distance computation will be
  // done in fp16 anyway: dispatch matches the SIMT/WMMA decision in local_join() (FP16 explicit, or
  // AUTO with dim > 16).
  using DCT = cuvs::neighbors::nn_descent::DIST_COMP_DTYPE;
  bool fp32_input_uses_fp16_dist =
    std::is_same_v<input_t, float> &&
    (build_config_.dist_comp_dtype == DCT::FP16 ||
     (build_config_.dist_comp_dtype == DCT::AUTO && build_config_.dataset_dim > 16));
  bool downcast_host_data = !data_on_device && fp32_input_uses_fp16_dist;

  if (data_on_device) {
    // When user-given data is on device, we use it directly. This can be any type (fp32, fp16,
    // int8, uint8)
    d_data_ptr_ = data;
  } else if (downcast_host_data) {
    // When user-given data is fp32 host data and distances will be computed in fp16, we allocate
    // an fp16 device buffer and downcast at copy-in time. Storing the dataset on device in fp16
    // (instead of fp32) for this path halves both the device memory footprint and the per-
    // iteration read bandwidth of the WMMA kernel.
    if (!d_data_half_.has_value()) {
      d_data_half_.emplace(raft::make_device_matrix<half, size_t, raft::row_major>(
        res, build_config_.max_dataset_size, build_config_.dataset_dim));
    }
    size_t batch_size = 100000;
    auto vec_batches  = cuvs::spatial::knn::detail::utils::make_batch_load_iterator<Data_t>(
      res,
      data,
      static_cast<int64_t>(nrow_),
      static_cast<int64_t>(build_config_.dataset_dim),
      batch_size,
      stream);
    constexpr int TPB = 256;
    for (auto const& batch : vec_batches) {
      size_t n_elems    = batch.size() * build_config_.dataset_dim;
      int num_blocks    = raft::ceildiv(n_elems, static_cast<size_t>(TPB));
      size_t dst_offset = batch.offset() * build_config_.dataset_dim;
      if (needs_l2_norms) {
        // Compute l2 norms on the fp32 batches before they're downcast to fp16.
        compute_l2_norms_kernel<<<batch.size(),
                                  raft::warp_size(),
                                  sizeof(float) *
                                    raft::ceildiv(build_config_.dataset_dim,
                                                  static_cast<size_t>(raft::warp_size())) *
                                    raft::warp_size(),
                                  stream>>>(
          batch.data(), build_config_.dataset_dim, l2_norms_.data_handle() + batch.offset());
        RAFT_CUDA_TRY(cudaPeekAtLastError());
      }
      convert_copy_kernel<<<num_blocks, TPB, 0, stream>>>(
        batch.data(), d_data_half_.value().data_handle() + dst_offset, n_elems);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }
    d_data_ptr_ = d_data_half_.value().data_handle();
  } else {
    // Other cases: user-given data is not device-accessible, but we don't need a precision
    // conversion. Allocate a device buffer in input_t and copy as-is.
    if (!d_data_direct_.has_value()) {
      d_data_direct_.emplace(raft::make_device_matrix<input_t, size_t, raft::row_major>(
        res, build_config_.max_dataset_size, build_config_.dataset_dim));
    }
    raft::copy(d_data_direct_.value().data_handle(),
               data,
               static_cast<size_t>(nrow_) * build_config_.dataset_dim,
               stream);
    d_data_ptr_ = d_data_direct_.value().data_handle();
  }

  if (needs_l2_norms && !downcast_host_data) {
    compute_l2_norms_kernel<<<nrow_,
                              raft::warp_size(),
                              sizeof(input_t) *
                                raft::ceildiv(build_config_.dataset_dim,
                                              static_cast<size_t>(raft::warp_size())) *
                                raft::warp_size(),
                              stream>>>(
      static_cast<const input_t*>(d_data_ptr_), build_config_.dataset_dim, l2_norms_.data_handle());
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    raft::resource::sync_stream(res);
  }

  graph_.clear();
  graph_.init_random_graph();
  graph_.sample_graph(true);

  auto update_and_sample = [&](bool update_graph) {
    if (update_graph) {
      update_counter_ = 0;
      graph_.update_graph(graph_host_buffer_.data_handle(),
                          dists_host_buffer_.data_handle(),
                          DEGREE_ON_DEVICE,
                          update_counter_);
      if (update_counter_ < build_config_.termination_threshold * nrow_ *
                              build_config_.dataset_dim / counter_interval) {
        update_counter_ = -1;
      }
    }
    graph_.sample_graph(false);
  };

  for (size_t it = 0; it < build_config_.max_iterations; it++) {
    raft::copy(res, d_list_sizes_new_.view(), graph_.h_list_sizes_new.view());
    raft::copy(res, h_graph_old_.view(), graph_.h_graph_old.view());
    raft::copy(res, d_list_sizes_old_.view(), graph_.h_list_sizes_old.view());
    raft::resource::sync_stream(res);

    std::thread update_and_sample_thread(update_and_sample, it);

    RAFT_LOG_DEBUG("# GNND iteration: %lu / %lu", it + 1, build_config_.max_iterations);

    // Reuse dists_buffer_ to save GPU memory. graph_buffer_ cannot be reused, because it
    // contains some information for local_join.
    static_assert(DEGREE_ON_DEVICE * sizeof(*(dists_buffer_.data_handle())) >=
                  NUM_SAMPLES * sizeof(*(graph_buffer_.data_handle())));
    add_reverse_edges(graph_.h_graph_new.data_handle(),
                      h_rev_graph_new_.data_handle(),
                      (Index_t*)dists_buffer_.data_handle(),
                      d_list_sizes_new_.data_handle(),
                      stream);
    add_reverse_edges(h_graph_old_.data_handle(),
                      h_rev_graph_old_.data_handle(),
                      (Index_t*)dists_buffer_.data_handle(),
                      d_list_sizes_old_.data_handle(),
                      stream);

    // Tensor operations from `mma.h` are guarded with archicteture
    // __CUDA_ARCH__ >= 700. Since RAFT supports compilation for ARCH 600,
    // we need to ensure that `local_join_kernel` (which uses tensor) operations
    // is not only not compiled, but also a runtime error is presented to the user
    auto kernel       = compute_l2_norms_kernel<input_t>;
    void* kernel_ptr  = reinterpret_cast<void*>(kernel);
    auto runtime_arch = raft::util::arch::kernel_virtual_arch(kernel_ptr);
    auto wmma_range =
      raft::util::arch::SM_range(raft::util::arch::SM_70(), raft::util::arch::SM_future());

    if (wmma_range.contains(runtime_arch)) {
      local_join(stream, dist_epilogue);
    } else {
      THROW("NN_DESCENT cannot be run for __CUDA_ARCH__ < 700");
    }

    update_and_sample_thread.join();

    if (update_counter_ == -1) { break; }
    raft::copy(res, graph_host_buffer_.view(), graph_buffer_.view());
    raft::copy(res, dists_host_buffer_.view(), dists_buffer_.view());
    raft::resource::sync_stream(res);

    graph_.sample_graph_new(graph_host_buffer_.data_handle(), DEGREE_ON_DEVICE);
  }

  graph_.update_graph(graph_host_buffer_.data_handle(),
                      dists_host_buffer_.data_handle(),
                      DEGREE_ON_DEVICE,
                      update_counter_);
  raft::resource::sync_stream(res);
  graph_.sort_lists();

  // Reuse graph_.h_dists as the buffer for shrink the lists in graph
  static_assert(sizeof(decltype(*(graph_.h_dists.data_handle()))) >= sizeof(Index_t));

  if (return_distances) {
    auto graph_h_dists = raft::make_host_matrix<DistData_t, int64_t, raft::row_major>(
      nrow_, build_config_.output_graph_degree);

// slice on host
#pragma omp parallel for
    for (size_t i = 0; i < (size_t)nrow_; i++) {
      for (size_t j = 0; j < build_config_.output_graph_degree; j++) {
        graph_h_dists(i, j) = graph_.h_dists(i, j);
      }
    }
    raft::copy(
      res,
      raft::make_device_vector_view(output_distances, nrow_ * build_config_.output_graph_degree),
      raft::make_host_vector_view(graph_h_dists.data_handle(),
                                  nrow_ * build_config_.output_graph_degree));

    auto output_dist_view = raft::make_device_matrix_view<DistData_t, int64_t, raft::row_major>(
      output_distances, nrow_, build_config_.output_graph_degree);
    // distance post-processing
    bool can_postprocess_dist = std::is_same_v<DistEpilogue_t, raft::identity_op>;
    if (build_config_.metric == cuvs::distance::DistanceType::L2SqrtExpanded &&
        can_postprocess_dist) {
      raft::linalg::map(
        res, output_dist_view, raft::sqrt_op{}, raft::make_const_mdspan(output_dist_view));
    } else if (!cuvs::distance::is_min_close(build_config_.metric) && can_postprocess_dist) {
      // revert negated innerproduct
      raft::linalg::map(res,
                        output_dist_view,
                        raft::mul_const_op<DistData_t>(-1),
                        raft::make_const_mdspan(output_dist_view));
    }
    raft::resource::sync_stream(res);
  }

  Index_t* graph_shrink_buffer = (Index_t*)graph_.h_dists.data_handle();

  // Copy the output graph while removing duplicates.
#pragma omp parallel for
  for (size_t i = 0; i < (size_t)nrow_; i++) {
    auto output_neighbor_list_ptr = graph_shrink_buffer + i * build_config_.node_degree;

    size_t out_j = 0;

    // Copy neighbor list while removing duplicates.
    for (size_t in_j = 0; in_j < build_config_.node_degree; in_j++) {
      size_t idx = graph_.h_graph[i * graph_.node_degree + in_j].id();

      bool dup = false;
      for (size_t exi_j = 0; exi_j < out_j; exi_j++) {
        if (static_cast<decltype(idx)>(output_neighbor_list_ptr[exi_j]) == idx || i == idx) {
          dup = true;
          break;
        }
      }
      if (!dup) {
        output_neighbor_list_ptr[out_j] = idx;
        out_j++;
      }
    }

    // Fill with random nodes if the length of the filled neighbor list is less than the degree.
    for (size_t j = out_j; j < build_config_.node_degree; j++) {
      uint64_t rnd = static_cast<uint64_t>(i * build_config_.node_degree + j + 1);
      uint64_t idx;
      bool dup = true;
      for (size_t attempts = 0; dup && attempts < build_config_.node_degree; attempts++) {
        rnd = cuvs::neighbors::detail::device::xorshift64(rnd);
        idx = rnd % nrow_;
        dup = false;
        for (size_t exi_j = 0; exi_j < j; exi_j++) {
          if (static_cast<decltype(idx)>(output_neighbor_list_ptr[exi_j]) == idx || i == idx) {
            dup = true;
            break;
          }
        }
      }
      output_neighbor_list_ptr[j] = static_cast<int>(idx);
    }
  }
  graph_.h_graph = nullptr;

#pragma omp parallel for
  for (size_t i = 0; i < (size_t)nrow_; i++) {
    for (size_t j = 0; j < build_config_.node_degree; j++) {
      output_graph[i * build_config_.node_degree + j] =
        graph_shrink_buffer[i * build_config_.node_degree + j];
    }
  }
}

template <typename Data_t, typename Index_t>
template <typename DistEpilogue_t>
void GNND<Data_t, Index_t>::build(cuvs::neighbors::device_bbq_dataset_view<int64_t> dataset,
                                  Index_t* output_graph,
                                  bool return_distances,
                                  DistData_t* output_distances,
                                  DistEpilogue_t dist_epilogue)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(res);
  nrow_               = static_cast<size_t>(dataset.n_rows());
  graph_.nrow         = nrow_;
  graph_.bloom_filter.set_nrow(nrow_);
  update_counter_ = 0;
  graph_.h_graph  = reinterpret_cast<InternalID_t<Index_t>*>(output_graph);

  graph_.clear();
  graph_.init_random_graph();
  graph_.sample_graph(true);

  auto update_and_sample = [&](bool update_graph) {
    if (update_graph) {
      update_counter_ = 0;
      graph_.update_graph(graph_host_buffer_.data_handle(),
                          dists_host_buffer_.data_handle(),
                          DEGREE_ON_DEVICE,
                          update_counter_);
      if (update_counter_ < build_config_.termination_threshold * nrow_ *
                              build_config_.dataset_dim / counter_interval) {
        update_counter_ = -1;
      }
    }
    graph_.sample_graph(false);
  };

  for (size_t it = 0; it < build_config_.max_iterations; ++it) {
    raft::copy(res, d_list_sizes_new_.view(), graph_.h_list_sizes_new.view());
    raft::copy(res, h_graph_old_.view(), graph_.h_graph_old.view());
    raft::copy(res, d_list_sizes_old_.view(), graph_.h_list_sizes_old.view());
    raft::resource::sync_stream(res);

    std::thread update_and_sample_thread(update_and_sample, it);
    RAFT_LOG_DEBUG("# GNND iteration: %lu / %lu", it + 1, build_config_.max_iterations);

    static_assert(DEGREE_ON_DEVICE * sizeof(*(dists_buffer_.data_handle())) >=
                  NUM_SAMPLES * sizeof(*(graph_buffer_.data_handle())));
    add_reverse_edges(graph_.h_graph_new.data_handle(),
                      h_rev_graph_new_.data_handle(),
                      reinterpret_cast<Index_t*>(dists_buffer_.data_handle()),
                      d_list_sizes_new_.data_handle(),
                      stream);
    add_reverse_edges(h_graph_old_.data_handle(),
                      h_rev_graph_old_.data_handle(),
                      reinterpret_cast<Index_t*>(dists_buffer_.data_handle()),
                      d_list_sizes_old_.data_handle(),
                      stream);

    local_join(stream, dataset, dist_epilogue);
    update_and_sample_thread.join();
    if (update_counter_ == -1) { break; }
    raft::copy(res, graph_host_buffer_.view(), graph_buffer_.view());
    raft::copy(res, dists_host_buffer_.view(), dists_buffer_.view());
    raft::resource::sync_stream(res);
    graph_.sample_graph_new(graph_host_buffer_.data_handle(), DEGREE_ON_DEVICE);
  }

  graph_.update_graph(graph_host_buffer_.data_handle(),
                      dists_host_buffer_.data_handle(),
                      DEGREE_ON_DEVICE,
                      update_counter_);
  raft::resource::sync_stream(res);
  graph_.sort_lists();

  static_assert(sizeof(decltype(*(graph_.h_dists.data_handle()))) >= sizeof(Index_t));
  if (return_distances) {
    auto graph_h_dists = raft::make_host_matrix<DistData_t, int64_t, raft::row_major>(
      nrow_, build_config_.output_graph_degree);
#pragma omp parallel for
    for (size_t i = 0; i < nrow_; ++i) {
      for (size_t j = 0; j < build_config_.output_graph_degree; ++j) {
        graph_h_dists(i, j) = graph_.h_dists(i, j);
      }
    }
    raft::copy(
      res,
      raft::make_device_vector_view(output_distances, nrow_ * build_config_.output_graph_degree),
      raft::make_host_vector_view(graph_h_dists.data_handle(),
                                  nrow_ * build_config_.output_graph_degree));

    auto output_dist_view = raft::make_device_matrix_view<DistData_t, int64_t, raft::row_major>(
      output_distances, nrow_, build_config_.output_graph_degree);
    const bool can_postprocess_dist = std::is_same_v<DistEpilogue_t, raft::identity_op>;
    if (build_config_.metric == cuvs::distance::DistanceType::L2SqrtExpanded &&
        can_postprocess_dist) {
      raft::linalg::map(
        res, output_dist_view, raft::sqrt_op{}, raft::make_const_mdspan(output_dist_view));
    } else if (!cuvs::distance::is_min_close(build_config_.metric) && can_postprocess_dist) {
      raft::linalg::map(res,
                        output_dist_view,
                        raft::mul_const_op<DistData_t>(-1),
                        raft::make_const_mdspan(output_dist_view));
    }
    raft::resource::sync_stream(res);
  }

  auto* graph_shrink_buffer = reinterpret_cast<Index_t*>(graph_.h_dists.data_handle());
#pragma omp parallel for
  for (size_t i = 0; i < nrow_; ++i) {
    for (size_t j = 0; j < build_config_.node_degree; ++j) {
      const size_t index = i * graph_.node_degree + j;
      const int id       = graph_.h_graph[index].id();
      graph_shrink_buffer[i * build_config_.node_degree + j] =
        id < static_cast<int>(nrow_) ? id
                                     : cuvs::neighbors::detail::device::xorshift64(index) % nrow_;
    }
  }
  graph_.h_graph = nullptr;

#pragma omp parallel for
  for (size_t i = 0; i < nrow_; ++i) {
    for (size_t j = 0; j < build_config_.node_degree; ++j) {
      output_graph[i * build_config_.node_degree + j] =
        graph_shrink_buffer[i * build_config_.node_degree + j];
    }
  }
}

template <typename IdxT = uint32_t>
void build(raft::resources const& res,
           const index_params& params,
           cuvs::neighbors::device_bbq_dataset_view<int64_t> dataset,
           index<IdxT>& idx)
{
  RAFT_EXPECTS(dataset.quantizers.size() > 0, "BBQ dataset must not be empty.");
  auto front_quantizer = dataset.quantizers[0];
  cuvs::common::nvtx::range<cuvs::common::nvtx::domain::cuvs> fun_scope(
    "neighbors::nn_descent::detail::build-bbq(%zu, %zu, %zu, %zu, %zu)",
    size_t(dataset.n_rows()),
    size_t(dataset.dim()),
    size_t(idx.graph().extent(1)),
    size_t(idx.metric()),
    size_t(front_quantizer.bits));
  RAFT_EXPECTS(idx.metric() == cuvs::distance::DistanceType::L2Expanded ||
                 idx.metric() == cuvs::distance::DistanceType::L2SqrtExpanded ||
                 idx.metric() == cuvs::distance::DistanceType::CosineExpanded ||
                 idx.metric() == cuvs::distance::DistanceType::InnerProduct,
               "BBQ NN-Descent supports L2Expanded, L2SqrtExpanded, CosineExpanded, and "
               "InnerProduct.");
  RAFT_EXPECTS(idx.metric() == front_quantizer.metric,
               "BBQ dataset metric does not match the NN-Descent metric.");

  size_t extended_graph_degree;
  size_t graph_degree;
  auto build_config = get_build_config(res,
                                       params,
                                       dataset.n_rows(),
                                       dataset.dim(),
                                       idx.metric(),
                                       extended_graph_degree,
                                       graph_degree);
  auto int_graph =
    raft::make_host_matrix<int, int64_t, raft::row_major>(dataset.n_rows(), extended_graph_degree);
  GNND<const uint8_t, int> nnd(res, build_config);

  if (idx.distances().has_value() || !params.return_distances) {
    nnd.build(dataset,
              int_graph.data_handle(),
              params.return_distances,
              idx.distances()
                .value_or(raft::make_device_matrix<float, int64_t>(res, 0, 0).view())
                .data_handle());
  } else {
    RAFT_FAIL(
      "Distance view not allocated. Using return_distances set to true requires "
      "distance view to be allocated.");
  }

#pragma omp parallel for
  for (size_t i = 0; i < static_cast<size_t>(dataset.n_rows()); ++i) {
    for (size_t j = 0; j < graph_degree; ++j) {
      idx.graph()(i, j) = int_graph(i, j);
    }
  }
}

template <typename T,
          typename IdxT = uint32_t,
          typename Accessor =
            raft::host_device_accessor<cuda::std::default_accessor<T>, raft::memory_type::host>>
void build(raft::resources const& res,
           const index_params& params,
           raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset,
           index<IdxT>& idx)
{
  size_t extended_graph_degree, graph_degree;
  auto build_config = get_build_config(res,
                                       params,
                                       static_cast<size_t>(dataset.extent(0)),
                                       static_cast<size_t>(dataset.extent(1)),
                                       idx.metric(),
                                       extended_graph_degree,
                                       graph_degree);

  auto int_graph =
    raft::make_host_matrix<int, int64_t, raft::row_major>(dataset.extent(0), extended_graph_degree);

  // When the graph will be a complete graph, output it without NND process for better performance.
  if (static_cast<size_t>(dataset.extent(0) - 1) == graph_degree && (!params.return_distances)) {
    auto graph = idx.graph().data_handle();
#pragma omp parallel for
    for (size_t i = 0; i < static_cast<size_t>(dataset.extent(0)); i++) {
      for (size_t j = 0; j < graph_degree; j++) {
        graph[i * graph_degree + j] = (i + j + 1) % dataset.extent(0);
      }
    }
    return;
  }

  GNND<const T, int> nnd(res, build_config);

  if (idx.distances().has_value() || !params.return_distances) {
    nnd.build(dataset.data_handle(),
              dataset.extent(0),
              int_graph.data_handle(),
              params.return_distances,
              idx.distances()
                .value_or(raft::make_device_matrix<float, int64_t>(res, 0, 0).view())
                .data_handle());
  } else {
    RAFT_EXPECTS(!params.return_distances,
                 "Distance view not allocated. Using return_distances set to true requires "
                 "distance view to be allocated.");
  }

#pragma omp parallel for
  for (size_t i = 0; i < static_cast<size_t>(dataset.extent(0)); i++) {
    for (size_t j = 0; j < graph_degree; j++) {
      auto graph                  = idx.graph().data_handle();
      graph[i * graph_degree + j] = int_graph.data_handle()[i * extended_graph_degree + j];
    }
  }
}

template <typename IdxT = uint32_t>
index<IdxT> build(raft::resources const& res,
                  const index_params& params,
                  cuvs::neighbors::device_bbq_dataset_view<int64_t> dataset)
{
  size_t graph_degree = params.graph_degree;
  if (params.intermediate_graph_degree < graph_degree) {
    RAFT_LOG_WARN(
      "Graph degree (%lu) cannot be larger than intermediate graph degree (%lu), reducing "
      "graph_degree.",
      graph_degree,
      params.intermediate_graph_degree);
    graph_degree = params.intermediate_graph_degree;
  }

  index<IdxT> idx{res,
                  static_cast<int64_t>(dataset.n_rows()),
                  static_cast<int64_t>(graph_degree),
                  params.return_distances,
                  params.metric};
  build(res, params, dataset, idx);
  return idx;
}

template <typename T,
          typename IdxT = uint32_t,
          typename Accessor =
            raft::host_device_accessor<cuda::std::default_accessor<T>, raft::memory_type::host>>
index<IdxT> build(
  raft::resources const& res,
  const index_params& params,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset)
{
  size_t intermediate_degree = params.intermediate_graph_degree;
  size_t graph_degree        = params.graph_degree;

  if (intermediate_degree < graph_degree) {
    RAFT_LOG_WARN(
      "Graph degree (%lu) cannot be larger than intermediate graph degree (%lu), reducing "
      "graph_degree.",
      graph_degree,
      intermediate_degree);
    graph_degree = intermediate_degree;
  }

  index<IdxT> idx{res,
                  dataset.extent(0),
                  static_cast<int64_t>(graph_degree),
                  params.return_distances,
                  params.metric};

  build(res, params, dataset, idx);

  return idx;
}

}  // namespace cuvs::neighbors::nn_descent::detail
