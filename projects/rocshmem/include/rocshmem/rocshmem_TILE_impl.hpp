/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

/**
 * @file rocshmem_TILE_impl.hpp
 * @brief Template implementations for rocSHMEM tile API functions.
 *
 * This header contains the inline template implementations for the tile API.
 * It is included at the end of rocshmem_TILE.hpp to allow template instantiation
 * for any conforming tensor types.
 */

#ifndef LIBRARY_INCLUDE_ROCSHMEM_TILE_IMPL_HPP
#define LIBRARY_INCLUDE_ROCSHMEM_TILE_IMPL_HPP

// This file should only be included from rocshmem_TILE.hpp
#ifndef LIBRARY_INCLUDE_ROCSHMEM_TILE_HPP
#error "rocshmem_TILE_impl.hpp should only be included from rocshmem_TILE.hpp"
#endif

// Only provide actual implementation when compiling device code
#ifdef __HIP_DEVICE_COMPILE__

// Forward declarations of type-erased namespace functions
// These are implemented in src/rocshmem_tile_gpu.cpp and provide the actual
// implementations. The extern "C" wrappers for JIT consumers are in
// src/device/rocshmem_wrapper.cc
namespace rocshmem {
  // RMA PUT operations
  __device__ int rocshmem_ctx_tile_put_internal(
      rocshmem_ctx_t ctx, void* dst_data, const void* src_data,
      const size_t* dst_strides, const size_t* src_strides,
      const size_t* start_coord, const size_t* boundary,
      int ndim, size_t element_size, int pe, uint64_t flags);

  __device__ int rocshmem_ctx_tile_put_wave_internal(
      rocshmem_ctx_t ctx, void* dst_data, const void* src_data,
      const size_t* dst_strides, const size_t* src_strides,
      const size_t* start_coord, const size_t* boundary,
      int ndim, size_t element_size, int pe, uint64_t flags);

  __device__ int rocshmem_ctx_tile_put_wg_internal(
      rocshmem_ctx_t ctx, void* dst_data, const void* src_data,
      const size_t* dst_strides, const size_t* src_strides,
      const size_t* start_coord, const size_t* boundary,
      int ndim, size_t element_size, int pe, uint64_t flags);

  // RMA GET operations
  __device__ int rocshmem_ctx_tile_get_internal(
      rocshmem_ctx_t ctx, void* dst_data, const void* src_data,
      const size_t* dst_strides, const size_t* src_strides,
      const size_t* start_coord, const size_t* boundary,
      int ndim, size_t element_size, int pe, uint64_t flags);

  __device__ int rocshmem_ctx_tile_get_wave_internal(
      rocshmem_ctx_t ctx, void* dst_data, const void* src_data,
      const size_t* dst_strides, const size_t* src_strides,
      const size_t* start_coord, const size_t* boundary,
      int ndim, size_t element_size, int pe, uint64_t flags);

  __device__ int rocshmem_ctx_tile_get_wg_internal(
      rocshmem_ctx_t ctx, void* dst_data, const void* src_data,
      const size_t* dst_strides, const size_t* src_strides,
      const size_t* start_coord, const size_t* boundary,
      int ndim, size_t element_size, int pe, uint64_t flags);

  // Collective - Allgather
  __device__ int rocshmem_tile_allgather_internal(
      rocshmem_team_t team, void* dst_data, const void* src_data,
      const size_t* dst_strides, const size_t* src_strides,
      const size_t* start_coord, const size_t* boundary,
      int ndim, size_t element_size, uint64_t flags);

  __device__ int rocshmem_tile_allgather_wave_internal(
      rocshmem_team_t team, void* dst_data, const void* src_data,
      const size_t* dst_strides, const size_t* src_strides,
      const size_t* start_coord, const size_t* boundary,
      int ndim, size_t element_size, uint64_t flags);

  __device__ int rocshmem_tile_allgather_wg_internal(
      rocshmem_team_t team, void* dst_data, const void* src_data,
      const size_t* dst_strides, const size_t* src_strides,
      const size_t* start_coord, const size_t* boundary,
      int ndim, size_t element_size, uint64_t flags);

  // Collective - Broadcast
  __device__ int rocshmem_tile_broadcast_internal(
      rocshmem_team_t team, void* dst_data, const void* src_data,
      const size_t* dst_strides, const size_t* src_strides,
      const size_t* start_coord, const size_t* boundary,
      int ndim, size_t element_size, int pe_root, uint64_t flags);

  __device__ int rocshmem_tile_broadcast_wave_internal(
      rocshmem_team_t team, void* dst_data, const void* src_data,
      const size_t* dst_strides, const size_t* src_strides,
      const size_t* start_coord, const size_t* boundary,
      int ndim, size_t element_size, int pe_root, uint64_t flags);

  __device__ int rocshmem_tile_broadcast_wg_internal(
      rocshmem_team_t team, void* dst_data, const void* src_data,
      const size_t* dst_strides, const size_t* src_strides,
      const size_t* start_coord, const size_t* boundary,
      int ndim, size_t element_size, int pe_root, uint64_t flags);

  // Collective - SUM Reduce
  __device__ int rocshmem_tile_sum_reduce_internal(
      rocshmem_team_t team, void* dst_data, const void* src_data,
      const size_t* dst_strides, const size_t* src_strides,
      const size_t* start_coord, const size_t* boundary,
      int ndim, size_t element_size, int root, uint64_t flags);

  __device__ int rocshmem_tile_sum_reduce_wave_internal(
      rocshmem_team_t team, void* dst_data, const void* src_data,
      const size_t* dst_strides, const size_t* src_strides,
      const size_t* start_coord, const size_t* boundary,
      int ndim, size_t element_size, int root, uint64_t flags);

  __device__ int rocshmem_tile_sum_reduce_wg_internal(
      rocshmem_team_t team, void* dst_data, const void* src_data,
      const size_t* dst_strides, const size_t* src_strides,
      const size_t* start_coord, const size_t* boundary,
      int ndim, size_t element_size, int root, uint64_t flags);

  // Collective - MAX Reduce
  __device__ int rocshmem_tile_max_reduce_internal(
      rocshmem_team_t team, void* dst_data, const void* src_data,
      const size_t* dst_strides, const size_t* src_strides,
      const size_t* start_coord, const size_t* boundary,
      int ndim, size_t element_size, int root, uint64_t flags);

  __device__ int rocshmem_tile_max_reduce_wave_internal(
      rocshmem_team_t team, void* dst_data, const void* src_data,
      const size_t* dst_strides, const size_t* src_strides,
      const size_t* start_coord, const size_t* boundary,
      int ndim, size_t element_size, int root, uint64_t flags);

  __device__ int rocshmem_tile_max_reduce_wg_internal(
      rocshmem_team_t team, void* dst_data, const void* src_data,
      const size_t* dst_strides, const size_t* src_strides,
      const size_t* start_coord, const size_t* boundary,
      int ndim, size_t element_size, int root, uint64_t flags);

  // Collective - MIN Reduce
  __device__ int rocshmem_tile_min_reduce_internal(
      rocshmem_team_t team, void* dst_data, const void* src_data,
      const size_t* dst_strides, const size_t* src_strides,
      const size_t* start_coord, const size_t* boundary,
      int ndim, size_t element_size, int root, uint64_t flags);

  __device__ int rocshmem_tile_min_reduce_wave_internal(
      rocshmem_team_t team, void* dst_data, const void* src_data,
      const size_t* dst_strides, const size_t* src_strides,
      const size_t* start_coord, const size_t* boundary,
      int ndim, size_t element_size, int root, uint64_t flags);

  __device__ int rocshmem_tile_min_reduce_wg_internal(
      rocshmem_team_t team, void* dst_data, const void* src_data,
      const size_t* dst_strides, const size_t* src_strides,
      const size_t* start_coord, const size_t* boundary,
      int ndim, size_t element_size, int root, uint64_t flags);

  // Collective wait
  __device__ int rocshmem_tile_collective_wait_internal(
      rocshmem_team_t team, uint64_t flags);

/******************************************************************************
 **************** RMA OPERATIONS - CONTEXT VERSIONS (5) ***********************
 *****************************************************************************/

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_put(rocshmem_ctx_t ctx, dst_tensor_t dst,
                                     const src_tensor_t src, tuple_t start_coord,
                                     tuple_t boundary, int pe, uint64_t flags) {
  using element_t = typename src_tensor_t::element_type;
  constexpr int ndim = src_tensor_t::ndim;

  // Extract tensor/tuple properties into arrays
  size_t dst_strides[ndim];
  size_t src_strides[ndim];
  size_t start_arr[ndim];
  size_t boundary_arr[ndim];

  #pragma unroll
  for (int i = 0; i < ndim; i++) {
    dst_strides[i] = dst.stride(i);
    src_strides[i] = src.stride(i);
    start_arr[i] = start_coord.get(i);
    boundary_arr[i] = boundary.get(i);
  }

  // Forward to type-erased bitcode implementation
  return rocshmem_ctx_tile_put_internal(ctx, dst.data_handle(), src.data_handle(),
                                  dst_strides, src_strides,
                                  start_arr, boundary_arr,
                                  ndim, sizeof(element_t), pe, flags);
}

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_put_wave(rocshmem_ctx_t ctx, dst_tensor_t dst,
                                          const src_tensor_t src, tuple_t start_coord,
                                          tuple_t boundary, int pe,
                                          uint64_t flags) {
  using element_t = typename src_tensor_t::element_type;
  constexpr int ndim = src_tensor_t::ndim;

  size_t dst_strides[ndim];
  size_t src_strides[ndim];
  size_t start_arr[ndim];
  size_t boundary_arr[ndim];

  #pragma unroll
  for (int i = 0; i < ndim; i++) {
    dst_strides[i] = dst.stride(i);
    src_strides[i] = src.stride(i);
    start_arr[i] = start_coord.get(i);
    boundary_arr[i] = boundary.get(i);
  }

  // Forward to type-erased bitcode implementation
  return rocshmem_ctx_tile_put_wave_internal(ctx, dst.data_handle(), src.data_handle(),
                                       dst_strides, src_strides,
                                       start_arr, boundary_arr,
                                       ndim, sizeof(element_t), pe, flags);
}

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_put_wg(rocshmem_ctx_t ctx, dst_tensor_t dst,
                                        const src_tensor_t src, tuple_t start_coord,
                                        tuple_t boundary, int pe,
                                        uint64_t flags) {
  using element_t = typename src_tensor_t::element_type;
  constexpr int ndim = src_tensor_t::ndim;

  size_t dst_strides[ndim];
  size_t src_strides[ndim];
  size_t start_arr[ndim];
  size_t boundary_arr[ndim];

  #pragma unroll
  for (int i = 0; i < ndim; i++) {
    dst_strides[i] = dst.stride(i);
    src_strides[i] = src.stride(i);
    start_arr[i] = start_coord.get(i);
    boundary_arr[i] = boundary.get(i);
  }

  // Forward to type-erased bitcode implementation
  return rocshmem_ctx_tile_put_wg_internal(ctx, dst.data_handle(), src.data_handle(),
                                     dst_strides, src_strides,
                                     start_arr, boundary_arr,
                                     ndim, sizeof(element_t), pe, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_get(rocshmem_ctx_t ctx, dst_tensor_t dst,
                                     src_tensor_t src, tuple_t start_coord,
                                     tuple_t boundary, int pe, uint64_t flags) {
  using element_t = typename src_tensor_t::element_type;
  constexpr int ndim = src_tensor_t::ndim;

  size_t dst_strides[ndim];
  size_t src_strides[ndim];
  size_t start_arr[ndim];
  size_t boundary_arr[ndim];

  #pragma unroll
  for (int i = 0; i < ndim; i++) {
    dst_strides[i] = dst.stride(i);
    src_strides[i] = src.stride(i);
    start_arr[i] = start_coord.get(i);
    boundary_arr[i] = boundary.get(i);
  }

  // Forward to type-erased bitcode implementation
  return rocshmem_ctx_tile_get_internal(ctx, dst.data_handle(), src.data_handle(),
                                  dst_strides, src_strides,
                                  start_arr, boundary_arr,
                                  ndim, sizeof(element_t), pe, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_get_wave(rocshmem_ctx_t ctx, dst_tensor_t dst,
                                          src_tensor_t src, tuple_t start_coord,
                                          tuple_t boundary, int pe,
                                          uint64_t flags) {
  using element_t = typename src_tensor_t::element_type;
  constexpr int ndim = src_tensor_t::ndim;

  size_t dst_strides[ndim];
  size_t src_strides[ndim];
  size_t start_arr[ndim];
  size_t boundary_arr[ndim];

  #pragma unroll
  for (int i = 0; i < ndim; i++) {
    dst_strides[i] = dst.stride(i);
    src_strides[i] = src.stride(i);
    start_arr[i] = start_coord.get(i);
    boundary_arr[i] = boundary.get(i);
  }

  // Forward to type-erased bitcode implementation
  return rocshmem_ctx_tile_get_wave_internal(ctx, dst.data_handle(), src.data_handle(),
                                       dst_strides, src_strides,
                                       start_arr, boundary_arr,
                                       ndim, sizeof(element_t), pe, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_get_wg(rocshmem_ctx_t ctx, dst_tensor_t dst,
                                        src_tensor_t src, tuple_t start_coord,
                                        tuple_t boundary, int pe,
                                        uint64_t flags) {
  using element_t = typename src_tensor_t::element_type;
  constexpr int ndim = src_tensor_t::ndim;

  size_t dst_strides[ndim];
  size_t src_strides[ndim];
  size_t start_arr[ndim];
  size_t boundary_arr[ndim];

  #pragma unroll
  for (int i = 0; i < ndim; i++) {
    dst_strides[i] = dst.stride(i);
    src_strides[i] = src.stride(i);
    start_arr[i] = start_coord.get(i);
    boundary_arr[i] = boundary.get(i);
  }

  // Forward to type-erased bitcode implementation
  return rocshmem_ctx_tile_get_wg_internal(ctx, dst.data_handle(), src.data_handle(),
                                     dst_strides, src_strides,
                                     start_arr, boundary_arr,
                                     ndim, sizeof(element_t), pe, flags);
}

/******************************************************************************
 *************** RMA OPERATIONS - DEFAULT CONTEXT WRAPPERS (5) ****************
 *****************************************************************************/

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_put(dst_tensor_t dst, const src_tensor_t src,
                                 tuple_t start_coord, tuple_t boundary, int pe,
                                 uint64_t flags) {
  return rocshmem_ctx_tile_put(ROCSHMEM_CTX_DEFAULT, dst, src, start_coord,
                               boundary, pe, flags);
}

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_put_wave(dst_tensor_t dst, const src_tensor_t src,
                                      tuple_t start_coord, tuple_t boundary,
                                      int pe, uint64_t flags) {
  return rocshmem_ctx_tile_put_wave(ROCSHMEM_CTX_DEFAULT, dst, src,
                                    start_coord, boundary, pe, flags);
}

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_put_wg(dst_tensor_t dst, const src_tensor_t src,
                                    tuple_t start_coord, tuple_t boundary,
                                    int pe, uint64_t flags) {
  return rocshmem_ctx_tile_put_wg(ROCSHMEM_CTX_DEFAULT, dst, src, start_coord,
                                  boundary, pe, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_get(dst_tensor_t dst, src_tensor_t src,
                                 tuple_t start_coord, tuple_t boundary, int pe,
                                 uint64_t flags) {
  return rocshmem_ctx_tile_get(ROCSHMEM_CTX_DEFAULT, dst, src, start_coord,
                               boundary, pe, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_get_wave(dst_tensor_t dst, src_tensor_t src,
                                      tuple_t start_coord, tuple_t boundary,
                                      int pe, uint64_t flags) {
  return rocshmem_ctx_tile_get_wave(ROCSHMEM_CTX_DEFAULT, dst, src, start_coord,
                                    boundary, pe, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_get_wg(dst_tensor_t dst, src_tensor_t src,
                                    tuple_t start_coord, tuple_t boundary,
                                    int pe, uint64_t flags) {
  return rocshmem_ctx_tile_get_wg(ROCSHMEM_CTX_DEFAULT, dst, src, start_coord,
                                  boundary, pe, flags);
}

/******************************************************************************
 *********** COLLECTIVE ALLGATHER - CONTEXT VERSIONS (3) **********************
 *****************************************************************************/

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_allgather(rocshmem_ctx_t ctx, rocshmem_team_t team,
                                           dst_tensor_t dst, const src_tensor_t src,
                                           tuple_t start_coord, tuple_t boundary,
                                           uint64_t flags) {
  using element_t = typename src_tensor_t::element_type;
  constexpr int ndim = src_tensor_t::ndim;

  size_t dst_strides[ndim];
  size_t src_strides[ndim];
  size_t start_arr[ndim];
  size_t boundary_arr[ndim];

  #pragma unroll
  for (int i = 0; i < ndim; i++) {
    dst_strides[i] = dst.stride(i);
    src_strides[i] = src.stride(i);
    start_arr[i] = start_coord.get(i);
    boundary_arr[i] = boundary.get(i);
  }

  // Forward to type-erased bitcode implementation
  return rocshmem_tile_allgather_internal(team, dst.data_handle(), src.data_handle(),
                                    dst_strides, src_strides,
                                    start_arr, boundary_arr,
                                    ndim, sizeof(element_t), flags);
}

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_allgather_wave(rocshmem_ctx_t ctx,
                                                rocshmem_team_t team,
                                                dst_tensor_t dst, const src_tensor_t src,
                                                tuple_t start_coord, tuple_t boundary,
                                                uint64_t flags) {
  using element_t = typename src_tensor_t::element_type;
  constexpr int ndim = src_tensor_t::ndim;

  size_t dst_strides[ndim];
  size_t src_strides[ndim];
  size_t start_arr[ndim];
  size_t boundary_arr[ndim];

  #pragma unroll
  for (int i = 0; i < ndim; i++) {
    dst_strides[i] = dst.stride(i);
    src_strides[i] = src.stride(i);
    start_arr[i] = start_coord.get(i);
    boundary_arr[i] = boundary.get(i);
  }

  // Forward to type-erased bitcode implementation
  return rocshmem_tile_allgather_wave_internal(team, dst.data_handle(), src.data_handle(),
                                         dst_strides, src_strides,
                                         start_arr, boundary_arr,
                                         ndim, sizeof(element_t), flags);
}

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_allgather_wg(rocshmem_ctx_t ctx,
                                              rocshmem_team_t team,
                                              dst_tensor_t dst, const src_tensor_t src,
                                              tuple_t start_coord, tuple_t boundary,
                                              uint64_t flags) {
  using element_t = typename src_tensor_t::element_type;
  constexpr int ndim = src_tensor_t::ndim;

  size_t dst_strides[ndim];
  size_t src_strides[ndim];
  size_t start_arr[ndim];
  size_t boundary_arr[ndim];

  #pragma unroll
  for (int i = 0; i < ndim; i++) {
    dst_strides[i] = dst.stride(i);
    src_strides[i] = src.stride(i);
    start_arr[i] = start_coord.get(i);
    boundary_arr[i] = boundary.get(i);
  }

  // Forward to type-erased bitcode implementation
  return rocshmem_tile_allgather_wg_internal(team, dst.data_handle(), src.data_handle(),
                                       dst_strides, src_strides,
                                       start_arr, boundary_arr,
                                       ndim, sizeof(element_t), flags);
}

/******************************************************************************
 ******** COLLECTIVE ALLGATHER - DEFAULT CONTEXT WRAPPERS (3) *****************
 *****************************************************************************/

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_allgather(rocshmem_team_t team, dst_tensor_t dst,
                                       const src_tensor_t src, tuple_t start_coord,
                                       tuple_t boundary, uint64_t flags) {
  return rocshmem_ctx_tile_allgather(ROCSHMEM_CTX_DEFAULT, team, dst, src,
                                     start_coord, boundary, flags);
}

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_allgather_wave(rocshmem_team_t team, dst_tensor_t dst,
                                            const src_tensor_t src, tuple_t start_coord,
                                            tuple_t boundary, uint64_t flags) {
  return rocshmem_ctx_tile_allgather_wave(ROCSHMEM_CTX_DEFAULT, team, dst, src,
                                          start_coord, boundary, flags);
}

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_allgather_wg(rocshmem_team_t team, dst_tensor_t dst,
                                          const src_tensor_t src, tuple_t start_coord,
                                          tuple_t boundary, uint64_t flags) {
  return rocshmem_ctx_tile_allgather_wg(ROCSHMEM_CTX_DEFAULT, team, dst, src,
                                        start_coord, boundary, flags);
}

/******************************************************************************
 *********** COLLECTIVE BROADCAST - CONTEXT VERSIONS (3) **********************
 *****************************************************************************/

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_broadcast(rocshmem_ctx_t ctx,
                                           rocshmem_team_t team,
                                           dst_tensor_t dst, const src_tensor_t src,
                                           tuple_t start_coord, tuple_t boundary,
                                           int pe_root, uint64_t flags) {
  using element_t = typename src_tensor_t::element_type;
  constexpr int ndim = src_tensor_t::ndim;

  size_t dst_strides[ndim];
  size_t src_strides[ndim];
  size_t start_arr[ndim];
  size_t boundary_arr[ndim];

  #pragma unroll
  for (int i = 0; i < ndim; i++) {
    dst_strides[i] = dst.stride(i);
    src_strides[i] = src.stride(i);
    start_arr[i] = start_coord.get(i);
    boundary_arr[i] = boundary.get(i);
  }

  // Forward to type-erased bitcode implementation
  return rocshmem_tile_broadcast_internal(team, dst.data_handle(), src.data_handle(),
                                    dst_strides, src_strides,
                                    start_arr, boundary_arr,
                                    ndim, sizeof(element_t), pe_root, flags);
}

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_broadcast_wave(rocshmem_ctx_t ctx,
                                                rocshmem_team_t team,
                                                dst_tensor_t dst, const src_tensor_t src,
                                                tuple_t start_coord, tuple_t boundary,
                                                int pe_root, uint64_t flags) {
  using element_t = typename src_tensor_t::element_type;
  constexpr int ndim = src_tensor_t::ndim;

  size_t dst_strides[ndim];
  size_t src_strides[ndim];
  size_t start_arr[ndim];
  size_t boundary_arr[ndim];

  #pragma unroll
  for (int i = 0; i < ndim; i++) {
    dst_strides[i] = dst.stride(i);
    src_strides[i] = src.stride(i);
    start_arr[i] = start_coord.get(i);
    boundary_arr[i] = boundary.get(i);
  }

  // Forward to type-erased bitcode implementation
  return rocshmem_tile_broadcast_wave_internal(team, dst.data_handle(), src.data_handle(),
                                         dst_strides, src_strides,
                                         start_arr, boundary_arr,
                                         ndim, sizeof(element_t), pe_root, flags);
}

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_broadcast_wg(rocshmem_ctx_t ctx,
                                              rocshmem_team_t team,
                                              dst_tensor_t dst, const src_tensor_t src,
                                              tuple_t start_coord, tuple_t boundary,
                                              int pe_root, uint64_t flags) {
  using element_t = typename src_tensor_t::element_type;
  constexpr int ndim = src_tensor_t::ndim;

  size_t dst_strides[ndim];
  size_t src_strides[ndim];
  size_t start_arr[ndim];
  size_t boundary_arr[ndim];

  #pragma unroll
  for (int i = 0; i < ndim; i++) {
    dst_strides[i] = dst.stride(i);
    src_strides[i] = src.stride(i);
    start_arr[i] = start_coord.get(i);
    boundary_arr[i] = boundary.get(i);
  }

  // Forward to type-erased bitcode implementation
  return rocshmem_tile_broadcast_wg_internal(team, dst.data_handle(), src.data_handle(),
                                       dst_strides, src_strides,
                                       start_arr, boundary_arr,
                                       ndim, sizeof(element_t), pe_root, flags);
}

/******************************************************************************
 ******** COLLECTIVE BROADCAST - DEFAULT CONTEXT WRAPPERS (3) *****************
 *****************************************************************************/

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_broadcast(rocshmem_team_t team,
                                       dst_tensor_t dst, const src_tensor_t src,
                                       tuple_t start_coord, tuple_t boundary,
                                       int pe_root, uint64_t flags) {
  return rocshmem_ctx_tile_broadcast(ROCSHMEM_CTX_DEFAULT, team, dst, src,
                                     start_coord, boundary, pe_root, flags);
}

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_broadcast_wave(rocshmem_team_t team,
                                            dst_tensor_t dst, const src_tensor_t src,
                                            tuple_t start_coord, tuple_t boundary,
                                            int pe_root, uint64_t flags) {
  return rocshmem_ctx_tile_broadcast_wave(ROCSHMEM_CTX_DEFAULT, team, dst, src,
                                          start_coord, boundary, pe_root, flags);
}

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_broadcast_wg(rocshmem_team_t team,
                                          dst_tensor_t dst, const src_tensor_t src,
                                          tuple_t start_coord, tuple_t boundary,
                                          int pe_root, uint64_t flags) {
  return rocshmem_ctx_tile_broadcast_wg(ROCSHMEM_CTX_DEFAULT, team, dst, src,
                                        start_coord, boundary, pe_root, flags);
}

/******************************************************************************
 ****************** COLLECTIVE WAIT - CONTEXT VERSION (1) *********************
 *****************************************************************************/

__device__ inline int rocshmem_ctx_tile_collective_wait(rocshmem_ctx_t ctx,
                                                 rocshmem_team_t team,
                                                 uint64_t flags) {
  // Forward to type-erased bitcode implementation
  return rocshmem_tile_collective_wait_internal(team, flags);
}

/******************************************************************************
 ************* COLLECTIVE WAIT - DEFAULT CONTEXT WRAPPER (1) ******************
 *****************************************************************************/

__device__ inline int rocshmem_tile_collective_wait(rocshmem_team_t team,
                                             uint64_t flags) {
  return rocshmem_ctx_tile_collective_wait(ROCSHMEM_CTX_DEFAULT, team, flags);
}

/******************************************************************************
 ******************* SUM REDUCTIONS - CONTEXT VERSIONS (6) ********************
 *****************************************************************************/

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_sum_reduce(rocshmem_ctx_t ctx,
                                            rocshmem_team_t team,
                                            dst_tensor_t dst, const src_tensor_t src,
                                            tuple_t start_coord,
                                            tuple_t boundary, int root, uint64_t flags) {
  using element_t = typename src_tensor_t::element_type;
  constexpr int ndim = src_tensor_t::ndim;

  size_t dst_strides[ndim];
  size_t src_strides[ndim];
  size_t start_arr[ndim];
  size_t boundary_arr[ndim];

  #pragma unroll
  for (int i = 0; i < ndim; i++) {
    dst_strides[i] = dst.stride(i);
    src_strides[i] = src.stride(i);
    start_arr[i] = start_coord.get(i);
    boundary_arr[i] = boundary.get(i);
  }

  // Forward to type-erased bitcode implementation
  return rocshmem_tile_sum_reduce_internal(team, dst.data_handle(), src.data_handle(),
                                     dst_strides, src_strides,
                                     start_arr, boundary_arr,
                                     ndim, sizeof(element_t), root, flags);
}

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_sum_reduce_wave(rocshmem_ctx_t ctx,
                                                 rocshmem_team_t team,
                                                 dst_tensor_t dst,
                                                 const src_tensor_t src,
                                                 tuple_t start_coord,
                                                 tuple_t boundary,
                                                 int root, uint64_t flags) {
  using element_t = typename src_tensor_t::element_type;
  constexpr int ndim = src_tensor_t::ndim;

  size_t dst_strides[ndim];
  size_t src_strides[ndim];
  size_t start_arr[ndim];
  size_t boundary_arr[ndim];

  #pragma unroll
  for (int i = 0; i < ndim; i++) {
    dst_strides[i] = dst.stride(i);
    src_strides[i] = src.stride(i);
    start_arr[i] = start_coord.get(i);
    boundary_arr[i] = boundary.get(i);
  }

  // Forward to type-erased bitcode implementation
  return rocshmem_tile_sum_reduce_wave_internal(team, dst.data_handle(), src.data_handle(),
                                          dst_strides, src_strides,
                                          start_arr, boundary_arr,
                                          ndim, sizeof(element_t), root, flags);
}

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_sum_reduce_wg(rocshmem_ctx_t ctx,
                                               rocshmem_team_t team,
                                               dst_tensor_t dst,
                                               const src_tensor_t src,
                                               tuple_t start_coord,
                                               tuple_t boundary,
                                               int root, uint64_t flags) {
  using element_t = typename src_tensor_t::element_type;
  constexpr int ndim = src_tensor_t::ndim;

  size_t dst_strides[ndim];
  size_t src_strides[ndim];
  size_t start_arr[ndim];
  size_t boundary_arr[ndim];

  #pragma unroll
  for (int i = 0; i < ndim; i++) {
    dst_strides[i] = dst.stride(i);
    src_strides[i] = src.stride(i);
    start_arr[i] = start_coord.get(i);
    boundary_arr[i] = boundary.get(i);
  }

  // Forward to type-erased bitcode implementation
  return rocshmem_tile_sum_reduce_wg_internal(team, dst.data_handle(), src.data_handle(),
                                        dst_strides, src_strides,
                                        start_arr, boundary_arr,
                                        ndim, sizeof(element_t), root, flags);
}

/******************************************************************************
 ************** SUM REDUCTIONS - DEFAULT CONTEXT WRAPPERS (6) *****************
 *****************************************************************************/

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_sum_reduce(rocshmem_team_t team, dst_tensor_t dst,
                                        const src_tensor_t src, tuple_t start_coord,
                                        tuple_t boundary, int root, uint64_t flags) {
  return rocshmem_ctx_tile_sum_reduce(ROCSHMEM_CTX_DEFAULT, team, dst, src,
                                      start_coord, boundary, root, flags);
}

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_sum_reduce_wave(rocshmem_team_t team,
                                             dst_tensor_t dst,
                                             const src_tensor_t src,
                                             tuple_t start_coord,
                                             tuple_t boundary,
                                             int root, uint64_t flags) {
  return rocshmem_ctx_tile_sum_reduce_wave(ROCSHMEM_CTX_DEFAULT, team, dst,
                                           src, start_coord, boundary, root, flags);
}

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_sum_reduce_wg(rocshmem_team_t team, dst_tensor_t dst,
                                           const src_tensor_t src, tuple_t start_coord,
                                           tuple_t boundary,
                                           int root, uint64_t flags) {
  return rocshmem_ctx_tile_sum_reduce_wg(ROCSHMEM_CTX_DEFAULT, team, dst, src,
                                         start_coord, boundary, root, flags);
}

/******************************************************************************
 ******************* MAX REDUCTIONS - CONTEXT VERSIONS (3) ********************
 *****************************************************************************/

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_max_reduce(rocshmem_ctx_t ctx,
                                            rocshmem_team_t team,
                                            dst_tensor_t dst, const src_tensor_t src,
                                            tuple_t start_coord,
                                            tuple_t boundary, int root, uint64_t flags) {
  using element_t = typename src_tensor_t::element_type;
  constexpr int ndim = src_tensor_t::ndim;

  size_t dst_strides[ndim];
  size_t src_strides[ndim];
  size_t start_arr[ndim];
  size_t boundary_arr[ndim];

  #pragma unroll
  for (int i = 0; i < ndim; i++) {
    dst_strides[i] = dst.stride(i);
    src_strides[i] = src.stride(i);
    start_arr[i] = start_coord.get(i);
    boundary_arr[i] = boundary.get(i);
  }

  // Forward to type-erased bitcode implementation
  return rocshmem_tile_max_reduce_internal(team, dst.data_handle(), src.data_handle(),
                                     dst_strides, src_strides,
                                     start_arr, boundary_arr,
                                     ndim, sizeof(element_t), root, flags);
}

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_max_reduce_wave(rocshmem_ctx_t ctx,
                                                 rocshmem_team_t team,
                                                 dst_tensor_t dst,
                                                 const src_tensor_t src,
                                                 tuple_t start_coord,
                                                 tuple_t boundary,
                                                 int root, uint64_t flags) {
  using element_t = typename src_tensor_t::element_type;
  constexpr int ndim = src_tensor_t::ndim;

  size_t dst_strides[ndim];
  size_t src_strides[ndim];
  size_t start_arr[ndim];
  size_t boundary_arr[ndim];

  #pragma unroll
  for (int i = 0; i < ndim; i++) {
    dst_strides[i] = dst.stride(i);
    src_strides[i] = src.stride(i);
    start_arr[i] = start_coord.get(i);
    boundary_arr[i] = boundary.get(i);
  }

  // Forward to type-erased bitcode implementation
  return rocshmem_tile_max_reduce_wave_internal(team, dst.data_handle(), src.data_handle(),
                                          dst_strides, src_strides,
                                          start_arr, boundary_arr,
                                          ndim, sizeof(element_t), root, flags);
}

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_max_reduce_wg(rocshmem_ctx_t ctx,
                                               rocshmem_team_t team,
                                               dst_tensor_t dst,
                                               const src_tensor_t src,
                                               tuple_t start_coord,
                                               tuple_t boundary,
                                               int root, uint64_t flags) {
  using element_t = typename src_tensor_t::element_type;
  constexpr int ndim = src_tensor_t::ndim;

  size_t dst_strides[ndim];
  size_t src_strides[ndim];
  size_t start_arr[ndim];
  size_t boundary_arr[ndim];

  #pragma unroll
  for (int i = 0; i < ndim; i++) {
    dst_strides[i] = dst.stride(i);
    src_strides[i] = src.stride(i);
    start_arr[i] = start_coord.get(i);
    boundary_arr[i] = boundary.get(i);
  }

  // Forward to type-erased bitcode implementation
  return rocshmem_tile_max_reduce_wg_internal(team, dst.data_handle(), src.data_handle(),
                                        dst_strides, src_strides,
                                        start_arr, boundary_arr,
                                        ndim, sizeof(element_t), root, flags);
}

/******************************************************************************
 ************** MAX REDUCTIONS - DEFAULT CONTEXT WRAPPERS (3) *****************
 *****************************************************************************/

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_max_reduce(rocshmem_team_t team, dst_tensor_t dst,
                                        const src_tensor_t src, tuple_t start_coord,
                                        tuple_t boundary, int root, uint64_t flags) {
  return rocshmem_ctx_tile_max_reduce(ROCSHMEM_CTX_DEFAULT, team, dst, src,
                                      start_coord, boundary, root, flags);
}

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_max_reduce_wave(rocshmem_team_t team,
                                             dst_tensor_t dst,
                                             const src_tensor_t src,
                                             tuple_t start_coord,
                                             tuple_t boundary,
                                             int root, uint64_t flags) {
  return rocshmem_ctx_tile_max_reduce_wave(ROCSHMEM_CTX_DEFAULT, team, dst,
                                           src, start_coord, boundary, root, flags);
}

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_max_reduce_wg(rocshmem_team_t team, dst_tensor_t dst,
                                           const src_tensor_t src, tuple_t start_coord,
                                           tuple_t boundary,
                                           int root, uint64_t flags) {
  return rocshmem_ctx_tile_max_reduce_wg(ROCSHMEM_CTX_DEFAULT, team, dst, src,
                                         start_coord, boundary, root, flags);
}

/******************************************************************************
 ******************* MIN REDUCTIONS - CONTEXT VERSIONS (3) ********************
 *****************************************************************************/

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_min_reduce(rocshmem_ctx_t ctx,
                                            rocshmem_team_t team,
                                            dst_tensor_t dst, const src_tensor_t src,
                                            tuple_t start_coord,
                                            tuple_t boundary, int root, uint64_t flags) {
  using element_t = typename src_tensor_t::element_type;
  constexpr int ndim = src_tensor_t::ndim;

  size_t dst_strides[ndim];
  size_t src_strides[ndim];
  size_t start_arr[ndim];
  size_t boundary_arr[ndim];

  #pragma unroll
  for (int i = 0; i < ndim; i++) {
    dst_strides[i] = dst.stride(i);
    src_strides[i] = src.stride(i);
    start_arr[i] = start_coord.get(i);
    boundary_arr[i] = boundary.get(i);
  }

  // Forward to type-erased bitcode implementation
  return rocshmem_tile_min_reduce_internal(team, dst.data_handle(), src.data_handle(),
                                     dst_strides, src_strides,
                                     start_arr, boundary_arr,
                                     ndim, sizeof(element_t), root, flags);
}

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_min_reduce_wave(rocshmem_ctx_t ctx,
                                                 rocshmem_team_t team,
                                                 dst_tensor_t dst,
                                                 const src_tensor_t src,
                                                 tuple_t start_coord,
                                                 tuple_t boundary,
                                                 int root, uint64_t flags) {
  using element_t = typename src_tensor_t::element_type;
  constexpr int ndim = src_tensor_t::ndim;

  size_t dst_strides[ndim];
  size_t src_strides[ndim];
  size_t start_arr[ndim];
  size_t boundary_arr[ndim];

  #pragma unroll
  for (int i = 0; i < ndim; i++) {
    dst_strides[i] = dst.stride(i);
    src_strides[i] = src.stride(i);
    start_arr[i] = start_coord.get(i);
    boundary_arr[i] = boundary.get(i);
  }

  // Forward to type-erased bitcode implementation
  return rocshmem_tile_min_reduce_wave_internal(team, dst.data_handle(), src.data_handle(),
                                          dst_strides, src_strides,
                                          start_arr, boundary_arr,
                                          ndim, sizeof(element_t), root, flags);
}

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_min_reduce_wg(rocshmem_ctx_t ctx,
                                               rocshmem_team_t team,
                                               dst_tensor_t dst,
                                               const src_tensor_t src,
                                               tuple_t start_coord,
                                               tuple_t boundary,
                                               int root, uint64_t flags) {
  using element_t = typename src_tensor_t::element_type;
  constexpr int ndim = src_tensor_t::ndim;

  size_t dst_strides[ndim];
  size_t src_strides[ndim];
  size_t start_arr[ndim];
  size_t boundary_arr[ndim];

  #pragma unroll
  for (int i = 0; i < ndim; i++) {
    dst_strides[i] = dst.stride(i);
    src_strides[i] = src.stride(i);
    start_arr[i] = start_coord.get(i);
    boundary_arr[i] = boundary.get(i);
  }

  // Forward to type-erased bitcode implementation
  return rocshmem_tile_min_reduce_wg_internal(team, dst.data_handle(), src.data_handle(),
                                        dst_strides, src_strides,
                                        start_arr, boundary_arr,
                                        ndim, sizeof(element_t), root, flags);
}

/******************************************************************************
 ************** MIN REDUCTIONS - DEFAULT CONTEXT WRAPPERS (3) *****************
 *****************************************************************************/

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_min_reduce(rocshmem_team_t team, dst_tensor_t dst,
                                        const src_tensor_t src, tuple_t start_coord,
                                        tuple_t boundary, int root, uint64_t flags) {
  return rocshmem_ctx_tile_min_reduce(ROCSHMEM_CTX_DEFAULT, team, dst, src,
                                      start_coord, boundary, root, flags);
}

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_min_reduce_wave(rocshmem_team_t team,
                                             dst_tensor_t dst,
                                             const src_tensor_t src,
                                             tuple_t start_coord,
                                             tuple_t boundary,
                                             int root, uint64_t flags) {
  return rocshmem_ctx_tile_min_reduce_wave(ROCSHMEM_CTX_DEFAULT, team, dst,
                                           src, start_coord, boundary, root, flags);
}

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_min_reduce_wg(rocshmem_team_t team, dst_tensor_t dst,
                                           const src_tensor_t src, tuple_t start_coord,
                                           tuple_t boundary,
                                           int root, uint64_t flags) {
  return rocshmem_ctx_tile_min_reduce_wg(ROCSHMEM_CTX_DEFAULT, team, dst, src,
                                         start_coord, boundary, root, flags);
}

}  // namespace rocshmem

#endif  // __HIP_DEVICE_COMPILE__

#endif  // LIBRARY_INCLUDE_ROCSHMEM_TILE_IMPL_HPP
