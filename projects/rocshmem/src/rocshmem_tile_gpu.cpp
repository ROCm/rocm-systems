/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

/**
 * @file rocshmem_tile_gpu.cpp
 * @brief Type-erased implementations for rocSHMEM tile API functions.
 *
 * This file provides the type-erased namespace implementations that bridge
 * the public template API to the internal Context implementations. It is the
 * ONLY source file that includes context_incl.hpp, keeping internal headers
 * out of the public API.
 *
 * The public API (rocshmem_TILE_impl.hpp) extracts type information from
 * generic tensor types and forwards to these namespace functions (with _internal
 * suffix), which then delegate to the appropriate backend Context implementation.
 *
 * The extern "C" wrappers (without _internal suffix) for JIT consumers are in
 * src/device/rocshmem_wrapper.cc and forward to these namespace functions.
 */

#include <hip/hip_runtime.h>

#include "context_incl.hpp"
#include "rocshmem/rocshmem.hpp"

namespace rocshmem {

// Helper function to retrieve internal context from opaque handle
static __device__ Context* get_internal_ctx(rocshmem_ctx_t ctx) {
  return reinterpret_cast<Context*>(ctx.ctx_opaque);
}

/******************************************************************************
 *************************** RMA PUT OPERATIONS ********************************
 *****************************************************************************/

__device__ int rocshmem_ctx_tile_put_internal(
    rocshmem_ctx_t ctx,
    void* dst_data,
    const void* src_data,
    const size_t* dst_strides,
    const size_t* src_strides,
    const size_t* start_coord,
    const size_t* boundary,
    int ndim,
    size_t element_size,
    int pe,
    uint64_t flags) {

  auto* internal_ctx = get_internal_ctx(ctx);
  return internal_ctx->tile_put(dst_data, src_data, dst_strides, src_strides,
                                start_coord, boundary, ndim, element_size, pe, flags);
}

__device__ int rocshmem_ctx_tile_put_wave_internal(
    rocshmem_ctx_t ctx,
    void* dst_data,
    const void* src_data,
    const size_t* dst_strides,
    const size_t* src_strides,
    const size_t* start_coord,
    const size_t* boundary,
    int ndim,
    size_t element_size,
    int pe,
    uint64_t flags) {

  auto* internal_ctx = get_internal_ctx(ctx);
  return internal_ctx->tile_put_wave(dst_data, src_data, dst_strides, src_strides,
                                     start_coord, boundary, ndim, element_size, pe, flags);
}

__device__ int rocshmem_ctx_tile_put_wg_internal(
    rocshmem_ctx_t ctx,
    void* dst_data,
    const void* src_data,
    const size_t* dst_strides,
    const size_t* src_strides,
    const size_t* start_coord,
    const size_t* boundary,
    int ndim,
    size_t element_size,
    int pe,
    uint64_t flags) {

  auto* internal_ctx = get_internal_ctx(ctx);
  return internal_ctx->tile_put_wg(dst_data, src_data, dst_strides, src_strides,
                                   start_coord, boundary, ndim, element_size, pe, flags);
}

/******************************************************************************
 *************************** RMA GET OPERATIONS ********************************
 *****************************************************************************/

__device__ int rocshmem_ctx_tile_get_internal(
    rocshmem_ctx_t ctx,
    void* dst_data,
    const void* src_data,
    const size_t* dst_strides,
    const size_t* src_strides,
    const size_t* start_coord,
    const size_t* boundary,
    int ndim,
    size_t element_size,
    int pe,
    uint64_t flags) {

  auto* internal_ctx = get_internal_ctx(ctx);
  return internal_ctx->tile_get(dst_data, src_data, dst_strides, src_strides,
                                start_coord, boundary, ndim, element_size, pe, flags);
}

__device__ int rocshmem_ctx_tile_get_wave_internal(
    rocshmem_ctx_t ctx,
    void* dst_data,
    const void* src_data,
    const size_t* dst_strides,
    const size_t* src_strides,
    const size_t* start_coord,
    const size_t* boundary,
    int ndim,
    size_t element_size,
    int pe,
    uint64_t flags) {

  auto* internal_ctx = get_internal_ctx(ctx);
  return internal_ctx->tile_get_wave(dst_data, src_data, dst_strides, src_strides,
                                     start_coord, boundary, ndim, element_size, pe, flags);
}

__device__ int rocshmem_ctx_tile_get_wg_internal(
    rocshmem_ctx_t ctx,
    void* dst_data,
    const void* src_data,
    const size_t* dst_strides,
    const size_t* src_strides,
    const size_t* start_coord,
    const size_t* boundary,
    int ndim,
    size_t element_size,
    int pe,
    uint64_t flags) {

  auto* internal_ctx = get_internal_ctx(ctx);
  return internal_ctx->tile_get_wg(dst_data, src_data, dst_strides, src_strides,
                                   start_coord, boundary, ndim, element_size, pe, flags);
}

/******************************************************************************
 ************************ COLLECTIVE - ALLGATHER *******************************
 *****************************************************************************/

__device__ int rocshmem_tile_allgather_internal(
    rocshmem_team_t team,
    void* dst_data,
    const void* src_data,
    const size_t* dst_strides,
    const size_t* src_strides,
    const size_t* start_coord,
    const size_t* boundary,
    int ndim,
    size_t element_size,
    uint64_t flags) {

  auto* internal_ctx = get_internal_ctx(ROCSHMEM_CTX_DEFAULT);
  return internal_ctx->tile_allgather(team, dst_data, src_data, dst_strides, src_strides,
                                      start_coord, boundary, ndim, element_size, flags);
}

__device__ int rocshmem_tile_allgather_wave_internal(
    rocshmem_team_t team,
    void* dst_data,
    const void* src_data,
    const size_t* dst_strides,
    const size_t* src_strides,
    const size_t* start_coord,
    const size_t* boundary,
    int ndim,
    size_t element_size,
    uint64_t flags) {

  auto* internal_ctx = get_internal_ctx(ROCSHMEM_CTX_DEFAULT);
  return internal_ctx->tile_allgather_wave(team, dst_data, src_data, dst_strides, src_strides,
                                           start_coord, boundary, ndim, element_size, flags);
}

__device__ int rocshmem_tile_allgather_wg_internal(
    rocshmem_team_t team,
    void* dst_data,
    const void* src_data,
    const size_t* dst_strides,
    const size_t* src_strides,
    const size_t* start_coord,
    const size_t* boundary,
    int ndim,
    size_t element_size,
    uint64_t flags) {

  auto* internal_ctx = get_internal_ctx(ROCSHMEM_CTX_DEFAULT);
  return internal_ctx->tile_allgather_wg(team, dst_data, src_data, dst_strides, src_strides,
                                         start_coord, boundary, ndim, element_size, flags);
}

/******************************************************************************
 ************************ COLLECTIVE - BROADCAST *******************************
 *****************************************************************************/

__device__ int rocshmem_tile_broadcast_internal(
    rocshmem_team_t team,
    void* dst_data,
    const void* src_data,
    const size_t* dst_strides,
    const size_t* src_strides,
    const size_t* start_coord,
    const size_t* boundary,
    int ndim,
    size_t element_size,
    int pe_root,
    uint64_t flags) {

  auto* internal_ctx = get_internal_ctx(ROCSHMEM_CTX_DEFAULT);
  return internal_ctx->tile_broadcast(team, dst_data, src_data, dst_strides, src_strides,
                                      start_coord, boundary, ndim, element_size, pe_root, flags);
}

__device__ int rocshmem_tile_broadcast_wave_internal(
    rocshmem_team_t team,
    void* dst_data,
    const void* src_data,
    const size_t* dst_strides,
    const size_t* src_strides,
    const size_t* start_coord,
    const size_t* boundary,
    int ndim,
    size_t element_size,
    int pe_root,
    uint64_t flags) {

  auto* internal_ctx = get_internal_ctx(ROCSHMEM_CTX_DEFAULT);
  return internal_ctx->tile_broadcast_wave(team, dst_data, src_data, dst_strides, src_strides,
                                           start_coord, boundary, ndim, element_size, pe_root, flags);
}

__device__ int rocshmem_tile_broadcast_wg_internal(
    rocshmem_team_t team,
    void* dst_data,
    const void* src_data,
    const size_t* dst_strides,
    const size_t* src_strides,
    const size_t* start_coord,
    const size_t* boundary,
    int ndim,
    size_t element_size,
    int pe_root,
    uint64_t flags) {

  auto* internal_ctx = get_internal_ctx(ROCSHMEM_CTX_DEFAULT);
  return internal_ctx->tile_broadcast_wg(team, dst_data, src_data, dst_strides, src_strides,
                                         start_coord, boundary, ndim, element_size, pe_root, flags);
}

/******************************************************************************
 ************************ REDUCTION - SUM **************************************
 *****************************************************************************/

__device__ int rocshmem_tile_sum_reduce_internal(
    rocshmem_team_t team,
    void* dst_data,
    const void* src_data,
    const size_t* dst_strides,
    const size_t* src_strides,
    const size_t* start_coord,
    const size_t* boundary,
    int ndim,
    size_t element_size,
    int root,
    uint64_t flags) {

  auto* internal_ctx = get_internal_ctx(ROCSHMEM_CTX_DEFAULT);
  return internal_ctx->tile_sum_reduce(team, dst_data, src_data, dst_strides, src_strides,
                                       start_coord, boundary, ndim, element_size, root, flags);
}

__device__ int rocshmem_tile_sum_reduce_wave_internal(
    rocshmem_team_t team,
    void* dst_data,
    const void* src_data,
    const size_t* dst_strides,
    const size_t* src_strides,
    const size_t* start_coord,
    const size_t* boundary,
    int ndim,
    size_t element_size,
    int root,
    uint64_t flags) {

  auto* internal_ctx = get_internal_ctx(ROCSHMEM_CTX_DEFAULT);
  return internal_ctx->tile_sum_reduce_wave(team, dst_data, src_data, dst_strides, src_strides,
                                            start_coord, boundary, ndim, element_size, root, flags);
}

__device__ int rocshmem_tile_sum_reduce_wg_internal(
    rocshmem_team_t team,
    void* dst_data,
    const void* src_data,
    const size_t* dst_strides,
    const size_t* src_strides,
    const size_t* start_coord,
    const size_t* boundary,
    int ndim,
    size_t element_size,
    int root,
    uint64_t flags) {

  auto* internal_ctx = get_internal_ctx(ROCSHMEM_CTX_DEFAULT);
  return internal_ctx->tile_sum_reduce_wg(team, dst_data, src_data, dst_strides, src_strides,
                                          start_coord, boundary, ndim, element_size, root, flags);
}

/******************************************************************************
 ************************ COLLECTIVE WAIT **************************************
 *****************************************************************************/

/******************************************************************************
 ************************ REDUCTION - MAX **************************************
 *****************************************************************************/

__device__ int rocshmem_tile_max_reduce_internal(
    rocshmem_team_t team,
    void* dst_data,
    const void* src_data,
    const size_t* dst_strides,
    const size_t* src_strides,
    const size_t* start_coord,
    const size_t* boundary,
    int ndim,
    size_t element_size,
    int root,
    uint64_t flags) {

  return get_internal_ctx(ROCSHMEM_CTX_DEFAULT)->tile_max_reduce(
      team, dst_data, src_data, dst_strides, src_strides,
      start_coord, boundary, ndim, element_size, root, flags);
}

__device__ int rocshmem_tile_max_reduce_wave_internal(
    rocshmem_team_t team,
    void* dst_data,
    const void* src_data,
    const size_t* dst_strides,
    const size_t* src_strides,
    const size_t* start_coord,
    const size_t* boundary,
    int ndim,
    size_t element_size,
    int root,
    uint64_t flags) {

  return get_internal_ctx(ROCSHMEM_CTX_DEFAULT)->tile_max_reduce_wave(
      team, dst_data, src_data, dst_strides, src_strides,
      start_coord, boundary, ndim, element_size, root, flags);
}

__device__ int rocshmem_tile_max_reduce_wg_internal(
    rocshmem_team_t team,
    void* dst_data,
    const void* src_data,
    const size_t* dst_strides,
    const size_t* src_strides,
    const size_t* start_coord,
    const size_t* boundary,
    int ndim,
    size_t element_size,
    int root,
    uint64_t flags) {

  return get_internal_ctx(ROCSHMEM_CTX_DEFAULT)->tile_max_reduce_wg(
      team, dst_data, src_data, dst_strides, src_strides,
      start_coord, boundary, ndim, element_size, root, flags);
}

/******************************************************************************
 ************************ REDUCTION - MIN **************************************
 *****************************************************************************/

__device__ int rocshmem_tile_min_reduce_internal(
    rocshmem_team_t team,
    void* dst_data,
    const void* src_data,
    const size_t* dst_strides,
    const size_t* src_strides,
    const size_t* start_coord,
    const size_t* boundary,
    int ndim,
    size_t element_size,
    int root,
    uint64_t flags) {

  return get_internal_ctx(ROCSHMEM_CTX_DEFAULT)->tile_min_reduce(
      team, dst_data, src_data, dst_strides, src_strides,
      start_coord, boundary, ndim, element_size, root, flags);
}

__device__ int rocshmem_tile_min_reduce_wave_internal(
    rocshmem_team_t team,
    void* dst_data,
    const void* src_data,
    const size_t* dst_strides,
    const size_t* src_strides,
    const size_t* start_coord,
    const size_t* boundary,
    int ndim,
    size_t element_size,
    int root,
    uint64_t flags) {

  return get_internal_ctx(ROCSHMEM_CTX_DEFAULT)->tile_min_reduce_wave(
      team, dst_data, src_data, dst_strides, src_strides,
      start_coord, boundary, ndim, element_size, root, flags);
}

__device__ int rocshmem_tile_min_reduce_wg_internal(
    rocshmem_team_t team,
    void* dst_data,
    const void* src_data,
    const size_t* dst_strides,
    const size_t* src_strides,
    const size_t* start_coord,
    const size_t* boundary,
    int ndim,
    size_t element_size,
    int root,
    uint64_t flags) {

  return get_internal_ctx(ROCSHMEM_CTX_DEFAULT)->tile_min_reduce_wg(
      team, dst_data, src_data, dst_strides, src_strides,
      start_coord, boundary, ndim, element_size, root, flags);
}

__device__ int rocshmem_tile_collective_wait_internal(
    rocshmem_team_t team,
    uint64_t flags) {

  auto* internal_ctx = get_internal_ctx(ROCSHMEM_CTX_DEFAULT);
  return internal_ctx->tile_collective_wait(team, flags);
}

}  // namespace rocshmem
