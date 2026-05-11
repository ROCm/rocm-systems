/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
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

// We need the backend context implementations to be available
// The user must include context_incl.hpp before this header
#ifndef LIBRARY_SRC_CONTEXT_INCL_HPP_
#error "rocshmem_TILE_impl.hpp requires context_incl.hpp to be included first. Include '../../src/context_incl.hpp' before this header."
#endif

namespace rocshmem {

// Internal helper to get context pointer
// This typedef selects the appropriate context type based on build configuration
#if defined(ENABLE_IPC_BITCODE)
  typedef IPCContext ContextTy;
#elif defined(ENABLE_IBGDA_BITCODE)
  typedef GDAContext ContextTy;
#else
  typedef Context ContextTy;
#endif

static inline __device__ ContextTy *get_internal_ctx(rocshmem_ctx_t ctx) {
  return reinterpret_cast<ContextTy *>(ctx.ctx_opaque);
}

/******************************************************************************
 **************** RMA OPERATIONS - CONTEXT VERSIONS (5) ***********************
 *****************************************************************************/

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_put(rocshmem_ctx_t ctx, src_tensor_t src,
                                     dst_tensor_t dst, tuple_t start_coord,
                                     tuple_t boundary, int pe, uint64_t flags) {
  return get_internal_ctx(ctx)->tile_put(src, dst, start_coord, boundary, pe,
                                         flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_put_wave(rocshmem_ctx_t ctx, src_tensor_t src,
                                          dst_tensor_t dst, tuple_t start_coord,
                                          tuple_t boundary, int pe,
                                          uint64_t flags) {
  return get_internal_ctx(ctx)->tile_put_wave(src, dst, start_coord, boundary,
                                              pe, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_put_wg(rocshmem_ctx_t ctx, src_tensor_t src,
                                        dst_tensor_t dst, tuple_t start_coord,
                                        tuple_t boundary, int pe,
                                        uint64_t flags) {
  return get_internal_ctx(ctx)->tile_put_wg(src, dst, start_coord, boundary,
                                            pe, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_get(rocshmem_ctx_t ctx, dst_tensor_t dst,
                                     src_tensor_t src, tuple_t start_coord,
                                     tuple_t boundary, int pe, uint64_t flags) {
  return get_internal_ctx(ctx)->tile_get(dst, src, start_coord, boundary, pe,
                                         flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_get_wave(rocshmem_ctx_t ctx, dst_tensor_t dst,
                                          src_tensor_t src, tuple_t start_coord,
                                          tuple_t boundary, int pe,
                                          uint64_t flags) {
  return get_internal_ctx(ctx)->tile_get_wave(dst, src, start_coord, boundary,
                                              pe, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_get_wg(rocshmem_ctx_t ctx, dst_tensor_t dst,
                                        src_tensor_t src, tuple_t start_coord,
                                        tuple_t boundary, int pe,
                                        uint64_t flags) {
  return get_internal_ctx(ctx)->tile_get_wg(dst, src, start_coord, boundary,
                                            pe, flags);
}

/******************************************************************************
 *************** RMA OPERATIONS - DEFAULT CONTEXT WRAPPERS (5) ****************
 *****************************************************************************/

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_put(src_tensor_t src, dst_tensor_t dst,
                                 tuple_t start_coord, tuple_t boundary, int pe,
                                 uint64_t flags) {
  return rocshmem_ctx_tile_put(ROCSHMEM_CTX_DEFAULT, src, dst, start_coord,
                               boundary, pe, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_put_wave(src_tensor_t src, dst_tensor_t dst,
                                      tuple_t start_coord, tuple_t boundary,
                                      int pe, uint64_t flags) {
  return rocshmem_ctx_tile_put_wave(ROCSHMEM_CTX_DEFAULT, src, dst,
                                    start_coord, boundary, pe, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_put_wg(src_tensor_t src, dst_tensor_t dst,
                                    tuple_t start_coord, tuple_t boundary,
                                    int pe, uint64_t flags) {
  return rocshmem_ctx_tile_put_wg(ROCSHMEM_CTX_DEFAULT, src, dst, start_coord,
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

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_allgather(rocshmem_ctx_t ctx, src_tensor_t src,
                                           dst_tensor_t dst,
                                           tuple_t start_coord,
                                           tuple_t boundary,
                                           rocshmem_team_t team,
                                           uint64_t flags) {
  return get_internal_ctx(ctx)->tile_allgather(src, dst, start_coord, boundary,
                                               team, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_allgather_wave(rocshmem_ctx_t ctx,
                                                src_tensor_t src,
                                                dst_tensor_t dst,
                                                tuple_t start_coord,
                                                tuple_t boundary,
                                                rocshmem_team_t team,
                                                uint64_t flags) {
  return get_internal_ctx(ctx)->tile_allgather_wave(src, dst, start_coord,
                                                     boundary, team, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_allgather_wg(rocshmem_ctx_t ctx,
                                              src_tensor_t src, dst_tensor_t dst,
                                              tuple_t start_coord,
                                              tuple_t boundary,
                                              rocshmem_team_t team,
                                              uint64_t flags) {
  return get_internal_ctx(ctx)->tile_allgather_wg(src, dst, start_coord,
                                                   boundary, team, flags);
}

/******************************************************************************
 ******** COLLECTIVE ALLGATHER - DEFAULT CONTEXT WRAPPERS (3) *****************
 *****************************************************************************/

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_allgather(src_tensor_t src, dst_tensor_t dst,
                                       tuple_t start_coord, tuple_t boundary,
                                       rocshmem_team_t team, uint64_t flags) {
  return rocshmem_ctx_tile_allgather(ROCSHMEM_CTX_DEFAULT, src, dst,
                                     start_coord, boundary, team, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_allgather_wave(src_tensor_t src, dst_tensor_t dst,
                                            tuple_t start_coord,
                                            tuple_t boundary,
                                            rocshmem_team_t team,
                                            uint64_t flags) {
  return rocshmem_ctx_tile_allgather_wave(ROCSHMEM_CTX_DEFAULT, src, dst,
                                          start_coord, boundary, team, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_allgather_wg(src_tensor_t src, dst_tensor_t dst,
                                          tuple_t start_coord, tuple_t boundary,
                                          rocshmem_team_t team,
                                          uint64_t flags) {
  return rocshmem_ctx_tile_allgather_wg(ROCSHMEM_CTX_DEFAULT, src, dst,
                                        start_coord, boundary, team, flags);
}

/******************************************************************************
 *********** COLLECTIVE BROADCAST - CONTEXT VERSIONS (3) **********************
 *****************************************************************************/

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_broadcast(rocshmem_ctx_t ctx, src_tensor_t src,
                                           dst_tensor_t dst,
                                           tuple_t start_coord,
                                           tuple_t boundary, int pe_root,
                                           rocshmem_team_t team,
                                           uint64_t flags) {
  return get_internal_ctx(ctx)->tile_broadcast(src, dst, start_coord, boundary,
                                               pe_root, team, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_broadcast_wave(rocshmem_ctx_t ctx,
                                                src_tensor_t src,
                                                dst_tensor_t dst,
                                                tuple_t start_coord,
                                                tuple_t boundary, int pe_root,
                                                rocshmem_team_t team,
                                                uint64_t flags) {
  return get_internal_ctx(ctx)->tile_broadcast_wave(
      src, dst, start_coord, boundary, pe_root, team, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_broadcast_wg(rocshmem_ctx_t ctx,
                                              src_tensor_t src, dst_tensor_t dst,
                                              tuple_t start_coord,
                                              tuple_t boundary, int pe_root,
                                              rocshmem_team_t team,
                                              uint64_t flags) {
  return get_internal_ctx(ctx)->tile_broadcast_wg(src, dst, start_coord,
                                                   boundary, pe_root, team,
                                                   flags);
}

/******************************************************************************
 ******** COLLECTIVE BROADCAST - DEFAULT CONTEXT WRAPPERS (3) *****************
 *****************************************************************************/

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_broadcast(src_tensor_t src, dst_tensor_t dst,
                                       tuple_t start_coord, tuple_t boundary,
                                       int pe_root, rocshmem_team_t team,
                                       uint64_t flags) {
  return rocshmem_ctx_tile_broadcast(ROCSHMEM_CTX_DEFAULT, src, dst,
                                     start_coord, boundary, pe_root, team,
                                     flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_broadcast_wave(src_tensor_t src, dst_tensor_t dst,
                                            tuple_t start_coord,
                                            tuple_t boundary, int pe_root,
                                            rocshmem_team_t team,
                                            uint64_t flags) {
  return rocshmem_ctx_tile_broadcast_wave(ROCSHMEM_CTX_DEFAULT, src, dst,
                                          start_coord, boundary, pe_root, team,
                                          flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_broadcast_wg(src_tensor_t src, dst_tensor_t dst,
                                          tuple_t start_coord, tuple_t boundary,
                                          int pe_root, rocshmem_team_t team,
                                          uint64_t flags) {
  return rocshmem_ctx_tile_broadcast_wg(ROCSHMEM_CTX_DEFAULT, src, dst,
                                        start_coord, boundary, pe_root, team,
                                        flags);
}

/******************************************************************************
 ****************** COLLECTIVE WAIT - CONTEXT VERSION (1) *********************
 *****************************************************************************/

__device__ inline int rocshmem_ctx_tile_collective_wait(rocshmem_ctx_t ctx,
                                                 rocshmem_team_t team,
                                                 uint64_t flags) {
  return get_internal_ctx(ctx)->tile_collective_wait(team, flags);
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

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_sum_reduce(rocshmem_ctx_t ctx,
                                            rocshmem_team_t team,
                                            src_tensor_t src, dst_tensor_t dst,
                                            tuple_t start_coord,
                                            tuple_t boundary, uint64_t flags) {
  return get_internal_ctx(ctx)->tile_sum_reduce(src, dst, start_coord,
                                                 boundary, team, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_sum_reduce_wave(rocshmem_ctx_t ctx,
                                                 rocshmem_team_t team,
                                                 src_tensor_t src,
                                                 dst_tensor_t dst,
                                                 tuple_t start_coord,
                                                 tuple_t boundary,
                                                 uint64_t flags) {
  return get_internal_ctx(ctx)->tile_sum_reduce_wave(src, dst, start_coord,
                                                      boundary, team, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_sum_reduce_wg(rocshmem_ctx_t ctx,
                                               rocshmem_team_t team,
                                               src_tensor_t src,
                                               dst_tensor_t dst,
                                               tuple_t start_coord,
                                               tuple_t boundary,
                                               uint64_t flags) {
  return get_internal_ctx(ctx)->tile_sum_reduce_wg(src, dst, start_coord,
                                                    boundary, team, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_sum_rooted_reduce(rocshmem_ctx_t ctx,
                                                   rocshmem_team_t team,
                                                   src_tensor_t src,
                                                   dst_tensor_t dst,
                                                   tuple_t start_coord,
                                                   tuple_t boundary, int root,
                                                   uint64_t flags) {
  return get_internal_ctx(ctx)->tile_sum_rooted_reduce(
      src, dst, start_coord, boundary, root, team, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_sum_rooted_reduce_wave(
    rocshmem_ctx_t ctx, rocshmem_team_t team, src_tensor_t src,
    dst_tensor_t dst, tuple_t start_coord, tuple_t boundary, int root,
    uint64_t flags) {
  return get_internal_ctx(ctx)->tile_sum_rooted_reduce_wave(
      src, dst, start_coord, boundary, root, team, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_sum_rooted_reduce_wg(rocshmem_ctx_t ctx,
                                                      rocshmem_team_t team,
                                                      src_tensor_t src,
                                                      dst_tensor_t dst,
                                                      tuple_t start_coord,
                                                      tuple_t boundary, int root,
                                                      uint64_t flags) {
  return get_internal_ctx(ctx)->tile_sum_rooted_reduce_wg(
      src, dst, start_coord, boundary, root, team, flags);
}

/******************************************************************************
 ************** SUM REDUCTIONS - DEFAULT CONTEXT WRAPPERS (6) *****************
 *****************************************************************************/

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_sum_reduce(src_tensor_t src, dst_tensor_t dst,
                                        tuple_t start_coord, tuple_t boundary,
                                        rocshmem_team_t team, uint64_t flags) {
  return rocshmem_ctx_tile_sum_reduce(ROCSHMEM_CTX_DEFAULT, team, src, dst,
                                      start_coord, boundary, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_sum_reduce_wave(src_tensor_t src,
                                             dst_tensor_t dst,
                                             tuple_t start_coord,
                                             tuple_t boundary,
                                             rocshmem_team_t team,
                                             uint64_t flags) {
  return rocshmem_ctx_tile_sum_reduce_wave(ROCSHMEM_CTX_DEFAULT, team, src,
                                           dst, start_coord, boundary, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_sum_reduce_wg(src_tensor_t src, dst_tensor_t dst,
                                           tuple_t start_coord,
                                           tuple_t boundary,
                                           rocshmem_team_t team,
                                           uint64_t flags) {
  return rocshmem_ctx_tile_sum_reduce_wg(ROCSHMEM_CTX_DEFAULT, team, src, dst,
                                         start_coord, boundary, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_sum_rooted_reduce(src_tensor_t src,
                                               dst_tensor_t dst,
                                               tuple_t start_coord,
                                               tuple_t boundary, int pe_root,
                                               rocshmem_team_t team,
                                               uint64_t flags) {
  return rocshmem_ctx_tile_sum_rooted_reduce(ROCSHMEM_CTX_DEFAULT, team, src,
                                             dst, start_coord, boundary,
                                             pe_root, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_sum_rooted_reduce_wave(src_tensor_t src,
                                                    dst_tensor_t dst,
                                                    tuple_t start_coord,
                                                    tuple_t boundary,
                                                    int pe_root,
                                                    rocshmem_team_t team,
                                                    uint64_t flags) {
  return rocshmem_ctx_tile_sum_rooted_reduce_wave(
      ROCSHMEM_CTX_DEFAULT, team, src, dst, start_coord, boundary, pe_root,
      flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_sum_rooted_reduce_wg(src_tensor_t src,
                                                  dst_tensor_t dst,
                                                  tuple_t start_coord,
                                                  tuple_t boundary, int pe_root,
                                                  rocshmem_team_t team,
                                                  uint64_t flags) {
  return rocshmem_ctx_tile_sum_rooted_reduce_wg(
      ROCSHMEM_CTX_DEFAULT, team, src, dst, start_coord, boundary, pe_root,
      flags);
}

/******************************************************************************
 ******************* MAX REDUCTIONS - CONTEXT VERSIONS (6) ********************
 *****************************************************************************/

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_max_reduce(rocshmem_ctx_t ctx,
                                            rocshmem_team_t team,
                                            src_tensor_t src, dst_tensor_t dst,
                                            tuple_t start_coord,
                                            tuple_t boundary, uint64_t flags) {
  return get_internal_ctx(ctx)->tile_max_reduce(src, dst, start_coord,
                                                 boundary, team, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_max_reduce_wave(rocshmem_ctx_t ctx,
                                                 rocshmem_team_t team,
                                                 src_tensor_t src,
                                                 dst_tensor_t dst,
                                                 tuple_t start_coord,
                                                 tuple_t boundary,
                                                 uint64_t flags) {
  return get_internal_ctx(ctx)->tile_max_reduce_wave(src, dst, start_coord,
                                                      boundary, team, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_max_reduce_wg(rocshmem_ctx_t ctx,
                                               rocshmem_team_t team,
                                               src_tensor_t src,
                                               dst_tensor_t dst,
                                               tuple_t start_coord,
                                               tuple_t boundary,
                                               uint64_t flags) {
  return get_internal_ctx(ctx)->tile_max_reduce_wg(src, dst, start_coord,
                                                    boundary, team, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_max_rooted_reduce(rocshmem_ctx_t ctx,
                                                   rocshmem_team_t team,
                                                   src_tensor_t src,
                                                   dst_tensor_t dst,
                                                   tuple_t start_coord,
                                                   tuple_t boundary, int root,
                                                   uint64_t flags) {
  return get_internal_ctx(ctx)->tile_max_rooted_reduce(
      src, dst, start_coord, boundary, root, team, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_max_rooted_reduce_wave(
    rocshmem_ctx_t ctx, rocshmem_team_t team, src_tensor_t src,
    dst_tensor_t dst, tuple_t start_coord, tuple_t boundary, int root,
    uint64_t flags) {
  return get_internal_ctx(ctx)->tile_max_rooted_reduce_wave(
      src, dst, start_coord, boundary, root, team, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_max_rooted_reduce_wg(rocshmem_ctx_t ctx,
                                                      rocshmem_team_t team,
                                                      src_tensor_t src,
                                                      dst_tensor_t dst,
                                                      tuple_t start_coord,
                                                      tuple_t boundary, int root,
                                                      uint64_t flags) {
  return get_internal_ctx(ctx)->tile_max_rooted_reduce_wg(
      src, dst, start_coord, boundary, root, team, flags);
}

/******************************************************************************
 ************** MAX REDUCTIONS - DEFAULT CONTEXT WRAPPERS (6) *****************
 *****************************************************************************/

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_max_reduce(src_tensor_t src, dst_tensor_t dst,
                                        tuple_t start_coord, tuple_t boundary,
                                        rocshmem_team_t team, uint64_t flags) {
  return rocshmem_ctx_tile_max_reduce(ROCSHMEM_CTX_DEFAULT, team, src, dst,
                                      start_coord, boundary, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_max_reduce_wave(src_tensor_t src,
                                             dst_tensor_t dst,
                                             tuple_t start_coord,
                                             tuple_t boundary,
                                             rocshmem_team_t team,
                                             uint64_t flags) {
  return rocshmem_ctx_tile_max_reduce_wave(ROCSHMEM_CTX_DEFAULT, team, src,
                                           dst, start_coord, boundary, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_max_reduce_wg(src_tensor_t src, dst_tensor_t dst,
                                           tuple_t start_coord,
                                           tuple_t boundary,
                                           rocshmem_team_t team,
                                           uint64_t flags) {
  return rocshmem_ctx_tile_max_reduce_wg(ROCSHMEM_CTX_DEFAULT, team, src, dst,
                                         start_coord, boundary, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_max_rooted_reduce(src_tensor_t src,
                                               dst_tensor_t dst,
                                               tuple_t start_coord,
                                               tuple_t boundary, int pe_root,
                                               rocshmem_team_t team,
                                               uint64_t flags) {
  return rocshmem_ctx_tile_max_rooted_reduce(ROCSHMEM_CTX_DEFAULT, team, src,
                                             dst, start_coord, boundary,
                                             pe_root, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_max_rooted_reduce_wave(src_tensor_t src,
                                                    dst_tensor_t dst,
                                                    tuple_t start_coord,
                                                    tuple_t boundary,
                                                    int pe_root,
                                                    rocshmem_team_t team,
                                                    uint64_t flags) {
  return rocshmem_ctx_tile_max_rooted_reduce_wave(
      ROCSHMEM_CTX_DEFAULT, team, src, dst, start_coord, boundary, pe_root,
      flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_max_rooted_reduce_wg(src_tensor_t src,
                                                  dst_tensor_t dst,
                                                  tuple_t start_coord,
                                                  tuple_t boundary, int pe_root,
                                                  rocshmem_team_t team,
                                                  uint64_t flags) {
  return rocshmem_ctx_tile_max_rooted_reduce_wg(
      ROCSHMEM_CTX_DEFAULT, team, src, dst, start_coord, boundary, pe_root,
      flags);
}

/******************************************************************************
 ******************* MIN REDUCTIONS - CONTEXT VERSIONS (6) ********************
 *****************************************************************************/

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_min_reduce(rocshmem_ctx_t ctx,
                                            rocshmem_team_t team,
                                            src_tensor_t src, dst_tensor_t dst,
                                            tuple_t start_coord,
                                            tuple_t boundary, uint64_t flags) {
  return get_internal_ctx(ctx)->tile_min_reduce(src, dst, start_coord,
                                                 boundary, team, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_min_reduce_wave(rocshmem_ctx_t ctx,
                                                 rocshmem_team_t team,
                                                 src_tensor_t src,
                                                 dst_tensor_t dst,
                                                 tuple_t start_coord,
                                                 tuple_t boundary,
                                                 uint64_t flags) {
  return get_internal_ctx(ctx)->tile_min_reduce_wave(src, dst, start_coord,
                                                      boundary, team, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_min_reduce_wg(rocshmem_ctx_t ctx,
                                               rocshmem_team_t team,
                                               src_tensor_t src,
                                               dst_tensor_t dst,
                                               tuple_t start_coord,
                                               tuple_t boundary,
                                               uint64_t flags) {
  return get_internal_ctx(ctx)->tile_min_reduce_wg(src, dst, start_coord,
                                                    boundary, team, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_min_rooted_reduce(rocshmem_ctx_t ctx,
                                                   rocshmem_team_t team,
                                                   src_tensor_t src,
                                                   dst_tensor_t dst,
                                                   tuple_t start_coord,
                                                   tuple_t boundary, int root,
                                                   uint64_t flags) {
  return get_internal_ctx(ctx)->tile_min_rooted_reduce(
      src, dst, start_coord, boundary, root, team, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_min_rooted_reduce_wave(
    rocshmem_ctx_t ctx, rocshmem_team_t team, src_tensor_t src,
    dst_tensor_t dst, tuple_t start_coord, tuple_t boundary, int root,
    uint64_t flags) {
  return get_internal_ctx(ctx)->tile_min_rooted_reduce_wave(
      src, dst, start_coord, boundary, root, team, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_ctx_tile_min_rooted_reduce_wg(rocshmem_ctx_t ctx,
                                                      rocshmem_team_t team,
                                                      src_tensor_t src,
                                                      dst_tensor_t dst,
                                                      tuple_t start_coord,
                                                      tuple_t boundary, int root,
                                                      uint64_t flags) {
  return get_internal_ctx(ctx)->tile_min_rooted_reduce_wg(
      src, dst, start_coord, boundary, root, team, flags);
}

/******************************************************************************
 ************** MIN REDUCTIONS - DEFAULT CONTEXT WRAPPERS (6) *****************
 *****************************************************************************/

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_min_reduce(src_tensor_t src, dst_tensor_t dst,
                                        tuple_t start_coord, tuple_t boundary,
                                        rocshmem_team_t team, uint64_t flags) {
  return rocshmem_ctx_tile_min_reduce(ROCSHMEM_CTX_DEFAULT, team, src, dst,
                                      start_coord, boundary, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_min_reduce_wave(src_tensor_t src,
                                             dst_tensor_t dst,
                                             tuple_t start_coord,
                                             tuple_t boundary,
                                             rocshmem_team_t team,
                                             uint64_t flags) {
  return rocshmem_ctx_tile_min_reduce_wave(ROCSHMEM_CTX_DEFAULT, team, src,
                                           dst, start_coord, boundary, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_min_reduce_wg(src_tensor_t src, dst_tensor_t dst,
                                           tuple_t start_coord,
                                           tuple_t boundary,
                                           rocshmem_team_t team,
                                           uint64_t flags) {
  return rocshmem_ctx_tile_min_reduce_wg(ROCSHMEM_CTX_DEFAULT, team, src, dst,
                                         start_coord, boundary, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_min_rooted_reduce(src_tensor_t src,
                                               dst_tensor_t dst,
                                               tuple_t start_coord,
                                               tuple_t boundary, int pe_root,
                                               rocshmem_team_t team,
                                               uint64_t flags) {
  return rocshmem_ctx_tile_min_rooted_reduce(ROCSHMEM_CTX_DEFAULT, team, src,
                                             dst, start_coord, boundary,
                                             pe_root, flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_min_rooted_reduce_wave(src_tensor_t src,
                                                    dst_tensor_t dst,
                                                    tuple_t start_coord,
                                                    tuple_t boundary,
                                                    int pe_root,
                                                    rocshmem_team_t team,
                                                    uint64_t flags) {
  return rocshmem_ctx_tile_min_rooted_reduce_wave(
      ROCSHMEM_CTX_DEFAULT, team, src, dst, start_coord, boundary, pe_root,
      flags);
}

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ inline int rocshmem_tile_min_rooted_reduce_wg(src_tensor_t src,
                                                  dst_tensor_t dst,
                                                  tuple_t start_coord,
                                                  tuple_t boundary, int pe_root,
                                                  rocshmem_team_t team,
                                                  uint64_t flags) {
  return rocshmem_ctx_tile_min_rooted_reduce_wg(
      ROCSHMEM_CTX_DEFAULT, team, src, dst, start_coord, boundary, pe_root,
      flags);
}

}  // namespace rocshmem

#endif  // __HIP_DEVICE_COMPILE__

#endif  // LIBRARY_INCLUDE_ROCSHMEM_TILE_IMPL_HPP
