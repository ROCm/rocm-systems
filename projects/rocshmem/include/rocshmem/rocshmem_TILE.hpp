/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

#ifndef LIBRARY_INCLUDE_ROCSHMEM_TILE_HPP
#define LIBRARY_INCLUDE_ROCSHMEM_TILE_HPP

/**
 * @file rocshmem_TILE.hpp
 * @brief Tile-based granular API for rocSHMEM
 *
 * This file contains the tile-based API functions for rocSHMEM
 */

namespace rocshmem {

/******************************************************************************
 *************************** RMA PUT OPERATIONS *******************************
 *****************************************************************************/

/**
 * @brief Thread-granular tile put operation
 *
 * Writes a tile of data from source tensor to destination tensor at the
 * specified PE. Each thread operates independently on the tile region.
 *
 * @tparam src_tensor_t   Type of the source tensor
 * @tparam dst_tensor_t   Type of the destination tensor
 * @tparam tuple_t        Type of the coordinate tuple
 *
 * @param[in] src         Source tensor object
 * @param[in] dst         Destination tensor object
 * @param[in] start_coord Starting coordinates of the tile
 * @param[in] boundary    Boundary coordinates of the tile
 * @param[in] pe          PE of the remote process
 * @param[in] flags       Operation flags (reserved for future use)
 *
 * @return 0 on success, non-zero on failure
 */
template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ int rocshmem_tile_put(dst_tensor_t dst, const src_tensor_t src, tuple_t start_coord,
                                 tuple_t boundary, int pe, uint64_t flags);

/**
 * @brief Wave-granular tile put operation
 *
 * Writes a tile of data from source tensor to destination tensor at the
 * specified PE. All threads in the wave collectively participate in the
 * operation.
 *
 * @tparam src_tensor_t   Type of the source tensor
 * @tparam dst_tensor_t   Type of the destination tensor
 * @tparam tuple_t        Type of the coordinate tuple
 *
 * @param[in] src         Source tensor object
 * @param[in] dst         Destination tensor object
 * @param[in] start_coord Starting coordinates of the tile
 * @param[in] boundary    Boundary coordinates of the tile
 * @param[in] pe          PE of the remote process
 * @param[in] flags       Operation flags (reserved for future use)
 *
 * @return 0 on success, non-zero on failure
 */
template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ int rocshmem_tile_put_wave(dst_tensor_t dst, const src_tensor_t src,
                                      tuple_t start_coord, tuple_t boundary, int pe,
                                      uint64_t flags);

/**
 * @brief Workgroup-granular tile put operation
 *
 * Writes a tile of data from source tensor to destination tensor at the
 * specified PE. All threads in the workgroup collectively participate in the
 * operation.
 *
 * @tparam src_tensor_t   Type of the source tensor
 * @tparam dst_tensor_t   Type of the destination tensor
 * @tparam tuple_t        Type of the coordinate tuple
 *
 * @param[in] src         Source tensor object
 * @param[in] dst         Destination tensor object
 * @param[in] start_coord Starting coordinates of the tile
 * @param[in] boundary    Boundary coordinates of the tile
 * @param[in] pe          PE of the remote process
 * @param[in] flags       Operation flags (reserved for future use)
 *
 * @return 0 on success, non-zero on failure
 */
template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ int rocshmem_tile_put_wg(dst_tensor_t dst, const src_tensor_t src,
                                    tuple_t start_coord, tuple_t boundary, int pe,
                                    uint64_t flags);

/******************************************************************************
 *************************** RMA GET OPERATIONS *******************************
 *****************************************************************************/

/**
 * @brief Thread-granular tile get operation
 *
 * Reads a tile of data from source tensor at the specified PE to destination
 * tensor. Each thread operates independently on the tile region.
 *
 * @tparam src_tensor_t   Type of the source tensor
 * @tparam dst_tensor_t   Type of the destination tensor
 * @tparam tuple_t        Type of the coordinate tuple
 *
 * @param[out] dst        Destination tensor object on local PE
 * @param[in]  src        Source tensor object on remote PE
 * @param[in] start_coord Starting coordinates of the tile
 * @param[in] boundary    Boundary coordinates of the tile
 * @param[in] pe          PE of the remote process
 * @param[in] flags       Operation flags (reserved for future use)
 *
 * @return 0 on success, non-zero on failure
 */
template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ int rocshmem_tile_get(dst_tensor_t dst, src_tensor_t src, tuple_t start_coord,
                                 tuple_t boundary, int pe, uint64_t flags);

/**
 * @brief Wave-granular tile get operation
 *
 * Reads a tile of data from source tensor at the specified PE to destination
 * tensor. All threads in a wavefront (64 threads on AMD) collectively
 * participate in the operation.
 *
 * @tparam src_tensor_t   Type of the source tensor
 * @tparam dst_tensor_t   Type of the destination tensor
 * @tparam tuple_t        Type of the coordinate tuple
 *
 * @param[out] dst        Destination tensor object on local PE
 * @param[in]  src        Source tensor object on remote PE
 * @param[in] start_coord Starting coordinates of the tile
 * @param[in] boundary    Boundary coordinates of the tile
 * @param[in] pe          PE of the remote process
 * @param[in] flags       Operation flags (reserved for future use)
 *
 * @return 0 on success, non-zero on failure
 */
template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ int rocshmem_tile_get_wave(dst_tensor_t dst, src_tensor_t src,
                                      tuple_t start_coord, tuple_t boundary,
                                      int pe, uint64_t flags);

/**
 * @brief Workgroup-granular tile get operation
 *
 * Reads a tile of data from source tensor at the specified PE to destination
 * tensor. All threads in the workgroup collectively participate in the
 * operation.
 *
 * @tparam src_tensor_t   Type of the source tensor
 * @tparam dst_tensor_t   Type of the destination tensor
 * @tparam tuple_t        Type of the coordinate tuple
 *
 * @param[out] dst        Destination tensor object on local PE
 * @param[in]  src        Source tensor object on remote PE
 * @param[in] start_coord Starting coordinates of the tile
 * @param[in] boundary    Boundary coordinates of the tile
 * @param[in] pe          PE of the remote process
 * @param[in] flags       Operation flags (reserved for future use)
 *
 * @return 0 on success, non-zero on failure
 */
template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ int rocshmem_tile_get_wg(dst_tensor_t dst, src_tensor_t src,
                                    tuple_t start_coord, tuple_t boundary, int pe,
                                    uint64_t flags);

/******************************************************************************
 ************************** ALLGATHER OPERATIONS ******************************
 *****************************************************************************/

/**
 * @brief Thread-granular tile allgather operation
 *
 * Performs an allgather collective operation on a tile of data across all PEs
 * in the team. Each thread operates independently on the tile region.
 *
 * @tparam src_tensor_t   Type of the source tensor
 * @tparam dst_tensor_t   Type of the destination tensor
 * @tparam tuple_t        Type of the coordinate tuple
 *
 * @param[in] team        The team participating in the collective
 * @param[in] src         Source tensor object
 * @param[in] dst         Destination tensor object
 * @param[in] start_coord Starting coordinates of the tile
 * @param[in] boundary    Boundary coordinates of the tile
 * @param[in] flags       Operation flags (reserved for future use)
 *
 * @return 0 on success, non-zero on failure
 */
template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ int rocshmem_tile_allgather(rocshmem_team_t team, dst_tensor_t dst,
                                       const src_tensor_t src, tuple_t start_coord,
                                       tuple_t boundary, uint64_t flags);

/**
 * @brief Wave-granular tile allgather operation
 *
 * Performs an allgather collective operation on a tile of data across all PEs
 * in the team. All threads in the wave collectively participate in the
 * operation.
 *
 * @tparam src_tensor_t   Type of the source tensor
 * @tparam dst_tensor_t   Type of the destination tensor
 * @tparam tuple_t        Type of the coordinate tuple
 *
 * @param[in] team        The team participating in the collective
 * @param[in] src         Source tensor object
 * @param[in] dst         Destination tensor object
 * @param[in] start_coord Starting coordinates of the tile
 * @param[in] boundary    Boundary coordinates of the tile
 * @param[in] flags       Operation flags (reserved for future use)
 *
 * @return 0 on success, non-zero on failure
 */
template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ int rocshmem_tile_allgather_wave(rocshmem_team_t team, dst_tensor_t dst,
                                            const src_tensor_t src, tuple_t start_coord,
                                            tuple_t boundary, uint64_t flags);

/**
 * @brief Workgroup-granular tile allgather operation
 *
 * Performs an allgather collective operation on a tile of data across all PEs
 * in the team. All threads in the workgroup collectively participate in the
 * operation.
 *
 * @tparam src_tensor_t   Type of the source tensor
 * @tparam dst_tensor_t   Type of the destination tensor
 * @tparam tuple_t        Type of the coordinate tuple
 *
 * @param[in] team        The team participating in the collective
 * @param[in] src         Source tensor object
 * @param[in] dst         Destination tensor object
 * @param[in] start_coord Starting coordinates of the tile
 * @param[in] boundary    Boundary coordinates of the tile
 * @param[in] flags       Operation flags (reserved for future use)
 *
 * @return 0 on success, non-zero on failure
 */
template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ int rocshmem_tile_allgather_wg(rocshmem_team_t team, dst_tensor_t dst,
                                          const src_tensor_t src, tuple_t start_coord,
                                          tuple_t boundary, uint64_t flags);

/******************************************************************************
 ************************** BROADCAST OPERATIONS ******************************
 *****************************************************************************/

/**
 * @brief Thread-granular tile broadcast operation
 *
 * Broadcasts a tile of data from the root PE to all PEs in the team. Each
 * thread operates independently on the tile region.
 *
 * @tparam src_tensor_t   Type of the source tensor
 * @tparam dst_tensor_t   Type of the destination tensor
 * @tparam tuple_t        Type of the coordinate tuple
 *
 * @param[in] team        The team participating in the collective
 * @param[in] src         Source tensor object
 * @param[in] dst         Destination tensor object
 * @param[in] start_coord Starting coordinates of the tile
 * @param[in] boundary    Boundary coordinates of the tile
 * @param[in] flags       Operation flags (reserved for future use)
 *
 * @return 0 on success, non-zero on failure
 */
template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ int rocshmem_tile_broadcast(rocshmem_team_t team, dst_tensor_t dst,
                                       const src_tensor_t src, tuple_t start_coord,
                                       tuple_t boundary, int pe_root, uint64_t flags);

/**
 * @brief Wave-granular tile broadcast operation
 *
 * Broadcasts a tile of data from the root PE to all PEs in the team. All
 * threads in the wave collectively participate in the operation.
 *
 * @tparam src_tensor_t   Type of the source tensor
 * @tparam dst_tensor_t   Type of the destination tensor
 * @tparam tuple_t        Type of the coordinate tuple
 *
 * @param[in] team        The team participating in the collective
 * @param[in] src         Source tensor object
 * @param[in] dst         Destination tensor object
 * @param[in] start_coord Starting coordinates of the tile
 * @param[in] boundary    Boundary coordinates of the tile
 * @param[in] flags       Operation flags (reserved for future use)
 *
 * @return 0 on success, non-zero on failure
 */
template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ int rocshmem_tile_broadcast_wave(rocshmem_team_t team, dst_tensor_t dst,
                                            const src_tensor_t src, tuple_t start_coord,
                                            tuple_t boundary, int pe_root, uint64_t flags);

/**
 * @brief Workgroup-granular tile broadcast operation
 *
 * Broadcasts a tile of data from the root PE to all PEs in the team. All
 * threads in the workgroup collectively participate in the operation.
 *
 * @tparam src_tensor_t   Type of the source tensor
 * @tparam dst_tensor_t   Type of the destination tensor
 * @tparam tuple_t        Type of the coordinate tuple
 *
 * @param[in] team        The team participating in the collective
 * @param[in] src         Source tensor object
 * @param[in] dst         Destination tensor object
 * @param[in] start_coord Starting coordinates of the tile
 * @param[in] boundary    Boundary coordinates of the tile
 * @param[in] flags       Operation flags (reserved for future use)
 *
 * @return 0 on success, non-zero on failure
 */
template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ int rocshmem_tile_broadcast_wg(rocshmem_team_t team, dst_tensor_t dst,
                                          const src_tensor_t src, tuple_t start_coord,
                                          tuple_t boundary, int pe_root, uint64_t flags);

/******************************************************************************
 ********************* COLLECTIVE WAIT OPERATIONS *****************************
 *****************************************************************************/

/**
 * @brief Wait for completion of tile collective operations
 *
 * Blocks until all outstanding tile collective operations on the specified
 * team have completed.
 *
 * @param[in] team  The team to wait on
 * @param[in] flags Operation flags (reserved for future use)
 *
 * @return 0 on success, non-zero on failure
 */
__device__ int rocshmem_tile_collective_wait(rocshmem_team_t team, uint64_t flags);

/******************************************************************************
 ************************** REDUCTION OPERATIONS ******************************
 *****************************************************************************/

// SUM Reductions

/**
 * @brief Thread-granular tile sum reduction operation
 *
 * Performs a sum reduction on a tile of data across all PEs in the team.
 * The result is stored on all PEs. Each thread operates independently on the
 * tile region.
 *
 * @tparam src_tensor_t   Type of the source tensor
 * @tparam dst_tensor_t   Type of the destination tensor
 * @tparam tuple_t        Type of the coordinate tuple
 *
 * @param[in] team        The team participating in the collective
 * @param[in] src         Source tensor object
 * @param[in] dst         Destination tensor object
 * @param[in] start_coord Starting coordinates of the tile
 * @param[in] boundary    Boundary coordinates of the tile
 * @param[in] root        Root PE for the reduction (relative to team)
 * @param[in] flags       Operation flags (reserved for future use)
 *
 * @return 0 on success, non-zero on failure
 */
template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ int rocshmem_tile_sum_reduce(rocshmem_team_t team, dst_tensor_t dst,
                                        const src_tensor_t src, tuple_t start_coord,
                                        tuple_t boundary, int root, uint64_t flags);

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ int rocshmem_tile_sum_reduce_wave(rocshmem_team_t team, dst_tensor_t dst,
                                             const src_tensor_t src, tuple_t start_coord,
                                             tuple_t boundary, int root, uint64_t flags);

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ int rocshmem_tile_sum_reduce_wg(rocshmem_team_t team, dst_tensor_t dst,
                                           const src_tensor_t src, tuple_t start_coord,
                                           tuple_t boundary, int root, uint64_t flags);

// MAX Reductions

/**
 * @brief Thread-granular tile max reduction operation
 *
 * Performs a max reduction on a tile of data across all PEs in the team.
 * The result is stored on all PEs. Each thread operates independently on the
 * tile region.
 *
 * @tparam src_tensor_t   Type of the source tensor
 * @tparam dst_tensor_t   Type of the destination tensor
 * @tparam tuple_t        Type of the coordinate tuple
 *
 * @param[in] team        The team participating in the collective
 * @param[in] src         Source tensor object
 * @param[in] dst         Destination tensor object
 * @param[in] start_coord Starting coordinates of the tile
 * @param[in] boundary    Boundary coordinates of the tile
 * @param[in] root        Root PE for the reduction (relative to team)
 * @param[in] flags       Operation flags (reserved for future use)
 *
 * @return 0 on success, non-zero on failure
 */
template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ int rocshmem_tile_max_reduce(rocshmem_team_t team, dst_tensor_t dst,
                                        const src_tensor_t src, tuple_t start_coord,
                                        tuple_t boundary, int root, uint64_t flags);

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ int rocshmem_tile_max_reduce_wave(rocshmem_team_t team, dst_tensor_t dst,
                                             const src_tensor_t src, tuple_t start_coord,
                                             tuple_t boundary, int root, uint64_t flags);

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ int rocshmem_tile_max_reduce_wg(rocshmem_team_t team, dst_tensor_t dst,
                                           const src_tensor_t src, tuple_t start_coord,
                                           tuple_t boundary, int root, uint64_t flags);

// MIN Reductions

/**
 * @brief Thread-granular tile min reduction operation
 *
 * Performs a min reduction on a tile of data across all PEs in the team.
 * The result is stored on all PEs. Each thread operates independently on the
 * tile region.
 *
 * @tparam src_tensor_t   Type of the source tensor
 * @tparam dst_tensor_t   Type of the destination tensor
 * @tparam tuple_t        Type of the coordinate tuple
 *
 * @param[in] team        The team participating in the collective
 * @param[in] src         Source tensor object
 * @param[in] dst         Destination tensor object
 * @param[in] start_coord Starting coordinates of the tile
 * @param[in] boundary    Boundary coordinates of the tile
 * @param[in] root        Root PE for the reduction (relative to team)
 * @param[in] flags       Operation flags (reserved for future use)
 *
 * @return 0 on success, non-zero on failure
 */
template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ int rocshmem_tile_min_reduce(rocshmem_team_t team, dst_tensor_t dst,
                                        const src_tensor_t src, tuple_t start_coord,
                                        tuple_t boundary, int root, uint64_t flags);

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ int rocshmem_tile_min_reduce_wave(rocshmem_team_t team, dst_tensor_t dst,
                                             const src_tensor_t src, tuple_t start_coord,
                                             tuple_t boundary, int root, uint64_t flags);

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ int rocshmem_tile_min_reduce_wg(rocshmem_team_t team, dst_tensor_t dst,
                                           const src_tensor_t src, tuple_t start_coord,
                                           tuple_t boundary, int root, uint64_t flags);

/******************************************************************************
 ****************** CONTEXT VERSIONS OF TILE API FUNCTIONS ********************
 *****************************************************************************/

/******************************************************************************
 ********************* RMA PUT OPERATIONS (CTX VERSIONS) **********************
 *****************************************************************************/

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ int rocshmem_ctx_tile_put(rocshmem_ctx_t ctx, dst_tensor_t dst, const src_tensor_t src,
                                     tuple_t start_coord, tuple_t boundary, int pe,
                                     uint64_t flags);

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ int rocshmem_ctx_tile_put_wave(rocshmem_ctx_t ctx, dst_tensor_t dst,
                                          const src_tensor_t src, tuple_t start_coord,
                                          tuple_t boundary, int pe, uint64_t flags);

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ int rocshmem_ctx_tile_put_wg(rocshmem_ctx_t ctx, dst_tensor_t dst, const src_tensor_t src,
                                        tuple_t start_coord, tuple_t boundary, int pe,
                                        uint64_t flags);

/******************************************************************************
 ********************* RMA GET OPERATIONS (CTX VERSIONS) **********************
 *****************************************************************************/

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ int rocshmem_ctx_tile_get(rocshmem_ctx_t ctx, dst_tensor_t dst, src_tensor_t src,
                                     tuple_t start_coord, tuple_t boundary, int pe,
                                     uint64_t flags);

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ int rocshmem_ctx_tile_get_wave(rocshmem_ctx_t ctx, dst_tensor_t dst, src_tensor_t src,
                                          tuple_t start_coord, tuple_t boundary, int pe,
                                          uint64_t flags);

template <typename src_tensor_t, typename dst_tensor_t, typename tuple_t>
__device__ int rocshmem_ctx_tile_get_wg(rocshmem_ctx_t ctx, dst_tensor_t dst, src_tensor_t src,
                                        tuple_t start_coord, tuple_t boundary, int pe,
                                        uint64_t flags);

/******************************************************************************
 ****************** COLLECTIVE ALLGATHER (CTX VERSIONS) ***********************
 *****************************************************************************/

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ int rocshmem_ctx_tile_allgather(rocshmem_ctx_t ctx, rocshmem_team_t team,
                                           dst_tensor_t dst, const src_tensor_t src,
                                           tuple_t start_coord, tuple_t boundary,
                                           uint64_t flags);

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ int rocshmem_ctx_tile_allgather_wave(rocshmem_ctx_t ctx, rocshmem_team_t team,
                                                dst_tensor_t dst, const src_tensor_t src,
                                                tuple_t start_coord, tuple_t boundary,
                                                uint64_t flags);

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ int rocshmem_ctx_tile_allgather_wg(rocshmem_ctx_t ctx, rocshmem_team_t team,
                                              dst_tensor_t dst, const src_tensor_t src,
                                              tuple_t start_coord, tuple_t boundary,
                                              uint64_t flags);

/******************************************************************************
 ****************** COLLECTIVE BROADCAST (CTX VERSIONS) ***********************
 *****************************************************************************/

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ int rocshmem_ctx_tile_broadcast(rocshmem_ctx_t ctx, rocshmem_team_t team,
                                           dst_tensor_t dst, const src_tensor_t src,
                                           tuple_t start_coord, tuple_t boundary,
                                           int pe_root, uint64_t flags);

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ int rocshmem_ctx_tile_broadcast_wave(rocshmem_ctx_t ctx, rocshmem_team_t team,
                                                dst_tensor_t dst, const src_tensor_t src,
                                                tuple_t start_coord, tuple_t boundary,
                                                int pe_root, uint64_t flags);

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ int rocshmem_ctx_tile_broadcast_wg(rocshmem_ctx_t ctx, rocshmem_team_t team,
                                              dst_tensor_t dst, const src_tensor_t src,
                                              tuple_t start_coord, tuple_t boundary,
                                              int pe_root, uint64_t flags);

/******************************************************************************
 ******************** COLLECTIVE WAIT (CTX VERSION) ***************************
 *****************************************************************************/

__device__ int rocshmem_ctx_tile_collective_wait(rocshmem_ctx_t ctx, rocshmem_team_t team,
                                                 uint64_t flags);

/******************************************************************************
 ******************** SUM REDUCTIONS (CTX VERSIONS) ***************************
 *****************************************************************************/

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ int rocshmem_ctx_tile_sum_reduce(rocshmem_ctx_t ctx, rocshmem_team_t team,
                                            dst_tensor_t dst, const src_tensor_t src,
                                            tuple_t start_coord, tuple_t boundary,
                                            int root, uint64_t flags);

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ int rocshmem_ctx_tile_sum_reduce_wave(rocshmem_ctx_t ctx, rocshmem_team_t team,
                                                 dst_tensor_t dst, const src_tensor_t src,
                                                 tuple_t start_coord, tuple_t boundary,
                                                 int root, uint64_t flags);

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ int rocshmem_ctx_tile_sum_reduce_wg(rocshmem_ctx_t ctx, rocshmem_team_t team,
                                               dst_tensor_t dst, const src_tensor_t src,
                                               tuple_t start_coord, tuple_t boundary,
                                               int root, uint64_t flags);

/******************************************************************************
 ******************** MAX REDUCTIONS (CTX VERSIONS) ***************************
 *****************************************************************************/

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ int rocshmem_ctx_tile_max_reduce(rocshmem_ctx_t ctx, rocshmem_team_t team,
                                            dst_tensor_t dst, const src_tensor_t src,
                                            tuple_t start_coord, tuple_t boundary,
                                            int root, uint64_t flags);

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ int rocshmem_ctx_tile_max_reduce_wave(rocshmem_ctx_t ctx, rocshmem_team_t team,
                                                 dst_tensor_t dst, const src_tensor_t src,
                                                 tuple_t start_coord, tuple_t boundary,
                                                 int root, uint64_t flags);

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ int rocshmem_ctx_tile_max_reduce_wg(rocshmem_ctx_t ctx, rocshmem_team_t team,
                                               dst_tensor_t dst, const src_tensor_t src,
                                               tuple_t start_coord, tuple_t boundary,
                                               int root, uint64_t flags);

/******************************************************************************
 ******************** MIN REDUCTIONS (CTX VERSIONS) ***************************
 *****************************************************************************/

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ int rocshmem_ctx_tile_min_reduce(rocshmem_ctx_t ctx, rocshmem_team_t team,
                                            dst_tensor_t dst, const src_tensor_t src,
                                            tuple_t start_coord, tuple_t boundary,
                                            int root, uint64_t flags);

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ int rocshmem_ctx_tile_min_reduce_wave(rocshmem_ctx_t ctx, rocshmem_team_t team,
                                                 dst_tensor_t dst, const src_tensor_t src,
                                                 tuple_t start_coord, tuple_t boundary,
                                                 int root, uint64_t flags);

template <typename dst_tensor_t, typename src_tensor_t, typename tuple_t>
__device__ int rocshmem_ctx_tile_min_reduce_wg(rocshmem_ctx_t ctx, rocshmem_team_t team,
                                               dst_tensor_t dst, const src_tensor_t src,
                                               tuple_t start_coord, tuple_t boundary,
                                               int root, uint64_t flags);

}  // namespace rocshmem

// Note: Template implementations are in rocshmem_TILE_impl.hpp
// That header is included automatically by rocshmem_tile_gpu.cpp when building the library
// and should be explicitly included by device test/application code that uses the Tile API

#endif  // LIBRARY_INCLUDE_ROCSHMEM_TILE_HPP
