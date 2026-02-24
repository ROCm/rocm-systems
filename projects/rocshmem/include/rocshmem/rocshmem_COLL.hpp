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

#ifndef LIBRARY_INCLUDE_ROCSHMEM_COLL_HPP
#define LIBRARY_INCLUDE_ROCSHMEM_COLL_HPP

namespace rocshmem {

/**
 * @name SHMEM_ALLTOALL
 * @brief Exchanges a fixed amount of contiguous data blocks between all pairs
 * of PEs participating in the collective routine.
 *
 * This function must be called as a work-group collective.
 *
 * @param[in] team         The team participating in the collective.
 * @param[in] dest         Destination address. Must be an address on the
 *                         symmetric heap.
 * @param[in] source       Source address. Must be an address on the symmetric
                           heap.
 * @param[in] nelems       Number of data blocks transferred per pair of PEs.
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_float_alltoall_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, float *dest,
    const float *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_double_alltoall_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, double *dest,
    const double *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_char_alltoall_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, char *dest,
    const char *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_schar_alltoall_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, signed char *dest,
    const signed char *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_short_alltoall_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, short *dest,
    const short *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int_alltoall_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, int *dest,
    const int *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_long_alltoall_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long *dest,
    const long *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_longlong_alltoall_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long long *dest,
    const long long *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uchar_alltoall_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned char *dest,
    const unsigned char *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ushort_alltoall_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned short *dest,
    const unsigned short *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint_alltoall_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned int *dest,
    const unsigned int *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulong_alltoall_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned long *dest,
    const unsigned long *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulonglong_alltoall_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned long long *dest,
    const unsigned long long *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_float_alltoall_wg(
    rocshmem_team_t team, float *dest, const float *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_double_alltoall_wg(
    rocshmem_team_t team, double *dest, const double *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_char_alltoall_wg(
    rocshmem_team_t team, char *dest, const char *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_schar_alltoall_wg(
    rocshmem_team_t team, signed char *dest, const signed char *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_short_alltoall_wg(
    rocshmem_team_t team, short *dest, const short *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_int_alltoall_wg(
    rocshmem_team_t team, int *dest, const int *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_long_alltoall_wg(
    rocshmem_team_t team, long *dest, const long *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_longlong_alltoall_wg(
    rocshmem_team_t team, long long *dest, const long long *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_uchar_alltoall_wg(
    rocshmem_team_t team, unsigned char *dest, const unsigned char *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ushort_alltoall_wg(
    rocshmem_team_t team, unsigned short *dest, const unsigned short *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_uint_alltoall_wg(
    rocshmem_team_t team, unsigned int *dest, const unsigned int *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ulong_alltoall_wg(
    rocshmem_team_t team, unsigned long *dest, const unsigned long *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ulonglong_alltoall_wg(
    rocshmem_team_t team, unsigned long long *dest, const unsigned long long *source, int nelems);

/**
 * @name SHMEM_ALLTOALLV
 * @brief PE i sends source_nelems[j] of data from source + source_displs[j] to PE j.
 * At the same time, PE i receives dest_nelems[j] of data from PE j to be placed at dest + dest_displs[j].
 *
 * This function must be called as a work-group collective.
 *
 * @param[in] team          The team participating in the collective.
 * @param[in] dest          Destination address. Must be an address on the symmetric heap.
 * @param[in] dest_nelems   Array containing number of elements to receive from each participating PE
 * @param[in] dest_displs   Array of offsets into dest buffer for each participating PE
 * @param[in] source        Source address. Must be an address on the symmetric heap.
 * @param[in] source_nelems Array containing number of elements to send from each participating PE
 * @param[in] source_displs Array of offsets into source buffer for each participating PE
 *
 * @return void
 */

__device__ ATTR_NO_INLINE void rocshmem_float_alltoallv_wg(rocshmem_team_t team,
                                                           float *dest, const size_t dest_nelems[],
                                                           const size_t dest_displs[],
                                                           float *source, const size_t source_nelems[],
                                                           const size_t source_displs[]);

__device__ ATTR_NO_INLINE void rocshmem_double_alltoallv_wg(rocshmem_team_t team,
                                                            double *dest, const size_t dest_nelems[],
                                                            const size_t dest_displs[],
                                                            double *source, const size_t source_nelems[],
                                                            const size_t source_displs[]);

__device__ ATTR_NO_INLINE void rocshmem_char_alltoallv_wg(rocshmem_team_t team,
                                                          char *dest, const size_t dest_nelems[],
                                                          const size_t dest_displs[],
                                                          char *source, const size_t source_nelems[],
                                                          const size_t source_displs[]);

__device__ ATTR_NO_INLINE void rocshmem_schar_alltoallv_wg(rocshmem_team_t team,
                                                           signed char *dest, const size_t dest_nelems[],
                                                           const size_t dest_displs[],
                                                           signed char *source, const size_t source_nelems[],
                                                           const size_t source_displs[]);

__device__ ATTR_NO_INLINE void rocshmem_short_alltoallv_wg(rocshmem_team_t team,
                                                           short *dest, const size_t dest_nelems[],
                                                           const size_t dest_displs[],
                                                           short *source, const size_t source_nelems[],
                                                           const size_t source_displs[]);

__device__ ATTR_NO_INLINE void rocshmem_int_alltoallv_wg(rocshmem_team_t team,
                                                         int *dest, const size_t dest_nelems[],
                                                         const size_t dest_displs[],
                                                         int *source, const size_t source_nelems[],
                                                         const size_t source_displs[]);

__device__ ATTR_NO_INLINE void rocshmem_long_alltoallv_wg(rocshmem_team_t team,
                                                          long *dest, const size_t dest_nelems[],
                                                          const size_t dest_displs[],
                                                          long *source, const size_t source_nelems[],
                                                          const size_t source_displs[]);

__device__ ATTR_NO_INLINE void rocshmem_longlong_alltoallv_wg(rocshmem_team_t team,
                                                              long long *dest, const size_t dest_nelems[],
                                                              const size_t dest_displs[],
                                                              long long *source, const size_t source_nelems[],
                                                              const size_t source_displs[]);

__device__ ATTR_NO_INLINE void rocshmem_uchar_alltoallv_wg(rocshmem_team_t team,
                                                           unsigned char *dest, const size_t dest_nelems[],
                                                           const size_t dest_displs[],
                                                           unsigned char *source, const size_t source_nelems[],
                                                           const size_t source_displs[]);

__device__ ATTR_NO_INLINE void rocshmem_ushort_alltoallv_wg(rocshmem_team_t team,
                                                            unsigned short *dest, const size_t dest_nelems[],
                                                            const size_t dest_displs[],
                                                            unsigned short *source, const size_t source_nelems[],
                                                            const size_t source_displs[]);

__device__ ATTR_NO_INLINE void rocshmem_uint_alltoallv_wg(rocshmem_team_t team,
                                                          unsigned int *dest, const size_t dest_nelems[],
                                                          const size_t dest_displs[],
                                                          unsigned int *source, const size_t source_nelems[],
                                                          const size_t source_displs[]);

__device__ ATTR_NO_INLINE void rocshmem_ulong_alltoallv_wg(rocshmem_team_t team,
                                                           unsigned long *dest, const size_t dest_nelems[],
                                                           const size_t dest_displs[],
                                                           unsigned long *source, const size_t source_nelems[],
                                                           const size_t source_displs[]);

__device__ ATTR_NO_INLINE void rocshmem_ulonglong_alltoallv_wg(rocshmem_team_t team,
                                                               unsigned long long *dest, const size_t dest_nelems[],
                                                               const size_t dest_displs[],
                                                               unsigned long long *source, const size_t source_nelems[],
                                                               const size_t source_displs[]);

/**
 * @name SHMEM_BROADCAST
 * @brief Perform a broadcast between PEs in the active set. The caller
 * is blocked until the broadcase completes.
 *
 * This function must be called as a work-group collective.
 *
 * @param[in] dest         Destination address. Must be an address on the
 *                         symmetric heap.
 * @param[in] source       Source address. Must be an address on the symmetric
                           heap.
 * @param[in] nelement     Size of the buffer to participate in the broadcast.
 * @param[in] PE_root      Zero-based ordinal of the PE, with respect to the
                           active set, from which the data is copied
 * @param[in] PE_start     PE to start the reduction.
 * @param[in] logPE_stride Stride of PEs participating in the reduction.
 * @param[in] PE_size      Number PEs participating in the reduction.
 * @param[in] pSync        Temporary sync buffer provided to ROCSHMEM. Must
                           be of size at least ROCSHMEM_REDUCE_SYNC_SIZE.
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_float_broadcast_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, float *dest,
    const float *source, int nelems, int pe_root);
__host__ void rocshmem_ctx_float_broadcast(
    rocshmem_ctx_t ctx, float *dest, const float *source,
    int nelems, int pe_root, int pe_start, int log_pe_stride,
    int pe_size, long *p_sync);
__host__ void rocshmem_ctx_float_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, float *dest,
    const float *source, int nelems, int pe_root);

__device__ ATTR_NO_INLINE void rocshmem_ctx_double_broadcast_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, double *dest,
    const double *source, int nelems, int pe_root);
__host__ void rocshmem_ctx_double_broadcast(
    rocshmem_ctx_t ctx, double *dest, const double *source,
    int nelems, int pe_root, int pe_start, int log_pe_stride,
    int pe_size, long *p_sync);
__host__ void rocshmem_ctx_double_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, double *dest,
    const double *source, int nelems, int pe_root);

__device__ ATTR_NO_INLINE void rocshmem_ctx_char_broadcast_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, char *dest,
    const char *source, int nelems, int pe_root);
__host__ void rocshmem_ctx_char_broadcast(
    rocshmem_ctx_t ctx, char *dest, const char *source,
    int nelems, int pe_root, int pe_start, int log_pe_stride,
    int pe_size, long *p_sync);
__host__ void rocshmem_ctx_char_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, char *dest,
    const char *source, int nelems, int pe_root);

__device__ ATTR_NO_INLINE void rocshmem_ctx_schar_broadcast_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, signed char *dest,
    const signed char *source, int nelems, int pe_root);
__host__ void rocshmem_ctx_schar_broadcast(
    rocshmem_ctx_t ctx, signed char *dest, const signed char *source,
    int nelems, int pe_root, int pe_start, int log_pe_stride,
    int pe_size, long *p_sync);
__host__ void rocshmem_ctx_schar_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, signed char *dest,
    const signed char *source, int nelems, int pe_root);

__device__ ATTR_NO_INLINE void rocshmem_ctx_short_broadcast_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, short *dest,
    const short *source, int nelems, int pe_root);
__host__ void rocshmem_ctx_short_broadcast(
    rocshmem_ctx_t ctx, short *dest, const short *source,
    int nelems, int pe_root, int pe_start, int log_pe_stride,
    int pe_size, long *p_sync);
__host__ void rocshmem_ctx_short_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, short *dest,
    const short *source, int nelems, int pe_root);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int_broadcast_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, int *dest,
    const int *source, int nelems, int pe_root);
__host__ void rocshmem_ctx_int_broadcast(
    rocshmem_ctx_t ctx, int *dest, const int *source,
    int nelems, int pe_root, int pe_start, int log_pe_stride,
    int pe_size, long *p_sync);
__host__ void rocshmem_ctx_int_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, int *dest,
    const int *source, int nelems, int pe_root);

__device__ ATTR_NO_INLINE void rocshmem_ctx_long_broadcast_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long *dest,
    const long *source, int nelems, int pe_root);
__host__ void rocshmem_ctx_long_broadcast(
    rocshmem_ctx_t ctx, long *dest, const long *source,
    int nelems, int pe_root, int pe_start, int log_pe_stride,
    int pe_size, long *p_sync);
__host__ void rocshmem_ctx_long_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long *dest,
    const long *source, int nelems, int pe_root);

__device__ ATTR_NO_INLINE void rocshmem_ctx_longlong_broadcast_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long long *dest,
    const long long *source, int nelems, int pe_root);
__host__ void rocshmem_ctx_longlong_broadcast(
    rocshmem_ctx_t ctx, long long *dest, const long long *source,
    int nelems, int pe_root, int pe_start, int log_pe_stride,
    int pe_size, long *p_sync);
__host__ void rocshmem_ctx_longlong_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long long *dest,
    const long long *source, int nelems, int pe_root);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uchar_broadcast_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned char *dest,
    const unsigned char *source, int nelems, int pe_root);
__host__ void rocshmem_ctx_uchar_broadcast(
    rocshmem_ctx_t ctx, unsigned char *dest, const unsigned char *source,
    int nelems, int pe_root, int pe_start, int log_pe_stride,
    int pe_size, long *p_sync);
__host__ void rocshmem_ctx_uchar_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned char *dest,
    const unsigned char *source, int nelems, int pe_root);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ushort_broadcast_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned short *dest,
    const unsigned short *source, int nelems, int pe_root);
__host__ void rocshmem_ctx_ushort_broadcast(
    rocshmem_ctx_t ctx, unsigned short *dest, const unsigned short *source,
    int nelems, int pe_root, int pe_start, int log_pe_stride,
    int pe_size, long *p_sync);
__host__ void rocshmem_ctx_ushort_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned short *dest,
    const unsigned short *source, int nelems, int pe_root);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint_broadcast_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned int *dest,
    const unsigned int *source, int nelems, int pe_root);
__host__ void rocshmem_ctx_uint_broadcast(
    rocshmem_ctx_t ctx, unsigned int *dest, const unsigned int *source,
    int nelems, int pe_root, int pe_start, int log_pe_stride,
    int pe_size, long *p_sync);
__host__ void rocshmem_ctx_uint_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned int *dest,
    const unsigned int *source, int nelems, int pe_root);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulong_broadcast_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned long *dest,
    const unsigned long *source, int nelems, int pe_root);
__host__ void rocshmem_ctx_ulong_broadcast(
    rocshmem_ctx_t ctx, unsigned long *dest, const unsigned long *source,
    int nelems, int pe_root, int pe_start, int log_pe_stride,
    int pe_size, long *p_sync);
__host__ void rocshmem_ctx_ulong_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned long *dest,
    const unsigned long *source, int nelems, int pe_root);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulonglong_broadcast_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned long long *dest,
    const unsigned long long *source, int nelems, int pe_root);
__host__ void rocshmem_ctx_ulonglong_broadcast(
    rocshmem_ctx_t ctx, unsigned long long *dest, const unsigned long long *source,
    int nelems, int pe_root, int pe_start, int log_pe_stride,
    int pe_size, long *p_sync);
__host__ void rocshmem_ctx_ulonglong_broadcast(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned long long *dest,
    const unsigned long long *source, int nelems, int pe_root);


/**
 * @name SHMEM_FCOLLECT
 * @brief Concatenates blocks of data from multiple PEs to an array in every
 * PE participating in the collective routine.
 *
 * This function must be called as a work-group collective.
 *
 * @param[in] team         The team participating in the collective.
 * @param[in] dest         Destination address. Must be an address on the
 *                         symmetric heap.
 * @param[in] source       Source address. Must be an address on the symmetric
                           heap.
 * @param[in] nelems       Number of data blocks in source array.
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_float_fcollect_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, float *dest,
    const float *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_double_fcollect_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, double *dest,
    const double *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_char_fcollect_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, char *dest,
    const char *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_schar_fcollect_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, signed char *dest,
    const signed char *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_short_fcollect_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, short *dest,
    const short *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_int_fcollect_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, int *dest,
    const int *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_long_fcollect_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long *dest,
    const long *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_longlong_fcollect_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long long *dest,
    const long long *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uchar_fcollect_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned char *dest,
    const unsigned char *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ushort_fcollect_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned short *dest,
    const unsigned short *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_uint_fcollect_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned int *dest,
    const unsigned int *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulong_fcollect_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned long *dest,
    const unsigned long *source, int nelems);

__device__ ATTR_NO_INLINE void rocshmem_ctx_ulonglong_fcollect_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, unsigned long long *dest,
    const unsigned long long *source, int nelems);


/**
 * @name SHMEM_REDUCTIONS
 * @brief Perform an allreduce between PEs in the active set. The caller
 * is blocked until the reduction completes.
 *
 * This function must be called as a work-group collective.
 *
 * @param[in] team         The team participating in the collective.
 * @param[in] dest         Destination address. Must be an address on the
 *                         symmetric heap.
 * @param[in] source       Source address. Must be an address on the symmetric
                           heap.
 * @param[in] nreduce      Size of the buffer to participate in the reduction.
 *
 * @return int (Zero on successful local completion. Nonzero otherwise.)
 */
__device__ ATTR_NO_INLINE int rocshmem_ctx_short_sum_reduce_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, short *dest, const short *source,
    int nreduce);
__host__ int rocshmem_ctx_short_sum_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, short *dest, const short *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_short_min_reduce_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, short *dest, const short *source,
    int nreduce);
__host__ int rocshmem_ctx_short_min_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, short *dest, const short *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_short_max_reduce_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, short *dest, const short *source,
    int nreduce);
__host__ int rocshmem_ctx_short_max_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, short *dest, const short *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_short_prod_reduce_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, short *dest, const short *source,
    int nreduce);
__host__ int rocshmem_ctx_short_prod_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, short *dest, const short *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_short_or_reduce_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, short *dest, const short *source,
    int nreduce);
__host__ int rocshmem_ctx_short_or_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, short *dest, const short *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_short_and_reduce_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, short *dest, const short *source,
    int nreduce);
__host__ int rocshmem_ctx_short_and_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, short *dest, const short *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_short_xor_reduce_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, short *dest, const short *source,
    int nreduce);
__host__ int rocshmem_ctx_short_xor_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, short *dest, const short *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_int_sum_reduce_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, int *dest, const int *source,
    int nreduce);
__host__ int rocshmem_ctx_int_sum_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, int *dest, const int *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_int_min_reduce_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, int *dest, const int *source,
    int nreduce);
__host__ int rocshmem_ctx_int_min_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, int *dest, const int *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_int_max_reduce_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, int *dest, const int *source,
    int nreduce);
__host__ int rocshmem_ctx_int_max_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, int *dest, const int *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_int_prod_reduce_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, int *dest, const int *source,
    int nreduce);
__host__ int rocshmem_ctx_int_prod_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, int *dest, const int *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_int_or_reduce_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, int *dest, const int *source,
    int nreduce);
__host__ int rocshmem_ctx_int_or_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, int *dest, const int *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_int_and_reduce_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, int *dest, const int *source,
    int nreduce);
__host__ int rocshmem_ctx_int_and_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, int *dest, const int *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_int_xor_reduce_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, int *dest, const int *source,
    int nreduce);
__host__ int rocshmem_ctx_int_xor_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, int *dest, const int *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_long_sum_reduce_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long *dest, const long *source,
    int nreduce);
__host__ int rocshmem_ctx_long_sum_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long *dest, const long *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_long_min_reduce_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long *dest, const long *source,
    int nreduce);
__host__ int rocshmem_ctx_long_min_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long *dest, const long *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_long_max_reduce_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long *dest, const long *source,
    int nreduce);
__host__ int rocshmem_ctx_long_max_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long *dest, const long *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_long_prod_reduce_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long *dest, const long *source,
    int nreduce);
__host__ int rocshmem_ctx_long_prod_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long *dest, const long *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_long_or_reduce_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long *dest, const long *source,
    int nreduce);
__host__ int rocshmem_ctx_long_or_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long *dest, const long *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_long_and_reduce_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long *dest, const long *source,
    int nreduce);
__host__ int rocshmem_ctx_long_and_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long *dest, const long *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_long_xor_reduce_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long *dest, const long *source,
    int nreduce);
__host__ int rocshmem_ctx_long_xor_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long *dest, const long *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_longlong_sum_reduce_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long long *dest, const long long *source,
    int nreduce);
__host__ int rocshmem_ctx_longlong_sum_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long long *dest, const long long *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_longlong_min_reduce_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long long *dest, const long long *source,
    int nreduce);
__host__ int rocshmem_ctx_longlong_min_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long long *dest, const long long *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_longlong_max_reduce_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long long *dest, const long long *source,
    int nreduce);
__host__ int rocshmem_ctx_longlong_max_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long long *dest, const long long *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_longlong_prod_reduce_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long long *dest, const long long *source,
    int nreduce);
__host__ int rocshmem_ctx_longlong_prod_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long long *dest, const long long *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_longlong_or_reduce_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long long *dest, const long long *source,
    int nreduce);
__host__ int rocshmem_ctx_longlong_or_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long long *dest, const long long *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_longlong_and_reduce_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long long *dest, const long long *source,
    int nreduce);
__host__ int rocshmem_ctx_longlong_and_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long long *dest, const long long *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_longlong_xor_reduce_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long long *dest, const long long *source,
    int nreduce);
__host__ int rocshmem_ctx_longlong_xor_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, long long *dest, const long long *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_float_sum_reduce_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, float *dest, const float *source,
    int nreduce);
__host__ int rocshmem_ctx_float_sum_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, float *dest, const float *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_float_min_reduce_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, float *dest, const float *source,
    int nreduce);
__host__ int rocshmem_ctx_float_min_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, float *dest, const float *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_float_max_reduce_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, float *dest, const float *source,
    int nreduce);
__host__ int rocshmem_ctx_float_max_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, float *dest, const float *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_float_prod_reduce_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, float *dest, const float *source,
    int nreduce);
__host__ int rocshmem_ctx_float_prod_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, float *dest, const float *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_double_sum_reduce_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, double *dest, const double *source,
    int nreduce);
__host__ int rocshmem_ctx_double_sum_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, double *dest, const double *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_double_min_reduce_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, double *dest, const double *source,
    int nreduce);
__host__ int rocshmem_ctx_double_min_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, double *dest, const double *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_double_max_reduce_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, double *dest, const double *source,
    int nreduce);
__host__ int rocshmem_ctx_double_max_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, double *dest, const double *source,
    int nreduce);

__device__ ATTR_NO_INLINE int rocshmem_ctx_double_prod_reduce_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team, double *dest, const double *source,
    int nreduce);
__host__ int rocshmem_ctx_double_prod_reduce(
    rocshmem_ctx_t ctx, rocshmem_team_t team, double *dest, const double *source,
    int nreduce);

/**
 * @brief kernel for performing a barrier synchronization.
 * Caller enqueues the kernel on given stream
 *
 * @return void
 */
__global__ ATTR_NO_INLINE void rocshmem_barrier_all_kernel();

/**
 * @brief kernel for performing an alltoall collective operation.
 * Caller enqueues the kernel on given stream
 *
 * @param[in] team    The team participating in the collective.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] size    Number of bytes to transfer per pair of PEs.
 *
 * @return void
 */
__global__ ATTR_NO_INLINE void rocshmem_alltoallmem_kernel(rocshmem_team_t team,
                                                           void *dest,
                                                           const void *source,
                                                           size_t size);

/**
 * @brief kernel for performing a broadcast collective operation.
 * Caller enqueues the kernel on given stream
 *
 * @param[in] team    The team participating in the collective.
 * @param[in] dest    Destination address. Must be an address on the symmetric
 *                    heap.
 * @param[in] source  Source address. Must be an address on the symmetric heap.
 * @param[in] nelems  Number of bytes to broadcast.
 * @param[in] pe_root Root PE (relative to team) from which to broadcast.
 *
 * @return void
 */
__global__ ATTR_NO_INLINE void rocshmem_broadcastmem_kernel(
    rocshmem_team_t team, void *dest, const void *source, size_t nelems,
    int pe_root);

/**
 * @brief perform a collective barrier between all PEs in the system.
 * The caller is blocked until the barrier is resolved.
 *
 * This function must be invoked by a single thread within the PE.
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_barrier_all();

/**
 * @brief perform a collective barrier between all PEs in the system.
 * The caller is blocked until the barrier is resolved.
 *
 * This function must be called as a wave-front collective.
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_barrier_all_wave();

/**
 * @brief perform a collective barrier between all PEs in the system.
 * The caller is blocked until the barrier is resolved.
 *
 * This function must be called as a work-group collective.
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_barrier_all_wg();

/**
 * @brief perform a collective barrier between all PEs in the team.
 * The caller is blocked until the barrier is resolved.
 *
 * This function must be invoked by a single thread within the PE.
 *
 * @param[in] handle GPU side handle.
 *
 * @param[in] team The team on which to perform barrier synchronization
 *
 * @return void
 */
__device__ void rocshmem_ctx_barrier(rocshmem_ctx_t ctx, rocshmem_team_t team);

/**
 * @brief perform a collective barrier between all PEs in the team.
 * The caller is blocked until the barrier is resolved.
 *
 * This function must be called as a wave-front collective.
 *
 * @param[in] handle GPU side handle.
 *
 * @param[in] team The team on which to perform barrier synchronization
 *
 * @return void
 */
__device__ void rocshmem_ctx_barrier_wave(rocshmem_ctx_t ctx, rocshmem_team_t team);

/**
 * @brief perform a collective barrier between all PEs in the team.
 * The caller is blocked until the barrier is resolved.
 *
 * This function must be called as a work-group collective.
 *
 * @param[in] handle GPU side handle.
 *
 * @param[in] team The team on which to perform barrier synchronization
 *
 * @return void
 */
__device__ void rocshmem_ctx_barrier_wg(rocshmem_ctx_t ctx, rocshmem_team_t team);

/**
 * @brief registers the arrival of a PE at a barrier.
 * The caller is blocked until the synchronization is resolved.
 *
 * In contrast with the shmem_barrier_all routine, shmem_sync_all only ensures
 * completion and visibility of previously issued memory stores and does not
 * ensure completion of remote memory updates issued via OpenSHMEM routines.
 *
 * This function must be invoked by a single thread within the PE.
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_sync_all();

/**
 * @brief registers the arrival of a PE at a barrier.
 * The caller is blocked until the synchronization is resolved.
 *
 * In contrast with the shmem_barrier_all routine, shmem_sync_all only ensures
 * completion and visibility of previously issued memory stores and does not
 * ensure completion of remote memory updates issued via OpenSHMEM routines.
 *
 * This function must be called as a wave-front collective.
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_sync_all_wave();

/**
 * @brief registers the arrival of a PE at a barrier.
 * The caller is blocked until the synchronization is resolved.
 *
 * In contrast with the shmem_barrier_all routine, shmem_sync_all only ensures
 * completion and visibility of previously issued memory stores and does not
 * ensure completion of remote memory updates issued via OpenSHMEM routines.
 *
 * This function must be called as a work-group collective.
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_sync_all_wg();

/**
 * @brief registers the arrival of a PE at a barrier.
 * The caller is blocked until the synchronization is resolved.
 *
 * In contrast with the shmem_barrier_all routine, shmem_team_sync only ensures
 * completion and visibility of previously issued memory stores and does not
 * ensure completion of remote memory updates issued via OpenSHMEM routines.
 *
 * This function must be invoked by a single thread within the PE.
 *
 * @param[in] handle GPU side handle.
 * @param[in] team  Handle of the team being synchronized
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_sync(
    rocshmem_ctx_t ctx, rocshmem_team_t team);

/**
 * @brief registers the arrival of a PE at a barrier.
 * The caller is blocked until the synchronization is resolved.
 *
 * In contrast with the shmem_barrier_all routine, shmem_team_sync only ensures
 * completion and visibility of previously issued memory stores and does not
 * ensure completion of remote memory updates issued via OpenSHMEM routines.
 *
 * This function must be called as a wave-front collective.
 *
 * @param[in] handle GPU side handle.
 * @param[in] team  Handle of the team being synchronized
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_sync_wave(
    rocshmem_ctx_t ctx, rocshmem_team_t team);

/**
 * @brief registers the arrival of a PE at a barrier.
 * The caller is blocked until the synchronization is resolved.
 *
 * In contrast with the shmem_barrier_all routine, shmem_team_sync only ensures
 * completion and visibility of previously issued memory stores and does not
 * ensure completion of remote memory updates issued via OpenSHMEM routines.
 *
 * This function must be called as a work-group collective.
 *
 * @param[in] handle GPU side handle.
 * @param[in] team  Handle of the team being synchronized
 *
 * @return void
 */
__device__ ATTR_NO_INLINE void rocshmem_ctx_sync_wg(
    rocshmem_ctx_t ctx, rocshmem_team_t team);

}  // namespace rocshmem

#endif  // LIBRARY_INCLUDE_ROCSHMEM_COLL_HPP
