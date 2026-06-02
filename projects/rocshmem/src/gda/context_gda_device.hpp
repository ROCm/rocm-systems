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

#ifndef LIBRARY_SRC_GDA_CONTEXT_DEVICE_HPP_
#define LIBRARY_SRC_GDA_CONTEXT_DEVICE_HPP_

#include "context.hpp"
#include "team.hpp"
#include "queue_pair.hpp"

namespace rocshmem {

class QueuePair;

class GDAContext : public Context {
 public:
  __host__ GDAContext(Backend *b, unsigned int ctx_id);

  __host__ ~GDAContext();

  __device__ GDAContext(Backend *b, unsigned int ctx_id); //TODO is this used?

  __device__ void putmem(void *dest, const void *source, size_t nelems, int pe);

  __device__ void getmem(void *dest, const void *source, size_t nelems, int pe);

  __device__ void putmem_nbi(void *dest, const void *source, size_t nelems,
                             int pe);

  __device__ void getmem_nbi(void *dest, const void *source, size_t size,
                             int pe);

  __device__ void fence();

  __device__ void fence(int pe);

  __device__ void quiet();

  __device__ void pe_quiet(size_t pe);

  __device__ void *shmem_ptr(const void *dest, int pe);

  __device__ void barrier_all();

  __device__ void barrier_all_wave();

  __device__ void barrier_all_wg();

  __device__ void barrier(rocshmem_team_t team);

  __device__ void barrier_wave(rocshmem_team_t team);

  __device__ void barrier_wg(rocshmem_team_t team);

  __device__ void sync_all();

  __device__ void sync_all_wave();

  __device__ void sync_all_wg();

  __device__ void sync(rocshmem_team_t team);

  __device__ void sync_wave(rocshmem_team_t team);

  __device__ void sync_wg(rocshmem_team_t team);

  template <typename T>
  __device__ void p(T *dest, T value, int pe);

  template <typename T>
  __device__ void put(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ void put_nbi(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ T g(const T *source, int pe);

  template <typename T>
  __device__ void get(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ void get_nbi(T *dest, const T *source, size_t nelems, int pe);

  // Atomic operations
  template <typename T>
  __device__ void amo_add(void *dst, T value, int pe);

  template <typename T>
  __device__ void amo_set(void *dst, T value, int pe);

  template <typename T>
  __device__ T amo_swap(void *dst, T value, int pe);

  template <typename T>
  __device__ T amo_fetch_and(void *dst, T value, int pe);

  template <typename T>
  __device__ void amo_and(void *dst, T value, int pe);

  template <typename T>
  __device__ T amo_fetch_or(void *dst, T value, int pe);

  template <typename T>
  __device__ void amo_or(void *dst, T value, int pe);

  template <typename T>
  __device__ T amo_fetch_xor(void *dst, T value, int pe);

  template <typename T>
  __device__ void amo_xor(void *dst, T value, int pe);

  template <typename T>
  __device__ void amo_cas(void *dst, T value, T cond, int pe);

  template <typename T>
  __device__ T amo_fetch_add(void *dst, T value, int pe);

  template <typename T>
  __device__ T amo_fetch_cas(void *dst, T value, T cond, int pe);

  // Collectives
  template <typename T, ROCSHMEM_OP Op>
  __device__ int reduce(rocshmem_team_t team, T *dest, const T *source, int nreduce);

  template <typename T>
  __device__ void broadcast(rocshmem_team_t team, T *dest, const T *source,
                            int nelems, int pe_root);

  template <typename T>
  __device__ void alltoall(rocshmem_team_t team, T *dest, const T *source,
                           int nelems);

  template <typename T>
  __device__ void alltoallv(rocshmem_team_t team,
                            T *dest, const size_t dest_nelems[],
                            const size_t dest_displs[],
                            T *source, const size_t source_nelems[],
                            const size_t source_displs[]);

  template <typename T>
  __device__ void alltoallv_copy(rocshmem_team_t team,
                                 T *dest, const size_t dest_nelems[],
                                 const size_t dest_displs[],
                                 T *source, const size_t source_nelems[],
                                 const size_t source_displs[]);

  template <typename T>
  __device__ void alltoallv_get(rocshmem_team_t team,
                                T *dest, const size_t dest_nelems[],
                                const size_t dest_displs[],
                                T *source, const size_t source_nelems[],
                                const size_t source_displs[]);

  template <typename T>
  __device__ void fcollect(rocshmem_team_t team, T *dest, const T *source,
                           int nelems);


  // Block/wave functions
  __device__ void putmem_wg(void *dest, const void *source, size_t nelems,
                            int pe);

  __device__ void getmem_wg(void *dest, const void *source, size_t nelems,
                            int pe);

  __device__ void putmem_nbi_wg(void *dest, const void *source, size_t nelems,
                                int pe);

  __device__ void getmem_nbi_wg(void *dest, const void *source, size_t size,
                                int pe);

  __device__ void putmem_wave(void *dest, const void *source, size_t nelems,
                              int pe);

  __device__ void getmem_wave(void *dest, const void *source, size_t nelems,
                              int pe);

  __device__ void putmem_nbi_wave(void *dest, const void *source, size_t nelems,
                                  int pe);

  __device__ void getmem_nbi_wave(void *dest, const void *source, size_t size,
                                  int pe);

  template <typename T>
  __device__ void put_wg(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ void put_nbi_wg(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ void put_wave(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ void put_nbi_wave(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ void get_wg(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ void get_nbi_wg(T *dest, const T *source, size_t nelems, int pe);


  template <typename T>
  __device__ void get_wave(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ void get_nbi_wave(T *dest, const T *source, size_t nelems, int pe);

#define GDA_CONTEXT_PUT_SIGNAL_DEC(SUFFIX)                                               \
  template <typename T>                                                                  \
  __device__ void put_signal##SUFFIX(T *dest, const T *source, size_t nelems,            \
                                     uint64_t *sig_addr, uint64_t signal, int sig_op,    \
                                     int pe);                                            \
                                                                                         \
  __device__ void putmem_signal##SUFFIX(void *dest, const void *source, size_t nelems,   \
                                        uint64_t *sig_addr, uint64_t signal, int sig_op, \
                                        int pe);

  GDA_CONTEXT_PUT_SIGNAL_DEC()
  GDA_CONTEXT_PUT_SIGNAL_DEC(_wg)
  GDA_CONTEXT_PUT_SIGNAL_DEC(_wave)
  GDA_CONTEXT_PUT_SIGNAL_DEC(_nbi)
  GDA_CONTEXT_PUT_SIGNAL_DEC(_nbi_wg)
  GDA_CONTEXT_PUT_SIGNAL_DEC(_nbi_wave)

  __device__ uint64_t signal_fetch(const uint64_t *sig_addr);
  __device__ uint64_t signal_fetch_wg(const uint64_t *sig_addr);
  __device__ uint64_t signal_fetch_wave(const uint64_t *sig_addr);

 private:

  //internal functions used by collective operations
  template <typename T>
  __device__ void internal_broadcast(T *dest, const T *source, int nelems,
      int pe_root, int pe_start, int stride, int pe_size, long *p_sync);  // NOLINT(runtime/int)

  template <typename T>
  __device__ void internal_put_broadcast(T *dst, const T *src, int nelems,
      int pe_root, int PE_start, int logPE_stride, int PE_size,
      ActiveWFInfo &wf_info);  // NOLINT(runtime/int)

  template <typename T>
  __device__ void internal_get_broadcast(T *dst, const T *src, int nelems,
      int pe_root, ActiveWFInfo &wf_info);  // NOLINT(runtime/int)

  template <typename T>
  __device__ void fcollect_linear(rocshmem_team_t team, T *dest,
      const T *source, int nelems);

  template <typename T>
  __device__ void alltoall_linear(rocshmem_team_t team, T *dest,
    const T *source, int nelems);

  template <typename T>
  __device__ void alltoall_linear_thread_puts(rocshmem_team_t team, T *dest,
                                              const T *source, int nelems);

  __device__ void internal_sync(int pe, int PE_start, int stride, int PE_size,
      int64_t *pSync, ActiveWFInfo &wf_info);

  __device__ void internal_sync_wave(int pe, int PE_start, int stride,
      int PE_size, int64_t *pSync, ActiveWFInfo &wf_info);

  __device__ void internal_sync_wg(int pe, int PE_start, int stride,
    int PE_size, int64_t *pSync, ActiveWFInfo &wf_info);

  __device__ void internal_direct_barrier(int pe, int PE_start, int stride,
      int n_pes, int64_t *pSync, ActiveWFInfo &wf_info);

  __device__ void internal_direct_barrier_wg(int pe, int PE_start, int stride,
      int n_pes, int64_t *pSync, ActiveWFInfo &wf_info);

  __device__ void internal_atomic_barrier(int pe, int PE_start, int stride,
      int n_pes, int64_t *pSync, ActiveWFInfo &wf_info);

  template <typename T, ROCSHMEM_OP Op>
  __device__ void internal_direct_allreduce(T *dst, const T *src, int nelems,
      GDATeam *team_obj, ActiveWFInfo &wf_info);

  template <typename T, ROCSHMEM_OP Op>
  __device__ void internal_ring_allreduce(T *dst, const T *src, int nelems,
      GDATeam *team_obj, int n_seg, int seg_size, int chunk_size,
      ActiveWFInfo &wf_info);

  __device__ void internal_putmem(void *dest, const void *source,
      size_t nelems, int pe, int qp_index, ActiveWFInfo &wf_info);

  __device__ void internal_getmem(void *dest, const void *source,
      size_t nelems, int pe, int qp_index, ActiveWFInfo &wf_info);

  __device__ void internal_putmem_wg(void *dest, const void *source,
      size_t nelems, int pe, int qp_index, ActiveWFInfo &wf_info);

  __device__ void internal_getmem_wg(void *dest, const void *source,
      size_t nelems, int pe, int qp_index, ActiveWFInfo &wf_info);

  __device__ void internal_putmem_wave(void *dest, const void *source,
      size_t nelems, int pe, int qp_index, ActiveWFInfo &wf_info);

  __device__ void internal_getmem_wave(void *dest, const void *source,
      size_t nelems, int pe, int qp_index, ActiveWFInfo &wf_info);

  __device__ void internal_putmem_nbi(void *dest, const void *source,
      size_t nelems, int pe, int qp_index, ActiveWFInfo &wf_info);

  __device__ void internal_getmem_nbi(void *dest, const void *source,
      size_t nelems, int pe, int qp_index, ActiveWFInfo &wf_info);

  __device__ void internal_putmem_nbi_wg(void *dest, const void *source,
      size_t nelems, int pe, int qp_index, ActiveWFInfo &wf_info);

  __device__ void internal_getmem_nbi_wg(void *dest, const void *source,
      size_t nelems, int pe, int qp_index, ActiveWFInfo &wf_info);

  __device__ void internal_putmem_nbi_wave(void *dest, const void *source,
      size_t nelems, int pe, int qp_index, ActiveWFInfo &wf_info);

  __device__ void internal_getmem_nbi_wave(void *dest, const void *source,
      size_t nelems, int pe, int qp_index, ActiveWFInfo &wf_info);

  __device__
  void internal_quiet(ActiveWFInfo &wf_info);

  template <typename T>
  __device__ void internal_amo_add(void *dst, T value, int pe, int qp_index,
      ActiveWFInfo &wf_info);

  template <typename T>
  __device__ T internal_amo_fetch_add(void *dst, T value, int pe, int qp_index,
      ActiveWFInfo &wf_info);

  template <typename T>
  __device__ T internal_amo_swap(void *dst, T value, int pe, int qp_index,
      ActiveWFInfo &wf_info);

  /**
   * @brief Get the Queue Pair index to use for a given PE
   */
  __device__ __forceinline__ uint32_t get_qp_index(int pe, ActiveWFInfo wf_info);

  //Temporary scratchpad memory used by internal barrier algorithms.
  int64_t *barrier_sync{nullptr};

  /**
   * @brief Array containing the addresses of the work/sync buffer bases
   * of other PEs
  */
  char **wrk_sync_pool_bases_{nullptr};

  /**
   * @brief Device context Id
   */
  unsigned int ctx_id_{};

  /**
   * @brief Number of Queue Pairs allocated per PE
   */
  uint32_t num_qps_per_pe {1};

  /**
   * @brief Total number of Queue Pairs allocated = num_qps_per_pe * num_pes
   */
  uint32_t num_qps {1};

  /**
   * @brief Device pointer to the qp_counter variable to pick next qp index
   */
  uint32_t *qp_counter {nullptr};

 public:
  /**************************************************************************
   ****************** TILE API METHODS (NOT IMPLEMENTED) ********************
   *************************************************************************/

  // RMA PUT operations - Type-erased interface
  __device__ int tile_put(void* dst_data, const void* src_data,
                          const size_t* dst_strides, const size_t* src_strides,
                          const size_t* start_coord, const size_t* boundary,
                          int ndim, size_t element_size, int pe, uint64_t flags);

  __device__ int tile_put_wave(void* dst_data, const void* src_data,
                               const size_t* dst_strides, const size_t* src_strides,
                               const size_t* start_coord, const size_t* boundary,
                               int ndim, size_t element_size, int pe, uint64_t flags);

  __device__ int tile_put_wg(void* dst_data, const void* src_data,
                             const size_t* dst_strides, const size_t* src_strides,
                             const size_t* start_coord, const size_t* boundary,
                             int ndim, size_t element_size, int pe, uint64_t flags);

  // RMA GET operations - Type-erased interface
  __device__ int tile_get(void* dst_data, const void* src_data,
                          const size_t* dst_strides, const size_t* src_strides,
                          const size_t* start_coord, const size_t* boundary,
                          int ndim, size_t element_size, int pe, uint64_t flags);

  __device__ int tile_get_wave(void* dst_data, const void* src_data,
                               const size_t* dst_strides, const size_t* src_strides,
                               const size_t* start_coord, const size_t* boundary,
                               int ndim, size_t element_size, int pe, uint64_t flags);

  __device__ int tile_get_wg(void* dst_data, const void* src_data,
                             const size_t* dst_strides, const size_t* src_strides,
                             const size_t* start_coord, const size_t* boundary,
                             int ndim, size_t element_size, int pe, uint64_t flags);

  // Collective operations - Allgather - Type-erased interface
  __device__ int tile_allgather(rocshmem_team_t team, void* dst_data, const void* src_data,
                                const size_t* dst_strides, const size_t* src_strides,
                                const size_t* start_coord, const size_t* boundary,
                                int ndim, size_t element_size, uint64_t flags);

  __device__ int tile_allgather_wave(rocshmem_team_t team, void* dst_data, const void* src_data,
                                     const size_t* dst_strides, const size_t* src_strides,
                                     const size_t* start_coord, const size_t* boundary,
                                     int ndim, size_t element_size, uint64_t flags);

  __device__ int tile_allgather_wg(rocshmem_team_t team, void* dst_data, const void* src_data,
                                   const size_t* dst_strides, const size_t* src_strides,
                                   const size_t* start_coord, const size_t* boundary,
                                   int ndim, size_t element_size, uint64_t flags);

  // Collective operations - Broadcast - Type-erased interface
  __device__ int tile_broadcast(rocshmem_team_t team, void* dst_data, const void* src_data,
                                const size_t* dst_strides, const size_t* src_strides,
                                const size_t* start_coord, const size_t* boundary,
                                int ndim, size_t element_size, int pe_root, uint64_t flags);

  __device__ int tile_broadcast_wave(rocshmem_team_t team, void* dst_data, const void* src_data,
                                     const size_t* dst_strides, const size_t* src_strides,
                                     const size_t* start_coord, const size_t* boundary,
                                     int ndim, size_t element_size, int pe_root, uint64_t flags);

  __device__ int tile_broadcast_wg(rocshmem_team_t team, void* dst_data, const void* src_data,
                                   const size_t* dst_strides, const size_t* src_strides,
                                   const size_t* start_coord, const size_t* boundary,
                                   int ndim, size_t element_size, int pe_root, uint64_t flags);

  // Collective wait - No change needed
  __device__ int tile_collective_wait(rocshmem_team_t team, uint64_t flags);

  // SUM Reduction operations - Type-erased interface
  __device__ int tile_sum_reduce(rocshmem_team_t team, void* dst_data, const void* src_data,
                                 const size_t* dst_strides, const size_t* src_strides,
                                 const size_t* start_coord, const size_t* boundary,
                                 int ndim, size_t element_size, int root, uint64_t flags);

  __device__ int tile_sum_reduce_wave(rocshmem_team_t team, void* dst_data, const void* src_data,
                                      const size_t* dst_strides, const size_t* src_strides,
                                      const size_t* start_coord, const size_t* boundary,
                                      int ndim, size_t element_size, int root, uint64_t flags);

  __device__ int tile_sum_reduce_wg(rocshmem_team_t team, void* dst_data, const void* src_data,
                                    const size_t* dst_strides, const size_t* src_strides,
                                    const size_t* start_coord, const size_t* boundary,
                                    int ndim, size_t element_size, int root, uint64_t flags);

  // MAX Reduction operations - Type-erased interface
  __device__ int tile_max_reduce(rocshmem_team_t team, void* dst_data, const void* src_data,
                                 const size_t* dst_strides, const size_t* src_strides,
                                 const size_t* start_coord, const size_t* boundary,
                                 int ndim, size_t element_size, int root, uint64_t flags);

  __device__ int tile_max_reduce_wave(rocshmem_team_t team, void* dst_data, const void* src_data,
                                      const size_t* dst_strides, const size_t* src_strides,
                                      const size_t* start_coord, const size_t* boundary,
                                      int ndim, size_t element_size, int root, uint64_t flags);

  __device__ int tile_max_reduce_wg(rocshmem_team_t team, void* dst_data, const void* src_data,
                                    const size_t* dst_strides, const size_t* src_strides,
                                    const size_t* start_coord, const size_t* boundary,
                                    int ndim, size_t element_size, int root, uint64_t flags);

  // MIN Reduction operations - Type-erased interface
  __device__ int tile_min_reduce(rocshmem_team_t team, void* dst_data, const void* src_data,
                                 const size_t* dst_strides, const size_t* src_strides,
                                 const size_t* start_coord, const size_t* boundary,
                                 int ndim, size_t element_size, int root, uint64_t flags);

  __device__ int tile_min_reduce_wave(rocshmem_team_t team, void* dst_data, const void* src_data,
                                      const size_t* dst_strides, const size_t* src_strides,
                                      const size_t* start_coord, const size_t* boundary,
                                      int ndim, size_t element_size, int root, uint64_t flags);

  __device__ int tile_min_reduce_wg(rocshmem_team_t team, void* dst_data, const void* src_data,
                                    const size_t* dst_strides, const size_t* src_strides,
                                    const size_t* start_coord, const size_t* boundary,
                                    int ndim, size_t element_size, int root, uint64_t flags);

  // Rooted SUM Reduction operations
  // Rooted MAX Reduction operations
  // Rooted MIN Reduction operations
 public:
  QueuePair *qps{nullptr};

  /**
   * @brief Base heap pointers of all PEs
   */
  char *const *base_heap{nullptr};

  //TODO(Avinash):
  //Make tinfo private variable, it requires changes to the context
  //creation API in backend

  //Team information for the team associated with the context
  TeamInfo *tinfo{nullptr};
};

}  // namespace rocshmem

#endif // LIBRARY_SRC_GDA_CONTEXT_DEVICE_HPP_
