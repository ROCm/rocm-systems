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

#ifndef LIBRARY_SRC_REVERSE_OFFLOAD_CONTEXT_RO_DEVICE_HPP_
#define LIBRARY_SRC_REVERSE_OFFLOAD_CONTEXT_RO_DEVICE_HPP_

#include "context.hpp"
#include "block_handle.hpp"
#include "commands_types.hpp"
#include "queue.hpp"

namespace rocshmem {

__device__ void build_queue_element(
    ro_net_cmds type, void *dst, void *src, size_t size, int pe,
    int logPE_stride, int PE_size, int PE_root, void *pWrk, long *pSync,
    intptr_t team_comm, int ro_net_win_id, BlockHandle *handle,
    bool blocking, volatile char *status = nullptr, bool default_ctx = false,
    ROCSHMEM_OP op = ROCSHMEM_SUM, ro_net_types datatype = RO_NET_INT);

class ROContext : public Context {
 public:
  __host__ ROContext(Backend *b, size_t block_id = 0, bool default_ctx = false);

  __device__ void ctx_destroy();

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
  __device__ T g(const T *source, int pe);

  template <typename T, ROCSHMEM_OP Op>
  __device__ int reduce(rocshmem_team_t team, T *dest, const T *source,
                        int nreduce);

  template <typename T>
  __device__ void put(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ void put_nbi(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ void get(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ void get_nbi(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ T amo_fetch_cas(void *dst, T value, T cond, int pe);

  template <typename T>
  __device__ void amo_cas(void *dst, T value, T cond, int pe);

  template <typename T>
  __device__ T amo_fetch_add(void *dst, T value, int pe);

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
  __device__ void broadcast(rocshmem_team_t team, T *dest, const T *source,
                            int nelems, int pe_root);

  template <typename T>
  __device__ void broadcast(T *dest, const T *source, int nelems, int pe_root,
                            int pe_start, int log_pe_stride, int pe_size,
                            long *p_sync);  // NOLINT(runtime/int)

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
  __device__ void fcollect(rocshmem_team_t team, T *dest, const T *source,
                           int nelems);

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
  __device__ void get_wg(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ void get_nbi_wg(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ void put_wave(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ void put_nbi_wave(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ void get_wave(T *dest, const T *source, size_t nelems, int pe);

  template <typename T>
  __device__ void get_nbi_wave(T *dest, const T *source, size_t nelems, int pe);

#define RO_CONTEXT_PUT_SIGNAL_DEC(SUFFIX)                                                \
  template <typename T>                                                                  \
  __device__ void put_signal##SUFFIX(T *dest, const T *source, size_t nelems,            \
                                     uint64_t *sig_addr, uint64_t signal, int sig_op,    \
                                     int pe);                                            \
                                                                                         \
  __device__ void putmem_signal##SUFFIX(void *dest, const void *source, size_t nelems,   \
                                        uint64_t *sig_addr, uint64_t signal, int sig_op, \
                                        int pe);

  RO_CONTEXT_PUT_SIGNAL_DEC()
  RO_CONTEXT_PUT_SIGNAL_DEC(_wg)
  RO_CONTEXT_PUT_SIGNAL_DEC(_wave)
  RO_CONTEXT_PUT_SIGNAL_DEC(_nbi)
  RO_CONTEXT_PUT_SIGNAL_DEC(_nbi_wg)
  RO_CONTEXT_PUT_SIGNAL_DEC(_nbi_wave)

  __device__ uint64_t signal_fetch(const uint64_t *sig_addr);
  __device__ uint64_t signal_fetch_wg(const uint64_t *sig_addr);
  __device__ uint64_t signal_fetch_wave(const uint64_t *sig_addr);

 private:
  __device__ uint64_t *get_atomic_ret_buf();

  __device__ uint64_t *get_g_ret_buf();

  __device__ volatile char *get_status_flag();

  BlockHandle *block_handle{nullptr};

  int ro_net_win_id{-1};

  bool is_default_ctx{false};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_REVERSE_OFFLOAD_CONTEXT_RO_DEVICE_HPP_
