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

#ifndef LIBRARY_SRC_CONTEXT_HPP_
#define LIBRARY_SRC_CONTEXT_HPP_

#include <hip/hip_runtime.h>

#include "backend_type.hpp"
#include "host/host.hpp"
#include "ipc_policy.hpp"
#include "stats.hpp"
#include "wf_coal_policy.hpp"

namespace rocshmem {

class Backend;

/**
 * @file context.hpp
 * @brief Context class corresponds directly to an OpenSHMEM context.
 *
 * GPUs perform networking operations on a context that is created by the
 * application programmer or a "default context" managed by the runtime.
 *
 * Contexts can be allocated in shared memory, in which case they are private
 * to the creating workgroup, or they can be allocated in global memory, in
 * which case they are shareable across workgroups.
 *
 * This is an 'abstract' class, as much as there is such a thing on a GPU.
 * It uses 'type' to dispatch to a derived class for most of the interesting
 * behavior.
 */
class Context {
 public:
  __host__ Context(Backend* handle);

  __device__ Context(Backend* handle);

  __host__ virtual ~Context();

  /*
   * Dispatch functions to get runtime polymorphism without 'virtual' or
   * function pointers. Each one of these guys will use 'type' to
   * static_cast themselves and dispatch to the appropriate derived class.
   * It's basically doing part of what the 'virtual' keyword does, so when
   * we get that working in ROCm it will be super easy to adapt to it by
   * just removing the dispatch implementations.
   *
   * No comments for these guys since its basically the same as in the
   * rocshmem.hpp public header.
   */

  /**************************************************************************
   ***************************** DEVICE METHODS *****************************
   *************************************************************************/
  template <typename T>
  __device__ void wait_until(T *ivars, int cmp, T val);

  template <typename T>
  __device__ void wait_until_all(T *ivars, size_t nelems,
                                 const int *status,
                                 int cmp, T val);

  template <typename T>
  __device__ size_t wait_until_any(T *ivars, size_t nelems,
                                   const int *status,
                                   int cmp, T val);

  template <typename T>
  __device__ size_t wait_until_some(T *ivars, size_t nelems,
                                    size_t* indices,
                                    const int *status,
                                    int cmp, T val);

  template <typename T>
  __device__ void wait_until_all_vector(T *ivars, size_t nelems,
                                        const int *status,
                                        int cmp, T* vals);

  template <typename T>
  __device__ size_t wait_until_any_vector(T *ivars, size_t nelems,
                                          const int *status,
                                          int cmp, T* vals);

  template <typename T>
  __device__ size_t wait_until_some_vector(T *ivars, size_t nelems,
                                           size_t* indices,
                                           const int *status,
                                           int cmp, T* vals);

  template <typename T>
  __device__ int test(T *ivars, int cmp, T val);

  __device__ void threadfence_system();

  __device__ void ctx_create();

  __device__ void ctx_destroy();

  __device__ void putmem(void* dest, const void* source, size_t nelems, int pe);

  __device__ void getmem(void* dest, const void* source, size_t nelems, int pe);

  __device__ void putmem_nbi(void* dest, const void* source, size_t nelems,
                             int pe);

  __device__ void getmem_nbi(void* dest, const void* source, size_t size,
                             int pe);

  __device__ void fence();

  __device__ void fence(int pe);

  __device__ void quiet();

  __device__ void pe_quiet(size_t pe);

  __device__ void* shmem_ptr(const void* dest, int pe);

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
  __device__ T amo_fetch(void* dst, T value, T cond, int pe, uint8_t atomic_op);

  template <typename T>
  __device__ void amo_add(void* dst, T value, int pe);

  template <typename T>
  __device__ void amo_set(void* dst, T value, int pe);

  template <typename T>
  __device__ T amo_swap(void* dst, T value, int pe);

  template <typename T>
  __device__ T amo_fetch_and(void* dst, T value, int pe);

  template <typename T>
  __device__ void amo_and(void* dst, T value, int pe);

  template <typename T>
  __device__ T amo_fetch_or(void* dst, T value, int pe);

  template <typename T>
  __device__ void amo_or(void* dst, T value, int pe);

  template <typename T>
  __device__ T amo_fetch_xor(void* dst, T value, int pe);

  template <typename T>
  __device__ void amo_xor(void* dst, T value, int pe);

  template <typename T>
  __device__ void amo_cas(void* dst, T value, T cond, int pe);

  template <typename T>
  __device__ T amo_fetch_add(void* dst, T value, int pe);

  template <typename T>
  __device__ T amo_fetch_cas(void* dst, T value, T cond, int pe);

  template <typename T>
  __device__ void p(T* dest, T value, int pe);

  template <typename T>
  __device__ T g(T* source, int pe);

  template <typename T, ROCSHMEM_OP Op>
  __device__ void to_all(T* dest, const T* source, int nreduce, int PE_start,
                         int logPE_stride, int PE_size, T* pWrk,
                         long* pSync);  // NOLINT(runtime/int)

  template <typename T, ROCSHMEM_OP Op>
  __device__ int reduce(rocshmem_team_t team, T* dest, const T* source, int nreduce);

  template <typename T>
  __device__ void put(T* dest, const T* source, size_t nelems, int pe);

  template <typename T>
  __device__ void put_nbi(T* dest, const T* source, size_t nelems, int pe);

  template <typename T>
  __device__ void get(T* dest, const T* source, size_t nelems, int pe);

  template <typename T>
  __device__ void get_nbi(T* dest, const T* source, size_t nelems, int pe);

  template <typename T>
  __device__ void alltoall(rocshmem_team_t team, T* dest, const T* source,
                           int nelems);

  template <typename T>
  __device__ void fcollect(rocshmem_team_t team, T* dest, const T* source,
                           int nelems);

  template <typename T>
  __device__ void broadcast(rocshmem_team_t team, T* dest, const T* source,
                            int nelems, int pe_root);

  template <typename T>
  __device__ void broadcast(T* dest, const T* source, int nelems, int pe_root,
                            int pe_start, int log_pe_stride, int pe_size,
                            long* p_sync);  // NOLINT(runtime/int)

  __device__ void putmem_wg(void* dest, const void* source, size_t nelems,
                            int pe);

  __device__ void getmem_wg(void* dest, const void* source, size_t nelems,
                            int pe);

  __device__ void putmem_nbi_wg(void* dest, const void* source, size_t nelems,
                                int pe);

  __device__ void getmem_nbi_wg(void* dest, const void* source, size_t size,
                                int pe);

  __device__ void putmem_wave(void* dest, const void* source, size_t nelems,
                              int pe);

  __device__ void getmem_wave(void* dest, const void* source, size_t nelems,
                              int pe);

  __device__ void putmem_nbi_wave(void* dest, const void* source, size_t nelems,
                                  int pe);

  __device__ void getmem_nbi_wave(void* dest, const void* source, size_t size,
                                  int pe);

  template <typename T>
  __device__ void put_wg(T* dest, const T* source, size_t nelems, int pe);

  template <typename T>
  __device__ void put_nbi_wg(T* dest, const T* source, size_t nelems, int pe);

  template <typename T>
  __device__ void get_wg(T* dest, const T* source, size_t nelems, int pe);

  template <typename T>
  __device__ void get_nbi_wg(T* dest, const T* source, size_t nelems, int pe);

  template <typename T>
  __device__ void put_wave(T* dest, const T* source, size_t nelems, int pe);

  template <typename T>
  __device__ void put_nbi_wave(T* dest, const T* source, size_t nelems, int pe);

  template <typename T>
  __device__ void get_wave(T* dest, const T* source, size_t nelems, int pe);

  template <typename T>
  __device__ void get_nbi_wave(T* dest, const T* source, size_t nelems, int pe);

#define CONTEXT_PUTMEM_SIGNAL_DEC(SUFFIX)                                              \
  __device__ void putmem_signal##SUFFIX(void *dest, const void *source, size_t nelems, \
                                        uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);

  CONTEXT_PUTMEM_SIGNAL_DEC()
  CONTEXT_PUTMEM_SIGNAL_DEC(_wg)
  CONTEXT_PUTMEM_SIGNAL_DEC(_wave)
  CONTEXT_PUTMEM_SIGNAL_DEC(_nbi)
  CONTEXT_PUTMEM_SIGNAL_DEC(_nbi_wg)
  CONTEXT_PUTMEM_SIGNAL_DEC(_nbi_wave)

#define CONTEXT_PUT_SIGNAL_DEC(SUFFIX)                                        \
  template <typename T>                                                       \
  __device__ void put_signal##SUFFIX(T *dest, const T *source, size_t nelems, \
                                     uint64_t *sig_addr, uint64_t signal, int sig_op, int pe);

  CONTEXT_PUT_SIGNAL_DEC()
  CONTEXT_PUT_SIGNAL_DEC(_wg)
  CONTEXT_PUT_SIGNAL_DEC(_wave)
  CONTEXT_PUT_SIGNAL_DEC(_nbi)
  CONTEXT_PUT_SIGNAL_DEC(_nbi_wg)
  CONTEXT_PUT_SIGNAL_DEC(_nbi_wave)

  __device__ uint64_t signal_fetch(const uint64_t *sig_addr);
  __device__ uint64_t signal_fetch_wg(const uint64_t *sig_addr);
  __device__ uint64_t signal_fetch_wave(const uint64_t *sig_addr);

  /**************************************************************************
   ****************************** HOST METHODS ******************************
   *************************************************************************/
  template <typename T>
  __host__ void p(T* dest, T value, int pe);

  template <typename T>
  __host__ T g(const T* source, int pe);

  template <typename T>
  __host__ void put(T* dest, const T* source, size_t nelems, int pe);

  template <typename T>
  __host__ void get(T* dest, const T* source, size_t nelems, int pe);

  template <typename T>
  __host__ void put_nbi(T* dest, const T* source, size_t nelems, int pe);

  template <typename T>
  __host__ void get_nbi(T* dest, const T* source, size_t nelems, int pe);

  __host__ void putmem(void* dest, const void* source, size_t nelems, int pe);

  __host__ void getmem(void* dest, const void* source, size_t nelems, int pe);

  __host__ void putmem_nbi(void* dest, const void* source, size_t nelems,
                           int pe);

  __host__ void getmem_nbi(void* dest, const void* source, size_t size, int pe);

  template <typename T>
  __host__ void amo_add(void* dst, T value, int pe);

  template <typename T>
  __host__ void amo_set(void* dst, T value, int pe);

  template <typename T>
  __host__ T amo_swap(void* dst, T value, int pe);

  template <typename T>
  __host__ T amo_fetch_and(void* dst, T value, int pe);

  template <typename T>
  __host__ void amo_and(void* dst, T value, int pe);

  template <typename T>
  __host__ T amo_fetch_or(void* dst, T value, int pe);

  template <typename T>
  __host__ void amo_or(void* dst, T value, int pe);

  template <typename T>
  __host__ T amo_fetch_xor(void* dst, T value, int pe);

  template <typename T>
  __host__ void amo_xor(void* dst, T value, int pe);

  template <typename T>
  __host__ void amo_cas(void* dst, T value, T cond, int pe);

  template <typename T>
  __host__ T amo_fetch_add(void* dst, T value, int pe);

  template <typename T>
  __host__ T amo_fetch_cas(void* dst, T value, T cond, int pe);

  __host__ void fence();

  __host__ void quiet();

  __host__ void* shmem_ptr(const void* dest, int pe);

  __host__ void barrier_all();

  __host__ void barrier_all_on_stream(hipStream_t stream);

  __host__ void alltoallmem_on_stream(rocshmem_team_t team, void *dest,
                                      const void *source, size_t size,
                                      hipStream_t stream);

  __host__ void broadcastmem_on_stream(rocshmem_team_t team, void *dest,
                                       const void *source, size_t nelems,
                                       int pe_root, hipStream_t stream);

  __host__ void getmem_on_stream(void *dest, const void *source, size_t nelems,
                                 int pe, hipStream_t stream);

  __host__ void putmem_on_stream(void *dest, const void *source, size_t nelems,
                                 int pe, hipStream_t stream);

  __host__ void putmem_signal_on_stream(void *dest, const void *source,
                                        size_t nelems, uint64_t *sig_addr,
                                        uint64_t signal, int sig_op, int pe,
                                        hipStream_t stream);

  __host__ void signal_wait_until_on_stream(uint64_t *sig_addr, int cmp,
                                            uint64_t cmp_value,
                                            hipStream_t stream);

  __host__ void sync_all();

  template <typename T>
  __host__ void broadcast(T* dest, const T* source, int nelems, int pe_root,
                          int pe_start, int log_pe_stride, int pe_size,
                          long* p_sync);  // NOLINT(runtime/int)

  template <typename T>
  __host__ void broadcast(rocshmem_team_t team, T* dest, const T* source,
                          int nelems, int pe_root);

  template <typename T, ROCSHMEM_OP Op>
  __host__ void to_all(T* dest, const T* source, int nreduce, int PE_start,
                       int logPE_stride, int PE_size, T* pWrk,
                       long* pSync);  // NOLINT(runtime/int)

  template <typename T, ROCSHMEM_OP Op>
  __host__ int reduce(rocshmem_team_t team, T* dest, const T* source, int nreduce);

  template <typename T>
  __host__ void wait_until(T *ivars, int cmp, T val);

  template <typename T>
  __host__ void wait_until_all(T *ivars, size_t nelems,
                               const int *status,
                               int cmp, T val);

  template <typename T>
  __host__ size_t wait_until_any(T *ivars, size_t nelems,
                                 const int *status,
                                 int cmp, T val);

  template <typename T>
  __host__ size_t wait_until_some(T *ivars, size_t nelems,
                                  size_t* indices,
                                  const int *status,
                                  int cmp, T val);

  template <typename T>
  __host__ void wait_until_all_vector(T *ivars, size_t nelems,
                                      const int *status,
                                      int cmp, T* vals);

  template <typename T>
  __host__ size_t wait_until_any_vector(T *ivars, size_t nelems,
                                        const int *status,
                                        int cmp, T* vals);

  template <typename T>
  __host__ size_t wait_until_some_vector(T *ivars, size_t nelems,
                                         size_t* indices,
                                         const int *status,
                                         int cmp, T* vals);

  template <typename T>
  __host__ int test(T *ivars, int cmp, T val);

 public:
  /**************************************************************************
   ***************************** PUBLIC MEMBERS *****************************
   *************************************************************************/
  /**
   * @brief Duplicated local copy of backend's num_pes
   */
  int num_pes{0};

  /**
   * @brief Duplicated local copy of backend's my_pe
   */
  int my_pe{-1};

  /**
   * @brief Duplicated local copy of backend's type
   */
  BackendType btype;

  /**
   * @brief Stats common to all types of device contexts.
   */
  ROCStats ctxStats{};

  /**
   * @brief Stats common to all types of host contexts.
   */
  ROCHostStats ctxHostStats{};

 protected:
  /**************************************************************************
   ***************************** POLICY MEMBERS *****************************
   *************************************************************************/

  /**
   * @brief Coalesce policy for 'multi' configuration builds
   */
  WavefrontCoalescer wf_coal_{};

 public:
  /**
   * @brief Inter-Process Communication (IPC) interface for context class
   *
   * This member is an interface to allow intra-node interprocess
   * communication through shared memory.
   */
  IpcImpl ipcImpl_{};
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_CONTEXT_HPP_
