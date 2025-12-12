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

#include "context_ipc_host.hpp"

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "backend_type.hpp"
#include "context_incl.hpp"
#include "backend_ipc.hpp"
#include "host/host.hpp"
#include "ipc_policy.hpp"

/* IPC non-MPI host AMO kernels — templated over the element type. */
template <typename T>
__global__ static void ipc_fadd(T *dst, T val, T *result) {
  rocshmem::IpcImpl impl;
  *result = impl.ipcAMOFetchAdd(dst, val);
}

template <typename T>
__global__ static void ipc_fcas(T *dst, T cond, T val, T *result) {
  rocshmem::IpcImpl impl;
  *result = impl.ipcAMOFetchCas(dst, cond, val);
}

namespace rocshmem {

__host__ IPCHostContext::IPCHostContext(Backend *backend,
                                        [[maybe_unused]] int64_t options)
    : Context(backend) {
  IPCBackend *b{static_cast<IPCBackend *>(backend)};

  host_interface = b->host_interface;

  context_window_info = host_interface->acquire_window_context();

  char** ipc_bases = new char*[b->ipcImpl.shm_size];

  CHECK_HIP(hipMemcpy(ipc_bases,
                  b->ipcImpl.ipc_bases,
                  b->ipcImpl.shm_size * sizeof(char *),
                  hipMemcpyDeviceToHost));

  ipcImpl_.ipc_bases = ipc_bases;

  if (is_ipc_non_mpi()) {
    CHECK_HIP(hipStreamCreate(&ctx_stream_));
    CHECK_HIP(hipExtMallocWithFlags(reinterpret_cast<void **>(&ipc_staging_buf_),
                                    sizeof(uint64_t),
                                    hipDeviceMallocFinegrained));
  }
}

__host__ IPCHostContext::~IPCHostContext() {
  if (is_ipc_non_mpi()) {
    CHECK_HIP(hipStreamSynchronize(ctx_stream_));
    CHECK_HIP(hipStreamDestroy(ctx_stream_));
    CHECK_HIP(hipFree(ipc_staging_buf_));
  }

  delete[] ipcImpl_.ipc_bases;

  host_interface->release_window_context(context_window_info);
}

__host__ void IPCHostContext::putmem_nbi(void *dest, const void *source,
                                         size_t nelems, int pe) {
  host_interface->putmem_nbi(dest, source, nelems, pe, context_window_info);
}

__host__ void IPCHostContext::getmem_nbi(void *dest, const void *source,
                                         size_t nelems, int pe) {
  host_interface->getmem_nbi(dest, source, nelems, pe, context_window_info);
}

__host__ void IPCHostContext::putmem(void *dest, const void *source,
                                     size_t nelems, int pe) {
  if (is_ipc_non_mpi()) {
    CHECK_HIP(hipMemcpyAsync(shmem_ptr(dest, pe), source, nelems,
                             hipMemcpyDefault, ctx_stream_));
    // Blocking: drain stream so source is safe to reuse on return.
    CHECK_HIP(hipStreamSynchronize(ctx_stream_));
    return;
  }
  host_interface->putmem(dest, source, nelems, pe, context_window_info);
}

__host__ void IPCHostContext::getmem(void *dest, const void *source,
                                     size_t nelems, int pe) {
  if (is_ipc_non_mpi()) {
    CHECK_HIP(hipMemcpyAsync(dest, shmem_ptr(source, pe), nelems,
                             hipMemcpyDefault, ctx_stream_));
    // Blocking get: destination buffer must be populated on return.
    CHECK_HIP(hipStreamSynchronize(ctx_stream_));
    return;
  }
  host_interface->getmem(dest, source, nelems, pe, context_window_info);
}

__host__ void IPCHostContext::fence() {
  if (is_ipc_non_mpi()) {
    // All host-initiated ops (putmem, getmem, AMOs) synchronize ctx_stream_
    // before returning, so ordering is already guaranteed. fence() is a no-op.
    return;
  }
  host_interface->fence(context_window_info);
}

__host__ void IPCHostContext::quiet() {
  if (is_ipc_non_mpi()) {
    CHECK_HIP(hipStreamSynchronize(ctx_stream_));
    return;
  }
  host_interface->quiet(context_window_info);
}

__host__ void *IPCHostContext::shmem_ptr(const void *dest, int pe) {
  void *ret = nullptr;
  void *dst = const_cast<void *>(dest);
  uint64_t L_offset = reinterpret_cast<char *>(dst) - ipcImpl_.ipc_bases[my_pe];
  ret = ipcImpl_.ipc_bases[pe] + L_offset;
  return ret;
}

template <typename T>
__host__ T IPCHostContext::ipc_amo_fadd(T *dst, T val, bool fetch) {
  ipc_fadd<T><<<1, 1, 0, ctx_stream_>>>(dst, val,
                                         reinterpret_cast<T *>(ipc_staging_buf_));
  if (!fetch) return T{};
  CHECK_HIP(hipStreamSynchronize(ctx_stream_));
  return *reinterpret_cast<T *>(ipc_staging_buf_);
}

template <typename T>
__host__ T IPCHostContext::ipc_amo_fcas(T *dst, T cond, T val) {
  ipc_fcas<T><<<1, 1, 0, ctx_stream_>>>(dst, cond, val,
                                          reinterpret_cast<T *>(ipc_staging_buf_));
  CHECK_HIP(hipStreamSynchronize(ctx_stream_));
  return *reinterpret_cast<T *>(ipc_staging_buf_);
}

#define IPC_AMO_STANDARD_INST(T)                                          \
  template __host__ T IPCHostContext::ipc_amo_fadd(T *, T, bool);         \
  template __host__ T IPCHostContext::ipc_amo_fcas(T *, T, T);

#define IPC_AMO_EXTENDED_INST(T)                                          \
  template __host__ T IPCHostContext::ipc_amo_fadd(T *, T, bool);

IPC_AMO_STANDARD_INST(int)
IPC_AMO_STANDARD_INST(long)
IPC_AMO_STANDARD_INST(long long)
IPC_AMO_STANDARD_INST(unsigned int)
IPC_AMO_STANDARD_INST(unsigned long)
IPC_AMO_STANDARD_INST(unsigned long long)
IPC_AMO_EXTENDED_INST(float)
IPC_AMO_EXTENDED_INST(double)

__host__ void IPCHostContext::sync_all() {
  host_interface->sync_all(context_window_info);
}

__host__ void IPCHostContext::sync(rocshmem_team_t team) {
  host_interface->sync(team, context_window_info);
}

__host__ void IPCHostContext::barrier_all() {
  host_interface->barrier_all(context_window_info);
}

__host__ void IPCHostContext::barrier(rocshmem_team_t team) {
  host_interface->barrier(team, context_window_info);
}

__host__ void IPCHostContext::barrier_all_on_stream(hipStream_t stream) {
  host_interface->barrier_all_on_stream(stream);
}

__host__ void IPCHostContext::barrier_on_stream(rocshmem_team_t team,
                                                hipStream_t stream) {
  host_interface->barrier_on_stream(team, stream);
}

__host__ void IPCHostContext::quiet_on_stream(hipStream_t stream) {
  LOG_TRACE("IPC backend: quiet_on_stream");
  host_interface->quiet_on_stream(stream);
}

__host__ void IPCHostContext::sync_all_on_stream(hipStream_t stream) {
  host_interface->sync_all_on_stream(stream);
}

__host__ void IPCHostContext::sync_on_stream(rocshmem_team_t team,
                                             hipStream_t stream) {
  host_interface->sync_on_stream(team, stream);
}

__host__ void IPCHostContext::alltoallmem_on_stream(rocshmem_team_t team,
                                                    void *dest,
                                                    const void *source,
                                                    size_t size,
                                                    hipStream_t stream) {
  host_interface->alltoallmem_on_stream(team, dest, source, size, stream);
}

__host__ void IPCHostContext::broadcastmem_on_stream(rocshmem_team_t team,
                                                     void *dest,
                                                     const void *source,
                                                     size_t nelems, int pe_root,
                                                     hipStream_t stream) {
  host_interface->broadcastmem_on_stream(team, dest, source, nelems, pe_root,
                                         stream);
}

__host__ void IPCHostContext::getmem_on_stream(void *dest, const void *source,
                                               size_t nelems, int pe,
                                               hipStream_t stream) {
  host_interface->getmem_on_stream(dest, source, nelems, pe, stream);
}

__host__ void IPCHostContext::putmem_on_stream(void *dest, const void *source,
                                               size_t nelems, int pe,
                                               hipStream_t stream) {
  LOG_TRACE("IPC backend: putmem_on_stream (pe=%d, size=%zu)", pe, nelems);
  host_interface->putmem_on_stream(dest, source, nelems, pe, stream);
}

__host__ void IPCHostContext::putmem_signal_on_stream(
    void *dest, const void *source, size_t nelems, uint64_t *sig_addr,
    uint64_t signal, int sig_op, int pe, hipStream_t stream) {
  host_interface->putmem_signal_on_stream(dest, source, nelems, sig_addr,
                                          signal, sig_op, pe, stream);
}

__host__ void IPCHostContext::signal_wait_until_on_stream(uint64_t *sig_addr,
                                                          int cmp,
                                                          uint64_t cmp_value,
                                                          hipStream_t stream) {
  host_interface->signal_wait_until_on_stream(sig_addr, cmp, cmp_value, stream);
}

}  // namespace rocshmem
