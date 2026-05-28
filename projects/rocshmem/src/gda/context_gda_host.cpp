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


#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "context_gda_host.hpp"
#include "backend_type.hpp"
#include "context_incl.hpp"
#include "backend_gda.hpp"
#include "host/host.hpp"

namespace rocshmem {

__host__ GDAHostContext::GDAHostContext(Backend *backend,
                                        [[maybe_unused]] int64_t options)
    : Context(backend, true) {
  GDABackend *b{static_cast<GDABackend *>(backend)};

  host_interface = b->host_interface;

  context_window_info = host_interface->acquire_window_context();

  int *pes_with_ipc_avail = new int[backend->ipcImpl.shm_size];
  char** ipc_bases = new char*[b->ipcImpl.shm_size];
  if (backend->ipcImpl.pes_with_ipc_avail != nullptr) {
    CHECK_HIP(hipMemcpy(pes_with_ipc_avail,
                  backend->ipcImpl.pes_with_ipc_avail,
                  backend->ipcImpl.shm_size * sizeof(int),
                  hipMemcpyDeviceToHost));
    CHECK_HIP(hipMemcpy(ipc_bases,
                  backend->ipcImpl.ipc_bases,
                  backend->ipcImpl.shm_size * sizeof(char *),
                  hipMemcpyDeviceToHost));
  }
  ipcImpl_.pes_with_ipc_avail = pes_with_ipc_avail;
  ipcImpl_.ipc_bases = ipc_bases;
  ipcImpl_.shm_size = backend->ipcImpl.shm_size;
  ipcImpl_.shm_rank = backend->ipcImpl.shm_rank;
}

__host__ GDAHostContext::~GDAHostContext() {
  delete[] ipcImpl_.pes_with_ipc_avail;
  delete[] ipcImpl_.ipc_bases;

  host_interface->release_window_context(context_window_info);
}

__host__ void GDAHostContext::putmem_nbi(void *dest, const void *source,
                                         size_t nelems, int pe) {
  host_interface->putmem_nbi(dest, source, nelems, pe, context_window_info);
}

__host__ void GDAHostContext::getmem_nbi(void *dest, const void *source,
                                         size_t nelems, int pe) {
  host_interface->getmem_nbi(dest, source, nelems, pe, context_window_info);
}

__host__ void GDAHostContext::putmem(void *dest, const void *source,
                                     size_t nelems, int pe) {
  host_interface->putmem(dest, source, nelems, pe, context_window_info);
}

__host__ void GDAHostContext::getmem(void *dest, const void *source,
                                     size_t nelems, int pe) {
  host_interface->getmem(dest, source, nelems, pe, context_window_info);
}

__host__ void GDAHostContext::fence() {
  host_interface->fence(context_window_info);
}

__host__ void GDAHostContext::quiet() {
  host_interface->quiet(context_window_info);
}

__host__ void *GDAHostContext::shmem_ptr(const void *dest, int pe) {
  void *ret = nullptr;
  int local_pe{-1};
  if (ipcImpl_.isIpcAvailable(my_pe, pe, &local_pe)) {
    void *dst = const_cast<void *>(dest);
    uint64_t L_offset = reinterpret_cast<char *>(dst) - ipcImpl_.ipc_bases[ipcImpl_.shm_rank];
    ret = ipcImpl_.ipc_bases[local_pe] + L_offset;
  }
  return ret;
}

__host__ void GDAHostContext::sync_all() {
  host_interface->sync_all(context_window_info);
}

__host__ void GDAHostContext::barrier_all() {
  host_interface->barrier_all(context_window_info);
}

}  // namespace rocshmem
