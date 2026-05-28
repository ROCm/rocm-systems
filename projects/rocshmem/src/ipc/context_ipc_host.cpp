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

namespace rocshmem {

__host__ IPCHostContext::IPCHostContext(Backend *backend,
                                        [[maybe_unused]] int64_t options)
    : Context(backend, true) {
  IPCBackend *b{static_cast<IPCBackend *>(backend)};

  host_interface = b->host_interface;

  context_window_info = host_interface->acquire_window_context();

  char** ipc_bases = new char*[b->ipcImpl.shm_size];

  CHECK_HIP(hipMemcpy(ipc_bases,
                  b->ipcImpl.ipc_bases,
                  b->ipcImpl.shm_size * sizeof(char *),
                  hipMemcpyDeviceToHost));

  ipcImpl_.ipc_bases = ipc_bases;
}

__host__ IPCHostContext::~IPCHostContext() {
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
  host_interface->putmem(dest, source, nelems, pe, context_window_info);
}

__host__ void IPCHostContext::getmem(void *dest, const void *source,
                                     size_t nelems, int pe) {
  host_interface->getmem(dest, source, nelems, pe, context_window_info);
}

__host__ void IPCHostContext::fence() {
  host_interface->fence(context_window_info);
}

__host__ void IPCHostContext::quiet() {
  host_interface->quiet(context_window_info);
}

__host__ void *IPCHostContext::shmem_ptr(const void *dest, int pe) {
  void *ret = nullptr;
  void *dst = const_cast<void *>(dest);
  uint64_t L_offset = reinterpret_cast<char *>(dst) - ipcImpl_.ipc_bases[my_pe];
  ret = ipcImpl_.ipc_bases[pe] + L_offset;
  return ret;
}

__host__ void IPCHostContext::sync_all() {
  host_interface->sync_all(context_window_info);
}

__host__ void IPCHostContext::barrier_all() {
  host_interface->barrier_all(context_window_info);
}

__host__ void IPCHostContext::barrier_all_on_stream(hipStream_t stream) {
  host_interface->barrier_all_on_stream(stream);
}

}  // namespace rocshmem
