/*
 * Copyright © 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including
 * the next paragraph) shall be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "runtime.h"

#include <unistd.h>

namespace rocdxg {

namespace {

void PrepareForkHandler() { Runtime::Instance().PrepareFork(); }
void ParentForkHandler() { Runtime::Instance().ParentFork(); }
void ChildForkHandler() { Runtime::Instance().ChildFork(); }

} // namespace

Runtime &Runtime::Instance() {
  static Runtime *runtime = new Runtime();
  return *runtime;
}

Runtime::Runtime()
    : hsakmt_mutex(PTHREAD_MUTEX_INITIALIZER) {
  ResetState();
}

void Runtime::ResetState() {
  dxg_fd = -1;
  parent_pid = getpid();
  is_forked = false;
  hsakmt_debug_level = HSAKMT_DEBUG_LEVEL_DEFAULT;
  dxg_open_count = 0;
  hsakmt_is_dgpu = false;
  is_svm_api_supported = false;
  zfb_support = 0;
  vendor_packet_process = 0;
  check_avail_sysram = false;
  max_single_alloc_size = 0;
  enable_thunk_sub_allocator = 0;
  heap.Reset();
  default_node = 1;
}

void Runtime::Reset() {
  if (dxg_fd >= 0) {
    close(dxg_fd);
  }
  ResetState();
}

/* Detect when the process has forked since the last call. We cannot rely
 * on pthread_atfork because the process can fork without calling the fork
 * function in libc, such as through clone or a direct syscall.
 */
bool Runtime::IsForkedChild() {
  if (is_forked)
    return true;

  pid_t cur_pid = getpid();
  if (parent_pid != cur_pid) {
    is_forked = true;
    parent_pid = cur_pid;
    return true;
  }

  return false;
}

/* Call from the child process after fork. This clears data duplicated from
 * the parent process that is not valid in the child. Topology information
 * remains valid in the child process, so it is not cleared.
 */
void Runtime::ClearAfterFork() {
  reset_suballocator();
  clear_allocation_map();
  Reset();
}

void Runtime::InstallAtForkHandlers() {
  static bool atfork_installed = false;
  /* Atfork handlers cannot be uninstalled and must be installed only once.
   * Otherwise prepare will deadlock when trying to take the same lock
   * multiple times.
   */
  if (atfork_installed)
    return;

  pthread_atfork(PrepareForkHandler, ParentForkHandler, ChildForkHandler);
  atfork_installed = true;
}

void Runtime::PrepareFork() {
  pthread_mutex_lock(&hsakmt_mutex);
}

void Runtime::ParentFork() {
  pthread_mutex_unlock(&hsakmt_mutex);
}

void Runtime::ChildFork() {
  pthread_mutex_init(&hsakmt_mutex, NULL);
  is_forked = true;
}

void Runtime::HeapInit() { heap.Init(); }

void Runtime::HeapFini() { heap.Fini(); }

bool Runtime::ReserveSvmSpace(uint64_t &base, uint64_t &size, uint64_t align) {
  return heap.ReserveSvmSpace(base, size, align);
}

bool Runtime::FreeSvmSpace(uint64_t &base, uint64_t &size) {
  return heap.FreeSvmSpace(base, size);
}

bool Runtime::ReserveLocalHeapSpace() { return heap.ReserveLocalHeapSpace(); }

bool Runtime::FreeLocalHeapSpace() { return heap.FreeLocalHeapSpace(); }

bool Runtime::ReserveSystemHeapSpace() { return heap.ReserveSystemHeapSpace(); }

bool Runtime::FreeSystemHeapSpace() { return heap.FreeSystemHeapSpace(); }

bool Runtime::CommitSystemHeapSpace(void* addr, int64_t size, bool lock) {
  return heap.CommitSystemHeapSpace(addr, size, lock);
}

bool Runtime::DecommitSystemHeapSpace(void* addr, int64_t size) {
  return heap.DecommitSystemHeapSpace(addr, size);
}

ErrorCode Runtime::ReserveGpuVirtualAddress(
    const thunk_proxy::AllocDomain domain, gpusize hit_base_addr, gpusize size,
    gpusize *out_gpu_virt_addr, gpusize alignment, bool lock) {
  return heap.ReserveGpuVirtualAddress(domain, hit_base_addr, size,
                                       out_gpu_virt_addr, alignment, lock);
}

ErrorCode Runtime::FreeGpuVirtualAddress(
    const thunk_proxy::AllocDomain domain, gpusize gpu_addr, gpusize size) {
  return heap.FreeGpuVirtualAddress(domain, gpu_addr, size);
}

bool Runtime::CommitSystemHeapSpaceIPC(void* addr, int64_t size, int &fd,
                                       bool lock) {
  return heap.CommitSystemHeapSpaceIPC(addr, size, fd, lock);
}

bool Runtime::DecommitSystemHeapSpaceIPC(void* addr, int64_t size, int &memfd) {
  return heap.DecommitSystemHeapSpaceIPC(addr, size, memfd);
}

ErrorCode Runtime::ReserveIPCSysMem(gpusize size, gpusize *out_gpu_virt_addr,
                                    gpusize alignment, int &memfd, bool lock) {
  return heap.ReserveIPCSysMem(size, out_gpu_virt_addr, alignment, memfd, lock);
}

ErrorCode Runtime::FreeIPCSysMem(gpusize gpu_addr, gpusize size, int &memfd) {
  return heap.FreeIPCSysMem(gpu_addr, size, memfd);
}

bool Runtime::InitHandleApertureSpace() { return heap.InitHandleApertureSpace(); }

ErrorCode Runtime::HandleApertureAlloc(gpusize size, gpusize *out_gpu_virt_addr) {
  return heap.HandleApertureAlloc(size, out_gpu_virt_addr);
}

void Runtime::HandleApertureFree(gpusize gpu_addr) {
  heap.HandleApertureFree(gpu_addr);
}

} // namespace rocdxg
