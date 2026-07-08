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

#ifndef LIBROCDXG_RUNTIME_H_INCLUDED
#define LIBROCDXG_RUNTIME_H_INCLUDED

#include <pthread.h>
#include <stdint.h>
#include <memory>
#include <unistd.h>

#include "heap_manager.h"
#include "impl/wddm/va_mgr.h"
#include "memory_allocation_registry.h"
#include "shared/include/d3dkmt_types.h"
#include "shared/include/dxcore_loader.h"
#include "shared/include/status.h"
#include "shared/include/thunk_proxy/thunk_proxy.h"
#include "util/simple_heap.h"

namespace rocdxg {

class BlockAllocator {
private:
    static const size_t block_size_ = 128 * 1024 * 1024;  // 128MB blocks.

public:
    void* alloc(size_t request_size, size_t& allocated_size) const;
    void free(void* ptr, size_t length) const;
    size_t block_size() const { return block_size_; }
};

class Runtime {
public:
  static Runtime &Instance();

  Runtime();
  ~Runtime() = default;

  void Reset();
  bool IsForkedChild();
  void ClearAfterFork();
  void InstallAtForkHandlers();
  void PrepareFork();
  void ParentFork();
  void ChildFork();
  void ResetMemoryState();
  AllocationRegistry &Allocations() { return allocation_registry_; }
  wsl::SimpleHeap<BlockAllocator> &FragmentAllocator() {
    return fragment_allocator_;
  }

  void HeapInit();
  void HeapFini();
  bool ReserveSvmSpace(uint64_t &base, uint64_t &size, uint64_t align);
  bool FreeSvmSpace(uint64_t &base, uint64_t &size);
  bool ReserveLocalHeapSpace();
  bool FreeLocalHeapSpace();
  bool ReserveSystemHeapSpace();
  uint64_t SystemHeapSize() const { return heap.SystemHeapSize(); }
  bool FreeSystemHeapSpace();
  bool CommitSystemHeapSpace(void* addr, int64_t size, bool lock);
  bool DecommitSystemHeapSpace(void* addr, int64_t size);
  ErrorCode ReserveGpuVirtualAddress(const thunk_proxy::AllocDomain domain,
          gpusize hit_base_addr, gpusize size,
          gpusize *out_gpu_virt_addr, gpusize alignment, bool lock);
  ErrorCode FreeGpuVirtualAddress(const thunk_proxy::AllocDomain domain,
          gpusize gpu_addr, gpusize size);
  bool CommitSystemHeapSpaceIPC(void* addr, int64_t size, int &fd, bool lock=false);
  bool DecommitSystemHeapSpaceIPC(void* addr, int64_t size, int &memfd);
  ErrorCode ReserveIPCSysMem(gpusize size,
          gpusize *out_gpu_virt_addr, gpusize alignment,
          int &memfd, bool lock);
  ErrorCode FreeIPCSysMem(gpusize gpu_addr, gpusize size, int &memfd);
  bool InitHandleApertureSpace();
  ErrorCode HandleApertureAlloc(gpusize size, gpusize *out_gpu_virt_addr);
  void HandleApertureFree(gpusize gpu_addr);

  pthread_mutex_t hsakmt_mutex;
  const char *dxg_device_name = "/dev/dxg";
  long page_size;
  int page_shift;
  int dxg_fd = -1;
  pid_t parent_pid = -1;
  bool is_forked = false;
  int hsakmt_debug_level;
  unsigned long dxg_open_count;
  bool hsakmt_is_dgpu;
  bool is_svm_api_supported;
  int zfb_support;
  int vendor_packet_process;
  bool check_avail_sysram;
  size_t max_single_alloc_size;
  int enable_thunk_sub_allocator;
  uint32_t default_node;

private:
  void ResetState();

  HeapManager heap;
  AllocationRegistry allocation_registry_;
  wsl::SimpleHeap<BlockAllocator> fragment_allocator_;
};

} // namespace rocdxg

inline rocdxg::Runtime &dxg_runtime() { return rocdxg::Runtime::Instance(); }

#endif
