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

#ifndef LIBROCDXG_HEAP_MANAGER_H_INCLUDED
#define LIBROCDXG_HEAP_MANAGER_H_INCLUDED

#include <cstdint>
#include <memory>

#include "impl/wddm/va_mgr.h"
#include "shared/include/d3dkmt_types.h"
#include "shared/include/status.h"
#include "shared/include/thunk_proxy/thunk_proxy.h"

namespace rocdxg {

class HeapManager {
public:
  void Init();
  void Fini();
  void Reset();

  bool ReserveSvmSpace(uint64_t &base, uint64_t &size, uint64_t align);
  bool FreeSvmSpace(uint64_t &base, uint64_t &size);
  bool ReserveLocalHeapSpace();
  bool FreeLocalHeapSpace();
  bool ReserveSystemHeapSpace();
  uint64_t SystemHeapSize() const { return system_heap_space_size_; }
  bool FreeSystemHeapSpace();

  bool CommitSystemHeapSpace(void *addr, int64_t size, bool lock);
  bool DecommitSystemHeapSpace(void *addr, int64_t size);
  ErrorCode ReserveGpuVirtualAddress(thunk_proxy::AllocDomain domain,
                                     gpusize hint_base_addr, gpusize size,
                                     gpusize *out_gpu_virt_addr,
                                     gpusize alignment, bool lock);
  ErrorCode FreeGpuVirtualAddress(thunk_proxy::AllocDomain domain,
                                  gpusize gpu_addr, gpusize size);

  bool CommitSystemHeapSpaceIPC(void *addr, int64_t size, int &fd,
                                bool lock = false);
  bool DecommitSystemHeapSpaceIPC(void *addr, int64_t size, int &memfd);
  ErrorCode ReserveIPCSysMem(gpusize size, gpusize *out_gpu_virt_addr,
                             gpusize alignment, int &memfd, bool lock);
  ErrorCode FreeIPCSysMem(gpusize gpu_addr, gpusize size, int &memfd);

  bool InitHandleApertureSpace();
  ErrorCode HandleApertureAlloc(gpusize size, gpusize *out_gpu_virt_addr);
  void HandleApertureFree(gpusize gpu_addr);

private:
  void InitLocalHeapMgr();
  void InitSystemHeapMgr();
  void InitHandleApertureMgr();

  uint64_t local_heap_space_start_ = 0;
  uint64_t local_heap_space_size_ = 0;
  std::unique_ptr<wsl::thunk::VaMgr> local_heap_mgr_;

  uint64_t system_heap_space_start_ = 0;
  uint64_t system_heap_space_size_ = 0;
  std::unique_ptr<wsl::thunk::VaMgr> system_heap_mgr_;

  uint64_t handle_aperture_start_ = 0;
  uint64_t handle_aperture_size_ = 0;
  std::unique_ptr<wsl::thunk::VaMgr> handle_aperture_mgr_;
};

} // namespace rocdxg

#endif
