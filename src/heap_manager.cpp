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

#include "heap_manager.h"

#include <algorithm>
#include <cassert>
#include <fcntl.h>
#include <linux/mman.h>
#include <sys/mman.h>
#include <sys/sysinfo.h>
#include <unistd.h>

namespace rocdxg {

void HeapManager::Init() {
  ReserveLocalHeapSpace();
  ReserveSystemHeapSpace();
  InitHandleApertureSpace();
  InitLocalHeapMgr();
  InitSystemHeapMgr();
  InitHandleApertureMgr();
}

void HeapManager::Fini() {
  FreeSystemHeapSpace();
  FreeLocalHeapSpace();
}

void HeapManager::Reset() {
  Fini();
  local_heap_mgr_.reset();
  system_heap_mgr_.reset();
  system_heap_space_start_ = 0;
  system_heap_space_size_ = 0;
  handle_aperture_mgr_.reset();
  handle_aperture_start_ = 0;
  handle_aperture_size_ = 0;
}

bool HeapManager::ReserveSvmSpace(uint64_t &base, uint64_t &size,
                                  uint64_t align) {
  uint64_t sys_va[16] = {0};
  uint64_t local_va;
  uint64_t sys_va_size;
  int match_index = -1;
  void* ptr = NULL;

  wsl::thunk::WDDMDevice* device;
  size_t num_adapters = get_num_wddmdev();

  base = 0;
  sys_va_size = size + align;

  /* it will retry 16 times to find the avaliable range. */
  for (int i = 0; i < 16; i++) {
    local_va = 0;
    ptr = mmap(NULL, sys_va_size , PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
      pr_err("fail to reserve cpu va in %d time!\n", i);
      break;
    }

    sys_va[i] = (uint64_t)ptr;

    int match_cnt = 0;
    for (uint32_t j = 0; j < num_adapters; j++) {
      device = get_wddmdev(j+1);
      uint64_t start = (base == 0) ? (uint64_t)ptr : base;
      uint64_t end = start + ((base == 0) ? sys_va_size : size) + 1;

      if (wsl::thunk::d3dthunk::ReserveGpuVirtualAddress(
              device->GetAdapter(), size, start, end, &local_va) ==
          ErrorCode::Success) {
        match_cnt++;
        base = local_va;
        pr_debug("success to reserve gpu va %lx and va cpu %p in %d time\n",
                 local_va, ptr, i);
      } else {
        pr_err("%s fail to reserve gpu va for cpu va %p in %d time!\n",
               __FUNCTION__, ptr, i);
      }
    }

    if (match_cnt == num_adapters) {
      match_index = i;
      break;
    }
  }

  if (match_index >= 0) {
    /* release cpu unused ranges*/
    uint64_t left_size = local_va - sys_va[match_index];
    uint64_t right_size = align - left_size;
    if ((left_size > 0) && munmap((void*)sys_va[match_index], left_size))
      pr_err("fail to unmap left %lx with size %lx\n", sys_va[match_index], left_size);
    if ((right_size > 0) && munmap((void*)(local_va + size), right_size))
      pr_err("fail to unmap right %lx with size %lx\n", (local_va + size), right_size);
  } else {
    pr_err("fail to reserve Local Heap Space!\n");
    base = 0;
    size = 0;
  }

  /* free match fail address for cpu va */
  int free = match_index >= 0 ? match_index : 16;
  for (int j = 0; j < free; j++) {
    if (sys_va[j] != 0 && munmap((void*)sys_va[j], sys_va_size)) {
      pr_err("fail to unmap %d %lx\n", j, sys_va[j]);
    }
  }

  return match_index >= 0;
}

bool HeapManager::ReserveLocalHeapSpace() {
  wsl::thunk::WDDMDevice* device;
  uint64_t total_local_size = 0;
  uint64_t align = 0x40000000; /* 1G */
  size_t num_adapters = get_num_wddmdev();

  for (uint32_t j = 0; j < num_adapters; j++) {
    device = get_wddmdev(j+1);
    if (device == nullptr)
      return -1;
    /*
     * For APU, use non local memory(shared GPU memory) as GPU memory,
     * because it has small local memory
    */
    if (device->IsDgpu())
      total_local_size = wsl::Max(device->LocalHeapSize(), total_local_size);
    else
      total_local_size = wsl::Max(device->LocalHeapSize() + device->NonLocalHeapSize(), total_local_size);
  }

  total_local_size = wsl::AlignUp(total_local_size, align) * 4;
  local_heap_space_start_ = 0;
  local_heap_space_size_ = total_local_size;

  return ReserveSvmSpace(local_heap_space_start_, local_heap_space_size_, align);
}

bool HeapManager::FreeSvmSpace(uint64_t &base, uint64_t &size) {
  if (base == 0 || size == 0) {
    base = 0;
    size = 0;
    return true;
  }

  wsl::thunk::WDDMDevice* device;
  size_t num_adapters = get_num_wddmdev();
  for (uint32_t j = 0; j < num_adapters; j++) {
    device = get_wddmdev(j+1);
    if (device == nullptr)
      return -1;
    wsl::thunk::d3dthunk::FreeGpuVirtualAddress(device->GetAdapter(), base, size);
  }

  void *cpu = (void *)base;
  auto r = (munmap(cpu, size) == 0);
  base = 0;
  size = 0;
  return r;
}

bool HeapManager::FreeLocalHeapSpace() {
  return FreeSvmSpace(local_heap_space_start_, local_heap_space_size_);
}

void HeapManager::InitLocalHeapMgr() {
  local_heap_mgr_ = std::make_unique<wsl::thunk::VaMgr>(
      local_heap_space_start_, local_heap_space_size_, DEFAULT_GPU_PAGE_SIZE);
}

bool HeapManager::ReserveSystemHeapSpace() {
  struct sysinfo info;
  int ret = sysinfo(&info);
  uint64_t max_ram = 0x10000000000;
  uint64_t alignment = 0x100000000;
  assert(!ret);

  // minimum of reserve size is 8G, maximum of reserve size is 1T.
  system_heap_space_size_ = std::min(wsl::AlignUp(info.totalram, alignment) * 2, max_ram);

  return ReserveSvmSpace(system_heap_space_start_, system_heap_space_size_, alignment);
}

bool HeapManager::FreeSystemHeapSpace(void) {
  return FreeSvmSpace(system_heap_space_start_, system_heap_space_size_);
}

bool HeapManager::CommitSystemHeapSpace(void* addr, int64_t size, bool lock) {
  int32_t protFlags = PROT_READ | PROT_WRITE | PROT_EXEC;
  int32_t mapFlags = MAP_PRIVATE|MAP_ANONYMOUS|MAP_FIXED|
      MAP_NORESERVE|MAP_UNINITIALIZED;
  if (lock)
    mapFlags |= MAP_LOCKED;
  void* paddr = mmap(addr, size, protFlags, mapFlags, -1, 0);
  if (paddr == MAP_FAILED) {
    pr_err("fail to commit %s addr = %p, paddr = %p\n", (lock ? "locked" : ""), addr, paddr);
    return false;
  }
  assert(addr == paddr);

  if (madvise(addr, size, MADV_DONTFORK))
    pr_err("fail to set MADV_DONTFORK for addr = %p\n", addr);

  return true;
}

bool HeapManager::DecommitSystemHeapSpace(void* addr, int64_t size) {
  int32_t protFlags = PROT_NONE;
  int32_t mapFlags = MAP_PRIVATE|MAP_ANONYMOUS|MAP_FIXED|
      MAP_NORESERVE|MAP_UNINITIALIZED;
  void* paddr = mmap(addr, size, protFlags, mapFlags, -1, 0);
  if (paddr == MAP_FAILED) {
    pr_err("fail to decommit addr = %p, paddr = %p\n", addr, paddr);
    return false;
  }
  assert(addr == paddr);
  return true;
}

void HeapManager::InitSystemHeapMgr() {
  system_heap_mgr_ = std::make_unique<wsl::thunk::VaMgr>(
      system_heap_space_start_, system_heap_space_size_, DEFAULT_GPU_PAGE_SIZE);
}

ErrorCode HeapManager::ReserveGpuVirtualAddress(
    const thunk_proxy::AllocDomain domain, gpusize hit_base_addr, gpusize size,
    gpusize *out_gpu_virt_addr, gpusize alignment, bool lock) {
  gpusize gpu_addr = 0;
  ErrorCode code = ErrorCode::Success;

  uint64_t align = alignment == 0 ? (64 * 1024) : alignment; // default 64K alignment
  if (size >= GPU_HUGE_PAGE_SIZE)
    align = GPU_HUGE_PAGE_SIZE;

  if (domain == thunk_proxy::kSystem) {
    gpu_addr = system_heap_mgr_->Alloc(size, align, hit_base_addr);
    if (gpu_addr == 0)
      code = ErrorCode::OutOfMemory;

    if (!CommitSystemHeapSpace((void*)gpu_addr, size, lock)) {
      system_heap_mgr_->Free(gpu_addr);
      code = ErrorCode::SyscallFail;
    }
  } else {
    gpu_addr = local_heap_mgr_->Alloc(size, align, hit_base_addr);
    if (gpu_addr == 0)
      code = ErrorCode::OutOfGpuMemory;
  }

  *out_gpu_virt_addr = (code == ErrorCode::Success) ? gpu_addr : 0;
  return code;
}

ErrorCode HeapManager::FreeGpuVirtualAddress(
    const thunk_proxy::AllocDomain domain, gpusize gpu_addr, gpusize size) {
  auto code = ErrorCode::Success;

  if (domain == thunk_proxy::kSystem) {
    DecommitSystemHeapSpace((void *)gpu_addr, size);
    system_heap_mgr_->Free(gpu_addr);
  } else {
    local_heap_mgr_->Free(gpu_addr);
  }

  return code;
}

bool HeapManager::CommitSystemHeapSpaceIPC(void* addr, int64_t size, int &memfd, bool lock) {
  int fd = -1;

  if (memfd == -1) {
    fd = memfd_create("rocr4wsl_gtt", MFD_CLOEXEC);
    if (fd < 0) {
      pr_err("memfd_create failed\n");
      return false;
    }

    ftruncate(fd, size);
  } else {
    fd = memfd;
  }

  int32_t protFlags = PROT_READ | PROT_WRITE;
  int32_t mapFlags = MAP_SHARED | MAP_FIXED | MAP_NORESERVE |
      MAP_UNINITIALIZED | (lock ? MAP_LOCKED : 0);

  void* paddr = mmap(addr, size, protFlags, mapFlags, fd, 0);
  if (paddr == MAP_FAILED) {
    pr_err("fail to commit %s addr = %p, paddr = %p\n", (lock ? "locked" : ""), addr, paddr);
    if (memfd == -1)
      close(fd);
    return false;
  }
  assert(addr == paddr);

  memfd = fd;

  if (madvise(addr, size, MADV_DONTFORK))
    pr_err("fail to set MADV_DONTFORK for addr = %p\n", addr);

  return true;
}

bool HeapManager::DecommitSystemHeapSpaceIPC(void* addr, int64_t size, int &memfd) {
  if (munmap(addr, size) != 0) {
    pr_err("fail to unmap = %p \n", addr);
    return false;
  }
  close(memfd);
  memfd = -1;
  return true;
}

ErrorCode HeapManager::ReserveIPCSysMem(gpusize size,
    gpusize *out_gpu_virt_addr, gpusize alignment, int &memfd, bool lock) {
  gpusize gpu_addr = 0;
  ErrorCode code = ErrorCode::Success;
  gpu_addr = system_heap_mgr_->Alloc(size, alignment, 0);
  if (gpu_addr == 0)
    return ErrorCode::OutOfMemory;

  if (!CommitSystemHeapSpaceIPC((void*)gpu_addr, size, memfd, lock)) {
    system_heap_mgr_->Free(gpu_addr);
    code = ErrorCode::SyscallFail;
  }

  *out_gpu_virt_addr = (code == ErrorCode::Success) ? gpu_addr : 0;
  return code;
}

ErrorCode HeapManager::FreeIPCSysMem(gpusize gpu_addr, gpusize size, int &memfd) {
  auto code = ErrorCode::Success;

  DecommitSystemHeapSpaceIPC((void *)gpu_addr, size, memfd);

  system_heap_mgr_->Free(gpu_addr);
  return code;
}

bool HeapManager::InitHandleApertureSpace() {
  wsl::thunk::WDDMDevice* device;
  size_t num_adapters = get_num_wddmdev();
  handle_aperture_start_ = START_NON_CANONICAL_ADDR;
  handle_aperture_size_ = 1ULL << 47;

  while (handle_aperture_start_ < END_NON_CANONICAL_ADDR - 1) {
    for (uint32_t j = 0; j < num_adapters;) {
      device = get_wddmdev(j+1);
      if (device == nullptr)
        return -1;

      if (device->PrivateApertureBase() &&
          IS_OVERLAPPING(device->PrivateApertureBase(),
                         device->PrivateApertureSize(),
                         handle_aperture_start_,
                         handle_aperture_size_)) {
        handle_aperture_start_ += (1ULL << 47);
        continue;
      }

      if (device->SharedApertureBase() &&
          IS_OVERLAPPING(device->SharedApertureBase(),
                         device->SharedApertureSize(),
                         handle_aperture_start_,
                         handle_aperture_size_)) {
        handle_aperture_start_ += (1ULL << 47);
        continue;
      }

      j++;
    }
    pr_debug("handle aperture start %lx, size %lx\n", handle_aperture_start_, handle_aperture_size_);
    return true;
  }

  handle_aperture_start_ = 0;
  pr_err("fail\n");

  return false;
}

void HeapManager::InitHandleApertureMgr() {
  handle_aperture_mgr_ = std::make_unique<wsl::thunk::VaMgr>(
      handle_aperture_start_, handle_aperture_size_, DEFAULT_GPU_PAGE_SIZE);
}

ErrorCode HeapManager::HandleApertureAlloc(gpusize size, gpusize *out_gpu_virt_addr) {
  uint64_t align = DEFAULT_GPU_PAGE_SIZE;

  if (size >= GPU_HUGE_PAGE_SIZE)
    align = GPU_HUGE_PAGE_SIZE;

  *out_gpu_virt_addr = handle_aperture_mgr_->Alloc(size, align);
  if (*out_gpu_virt_addr == 0)
    return ErrorCode::OutOfHandleApeMemory;

  return ErrorCode::Success;
}

void HeapManager::HandleApertureFree(gpusize gpu_addr) {
  handle_aperture_mgr_->Free(gpu_addr);
}

} // namespace rocdxg
