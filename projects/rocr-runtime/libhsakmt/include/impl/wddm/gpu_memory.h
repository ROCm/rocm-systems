////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2020, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cstddef>
#include <cstdint>
#include "util/utils.h"
#include "impl/wddm/types.h"
#include "impl/wddm/thunks.h"
#include "impl/thunk_proxy/thunk_proxy.h"

namespace wsl {
namespace thunk {

#define INVALID_DMABUF_FD ((uint64_t)-1)

class WDDMDevice;

union GpuMemoryCreateFlags {
  struct {
    uint64_t virtual_alloc              : 1; // only allocate virtual address, without physical buffer
    uint64_t physical_only              : 1; // only allocate physical buffer, without virtual address
    uint64_t interprocess               : 1; // physical buffer need share info between exporter and importer
    uint64_t locked                     : 1; // lock virtual address space into RAM, preventing that memory from being paged to the swap area
    uint64_t physical_contiguous        : 1; // contiguous physical pages
    uint64_t sysmem_ipc_sig_importer    : 1; // allocate system memory for IPC signal
    uint64_t sysmem_ipc_sig_exporter    : 1; // allocate system memory for IPC signal, prepare to export
    uint64_t alloc_va                   : 1; // allocate va. 0 for vmem import
    uint64_t blit_kernel_object         : 1; // allocate executable blit kernel object
    uint64_t kmt_handle_importer        : 1; // import from KMT handle
    uint64_t unused                     : 54;
  };
  uint64_t reserved;
};

union GpuMemoryDescFlags {
  struct {
    uint32_t is_virtual  : 1;
    uint32_t is_shared   : 1;
    uint32_t is_external : 1;
    uint32_t is_physical_only : 1;
    uint32_t is_locked : 1;
    uint32_t is_queue_referenced : 1;
    uint32_t is_physical_contiguous : 1;
    uint32_t is_imported_sys_memfd : 1;     // 0 - ignored; 1 - va from system heap
    uint32_t is_sysmem_exporter : 1; // allocate system memory for IPC signal, prepare to export
    uint32_t is_va_required :1;
    uint32_t is_imported_vram_vmem	:1;
    uint32_t is_imported_vram_ipc	:1;
    uint32_t is_imported_from_same_process : 3; // imported from same process, record shared cnt
    uint32_t is_blit_kernel_object : 1; // blit kernel object
    uint32_t unused : 16;
  };

  uint32_t reserved;
};

struct GpuMemoryCreateInfo {
  GpuMemoryCreateInfo() {
    flags.reserved = 0;
    domain = thunk_proxy::kLocal;
    size = 0;
    alignment = 0;
    mem_flags = 0;
    engine_flag = 0;
    va_hint = 0;
    user_ptr = nullptr;
    dmabuf_fd = INVALID_DMABUF_FD;
  }

  GpuMemoryCreateFlags flags;
  thunk_proxy::AllocDomain domain;
  gpusize size;
  gpusize alignment;
  int mem_flags;
  int engine_flag;
  uint64_t dmabuf_fd;  // Import from dmabuf

  void *user_ptr;
  gpusize va_hint;
};

struct GpuMemoryDesc {
  GpuMemoryDesc() {
    gpu_addr = 0;
    cpu_addr = nullptr;
    client_size = 0;
    size = alignment = 0;
    flags.reserved = 0;
    mem_flags = 0;
    engine_flag = 0;
    handle_ape_addr = 0;
  }

  thunk_proxy::AllocDomain domain;
  LUID adapter_luid;      // Where is the backing store location
  gpusize gpu_addr;
  void *cpu_addr;
  gpusize client_size;    // user request size
  gpusize size;
  gpusize alignment;
  gpusize handle_ape_addr;

  GpuMemoryDescFlags flags;
  int mem_flags;
  int engine_flag;
};

struct SharedHandleInfo {
  thunk_proxy::AllocDomain domain;
  LUID adapter_luid;
  gpusize client_size;    // user request size
  uint64_t size;
  uint32_t flags;
  int mem_flags;
  int pid;
  gpusize gpu_addr;
};

using GpuMemoryHandle = void *;

class GpuMemory {
public:
  static size_t CalcChunkNumbers(gpusize size);

  ErrorCode Init(const GpuMemoryCreateInfo &create_info);

  WDDMDevice *GetDevice() const { return device_; }
  gpusize Size() const { return desc_.size; }
  gpusize ClientSize() const { return desc_.client_size; }
  uint64_t GpuAddress() const { return desc_.gpu_addr; }
  void *CpuAddress() const { return desc_.cpu_addr; }
  uint64_t HandleApeAddress() const { return desc_.handle_ape_addr; }

  inline bool IsLocal() const { return desc_.domain == thunk_proxy::kLocal; }
  inline bool IsUserMemory() const { return desc_.domain == thunk_proxy::kUserMemory; }
  inline bool IsSystem() const { return desc_.domain == thunk_proxy::kSystem; }
  inline bool IsSysMemExporter() const { return desc_.flags.is_sysmem_exporter; }
  inline bool IsSysMemFd() const { return desc_.flags.is_imported_sys_memfd; }
  inline bool IsUserQueue() const { return desc_.domain == thunk_proxy::kUserQueue; }
  inline bool IsPhysicalOnly() const { return desc_.flags.is_physical_only; }
  inline bool IsPhysicalContiguous() const { return desc_.flags.is_physical_contiguous; }
  inline bool IsVirtual() const { return desc_.flags.is_virtual; }
  inline bool IsShared() const { return desc_.flags.is_shared; }
  inline bool IsExternal() const { return desc_.flags.is_external; }
  inline bool IsVaAllocated() const { return desc_.flags.is_va_required; }
  inline bool IsBlitKernelObject() const { return desc_.flags.is_blit_kernel_object; }
  inline void forceSysMem() { desc_.domain = thunk_proxy::kSystem; }
  inline void SetGpuAddress(uint64_t gpu_addr) { desc_.gpu_addr = gpu_addr; }
  inline void SetCpuAddress(void* cpu_addr) { desc_.cpu_addr = cpu_addr; }

  inline uint32_t Flags() const { return desc_.flags.reserved; }
  inline int GetAllocInfo() const { return desc_.mem_flags; }
  inline bool IsFineGrain() const { return (desc_.mem_flags & thunk_proxy::kFineGrain); }
  inline bool IsSameAdapter(const LUID &luid) const {
    return (desc_.adapter_luid.HighPart == luid.HighPart &&
      desc_.adapter_luid.LowPart == luid.LowPart);
  }
  inline void GetQueueReference() { desc_.flags.is_queue_referenced = 1; }
  inline void PutQueueReference() { desc_.flags.is_queue_referenced = 0; }
  inline bool IsQueueReferenced() const { return desc_.flags.is_queue_referenced; }
  inline void IncSharedReference() { desc_.flags.is_imported_from_same_process++; }
  inline uint32_t DecSharedReference() { return (desc_.flags.is_imported_from_same_process == 0) ? 0 : --desc_.flags.is_imported_from_same_process; }
  inline bool IsSharedFromSameProcess() const { return desc_.flags.is_imported_from_same_process > 0; }
  inline bool IsPhysicalCreated() const { return is_phymem_created; }

  WinAllocationHandle GetAllocationHandle(size_t index) const { return alloc_handles_ptr_[index]; }
  size_t NumChunks() const { return num_allocations_; }

  const GpuMemoryHandle GetGpuMemoryHandle() const {
    return reinterpret_cast<GpuMemoryHandle>(const_cast<GpuMemory*>(this));
  }

  static GpuMemory *Convert(GpuMemoryHandle handle) { return reinterpret_cast<GpuMemory *>(handle); }

  ErrorCode ReserveGpuVirtualAddress(gpusize base_virt_addr, gpusize va_size, gpusize alignment);
  ErrorCode FreeGpuVirtualAddress(gpusize va_start_address, gpusize va_size);

  ErrorCode MapGpuVirtualAddress(const gpusize map_addr, const gpusize size, gpusize offset = 0);
  ErrorCode UnmapGpuVirtualAddress(const gpusize map_addr, const gpusize size, gpusize offset = 0);
  ErrorCode MapMemoryToVirtualAddress(bool create_phys_mem = true);
  ErrorCode MakeResident();
  ErrorCode Evict();
  ErrorCode CreatePhysicalMemory();
  ErrorCode FreePhysicalMemory();

  ErrorCode OpenResourceFromKMTHandle(D3DKMT_HANDLE buffer_handle, D3DKMT_HANDLE device_handle,
                                      D3DKMT_OPENRESOURCE** out_open_resource);
  ErrorCode OpenResourceFromNTHandle(HANDLE buffer_handle, D3DKMT_HANDLE device_handle,
                                     D3DKMT_OPENRESOURCEFROMNTHANDLE** out_open_resource);

  ErrorCode ExportPhysicalHandle(int* dmabuf_fd, uint32_t flags = SHARED_ALLOCATION_ALL_ACCESS);
  ErrorCode ImportPhysicalFD(const GpuMemoryCreateInfo& create_info, gpusize* gpu_addr = nullptr);
  ErrorCode ImportPhysicalKMTHandle(const GpuMemoryCreateInfo& create_info,
                                    gpusize* gpu_addr = nullptr);
  ErrorCode ImportPhysicalNTHandle(const GpuMemoryCreateInfo& create_info,
                                   gpusize* gpu_addr = nullptr);
  ErrorCode ImportPhysicalHandle(const GpuMemoryCreateInfo& create_info,
                                 gpusize* gpu_addr = nullptr);
  WinAllocationHandle KmtHandle() const { return alloc_handle_; }
  ~GpuMemory();

 protected:
  explicit GpuMemory(WDDMDevice *device);
private:
  uint64_t AdjustSize(gpusize size) const;
private:
  friend class WDDMDevice;

  WDDMDevice *const device_;

  GpuMemoryDesc desc_;

  size_t num_allocations_;
  WinAllocationHandle *alloc_handles_ptr_;
  WinAllocationHandle alloc_handle_; // Optimization for num_allocations_ is 1

  WinResourceHandle resource_;     // Handle to a resource object that wraps the allocation. Used for shared resources

  int mem_fd_; // IPC sigal's sys mem fd

  bool is_phymem_created = false; // status of physical memory allocation

  DISALLOW_COPY_AND_ASSIGN(GpuMemory);
};

} // namespace thunk
} // namespace wsl

