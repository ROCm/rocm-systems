////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
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

#include "core/inc/amd_gpu_driver.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <amdgpu.h>
#include <amdgpu_drm.h>
#include <link.h>

#include "hsakmt/linux/kfd_ioctl.h"

#include "core/inc/amd_gpu_agent.h"
#include "core/inc/amd_memory_region.h"
#include "core/inc/runtime.h"
#include "core/util/utils.h"

extern r_debug _amdgpu_r_debug;

namespace rocr {
namespace AMD {

namespace fs = std::filesystem;

static const fs::path kKfdSysfsPath = "/sys/devices/virtual/kfd/kfd/topology";
static const long kPageSize = sysconf(_SC_PAGE_SIZE);

/// @brief Internal queue tracking for doorbell and kernel queue ID.
struct QueueInfo {
  uint32_t queue_id;
  uint32_t gpu_id;
  volatile uint64_t* doorbell_ptr;
  void* doorbell_mmap_base;
  void* eop_buffer = nullptr;
  uint64_t eop_handle = 0;
  uint32_t eop_size = 0;
  void* ctx_save_restore = nullptr;
  uint64_t ctx_handle = 0;
  uint32_t ctx_size = 0;
  bool ctx_is_mmap = false;  // true if CWSR was allocated via anonymous mmap+SVM
  // GPU-accessible memory for non-AQL queue wptr/rptr.
  // Matches libhsakmt's struct queue allocation.
  void* queue_struct = nullptr;
  uint64_t queue_struct_handle = 0;
  uint32_t queue_struct_size = 0;
};

/// @brief Process-level state for the KFD driver.
///
/// Matches libhsakmt behavior: the KFD fd, DRM render fds, ACQUIRE_VM state,
/// and events page persist across init/shutdown cycles. They are only cleaned
/// up after a fork (child process) or on process exit.
struct GpuDriver::ProcessState {
  int kfd_fd = -1;
  std::vector<int> drm_render_fds;   // indexed by node_id, -1 if not opened
  std::vector<amdgpu_device_handle> amdgpu_handles;  // cached per node_id
  bool vm_acquired = false;
  void* events_page = nullptr;
  uint64_t events_page_handle = 0;
  size_t events_page_size = 0;
  pid_t pid = -1;                    // PID at Open() time, for fork detection
};

bool GpuDriver::IsForkedChild() const {
  if (!process_state_) return false;
  return process_state_->pid != -1 && getpid() != process_state_->pid;
}

void GpuDriver::ClearAfterFork() {
  if (process_state_) {
    // Match libhsakmt clear_after_fork ordering:
    // 1. Clear doorbells (munmap non-GPUVM), 2. Clear events page,
    // 3. Close DRM fds / amdgpu handles, 4. Close KFD fd.
    // Closing DRM fds before KFD fd is important: the kernel's KFD close
    // handler may reference DRM state, so DRM must be cleaned up first.

    // Deinitialize amdgpu device handles (closes DRM fds internally).
    // Only close raw fds that don't have an associated amdgpu handle.
    for (size_t i = 0; i < process_state_->drm_render_fds.size(); ++i) {
      if (i < process_state_->amdgpu_handles.size() &&
          process_state_->amdgpu_handles[i]) {
        amdgpu_device_deinitialize(process_state_->amdgpu_handles[i]);
      } else if (process_state_->drm_render_fds[i] >= 0) {
        close(process_state_->drm_render_fds[i]);
      }
    }
    process_state_->drm_render_fds.clear();
    process_state_->amdgpu_handles.clear();
    process_state_->vm_acquired = false;

    // Clear events page pointer (the mapping itself was not inherited due to
    // MADV_DONTFORK, so no munmap needed).
    process_state_->events_page = nullptr;
    process_state_->events_page_handle = 0;
    process_state_->events_page_size = 0;

    // Close KFD fd last (after DRM cleanup), matching libhsakmt ordering.
    if (process_state_->kfd_fd >= 0) {
      close(process_state_->kfd_fd);
      process_state_->kfd_fd = -1;
    }
    process_state_->pid = -1;
  }

  // Clear per-instance state.
  mem_handles_.clear();
  runtime_caps_mask_ = 0;
  fd_ = -1;
  gpu_ids_.clear();
  all_gpu_id_array_.clear();
  doorbells_.clear();
}

uint32_t GpuDriver::NodeToGpuId(uint32_t node_id) const {
  if (node_id < gpu_ids_.size()) return gpu_ids_[node_id];
  return 0;
}

uint32_t GpuDriver::GpuIdToNodeId(uint32_t gpu_id) const {
  for (size_t i = 0; i < gpu_ids_.size(); ++i) {
    if (gpu_ids_[i] == gpu_id) return static_cast<uint32_t>(i);
  }
  return INVALID_NODEID;
}

int GpuDriver::DrmFdForGpuId(uint32_t gpu_id) const {
  if (!process_state_) return -1;
  for (size_t i = 0; i < gpu_ids_.size() &&
       i < process_state_->drm_render_fds.size(); ++i) {
    if (gpu_ids_[i] == gpu_id) return process_state_->drm_render_fds[i];
  }
  return -1;
}

///////////////////////////////////////////////////////////////////////////////
// Sysfs helpers
///////////////////////////////////////////////////////////////////////////////

/// @brief Read a single unsigned integer from a sysfs file.
static bool SysfsReadUint(const fs::path& path, uint64_t& value) {
  std::ifstream f(path);
  if (!f.is_open()) return false;
  f >> value;
  return !f.fail();
}

/// @brief Read key-value property file ("key value\n" format).
static std::map<std::string, uint64_t> SysfsReadProperties(const fs::path& path) {
  std::map<std::string, uint64_t> props;
  std::ifstream f(path);
  std::string key;
  uint64_t value;
  while (f >> key >> value) {
    props[key] = value;
  }
  return props;
}

/// @brief Count subdirectories in a sysfs directory.
static uint32_t SysfsCountSubdirs(const fs::path& dir) {
  uint32_t count = 0;
  std::error_code ec;
  for (auto& entry : fs::directory_iterator(dir, ec)) {
    if (entry.is_directory(ec)) ++count;
  }
  return count;
}

///////////////////////////////////////////////////////////////////////////////
// Static helpers (platform-specific, compiled only on Linux)
///////////////////////////////////////////////////////////////////////////////

int GpuDriver::GpuIoctl(int fd, unsigned long request, void* arg) {
  int ret;
  do {
    ret = ioctl(fd, request, arg);
  } while (ret == -1 && (errno == EINTR || errno == EAGAIN));

  if (ret == -1 && errno == EBADF) {
    debug_print("GPU driver fd not valid in this process\n");
  }
  return ret;
}

bool GpuDriver::BindXnackMode(int fd) {
  HSAint32 mode = core::Runtime::runtime_singleton_->flag().xnack();
  bool config_xnack = (mode != Flag::XNACK_REQUEST::XNACK_UNCHANGED);

  if (config_xnack) {
    kfd_ioctl_set_xnack_mode_args args = {};
    args.xnack_enabled = mode;
    if (GpuIoctl(fd, AMDKFD_IOC_SET_XNACK_MODE, &args) == 0) {
      return (mode != Flag::XNACK_DISABLE);
    }
  }

  // Query current xnack mode.
  kfd_ioctl_set_xnack_mode_args args = {};
  args.xnack_enabled = -1;
  if (GpuIoctl(fd, AMDKFD_IOC_SET_XNACK_MODE, &args) != 0) {
    debug_print(
        "Driver does not support xnack mode query.\n"
        "ROCr must assume xnack is disabled.\n");
    return false;
  }
  return (args.xnack_enabled != Flag::XNACK_DISABLE);
}

void *GpuDriver::AllocateDriverMemory(int fd, int drm_fd,
                                       const HsaMemFlags &flags,
                                       uint32_t gpu_id, size_t size,
                                       uint64_t *out_handle,
                                       bool is_device_alloc) {
  // WRITABLE is set by default unless ReadOnly is specified.
  // Matches libhsakmt fmm_translate_hsa_to_ioc_flags.
  uint32_t ioc_flags = 0;
  if (!flags.ui32.ReadOnly)
    ioc_flags |= KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;

  // Memory type selection.
  // Device allocations always go to VRAM (with PUBLIC if host-accessible).
  // Host allocations: paged memory uses USERPTR (anonymous mmap pinned for GPU
  // access), non-paged uses GTT (kernel-allocated pinned memory).
  if (is_device_alloc) {
    ioc_flags |= KFD_IOC_ALLOC_MEM_FLAGS_VRAM;
  } else if (flags.ui32.Scratch) {
    ioc_flags |= KFD_IOC_ALLOC_MEM_FLAGS_VRAM;
  } else if (flags.ui32.HostAccess && !flags.ui32.NonPaged) {
    ioc_flags |= KFD_IOC_ALLOC_MEM_FLAGS_USERPTR;
  } else if (flags.ui32.HostAccess || flags.ui32.NonPaged) {
    ioc_flags |= KFD_IOC_ALLOC_MEM_FLAGS_GTT;
  } else {
    ioc_flags |= KFD_IOC_ALLOC_MEM_FLAGS_VRAM;
  }

  // Attribute flags (must match libhsakmt's fmm_translate_hsa_to_ioc_flags).
  if (flags.ui32.NoSubstitute)
    ioc_flags |= KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE;
  if (flags.ui32.ExecuteAccess)
    ioc_flags |= KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE;
  if (!flags.ui32.CoarseGrain)
    ioc_flags |= KFD_IOC_ALLOC_MEM_FLAGS_COHERENT;
  if (flags.ui32.ExtendedCoherent)
    ioc_flags |= KFD_IOC_ALLOC_MEM_FLAGS_EXT_COHERENT;
  if (flags.ui32.Uncached)
    ioc_flags |= KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED;
  if (flags.ui32.AQLQueueMemory)
    ioc_flags |= (KFD_IOC_ALLOC_MEM_FLAGS_AQL_QUEUE_MEM |
                  KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED);
  // PUBLIC is only needed for VRAM with HostAccess (cross-GPU access).
  // GTT/USERPTR memory is inherently host-accessible.
  bool is_vram = (ioc_flags & KFD_IOC_ALLOC_MEM_FLAGS_VRAM) != 0;
  if (is_vram && flags.ui32.HostAccess)
    ioc_flags |= KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC;
  if (flags.ui32.Contiguous)
    ioc_flags |= KFD_IOC_ALLOC_MEM_FLAGS_CONTIGUOUS_BEST_EFFORT;

  bool no_address = flags.ui32.NoAddress;
  bool is_userptr = (ioc_flags & KFD_IOC_ALLOC_MEM_FLAGS_USERPTR) != 0;

  // Reserve a VA range. For USERPTR, allocate real anonymous pages (the CPU
  // mapping IS the memory). For VRAM/GTT, reserve an unmapped VA that the
  // DRM render mmap will later replace.
  void *reserved_va = nullptr;
  if (!no_address) {
    if (is_userptr) {
      reserved_va = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    } else {
      reserved_va = mmap(nullptr, size, PROT_NONE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    }
    if (reserved_va == MAP_FAILED)
      return nullptr;
  }


  kfd_ioctl_alloc_memory_of_gpu_args args = {};
  args.size = size;
  args.gpu_id = gpu_id;
  args.flags = ioc_flags;
  args.va_addr = reinterpret_cast<uint64_t>(reserved_va);
  // For USERPTR, mmap_offset is the user VA (tells KFD where the pages are).
  if (is_userptr)
    args.mmap_offset = reinterpret_cast<uint64_t>(reserved_va);

  if (GpuIoctl(fd, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &args) != 0) {
    debug_print("AllocateDriverMemory ioctl failed: va=%p size=%zu gpu_id=%u "
                "flags=0x%x errno=%d\n",
                reserved_va, size, gpu_id, ioc_flags, errno);
    if (reserved_va) munmap(reserved_va, size);
    return nullptr;
  }

  *out_handle = args.handle;

  if (no_address) {
    // NoAddress allocations have no GPU VA. Allocate a minimal anonymous
    // placeholder page so the caller gets a unique non-null key for tracking
    // in mem_handles_. libhsakmt uses an internal aperture for this purpose.
    void *token = mmap(nullptr, kPageSize, PROT_NONE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (token == MAP_FAILED) {
      kfd_ioctl_free_memory_of_gpu_args free_args = {};
      free_args.handle = args.handle;
      GpuIoctl(fd, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &free_args);
      return nullptr;
    }
    return token;
  }

  void *mem = reserved_va;

  // For VRAM/GTT, replace the anonymous reservation with the real GPU memory
  // mapping via the DRM render node fd. USERPTR memory is already mapped.
  // Only map with CPU access if HostAccess is set (not just NonPaged).
  // Matches libhsakmt: GTT with NonPaged=1 but HostAccess=0 is not CPU-mapped.
  if (!is_userptr) {
    bool host_access = flags.ui32.HostAccess;
    int prot = host_access ? (PROT_READ | PROT_WRITE) : PROT_NONE;
    mem = mmap(reserved_va, size, prot,
               MAP_SHARED | MAP_FIXED, drm_fd, args.mmap_offset);
    if (mem == MAP_FAILED) {
      debug_print("AllocateDriverMemory mmap failed: va=%p size=%zu drm_fd=%d "
                  "mmap_off=0x%lx errno=%d\n",
                  reserved_va, size, drm_fd,
                  (unsigned long)args.mmap_offset, errno);
      munmap(reserved_va, size);
      kfd_ioctl_free_memory_of_gpu_args free_args = {};
      free_args.handle = args.handle;
      GpuIoctl(fd, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &free_args);
      return nullptr;
    }
  }

  // Prevent child processes from inheriting this mapping (avoids BO refcount
  // issues after fork).
  madvise(mem, size, MADV_DONTFORK);

  return mem;
}

bool GpuDriver::FreeDriverMemory(int fd, void *mem, uint64_t handle,
                                  size_t size) {
  if (mem == nullptr || size == 0) {
    debug_print("Invalid free ptr:%p size:%lu\n", mem, size);
    return false;
  }

  // Free the BO before unmapping the pages. If memory is user memory and
  // it's still GPU mapped, munmap would cause an eviction. If the restore
  // happens quickly enough, restore would also fail with an error message.
  // (Matches libhsakmt __fmm_release ordering.)
  kfd_ioctl_free_memory_of_gpu_args args = {};
  args.handle = handle;
  if (GpuIoctl(fd, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &args) != 0) {
    debug_print("Failed to free ptr:%p size:%lu handle=0x%llx errno=%d\n",
                mem, size, (unsigned long long)handle, errno);
    return false;
  }

  munmap(mem, size);
  return true;
}

bool GpuDriver::MakeDriverMemoryResident(int fd, size_t num_node,
                                          const uint32_t *nodes,
                                          void *mem, uint64_t handle,
                                          size_t size,
                                          uint64_t *alternate_va,
                                          HsaMemMapFlags map_flag) {
  assert(num_node > 0);
  assert(nodes);

  *alternate_va = 0;

  kfd_ioctl_map_memory_to_gpu_args args = {};
  args.handle = handle;
  args.device_ids_array_ptr = reinterpret_cast<uint64_t>(nodes);
  args.n_devices = static_cast<uint32_t>(num_node);
  args.n_success = 0;

  int ret = GpuIoctl(fd, AMDKFD_IOC_MAP_MEMORY_TO_GPU, &args);
  if (ret != 0) {
    debug_print("MAP_MEM failed: handle=0x%llx n_devices=%u n_success=%u errno=%d\n",
                (unsigned long long)args.handle, args.n_devices, args.n_success, errno);
  }
  return (ret == 0);
}

void GpuDriver::MakeDriverMemoryUnresident(int fd, void *mem,
                                            uint64_t handle,
                                            size_t num_nodes,
                                            const uint32_t *gpu_ids) {
  if (num_nodes == 0 || gpu_ids == nullptr) return;

  kfd_ioctl_unmap_memory_from_gpu_args args = {};
  args.handle = handle;
  args.device_ids_array_ptr = reinterpret_cast<uint64_t>(gpu_ids);
  args.n_devices = static_cast<uint32_t>(num_nodes);
  args.n_success = 0;
  GpuIoctl(fd, AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, &args);
}

///////////////////////////////////////////////////////////////////////////////
// Driver lifecycle
///////////////////////////////////////////////////////////////////////////////

hsa_status_t GpuDriver::DiscoverDriver(std::unique_ptr<core::Driver>& driver) {
  auto tmp_driver = std::unique_ptr<core::Driver>(new GpuDriver("/dev/kfd"));

  if (tmp_driver->Open() == HSA_STATUS_SUCCESS) {
    driver = std::move(tmp_driver);
    return HSA_STATUS_SUCCESS;
  }

  return HSA_STATUS_ERROR;
}

hsa_status_t GpuDriver::Open() {
  if (IsForkedChild()) {
    ClearAfterFork();
  }

  // Lazily allocate process state on first Open.
  if (!process_state_) {
    process_state_ = new ProcessState();
  }

  // Reuse existing process-level KFD fd if already open.
  if (process_state_->kfd_fd >= 0) {
    fd_ = process_state_->kfd_fd;
    return HSA_STATUS_SUCCESS;
  }

  process_state_->kfd_fd = open("/dev/kfd", O_RDWR | O_CLOEXEC);
  if (process_state_->kfd_fd < 0) return HSA_STATUS_ERROR;

  fd_ = process_state_->kfd_fd;
  process_state_->pid = getpid();

  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::Close() {
  // Process-level fd persists across driver instances.
  // Just disconnect this instance; the fd stays open.
  fd_ = -1;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::Init() {
  // Get driver version.
  kfd_ioctl_get_version_args ver_args = {};
  if (GpuIoctl(fd_, AMDKFD_IOC_GET_VERSION, &ver_args) != 0)
    return HSA_STATUS_ERROR;

  version_.KernelInterfaceMajorVersion = ver_args.major_version;
  version_.KernelInterfaceMinorVersion = ver_args.minor_version;

  if (version_.KernelInterfaceMajorVersion == version_major_min &&
      version_.KernelInterfaceMinorVersion < version_minor_min)
    return HSA_STATUS_ERROR;

  core::Runtime::runtime_singleton_->KfdVersion(version_);

  if (version_.KernelInterfaceMajorVersion == 1 &&
      version_.KernelInterfaceMinorVersion == 0)
    core::g_use_interrupt_wait = false;

  bool xnack_mode = BindXnackMode(fd_);
  core::Runtime::runtime_singleton_->XnackEnabled(xnack_mode);

  // Build GPU ID mapping from sysfs topology.
  auto nodes_path = kKfdSysfsPath / "nodes";
  uint32_t sysfs_node_count = SysfsCountSubdirs(nodes_path);

  gpu_ids_.resize(sysfs_node_count, 0);
  all_gpu_id_array_.clear();
  doorbells_.resize(sysfs_node_count);
  is_dgpu_ = false;

  for (uint32_t i = 0; i < sysfs_node_count; ++i) {
    uint64_t gpu_id = 0;
    SysfsReadUint(nodes_path / std::to_string(i) / "gpu_id", gpu_id);
    gpu_ids_[i] = static_cast<uint32_t>(gpu_id);
    if (gpu_id != 0) {
      all_gpu_id_array_.push_back(static_cast<uint32_t>(gpu_id));
      // Detect dGPU: any node with compute cores but no CPU cores means
      // the system has a discrete GPU (matches libhsakmt's hsakmt_is_dgpu).
      if (!is_dgpu_) {
        auto nprops = SysfsReadProperties(nodes_path / std::to_string(i) / "properties");
        auto cpu_it = nprops.find("cpu_cores_count");
        auto fcu_it = nprops.find("simd_count");
        uint64_t cpu_cores = (cpu_it != nprops.end()) ? cpu_it->second : 0;
        uint64_t simd_count = (fcu_it != nprops.end()) ? fcu_it->second : 0;
        if (cpu_cores == 0 && simd_count > 0)
          is_dgpu_ = true;
      }
    }
  }

  // Acquire VM for each GPU via DRM render node (once per process).
  // IMPORTANT: ACQUIRE_VM must happen BEFORE RUNTIME_ENABLE so that
  // per-device process data (pdds) exist in the kernel. RUNTIME_ENABLE
  // iterates pdds and calls kfd_dbg_set_mes_debug_mode which sends
  // SET_SHADER_DEBUGGER to MES firmware — this clears stale process
  // context data in MES. Without pdds, this cleanup is skipped, and
  // MES can enter an unrecoverable state when handling subsequent
  // VM faults (matching libhsakmt ordering: open → topology/acquire_vm
  // → runtime_enable).
  // DRM render fds and ACQUIRE_VM persist in process_state_ across
  // init/shutdown cycles, matching libhsakmt behavior.
  if (!process_state_->vm_acquired) {
    process_state_->drm_render_fds.resize(sysfs_node_count, -1);
    process_state_->amdgpu_handles.resize(sysfs_node_count, nullptr);

    for (uint32_t i = 0; i < sysfs_node_count; ++i) {
      if (gpu_ids_[i] == 0) continue;

      auto props = SysfsReadProperties(nodes_path / std::to_string(i) / "properties");
      auto it = props.find("drm_render_minor");
      if (it == props.end()) continue;

      int render_minor = static_cast<int>(it->second);
      std::string render_path = "/dev/dri/renderD" + std::to_string(render_minor);

      int drm_fd = open(render_path.c_str(), O_RDWR | O_CLOEXEC);
      if (drm_fd < 0) continue;

      // Initialize amdgpu device handle and cache it. Matches libhsakmt's
      // hsakmt_open_drm_render_device which caches per render minor.
      uint32_t major_drm = 0, minor_drm = 0;
      amdgpu_device_handle amdgpu_dev = nullptr;
      if (amdgpu_device_initialize(drm_fd, &major_drm, &minor_drm,
                                    &amdgpu_dev) == 0) {
        // Use libdrm's internal fd if available (matches libhsakmt pattern).
        int libdrm_fd = amdgpu_device_get_fd(amdgpu_dev);
        if (libdrm_fd > 0) {
          close(drm_fd);
          drm_fd = libdrm_fd;
        }
        process_state_->amdgpu_handles[i] = amdgpu_dev;
      }

      kfd_ioctl_acquire_vm_args acq_args = {};
      acq_args.drm_fd = static_cast<uint32_t>(drm_fd);
      acq_args.gpu_id = gpu_ids_[i];
      if (GpuIoctl(fd_, AMDKFD_IOC_ACQUIRE_VM, &acq_args) != 0) {
        debug_print("ACQUIRE_VM failed: gpu_id=%u drm_fd=%d errno=%d\n",
                    gpu_ids_[i], drm_fd, errno);
        if (amdgpu_dev) {
          amdgpu_device_deinitialize(amdgpu_dev);
          process_state_->amdgpu_handles[i] = nullptr;
        } else {
          close(drm_fd);
        }
        continue;
      }
      process_state_->drm_render_fds[i] = drm_fd;
    }
    process_state_->vm_acquired = true;
  }

  // Runtime enable. Must be called AFTER ACQUIRE_VM so that kernel pdds
  // exist — the kernel's runtime_enable iterates pdds and calls
  // kfd_dbg_set_mes_debug_mode (SET_SHADER_DEBUGGER) which clears stale
  // MES process context. Without this, MES can hang on VM faults.
  kfd_ioctl_runtime_enable_args re_args = {};
  re_args.r_debug = reinterpret_cast<uint64_t>(&_amdgpu_r_debug);
  re_args.mode_mask = KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK;
  if (core::Runtime::runtime_singleton_->flag().debug())
    re_args.mode_mask |= KFD_RUNTIME_ENABLE_MODE_TTMP_SAVE_MASK;

  int ret = GpuIoctl(fd_, AMDKFD_IOC_RUNTIME_ENABLE, &re_args);
  bool runtime_enable_supported = (ret == 0);

  if (ret != 0 && errno != ENOSYS && errno != EINVAL)
    return HSA_STATUS_ERROR;

  runtime_caps_mask_ = re_args.capabilities_mask;

  core::Runtime::runtime_singleton_->KfdVersion(
      runtime_enable_supported,
      !!(runtime_caps_mask_ & KFD_RUNTIME_ENABLE_CAPS_SUPPORTS_CORE_DUMP_MASK));

  // Query per-GPU process apertures (LDS, scratch, GPUVM addresses).
  gpu_apertures_.resize(sysfs_node_count);
  {
    uint32_t num_gpus = static_cast<uint32_t>(all_gpu_id_array_.size());
    if (num_gpus > 0) {
      std::vector<kfd_process_device_apertures> apertures(num_gpus);
      kfd_ioctl_get_process_apertures_new_args ap_args = {};
      ap_args.kfd_process_device_apertures_ptr =
          reinterpret_cast<uint64_t>(apertures.data());
      ap_args.num_of_nodes = num_gpus;

      if (GpuIoctl(fd_, AMDKFD_IOC_GET_PROCESS_APERTURES_NEW, &ap_args) == 0) {
        for (uint32_t a = 0; a < ap_args.num_of_nodes; ++a) {
          // Find the node_id for this gpu_id.
          for (uint32_t n = 0; n < sysfs_node_count; ++n) {
            if (gpu_ids_[n] == apertures[a].gpu_id) {
              gpu_apertures_[n].lds_base = apertures[a].lds_base;
              gpu_apertures_[n].lds_limit = apertures[a].lds_limit;
              gpu_apertures_[n].scratch_base = apertures[a].scratch_base;
              gpu_apertures_[n].scratch_limit = apertures[a].scratch_limit;
              gpu_apertures_[n].gpuvm_base = apertures[a].gpuvm_base;
              gpu_apertures_[n].gpuvm_limit = apertures[a].gpuvm_limit;

              // Set memory policy for this GPU: default non-coherent,
              // alternate coherent for the GPUVM aperture.
              if (apertures[a].gpuvm_limit != 0) {
                kfd_ioctl_set_memory_policy_args mp_args = {};
                mp_args.gpu_id = apertures[a].gpu_id;
                mp_args.default_policy = KFD_IOC_CACHE_POLICY_NONCOHERENT;
                mp_args.alternate_policy = KFD_IOC_CACHE_POLICY_COHERENT;
                mp_args.alternate_aperture_base = apertures[a].gpuvm_base;
                mp_args.alternate_aperture_size =
                    apertures[a].gpuvm_limit - apertures[a].gpuvm_base + 1;
                GpuIoctl(fd_, AMDKFD_IOC_SET_MEMORY_POLICY, &mp_args);
              }
              break;
            }
          }
        }
      }
    }
  }

  // Events page mapping persists across init/shutdown cycles (the kernel
  // keeps GPU page table entries alive as long as the BO is mapped).
  // Only re-map if the mapping was lost (e.g., after GPU reset).
  // Matching libhsakmt behavior: never re-maps the events page.

  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::ShutDown() {
  // Disable runtime.
  // Note: RuntimeDisable sends SET_SHADER_DEBUGGER to MES for each pdd.
  // The old driver (via libhsakmt) does the same call here.
  kfd_ioctl_runtime_enable_args re_args = {};
  re_args.mode_mask = 0;
  int disable_ret = GpuIoctl(fd_, AMDKFD_IOC_RUNTIME_ENABLE, &re_args);
  if (disable_ret != 0)
    debug_print("ShutDown: RUNTIME_DISABLE failed errno=%d\n", errno);

  // Free all tracked allocations (mirrors libhsakmt's fmm_clear_all_aperture).
  for (auto& [addr, mh] : mem_handles_) {
    if (mh.handle != 0) {
      HsaMemFlags flags;
      flags.Value = mh.mflags;
      if (flags.ui32.NoAddress) {
        // NoAddress allocations only mapped a token page — munmap that, then
        // free the KFD handle.
        munmap(addr, kPageSize);
        kfd_ioctl_free_memory_of_gpu_args free_args = {};
        free_args.handle = mh.handle;
        GpuIoctl(fd_, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &free_args);
      } else {
        if (!mh.mapped_gpu_ids.empty()) {
          MakeDriverMemoryUnresident(fd_, addr, mh.handle,
                                     mh.mapped_gpu_ids.size(),
                                     mh.mapped_gpu_ids.data());
        }
        FreeDriverMemory(fd_, addr, mh.handle, mh.size);
      }
    } else {
      // Anonymous mmap (e.g., scratch backing).
      munmap(addr, mh.size);
    }
  }
  mem_handles_.clear();

  // Unmap doorbells: GPUVM doorbells need unmap-from-GPU + free BO;
  // APU doorbells just need munmap. Matches libhsakmt's
  // hsakmt_destroy_process_doorbells.
  for (auto& db : doorbells_) {
    if (!db.mapping || !db.size) continue;
    if (db.use_gpuvm) {
      // Unmap from GPU, then free the GPUVM BO.
      MakeDriverMemoryUnresident(fd_, db.mapping, db.handle,
                                  1, &db.gpu_id);
      kfd_ioctl_free_memory_of_gpu_args free_args = {};
      free_args.handle = db.handle;
      GpuIoctl(fd_, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &free_args);
      // munmap the CPU mapping (MAP_FIXED over the GPUVM VA).
      munmap(db.mapping, db.size);
    } else {
      munmap(db.mapping, db.size);
    }
  }
  doorbells_.clear();

  // Clear per-instance state.
  gpu_ids_.clear();
  all_gpu_id_array_.clear();
  gpu_apertures_.clear();

  // Process-level state (KFD fd, DRM fds, ACQUIRE_VM, events page) persists
  // across init/shutdown cycles. The KFD process can't be recreated because
  // the MMU notifier holds a reference; closing and reopening /dev/kfd just
  // returns the same kernel process. Do NOT close fds here.
  fd_ = -1;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::QueryKernelModeDriver(core::DriverQuery query) {
  return HSA_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
// Topology (sysfs readers)
///////////////////////////////////////////////////////////////////////////////

hsa_status_t GpuDriver::GetSystemProperties(HsaSystemProperties& sys_props) const {
  memset(&sys_props, 0, sizeof(sys_props));

  auto props = SysfsReadProperties(kKfdSysfsPath / "system_properties");

  auto get = [&](const char* key) -> uint64_t {
    auto it = props.find(key);
    return (it != props.end()) ? it->second : 0;
  };

  sys_props.PlatformOem = static_cast<uint32_t>(get("platform_oem"));
  sys_props.PlatformId = static_cast<uint32_t>(get("platform_id"));
  sys_props.PlatformRev = static_cast<uint32_t>(get("platform_rev"));
  sys_props.NumNodes = SysfsCountSubdirs(kKfdSysfsPath / "nodes");

  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::GetNodeProperties(HsaNodeProperties& node_props,
                                           uint32_t node_id) const {
  memset(&node_props, 0, sizeof(node_props));

  auto node_path = kKfdSysfsPath / "nodes" / std::to_string(node_id);
  auto props = SysfsReadProperties(node_path / "properties");
  if (props.empty()) return HSA_STATUS_ERROR;

  auto get = [&](const char* key, uint64_t def = 0) -> uint64_t {
    auto it = props.find(key);
    return (it != props.end()) ? it->second : def;
  };

  node_props.NumCPUCores = static_cast<uint32_t>(get("cpu_cores_count"));
  node_props.NumFComputeCores = static_cast<uint32_t>(get("simd_count"));
  node_props.NumMemoryBanks = static_cast<uint32_t>(get("mem_banks_count"));

  // Add synthetic memory banks for GPU nodes (LDS, scratch, SVM).
  // These are not in sysfs; they are populated from kernel aperture data.
  if (NodeToGpuId(node_id) != 0) {
    node_props.NumMemoryBanks += 3;  // LDS + scratch + SVM
  }

  node_props.NumCaches = static_cast<uint32_t>(get("caches_count"));
  node_props.NumIOLinks = static_cast<uint32_t>(get("io_links_count")) +
                           static_cast<uint32_t>(get("p2p_links_count"));
  node_props.CComputeIdLo = static_cast<uint32_t>(get("cpu_core_id_base"));
  node_props.FComputeIdLo = static_cast<uint32_t>(get("simd_id_base"));
  node_props.Capability.Value = static_cast<uint32_t>(get("capability"));
  node_props.MaxWavesPerSIMD = static_cast<uint32_t>(get("max_waves_per_simd"));
  node_props.LDSSizeInKB = static_cast<uint32_t>(get("lds_size_in_kb"));
  node_props.GDSSizeInKB = static_cast<uint32_t>(get("gds_size_in_kb"));
  node_props.WaveFrontSize = static_cast<uint32_t>(get("wave_front_size"));
  node_props.NumArrays = static_cast<uint32_t>(get("simd_arrays_per_engine"));
  // NumShaderBanks = array_count / simd_arrays_per_engine (number of shader
  // engines). Matches libhsakmt's topology.c:1268.
  {
    uint32_t simd_arrays_count = static_cast<uint32_t>(get("array_count"));
    node_props.NumShaderBanks = (node_props.NumArrays != 0)
        ? simd_arrays_count / node_props.NumArrays
        : simd_arrays_count;
  }
  node_props.NumCUPerArray = static_cast<uint32_t>(get("cu_per_simd_array"));
  node_props.NumSIMDPerCU = static_cast<uint32_t>(get("simd_per_cu"));
  node_props.MaxSlotsScratchCU = static_cast<uint32_t>(get("max_slots_scratch_cu"));
  node_props.VendorId = static_cast<uint32_t>(get("vendor_id"));
  node_props.DeviceId = static_cast<uint32_t>(get("device_id"));
  node_props.LocationId = static_cast<uint32_t>(get("location_id"));
  node_props.Domain = static_cast<uint32_t>(get("domain"));
  node_props.MaxEngineClockMhzFCompute = static_cast<uint32_t>(get("max_engine_clk_fcompute"));
  node_props.MaxEngineClockMhzCCompute = static_cast<uint32_t>(get("max_engine_clk_ccompute"));
  node_props.LocalMemSize = get("local_mem_size");
  node_props.EngineId.Value = static_cast<uint32_t>(get("fw_version")) & 0x3ff;
  node_props.FamilyID = static_cast<uint32_t>(get("family_id"));

  // Parse gfx_target_version into EngineId Major/Minor/Stepping.
  // Format: decimal MMMMNNSS (e.g. 110002 → Major=11, Minor=0, Stepping=2).
  uint32_t gfxv = static_cast<uint32_t>(get("gfx_target_version"));
  if (gfxv) {
    uint32_t gfxv_major = (gfxv / 10000) % 100;
    uint32_t gfxv_minor = (gfxv / 100) % 100;
    uint32_t gfxv_stepping = gfxv % 100;

    // Check for HSA_OVERRIDE_GFX_VERSION[_<node_id>] env var.
    char per_node_override[64];
    snprintf(per_node_override, sizeof(per_node_override),
             "HSA_OVERRIDE_GFX_VERSION_%d", node_id);
    const char* envvar = getenv(per_node_override);
    if (!envvar) envvar = getenv("HSA_OVERRIDE_GFX_VERSION");
    if (envvar) {
      unsigned major, minor, step;
      char dummy;
      if (sscanf(envvar, "%u.%u.%u%c", &major, &minor, &step, &dummy) == 3 &&
          major <= 63 && minor <= 255 && step <= 255) {
        node_props.OverrideEngineId.ui32.Major = major & 0x3f;
        node_props.OverrideEngineId.ui32.Minor = minor & 0xff;
        node_props.OverrideEngineId.ui32.Stepping = step & 0xff;
      }
    }

    node_props.EngineId.ui32.Major = gfxv_major & 0x3f;
    node_props.EngineId.ui32.Minor = gfxv_minor & 0xff;
    node_props.EngineId.ui32.Stepping = gfxv_stepping & 0xff;

    // Compute VGPR/SGPR register file sizes per CU.
    uint32_t gfxv_full = (gfxv_major << 16) | (gfxv_minor << 8) | gfxv_stepping;
    node_props.SGPRSizePerCU = 0x4000;

    // 512KB for MI-class GPUs, 384KB for RDNA3.5/4, 256KB default.
    // Packed gfxv format: (major << 16) | (minor << 8) | stepping.
    // Matches libhsakmt's hsakmt_get_vgpr_size_per_cu.
    if (gfxv_full == 0x090500 ||                     // gfx950
        (gfxv_full & ~0xff) == 0x090400 ||           // gfx94x (Aqua Vanjaram)
        gfxv_full == 0x09000A ||                     // Aldebaran (gfx90a)
        gfxv_full == 0x090008)                       // Arcturus (gfx908)
      node_props.VGPRSizePerCU = 0x80000;
    else if (gfxv_full == 0x0B0000 ||                // Plum Bonito (gfx1100)
             gfxv_full == 0x0B0001 ||                // Wheat Nas (gfx1101)
             gfxv_full == 0x0c0000 ||                // gfx1200
             gfxv_full == 0x0c0001)                  // gfx1201
      node_props.VGPRSizePerCU = 0x60000;
    else
      node_props.VGPRSizePerCU = 0x40000;
  }

  node_props.uCodeEngineVersions.Value = static_cast<uint32_t>(get("sdma_fw_version")) & 0x3ff;
  node_props.NumSdmaEngines = static_cast<uint32_t>(get("num_sdma_engines"));
  node_props.NumSdmaXgmiEngines = static_cast<uint32_t>(get("num_sdma_xgmi_engines"));
  node_props.NumSdmaQueuesPerEngine = static_cast<uint32_t>(get("num_sdma_queues_per_engine"));
  node_props.NumGws = static_cast<uint32_t>(get("num_gws"));
  node_props.NumCpQueues = static_cast<uint32_t>(get("num_cp_queues"));
  node_props.CwsrSize = static_cast<uint32_t>(get("cwsr_size"));
  node_props.CtlStackSize = static_cast<uint32_t>(get("ctl_stack_size"));
  node_props.DrmRenderMinor = static_cast<int32_t>(get("drm_render_minor", static_cast<uint64_t>(-1)));
  node_props.NumXcc = static_cast<uint32_t>(get("num_xcc", 1));
  node_props.Capability2.Value = static_cast<uint32_t>(get("capability2"));
  node_props.DebugProperties.Value = static_cast<uint32_t>(get("debug_prop"));
  node_props.UniqueID = get("unique_id");
  node_props.HiveID = get("hive_id");
  node_props.KFDGpuID = NodeToGpuId(node_id);

  // Marketing name.
  std::ifstream name_file(node_path / "name");
  if (name_file.is_open()) {
    std::string name;
    std::getline(name_file, name);
    for (size_t i = 0; i < name.size() && i < HSA_PUBLIC_NAME_SIZE - 1; ++i)
      node_props.MarketingName[i] = static_cast<HSAuint16>(name[i]);
    node_props.MarketingName[std::min(name.size(),
        static_cast<size_t>(HSA_PUBLIC_NAME_SIZE - 1))] = 0;
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::GetEdgeProperties(
    std::vector<HsaIoLinkProperties>& io_link_props, uint32_t node_id) const {
  auto node_path = kKfdSysfsPath / "nodes" / std::to_string(node_id);
  const fs::path dirs[] = {node_path / "io_links", node_path / "p2p_links"};

  uint32_t idx = 0;
  for (const auto& dir : dirs) {
    for (uint32_t i = 0; idx < io_link_props.size(); ++i) {
      auto props = SysfsReadProperties(dir / std::to_string(i) / "properties");
      if (props.empty()) break;

      auto& link = io_link_props[idx];
      memset(&link, 0, sizeof(link));

      auto get = [&](const char* key, uint64_t def = 0) -> uint64_t {
        auto it = props.find(key);
        return (it != props.end()) ? it->second : def;
      };

      link.IoLinkType = static_cast<HSA_IOLINKTYPE>(get("type"));
      link.VersionMajor = static_cast<uint32_t>(get("version_major"));
      link.VersionMinor = static_cast<uint32_t>(get("version_minor"));
      link.NodeFrom = static_cast<uint32_t>(get("node_from"));
      link.NodeTo = static_cast<uint32_t>(get("node_to"));
      link.Weight = static_cast<uint32_t>(get("weight"));
      link.MinimumLatency = static_cast<uint32_t>(get("min_latency"));
      link.MaximumLatency = static_cast<uint32_t>(get("max_latency"));
      link.MinimumBandwidth = static_cast<uint32_t>(get("min_bandwidth"));
      link.MaximumBandwidth = static_cast<uint32_t>(get("max_bandwidth"));
      link.RecTransferSize = static_cast<uint32_t>(get("recommended_transfer_size"));
      link.RecSdmaEngIdMask =
          static_cast<uint32_t>(get("recommended_sdma_engine_id_mask"));
      link.Flags.LinkProperty = static_cast<uint32_t>(get("flags"));
      ++idx;
    }
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::GetMemoryProperties(
    uint32_t node_id, std::vector<HsaMemoryProperties>& mem_props) const {
  if (!mem_props.data()) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  auto banks_path = kKfdSysfsPath / "nodes" / std::to_string(node_id) / "mem_banks";

  // Read actual hardware memory banks from sysfs.
  uint32_t i = 0;
  for (; i < mem_props.size(); ++i) {
    auto props = SysfsReadProperties(banks_path / std::to_string(i) / "properties");
    if (props.empty()) break;

    auto& mem = mem_props[i];
    memset(&mem, 0, sizeof(mem));

    auto get = [&](const char* key, uint64_t def = 0) -> uint64_t {
      auto it = props.find(key);
      return (it != props.end()) ? it->second : def;
    };

    mem.HeapType = static_cast<HSA_HEAPTYPE>(get("heap_type"));
    mem.SizeInBytes = get("size_in_bytes");
    mem.Flags.MemoryProperty = static_cast<uint32_t>(get("flags"));
    mem.Width = static_cast<uint32_t>(get("width"));
    mem.MemoryClockMax = static_cast<uint32_t>(get("mem_clk_max"));
    mem.VirtualBaseAddress = get("virtual_base_address");
  }

  // Synthesize virtual memory regions for GPU nodes using kernel apertures.
  uint32_t gpu_id = NodeToGpuId(node_id);
  if (gpu_id != 0 && node_id < gpu_apertures_.size()) {
    const auto& ap = gpu_apertures_[node_id];

    // Read LDSSizeInKB from sysfs for the LDS entry size.
    auto node_props = SysfsReadProperties(
        kKfdSysfsPath / "nodes" / std::to_string(node_id) / "properties");
    uint64_t lds_size_kb = 0;
    {
      auto it = node_props.find("lds_size_in_kb");
      if (it != node_props.end()) lds_size_kb = it->second;
    }

    // Add LDS region.
    if (i < mem_props.size() && ap.lds_base != 0) {
      memset(&mem_props[i], 0, sizeof(mem_props[i]));
      mem_props[i].HeapType = HSA_HEAPTYPE_GPU_LDS;
      mem_props[i].SizeInBytes = lds_size_kb * 1024;
      mem_props[i].VirtualBaseAddress = ap.lds_base;
      ++i;
    }

    // Add scratch region.
    if (i < mem_props.size() && ap.scratch_base != 0) {
      memset(&mem_props[i], 0, sizeof(mem_props[i]));
      mem_props[i].HeapType = HSA_HEAPTYPE_GPU_SCRATCH;
      mem_props[i].SizeInBytes = (ap.scratch_limit - ap.scratch_base) + 1;
      mem_props[i].VirtualBaseAddress = ap.scratch_base;
      ++i;
    }

    // Add SVM aperture (for GPUs that need non-canonical GPUVM aperture).
    if (i < mem_props.size() && ap.gpuvm_base != 0 && ap.gpuvm_limit != 0) {
      memset(&mem_props[i], 0, sizeof(mem_props[i]));
      mem_props[i].HeapType = HSA_HEAPTYPE_DEVICE_SVM;
      mem_props[i].SizeInBytes = (ap.gpuvm_limit - ap.gpuvm_base) + 1;
      mem_props[i].VirtualBaseAddress = ap.gpuvm_base;
      ++i;
    }
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::GetCacheProperties(
    uint32_t node_id, uint32_t processor_id,
    std::vector<HsaCacheProperties>& cache_props) const {
  if (!cache_props.data()) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  auto caches_path = kKfdSysfsPath / "nodes" / std::to_string(node_id) / "caches";

  for (uint32_t i = 0; i < cache_props.size(); ++i) {
    auto props = SysfsReadProperties(caches_path / std::to_string(i) / "properties");
    if (props.empty()) break;

    auto& cache = cache_props[i];
    memset(&cache, 0, sizeof(cache));

    auto get = [&](const char* key, uint64_t def = 0) -> uint64_t {
      auto it = props.find(key);
      return (it != props.end()) ? it->second : def;
    };

    cache.ProcessorIdLow = static_cast<uint32_t>(get("processor_id_low"));
    cache.CacheLevel = static_cast<uint32_t>(get("level"));
    cache.CacheSize = static_cast<uint32_t>(get("size"));
    cache.CacheLineSize = static_cast<uint32_t>(get("cache_line_size"));
    cache.CacheLinesPerTag = static_cast<uint32_t>(get("cache_lines_per_tag"));
    cache.CacheAssociativity = static_cast<uint32_t>(get("association"));
    cache.CacheLatency = static_cast<uint32_t>(get("latency"));
    cache.CacheType.Value = static_cast<uint32_t>(get("type"));

    // sibling_map is embedded in the properties file (not a separate file).
    // Parse it by reading the raw file and extracting comma/newline-separated
    // uint32 values after the "sibling_map" key. Matches libhsakmt's
    // topology_sysfs_get_cache_props.
    {
      auto prop_path = caches_path / std::to_string(i) / "properties";
      std::ifstream f(prop_path);
      if (f.is_open()) {
        std::string line;
        while (std::getline(f, line)) {
          if (line.compare(0, 11, "sibling_map") == 0) {
            // Values start after "sibling_map " — may be space or tab separated.
            const char* p = line.c_str() + 11;
            while (*p == ' ' || *p == '\t') ++p;
            uint32_t j = 0;
            unsigned val;
            int n;
            while (j < HSA_CPU_SIBLINGS &&
                   sscanf(p, "%u%n", &val, &n) == 1) {
              cache.SiblingMap[j++] = val;
              p += n;
              // Skip comma/whitespace separators.
              while (*p == ',' || *p == ' ' || *p == '\t') ++p;
            }
            // Values may continue on subsequent lines.
            while (j < HSA_CPU_SIBLINGS && std::getline(f, line)) {
              p = line.c_str();
              while (*p == ' ' || *p == '\t') ++p;
              if (*p == '\0') break;
              while (j < HSA_CPU_SIBLINGS &&
                     sscanf(p, "%u%n", &val, &n) == 1) {
                cache.SiblingMap[j++] = val;
                p += n;
                while (*p == ',' || *p == ' ' || *p == '\t') ++p;
              }
            }
            break;
          }
        }
      }
    }
  }

  return HSA_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
// Memory management
///////////////////////////////////////////////////////////////////////////////

hsa_status_t
GpuDriver::AllocateMemory(const core::MemoryRegion &mem_region,
                          core::MemoryRegion::AllocateFlags alloc_flags,
                          void **mem, size_t size, uint32_t agent_node_id) {
  const MemoryRegion &m_region(static_cast<const MemoryRegion &>(mem_region));
  HsaMemFlags kmt_alloc_flags(m_region.mem_flags());

  kmt_alloc_flags.ui32.ExecuteAccess =
      (alloc_flags & core::MemoryRegion::AllocateExecutable ? 1 : 0);

  if (m_region.IsSystem() &&
      (alloc_flags & core::MemoryRegion::AllocateNonPaged)) {
    kmt_alloc_flags.ui32.NonPaged = 1;
  }

  if (!m_region.IsLocalMemory() &&
      (alloc_flags & core::MemoryRegion::AllocateMemoryOnly)) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  kmt_alloc_flags.ui32.NoAddress =
      !!(alloc_flags & core::MemoryRegion::AllocateMemoryOnly);

  kmt_alloc_flags.ui32.CoarseGrain =
      (alloc_flags & core::MemoryRegion::AllocatePCIeRW
           ? 0
           : kmt_alloc_flags.ui32.CoarseGrain);

  kmt_alloc_flags.ui32.NoSubstitute =
      (alloc_flags & core::MemoryRegion::AllocatePinned
           ? 1
           : kmt_alloc_flags.ui32.NoSubstitute);

  kmt_alloc_flags.ui32.GTTAccess =
      (alloc_flags & core::MemoryRegion::AllocateGTTAccess
           ? 1
           : kmt_alloc_flags.ui32.GTTAccess);

  kmt_alloc_flags.ui32.Uncached =
      (alloc_flags & core::MemoryRegion::AllocateUncached
            ? 1
            : kmt_alloc_flags.ui32.Uncached);

  kmt_alloc_flags.ui32.QueueObject =
      (alloc_flags & core::MemoryRegion::AllocateQueueObject ? 1
                                                             : kmt_alloc_flags.ui32.QueueObject);
  if (kmt_alloc_flags.ui32.Uncached) {
    kmt_alloc_flags.ui32.CoarseGrain = 0;
    kmt_alloc_flags.ui32.ExtendedCoherent = 0;
  }

  kmt_alloc_flags.ui32.ExecuteBlit =
    !!(alloc_flags & core::MemoryRegion::AllocateExecutableBlitKernelObject);

  if (m_region.IsLocalMemory()) {
    kmt_alloc_flags.ui32.Contiguous =
        (alloc_flags & core::MemoryRegion::AllocateContiguous
             ? 1
             : kmt_alloc_flags.ui32.Contiguous);
  }

  // Sub-allocator path for ordinary VRAM.
  if (m_region.IsLocalMemory() && !kmt_alloc_flags.ui32.NoAddress) {
    bool subAllocEnabled =
        !core::Runtime::runtime_singleton_->flag().disable_fragment_alloc();
    bool useSubAlloc = subAllocEnabled;
    useSubAlloc &=
        ((alloc_flags & (~core::MemoryRegion::AllocateRestrict)) == 0);

    if (useSubAlloc) {
      *mem = m_region.fragment_alloc(size);
      return HSA_STATUS_SUCCESS;
    }
  }

  const uint32_t node_id =
      (alloc_flags & core::MemoryRegion::AllocateGTTAccess)
          ? agent_node_id
          : m_region.owner()->node_id();

  uint32_t gpu_id = NodeToGpuId(node_id);

  // For system memory owned by a CPU node (gpu_id == 0), use the first
  // available GPU's id. The kernel needs a valid gpu_id for GTT allocations.
  if (gpu_id == 0 && !all_gpu_id_array_.empty())
    gpu_id = all_gpu_id_array_[0];

  // Allocate memory.
  uint64_t handle = 0;
  int drm_fd = DrmFdForGpuId(gpu_id);
  bool is_device = m_region.IsLocalMemory();
  *mem = AllocateDriverMemory(fd_, drm_fd, kmt_alloc_flags, gpu_id, size,
                              &handle, is_device);
  if (*mem == nullptr) {
    m_region.owner()->Trim();
    *mem = AllocateDriverMemory(fd_, drm_fd, kmt_alloc_flags, gpu_id, size,
                                &handle, is_device);
  }

  if (*mem != nullptr) {
    // Track the allocation (not yet mapped to GPU).
    mem_handles_[*mem] = {handle, size, node_id, kmt_alloc_flags.Value, {}};

    if (kmt_alloc_flags.ui32.NoAddress)
      return HSA_STATUS_SUCCESS;

    HsaMemMapFlags map_flag = m_region.map_flags();
    size_t map_node_count = 1;
    const uint32_t owner_gpu_id = NodeToGpuId(m_region.owner()->node_id());
    const uint32_t *map_gpu_ids = &owner_gpu_id;

    if (m_region.IsSystem()) {
      if ((alloc_flags & core::MemoryRegion::AllocateRestrict) == 0) {
        map_node_count = all_gpu_id_array_.size();
        if (map_node_count == 0) return HSA_STATUS_SUCCESS;
        map_gpu_ids = all_gpu_id_array_.data();
      } else {
        return HSA_STATUS_SUCCESS;
      }
    }

    uint64_t alternate_va = 0;
    const bool is_resident = MakeDriverMemoryResident(
        fd_, map_node_count, map_gpu_ids, *mem, handle, size,
        &alternate_va, map_flag);

    if (is_resident)
      mem_handles_[*mem].mapped_gpu_ids.assign(map_gpu_ids,
                                               map_gpu_ids + map_node_count);

    const bool require_pinning =
        (!m_region.full_profile() || m_region.IsLocalMemory() ||
         m_region.IsScratch());

    if (require_pinning && !is_resident) {
      debug_print("AllocMem: map failed, freeing. node_id=%u gpu_id=%u size=%zu full_prof=%d local=%d scratch=%d\n",
                  node_id, gpu_id, size, m_region.full_profile(), m_region.IsLocalMemory(), m_region.IsScratch());
      FreeDriverMemory(fd_, *mem, handle, size);
      mem_handles_.erase(*mem);
      *mem = nullptr;
      return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }

    return HSA_STATUS_SUCCESS;
  }

  return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
}

hsa_status_t GpuDriver::FreeMemory(void *mem, size_t size) {
  auto it = mem_handles_.find(mem);
  if (it != mem_handles_.end()) {
    if (it->second.handle == 0) {
      // Anonymous mmap (e.g., scratch backing) — just munmap.
      munmap(mem, it->second.size);
      mem_handles_.erase(it);
      return HSA_STATUS_SUCCESS;
    }
    // NoAddress allocations (VRAM-only handles) use a placeholder token page.
    // Skip GPU unmap (was never mapped) and only free the token + KFD handle.
    HsaMemFlags flags;
    flags.Value = it->second.mflags;
    if (flags.ui32.NoAddress) {
      munmap(mem, kPageSize);
      kfd_ioctl_free_memory_of_gpu_args free_args = {};
      free_args.handle = it->second.handle;
      GpuIoctl(fd_, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &free_args);
      mem_handles_.erase(it);
      return HSA_STATUS_SUCCESS;
    }

    if (!it->second.mapped_gpu_ids.empty()) {
      MakeDriverMemoryUnresident(fd_, mem, it->second.handle,
                                 it->second.mapped_gpu_ids.size(),
                                 it->second.mapped_gpu_ids.data());
    }
    // Use tracked size for correct munmap (more reliable than caller-provided size).
    bool ok = FreeDriverMemory(fd_, mem, it->second.handle, it->second.size);
    mem_handles_.erase(it);
    return ok ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
  }

  // Fallback: try to free without tracked handle.
  debug_print("FreeMemory: untracked ptr:%p size:%lu\n", mem, size);
  return HSA_STATUS_ERROR;
}

hsa_status_t GpuDriver::AllocateScratchMemory(uint32_t node_id, uint64_t size,
                                               void** mem) const {
  assert(mem);
  assert(size > 0);

  // Scratch backing is allocated as anonymous memory (not via ALLOC_MEMORY_OF_GPU).
  // The GPU accesses scratch through SH_HIDDEN_PRIVATE_BASE, not GPUVM translation.
  static constexpr uint64_t kScratchAlign = 0x10000;
  uint64_t aligned_size = (size + kScratchAlign - 1) & ~(kScratchAlign - 1);

  void* ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  // Track with handle=0 to indicate anonymous mmap (FreeMemory will just munmap).
  mem_handles_[ptr] = {0, static_cast<size_t>(aligned_size), node_id, 0, {}};

  // Program SH_HIDDEN_PRIVATE_BASE so the GPU knows where scratch memory is.
  uint32_t gpu_id = NodeToGpuId(node_id);
  kfd_ioctl_set_scratch_backing_va_args scratch_args = {};
  scratch_args.gpu_id = gpu_id;
  scratch_args.va_addr = reinterpret_cast<uint64_t>(ptr) >> 16;
  if (GpuIoctl(fd_, AMDKFD_IOC_SET_SCRATCH_BACKING_VA, &scratch_args) != 0) {
    debug_print("SET_SCRATCH_BACKING_VA failed: gpu_id=%u errno=%d\n",
                gpu_id, errno);
    munmap(ptr, aligned_size);
    mem_handles_.erase(ptr);
    return HSA_STATUS_ERROR;
  }

  *mem = ptr;
  return HSA_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
// Queue management
///////////////////////////////////////////////////////////////////////////////

hsa_status_t GpuDriver::CreateQueue(uint32_t node_id, HSA_QUEUE_TYPE type,
                                    uint32_t queue_pct,
                                    HSA::hsa_amd_queue_priority_internal_t priority,
                                    uint32_t sdma_engine_id, void* queue_addr,
                                    uint64_t queue_size_bytes, HsaEvent* event,
                                    HsaQueueResource& queue_resource) const {
  HSA_QUEUE_PRIORITY drv_priority = HsaInternalToDriverPriority(priority);
  uint32_t gpu_id = NodeToGpuId(node_id);
  int drm_fd = DrmFdForGpuId(gpu_id);

  // Convert HSA_QUEUE_TYPE enum to KFD ioctl queue type.
  uint32_t kfd_queue_type;
  switch (type) {
    case HSA_QUEUE_COMPUTE:
      kfd_queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE; break;
    case HSA_QUEUE_SDMA:
      kfd_queue_type = KFD_IOC_QUEUE_TYPE_SDMA; break;
    case HSA_QUEUE_SDMA_XGMI:
      kfd_queue_type = KFD_IOC_QUEUE_TYPE_SDMA_XGMI; break;
    case HSA_QUEUE_SDMA_BY_ENG_ID:
      kfd_queue_type = KFD_IOC_QUEUE_TYPE_SDMA_BY_ENG_ID; break;
    case HSA_QUEUE_COMPUTE_AQL:
      kfd_queue_type = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL; break;
    default:
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  // Map HSA_QUEUE_PRIORITY enum to KFD priority values.
  // Matches libhsakmt's priority_map[] = {0, 3, 5, 7, 9, 11, 15}.
  static constexpr uint32_t kPriorityMap[] = {0, 3, 5, 7, 9, 11, 15};
  int prio_idx = static_cast<int>(drv_priority) + 3;
  if (prio_idx < 0) prio_idx = 0;
  if (prio_idx > 6) prio_idx = 6;
  uint32_t kfd_priority = kPriorityMap[prio_idx];

  auto* qinfo = new QueueInfo();
  qinfo->gpu_id = gpu_id;

  kfd_ioctl_create_queue_args args = {};
  args.ring_base_address = reinterpret_cast<uint64_t>(queue_addr);
  args.ring_size = static_cast<uint32_t>(queue_size_bytes);
  args.gpu_id = gpu_id;
  args.queue_type = kfd_queue_type;
  args.queue_percentage = queue_pct;
  args.queue_priority = kfd_priority;
  args.sdma_engine_id = sdma_engine_id;

  // For non-AQL queues, allocate GPU-accessible memory for wptr/rptr.
  // Matches libhsakmt's queue struct allocation (queues.c:689-746).
  // AQL queues have wptr/rptr already set by the caller (amd_aql_queue.cpp).
  if (kfd_queue_type != KFD_IOC_QUEUE_TYPE_COMPUTE_AQL) {
    // Allocate a small GPU-accessible buffer for the queue control structure.
    // libhsakmt allocates sizeof(struct queue) via allocate_exec_aligned_memory,
    // which uses HostAccess+ExecuteAccess flags. We only need space for wptr/rptr.
    uint32_t qstruct_size = kPageSize;
    HsaMemFlags qstruct_flags = {};
    qstruct_flags.ui32.HostAccess = 1;
    qstruct_flags.ui32.ExecuteAccess = 1;
    qstruct_flags.ui32.NonPaged = 1;
    qstruct_flags.ui32.Uncached = 1;
    qinfo->queue_struct = AllocateDriverMemory(fd_, drm_fd, qstruct_flags,
                                                gpu_id, qstruct_size,
                                                &qinfo->queue_struct_handle);
    if (qinfo->queue_struct) {
      qinfo->queue_struct_size = qstruct_size;
      memset(qinfo->queue_struct, 0, qstruct_size);
      // Map to GPU so the device can write rptr.
      uint64_t alt_va = 0;
      HsaMemMapFlags map_flags = {};
      MakeDriverMemoryResident(fd_, 1, &gpu_id, qinfo->queue_struct,
                                qinfo->queue_struct_handle, qstruct_size,
                                &alt_va, map_flags);
      // wptr is at offset 0, rptr is at offset 8 within the struct.
      auto* base = static_cast<uint64_t*>(qinfo->queue_struct);
      queue_resource.QueueWptrValue = reinterpret_cast<uint64_t>(&base[0]);
      queue_resource.QueueRptrValue = reinterpret_cast<uint64_t>(&base[1]);
    }
  }

  args.write_pointer_address = queue_resource.QueueWptrValue;
  args.read_pointer_address = queue_resource.QueueRptrValue;

  // Compute queues need EOP buffer and context save/restore.
  bool is_compute = (kfd_queue_type == KFD_IOC_QUEUE_TYPE_COMPUTE ||
                     kfd_queue_type == KFD_IOC_QUEUE_TYPE_COMPUTE_AQL);
  if (is_compute) {
    // Allocate EOP buffer (4096 bytes of VRAM).
    uint32_t eop_size = 4096;
    HsaMemFlags eop_flags = {};
    eop_flags.ui32.ExecuteAccess = 1;
    eop_flags.ui32.CoarseGrain = 1;  // VRAM, no host access
    qinfo->eop_buffer = AllocateDriverMemory(fd_, drm_fd, eop_flags, gpu_id,
                                              eop_size, &qinfo->eop_handle);
    if (qinfo->eop_buffer) {
      qinfo->eop_size = eop_size;
      // Map EOP buffer to GPU.
      uint64_t alt_va = 0;
      HsaMemMapFlags map_flags = {};
      MakeDriverMemoryResident(fd_, 1, &gpu_id, qinfo->eop_buffer,
                                qinfo->eop_handle, eop_size, &alt_va, map_flags);
      args.eop_buffer_address = reinterpret_cast<uint64_t>(qinfo->eop_buffer);
      args.eop_buffer_size = eop_size;
    }

    // Compute context save/restore sizes.
    // Must match kernel's kfd_queue_ctx_save_restore_size() exactly.
    HsaNodeProperties nprops = {};
    GetNodeProperties(nprops, node_id);

    if (nprops.NumFComputeCores > 0 && nprops.NumSIMDPerCU > 0) {
      // Use gfx_target_version in the same format as the kernel (decimal).
      uint32_t gfxv = nprops.EngineId.ui32.Major * 10000 +
                      nprops.EngineId.ui32.Minor * 100 +
                      nprops.EngineId.ui32.Stepping;

      uint32_t num_xcc = nprops.NumXcc > 0 ? nprops.NumXcc : 1;
      uint32_t cu_num = nprops.NumFComputeCores / nprops.NumSIMDPerCU / num_xcc;
      uint32_t wave_num;

      if (gfxv < 100100)  // pre-NAVI10
        wave_num = std::min(cu_num * 40, nprops.NumShaderBanks / nprops.NumArrays * 512);
      else
        wave_num = cu_num * 32;

      // CNTL_STACK_BYTES_PER_WAVE: 12 for NAVI10+, 8 for older.
      uint32_t cntl_bytes = (gfxv >= 100100) ? 12 : 8;
      uint32_t ctl_stack_size_raw = wave_num * cntl_bytes + 8;

      // Kernel adds sizeof(HsaUserContextSaveAreaHeader) = 40 bytes.
      static constexpr uint32_t kCwsrHeaderSize = 40;
      uint32_t ctl_stack_size = (kCwsrHeaderSize + ctl_stack_size_raw +
                                  kPageSize - 1) & ~(kPageSize - 1);

      // GFX10 family caps ctl_stack to 0x7000.
      if ((gfxv / 10000 * 10000) == 100000)
        ctl_stack_size = std::min(ctl_stack_size, 0x7000u);

      // WG_CONTEXT_DATA_SIZE_PER_CU = VGPR + SGPR + LDS + HWREG.
      // Matches libhsakmt: always use node.LDSSizeInKB from topology.
      uint32_t lds_per_cu = nprops.LDSSizeInKB * 1024;
      uint32_t wg_data_per_cu = nprops.VGPRSizePerCU + nprops.SGPRSizePerCU +
                                 lds_per_cu + 0x1000;
      uint32_t wg_data_size = (cu_num * wg_data_per_cu + kPageSize - 1) & ~(kPageSize - 1);

      uint32_t cwsr_size = ctl_stack_size + wg_data_size;

      // If the kernel provides CwsrSize/CtlStackSize via sysfs, use those
      // as definitive values. Matches libhsakmt's update_ctx_save_restore_size.
      if (nprops.CwsrSize > 0)
        cwsr_size = nprops.CwsrSize;
      if (nprops.CtlStackSize > 0)
        ctl_stack_size = nprops.CtlStackSize;

      // Kernel also adds debug_memory_size to the total allocation.
      uint32_t debug_memory_size = (wave_num * 32 + 63) & ~63u;
      uint32_t total_alloc = (cwsr_size + debug_memory_size) * num_xcc;
      total_alloc = (total_alloc + kPageSize - 1) & ~(kPageSize - 1);

      // Allocate CWSR buffer via anonymous mmap + SVM registration.
      // Matches libhsakmt: 2MB-aligned allocation for TLB efficiency.
      static constexpr uint32_t kGpuHugePageSize = 2u << 20;  // 2MB
      uint32_t alloc_size = (total_alloc + kPageSize - 1) & ~(kPageSize - 1);

      // Allocate with 2MB alignment (matching hsakmt_mmap_allocate_aligned).
      // Over-allocate, then trim padding to get an aligned region.
      void* ctx_addr = MAP_FAILED;
      {
        size_t padded = alloc_size + kGpuHugePageSize - kPageSize;
        void* raw = mmap(nullptr, padded, PROT_NONE,
                         MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
        if (raw != MAP_FAILED) {
          uintptr_t aligned = (reinterpret_cast<uintptr_t>(raw) +
                               kGpuHugePageSize - 1) & ~(uintptr_t)(kGpuHugePageSize - 1);
          void* aligned_ptr = reinterpret_cast<void*>(aligned);

          // Unmap padding before aligned region.
          if (aligned_ptr > raw)
            munmap(raw, reinterpret_cast<char*>(aligned_ptr) -
                        reinterpret_cast<char*>(raw));
          // Unmap padding after aligned region.
          void* end = reinterpret_cast<void*>(aligned + alloc_size);
          void* mapping_end = reinterpret_cast<void*>(
              reinterpret_cast<uintptr_t>(raw) + padded);
          if (mapping_end > end)
            munmap(end, reinterpret_cast<char*>(mapping_end) -
                        reinterpret_cast<char*>(end));

          // Re-map with correct protection (MAP_FIXED).
          ctx_addr = mmap(aligned_ptr, alloc_size, PROT_READ | PROT_WRITE,
                          MAP_ANONYMOUS | MAP_PRIVATE | MAP_FIXED, -1, 0);
        }
      }
      if (ctx_addr != MAP_FAILED) {
        madvise(ctx_addr, alloc_size, MADV_DONTFORK);

        // Fill CWSR header per XCC with event/error info.
        for (uint32_t x = 0; x < num_xcc; x++) {
          auto* header = reinterpret_cast<HsaUserContextSaveAreaHeader*>(
              static_cast<char*>(ctx_addr) + x * cwsr_size);
          header->ErrorEventId = event ? event->EventId : 0;
          header->ErrorReason = queue_resource.ErrorReason;
          header->DebugOffset = (num_xcc - x) * cwsr_size;
          header->DebugSize = debug_memory_size * num_xcc;
        }

        // Register the CWSR buffer as an SVM range with the GPU.
        constexpr uint32_t nattr = 6;
        size_t attr_bytes = nattr * sizeof(kfd_ioctl_svm_attribute);
        size_t svm_size = sizeof(kfd_ioctl_svm_args) + attr_bytes;
        auto* svm_args = reinterpret_cast<kfd_ioctl_svm_args*>(alloca(svm_size));
        memset(svm_args, 0, svm_size);
        svm_args->start_addr = reinterpret_cast<uint64_t>(ctx_addr);
        svm_args->size = alloc_size;
        svm_args->op = KFD_IOCTL_SVM_OP_SET_ATTR;
        svm_args->nattr = nattr;

        uint32_t svm_flags = KFD_IOCTL_SVM_FLAG_HOST_ACCESS |
                              KFD_IOCTL_SVM_FLAG_GPU_EXEC |
                              KFD_IOCTL_SVM_FLAG_GPU_ALWAYS_MAPPED;
        svm_args->attrs[0] = {KFD_IOCTL_SVM_ATTR_PREFETCH_LOC, gpu_id};
        svm_args->attrs[1] = {KFD_IOCTL_SVM_ATTR_PREFERRED_LOC, 0};
        svm_args->attrs[2] = {KFD_IOCTL_SVM_ATTR_CLR_FLAGS, ~svm_flags};
        svm_args->attrs[3] = {KFD_IOCTL_SVM_ATTR_SET_FLAGS, svm_flags};
        svm_args->attrs[4] = {KFD_IOCTL_SVM_ATTR_ACCESS, gpu_id};
        svm_args->attrs[5] = {KFD_IOCTL_SVM_ATTR_GRANULARITY, 0xFF};

        unsigned long svm_cmd = AMDKFD_IOC_SVM +
            (static_cast<unsigned long>(attr_bytes) << _IOC_SIZESHIFT);
        if (GpuIoctl(fd_, svm_cmd, svm_args) == 0) {
          qinfo->ctx_save_restore = ctx_addr;
          qinfo->ctx_size = alloc_size;
          qinfo->ctx_is_mmap = true;
        } else {
          debug_print("SVM register for CWSR failed: errno=%d\n", errno);
          munmap(ctx_addr, alloc_size);
          ctx_addr = MAP_FAILED;
        }
      }

      // Fallback: allocate CWSR via ALLOC_MEMORY_OF_GPU if SVM failed.
      // Matches libhsakmt: allocate_exec_aligned_memory(nonPaged=false,
      // DeviceLocal=false, Uncached=false) → USERPTR path.
      if (ctx_addr == MAP_FAILED || !qinfo->ctx_save_restore) {
        HsaMemFlags ctx_flags = {};
        ctx_flags.ui32.HostAccess = 1;
        ctx_flags.ui32.ExecuteAccess = 1;
        qinfo->ctx_save_restore = AllocateDriverMemory(
            fd_, drm_fd, ctx_flags, gpu_id, total_alloc, &qinfo->ctx_handle);
        if (qinfo->ctx_save_restore) {
          qinfo->ctx_size = total_alloc;
          uint64_t alt_va = 0;
          HsaMemMapFlags map_flags = {};
          MakeDriverMemoryResident(fd_, 1, &gpu_id, qinfo->ctx_save_restore,
                                    qinfo->ctx_handle, total_alloc,
                                    &alt_va, map_flags);
          // Fill CWSR header even in fallback path.
          for (uint32_t x = 0; x < num_xcc; x++) {
            auto* header = reinterpret_cast<HsaUserContextSaveAreaHeader*>(
                static_cast<char*>(qinfo->ctx_save_restore) + x * cwsr_size);
            header->ErrorEventId = event ? event->EventId : 0;
            header->ErrorReason = queue_resource.ErrorReason;
            header->DebugOffset = (num_xcc - x) * cwsr_size;
            header->DebugSize = debug_memory_size * num_xcc;
          }
        }
      }

      if (qinfo->ctx_save_restore) {
        args.ctx_save_restore_address =
            reinterpret_cast<uint64_t>(qinfo->ctx_save_restore);
        args.ctx_save_restore_size = cwsr_size;
        args.ctl_stack_size = ctl_stack_size;
      }
    }
  }

  if (GpuIoctl(fd_, AMDKFD_IOC_CREATE_QUEUE, &args) != 0) {
    // Clean up allocated buffers (unmap from GPU before freeing).
    if (qinfo->eop_buffer) {
      MakeDriverMemoryUnresident(fd_, qinfo->eop_buffer, qinfo->eop_handle,
                                  1, &gpu_id);
      FreeDriverMemory(fd_, qinfo->eop_buffer, qinfo->eop_handle, qinfo->eop_size);
    }
    if (qinfo->ctx_save_restore) {
      if (qinfo->ctx_is_mmap) {
        munmap(qinfo->ctx_save_restore, qinfo->ctx_size);
      } else {
        MakeDriverMemoryUnresident(fd_, qinfo->ctx_save_restore, qinfo->ctx_handle,
                                    1, &gpu_id);
        FreeDriverMemory(fd_, qinfo->ctx_save_restore, qinfo->ctx_handle, qinfo->ctx_size);
      }
    }
    if (qinfo->queue_struct) {
      MakeDriverMemoryUnresident(fd_, qinfo->queue_struct, qinfo->queue_struct_handle,
                                  1, &gpu_id);
      FreeDriverMemory(fd_, qinfo->queue_struct, qinfo->queue_struct_handle,
                        qinfo->queue_struct_size);
    }
    delete qinfo;
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  qinfo->queue_id = args.queue_id;

  // Map doorbell page. SOC15+ uses 8-byte doorbells (8KB page = 1024 * 8).
  // Older ASICs use 4-byte doorbells (4KB page = 1024 * 4).
  // Matches libhsakmt's DOORBELL_SIZE and DOORBELLS_PAGE_SIZE macros.
  uint32_t gfxv_raw = 0;
  {
    HsaNodeProperties np = {};
    GetNodeProperties(np, node_id);
    gfxv_raw = np.EngineId.ui32.Major * 10000 +
               np.EngineId.ui32.Minor * 100 +
               np.EngineId.ui32.Stepping;
  }
  bool is_soc15 = (gfxv_raw >= 90000);
  uint32_t doorbell_size = is_soc15 ? 8 : 4;
  uint32_t doorbells_page_size = 1024 * doorbell_size;
  if (doorbells_page_size < static_cast<uint32_t>(kPageSize))
    doorbells_page_size = static_cast<uint32_t>(kPageSize);

  // Doorbell offset calculation differs between SOC15+ and pre-SOC15.
  // SOC15+: doorbell_offset from ioctl is absolute, mask to get page/within.
  // Pre-SOC15: mmap_offset is the full offset, within-page is queue_id based.
  uint64_t doorbell_mmap_offset;
  uint64_t doorbell_within_page;
  if (is_soc15) {
    uint64_t db_mask = static_cast<uint64_t>(doorbells_page_size - 1);
    doorbell_mmap_offset = args.doorbell_offset & ~db_mask;
    doorbell_within_page = args.doorbell_offset & db_mask;
  } else {
    doorbell_mmap_offset = args.doorbell_offset;
    doorbell_within_page = static_cast<uint64_t>(args.queue_id) * doorbell_size;
  }

  // Lazily map doorbell page per-node. Matches libhsakmt's map_doorbell:
  // On dGPU (except Tonga), use GPUVM path -- allocate a DOORBELL BO, mmap
  // with MAP_FIXED, and map to GPU. Fall back to APU-style mmap on failure.
  // On APU or Tonga, use direct mmap from KFD fd.
  static constexpr uint32_t kGfxvTonga = 80002;  // GFX_VERSION_TONGA
  if (node_id < doorbells_.size() && !doorbells_[node_id].mapping) {
    DoorbellInfo& db = doorbells_[node_id];
    bool want_gpuvm = (is_dgpu_ && gfxv_raw != kGfxvTonga);
    bool mapped = false;

    if (want_gpuvm) {
      // dGPU GPUVM path: allocate doorbell BO, mmap at that VA, map to GPU.
      // Matches libhsakmt's map_doorbell_dgpu + hsakmt_fmm_allocate_doorbell.
      void* va = mmap(nullptr, doorbells_page_size, PROT_NONE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
      if (va != MAP_FAILED) {
        kfd_ioctl_alloc_memory_of_gpu_args alloc_args = {};
        alloc_args.gpu_id = gpu_id;
        alloc_args.size = doorbells_page_size;
        alloc_args.va_addr = reinterpret_cast<uint64_t>(va);
        alloc_args.flags = KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL |
                           KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
                           KFD_IOC_ALLOC_MEM_FLAGS_COHERENT;

        if (GpuIoctl(fd_, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &alloc_args) == 0) {
          // CPU-map the doorbell page at the allocated GPUVM VA.
          void* cpu_map = mmap(va, doorbells_page_size,
                                PROT_READ | PROT_WRITE,
                                MAP_SHARED | MAP_FIXED, fd_,
                                doorbell_mmap_offset);
          if (cpu_map != MAP_FAILED) {
            // Map to GPU so doorbell is visible in GPU VA space.
            uint32_t db_gpu_id = gpu_id;
            uint64_t alt_va = 0;
            HsaMemMapFlags map_flag = {{0}};
            if (MakeDriverMemoryResident(fd_, 1, &db_gpu_id, cpu_map,
                                          alloc_args.handle, doorbells_page_size,
                                          &alt_va, map_flag)) {
              // Prevent fork child from inheriting doorbell mapping
              // (avoids COW MMU notifier evicting parent queues).
              madvise(cpu_map, doorbells_page_size, MADV_DONTFORK);
              db.mapping = cpu_map;
              db.size = doorbells_page_size;
              db.use_gpuvm = true;
              db.handle = alloc_args.handle;
              db.gpu_id = gpu_id;
              mapped = true;
            } else {
              // Map-to-GPU failed, clean up.
              kfd_ioctl_free_memory_of_gpu_args fa = {};
              fa.handle = alloc_args.handle;
              GpuIoctl(fd_, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &fa);
              munmap(va, doorbells_page_size);
            }
          } else {
            // CPU mmap failed, free the BO and VA reservation.
            kfd_ioctl_free_memory_of_gpu_args fa = {};
            fa.handle = alloc_args.handle;
            GpuIoctl(fd_, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &fa);
            munmap(va, doorbells_page_size);
          }
        } else {
          // ALLOC failed, release VA reservation.
          munmap(va, doorbells_page_size);
        }
      }
    }

    if (!mapped) {
      // APU path or GPUVM fallback: direct mmap from KFD fd.
      // Matches libhsakmt's map_doorbell_apu.
      void* ptr = mmap(nullptr, doorbells_page_size,
                        PROT_READ | PROT_WRITE,
                        MAP_SHARED, fd_, doorbell_mmap_offset);
      if (ptr != MAP_FAILED) {
        madvise(ptr, doorbells_page_size, MADV_DONTFORK);
        db.mapping = ptr;
        db.size = doorbells_page_size;
        db.use_gpuvm = false;
        db.handle = 0;
        db.gpu_id = 0;
      } else {
        debug_print("doorbell mmap failed: offset=0x%lx size=%u errno=%d\n",
                    (unsigned long)doorbell_mmap_offset, doorbells_page_size,
                    errno);
      }
    }
  }

  volatile uint64_t* doorbell_ptr = nullptr;
  if (node_id < doorbells_.size() && doorbells_[node_id].mapping) {
    doorbell_ptr = reinterpret_cast<volatile uint64_t*>(
        static_cast<char*>(doorbells_[node_id].mapping) + doorbell_within_page);
  }

  qinfo->doorbell_ptr = doorbell_ptr;
  qinfo->doorbell_mmap_base = nullptr;

  queue_resource.QueueId = reinterpret_cast<uint64_t>(qinfo);

  if (doorbell_ptr) {
    queue_resource.Queue_DoorBell_aql =
        const_cast<uint64_t*>(doorbell_ptr);
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::UpdateQueue(HSA_QUEUEID queue_id, uint32_t queue_pct,
                                    HSA::hsa_amd_queue_priority_internal_t priority,
                                    void* queue_addr, uint64_t queue_size,
                                    HsaEvent* event) const {
  auto* qinfo = reinterpret_cast<QueueInfo*>(queue_id);
  HSA_QUEUE_PRIORITY drv_priority = HsaInternalToDriverPriority(priority);

  // Map HSA_QUEUE_PRIORITY enum (-3..3) to KFD priority values.
  // Matches libhsakmt's priority_map[] = {0, 3, 5, 7, 9, 11, 15}.
  static constexpr uint32_t kPriorityMap[] = {0, 3, 5, 7, 9, 11, 15};
  int prio_idx = static_cast<int>(drv_priority) + 3;
  if (prio_idx < 0) prio_idx = 0;
  if (prio_idx > 6) prio_idx = 6;

  kfd_ioctl_update_queue_args args = {};
  args.queue_id = qinfo->queue_id;
  args.ring_base_address = reinterpret_cast<uint64_t>(queue_addr);
  args.ring_size = static_cast<uint32_t>(queue_size);
  args.queue_percentage = queue_pct;
  args.queue_priority = kPriorityMap[prio_idx];

  if (GpuIoctl(fd_, AMDKFD_IOC_UPDATE_QUEUE, &args) != 0) {
    return HSA_STATUS_ERROR;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::DestroyQueue(HSA_QUEUEID queue_id) const {
  auto* qinfo = reinterpret_cast<QueueInfo*>(queue_id);

  kfd_ioctl_destroy_queue_args args = {};
  args.queue_id = qinfo->queue_id;

  int ret = GpuIoctl(fd_, AMDKFD_IOC_DESTROY_QUEUE, &args);

  // Unmap and free EOP and context save/restore buffers.
  if (qinfo->eop_buffer) {
    MakeDriverMemoryUnresident(fd_, qinfo->eop_buffer, qinfo->eop_handle,
                                1, &qinfo->gpu_id);
    FreeDriverMemory(fd_, qinfo->eop_buffer, qinfo->eop_handle, qinfo->eop_size);
  }
  if (qinfo->ctx_save_restore) {
    if (qinfo->ctx_is_mmap) {
      munmap(qinfo->ctx_save_restore, qinfo->ctx_size);
    } else {
      MakeDriverMemoryUnresident(fd_, qinfo->ctx_save_restore, qinfo->ctx_handle,
                                  1, &qinfo->gpu_id);
      FreeDriverMemory(fd_, qinfo->ctx_save_restore, qinfo->ctx_handle, qinfo->ctx_size);
    }
  }
  if (qinfo->queue_struct) {
    MakeDriverMemoryUnresident(fd_, qinfo->queue_struct, qinfo->queue_struct_handle,
                                1, &qinfo->gpu_id);
    FreeDriverMemory(fd_, qinfo->queue_struct, qinfo->queue_struct_handle,
                      qinfo->queue_struct_size);
  }

  delete qinfo;

  return (ret == 0) ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
}

hsa_status_t GpuDriver::SetQueueCUMask(HSA_QUEUEID queue_id,
                                        uint32_t cu_mask_count,
                                        uint32_t* queue_cu_mask) const {
  auto* qinfo = reinterpret_cast<QueueInfo*>(queue_id);

  kfd_ioctl_set_cu_mask_args args = {};
  args.queue_id = qinfo->queue_id;
  args.num_cu_mask = cu_mask_count;
  args.cu_mask_ptr = reinterpret_cast<uint64_t>(queue_cu_mask);

  if (GpuIoctl(fd_, AMDKFD_IOC_SET_CU_MASK, &args) != 0)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::AllocQueueGWS(HSA_QUEUEID queue_id, uint32_t num_gws,
                                       uint32_t* first_gws) const {
  auto* qinfo = reinterpret_cast<QueueInfo*>(queue_id);

  kfd_ioctl_alloc_queue_gws_args args = {};
  args.queue_id = qinfo->queue_id;
  args.num_gws = num_gws;

  if (GpuIoctl(fd_, AMDKFD_IOC_ALLOC_QUEUE_GWS, &args) != 0)
    return HSA_STATUS_ERROR;

  *first_gws = args.first_gws;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::GetQueueSaveAreaInfo(HSA_QUEUEID queue_id,
                                              void** address,
                                              size_t* size) const {
  assert(address);
  assert(size);

  auto* qinfo = reinterpret_cast<QueueInfo*>(queue_id);

  kfd_ioctl_get_queue_wave_state_args args = {};
  args.queue_id = qinfo->queue_id;

  if (GpuIoctl(fd_, AMDKFD_IOC_GET_QUEUE_WAVE_STATE, &args) != 0)
    return HSA_STATUS_ERROR;

  *address = reinterpret_cast<void*>(args.ctl_stack_address);
  *size = args.save_area_used_size;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::RingDoorbell(HSA_QUEUEID queue_id, uint64_t value) const {
  auto* qinfo = reinterpret_cast<QueueInfo*>(queue_id);
  if (qinfo && qinfo->doorbell_ptr) {
    *qinfo->doorbell_ptr = value;
  }
  return HSA_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
// Event management
///////////////////////////////////////////////////////////////////////////////

hsa_status_t GpuDriver::CreateEvent(HsaEventDescriptor& event_descriptor,
                                    bool manual_reset,
                                    HsaEvent** event) const {
  // Allocate GPU-accessible events page once per process.
  // The events page must be in GPUVM so the GPU hardware (AQL completion,
  // trap handler) can write event_id to event_mailbox_ptr via s_store_dword.
  // Matches libhsakmt's dGPU path: ALLOC_MEMORY_OF_GPU + MAP_MEMORY_TO_GPU
  // then pass the handle as event_page_offset to CREATE_EVENT.
  // The events page persists in process_state_ across init/shutdown cycles.
  bool just_allocated_events_page = false;
  if (!process_state_->events_page) {
    const size_t ep_size = KFD_SIGNAL_EVENT_LIMIT * 8;
    uint32_t ep_gpu_id = all_gpu_id_array_.empty() ? 0 : all_gpu_id_array_[0];

    // GTT, NonPaged, HostAccess, Executable, Coherent, Uncached.
    uint32_t ioc_flags = KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
                         KFD_IOC_ALLOC_MEM_FLAGS_GTT |
                         KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE |
                         KFD_IOC_ALLOC_MEM_FLAGS_COHERENT |
                         KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED |
                         KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE;

    void* ep_va = mmap(nullptr, ep_size, PROT_NONE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (ep_va != MAP_FAILED) {
      kfd_ioctl_alloc_memory_of_gpu_args alloc_args = {};
      alloc_args.size = ep_size;
      alloc_args.gpu_id = ep_gpu_id;
      alloc_args.flags = ioc_flags;
      alloc_args.va_addr = reinterpret_cast<uint64_t>(ep_va);

      if (GpuIoctl(fd_, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &alloc_args) == 0) {
        int drm_fd = DrmFdForGpuId(ep_gpu_id);
        void* ep_mapped = mmap(ep_va, ep_size, PROT_READ | PROT_WRITE,
                               MAP_SHARED | MAP_FIXED, drm_fd,
                               alloc_args.mmap_offset);
        if (ep_mapped != MAP_FAILED) {
          madvise(ep_mapped, ep_size, MADV_DONTFORK);
          process_state_->events_page = ep_mapped;
          process_state_->events_page_handle = alloc_args.handle;
          process_state_->events_page_size = ep_size;
          just_allocated_events_page = true;

          // Map to all GPUs so GPU hardware can write to the events page.
          if (!all_gpu_id_array_.empty()) {
            uint64_t alt_va = 0;
            HsaMemMapFlags map_flags = {};
            MakeDriverMemoryResident(fd_, all_gpu_id_array_.size(),
                                     all_gpu_id_array_.data(),
                                     ep_mapped, alloc_args.handle,
                                     ep_size, &alt_va, map_flags);
          }
        } else {
          kfd_ioctl_free_memory_of_gpu_args free_args = {};
          free_args.handle = alloc_args.handle;
          GpuIoctl(fd_, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &free_args);
          munmap(ep_va, ep_size);
        }
      } else {
        munmap(ep_va, ep_size);
      }
    }
  }

  kfd_ioctl_create_event_args args = {};
  args.event_type = event_descriptor.EventType;
  args.auto_reset = manual_reset ? 0 : 1;
  args.node_id = event_descriptor.NodeId;

  // Pass the pre-allocated events page handle only on the first CREATE_EVENT
  // call (when we just allocated the page). The kernel registers it once;
  // subsequent calls must have event_page_offset=0.
  if (just_allocated_events_page)
    args.event_page_offset = process_state_->events_page_handle;

  int create_ret = GpuIoctl(fd_, AMDKFD_IOC_CREATE_EVENT, &args);
  if (create_ret != 0)
    return HSA_STATUS_ERROR;

  // Allocate HsaEvent struct.
  auto* ev = static_cast<HsaEvent*>(calloc(1, sizeof(HsaEvent)));
  if (!ev) {
    kfd_ioctl_destroy_event_args d_args = {};
    d_args.event_id = args.event_id;
    GpuIoctl(fd_, AMDKFD_IOC_DESTROY_EVENT, &d_args);
    return HSA_STATUS_ERROR;
  }

  ev->EventData.EventType = static_cast<HSA_EVENTTYPE>(event_descriptor.EventType);
  ev->EventId = args.event_id;
  // HWData1 = event_id (used internally by the runtime)
  ev->EventData.HWData1 = args.event_id;
  // HWData3 = event_trigger_data (the GPU writes this to the mailbox)
  ev->EventData.HWData3 = args.event_trigger_data;
  // Copy SyncVar from descriptor (matches libhsakmt hsaKmtCreateEvent).
  ev->EventData.EventData.SyncVar.SyncVar.UserData =
      event_descriptor.SyncVar.SyncVar.UserData;
  ev->EventData.EventData.SyncVar.SyncVarSize =
      event_descriptor.SyncVar.SyncVarSize;

  // HWData2 = events page slot address (the GPU's event mailbox pointer).
  // The runtime stores this in signal_.event_mailbox_ptr and the GPU writes
  // to this address to signal event completion.
  if (process_state_->events_page &&
      event_descriptor.EventType == HSA_EVENTTYPE_SIGNAL) {
    ev->EventData.HWData2 = reinterpret_cast<uint64_t>(
        static_cast<char*>(process_state_->events_page) +
        args.event_slot_index * 8);
  }

  *event = ev;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::DestroyEvent(HsaEvent* event) const {
  if (!event) return HSA_STATUS_SUCCESS;

  kfd_ioctl_destroy_event_args args = {};
  args.event_id = event->EventId;
  GpuIoctl(fd_, AMDKFD_IOC_DESTROY_EVENT, &args);

  free(event);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::WaitOnEvent(HsaEvent* event, uint32_t timeout_ms,
                                    uint64_t* event_age) const {
  if (!event) return HSA_STATUS_ERROR;

  kfd_event_data ev_data = {};
  ev_data.event_id = event->EventId;
  if (event_age) {
    ev_data.signal_event_data.last_event_age = *event_age;
  }

  kfd_ioctl_wait_events_args args = {};
  args.events_ptr = reinterpret_cast<uint64_t>(&ev_data);
  args.num_events = 1;
  args.wait_for_all = 1;
  args.timeout = timeout_ms;

  GpuIoctl(fd_, AMDKFD_IOC_WAIT_EVENTS, &args);

  if (event_age) {
    *event_age = ev_data.signal_event_data.last_event_age;
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::WaitOnMultipleEvents(HsaEvent** events,
                                              uint32_t num_events,
                                              bool wait_on_all,
                                              uint32_t timeout_ms,
                                              uint64_t* event_age) const {
  if (!events || num_events == 0) return HSA_STATUS_ERROR;

  std::vector<kfd_event_data> ev_data(num_events);
  for (uint32_t i = 0; i < num_events; ++i) {
    memset(&ev_data[i], 0, sizeof(kfd_event_data));
    ev_data[i].event_id = events[i]->EventId;
    if (event_age && events[i]->EventData.EventType == HSA_EVENTTYPE_SIGNAL) {
      ev_data[i].signal_event_data.last_event_age = event_age[i];
    }
  }

  kfd_ioctl_wait_events_args args = {};
  args.events_ptr = reinterpret_cast<uint64_t>(ev_data.data());
  args.num_events = num_events;
  args.wait_for_all = wait_on_all ? 1 : 0;
  args.timeout = timeout_ms;

  GpuIoctl(fd_, AMDKFD_IOC_WAIT_EVENTS, &args);

  // Process event data returned by the kernel.
  // Matches libhsakmt hsaKmtWaitOnMultipleEvents_ExtCtx (events.c:467-501).
  for (uint32_t i = 0; i < num_events; ++i) {
    if (events[i]->EventData.EventType == HSA_EVENTTYPE_MEMORY &&
        ev_data[i].memory_exception_data.gpu_id) {
      auto& fault = events[i]->EventData.EventData.MemoryAccessFault;
      fault.VirtualAddress = ev_data[i].memory_exception_data.va;
      // Convert gpu_id to node_id.
      fault.NodeId = 0;
      for (size_t n = 0; n < gpu_ids_.size(); ++n) {
        if (gpu_ids_[n] == ev_data[i].memory_exception_data.gpu_id) {
          fault.NodeId = static_cast<uint32_t>(n);
          break;
        }
      }
      fault.Failure.NotPresent = ev_data[i].memory_exception_data.failure.NotPresent;
      fault.Failure.ReadOnly = ev_data[i].memory_exception_data.failure.ReadOnly;
      fault.Failure.NoExecute = ev_data[i].memory_exception_data.failure.NoExecute;
      fault.Failure.Imprecise = ev_data[i].memory_exception_data.failure.imprecise;
      fault.Failure.ErrorType = ev_data[i].memory_exception_data.ErrorType;
      fault.Failure.ECC =
          (ev_data[i].memory_exception_data.ErrorType == 1 ||
           ev_data[i].memory_exception_data.ErrorType == 2) ? 1 : 0;
      fault.Flags = HSA_EVENTID_MEMORY_FATAL_PROCESS;
    } else if (events[i]->EventData.EventType == HSA_EVENTTYPE_HW_EXCEPTION) {
      if (ev_data[i].hw_exception_data.gpu_id) {
      auto& hw = events[i]->EventData.EventData.HwException;
      hw.NodeId = 0;
      for (size_t n = 0; n < gpu_ids_.size(); ++n) {
        if (gpu_ids_[n] == ev_data[i].hw_exception_data.gpu_id) {
          hw.NodeId = static_cast<uint32_t>(n);
          break;
        }
      }
      hw.ResetType = ev_data[i].hw_exception_data.reset_type;
      hw.ResetCause =
          static_cast<HSA_EVENTID_HW_EXCEPTION_CAUSE>(ev_data[i].hw_exception_data.reset_cause);
      hw.MemoryLost = ev_data[i].hw_exception_data.memory_lost;
      }
    }
  }

  // Update event ages for signal events.
  if (event_age) {
    for (uint32_t i = 0; i < num_events; ++i) {
      if (events[i]->EventData.EventType == HSA_EVENTTYPE_SIGNAL) {
        event_age[i] = ev_data[i].signal_event_data.last_event_age;
      }
    }
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::SetEvent(HsaEvent* event) const {
  if (!event) return HSA_STATUS_ERROR;

  kfd_ioctl_set_event_args args = {};
  args.event_id = event->EventId;
  GpuIoctl(fd_, AMDKFD_IOC_SET_EVENT, &args);
  return HSA_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
// Memory mapping and registration
///////////////////////////////////////////////////////////////////////////////

hsa_status_t GpuDriver::RegisterMemory(void* ptr, uint64_t size,
                                        HsaMemFlags mem_flags) const {
  assert(ptr);
  assert(size > 0);

  // Page-align the address and size (matches libhsakmt fmm_register_user_memory).
  uint64_t page_offset = reinterpret_cast<uint64_t>(ptr) & (kPageSize - 1);
  uint64_t aligned_addr = reinterpret_cast<uint64_t>(ptr) - page_offset;
  uint64_t aligned_size = (page_offset + size + kPageSize - 1) & ~(kPageSize - 1);

  // Use the first GPU's gpu_id (matches libhsakmt which uses
  // fmm_ctx->first_gpu_mem->gpu_id for USERPTR registration).
  uint32_t gpu_id = all_gpu_id_array_.empty() ? 0 : all_gpu_id_array_[0];

  // Register user memory via ALLOC with USERPTR flag.
  // Flags match libhsakmt fmm_register_user_memory: USERPTR | WRITABLE |
  // EXECUTABLE, plus coherency flags.
  kfd_ioctl_alloc_memory_of_gpu_args args = {};
  args.va_addr = aligned_addr;
  args.size = aligned_size;
  args.mmap_offset = aligned_addr;
  args.gpu_id = gpu_id;
  args.flags = KFD_IOC_ALLOC_MEM_FLAGS_USERPTR |
               KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
               KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE |
               KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE;

  if (!mem_flags.ui32.CoarseGrain)
    args.flags |= KFD_IOC_ALLOC_MEM_FLAGS_COHERENT;
  if (mem_flags.ui32.ExtendedCoherent)
    args.flags |= KFD_IOC_ALLOC_MEM_FLAGS_EXT_COHERENT;

  if (GpuIoctl(fd_, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &args) != 0)
    return HSA_STATUS_ERROR;

  mem_handles_[ptr] = {args.handle, static_cast<size_t>(size), 0, mem_flags.Value, {}};
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::DeregisterMemory(void* ptr) const {
  auto it = mem_handles_.find(ptr);
  if (it == mem_handles_.end()) {
    return HSA_STATUS_ERROR;
  }

  kfd_ioctl_free_memory_of_gpu_args args = {};
  args.handle = it->second.handle;
  GpuIoctl(fd_, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &args);

  mem_handles_.erase(it);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::MakeMemoryResident(const void* mem, size_t size,
                                            uint64_t* alternate_va,
                                            const HsaMemMapFlags* mem_flags,
                                            uint32_t num_nodes,
                                            const uint32_t* nodes) const {
  auto it = mem_handles_.find(const_cast<void*>(mem));
  if (it == mem_handles_.end()) {
    debug_print("MakeMemoryResident: untracked ptr:%p\n", mem);
    return HSA_STATUS_ERROR;
  }

  // Anonymous mmap (e.g., scratch backing) has no driver handle and doesn't
  // need GPU mapping — the GPU accesses it through a dedicated register.
  if (it->second.handle == 0) {
    if (alternate_va) *alternate_va = reinterpret_cast<uint64_t>(mem);
    return HSA_STATUS_SUCCESS;
  }

  const uint32_t* map_gpu_ids = nullptr;
  size_t map_count = 0;
  std::vector<uint32_t> gpu_ids_buf;
  HsaMemMapFlags effective_flags = {};

  if (mem_flags == nullptr && nodes == nullptr) {
    // Map to all GPUs.
    if (all_gpu_id_array_.empty()) {
      if (alternate_va) *alternate_va = 0;
      return HSA_STATUS_SUCCESS;
    }
    map_gpu_ids = all_gpu_id_array_.data();
    map_count = all_gpu_id_array_.size();
  } else if (mem_flags != nullptr && nodes != nullptr) {
    // Callers pass node_ids; convert to gpu_ids for the ioctl.
    gpu_ids_buf.resize(num_nodes);
    for (uint32_t i = 0; i < num_nodes; ++i)
      gpu_ids_buf[i] = NodeToGpuId(nodes[i]);
    map_gpu_ids = gpu_ids_buf.data();
    map_count = num_nodes;
    effective_flags = *mem_flags;
  } else {
    debug_print("Invalid memory flags ptr:%p nodes ptr:%p\n", mem_flags, nodes);
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  bool ok = MakeDriverMemoryResident(fd_, map_count, map_gpu_ids,
                                      const_cast<void*>(mem), it->second.handle,
                                      size, alternate_va, effective_flags);
  if (ok)
    it->second.mapped_gpu_ids.assign(map_gpu_ids, map_gpu_ids + map_count);
  return ok ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
}

hsa_status_t GpuDriver::MakeMemoryUnresident(const void* mem) const {
  auto it = mem_handles_.find(const_cast<void*>(mem));
  if (it != mem_handles_.end() && it->second.handle != 0 &&
      !it->second.mapped_gpu_ids.empty()) {
    MakeDriverMemoryUnresident(fd_, const_cast<void*>(mem), it->second.handle,
                               it->second.mapped_gpu_ids.size(),
                               it->second.mapped_gpu_ids.data());
    it->second.mapped_gpu_ids.clear();
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::MapMemoryToGPU(const void* mem, size_t size,
                                        uint64_t* alternate_va) const {
  auto it = mem_handles_.find(const_cast<void*>(mem));
  if (it == mem_handles_.end()) return HSA_STATUS_ERROR;

  if (it->second.handle == 0) {
    if (alternate_va) *alternate_va = reinterpret_cast<uint64_t>(mem);
    return HSA_STATUS_SUCCESS;
  }

  if (all_gpu_id_array_.empty()) {
    if (alternate_va) *alternate_va = 0;
    return HSA_STATUS_SUCCESS;
  }

  HsaMemMapFlags flags = {};
  bool ok = MakeDriverMemoryResident(fd_, all_gpu_id_array_.size(),
                                     all_gpu_id_array_.data(),
                                     const_cast<void*>(mem), it->second.handle,
                                     size, alternate_va, flags);
  if (ok)
    it->second.mapped_gpu_ids = all_gpu_id_array_;
  return ok ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
}

hsa_status_t GpuDriver::MapMemoryToGPUNodes(const void* mem, size_t size,
                                             uint64_t* alternate_va,
                                             HsaMemMapFlags flags,
                                             uint32_t num_nodes,
                                             const uint32_t* nodes) const {
  auto it = mem_handles_.find(const_cast<void*>(mem));
  if (it == mem_handles_.end()) return HSA_STATUS_ERROR;

  if (it->second.handle == 0) {
    if (alternate_va) *alternate_va = reinterpret_cast<uint64_t>(mem);
    return HSA_STATUS_SUCCESS;
  }

  // Callers pass node_ids; convert to gpu_ids for the ioctl.
  std::vector<uint32_t> new_gpu_ids(num_nodes);
  for (uint32_t i = 0; i < num_nodes; ++i)
    new_gpu_ids[i] = NodeToGpuId(nodes[i]);

  auto& cur = it->second.mapped_gpu_ids;

  // Differential unmap/remap matching libhsakmt _fmm_map_to_gpu_gtt_bo:
  // 1) Unmap nodes currently mapped but NOT in the new set.
  if (!cur.empty()) {
    std::vector<uint32_t> to_unmap;
    for (uint32_t id : cur) {
      if (std::find(new_gpu_ids.begin(), new_gpu_ids.end(), id) ==
          new_gpu_ids.end())
        to_unmap.push_back(id);
    }
    if (!to_unmap.empty()) {
      MakeDriverMemoryUnresident(fd_, const_cast<void*>(mem),
                                 it->second.handle,
                                 to_unmap.size(), to_unmap.data());
    }
  }

  // 2) Build list of nodes to actually map (not already mapped).
  std::vector<uint32_t> to_map;
  for (uint32_t id : new_gpu_ids) {
    if (std::find(cur.begin(), cur.end(), id) == cur.end())
      to_map.push_back(id);
  }

  if (!to_map.empty()) {
    bool ok = MakeDriverMemoryResident(fd_, to_map.size(), to_map.data(),
                                       const_cast<void*>(mem),
                                       it->second.handle,
                                       size, alternate_va, flags);
    if (!ok) return HSA_STATUS_ERROR;
  } else {
    if (alternate_va) *alternate_va = 0;
  }

  // 3) Update tracked mapping to the new set.
  cur = std::move(new_gpu_ids);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::AvailableMemory(uint32_t node_id,
                                         uint64_t* available_size) const {
  assert(available_size);

  kfd_ioctl_get_available_memory_args args = {};
  args.gpu_id = NodeToGpuId(node_id);

  if (GpuIoctl(fd_, AMDKFD_IOC_AVAILABLE_MEMORY, &args) != 0)
    return HSA_STATUS_ERROR;

  *available_size = args.available;
  return HSA_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
// Simple ioctl wrappers
///////////////////////////////////////////////////////////////////////////////

hsa_status_t GpuDriver::SetTrapHandler(uint32_t node_id, const void* base,
                                        uint64_t base_size,
                                        const void* buffer_base,
                                        uint64_t buffer_base_size) const {
  kfd_ioctl_set_trap_handler_args args = {};
  args.tba_addr = reinterpret_cast<uint64_t>(base);
  args.tma_addr = reinterpret_cast<uint64_t>(buffer_base);
  args.gpu_id = NodeToGpuId(node_id);

  if (GpuIoctl(fd_, AMDKFD_IOC_SET_TRAP_HANDLER, &args) != 0)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::GetClockCounters(uint32_t node_id,
                                          HsaClockCounters* clock_counter) const {
  assert(clock_counter);

  kfd_ioctl_get_clock_counters_args args = {};
  args.gpu_id = NodeToGpuId(node_id);

  if (GpuIoctl(fd_, AMDKFD_IOC_GET_CLOCK_COUNTERS, &args) != 0)
    return HSA_STATUS_ERROR;

  clock_counter->GPUClockCounter = args.gpu_clock_counter;
  clock_counter->CPUClockCounter = args.cpu_clock_counter;
  clock_counter->SystemClockCounter = args.system_clock_counter;
  clock_counter->SystemClockFrequencyHz = args.system_clock_freq;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::GetTileConfig(uint32_t node_id,
                                       HsaGpuTileConfig* config) const {
  assert(config);

  kfd_ioctl_get_tile_config_args args = {};
  args.gpu_id = NodeToGpuId(node_id);
  args.tile_config_ptr = reinterpret_cast<uint64_t>(config->TileConfig);
  args.macro_tile_config_ptr = reinterpret_cast<uint64_t>(config->MacroTileConfig);
  args.num_tile_configs = config->NumTileConfigs;
  args.num_macro_tile_configs = config->NumMacroTileConfigs;

  if (GpuIoctl(fd_, AMDKFD_IOC_GET_TILE_CONFIG, &args) != 0)
    return HSA_STATUS_ERROR;

  config->NumTileConfigs = args.num_tile_configs;
  config->NumMacroTileConfigs = args.num_macro_tile_configs;
  config->GbAddrConfig = args.gb_addr_config;
  config->NumBanks = args.num_banks;
  config->NumRanks = args.num_ranks;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::GetWallclockFrequency(uint32_t node_id,
                                               uint64_t* frequency) const {
  assert(frequency);

  kfd_ioctl_get_clock_counters_args args = {};
  args.gpu_id = NodeToGpuId(node_id);

  if (GpuIoctl(fd_, AMDKFD_IOC_GET_CLOCK_COUNTERS, &args) != 0)
    return HSA_STATUS_ERROR;

  *frequency = args.system_clock_freq;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::OpenSMI(uint32_t node_id, int* fd) const {
  kfd_ioctl_smi_events_args args = {};
  args.gpuid = NodeToGpuId(node_id);

  if (GpuIoctl(fd_, AMDKFD_IOC_SMI_EVENTS, &args) != 0)
    return HSA_STATUS_ERROR;

  *fd = static_cast<int>(args.anon_fd);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::IsModelEnabled(bool* enable) const {
  assert(enable);
  const char* env = getenv("HSA_ENABLE_MODEL");
  *enable = (env && std::string(env) == "1");
  return HSA_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
// SPM (Streaming Performance Monitor)
///////////////////////////////////////////////////////////////////////////////

hsa_status_t GpuDriver::SPMAcquire(uint32_t preferred_node_id) const {
  kfd_ioctl_spm_args args = {};
  args.op = KFD_IOCTL_SPM_OP_ACQUIRE;
  args.gpu_id = NodeToGpuId(preferred_node_id);

  if (GpuIoctl(fd_, AMDKFD_IOC_RLC_SPM, &args) != 0)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::SPMRelease(uint32_t preferred_node_id) const {
  kfd_ioctl_spm_args args = {};
  args.op = KFD_IOCTL_SPM_OP_RELEASE;
  args.gpu_id = NodeToGpuId(preferred_node_id);

  if (GpuIoctl(fd_, AMDKFD_IOC_RLC_SPM, &args) != 0)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::SPMSetDestBuffer(uint32_t preferred_node_id,
                                          uint32_t size_bytes,
                                          uint32_t* timeout,
                                          uint32_t* size_copied,
                                          void* dest_mem_addr,
                                          bool* is_spm_data_loss) const {
  kfd_ioctl_spm_args args = {};
  args.op = KFD_IOCTL_SPM_OP_SET_DEST_BUF;
  args.gpu_id = NodeToGpuId(preferred_node_id);
  args.dest_buf = reinterpret_cast<uint64_t>(dest_mem_addr);
  args.buf_size = size_bytes;
  args.timeout = timeout ? *timeout : 0;

  if (GpuIoctl(fd_, AMDKFD_IOC_RLC_SPM, &args) != 0)
    return HSA_STATUS_ERROR;

  if (timeout) *timeout = args.timeout;
  if (size_copied) *size_copied = args.bytes_copied;
  if (is_spm_data_loss) *is_spm_data_loss = (args.has_data_loss != 0);
  return HSA_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
// SVM (Shared Virtual Memory)
///////////////////////////////////////////////////////////////////////////////

hsa_status_t GpuDriver::SVMSetAttr(void* addr, size_t size, uint32_t count,
                                    HSA_SVM_ATTRIBUTE* attrs) const {
  size_t s_attr = count * sizeof(kfd_ioctl_svm_attribute);
  size_t args_size = sizeof(kfd_ioctl_svm_args) + s_attr;
  std::vector<uint8_t> buf(args_size, 0);
  auto* args = reinterpret_cast<kfd_ioctl_svm_args*>(buf.data());

  args->start_addr = reinterpret_cast<uint64_t>(addr);
  args->size = size;
  args->op = KFD_IOCTL_SVM_OP_SET_ATTR;
  args->nattr = count;

  static_assert(sizeof(kfd_ioctl_svm_attribute) == sizeof(HSA_SVM_ATTRIBUTE),
                "SVM attribute size mismatch");
  memcpy(args->attrs, attrs, s_attr);

  // Translate node_id to gpu_id for location/access attributes.
  for (uint32_t i = 0; i < count; ++i) {
    if (attrs[i].type != KFD_IOCTL_SVM_ATTR_PREFERRED_LOC &&
        attrs[i].type != KFD_IOCTL_SVM_ATTR_PREFETCH_LOC &&
        attrs[i].type != KFD_IOCTL_SVM_ATTR_ACCESS &&
        attrs[i].type != KFD_IOCTL_SVM_ATTR_ACCESS_IN_PLACE &&
        attrs[i].type != KFD_IOCTL_SVM_ATTR_NO_ACCESS)
      continue;

    if ((attrs[i].type == KFD_IOCTL_SVM_ATTR_PREFERRED_LOC ||
         attrs[i].type == KFD_IOCTL_SVM_ATTR_PREFETCH_LOC) &&
        attrs[i].value == INVALID_NODEID) {
      args->attrs[i].value = KFD_IOCTL_SVM_LOCATION_UNDEFINED;
      continue;
    }

    uint32_t gpu_id = NodeToGpuId(attrs[i].value);
    if (gpu_id == 0 &&
        (attrs[i].type == KFD_IOCTL_SVM_ATTR_ACCESS ||
         attrs[i].type == KFD_IOCTL_SVM_ATTR_ACCESS_IN_PLACE ||
         attrs[i].type == KFD_IOCTL_SVM_ATTR_NO_ACCESS)) {
      debug_print("CPU node invalid for SVM access attribute\n");
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
    args->attrs[i].value = gpu_id;
  }

  // Encode variable-size ioctl command with attribute array size.
  unsigned long svm_cmd = AMDKFD_IOC_SVM +
      (static_cast<unsigned long>(s_attr) << _IOC_SIZESHIFT);
  if (GpuIoctl(fd_, svm_cmd, args) != 0)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::SVMGetAttr(void* addr, size_t size, uint32_t count,
                                    HSA_SVM_ATTRIBUTE* attrs) const {
  size_t s_attr = count * sizeof(kfd_ioctl_svm_attribute);
  size_t args_size = sizeof(kfd_ioctl_svm_args) + s_attr;
  std::vector<uint8_t> buf(args_size, 0);
  auto* args = reinterpret_cast<kfd_ioctl_svm_args*>(buf.data());

  args->start_addr = reinterpret_cast<uint64_t>(addr);
  args->size = size;
  args->op = KFD_IOCTL_SVM_OP_GET_ATTR;
  args->nattr = count;
  memcpy(args->attrs, attrs, s_attr);

  // Translate node_id to gpu_id for access attributes on input.
  for (uint32_t i = 0; i < count; ++i) {
    if (attrs[i].type != KFD_IOCTL_SVM_ATTR_ACCESS &&
        attrs[i].type != KFD_IOCTL_SVM_ATTR_ACCESS_IN_PLACE &&
        attrs[i].type != KFD_IOCTL_SVM_ATTR_NO_ACCESS)
      continue;

    uint32_t gpu_id = NodeToGpuId(attrs[i].value);
    if (gpu_id == 0) {
      debug_print("CPU node invalid for SVM access attribute\n");
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
    args->attrs[i].value = gpu_id;
  }

  // Encode variable-size ioctl command with attribute array size.
  unsigned long svm_cmd = AMDKFD_IOC_SVM +
      (static_cast<unsigned long>(s_attr) << _IOC_SIZESHIFT);
  if (GpuIoctl(fd_, svm_cmd, args) != 0)
    return HSA_STATUS_ERROR;

  memcpy(attrs, args->attrs, s_attr);

  // Reverse translate gpu_id back to node_id in response.
  for (uint32_t i = 0; i < count; ++i) {
    if (attrs[i].type != KFD_IOCTL_SVM_ATTR_PREFERRED_LOC &&
        attrs[i].type != KFD_IOCTL_SVM_ATTR_PREFETCH_LOC &&
        attrs[i].type != KFD_IOCTL_SVM_ATTR_ACCESS &&
        attrs[i].type != KFD_IOCTL_SVM_ATTR_ACCESS_IN_PLACE &&
        attrs[i].type != KFD_IOCTL_SVM_ATTR_NO_ACCESS)
      continue;

    switch (attrs[i].value) {
    case KFD_IOCTL_SVM_LOCATION_SYSMEM:
      attrs[i].value = 0;
      break;
    case KFD_IOCTL_SVM_LOCATION_UNDEFINED:
      attrs[i].value = INVALID_NODEID;
      break;
    default:
      attrs[i].value = GpuIdToNodeId(attrs[i].value);
      if (attrs[i].value == INVALID_NODEID) {
        debug_print("invalid GPU ID in SVM response: %d\n", attrs[i].value);
        return HSA_STATUS_ERROR;
      }
      break;
    }
  }

  return HSA_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
// PC Sampling
///////////////////////////////////////////////////////////////////////////////

hsa_status_t GpuDriver::PcSamplingQueryCapabilities(uint32_t node_id,
                                                     void* sample_info,
                                                     size_t size,
                                                     uint32_t* count) const {
  kfd_ioctl_pc_sample_args args = {};
  args.op = KFD_IOCTL_PCS_OP_QUERY_CAPABILITIES;
  args.gpu_id = NodeToGpuId(node_id);
  args.sample_info_ptr = reinterpret_cast<uint64_t>(sample_info);
  args.num_sample_info = count ? *count : 0;

  if (GpuIoctl(fd_, AMDKFD_IOC_PC_SAMPLE, &args) != 0)
    return HSA_STATUS_ERROR;

  if (count) *count = args.num_sample_info;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::PcSamplingCreate(uint32_t node_id,
                                          HsaPcSamplingInfo* sample_info,
                                          HsaPcSamplingTraceId* trace_id) const {
  kfd_ioctl_pc_sample_args args = {};
  args.op = KFD_IOCTL_PCS_OP_CREATE;
  args.gpu_id = NodeToGpuId(node_id);
  args.sample_info_ptr = reinterpret_cast<uint64_t>(sample_info);
  args.num_sample_info = 1;

  int ret = GpuIoctl(fd_, AMDKFD_IOC_PC_SAMPLE, &args);
  if (ret != 0) {
    if (errno == EBUSY) return (hsa_status_t)HSA_STATUS_ERROR_RESOURCE_BUSY;
    return HSA_STATUS_ERROR;
  }

  if (trace_id) *trace_id = args.trace_id;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::PcSamplingDestroy(uint32_t node_id,
                                           HsaPcSamplingTraceId trace_id) const {
  kfd_ioctl_pc_sample_args args = {};
  args.op = KFD_IOCTL_PCS_OP_DESTROY;
  args.gpu_id = NodeToGpuId(node_id);
  args.trace_id = trace_id;

  if (GpuIoctl(fd_, AMDKFD_IOC_PC_SAMPLE, &args) != 0)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::PcSamplingStart(uint32_t node_id,
                                         HsaPcSamplingTraceId trace_id) const {
  kfd_ioctl_pc_sample_args args = {};
  args.op = KFD_IOCTL_PCS_OP_START;
  args.gpu_id = NodeToGpuId(node_id);
  args.trace_id = trace_id;

  if (GpuIoctl(fd_, AMDKFD_IOC_PC_SAMPLE, &args) != 0)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::PcSamplingStop(uint32_t node_id,
                                        HsaPcSamplingTraceId trace_id) const {
  kfd_ioctl_pc_sample_args args = {};
  args.op = KFD_IOCTL_PCS_OP_STOP;
  args.gpu_id = NodeToGpuId(node_id);
  args.trace_id = trace_id;

  if (GpuIoctl(fd_, AMDKFD_IOC_PC_SAMPLE, &args) != 0)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
// Debug
///////////////////////////////////////////////////////////////////////////////

hsa_status_t GpuDriver::DbgEnable(void** runtime_ptr,
                                   uint32_t* runtime_size) const {
  // Allocate runtime info buffer, matching hsaKmtDbgEnableCtx.
  uint32_t rinfo_sz = sizeof(struct kfd_runtime_info);
  void* rinfo = malloc(rinfo_sz);
  if (!rinfo) return HSA_STATUS_ERROR;

  kfd_ioctl_dbg_trap_args args = {};
  args.pid = getpid();
  args.op = KFD_IOC_DBG_TRAP_ENABLE;
  args.enable.rinfo_ptr = reinterpret_cast<uint64_t>(rinfo);
  args.enable.rinfo_size = rinfo_sz;
  args.enable.dbg_fd = fd_;

  if (GpuIoctl(fd_, AMDKFD_IOC_DBG_TRAP, &args) != 0) {
    free(rinfo);
    return HSA_STATUS_ERROR;
  }

  if (runtime_ptr) *runtime_ptr = rinfo;
  if (runtime_size) *runtime_size = rinfo_sz;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::DbgDisable() const {
  kfd_ioctl_dbg_trap_args args = {};
  args.pid = getpid();
  args.op = KFD_IOC_DBG_TRAP_DISABLE;
  args.enable.dbg_fd = fd_;

  if (GpuIoctl(fd_, AMDKFD_IOC_DBG_TRAP, &args) != 0)
    return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::DbgGetDeviceData(void** data, uint32_t* count,
                                           uint32_t* entry_size) const {
  // Allocate buffer internally, matching hsaKmtDbgGetDeviceDataCtx.
  // Caller expects *data to be malloc'd and will free it with std::free.
  // Use a generous max (4096 devices) rather than UINT32_MAX to avoid
  // multi-GB allocations that cause heap issues in forked children.
  constexpr uint32_t kMaxDevices = 4096;
  uint32_t esz = sizeof(struct kfd_dbg_device_info_entry);

  void* buf = malloc(static_cast<size_t>(esz) * kMaxDevices);
  if (!buf) return HSA_STATUS_ERROR;

  kfd_ioctl_dbg_trap_args args = {};
  args.pid = getpid();
  args.op = KFD_IOC_DBG_TRAP_GET_DEVICE_SNAPSHOT;
  args.device_snapshot.snapshot_buf_ptr = reinterpret_cast<uint64_t>(buf);
  args.device_snapshot.num_devices = kMaxDevices;
  args.device_snapshot.entry_size = esz;

  if (GpuIoctl(fd_, AMDKFD_IOC_DBG_TRAP, &args) != 0) {
    free(buf);
    return HSA_STATUS_ERROR;
  }

  *data = buf;
  if (count) *count = args.device_snapshot.num_devices;
  if (entry_size) *entry_size = args.device_snapshot.entry_size;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::DbgGetQueueData(void** data, uint32_t* count,
                                          uint32_t* entry_size,
                                          bool suspend) const {
  // Two-pass approach matching hsaKmtDbgGetQueueDataCtx.
  // Pass 1: query number of queues (NULL buffer, n_entries=0).
  uint32_t esz = sizeof(struct kfd_queue_snapshot_entry);
  uint32_t n = 0;

  {
    kfd_ioctl_dbg_trap_args args = {};
    args.pid = getpid();
    args.op = KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT;
    args.queue_snapshot.snapshot_buf_ptr = 0;
    args.queue_snapshot.num_queues = 0;
    args.queue_snapshot.entry_size = esz;
    args.queue_snapshot.exception_mask = KFD_EC_MASK(EC_QUEUE_NEW);

    if (GpuIoctl(fd_, AMDKFD_IOC_DBG_TRAP, &args) != 0)
      return HSA_STATUS_ERROR;

    n = args.queue_snapshot.num_queues;
  }

  // Allocate buffer for queue data (may be 0 entries — still valid).
  size_t alloc_size = static_cast<size_t>(n) * esz;
  void* buf = nullptr;
  if (alloc_size > 0) {
    buf = malloc(alloc_size);
    if (!buf) return HSA_STATUS_ERROR;
  }

  // If no queues, return success with empty data.
  if (n == 0) {
    *data = buf;
    if (count) *count = 0;
    if (entry_size) *entry_size = esz;
    return HSA_STATUS_SUCCESS;
  }

  // Pass 2: fill buffer.
  {
    kfd_ioctl_dbg_trap_args args = {};
    args.pid = getpid();
    args.op = KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT;
    args.queue_snapshot.snapshot_buf_ptr = reinterpret_cast<uint64_t>(buf);
    args.queue_snapshot.num_queues = n;
    args.queue_snapshot.entry_size = esz;
    args.queue_snapshot.exception_mask = KFD_EC_MASK(EC_QUEUE_NEW);

    if (GpuIoctl(fd_, AMDKFD_IOC_DBG_TRAP, &args) != 0) {
      free(buf);
      return HSA_STATUS_ERROR;
    }

    n = args.queue_snapshot.num_queues;
  }

  // Suspend queues and re-snapshot to get updated state.
  if (suspend && n > 0) {
    // Extract queue IDs from snapshot entries.
    uint32_t* queue_ids = static_cast<uint32_t*>(malloc(sizeof(uint32_t) * n));
    if (!queue_ids) {
      free(buf);
      return HSA_STATUS_ERROR;
    }

    auto* entries = reinterpret_cast<struct kfd_queue_snapshot_entry*>(buf);
    for (uint32_t i = 0; i < n; i++)
      queue_ids[i] = entries[i].queue_id;

    kfd_ioctl_dbg_trap_args suspend_args = {};
    suspend_args.pid = getpid();
    suspend_args.op = KFD_IOC_DBG_TRAP_SUSPEND_QUEUES;
    suspend_args.suspend_queues.queue_array_ptr = reinterpret_cast<uint64_t>(queue_ids);
    suspend_args.suspend_queues.num_queues = n;
    suspend_args.suspend_queues.exception_mask = KFD_EC_MASK(EC_QUEUE_NEW);

    // Suspend ioctl returns positive value (count of suspended queues) on
    // success, so check < 0 for error (matching libhsakmt).
    if (GpuIoctl(fd_, AMDKFD_IOC_DBG_TRAP, &suspend_args) < 0) {
      free(queue_ids);
      free(buf);
      return HSA_STATUS_ERROR;
    }

    // Pass 3: re-snapshot after suspend.
    kfd_ioctl_dbg_trap_args args = {};
    args.pid = getpid();
    args.op = KFD_IOC_DBG_TRAP_GET_QUEUE_SNAPSHOT;
    args.queue_snapshot.snapshot_buf_ptr = reinterpret_cast<uint64_t>(buf);
    args.queue_snapshot.num_queues = n;
    args.queue_snapshot.entry_size = esz;
    args.queue_snapshot.exception_mask = KFD_EC_MASK(EC_QUEUE_NEW);

    if (GpuIoctl(fd_, AMDKFD_IOC_DBG_TRAP, &args) != 0) {
      free(queue_ids);
      free(buf);
      return HSA_STATUS_ERROR;
    }

    n = args.queue_snapshot.num_queues;
    free(queue_ids);
  }

  *data = buf;
  if (count) *count = n;
  if (entry_size) *entry_size = esz;
  return HSA_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
// AIS (AMD Infinity Storage)
///////////////////////////////////////////////////////////////////////////////

hsa_status_t GpuDriver::AisReadWriteFile(void* device_ptr, size_t size, int fd,
                                          int64_t file_offset,
                                          HsaAisFlags operation,
                                          uint64_t* size_copied,
                                          int32_t* status) const {
  // Look up handle for the device pointer.
  auto it = mem_handles_.find(device_ptr);
  if (it == mem_handles_.end()) return HSA_STATUS_ERROR;

  kfd_ioctl_ais_args args = {};
  args.in.handle = it->second.handle;
  args.in.handle_offset = 0;
  args.in.file_offset = file_offset;
  args.in.size = size;
  args.in.op = static_cast<uint32_t>(operation);
  args.in.fd = fd;

  if (GpuIoctl(fd_, AMDKFD_IOC_AIS_OP, &args) != 0)
    return HSA_STATUS_ERROR;

  if (size_copied) *size_copied = args.out.size_copied;
  if (status) *status = args.out.status;
  return HSA_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
// DMA-buf and sharing
///////////////////////////////////////////////////////////////////////////////

hsa_status_t GpuDriver::GetShareableHandle(void* va, void* mem, size_t size,
                                            core::ShareableHandle* handle) {
  // Windows-only operation.
  return HSA_STATUS_ERROR;
}

hsa_status_t GpuDriver::ExportDMABuf(void *mem, size_t size, int *dmabuf_fd,
                                      size_t *offset) {
  // Find the allocation containing mem (may be a sub-allocation/fragment).
  // Matches libhsakmt's vm_find_object_by_address_range behavior.
  uint64_t handle = 0;
  size_t byte_offset = 0;
  auto it = mem_handles_.upper_bound(mem);
  if (it != mem_handles_.begin()) {
    --it;
    auto* base = reinterpret_cast<uint8_t*>(it->first);
    auto* addr = reinterpret_cast<uint8_t*>(mem);
    if (addr >= base && addr < base + it->second.size) {
      handle = it->second.handle;
      byte_offset = static_cast<size_t>(addr - base);
    }
  }
  if (!handle) return HSA_STATUS_ERROR;

  kfd_ioctl_export_dmabuf_args args = {};
  args.handle = handle;
  args.flags = O_CLOEXEC;

  if (GpuIoctl(fd_, AMDKFD_IOC_EXPORT_DMABUF, &args) != 0) {
    if (errno == EINVAL) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  *dmabuf_fd = static_cast<int>(args.dmabuf_fd);
  if (offset) *offset = byte_offset;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::ImportDMABuf(int dmabuf_fd, core::Agent &agent,
                                      core::ShareableHandle &handle) {
  // Import the DMA-buf as an amdgpu BO. The caller (VMemoryHandleMap) passes
  // the result to GetMemoryCpuAddr which expects an amdgpu_bo_handle, not a
  // raw KFD handle. Matches old KfdDriver flow via hsaKmtHandleImport.
  auto &gpu_agent = static_cast<GpuAgent &>(agent);
  amdgpu_device_handle dev = nullptr;
  void* dev_handle = nullptr;
  if (GetDeviceHandle(gpu_agent.node_id(), &dev_handle) != HSA_STATUS_SUCCESS)
    return HSA_STATUS_ERROR;
  dev = reinterpret_cast<amdgpu_device_handle>(dev_handle);

  struct amdgpu_bo_import_result import_result = {};
  int ret = amdgpu_bo_import(dev, amdgpu_bo_handle_type_dma_buf_fd,
                              static_cast<uint32_t>(dmabuf_fd), &import_result);
  if (ret != 0) return HSA_STATUS_ERROR;

  handle.handle = reinterpret_cast<uint64_t>(import_result.buf_handle);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::Map(core::ShareableHandle handle, void *mem,
                             size_t offset, size_t size,
                             hsa_access_permission_t perms) {
  // handle.handle is an amdgpu_bo_handle from ImportDMABuf.
  amdgpu_bo_handle bo = reinterpret_cast<amdgpu_bo_handle>(handle.handle);
  if (!bo) return HSA_STATUS_ERROR;

  // Convert HSA permission to DRM VA flags (matches libhsakmt MapDrmPerm).
  HsaMemoryMapFlags map_flags = mem_perm(perms);
  uint64_t flags = 0;
  switch (map_flags) {
  case HSA_MEMORY_ACCESS_RO:
    flags = AMDGPU_VM_PAGE_READABLE;
    break;
  case HSA_MEMORY_ACCESS_WO:
    flags = AMDGPU_VM_PAGE_WRITEABLE;
    break;
  case HSA_MEMORY_ACCESS_RW:
    flags = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE;
    break;
  case HSA_MEMORY_ACCESS_NONE:
  default:
    flags = 0;
    break;
  }

  uint64_t va = reinterpret_cast<uint64_t>(mem);
  int ret = amdgpu_bo_va_op(bo, offset, size, va, flags, AMDGPU_VA_OP_MAP);
  if (ret != 0) return HSA_STATUS_ERROR;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::Unmap(core::ShareableHandle handle, void *mem,
                               size_t offset, size_t size) {
  amdgpu_bo_handle bo = reinterpret_cast<amdgpu_bo_handle>(handle.handle);
  if (!bo) return HSA_STATUS_ERROR;

  uint64_t va = reinterpret_cast<uint64_t>(mem);
  int ret = amdgpu_bo_va_op(bo, offset, size, va, 0, AMDGPU_VA_OP_UNMAP);
  if (ret != 0) return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::ReleaseShareableHandle(core::ShareableHandle &handle) {
  // The handle is an amdgpu_bo_handle from ImportDMABuf (via amdgpu_bo_import).
  // Reset metadata then free, matching libhsakmt hsaKmtMemHandleFree.
  amdgpu_bo_handle bo = reinterpret_cast<amdgpu_bo_handle>(handle.handle);

  struct amdgpu_bo_metadata zero_metadata = {};
  int ret = amdgpu_bo_set_metadata(bo, &zero_metadata);
  if (ret != 0) return HSA_STATUS_ERROR;

  ret = amdgpu_bo_free(bo);
  if (ret != 0) return HSA_STATUS_ERROR;

  handle = {};
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::ExportDMABufHandle(const void* mem, size_t size,
                                            int* dmabuf_fd,
                                            uint64_t* offset) const {
  // Find the allocation containing mem (may be a sub-allocation/fragment).
  uint64_t handle = 0;
  uint64_t byte_offset = 0;
  auto it = mem_handles_.upper_bound(const_cast<void*>(mem));
  if (it != mem_handles_.begin()) {
    --it;
    auto* base = reinterpret_cast<const uint8_t*>(it->first);
    auto* addr = reinterpret_cast<const uint8_t*>(mem);
    if (addr >= base && addr < base + it->second.size) {
      handle = it->second.handle;
      byte_offset = static_cast<uint64_t>(addr - base);
    }
  }
  if (!handle) return HSA_STATUS_ERROR;

  kfd_ioctl_export_dmabuf_args args = {};
  args.handle = handle;
  args.flags = O_CLOEXEC;

  if (GpuIoctl(fd_, AMDKFD_IOC_EXPORT_DMABUF, &args) != 0)
    return HSA_STATUS_ERROR;

  *dmabuf_fd = static_cast<int>(args.dmabuf_fd);
  if (offset) *offset = byte_offset;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::HandleImport(const HsaExternalHandleDesc* desc,
                                      HsaHandleImportResult* result,
                                      HsaHandleImportFlags* flags) const {
  if (!desc || !result) return HSA_STATUS_ERROR;

  // Use libdrm_amdgpu for DMA-buf import.
  // Matches libhsakmt hsaKmtHandleImport logic exactly.
  amdgpu_device_handle dev = reinterpret_cast<amdgpu_device_handle>(desc->device_handle);
  if (!dev) return HSA_STATUS_ERROR;

  enum amdgpu_bo_handle_type type;
  switch (desc->type) {
  case HSA_EXTERNAL_HANDLE_GEM_FLINK_NAME:
    type = amdgpu_bo_handle_type_gem_flink_name;
    break;
  case HSA_EXTERNAL_HANDLE_KMS:
    type = amdgpu_bo_handle_type_kms;
    break;
  case HSA_EXTERNAL_HANDLE_DMA_BUF:
  default:
    type = amdgpu_bo_handle_type_dma_buf_fd;
    break;
  }

  struct amdgpu_bo_import_result import_result = {};
  int ret = amdgpu_bo_import(dev, type, static_cast<uint32_t>(desc->fd),
                              &import_result);
  if (ret != 0) return HSA_STATUS_ERROR;

  if (flags->ui32.IPCHandle) {
    // Query buffer object for pre-existing metadata.
    struct amdgpu_bo_info info = {};
    ret = amdgpu_bo_query_info(import_result.buf_handle, &info);
    if (ret != 0) return HSA_STATUS_ERROR;

    uint32_t metadata = info.metadata.umd_metadata[0];
    uint32_t size_metadata = info.metadata.size_metadata;

    if (flags->ui32.UpdateMetadata && !flags->ui32.SysMem) {
      if (size_metadata != 0) {
        // Return pre-existing metadata.
        result->metadata = static_cast<HSAuint32>(metadata);
      } else {
        // Set metadata with the IPC handle token.
        struct amdgpu_bo_metadata buf_info = {};
        buf_info.size_metadata = sizeof(HSAuint32);
        buf_info.umd_metadata[0] = static_cast<uint32_t>(desc->metadata);
        amdgpu_bo_set_metadata(import_result.buf_handle, &buf_info);
      }
    } else if (desc->metadata != metadata) {
      // Metadata validation failed — return what was on the BO.
      result->metadata = static_cast<HSAuint32>(metadata);
      result->buf_handle = reinterpret_cast<HsaMemoryObjectHandle>(import_result.buf_handle);
      result->alloc_size = static_cast<HSAuint64>(import_result.alloc_size);
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
  }

  result->buf_handle = reinterpret_cast<HsaMemoryObjectHandle>(import_result.buf_handle);
  result->alloc_size = static_cast<HSAuint64>(import_result.alloc_size);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::ShareMemory(void* mem, size_t size,
                                     HsaSharedMemoryHandle* handle) const {
  auto it = mem_handles_.find(mem);
  if (it == mem_handles_.end()) return HSA_STATUS_ERROR;

  kfd_ioctl_ipc_export_handle_args args = {};
  args.handle = it->second.handle;
  args.gpu_id = NodeToGpuId(it->second.node_id);
  args.flags = it->second.mflags;

  if (GpuIoctl(fd_, AMDKFD_IOC_IPC_EXPORT_HANDLE, &args) != 0)
    return HSA_STATUS_ERROR;

  // Populate the full HsaSharedMemoryHandle (HSAuint32[8]) matching
  // libhsakmt's HsaSharedMemoryStruct layout:
  //   [0..3] ShareHandle (16 bytes, kernel IPC handle)
  //   [4]    ApeInfo.type (aperture type -- 0 for GpuDriver)
  //   [5]    ApeInfo.idx  (aperture index -- 0 for GpuDriver)
  //   [6]    SizeInPages
  //   [7]    ExportGpuId
  auto* h = reinterpret_cast<HSAuint32*>(handle);
  memcpy(h, args.share_handle, sizeof(args.share_handle));
  h[4] = 0;  // ApeInfo.type
  h[5] = 0;  // ApeInfo.idx
  h[6] = static_cast<HSAuint32>(size >> 12);  // SizeInPages (PAGE_SHIFT=12)
  h[7] = args.gpu_id;                         // ExportGpuId
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::RegisterSharedHandle(const HsaSharedMemoryHandle* handle,
                                              void** address,
                                              HSAuint64* size) const {
  // The HsaSharedMemoryHandle (HSAuint32[8]) has an internal structure matching
  // libhsakmt's HsaSharedMemoryStruct:
  //   [0..3] ShareHandle (16 bytes, kernel IPC handle)
  //   [4]    ApeInfo.type
  //   [5]    ApeInfo.idx
  //   [6]    SizeInPages
  //   [7]    ExportGpuId
  const HSAuint32* h = reinterpret_cast<const HSAuint32*>(handle);
  uint32_t size_in_pages = h[6];
  uint32_t export_gpu_id = h[7];
  size_t alloc_size = static_cast<size_t>(size_in_pages) << 12;  // PAGE_SHIFT

  if (alloc_size == 0) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  // Reserve VA space for the imported buffer.
  void* reserved_va = mmap(nullptr, alloc_size, PROT_NONE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (reserved_va == MAP_FAILED) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  kfd_ioctl_ipc_import_handle_args args = {};
  // Copy only the IPC share handle (first 16 bytes).
  memcpy(args.share_handle, h, sizeof(args.share_handle));
  args.gpu_id = export_gpu_id;
  args.va_addr = reinterpret_cast<uint64_t>(reserved_va);

  if (GpuIoctl(fd_, AMDKFD_IOC_IPC_IMPORT_HANDLE, &args) != 0) {
    munmap(reserved_va, alloc_size);
    return HSA_STATUS_ERROR;
  }

  // mmap for CPU access if the kernel provides an mmap offset.
  if (args.mmap_offset) {
    int drm_fd = DrmFdForGpuId(args.gpu_id);
    if (drm_fd < 0) drm_fd = fd_;
    void* mapped = mmap(reserved_va, alloc_size, PROT_READ | PROT_WRITE,
                        MAP_SHARED | MAP_FIXED, drm_fd, args.mmap_offset);
    if (mapped == MAP_FAILED) {
      kfd_ioctl_free_memory_of_gpu_args free_args = {};
      free_args.handle = args.handle;
      GpuIoctl(fd_, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &free_args);
      munmap(reserved_va, alloc_size);
      return HSA_STATUS_ERROR;
    }
    madvise(mapped, alloc_size, MADV_DONTFORK);
  }

  if (address) *address = reserved_va;
  if (size) *size = static_cast<HSAuint64>(alloc_size);

  mem_handles_[reserved_va] = {args.handle, alloc_size, 0, args.flags, {}};
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::RegisterGraphicsHandleToNodes(
    int dmabuf_fd, HsaGraphicsResourceInfo* info, uint32_t num_nodes,
    uint32_t* nodes) const {
  if (!info) return HSA_STATUS_ERROR;

  // Step 1: Query dmabuf info to get size, gpu_id, metadata, and flags.
  // Matches libhsakmt's hsakmt_fmm_register_graphics_handle flow.
  static constexpr uint32_t kMetadataDefaultSize = 64;
  std::vector<uint8_t> metadata(kMetadataDefaultSize, 0);

  kfd_ioctl_get_dmabuf_info_args info_args = {};
  info_args.dmabuf_fd = static_cast<uint32_t>(dmabuf_fd);
  info_args.metadata_size = static_cast<uint32_t>(metadata.size());
  info_args.metadata_ptr = reinterpret_cast<uint64_t>(metadata.data());

  int r = GpuIoctl(fd_, AMDKFD_IOC_GET_DMABUF_INFO, &info_args);
  if (r != 0 && info_args.metadata_size > kMetadataDefaultSize) {
    // Retry with the actual metadata size reported by the kernel.
    metadata.resize(info_args.metadata_size, 0);
    info_args.metadata_ptr = reinterpret_cast<uint64_t>(metadata.data());
    r = GpuIoctl(fd_, AMDKFD_IOC_GET_DMABUF_INFO, &info_args);
  }
  if (r != 0) return HSA_STATUS_ERROR;

  // Step 2: Reserve VA space for the imported buffer.
  void* mem = mmap(nullptr, info_args.size, PROT_NONE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (mem == MAP_FAILED) return HSA_STATUS_ERROR;

  // Step 3: Import the dmabuf. Use the gpu_id from GET_DMABUF_INFO.
  kfd_ioctl_import_dmabuf_args import_args = {};
  import_args.va_addr = reinterpret_cast<uint64_t>(mem);
  import_args.gpu_id = info_args.gpu_id;
  import_args.dmabuf_fd = static_cast<uint32_t>(dmabuf_fd);

  if (GpuIoctl(fd_, AMDKFD_IOC_IMPORT_DMABUF, &import_args) != 0) {
    munmap(mem, info_args.size);
    return HSA_STATUS_ERROR;
  }

  // Step 4: Convert gpu_id to node_id.
  // Matches libhsakmt's hsakmt_gpuid_to_nodeid.
  uint32_t node_id = 0;
  for (size_t i = 0; i < gpu_ids_.size(); ++i) {
    if (gpu_ids_[i] == info_args.gpu_id) {
      node_id = static_cast<uint32_t>(i);
      break;
    }
  }

  // Step 5: Track the allocation and populate the output.
  // Store import flags with CoarseGrain=1, matching libhsakmt's
  // fmm_translate_ioc_to_hsa_flags for graphics-imported BOs.
  HsaMemFlags mflags;
  mflags.Value = 0;
  mflags.ui32.CoarseGrain = 1;
  mem_handles_[mem] = {import_args.handle, static_cast<size_t>(info_args.size),
                       node_id, mflags.Value, {}};

  // Transfer ownership of the metadata buffer to a heap allocation so the
  // caller gets a stable pointer. The local vector would go out of scope.
  void* metadata_out = nullptr;
  if (info_args.metadata_size > 0) {
    metadata_out = malloc(info_args.metadata_size);
    if (metadata_out)
      memcpy(metadata_out, metadata.data(), info_args.metadata_size);
  }

  info->MemoryAddress = mem;
  info->SizeInBytes = info_args.size;
  info->Metadata = metadata_out;
  info->MetadataSizeInBytes = info_args.metadata_size;
  info->NodeId = node_id;

  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::RegisterGraphicsHandleToNodesExt(
    HSAuint64 dmabuf_fd, HsaGraphicsResourceInfo* info, HSAuint64 num_nodes,
    uint32_t* nodes, HSA_REGISTER_MEM_FLAGS flags) const {
  auto ret = RegisterGraphicsHandleToNodes(static_cast<int>(dmabuf_fd), info,
                                            static_cast<uint32_t>(num_nodes), nodes);
  return ret;
}

///////////////////////////////////////////////////////////////////////////////
// DRM-based operations
///////////////////////////////////////////////////////////////////////////////

hsa_status_t GpuDriver::GetDeviceHandle(uint32_t node_id,
                                         void** device_handle) const {
  assert(device_handle);

  if (!process_state_ ||
      static_cast<size_t>(node_id) >= process_state_->amdgpu_handles.size() ||
      !process_state_->amdgpu_handles[node_id]) {
    return HSA_STATUS_ERROR;
  }

  *device_handle = reinterpret_cast<void*>(
      process_state_->amdgpu_handles[node_id]);
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::GetMemoryCpuAddr(void* device_handle,
                                           void* mem_handle, int* drm_fd,
                                           uint64_t* cpu_addr) const {
  amdgpu_device_handle dev = reinterpret_cast<amdgpu_device_handle>(device_handle);
  amdgpu_bo_handle bo = reinterpret_cast<amdgpu_bo_handle>(mem_handle);

  int render_fd = amdgpu_device_get_fd(dev);
  if (render_fd < 0) return HSA_STATUS_ERROR;

  // Export to GEM handle, then query the DRM mmap offset.
  // The offset is used by the caller with mmap(MAP_SHARED | MAP_FIXED, fd, offset).
  uint32_t gem_handle = 0;
  int ret = amdgpu_bo_export(bo, amdgpu_bo_handle_type_kms, &gem_handle);
  if (ret != 0) return HSA_STATUS_ERROR;

  union drm_amdgpu_gem_mmap args = {};
  args.in.handle = gem_handle;
  ret = ioctl(render_fd, DRM_IOCTL_AMDGPU_GEM_MMAP, &args);
  if (ret != 0) return HSA_STATUS_ERROR;

  if (drm_fd) *drm_fd = render_fd;
  if (cpu_addr) *cpu_addr = args.out.addr_ptr;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::AllocateMemoryAlign(uint32_t node, size_t size,
                                             size_t alignment,
                                             HsaMemFlags flags,
                                             void** mem) const {
  // OnlyAddress: reserve a VA range without allocating GPU memory.
  // Matches libhsakmt's fmm_allocate_va (aperture_allocate_area_aligned).
  if (flags.ui32.OnlyAddress) {
    if (alignment == 0) alignment = kPageSize;
    size_t padded = size + alignment - kPageSize;
    void* raw = mmap(nullptr, padded, PROT_NONE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (raw == MAP_FAILED) return HSA_STATUS_ERROR;
    // Align within the padded reservation.
    uintptr_t aligned = (reinterpret_cast<uintptr_t>(raw) + alignment - 1) & ~(alignment - 1);
    // Trim leading/trailing excess.
    if (aligned > reinterpret_cast<uintptr_t>(raw))
      munmap(raw, aligned - reinterpret_cast<uintptr_t>(raw));
    uintptr_t end = reinterpret_cast<uintptr_t>(raw) + padded;
    uintptr_t aligned_end = aligned + size;
    if (end > aligned_end)
      munmap(reinterpret_cast<void*>(aligned_end), end - aligned_end);

    void* ptr = reinterpret_cast<void*>(aligned);
    mem_handles_[ptr] = {0, size, node, flags.Value, {}};
    *mem = ptr;
    return HSA_STATUS_SUCCESS;
  }

  uint64_t handle = 0;
  uint32_t gpu_id = NodeToGpuId(node);
  if (gpu_id == 0 && !all_gpu_id_array_.empty())
    gpu_id = all_gpu_id_array_[0];
  void* ptr = AllocateDriverMemory(fd_, DrmFdForGpuId(gpu_id), flags, gpu_id, size, &handle);
  if (!ptr) return HSA_STATUS_ERROR;

  mem_handles_[ptr] = {handle, size, node, flags.Value, {}};
  *mem = ptr;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::MemoryCpuMap(HsaMemoryObjectHandle handle,
                                      void** cpu_ptr) const {
  amdgpu_bo_handle bo = reinterpret_cast<amdgpu_bo_handle>(handle);
  void* ptr = nullptr;
  int ret = amdgpu_bo_cpu_map(bo, &ptr);
  if (ret != 0) return HSA_STATUS_ERROR;

  *cpu_ptr = ptr;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::MemoryVaMap(HsaMemoryObjectHandle handle,
                                     uint64_t offset, uint64_t size,
                                     uint64_t va, uint32_t access) const {
  // Use amdgpu_bo_va_op to properly extract the GEM handle from the BO.
  // Matches libhsakmt hsaKmtMemoryVaMap (memory.c:1005).
  amdgpu_bo_handle bo = reinterpret_cast<amdgpu_bo_handle>(handle);
  if (!bo) return HSA_STATUS_ERROR;

  uint64_t flags = 0;
  switch (access) {
  case HSA_MEMORY_ACCESS_RO:
    flags = AMDGPU_VM_PAGE_READABLE;
    break;
  case HSA_MEMORY_ACCESS_WO:
    flags = AMDGPU_VM_PAGE_WRITEABLE;
    break;
  case HSA_MEMORY_ACCESS_RW:
    flags = AMDGPU_VM_PAGE_READABLE | AMDGPU_VM_PAGE_WRITEABLE;
    break;
  case HSA_MEMORY_ACCESS_NONE:
  default:
    flags = 0;
    break;
  }

  int va_ret = amdgpu_bo_va_op(bo, offset, size, va, flags, AMDGPU_VA_OP_MAP);
  if (va_ret != 0) return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::MemoryVaUnmap(HsaMemoryObjectHandle handle,
                                       uint64_t offset, uint64_t size,
                                       uint64_t va) const {
  // Use amdgpu_bo_va_op to properly extract the GEM handle from the BO.
  // Matches libhsakmt hsaKmtMemoryVaUnmap (memory.c:1024).
  amdgpu_bo_handle bo = reinterpret_cast<amdgpu_bo_handle>(handle);
  if (!bo) return HSA_STATUS_ERROR;

  int ret = amdgpu_bo_va_op(bo, offset, size, va, 0, AMDGPU_VA_OP_UNMAP);
  if (ret != 0) return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////
// Pointer query and misc
///////////////////////////////////////////////////////////////////////////////

hsa_status_t GpuDriver::QueryPointerInfo(const void* ptr,
                                          HsaPointerInfo* info) const {
  if (!info) return HSA_STATUS_ERROR;
  memset(info, 0, sizeof(*info));

  // Search mem_handles_ for a range containing ptr.
  auto it = mem_handles_.upper_bound(const_cast<void*>(ptr));
  if (it != mem_handles_.begin()) {
    --it;
    uintptr_t base = reinterpret_cast<uintptr_t>(it->first);
    uintptr_t query = reinterpret_cast<uintptr_t>(ptr);
    if (query >= base && query < base + it->second.size) {
      const auto& mh = it->second;
      info->GPUAddress = base;
      info->SizeInBytes = mh.size;
      info->Node = mh.node_id;
      info->Type = HSA_POINTER_ALLOCATED;

      // CPUAddress: For USERPTR/GTT (system memory), CPU and GPU VA are the
      // same. For VRAM without HostAccess, CPU address is NULL. Matches
      // libhsakmt fmm_get_mem_info behavior.
      HsaMemFlags flags;
      flags.Value = mh.mflags;
      // If the allocation has HostAccess or is on a CPU node (system memory),
      // CPU VA == GPU VA. Otherwise NULL (e.g., device-only VRAM).
      if (flags.ui32.HostAccess || mh.node_id == 0)
        info->CPUAddress = it->first;

      info->MemFlags.Value = mh.mflags;

      // Convert mapped GPU IDs to node IDs for the caller.
      query_mapped_nodes_buf_.clear();
      for (uint32_t gid : mh.mapped_gpu_ids) {
        uint32_t nid = GpuIdToNodeId(gid);
        query_mapped_nodes_buf_.push_back(nid);
      }
      info->NMappedNodes = static_cast<uint32_t>(query_mapped_nodes_buf_.size());
      info->MappedNodes = query_mapped_nodes_buf_.empty()
                              ? nullptr
                              : query_mapped_nodes_buf_.data();

      return HSA_STATUS_SUCCESS;
    }
  }

  return HSA_STATUS_ERROR;
}

hsa_status_t GpuDriver::SetMemoryUserData(const void* ptr,
                                            void* user_data) const {
  // Simplified: no separate user_data tracking in this implementation.
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::FreeMemoryHandle(HsaMemoryObjectHandle handle) const {
  amdgpu_bo_handle bo = reinterpret_cast<amdgpu_bo_handle>(handle);

  // Reset metadata before freeing (matches libhsakmt hsaKmtMemHandleFree).
  struct amdgpu_bo_metadata zero_metadata = {};
  int ret = amdgpu_bo_set_metadata(bo, &zero_metadata);
  if (ret != 0) return HSA_STATUS_ERROR;

  ret = amdgpu_bo_free(bo);
  if (ret != 0) return HSA_STATUS_ERROR;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t GpuDriver::ReturnAsanHeaderPage(void* addr) const {
  // ASAN header page management not yet supported with direct ioctls.
  return HSA_STATUS_ERROR;
}

} // namespace AMD
} // namespace rocr
