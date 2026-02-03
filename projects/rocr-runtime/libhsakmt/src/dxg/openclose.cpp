/*
 * Copyright © 2014 Advanced Micro Devices, Inc.
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

#include <stdlib.h>
#include <cstring>
#include <sys/types.h>
#include <sys/stat.h>
#include <cstdint>
#if defined(__linux__)
#include <sys/mman.h>
#include <sys/sysinfo.h>
#include <linux/mman.h>
#endif
#include <fcntl.h>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <mutex>
#include <algorithm>
#include "util/os.h"
#include "util/utils.h"

hsakmtRuntime *dxg_runtime = new hsakmtRuntime();

void hsakmtRuntime::HeapInit() {
    if (ReserveLocalHeapSpace()) {
        InitLocalHeapMgr();
    } else {
        pr_err("Local heap reservation failed, local heap manager not initialized\n");
    }

    if (ReserveSystemHeapSpace()) {
        InitSystemHeapMgr();
    } else {
        pr_err("System heap reservation failed, system heap manager not initialized\n");
    }

    if (InitHandleApertureSpace()) {
        InitHandleApertureMgr();
    } else {
        pr_warn("InitHandleApertureSpace failed, handle aperture operations will be unavailable\n");
    }
}

void hsakmtRuntime::HeapFini() {
    if (local_heap_space_start_ != 0) {
        FreeLocalHeapSpace();
    }

    if (system_heap_space_start_ != 0) {
        FreeSystemHeapSpace();
    }
    // Destroy heap managers to prevent stale pointers
    // This is needed when topology refresh to re-initializes heaps
    system_heap_mgr_.reset();
    local_heap_mgr_.reset();
    handle_aperture_mgr_.reset();
}

bool hsakmtRuntime::ReserveSvmSpace(uint64_t &base, uint64_t &size, uint64_t align) {
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
        ptr = rocr::os::ReserveMemory(nullptr, sys_va_size, align);
        if (ptr == nullptr) {
            pr_err("fail to reserve cpu va in %d time!\n", i);
            break;
        }

        sys_va[i] = (uint64_t)ptr;

        int match_cnt = 0;
        for (uint32_t j = 0; j < num_adapters; j++) {
          device = WddmDevice(j);
            uint64_t start = (base == 0) ? (uint64_t)ptr : base;
            uint64_t end = start + ((base == 0) ? sys_va_size : size) + 1;

            if (wsl::thunk::d3dthunk::ReserveGpuVirtualAddress(
                        device->GetAdapter(), size,
                        start,
                        end, &local_va) == ErrorCode::Success) {

                match_cnt++;
                base = local_va;
                pr_debug("success to reserve gpu va %llx and va cpu %p in %d time\n",
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
      // Windows can't release a portion of reserved space.
      // The logic shouldn't add extra space in reservation - (size + align)
#if defined(__linux__)
        /* release cpu unused ranges*/
        uint64_t left_size = local_va - sys_va[match_index];
        uint64_t right_size = align - left_size;
        if ((left_size > 0) && !rocr::os::ReleaseMemory((void*)sys_va[match_index], left_size))
            pr_err("fail to unmap left %lx with size %llx\n", sys_va[match_index], left_size);
        if ((right_size > 0) && !rocr::os::ReleaseMemory((void*)(local_va + size), right_size))
            pr_err("fail to unmap right %lx with size %llx\n", (local_va + size), right_size);
#endif
    } else {
        pr_err("fail to reserve Local Heap Space!\n");
        base = 0;
        size = 0;
    }

    /* free match fail address for cpu va */
    int free = match_index >= 0 ? match_index : 16;
    for (int j = 0; j < free; j++) {
        if (sys_va[j] != 0 && !rocr::os::ReleaseMemory((void*)sys_va[j], sys_va_size)) {
            pr_err("fail to unmap %d %llx\n", j, sys_va[j]);
        }
    }

    return match_index >= 0;
}

/*
 * To find the avaliable same range for cpu
 * virtual space and gpu virtual space.
 * sys_va_size of cpu va range is larger 1G
 * than gpu va range, otherwise ReserveGPUVirtualAddress
 * will return error.
 */
bool hsakmtRuntime::ReserveLocalHeapSpace() {
  wsl::thunk::WDDMDevice* device;
  uint64_t total_local_size = 0;
  uint64_t align = 0x40000000; /* 1G */
  size_t num_adapters = get_num_wddmdev();

  for (uint32_t j = 0; j < num_adapters; j++) {
    device = WddmDevice(j);
    if (device == nullptr) {
      return false;
    }
    // For APU, use non local memory(shared GPU memory) as GPU memory,
    // because it has small local memory
    if (device->IsDgpu()) {
        total_local_size += rocr::AlignUp(device->LocalHeapSize(), align) * 2;
    } else {
        total_local_size += rocr::AlignUp(
            std::max(device->LocalHeapSize(), device->NonLocalHeapSize()), align) * 2;
    }
  }
  local_heap_space_start_ = 0;
  local_heap_space_size_ = total_local_size;
  return ReserveSvmSpace(local_heap_space_start_, local_heap_space_size_, align);
}

bool hsakmtRuntime::FreeSvmSpace(uint64_t &base, uint64_t &size) {
    wsl::thunk::WDDMDevice* device;
    size_t num_adapters = get_num_wddmdev();
    if (size == 0) {
        return true;
    }
    for (uint32_t j = 0; j < num_adapters; j++) {
      device = WddmDevice(j);
        if (device == nullptr) {
            pr_err("FreeSvmSpace: adapter %d is nullptr!\n", j);
            return false;
        }
        wsl::thunk::d3dthunk::FreeGpuVirtualAddress(device->GetAdapter(), base, size);
    }

    void *cpu = (void *)base;
    auto r = rocr::os::ReleaseMemory(cpu, size);
    base = 0;
    size = 0;
    return r;
}

bool hsakmtRuntime::FreeLocalHeapSpace() {
    return FreeSvmSpace(local_heap_space_start_, local_heap_space_size_);
}

void hsakmtRuntime::InitLocalHeapMgr() {
  local_heap_mgr_ = std::make_unique<wsl::thunk::VaMgr>(local_heap_space_start_,
                                          local_heap_space_size_,
                                          DEFAULT_GPU_PAGE_SIZE);
}

bool hsakmtRuntime::ReserveSystemHeapSpace() {
  constexpr uint64_t max_ram = 0x10000000000ULL;
  constexpr uint64_t alignment = 0x100000000ULL;
  uint64_t total_ram = rocr::os::HostTotalPhysicalMemory();
  // minimum of reserve size is 8G, maximum of reserve size is 1T.
  total_ram = rocr::AlignUp(total_ram, static_cast<size_t>(alignment) * 2);
  system_heap_space_size_ = (total_ram > max_ram) ? max_ram : total_ram;
  return ReserveSvmSpace(system_heap_space_start_, system_heap_space_size_, alignment);
}

bool hsakmtRuntime::FreeSystemHeapSpace(void) {
    return FreeSvmSpace(system_heap_space_start_, system_heap_space_size_);
}

bool hsakmtRuntime::CommitSystemHeapSpace(void* addr, int64_t size, bool lock) {
  rocr::os::MemProt protFlags = rocr::os::MEM_PROT_RWX;
    if (lock) assert(false);
        //mapFlags |= MAP_LOCKED;
    if (!rocr::os::CommitMemory(addr, size, protFlags)) {
        pr_err("fail to commit %s addr = %p\n", (lock ? "locked" : ""), addr);
        return false;
    }
    /*if (!Runtime::runtime_singleton_->PinWARequired())
      return true;*/
#if defined(__linux__)
    /*
     * Do not make the pages in this range available to the child
     * after a fork(2).  This is useful to prevent copy-on-write
     * semantics from changing the physical location of a page if
     * the parent writes to it after a fork(2).  (Such page
     * relocations cause problems for hardware that DMAs into the
     * page.)
     *
     * https://man7.org/linux/man-pages/man2/madvise.2.html
     */
    if (madvise(addr, size, MADV_DONTFORK))
        pr_err("fail to set MADV_DONTFORK for addr = %p\n", addr);
#endif
    return true;
}

bool hsakmtRuntime::DecommitSystemHeapSpace(void* addr, int64_t size) {
    if (!rocr::os::UncommitMemory(addr, size)) {
        pr_err("fail to decommit addr = %p\n", addr);
        return false;
    }
    return true;
}

void hsakmtRuntime::InitSystemHeapMgr() {
  system_heap_mgr_ = std::make_unique<wsl::thunk::VaMgr>(system_heap_space_start_,
                                          system_heap_space_size_,
                                          DEFAULT_GPU_PAGE_SIZE);
}

ErrorCode hsakmtRuntime::ReserveGpuVirtualAddress(const thunk_proxy::AllocDomain domain,
        gpusize hit_base_addr, gpusize size,
        gpusize *out_gpu_virt_addr, gpusize alignment, bool lock) {
    gpusize gpu_addr = 0;
    ErrorCode code = ErrorCode::Success;

    uint64_t align = alignment == 0 ? (64 * 1024) : alignment; // default 64K alignment
    if (size >= GPU_HUGE_PAGE_SIZE)
        align = GPU_HUGE_PAGE_SIZE;

    if (domain == thunk_proxy::kSystem) {
        if (!system_heap_mgr_) {
            *out_gpu_virt_addr = 0;
            return ErrorCode::OutOfMemory;
        }

        gpu_addr = system_heap_mgr_->Alloc(size, align, hit_base_addr);
        if (gpu_addr == 0)
            code = ErrorCode::OutOfMemory;
        else if (!CommitSystemHeapSpace((void*)gpu_addr, size, lock)) {
            system_heap_mgr_->Free(gpu_addr);
            code = ErrorCode::SyscallFail;
        }
    } else {
        if (!local_heap_mgr_) {
            *out_gpu_virt_addr = 0;
            return ErrorCode::OutOfGpuMemory;
        }

        gpu_addr = local_heap_mgr_->Alloc(size, align, hit_base_addr);
        if (gpu_addr == 0)
            code = ErrorCode::OutOfGpuMemory;
    }

    *out_gpu_virt_addr = (code == ErrorCode::Success) ? gpu_addr : 0;
    return code;
}

ErrorCode hsakmtRuntime::FreeGpuVirtualAddress(const thunk_proxy::AllocDomain domain,
        gpusize gpu_addr, gpusize size) {
    auto code = ErrorCode::Success;

    if (domain == thunk_proxy::kSystem) {
        DecommitSystemHeapSpace((void *)gpu_addr, size);
        if (system_heap_mgr_) {
            system_heap_mgr_->Free(gpu_addr);
        }
    } else {
        if (local_heap_mgr_) {
            local_heap_mgr_->Free(gpu_addr);
        }
    }

    return code;
}

bool hsakmtRuntime::CommitSystemHeapSpaceIPC(void* addr, int64_t size, int &memfd, bool lock) {
#if defined(__linux__)
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
#else
  assert(!"Unimplemented!");
#endif
    return true;
}

bool hsakmtRuntime::DecommitSystemHeapSpaceIPC(void* addr, int64_t size, int &memfd) {
#if defined(__linux__)
    if (munmap(addr, size) != 0) {
        pr_err("fail to unmap = %p \n", addr);
        return false;
    }
    close(memfd);
    memfd = -1;
#else
  assert(!"Unimplemented!");
#endif
    return true;
}

ErrorCode hsakmtRuntime::ReserveIPCSysMem(gpusize size,
        gpusize *out_gpu_virt_addr, gpusize alignment,
        int &memfd, bool lock) {
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

ErrorCode hsakmtRuntime::FreeIPCSysMem(gpusize gpu_addr, gpusize size, int &memfd) {
    auto code = ErrorCode::Success;

    DecommitSystemHeapSpaceIPC((void *)gpu_addr, size, memfd);

    system_heap_mgr_->Free(gpu_addr);
    return code;
}

bool hsakmtRuntime::InitHandleApertureSpace() {
	wsl::thunk::WDDMDevice* device;
	size_t num_adapters = get_num_wddmdev();
    handle_aperture_start_ = START_NON_CANONICAL_ADDR;
    handle_aperture_size_ = 1ULL << 47;

    while (handle_aperture_start_ < END_NON_CANONICAL_ADDR - 1) {
		for (uint32_t j = 0; j < num_adapters;) {
        device = WddmDevice(j);
	        if (device == nullptr)
	            return false;

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
    pr_err("failed to create non-canonical addr aperture\n");

    return false;
}

void hsakmtRuntime::InitHandleApertureMgr() {
  handle_aperture_mgr_ = std::make_unique<wsl::thunk::VaMgr>(handle_aperture_start_,
                                                 handle_aperture_size_,
                                                 DEFAULT_GPU_PAGE_SIZE);
}

ErrorCode hsakmtRuntime::HandleApertureAlloc(gpusize size, gpusize *out_gpu_virt_addr) {
    uint64_t align = DEFAULT_GPU_PAGE_SIZE;

    if (size >= GPU_HUGE_PAGE_SIZE)
        align = GPU_HUGE_PAGE_SIZE;

    *out_gpu_virt_addr = handle_aperture_mgr_->Alloc(size, align);
    if (*out_gpu_virt_addr == 0)
        return ErrorCode::OutOfHandleApeMemory;

    return ErrorCode::Success;
}

void hsakmtRuntime::HandleApertureFree(gpusize gpu_addr) {
    handle_aperture_mgr_->Free(gpu_addr);
}

/* is_forked_child detects when the process has forked since the last
 * time this function was called. We cannot rely on pthread_atfork
 * because the process can fork without calling the fork function in
 * libc (using clone or calling the system call directly).
 */
bool is_forked_child(void) {
  if (dxg_runtime->is_forked)
    return true;

  int cur_pid = getpid();
  if (dxg_runtime->parent_pid != cur_pid) {
    dxg_runtime->is_forked = true;
    dxg_runtime->parent_pid = cur_pid;
    return true;
  }

  return false;
}

/* Callbacks from pthread_atfork */
static void prepare_fork_handler(void) { dxg_runtime->hsakmt_mutex.lock(); }
static void parent_fork_handler(void) { dxg_runtime->hsakmt_mutex.unlock(); }
static void child_fork_handler(void) {
  dxg_runtime->is_forked = true;
}

/* Call this from the child process after fork. This will clear all
 * data that is duplicated from the parent process, that is not valid
 * in the child.
 * The topology information is duplicated from the parent is valid
 * in the child process so it is not cleared
 */
static void clear_after_fork(void) {
  reset_suballocator();
  clear_allocation_map();

  if (dxg_runtime->dxg_fd >= 0) {
#if defined(__linux__)
    close(dxg_runtime->dxg_fd);
#endif
    dxg_runtime->dxg_fd = -1;
  }
  delete dxg_runtime;
  dxg_runtime = new hsakmtRuntime();

}

static inline void init_page_size(void) {
#if defined(__linux__)
  dxg_runtime->page_size = sysconf(_SC_PAGESIZE);
  dxg_runtime->page_shift = ffs(dxg_runtime->page_size) - 1;
#else
  SYSTEM_INFO si;
  ::GetSystemInfo(&si);
  dxg_runtime->page_size = si.dwPageSize;
  //note ignore page_shift it's not used
#endif
}

static HSAKMT_STATUS init_vars_from_env(void) {
  const char* envvar;

  // Enable debug messages via HSAKMT_DEBUG_LEVEL environment variable.
  // Libraries normally suppress messages; this allows debugging output when needed.
  if ((envvar = getenv("HSAKMT_DEBUG_LEVEL")) != nullptr) {
    dxg_runtime->hsakmt_debug_level = atoi(envvar);
  }

  // Enable Zero Frame Buffer (ZFB) support if HSA_ZFB is set.
  if ((envvar = getenv("HSA_ZFB")) != nullptr) {
    dxg_runtime->zfb_support = atoi(envvar);
  }

  // Enable vendor-specific AQL packet processing if WSLKMT_VENDOR_PACKET is set.
  if ((envvar = getenv("WSLKMT_VENDOR_PACKET")) != nullptr) {
    dxg_runtime->vendor_packet_process = atoi(envvar);
  }

  // Enable thunk sub-allocator via WSL_ENABLE_THUNK_SUB_ALLOCATOR.
  if ((envvar = getenv("WSL_ENABLE_THUNK_SUB_ALLOCATOR")) != nullptr) {
    dxg_runtime->enable_thunk_sub_allocator = atoi(envvar);
  }

  // Enable PM4 packet usage if ROCR_USE_PM4 is set.
  if ((envvar = getenv("ROCR_USE_PM4")) != nullptr) {
    dxg_runtime->use_pm4_ = atoi(envvar);
  }

  // Disable wait timeout if ROCR_DISABLE_WAIT_TIMEOUT is set.
  if ((envvar = getenv("ROCR_DISABLE_WAIT_TIMEOUT")) != nullptr) {
    dxg_runtime->disable_wait_timeout_ = atoi(envvar);
  }

  // Check available system memory before allocation if WSL_CHECK_AVAIL_SYSRAM is "1".
  if ((envvar = getenv("WSL_CHECK_AVAIL_SYSRAM")) != nullptr) {
    dxg_runtime->check_avail_sysram = (envvar[0] == '1' && envvar[1] == '\0');
  }

  // Set default GPU node based on ROCR_VISIBLE_DEVICES. Extract the first numeric value
  // from the environment variable and add 1 to convert to 1-indexed node ID.
  if ((envvar = getenv("ROCR_VISIBLE_DEVICES")) != nullptr) {
    std::string devices(envvar);
    size_t first_num_pos = devices.find_first_of("0123456789");
    if (first_num_pos != std::string::npos) {
      dxg_runtime->default_node = std::stoi(devices.substr(first_num_pos));
    }
  }

  // Determine SVM (Shared Virtual Memory) API support. Enabled by default unless
  // HSA_USE_SVM is explicitly set to "0". Currently forced to false.
  envvar = getenv("HSA_USE_SVM");
  dxg_runtime->is_svm_api_supported =
      !(envvar && envvar[0] == '0' && envvar[1] == '\0') && false;

  if ((envvar = getenv("HSAKMT_DEBUG_SYSMEM")) != nullptr) {
    dxg_runtime->hsakmt_debug_sysmem = atoi(envvar);
  }

  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtOpenKFD(void) {
  HSAKMT_STATUS result;
  int fd = -1;
  HsaSystemProperties sys_props;
  char *error;

  std::lock_guard<std::recursive_mutex> lck(dxg_runtime->hsakmt_mutex);

  /* If the process has forked, the child process must re-initialize
   * it's connection to DXG. Any references tracked by dxg_open_count
   * belong to the parent
   */
  if (is_forked_child())
    clear_after_fork();

  if (dxg_runtime->dxg_open_count == 0) {
    static bool atfork_installed = false;

    result = init_vars_from_env();
    if (result != HSAKMT_STATUS_SUCCESS)
      goto open_failed;

    if (dxg_runtime->dxg_fd < 0) {
#if defined(__linux__)
      fd = open(dxg_runtime->dxg_device_name, O_RDWR | O_CLOEXEC);
#else
      fd = 1;
#endif

      if (fd == -1) {
        result = HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED;
        goto open_failed;
      }

      dxg_runtime->dxg_fd = fd;
    }
    if (!wsl::thunk::dxcore::DxcoreLoader::Instance().Initialize()) {
        pr_err("Failed to load libdxcore.so\n");
        result = HSAKMT_STATUS_ERROR;
        goto dxcore_loader_failed;
    }

#if defined(__linux__)
    hsakmt_hsa_loader_init();
#endif
    init_page_size();





    dxg_runtime->dxg_open_count = 1;

    if (!atfork_installed) {
      /* Atfork handlers cannot be uninstalled and
       * must be installed only once. Otherwise
       * prepare will deadlock when trying to take
       * the same lock multiple times.
       */
#if defined(__linux__)
      pthread_atfork(prepare_fork_handler, parent_fork_handler,
                     child_fork_handler);
#endif
      atfork_installed = true;
    }
  } else {
    dxg_runtime->dxg_open_count++;
    result = HSAKMT_STATUS_KERNEL_ALREADY_OPENED;
  }

  reset_suballocator();
  return result;
dxcore_loader_failed:
#if defined(__linux__)
  close(fd);
#endif
open_failed:

  return result;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtCloseKFD(void) {
  HSAKMT_STATUS result;
  std::lock_guard<std::recursive_mutex> lck(dxg_runtime->hsakmt_mutex);

  if (dxg_runtime->dxg_open_count > 0) {
    if (--dxg_runtime->dxg_open_count == 0) {
      dxg_runtime->HeapFini();
#if defined(__linux__)
      close(dxg_runtime->dxg_fd);
#endif
      dxg_runtime->dxg_fd = -1;
      wsl::thunk::dxcore::DxcoreLoader::Instance().Shutdown();
    }

    result = HSAKMT_STATUS_SUCCESS;
  } else
    result = HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED;

  return result;
}
