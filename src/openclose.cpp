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
#include <sys/mman.h>
#include <sys/sysinfo.h>
#include <linux/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <strings.h>
#include <cassert>
#include "libhsakmt.h"


hsakmtRuntime *dxg_runtime = new hsakmtRuntime();

void hsakmtRuntime::HeapInit() {
    ReserveLocalHeapSpace();
    ReserveSystemHeapSpace();
    InitLocalHeapMgr();
    InitSystemHeapMgr();
}

void hsakmtRuntime::HeapFini() {
    FreeSystemHeapSpace();
    FreeLocalHeapSpace();
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
                        device->GetAdapter(), size,
                        start,
                        end, &local_va) == ErrorCode::Success) {

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
        device = get_wddmdev(j+1);
        if (device == nullptr)
            return -1;
        total_local_size += wsl::AlignUp(device->LocalHeapSize(), align) * 4;
    }

    local_heap_space_start_ = 0;
    local_heap_space_size_ = total_local_size;

    return ReserveSvmSpace(local_heap_space_start_, local_heap_space_size_, align);
}

bool hsakmtRuntime::FreeSvmSpace(uint64_t &base, uint64_t &size) {
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

bool hsakmtRuntime::FreeLocalHeapSpace() {
    return FreeSvmSpace(local_heap_space_start_, local_heap_space_size_);
}

void hsakmtRuntime::InitLocalHeapMgr() {
  local_heap_mgr_ = std::make_unique<wsl::thunk::VaMgr>(local_heap_space_start_,
                                          local_heap_space_size_,
                                          DEFAULT_GPU_PAGE_SIZE);
}

bool hsakmtRuntime::ReserveSystemHeapSpace() {
    struct sysinfo info;
    int ret = sysinfo(&info);
    uint64_t max_ram = 0x10000000000;
    uint64_t alignment = 0x100000000;
    assert(!ret);

    int32_t protFlags = PROT_NONE;
    // minimum of reserve size is 8G, maximum of reserve size is 1T.
    system_heap_space_size_ = std::min(wsl::AlignUp(info.totalram, alignment) * 2, max_ram);

    return ReserveSvmSpace(system_heap_space_start_, system_heap_space_size_, alignment);
}

bool hsakmtRuntime::FreeSystemHeapSpace(void) {
    return FreeSvmSpace(system_heap_space_start_, system_heap_space_size_);
}

void hsakmtRuntime::InitSystemHeapMgr() {
  system_heap_mgr_ = std::make_unique<wsl::thunk::VaMgr>(system_heap_space_start_,
                                          system_heap_space_size_,
                                          DEFAULT_GPU_PAGE_SIZE);
}

/* is_forked_child detects when the process has forked since the last
 * time this function was called. We cannot rely on pthread_atfork
 * because the process can fork without calling the fork function in
 * libc (using clone or calling the system call directly).
 */
bool is_forked_child(void) {
  if (dxg_runtime->is_forked)
    return true;

  pid_t cur_pid = getpid();
  if (dxg_runtime->parent_pid != cur_pid) {
    dxg_runtime->is_forked = true;
    dxg_runtime->parent_pid = cur_pid;
    return true;
  }

  return false;
}

/* Callbacks from pthread_atfork */
static void prepare_fork_handler(void) { pthread_mutex_lock(&dxg_runtime->hsakmt_mutex); }
static void parent_fork_handler(void) { pthread_mutex_unlock(&dxg_runtime->hsakmt_mutex); }
static void child_fork_handler(void) {
  pthread_mutex_init(&dxg_runtime->hsakmt_mutex, NULL);
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
    close(dxg_runtime->dxg_fd);
    dxg_runtime->dxg_fd = -1;
  }
  delete dxg_runtime;
  dxg_runtime = new hsakmtRuntime();

}

static inline void init_page_size(void) {
  dxg_runtime->page_size = sysconf(_SC_PAGESIZE);
  dxg_runtime->page_shift = ffs(dxg_runtime->page_size) - 1;
}

static HSAKMT_STATUS init_vars_from_env(void) {
  char *envvar;
  int debug_level;

  /* Normally libraries don't print messages. For debugging purpose, we'll
   * print messages if an environment variable, HSAKMT_DEBUG_LEVEL, is set.
   */
  envvar = getenv("HSAKMT_DEBUG_LEVEL");
  if (envvar) {
    dxg_runtime->hsakmt_debug_level = atoi(envvar);
  }

  /* Check whether to support Zero frame buffer */
  envvar = getenv("HSA_ZFB");
  if (envvar)
    dxg_runtime->zfb_support = atoi(envvar);

  /* Check whether to handle vendor specific aql packet */
  envvar = getenv("WSLKMT_VENDOR_PACKET");
  if (envvar)
    dxg_runtime->vendor_packet_process = atoi(envvar);

  /* Decide whether to check available system memory before allocation */
  envvar = getenv("WSL_CHECK_AVAIL_SYSRAM");
  if (envvar)
    dxg_runtime->check_avail_sysram = !strcmp(envvar, "1");

  envvar = getenv("WSL_ENABLE_THUNK_SUB_ALLOCATOR");
  if (envvar)
    dxg_runtime->enable_thunk_sub_allocator = atoi(envvar);

  envvar = getenv("ROCR_VISIBLE_DEVICES");
  if (envvar) {
    std::string devices(envvar);
    size_t first_num_pos = devices.find_first_of("0123456789");
    if (first_num_pos != std::string::npos)
      dxg_runtime->default_node = std::stoi(devices.substr(first_num_pos)) + 1;
  }

  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtOpenKFD(void) {
  HSAKMT_STATUS result;
  int fd = -1;
  HsaSystemProperties sys_props;
  char *error;

  pthread_mutex_lock(&dxg_runtime->hsakmt_mutex);

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
      fd = open(dxg_runtime->dxg_device_name, O_RDWR | O_CLOEXEC);

      if (fd == -1) {
        result = HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED;
        goto open_failed;
      }

      dxg_runtime->dxg_fd = fd;
    }

    init_page_size();

    char *useSvmStr = getenv("HSA_USE_SVM");
    dxg_runtime->is_svm_api_supported = !(useSvmStr && !strcmp(useSvmStr, "0")) && false;

    // result = topology_sysfs_get_system_props(&sys_props);
    if (result != HSAKMT_STATUS_SUCCESS)
      goto topology_sysfs_failed;

    dxg_runtime->dxg_open_count = 1;

    if (!atfork_installed) {
      /* Atfork handlers cannot be uninstalled and
       * must be installed only once. Otherwise
       * prepare will deadlock when trying to take
       * the same lock multiple times.
       */
      pthread_atfork(prepare_fork_handler, parent_fork_handler,
                     child_fork_handler);
      atfork_installed = true;
    }
  } else {
    dxg_runtime->dxg_open_count++;
    result = HSAKMT_STATUS_KERNEL_ALREADY_OPENED;
  }

  reset_suballocator();
  pthread_mutex_unlock(&dxg_runtime->hsakmt_mutex);
  return result;
topology_sysfs_failed:
  close(fd);
open_failed:
  pthread_mutex_unlock(&dxg_runtime->hsakmt_mutex);

  return result;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtCloseKFD(void) {
  HSAKMT_STATUS result;

  pthread_mutex_lock(&dxg_runtime->hsakmt_mutex);

  if (dxg_runtime->dxg_open_count > 0) {
    if (--dxg_runtime->dxg_open_count == 0) {
      close(dxg_runtime->dxg_fd);
      dxg_runtime->dxg_fd = -1;
    }

    result = HSAKMT_STATUS_SUCCESS;
  } else
    result = HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED;

  pthread_mutex_unlock(&dxg_runtime->hsakmt_mutex);

  return result;
}
