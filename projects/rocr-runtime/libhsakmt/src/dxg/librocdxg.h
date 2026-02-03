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

#pragma once

#if defined(__linux__)
#include <pthread.h>
#else
#define WIN32_NO_STATUS
#include <Windows.h>
#undef WIN32_NO_STATUS
#include <d3dkmthk.h>
#include "hsakmt/drm/amdgpu.h"
#endif
#include <stdint.h>
#include <limits.h>
#include "hsakmt/hsakmt.h"
#if defined(__linux__)
#include "hsakmt/hsakmt_drm.h"
#endif
#include "impl/wddm/va_mgr.h"
#include "impl/wddm/types.h"
#include "impl/wddm/device.h"
#include "dxcore_loader.h"

wsl::thunk::WDDMDevice* get_wddmdev(uint32_t node_id);
wsl::thunk::WDDMDevice* WddmDevice(uint32_t dev_id);
int CpuNodes();

uint32_t get_num_wddmdev();
wsl::thunk::GpuMemory *get_gpu_mem(void *MemoryAddress);

#define HSAKMT_DEBUG_LEVEL_ERR      -1
#define HSAKMT_DEBUG_LEVEL_DEFAULT  3
#define HSAKMT_DEBUG_LEVEL_WARNING  4
#define HSAKMT_DEBUG_LEVEL_INFO     6
#define HSAKMT_DEBUG_LEVEL_DEBUG    7

struct hsakmtRuntime {
  hsakmtRuntime()
    : dxg_fd(-1),
    parent_pid(getpid()),
    hsakmt_debug_level(HSAKMT_DEBUG_LEVEL_DEFAULT),
    dxg_open_count(0),
    max_single_alloc_size(0),
    local_heap_space_start_(0),
    local_heap_space_size_(0),
    system_heap_space_start_(0),
    system_heap_space_size_(0),
    handle_aperture_start_(0),
    handle_aperture_size_(0),
    default_node(1) {}

  void HeapInit();
  void HeapFini();
  bool ReserveSvmSpace(uint64_t &base, uint64_t &size, uint64_t align);
  bool FreeSvmSpace(uint64_t &base, uint64_t &size);
  bool ReserveLocalHeapSpace();
  bool FreeLocalHeapSpace();
  void InitLocalHeapMgr();
  bool ReserveSystemHeapSpace();
  uint64_t SystemHeapSize() { return system_heap_space_size_; }
  bool FreeSystemHeapSpace();
  bool CommitSystemHeapSpace(void* addr, int64_t size, bool lock);
  bool DecommitSystemHeapSpace(void* addr, int64_t size);
  void InitSystemHeapMgr();
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
  void InitHandleApertureMgr();
  ErrorCode HandleApertureAlloc(gpusize size, gpusize *out_gpu_virt_addr);
  void HandleApertureFree(gpusize gpu_addr);

  std::recursive_mutex hsakmt_mutex;
  const char *dxg_device_name = "/dev/dxg";
  long page_size;
  int page_shift;
  int dxg_fd = -1;
  int parent_pid = -1;
  int hsakmt_debug_level = HSAKMT_DEBUG_LEVEL_DEFAULT;
  int hsakmt_debug_sysmem = 0;
  unsigned long dxg_open_count;

  size_t max_single_alloc_size;
  uint32_t default_node;

  /* local heap means bo's backend is vram of all GPUs */
  uint64_t local_heap_space_start_;
  uint64_t local_heap_space_size_;
  /* manage the reserved local heap space which shared by CPU and GPUs */
  std::unique_ptr<wsl::thunk::VaMgr> local_heap_mgr_;

  /* system heap means bo's backend is system ram */
  uint64_t system_heap_space_start_;
  uint64_t system_heap_space_size_;
  /* manage the reserved system heap space which shared by CPU and GPUs */
  std::unique_ptr<wsl::thunk::VaMgr> system_heap_mgr_;

  uint64_t handle_aperture_start_;
  uint64_t handle_aperture_size_;
  std::unique_ptr<wsl::thunk::VaMgr> handle_aperture_mgr_;
  union {
    struct {
      uint64_t use_pm4_ : 1;
      uint64_t is_forked : 1;
      uint64_t hsakmt_is_dgpu : 1;
      uint64_t check_avail_sysram : 1;
      uint64_t zfb_support : 1;
      uint64_t vendor_packet_process : 1;
      uint64_t enable_thunk_sub_allocator : 1;
      uint64_t is_svm_api_supported : 1;
      uint64_t disable_wait_timeout_ : 1;
    };
    uint64_t settings_bits_ = 0;
  };
};

extern hsakmtRuntime *dxg_runtime;

#undef HSAKMTAPI
#if defined(__linux__)
#define HSAKMTAPI __attribute__((visibility ("default")))
#else
#define HSAKMTAPI
#endif

#if defined(__clang__)
#if __has_feature(address_sanitizer)
#define SANITIZER_AMDGPU 1
#endif
#endif

/*Avoid pointer-to-int-cast warning*/
#define PORT_VPTR_TO_UINT64(vptr) ((uint64_t)(unsigned long)(vptr))

/*Avoid int-to-pointer-cast warning*/
#define PORT_UINT64_TO_VPTR(v) ((void*)(unsigned long)(v))

#define CHECK_DXG_OPEN() \
	do { if (dxg_runtime->dxg_open_count == 0 || dxg_runtime->is_forked) return HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED; } while (0)

/* 64KB BigK fragment size for TLB efficiency */
#define GPU_BIGK_PAGE_SIZE (1 << 16)

/* 2MB huge page size for 4-level page tables on Vega10 and later GPUs */
#define GPU_HUGE_PAGE_SIZE (2 << 20)

#define CHECK_PAGE_MULTIPLE(x) \
	do { if ((uint64_t)PORT_VPTR_TO_UINT64(x) % dxg_runtime->page_size) return HSAKMT_STATUS_INVALID_PARAMETER; } while(0)

#define ALIGN_UP(x,align) (((uint64_t)(x) + (align) - 1) & ~(uint64_t)((align)-1))
#define ALIGN_UP_32(x,align) (((uint32_t)(x) + (align) - 1) & ~(uint32_t)((align)-1))
#define PAGE_ALIGN_UP(x) ALIGN_UP(x,dxg_runtime->page_size)
#define BITMASK(n) ((n) ? (UINT64_MAX >> (sizeof(UINT64_MAX) * CHAR_BIT - (n))) : 0)
#define ARRAY_LEN(array) (sizeof(array) / sizeof(array[0]))

/* HSA Thunk logging usage */
#define get_thread_id()                                                                                                          \
    ([]() -> std::string {                                                                                                       \
        std::stringstream str_thrd_id;                                                                                           \
        str_thrd_id << std::hex << std::this_thread::get_id();                                                                   \
        return str_thrd_id.str();                                                                                                \
    })()
#define hsakmt_print_common(stream, fmt, ...)                                                                                    \
    do {                                                                                                                         \
        fprintf(stream, "pid:%d tid:0x%s [%s] " fmt, getpid(), get_thread_id().c_str(), __FUNCTION__, ##__VA_ARGS__);            \
        fflush(stream);                                                                                                          \
    } while (false)
#ifdef NDEBUG
#define hsakmt_print(level, fmt, ...)                                                                                            \
    do { } while (false)
#else
#define hsakmt_print(level, fmt, ...)                                                                                            \
    do {                                                                                                                         \
        if (level <= dxg_runtime->hsakmt_debug_level) {                                                                          \
            hsakmt_print_common(stdout, fmt, ##__VA_ARGS__);                                                                     \
        }                                                                                                                        \
    } while (false)
#endif

#define pr_err(fmt, ...) \
	hsakmt_print_common(stderr, fmt, ##__VA_ARGS__)
#define pr_warn(fmt, ...) \
	hsakmt_print(HSAKMT_DEBUG_LEVEL_WARNING, fmt, ##__VA_ARGS__)
#define pr_info(fmt, ...) \
	hsakmt_print(HSAKMT_DEBUG_LEVEL_INFO, fmt, ##__VA_ARGS__)
#define pr_debug(fmt, ...) \
	hsakmt_print(HSAKMT_DEBUG_LEVEL_DEBUG, fmt, ##__VA_ARGS__)
#define pr_err_once(fmt, ...)                   \
{                                               \
        static bool __print_once;               \
        if (!__print_once) {                    \
                __print_once = true;            \
                pr_err(fmt, ##__VA_ARGS__);     \
        }                                       \
}
#define pr_warn_once(fmt, ...)                  \
{                                               \
        static bool __print_once;               \
        if (!__print_once) {                    \
                __print_once = true;            \
                pr_warn(fmt, ##__VA_ARGS__);    \
        }                                       \
}

/* Expects HSA_ENGINE_ID.ui32, returns gfxv (full) in hex */
#define HSA_GET_GFX_VERSION_FULL(ui32) \
	(((ui32.Major) << 16) | ((ui32.Minor) << 8) | (ui32.Stepping))

HSAKMT_STATUS validate_nodeid(uint32_t nodeid, uint32_t *gpu_id);
HSAKMT_STATUS gpuid_to_nodeid(uint32_t gpu_id, uint32_t* node_id);
bool prefer_ats(HSAuint32 node_id);
uint16_t get_device_id_by_node_id(HSAuint32 node_id);
uint16_t get_device_id_by_gpu_id(HSAuint32 gpu_id);
uint32_t get_direct_link_cpu(uint32_t gpu_node);

HSAKMT_STATUS topology_sysfs_get_system_props(HsaSystemProperties& props);
HSAKMT_STATUS topology_get_node_props(HSAuint32 NodeId,
				      HsaNodeProperties *NodeProperties);
HSAKMT_STATUS topology_get_iolink_props(HSAuint32 NodeId,
					HSAuint32 NumIoLinks,
					HsaIoLinkProperties *IoLinkProperties);
void topology_setup_is_dgpu_param(HsaNodeProperties *props);

HSAuint32 PageSizeFromFlags(unsigned int pageSizeFlags);

uint32_t get_num_sysfs_nodes(void);

bool is_forked_child(void);

void clear_allocation_map(void);

class BlockAllocator {
private:
    static const size_t block_size_ = 128 * 1024 * 1024;  // 128MB blocks.

public:
    void* alloc(size_t request_size, size_t& allocated_size) const;
    void free(void* ptr, size_t length) const;
    size_t block_size() const { return block_size_; }
};

void reset_suballocator(void);
void trim_suballocator(void);

HSAKMT_STATUS hsaKmtAllocMemoryAlignInternal(HSAuint32 PreferredNode,
                                            HSAuint64 SizeInBytes,
                                            HSAuint64 Alignment,
                                            HsaMemFlags MemFlags,
                                            void **MemoryAddress,
                                            bool SkipSubAlloc = false);

HSAKMT_STATUS hsaKmtFreeMemoryInternal(void *MemoryAddress,
                                    HSAuint64 SizeInBytes,
                                    bool SkipSubAlloc = false);

bool queue_acquire_buffer(void *MemoryAddress);
bool queue_release_buffer(void *MemoryAddress);
/* Calculate VGPR and SGPR register file size per CU */
uint32_t get_vgpr_size_per_cu(HSA_ENGINE_ID id);
#define SGPR_SIZE_PER_CU 0x4000

bool is_ipc_sysmemfd(uint64_t fd);

HSAKMT_STATUS import_dmabuf_fd(uint64_t DMABufFd, uint32_t NodeId, bool alloc_va, bool is_ipc_memfd,
                               wsl::thunk::GpuMemoryHandle* GpuMemHandle, bool is_kmt_handle);

bool hsakmt_hsa_loader_init();

