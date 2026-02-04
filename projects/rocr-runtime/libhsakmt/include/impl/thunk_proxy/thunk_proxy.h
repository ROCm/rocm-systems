/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <vector>

namespace thunk_proxy {
enum AllocDomain {
  kSystem,
  kLocal,
  kUserMemory,
  kUserQueue,
  kDomainCount,
};

enum MemFlag {
  kFineGrain    = (1ULL << 0),
  kKernarg      = (1ULL << 1),
  kQueueObject  = (1ULL << 2),
};

enum EngineFlag {
  KCOMPUTE0   = (1ULL << 0),
  KDRMDMA     = (1ULL << 1),
  KDRMDMA1    = (1ULL << 2),
};

enum SchedLevel {
  kLow = 0,
  kNormal = 1,
  kHigh = 2,
};

struct HwsInfo {
  union {
    struct {
      uint32_t gfxHwsEnabled     : 1;
      uint32_t computeHwsEnabled : 1;
      uint32_t dmaHwsEnabled     : 1;
      uint32_t dma1HwsEnabled    : 1;
      uint32_t aql_queue         : 1;  //!< Kernel mode driver supports native AQL queue
      uint32_t reserved : 27;
    } hwsMask;
    uint32_t osHwsEnableFlags;
  };
  uint64_t engineOrdinalMask;  // Indicates which engines (by ordinal) support MES HWS
};

struct DeviceInfo {
  int major;
  int minor;
  int stepping;
  bool is_dgpu;
  char product_name[MAX_PATH];
  uint64_t uuid;
  uint32_t family;
  uint32_t device_id;
  uint32_t wavefront_size;
  uint32_t compute_unit_count;
  uint32_t max_engine_clock_mhz;
  uint32_t watch_points_num;
  uint32_t pci_bus_addr;
  uint32_t memory_bus_width;
  uint32_t max_memory_clock_mhz;
  uint64_t gpu_counter_frequency;
  uint32_t wave_per_cu;
  uint32_t simd_per_cu;
  uint32_t max_scratch_slots_per_cu;
  uint32_t num_shader_engine;
  uint32_t shader_array_per_shader_engine;
  uint32_t domain;
  uint32_t num_gws;
  uint32_t asic_revision;
  uint64_t local_visible_heap_size;
  uint64_t local_invisible_heap_size;
  uint64_t non_local_heap_size;
  uint64_t private_aperture_base;
  uint64_t private_aperture_size;
  uint64_t shared_aperture_base;
  uint64_t shared_aperture_size;
  uint32_t user_queue_size;
  uint32_t lds_size;
  uint32_t big_page_alignment_size;
  uint32_t hw_big_page_min_alignment_size;
  uint32_t hw_big_page_alignment_size;
  bool enable_big_page_alignment;
  uint32_t mec_fw_version;
  uint32_t sdma_fw_version;
  uint32_t l1_cache_size;
  uint32_t l2_cache_size;
  uint32_t l3_cache_size;
  uint32_t gl2_cacheline_size;
  uint32_t num_cp_queues;
  HwsInfo hwsInfo;
  std::vector<int> sdma_schedid;
  uint32_t compute_schedid;
  bool state_shadowing_by_cpfw;
  bool platform_atomic_support;
  void* adapter_info;
  uint32_t kmd_version;
  uint32_t gb_addr_config;  //!< Graphics block address configuration
  uint32_t num_xcc;         //!< Number of XCC units
};

int EngineOrdinal(int engine, DeviceInfo* device_info);
bool GetHwsEnabled(int engine, DeviceInfo* device_info);
bool ShouldDisableGpuTimeout(int engine, DeviceInfo* device_info);
bool QueryAdapterSupported(unsigned int device_id);

uint32_t QueueEngine2EngineFlag(uint32_t queue_engine);
void SetAllocationInfo(void* data, uint64_t size, AllocDomain domain, uint64_t addr,
                       uint32_t mem_flags, uint32_t engine_flag, const DeviceInfo& device_info);
void GetAllocPrivDataSize(int* priv_drv_data_size, int* priv_alloc_data_size);
void FillinAllocPrivDrvData(void* drv_priv, int priv_alloc_data_size);

int GetSubmitPrivDataSize();
void FillinSubmitPrivData(void* priv_data, D3DKMT_HANDLE queue, uint64_t command_addr,
                          uint64_t command_size, bool is_hw_queue);
int GetHwQueuePrivDataSize();
#if defined(__linux__)
void FillinHwQueuePrivData(void *priv_data, bool FwManagedGfxState, SchedLevel level = kNormal);
#else
void FillinHwQueuePrivData(void* priv_data, bool FwManagedGfxState, SchedLevel level = kNormal,
                           bool aql = false, uint64_t queue_va = 0, uint64_t queue_size = 0,
                           uint64_t wptr = 0, uint64_t rptr = 0, D3DKMT_HANDLE aql_queue_desc = 0);
#endif
int GetContextPrivDataSize();

int GetPowerOptPrivDataSize();
void FillinPowerOptPrivData(void* priv_data, bool restore);

#if defined(__linux__)
void FillinContextPrivData(void *priv_data, bool FwManagedGfxState);
bool ParseAdapterInfo(D3DKMT_HANDLE adapter, DeviceInfo* device_info);
#else
NTSTATUS ParseAdapterInfo(D3DKMT_HANDLE adapter, DeviceInfo* device_info);
void FillinContextPrivData(void* priv_data, bool FwManagedGfxState, uint32_t schedId = 0);

// AQL queue submit interfaces
int GetAqlSubmitPrivDataSize();
void FillinAqlSubmitPrivData(void* priv_data, uint64_t doorbell_value);

// User mode event interfaces
int GetRegisterEventPrivDataSize();
void FillinRegisterEventPrivData(void* priv_data, uint64_t handle, uint32_t event_id);
uint64_t GetRegisterEventMailbox(void* priv_data);
int GetUnregisterEventPrivDataSize();
void FillinUnregisterEventPrivData(void* priv_data, uint64_t handle);

// Interop memory interface to get allocation size
uint64_t GetMemoryAllocationSize(const void* priv_data);

// Get the size of the proxy resource info structure
size_t GetProxyResourceInfoSize();
#endif
}

