#ifndef SHARED_THUNK_PROXY_H
#define SHARED_THUNK_PROXY_H

#include <vector>
#include "d3dkmt_types.h"
#include "gpu_info.h"
#include "status.h"

namespace thunk_proxy {
enum AllocDomain {
  kSystem,
  kLocal,
  kUserMemory,
  kUserQueue,
  kDomainCount,
};

enum MemFlag {
  kFineGrain  = (1ULL << 0),
  kKernarg    = (1ULL << 1),
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
      uint32_t reserved          : 28;
    } hwsMask;
    uint32_t osHwsEnableFlags;
  };
  uint64_t engineOrdinalMask; // Indicates which engines (by ordinal) support MES HWS
};

typedef struct {
  // --- Identity ---
  uint32_t device_id;
  uint32_t family;
  uint32_t asic_revision;
  int major;
  int minor;
  int stepping;
  bool is_dgpu;
  char product_name[MAX_PATH];
  uint64_t uuid;

  // --- Shader / compute topology ---
  uint32_t wavefront_size;
  uint32_t compute_unit_count;
  uint32_t num_shader_engine;
  uint32_t shader_array_per_shader_engine;
  uint32_t simd_per_cu;
  uint32_t wave_per_cu;
  uint32_t max_scratch_slots_per_cu;
  uint32_t watch_points_num;
  uint32_t num_cp_queues;
  uint32_t user_queue_size;
  uint32_t lds_size;
  uint32_t domain;
  uint32_t num_gws;

  // --- Clock / performance ---
  uint32_t max_engine_clock_mhz;
  uint32_t max_memory_clock_mhz;
  uint64_t gpu_counter_frequency;

  // --- Cache sizes ---
  uint32_t l1_cache_size;
  uint32_t l2_cache_size;
  uint32_t l3_cache_size;
  uint32_t gl2_cacheline_size;
  uint32_t memory_bus_width;

  // --- Memory heaps ---
  uint64_t local_visible_heap_size;
  uint64_t local_invisible_heap_size;
  uint64_t non_local_heap_size;

  // --- Virtual address apertures ---
  uint64_t private_aperture_base;
  uint64_t private_aperture_size;
  uint64_t shared_aperture_base;
  uint64_t shared_aperture_size;
  uint32_t pci_bus_addr;

  // --- Big page alignment ---
  bool enable_big_page_alignment;
  uint32_t big_page_alignment_size;
  uint32_t hw_big_page_min_alignment_size;
  uint32_t hw_big_page_alignment_size;

  // --- Firmware versions ---
  uint32_t mec_fw_version;
  uint32_t sdma_fw_version;

  // --- Scheduler / HWS ---
  HwsInfo hwsInfo;
  std::vector<int> sdma_schedid;
  uint32_t compute_schedid;
  bool state_shadowing_by_cpfw;

  // --- Misc capabilities ---
  bool platform_atomic_support;

  // --- KMD adapter blob (opaque, managed by ParseAdapterInfo/DestroyDeviceInfo) ---
  void *adapter_info;
  uint32_t kmd_version;

  int EngineOrdinal(int engine) const;
  bool IsHwsEnabled(int engine) const;
  bool IsGpuTimeoutDisabled(int engine) const;
} DeviceInfo;

bool ParseAdapterInfo(D3DKMT_HANDLE adapter, DeviceInfo *device_info);
bool QueryAdapterSupported(unsigned int device_id);

uint32_t QueueEngine2EngineFlag(uint32_t queue_engine);
void SetAllocationInfo(void *data, uint64_t size, AllocDomain domain,
                      uint64_t addr, uint32_t mem_flags, uint32_t engine_flag, const DeviceInfo &device_info);

struct PrivData {
  std::vector<uint8_t> buf;
  int size() const { return static_cast<int>(buf.size()); }
  void *data() { return buf.data(); }
};

struct AllocPrivData {
  std::vector<uint8_t> buf;
  int drv_data_size;
  int per_alloc_size;
  void *drv_data() { return buf.data(); }
  void *alloc_data_at(int i) {
    return buf.data() + drv_data_size + i * per_alloc_size;
  }
};

PrivData MakePowerOptPrivData(bool restore);
PrivData MakeContextPrivData(bool fw_managed_gfx_state);
PrivData MakeSubmitPrivData(D3DKMT_HANDLE queue, uint64_t command_addr,
                            uint64_t command_size, bool is_hw_queue);
PrivData MakeHwQueuePrivData(bool fw_managed_gfx_state, SchedLevel level = kNormal);
AllocPrivData MakeAllocPrivData(int num_allocations);


class ChainContext {
public:
  // Construct with adapter identity only; topology is resolved by
  // QueryAdapterInfo() once a D3DKMT device handle is available.
  static ChainContext *Create(WinAdapterHandle adapter_handle,
                              LUID     luid);

  ~ChainContext();

  // Issue QAI escapes for every GPU in the chain.  On success:
  //   - NumChainedGpus() / VendorId() are updated from KMD data
  //   - out_infos is resized to one DeviceInfo per GPU slot
  ErrorCode QueryAdapterInfo(WinAdapterHandle device_handle,
                             std::vector<DeviceInfo> &out_infos);

  // Valid after QueryAdapterInfo() succeeds.
  WinAdapterHandle AdapterHandle()          const;
  uint32_t NumChainedGpus()         const;
  uint32_t VendorId(uint32_t index) const;

  // Create a DeviceContext for GPU slot chain_index.
  // Must be called after QueryAdapterInfo().
  class DeviceContext *CreateDevice(WinDeviceHandle device_handle,
                                    uint32_t chain_index) const;

  ChainContext(const ChainContext &) = delete;
  ChainContext &operator=(const ChainContext &) = delete;

private:
  ChainContext() = default;
  struct Impl;
  Impl *impl_ = nullptr;
};

class DeviceContext {
public:
  ~DeviceContext();

  // Returns false if the GPU slot does not meet the WDDM2 baseline.
  bool IsWddm2Supported() const;

  ErrorCode QueryVramInfo(wsl::thunk::VramInfo *info) const;

  // Query current VRAM usage in MB via KMD escape.
  ErrorCode QueryVramUsage(wsl::thunk::VramUsage *usage) const;

  // Query static ASIC information from KMD adapter info.
  ErrorCode QueryAsicInfo(wsl::thunk::AsicInfo *info) const;

  // Query RAS feature flags via KMD escape.
  ErrorCode QueryRasFeature(wsl::thunk::RasFeature *info) const;

  // Send a driver-escape packet on behalf of this GPU slot.
  ErrorCode Escape(void *pData, size_t dataSize, bool hardwareAccess = false) const;

  DeviceContext(const DeviceContext &) = delete;
  DeviceContext &operator=(const DeviceContext &) = delete;

private:
  friend class ChainContext;
  DeviceContext() = default;
  struct Impl;
  Impl *impl_ = nullptr;
};

}
#endif
