#ifndef SHARED_THUNK_PROXY_H
#define SHARED_THUNK_PROXY_H

#include <vector>
#include "shared/include/d3dkmt_types.h"
#include "shared/include/gpu_info.h"
#include "shared/include/status.h"

namespace thunk_proxy {

/**
 * @brief Per-process GPU usage information returned by EnumGpuProcesses.
 *
 * On WSL2, win_pid is the Windows-namespace PID returned by
 * D3DKMTEnumProcesses (which equals the Linux PID in a flat namespace).
 * vram_usage_bytes is populated via D3DKMTQueryVideoMemoryInfo
 * (LOCAL segment group); 0 when the query is unsupported or the process
 * has no current VRAM allocation.
 */
struct GpuProcessInfo {
  uint32_t win_pid;           ///< Windows PID of the GPU-using process
  uint64_t vram_usage_bytes;  ///< Current VRAM (LOCAL) usage in bytes
};

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

  // Query PCI BDF location from KMD adapter info.
  ErrorCode QueryBdfInfo(wsl::thunk::BdfInfo *info) const;

  // Query static ASIC information from KMD adapter info.
  ErrorCode QueryAsicInfo(wsl::thunk::AsicInfo *info) const;

  // Query cache sizes from KMD adapter info.
  ErrorCode QueryCacheInfo(wsl::thunk::CacheInfo *info) const;

  // Query RAS feature flags via KMD escape.
  ErrorCode QueryRasFeature(wsl::thunk::RasFeature *info) const;

  // Query VBIOS info via KMD CWDDE escape.
  ErrorCode QueryVBiosInfo(wsl::thunk::VBiosInfo *info) const;

  // Fetch sensor limits from KMD (call once after device creation).
  ErrorCode Init();

  // Query power / voltage readings via KMD PMLog escape.
  ErrorCode QueryPowerInfo(wsl::thunk::PowerInfo *info) const;

  // Query temperature metric via KMD PMLog escape.
  ErrorCode QueryTempMetric(uint32_t sensor_type, uint32_t metric,
                            int64_t *temperature) const;

  // Query GPU engine activity via KMD PMLog escape.
  ErrorCode QueryGpuActivity(wsl::thunk::GpuActivity *info) const;

  // Query firmware versions from KMD adapter info.
  ErrorCode QueryFwInfo(wsl::thunk::FwInfo *info) const;

  // Query clock frequency via KMD PMLog escape.
  ErrorCode QueryClockInfo(uint32_t clk_type, wsl::thunk::ClockInfo *info) const;

  ErrorCode QueryPCIeInfo(wsl::thunk::PCIeInfo *info) const;

  // Query board identification info from KMD adapter registry.
  ErrorCode QueryBoardInfo(wsl::thunk::BoardInfo *info) const;

  // Query driver version, name, and date from Windows registry keys.
  ErrorCode QueryDriverInfo(wsl::thunk::DriverInfo *info) const;

  // Query memory total size in bytes by type (0=VRAM, 1=VIS_VRAM, 2=GTT).
  ErrorCode QueryMemoryTotal(uint32_t mem_type, uint64_t *total) const;

  // Query memory used in bytes by type via D3DKMTQueryStatistics.
  ErrorCode QueryMemoryUsage(uint32_t mem_type, uint64_t *used) const;

  // Enumerate all Windows processes currently using this adapter's GPU.
  // Calls D3DKMTEnumProcesses (WSL2 dxgkrnl-specific API) and populates
  // |out| with one GpuProcessInfo per process.  Also queries VRAM usage
  // for each process via D3DKMTQueryVideoMemoryInfo.
  // Returns ErrorCode::Unsupported when the WSL2 API is unavailable.
  ErrorCode EnumGpuProcesses(std::vector<GpuProcessInfo> *out) const;

  // Query VRAM usage (LOCAL segment group) for a specific Windows PID.
  // |win_pid| == 0 queries the calling process itself.
  // Returns ErrorCode::Unsupported when the WSL2 API is unavailable.
  ErrorCode QueryProcessVram(uint32_t win_pid, uint64_t *vram_bytes) const;

  // Query live GPU metrics (clocks, temps, voltages, activity, fan) via PMLog escape.
  ErrorCode QueryGpuMetricsInfo(wsl::thunk::GpuMetricsInfo *info) const;

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
