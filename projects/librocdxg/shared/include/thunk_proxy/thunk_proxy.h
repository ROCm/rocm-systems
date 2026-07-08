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

class DeviceContext;

uint32_t QueueEngine2EngineFlag(uint32_t queue_engine);
void SetAllocationInfo(void *data, uint64_t size, AllocDomain domain,
                      uint64_t addr, uint32_t mem_flags, uint32_t engine_flag,
                      const DeviceContext &device_ctx);

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

  // Issue QAI escapes for every GPU in the chain. On success:
  //   - NumChainedGpus() / VendorId() are updated from KMD data
  //   - per-GPU info caches are refreshed.
  ErrorCode QueryAdapterInfo();

  // Valid after QueryAdapterInfo() succeeds.
  WinAdapterHandle AdapterHandle()          const;
  uint32_t NumChainedGpus()         const;
  uint32_t VendorId(uint32_t index) const;

  // Create a DeviceContext for GPU slot chain_index.
  // DeviceInfo is derived on demand from cached per-slot adapter info.
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

  int EngineOrdinal(int engine) const;
  bool IsHwsEnabled(int engine) const;
  bool IsGpuTimeoutDisabled(int engine) const;

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

  int Major() const;
  bool IsDgpu() const;
  uint64_t LocalVisibleHeapSize() const;
  uint64_t LocalInvisibleHeapSize() const;
  uint32_t NumSdmaEngines() const;
  uint32_t SdmaEngine(uint32_t idx) const;
  uint32_t ComputeEngine() const;
  uint32_t NumCpQueues() const;
  uint32_t UserQueueSize() const;
  uint32_t MecFwVersion() const;
  uint32_t SdmaFwVersion() const;
  uint32_t L1CacheSize() const;
  uint32_t L2CacheSize() const;
  uint32_t L3CacheSize() const;
  uint32_t Gl2CacheLineSize() const;
  bool SupportStateShadowingByCpFw() const;
  bool SupportPlatformAtomic() const;

  // Device version and identification
  int Minor() const;
  int Stepping() const;
  const char *ProductName() const;
  uint64_t Uuid() const;
  uint32_t Family() const;
  uint32_t DeviceId() const;
  uint32_t Domain() const;
  uint32_t AsicRevision() const;
  uint32_t PciBusAddr() const;

  // Processor configuration
  uint32_t WavefrontSize() const;
  uint32_t ComputeUnitCount() const;
  uint32_t WavePerCu() const;
  uint32_t SimdPerCu() const;
  uint32_t MaxScratchSlotsPerCu() const;
  uint32_t NumShaderEngine() const;
  uint32_t ShaderArrayPerShaderEngine() const;
  uint32_t NumGws() const;
  uint32_t LdsSize() const;
  uint32_t WatchPointsNum() const;

  // Memory and frequency
  uint32_t MemoryBusWidth() const;
  uint32_t MaxMemoryClockMhz() const;
  uint32_t MaxEngineClockMhz() const;
  uint64_t GpuCounterFrequency() const;
  uint64_t NonLocalHeapSize() const;
  uint64_t PrivateApertureBase() const;
  uint64_t PrivateApertureSize() const;
  uint64_t SharedApertureBase() const;
  uint64_t SharedApertureSize() const;

  // Memory alignment configuration
  bool EnableBigPageAlignment() const;
  uint32_t BigPageAlignmentSize() const;
  uint32_t HwBigPageMinAlignmentSize() const;
  uint32_t HwBigPageAlignmentSize() const;

  WinDeviceHandle DeviceHandle() const;
  WinAdapterHandle AdapterHandle() const;
  LUID AdapterLuid() const;

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
