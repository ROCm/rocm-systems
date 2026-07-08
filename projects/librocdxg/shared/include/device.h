#ifndef _WSL_SHARED_INC_DEVICE_H_
#define _WSL_SHARED_INC_DEVICE_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "shared/include/d3dkmt_types.h"
#include "shared/include/gpu_info.h"
#include "shared/include/status.h"
#include "shared/include/thunk_proxy/thunk_proxy.h"

namespace wsl {
namespace thunk {

constexpr gpusize PageSize = 0x1000u;

// WDDM memory segment buckets used for VRAM availability accounting.
enum class VramSegmentKind : uint8_t {
  kFb,        // local visible (framebuffer)
  kInvFb,     // local invisible
  kLocal,     // kFb + kInvFb (local heap)
  kNonLocal,  // shared system memory segment
};

struct VramSegmentIds {
  uint32_t fb = 0;
  uint32_t inv_fb = 1;
  uint32_t non_local = 3;
};

class Platform;
class LdaChain;

class Device {
public:
  static ErrorCode Create(Platform *platform, LdaChain *ldaChain,
                          u32 deviceIndex, u32 chainIndex, Device **deviceOut);

  LdaChain *GetLdaChain()   const { return lda_chain_; }
  u32       GetChainIndex() const { return chain_index_; }

  ErrorCode QueryVramUsage(VramUsage *usage) const;

  ErrorCode QueryVramInfo(VramInfo *info) const;

  ErrorCode QueryRasFeature(RasFeature *info) const;

  ErrorCode QueryBdfInfo(BdfInfo *info) const;

  ErrorCode QueryAsicInfo(AsicInfo *info) const;

  ErrorCode QueryCacheInfo(CacheInfo *info) const;

  ErrorCode QueryVBiosInfo(VBiosInfo *info) const;

  ErrorCode Init();

  // Query resident bytes for a VRAM segment bucket (see VramSegmentKind).
  ErrorCode QueryVramSegmentUsage(VramSegmentKind kind, uint64_t *usage) const;

  // Total VRAM reported to upper stacks.
  uint64_t VramTotal() const;

  // WDDM segment usage for VramTotal() buckets.
  ErrorCode QueryVramUsage(uint64_t *usage_bytes) const;

  ErrorCode QueryPowerInfo(PowerInfo *info) const;

  ErrorCode QueryTempMetric(uint32_t sensor_type, uint32_t metric,
                            int64_t *temperature) const;

  ErrorCode QueryGpuActivity(GpuActivity *info) const;

  ErrorCode QueryFwInfo(FwInfo *info) const;

  ErrorCode QueryClockInfo(uint32_t clk_type, ClockInfo *info) const;

  ErrorCode QueryPCIeInfo(PCIeInfo *info) const;

  ErrorCode QueryBoardInfo(BoardInfo *info) const;

  ErrorCode QueryDriverInfo(DriverInfo *info) const;

  ErrorCode QueryMemoryTotal(uint32_t mem_type, uint64_t *total) const;
  ErrorCode QueryMemoryUsage(uint32_t mem_type, uint64_t *used) const;

  // Enumerate all GPU processes visible on this adapter.
  ErrorCode EnumGpuProcesses(
      std::vector<thunk_proxy::GpuProcessInfo> *out) const;

  // Query live GPU metrics (clocks, temps, voltages, activity, fan) via PMLog.
  ErrorCode QueryGpuMetricsInfo(GpuMetricsInfo *info) const;

  ErrorCode Escape(void *pData, size_t dataSize,
                   bool hardwareAccess = false) const;

  WinDeviceHandle DeviceHandle() const;
  WinAdapterHandle AdapterHandle() const;
  LUID AdapterLuid() const;
  int EngineOrdinal(int engine) const;
  bool IsHwsEnabled(int engine) const;
  bool IsGpuTimeoutDisabled(int engine) const;

  int Major() const;
  int Minor() const;
  int Stepping() const;
  bool IsDgpu() const;
  const char *ProductName() const;
  uint64_t Uuid() const;
  uint32_t Family() const;
  uint32_t DeviceId() const;
  uint32_t WavefrontSize() const;
  uint32_t ComputeUnitCount() const;
  uint32_t MaxEngineClockMhz() const;
  uint32_t WatchPointsNum() const;
  uint32_t PciBusAddr() const;
  uint32_t MemoryBusWidth() const;
  uint32_t MaxMemoryClockMhz() const;
  uint32_t WavePerCu() const;
  uint32_t SimdPerCu() const;
  uint32_t MaxScratchSlotsPerCu() const;
  uint32_t NumShaderEngine() const;
  uint32_t ShaderArrayPerShaderEngine() const;
  uint32_t NumSdmaEngines() const;
  uint32_t SdmaEngine(uint32_t idx) const;
  uint32_t Domain() const;
  uint32_t NumGws() const;
  uint32_t AsicRevision() const;
  uint64_t LocalVisibleHeapSize() const;
  uint64_t LocalInvisibleHeapSize() const;
  uint64_t NonLocalHeapSize() const;
  uint64_t PrivateApertureBase() const;
  uint64_t PrivateApertureSize() const;
  uint64_t SharedApertureBase() const;
  uint64_t SharedApertureSize() const;
  uint32_t LdsSize() const;
  uint64_t GpuCounterFrequency() const;
  uint32_t UserQueueSize() const;
  uint32_t MecFwVersion() const;
  uint32_t SdmaFwVersion() const;
  uint32_t L1CacheSize() const;
  uint32_t L2CacheSize() const;
  uint32_t L3CacheSize() const;
  uint32_t Gl2CacheLineSize() const;
  bool SupportStateShadowingByCpFw() const;
  bool SupportPlatformAtomic() const;
  uint32_t ComputeEngine() const;
  uint32_t NumCpQueues() const;
  bool EnableBigPageAlignment() const;
  uint32_t BigPageAlignmentSize() const;
  uint32_t HwBigPageMinAlignmentSize() const;
  uint32_t HwBigPageAlignmentSize() const;

  void SetAllocationInfo(void *data, uint64_t size,
                         thunk_proxy::AllocDomain domain, uint64_t addr,
                         uint32_t mem_flags, uint32_t engine_flag) const;

private:
  Device(Platform *platform, LdaChain *lda_chain, u32 chainIndex,
         std::unique_ptr<thunk_proxy::DeviceContext> device_ctx);

  ErrorCode InitSegmentIds();

  std::unique_ptr<thunk_proxy::DeviceContext> device_ctx_;
  LdaChain *const lda_chain_ = nullptr;
  const u32       chain_index_ = 0;
  VramSegmentIds segment_ids_{};
};

} // namespace thunk
} // namespace wsl

#endif
