#ifndef _WSL_SHARED_INC_DEVICE_H_
#define _WSL_SHARED_INC_DEVICE_H_

#include <memory>
#include <vector>

#include "shared/include/d3dkmt_types.h"
#include "shared/include/gpu_info.h"
#include "shared/include/status.h"
#include "shared/include/thunk_proxy/thunk_proxy.h"

namespace wsl {
namespace thunk {

constexpr gpusize PageSize = 0x1000u;

class Platform;
class LdaChain;

class Device {
public:
  static ErrorCode Create(Platform *platform, LdaChain *ldaChain,
                          u32 deviceIndex, u32 chainIndex,
                          const thunk_proxy::DeviceInfo &deviceInfo,
                          Device **deviceOut);

  LdaChain *GetLdaChain()   const { return lda_chain_; }
  u32       GetChainIndex() const { return chain_index_; }

  ErrorCode QueryVramUsage(VramUsage *usage) const;

  ErrorCode QueryVramInfo(VramInfo *info) const;

  ErrorCode QueryRasFeature(RasFeature *info) const;

  ErrorCode QueryBdfInfo(BdfInfo *info) const;

  ErrorCode QueryAsicInfo(AsicInfo *info) const;

  ErrorCode QueryCacheInfo(CacheInfo *info) const;

  ErrorCode QueryVBiosInfo(VBiosInfo *info) const;

  // Called once after creation to pre-fetch sensor limits.
  ErrorCode Init();

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

private:
  Device(Platform *platform, LdaChain *lda_chain, u32 chainIndex,
         std::unique_ptr<thunk_proxy::DeviceContext> device_ctx);

  std::unique_ptr<thunk_proxy::DeviceContext> device_ctx_;
  LdaChain *const lda_chain_ = nullptr;
  const u32       chain_index_ = 0;
};

} // namespace thunk
} // namespace wsl

#endif
