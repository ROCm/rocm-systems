#include "shared/include/thunks.h"
#include "shared/include/device.h"
#include "shared/include/lda_chain.h"
#include "shared/include/thunk_proxy/thunk_proxy.h"
#include <memory>

using namespace std;

namespace wsl {
namespace thunk {

Device::Device(Platform *platform, LdaChain *lda_chain, u32 chainIndex,
               std::unique_ptr<thunk_proxy::DeviceContext> device_ctx)
    : device_ctx_(std::move(device_ctx)),
      lda_chain_(lda_chain),
      chain_index_(chainIndex) {
  (void)platform;
}

ErrorCode Device::Create(Platform *platform, LdaChain *ldaChain,
                         u32 deviceIndex, u32 chainIndex,
                         const thunk_proxy::DeviceInfo &deviceInfo,
                         Device **deviceOut) {
  (void)deviceIndex;
  (void)deviceInfo;

  auto dctx = std::unique_ptr<thunk_proxy::DeviceContext>(
      ldaChain->GetChainContext()->CreateDevice(
          ldaChain->DeviceHandle(), chainIndex));

  if (!dctx)
    return ErrorCode::InitializationFailed;

  if (!dctx->IsWddm2Supported())
    return ErrorCode::InitializationFailed;

  *deviceOut = new Device(platform, ldaChain, chainIndex, std::move(dctx));
  return ErrorCode::Success;
}

ErrorCode Device::QueryVramInfo(VramInfo *info) const {
  return device_ctx_->QueryVramInfo(info);
}

ErrorCode Device::QueryVramUsage(VramUsage *usage) const {
  return device_ctx_->QueryVramUsage(usage);
}

ErrorCode Device::QueryRasFeature(RasFeature *info) const {
  return device_ctx_->QueryRasFeature(info);
}

ErrorCode Device::QueryBdfInfo(BdfInfo *info) const {
  return device_ctx_->QueryBdfInfo(info);
}

ErrorCode Device::QueryAsicInfo(AsicInfo *info) const {
  return device_ctx_->QueryAsicInfo(info);
}

ErrorCode Device::QueryCacheInfo(CacheInfo *info) const {
  return device_ctx_->QueryCacheInfo(info);
}

ErrorCode Device::QueryVBiosInfo(VBiosInfo *info) const {
  return device_ctx_->QueryVBiosInfo(info);
}

ErrorCode Device::Init() {
  return device_ctx_->Init();
}

ErrorCode Device::QueryPowerInfo(PowerInfo *info) const {
  return device_ctx_->QueryPowerInfo(info);
}

ErrorCode Device::QueryTempMetric(uint32_t sensor_type, uint32_t metric,
                                  int64_t *temperature) const {
  return device_ctx_->QueryTempMetric(sensor_type, metric, temperature);
}

ErrorCode Device::QueryGpuActivity(GpuActivity *info) const {
  return device_ctx_->QueryGpuActivity(info);
}

ErrorCode Device::QueryFwInfo(FwInfo *info) const {
  return device_ctx_->QueryFwInfo(info);
}

ErrorCode Device::QueryClockInfo(uint32_t clk_type, ClockInfo *info) const {
  return device_ctx_->QueryClockInfo(clk_type, info);
}

ErrorCode Device::QueryPCIeInfo(PCIeInfo *info) const {
  return device_ctx_->QueryPCIeInfo(info);
}

ErrorCode Device::QueryBoardInfo(BoardInfo *info) const {
  return device_ctx_->QueryBoardInfo(info);
}

ErrorCode Device::QueryDriverInfo(DriverInfo *info) const {
  return device_ctx_->QueryDriverInfo(info);
}

ErrorCode Device::QueryMemoryTotal(uint32_t mem_type, uint64_t *total) const {
  return device_ctx_->QueryMemoryTotal(mem_type, total);
}

ErrorCode Device::QueryMemoryUsage(uint32_t mem_type, uint64_t *used) const {
  return device_ctx_->QueryMemoryUsage(mem_type, used);
}

ErrorCode Device::QueryGpuMetricsInfo(GpuMetricsInfo *info) const {
  return device_ctx_->QueryGpuMetricsInfo(info);
}

ErrorCode Device::Escape(void *pData, size_t dataSize, bool hardwareAccess) const {
  return device_ctx_->Escape(pData, dataSize, hardwareAccess);
}


ErrorCode Device::EnumGpuProcesses(
    std::vector<thunk_proxy::GpuProcessInfo> *out) const {
  return device_ctx_->EnumGpuProcesses(out);
}

} // namespace thunk
} // namespace wsl
