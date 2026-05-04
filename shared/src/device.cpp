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
                         u32 deviceIndex, u32 chainIndex, Device **deviceOut) {
  (void)deviceIndex;
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

WinDeviceHandle Device::DeviceHandle() const {
  return device_ctx_->DeviceHandle();
}

WinAdapterHandle Device::AdapterHandle() const {
  return device_ctx_->AdapterHandle();
}

LUID Device::AdapterLuid() const {
  return device_ctx_->AdapterLuid();
}

int Device::EngineOrdinal(int engine) const {
  return device_ctx_->EngineOrdinal(engine);
}

bool Device::IsHwsEnabled(int engine) const {
  return device_ctx_->IsHwsEnabled(engine);
}

bool Device::IsGpuTimeoutDisabled(int engine) const {
  return device_ctx_->IsGpuTimeoutDisabled(engine);
}

int Device::Major() const { return device_ctx_->Major(); }
int Device::Minor() const { return device_ctx_->Minor(); }
int Device::Stepping() const { return device_ctx_->Stepping(); }
bool Device::IsDgpu() const { return device_ctx_->IsDgpu(); }
const char *Device::ProductName() const { return device_ctx_->ProductName(); }
uint64_t Device::Uuid() const { return device_ctx_->Uuid(); }
uint32_t Device::Family() const { return device_ctx_->Family(); }
uint32_t Device::DeviceId() const { return device_ctx_->DeviceId(); }
uint32_t Device::WavefrontSize() const { return device_ctx_->WavefrontSize(); }
uint32_t Device::ComputeUnitCount() const { return device_ctx_->ComputeUnitCount(); }
uint32_t Device::MaxEngineClockMhz() const { return device_ctx_->MaxEngineClockMhz(); }
uint32_t Device::WatchPointsNum() const { return device_ctx_->WatchPointsNum(); }
uint32_t Device::PciBusAddr() const { return device_ctx_->PciBusAddr(); }
uint32_t Device::MemoryBusWidth() const { return device_ctx_->MemoryBusWidth(); }
uint32_t Device::MaxMemoryClockMhz() const { return device_ctx_->MaxMemoryClockMhz(); }
uint32_t Device::WavePerCu() const { return device_ctx_->WavePerCu(); }
uint32_t Device::SimdPerCu() const { return device_ctx_->SimdPerCu(); }
uint32_t Device::MaxScratchSlotsPerCu() const { return device_ctx_->MaxScratchSlotsPerCu(); }
uint32_t Device::NumShaderEngine() const { return device_ctx_->NumShaderEngine(); }
uint32_t Device::ShaderArrayPerShaderEngine() const { return device_ctx_->ShaderArrayPerShaderEngine(); }
uint32_t Device::NumSdmaEngines() const { return device_ctx_->NumSdmaEngines(); }
uint32_t Device::SdmaEngine(uint32_t idx) const { return device_ctx_->SdmaEngine(idx); }
uint32_t Device::Domain() const { return device_ctx_->Domain(); }
uint32_t Device::NumGws() const { return device_ctx_->NumGws(); }
uint32_t Device::AsicRevision() const { return device_ctx_->AsicRevision(); }
uint64_t Device::LocalVisibleHeapSize() const { return device_ctx_->LocalVisibleHeapSize(); }
uint64_t Device::LocalInvisibleHeapSize() const { return device_ctx_->LocalInvisibleHeapSize(); }
uint64_t Device::NonLocalHeapSize() const { return device_ctx_->NonLocalHeapSize(); }
uint64_t Device::PrivateApertureBase() const { return device_ctx_->PrivateApertureBase(); }
uint64_t Device::PrivateApertureSize() const { return device_ctx_->PrivateApertureSize(); }
uint64_t Device::SharedApertureBase() const { return device_ctx_->SharedApertureBase(); }
uint64_t Device::SharedApertureSize() const { return device_ctx_->SharedApertureSize(); }
uint32_t Device::LdsSize() const { return device_ctx_->LdsSize(); }
uint64_t Device::GpuCounterFrequency() const { return device_ctx_->GpuCounterFrequency(); }
uint32_t Device::UserQueueSize() const { return device_ctx_->UserQueueSize(); }
uint32_t Device::MecFwVersion() const { return device_ctx_->MecFwVersion(); }
uint32_t Device::SdmaFwVersion() const { return device_ctx_->SdmaFwVersion(); }
uint32_t Device::L1CacheSize() const { return device_ctx_->L1CacheSize(); }
uint32_t Device::L2CacheSize() const { return device_ctx_->L2CacheSize(); }
uint32_t Device::L3CacheSize() const { return device_ctx_->L3CacheSize(); }
uint32_t Device::Gl2CacheLineSize() const { return device_ctx_->Gl2CacheLineSize(); }
bool Device::SupportStateShadowingByCpFw() const { return device_ctx_->SupportStateShadowingByCpFw(); }
bool Device::SupportPlatformAtomic() const { return device_ctx_->SupportPlatformAtomic(); }
uint32_t Device::ComputeEngine() const { return device_ctx_->ComputeEngine(); }
uint32_t Device::NumCpQueues() const { return device_ctx_->NumCpQueues(); }
bool Device::EnableBigPageAlignment() const { return device_ctx_->EnableBigPageAlignment(); }
uint32_t Device::BigPageAlignmentSize() const { return device_ctx_->BigPageAlignmentSize(); }
uint32_t Device::HwBigPageMinAlignmentSize() const { return device_ctx_->HwBigPageMinAlignmentSize(); }
uint32_t Device::HwBigPageAlignmentSize() const { return device_ctx_->HwBigPageAlignmentSize(); }

void Device::SetAllocationInfo(void *data, uint64_t size,
                               thunk_proxy::AllocDomain domain, uint64_t addr,
                               uint32_t mem_flags,
                               uint32_t engine_flag) const {
  thunk_proxy::SetAllocationInfo(data, size, domain, addr, mem_flags,
                                 engine_flag, *device_ctx_);
}

} // namespace thunk
} // namespace wsl
