#include "shared/include/thunks.h"
#include "shared/include/device.h"
#include "shared/include/lda_chain.h"
#include "shared/include/platform.h"
#include "shared/include/thunk_proxy/thunk_proxy.h"
#include <limits>
#include <memory>
#include <vector>

using namespace std;

namespace wsl {
namespace thunk {

namespace dx = wsl::thunk::d3dthunk;

namespace {

constexpr uint32_t kInvalidSegmentId = std::numeric_limits<uint32_t>::max();

struct SegmentInfo {
  uint32_t segment_id = 0;
  uint32_t segment_type = 0;
};

ErrorCode QueryAllSegments(LUID luid, std::vector<SegmentInfo> *segments) {
  if (!segments)
    return ErrorCode::InvalidPointer;

  segments->clear();

  D3DKMT_QUERYSTATISTICS adapter_query{};
  adapter_query.Type = D3DKMT_QUERYSTATISTICS_ADAPTER;
  adapter_query.AdapterLuid = luid;

  ErrorCode ret = dx::QueryStatistics(&adapter_query);
  if (ret != ErrorCode::Success)
    return ret;

  const uint32_t segment_count =
      adapter_query.QueryResult.AdapterInformation.NbSegments;

  for (uint32_t i = 0; i < segment_count; i++) {
    D3DKMT_QUERYSTATISTICS seg_query{};
    seg_query.Type = D3DKMT_QUERYSTATISTICS_SEGMENT;
    seg_query.AdapterLuid = luid;
    seg_query.QuerySegment.SegmentId = i;

    ret = dx::QueryStatistics(&seg_query);
    if (ret != ErrorCode::Success)
      return ret;

    const auto &seg = seg_query.QueryResult.SegmentInformation;
    SegmentInfo info;
    info.segment_id = i;
    info.segment_type = seg.SegmentProperties.SegmentType;
    segments->push_back(info);
  }

  return ErrorCode::Success;
}

bool ResolveSegmentId(const std::vector<SegmentInfo> &segments,
                      D3DKMT_QUERYSTATISTICS_SEGMENT_TYPE segment_type,
                      uint32_t default_id, uint32_t *segment_id) {
  for (const auto &seg : segments) {
    if (seg.segment_type == segment_type) {
      *segment_id = seg.segment_id;
      return true;
    }
  }
  *segment_id = default_id;
  return false;
}

ErrorCode QuerySegmentGroupUsage(LUID luid, uint32_t segment_group,
                                 uint64_t *bytes_allocated) {
  D3DKMT_QUERYSTATISTICS stats{};
  stats.Type = D3DKMT_QUERYSTATISTICS_SEGMENT_GROUP_USAGE;
  stats.AdapterLuid = luid;
  stats.QuerySegmentGroupUsage.PhysicalAdapterIndex = 0;
  stats.QuerySegmentGroupUsage.SegmentGroup = segment_group;

  const ErrorCode ret = dx::QueryStatistics(&stats);
  if (ret != ErrorCode::Success) {
    *bytes_allocated = 0;
    return ret;
  }

  *bytes_allocated =
      stats.QueryResult.SegmentGroupUsageInformation.AllocatedBytes;
  return ErrorCode::Success;
}

ErrorCode QuerySegmentBytesResident(LUID luid, uint32_t segment_id,
                                    uint64_t *bytes_resident) {
  D3DKMT_QUERYSTATISTICS stats{};
  stats.Type = D3DKMT_QUERYSTATISTICS_SEGMENT;
  stats.AdapterLuid = luid;
  stats.QuerySegment.SegmentId = segment_id;

  const ErrorCode ret = dx::QueryStatistics(&stats);
  if (ret != ErrorCode::Success) {
    *bytes_resident = 0;
    return ret;
  }

  *bytes_resident = stats.QueryResult.SegmentInformation.BytesResident;
  return ErrorCode::Success;
}

} // namespace

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
  ErrorCode ret = device_ctx_->Init();
  if (ret != ErrorCode::Success)
    return ret;
  return InitSegmentIds();
}

ErrorCode Device::InitSegmentIds() {
  // Default to hardcoded segment ids; these act as fallbacks when the
  // runtime segment query is unavailable or fails.
  segment_ids_.fb = 0;
  segment_ids_.inv_fb = LocalInvisibleHeapSize() ? 1 : kInvalidSegmentId;
  segment_ids_.non_local = LocalInvisibleHeapSize() ? 4 : 3;

  if (Platform::instance().WddmVersion() >= KMT_DRIVERVERSION_WDDM_3_1) {
    const LUID luid = AdapterLuid();
    std::vector<SegmentInfo> segments;
    if (QueryAllSegments(luid, &segments) != ErrorCode::Success) {
      // Keep hardcoded fallback ids
      return ErrorCode::Success;
    }

    ResolveSegmentId(segments, D3DKMT_QUERYSTATISTICS_SEGMENT_TYPE_MEMORY, segment_ids_.fb,
                    &segment_ids_.fb);

    if (LocalInvisibleHeapSize())
      segment_ids_.inv_fb = segment_ids_.fb + 1;

    ResolveSegmentId(segments, D3DKMT_QUERYSTATISTICS_SEGMENT_TYPE_SYSMEM,
                    segment_ids_.non_local, &segment_ids_.non_local);
  }

  return ErrorCode::Success;
}

ErrorCode Device::QueryVramSegmentUsage(VramSegmentKind kind,
                                        uint64_t *usage) const {
  if (!usage)
    return ErrorCode::InvalidPointer;

  *usage = 0;

  if (kind == VramSegmentKind::kInvFb &&
      segment_ids_.inv_fb == kInvalidSegmentId)
    return ErrorCode::Success;

  const LUID luid = AdapterLuid();
  const bool seg_group_supported =
      Platform::instance().WddmVersion() >= KMT_DRIVERVERSION_WDDM_3_1;

  if (kind == VramSegmentKind::kLocal && seg_group_supported) {
    ErrorCode ret = QuerySegmentGroupUsage(
        luid, D3DKMT_MEMORY_SEGMENT_GROUP_LOCAL, usage);
    if (ret == ErrorCode::Success)
      return ErrorCode::Success;
  }

  if (kind == VramSegmentKind::kNonLocal && seg_group_supported) {
    ErrorCode ret = QuerySegmentGroupUsage(
        luid, D3DKMT_MEMORY_SEGMENT_GROUP_NON_LOCAL, usage);
    if (ret == ErrorCode::Success)
      return ErrorCode::Success;
  }

  if (kind == VramSegmentKind::kLocal) {
    uint64_t fb = 0;
    ErrorCode ret =
        QuerySegmentBytesResident(luid, segment_ids_.fb, &fb);
    if (ret != ErrorCode::Success)
      return ret;

    *usage = fb;
    if (segment_ids_.inv_fb == kInvalidSegmentId)
      return ErrorCode::Success;

    uint64_t inv = 0;
    ret = QuerySegmentBytesResident(luid, segment_ids_.inv_fb, &inv);
    if (ret != ErrorCode::Success)
      return ret;
    *usage += inv;
    return ErrorCode::Success;
  }

  uint32_t segment_id = 0;
  switch (kind) {
  case VramSegmentKind::kFb:
    segment_id = segment_ids_.fb;
    break;
  case VramSegmentKind::kInvFb:
    segment_id = segment_ids_.inv_fb;
    break;
  case VramSegmentKind::kNonLocal:
    segment_id = segment_ids_.non_local;
    break;
  default:
    return ErrorCode::InvalidParams;
  }

  return QuerySegmentBytesResident(luid, segment_id, usage);
}

uint64_t Device::VramTotal() const {
  uint64_t total = LocalVisibleHeapSize() + LocalInvisibleHeapSize();
  if (!IsDgpu())
    total += NonLocalHeapSize();
  return total;
}

ErrorCode Device::QueryVramUsage(uint64_t *usage_bytes) const {
  if (!usage_bytes)
    return ErrorCode::InvalidPointer;

  ErrorCode ret = QueryVramSegmentUsage(VramSegmentKind::kLocal, usage_bytes);
  if (ret != ErrorCode::Success)
    return ret;

  if (IsDgpu())
    return ErrorCode::Success;

  uint64_t used_non_local = 0;
  ret = QueryVramSegmentUsage(VramSegmentKind::kNonLocal, &used_non_local);
  if (ret != ErrorCode::Success)
    return ret;

  *usage_bytes += used_non_local;
  return ErrorCode::Success;
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
