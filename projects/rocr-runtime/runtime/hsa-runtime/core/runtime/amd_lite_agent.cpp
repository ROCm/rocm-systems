////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
////////////////////////////////////////////////////////////////////////////////

#if defined(__linux__)

#include "core/inc/amd_lite_agent.h"

#include <atomic>
#include <climits>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "core/inc/amd_lite_aql_queue.h"
#include "core/inc/amd_lite_linux_driver.h"
#include "core/inc/amd_memory_region.h"
#include "core/inc/exceptions.h"
#include "core/util/utils.h"

namespace rocr {
namespace AMD {
namespace {

constexpr uint32_t kDefaultCuCount = 64;
constexpr uint32_t kDefaultSimdPerCu = 2;
constexpr uint32_t kDefaultWavefrontSize = 32;
constexpr uint32_t kDefaultLdsSize = 64 * 1024;
constexpr uint32_t kDefaultMaxWavesPerSimd = 16;
constexpr uint64_t kFallbackLocalMemSize = 256ull * 1024 * 1024;
constexpr uint32_t kMinAqlSize = 0x40;
constexpr uint32_t kMaxAqlSize = 0x20000;

void CopyHsaString(void* value, const char* text) {
  constexpr size_t kHsaNameSize = 64;
  std::memset(value, 0, kHsaNameSize);
  std::snprintf(static_cast<char*>(value), kHsaNameSize, "%s", text);
}

bool TraceDmaCopy() {
  const char* value = std::getenv("ROCR_AMDGPU_LITE_TRACE_DMA_COPY");
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

}  // namespace

LiteGpuAgent::LiteGpuAgent(uint32_t node_id, core::DriverType driver_type)
    : GpuAgentInt(node_id, driver_type),
      node_props_{},
      current_coherency_type_(HSA_AMD_COHERENCY_TYPE_COHERENT),
      rec_sdma_eng_override_(false) {
  driver().GetNodeProperties(node_props_, node_id);

  const core::Isa* isa =
      core::IsaRegistry::GetIsa(core::Isa::Version(12, 0, 1),
                                core::IsaFeature::Unsupported,
                                core::IsaFeature::Unsupported);
  if (isa == nullptr) {
    throw hsa_exception(HSA_STATUS_ERROR_INVALID_ISA,
                        "amdgpu_lite gfx1201 ISA is not registered");
  }
  supported_isas_.push_back(isa);
  if (!isa->GetIsaGeneric().empty()) {
    supported_isas_.push_back(core::IsaRegistry::GetIsa(isa->GetIsaGeneric()));
  }

  HsaMemoryProperties fb = {};
  fb.HeapType = HSA_HEAPTYPE_FRAME_BUFFER_PUBLIC;
  fb.SizeInBytes = node_props_.LocalMemSize != 0 ? node_props_.LocalMemSize
                                                : kFallbackLocalMemSize;
  fb.VirtualBaseAddress = 0;
  fb.Width = 256;
  fb.MemoryClockMax = node_props_.MaxEngineClockMhzFCompute;
  regions_.push_back(std::make_shared<MemoryRegion>(
      false, false, false, false, true, this, fb));

  HsaMemoryProperties lds = {};
  lds.HeapType = HSA_HEAPTYPE_GPU_LDS;
  lds.SizeInBytes = kDefaultLdsSize;
  regions_.push_back(std::make_shared<MemoryRegion>(
      false, false, true, false, false, this, lds));
}

LiteGpuAgent::~LiteGpuAgent() = default;

void LiteGpuAgent::ReleaseResources() {}

hsa_status_t LiteGpuAgent::PostToolsInit() { return HSA_STATUS_SUCCESS; }

hsa_status_t LiteGpuAgent::VisitRegion(bool,
                                       hsa_status_t (*callback)(hsa_region_t, void*),
                                       void* data) const {
  if (callback == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  for (const auto& region : regions_) {
    hsa_region_t r = core::MemoryRegion::Convert(region.get());
    hsa_status_t status = callback(r, data);
    if (status != HSA_STATUS_SUCCESS) return status;
  }
  return HSA_STATUS_SUCCESS;
}

void LiteGpuAgent::AcquireQueueMainScratch(ScratchInfo&) {}
void LiteGpuAgent::AcquireQueueAltScratch(ScratchInfo&) {}
void LiteGpuAgent::ReleaseQueueMainScratch(ScratchInfo&) {}
void LiteGpuAgent::ReleaseQueueAltScratch(ScratchInfo&) {}

uint64_t LiteGpuAgent::TranslateTime(uint64_t tick) { return tick; }

void LiteGpuAgent::TranslateTime(core::Signal*, hsa_amd_profiling_dispatch_time_t& time) {
  time.start = 0;
  time.end = 0;
}

void LiteGpuAgent::TranslateTime(core::Signal*, hsa_amd_profiling_async_copy_time_t& time) {
  time.start = 0;
  time.end = 0;
}

void LiteGpuAgent::InvalidateCodeCaches(void*, size_t) {}

bool LiteGpuAgent::current_coherency_type(hsa_amd_coherency_type_t type) {
  current_coherency_type_ = type;
  return true;
}

void LiteGpuAgent::RegisterGangPeer(core::Agent&, unsigned int) {}
void LiteGpuAgent::RegisterRecSdmaEngIdMaskPeer(core::Agent&, uint32_t) {}

hsa_status_t LiteGpuAgent::IterateRegion(
    hsa_status_t (*callback)(hsa_region_t, void*), void* data) const {
  return VisitRegion(false, callback, data);
}

hsa_status_t LiteGpuAgent::IterateSupportedIsas(
    hsa_status_t (*callback)(hsa_isa_t, void*), void* data) const {
  if (callback == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  for (const auto* isa : supported_isas_) {
    hsa_isa_t handle = core::Isa::Handle(isa);
    hsa_status_t status = callback(handle, data);
    if (status != HSA_STATUS_SUCCESS) return status;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t LiteGpuAgent::IterateCache(
    hsa_status_t (*)(hsa_cache_t, void*), void*) const {
  return HSA_STATUS_SUCCESS;
}

hsa_status_t LiteGpuAgent::QueueCreate(size_t size, hsa_queue_type32_t queue_type,
                                       uint64_t flags,
                                       core::HsaEventCallback event_callback, void* data,
                                       uint32_t, uint32_t,
                                       bool metadata_queue,
                                       core::Queue** queue) {
  if (queue == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *queue = nullptr;
  if (metadata_queue) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (queue_type != HSA_QUEUE_TYPE_SINGLE && queue_type != HSA_QUEUE_TYPE_MULTI) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  if ((flags & HSA_AMD_QUEUE_CREATE_DEVICE_MEM_RING_BUF) != 0 ||
      (flags & HSA_AMD_QUEUE_CREATE_DEVICE_MEM_QUEUE_DESCRIPTOR) != 0) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  if (!IsPowerOfTwo(size) || size < kMinAqlSize || size > kMaxAqlSize) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  auto* shared_queue = static_cast<core::SharedQueue*>(
      core::Runtime::runtime_singleton_->system_allocator()(
          sizeof(core::SharedQueue), MemoryRegion::GetPageSize(),
          core::MemoryRegion::AllocateQueueObject, node_id()));
  if (shared_queue == nullptr) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  std::memset(shared_queue, 0, sizeof(*shared_queue));

  try {
    *queue = new LiteAqlQueue(shared_queue, this, size, flags, event_callback, data);
  } catch (...) {
    core::Runtime::runtime_singleton_->system_deallocator()(shared_queue);
    throw;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t LiteGpuAgent::DmaCopy(void* dst, const void* src, size_t size) {
  if ((dst == nullptr && size != 0) || (src == nullptr && size != 0)) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  if (TraceDmaCopy()) {
    std::fprintf(stderr, "ROCR amdgpu_lite DmaCopy dst=%p src=%p size=%zu\n",
                 dst, src, size);
  }
  if (size != 0) {
    std::memcpy(dst, src, size);
    static_cast<LinuxAmdgpuLiteDriver&>(driver()).RegisterVramShadow(dst, size, src);
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t LiteGpuAgent::DmaCopyStatus(core::Agent&, core::Agent&,
                                         uint32_t* engine_ids_mask) {
  if (engine_ids_mask == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *engine_ids_mask = 1;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t LiteGpuAgent::DmaPreferredEngine(core::Agent&, core::Agent&,
                                              uint32_t* recommended_ids_mask) {
  if (recommended_ids_mask == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  *recommended_ids_mask = 1;
  return HSA_STATUS_SUCCESS;
}

hsa_status_t LiteGpuAgent::GetInfo(hsa_agent_info_t attribute, void* value) const {
  if (value == nullptr) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  const size_t attribute_u = static_cast<size_t>(attribute);
  const core::Isa* isa = supported_isas_.empty() ? nullptr : supported_isas_[0];

  switch (attribute_u) {
    case HSA_AGENT_INFO_NAME:
      CopyHsaString(value, isa != nullptr ? isa->GetProcessorName().c_str() : "gfx1201");
      return HSA_STATUS_SUCCESS;
    case HSA_AGENT_INFO_VENDOR_NAME:
      CopyHsaString(value, "AMD");
      return HSA_STATUS_SUCCESS;
    case HSA_AGENT_INFO_FEATURE:
      *static_cast<hsa_agent_feature_t*>(value) = HSA_AGENT_FEATURE_KERNEL_DISPATCH;
      return HSA_STATUS_SUCCESS;
    case HSA_AGENT_INFO_MACHINE_MODEL:
#if defined(HSA_LARGE_MODEL)
      *static_cast<hsa_machine_model_t*>(value) = HSA_MACHINE_MODEL_LARGE;
#else
      *static_cast<hsa_machine_model_t*>(value) = HSA_MACHINE_MODEL_SMALL;
#endif
      return HSA_STATUS_SUCCESS;
    case HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE:
    case HSA_AGENT_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES:
      *static_cast<hsa_default_float_rounding_mode_t*>(value) =
          HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR;
      return HSA_STATUS_SUCCESS;
    case HSA_AGENT_INFO_FAST_F16_OPERATION:
      *static_cast<bool*>(value) = true;
      return HSA_STATUS_SUCCESS;
    case HSA_AGENT_INFO_DEVICE:
      *static_cast<hsa_device_type_t*>(value) = HSA_DEVICE_TYPE_GPU;
      return HSA_STATUS_SUCCESS;
    case HSA_AGENT_INFO_PROFILE:
      *static_cast<hsa_profile_t*>(value) = HSA_PROFILE_BASE;
      return HSA_STATUS_SUCCESS;
    case HSA_AGENT_INFO_WAVEFRONT_SIZE:
      *static_cast<uint32_t*>(value) = kDefaultWavefrontSize;
      return HSA_STATUS_SUCCESS;
    case HSA_AGENT_INFO_WORKGROUP_MAX_DIM: {
      const uint16_t group_size[3] = {1024, 1024, 1024};
      std::memcpy(value, group_size, sizeof(group_size));
      return HSA_STATUS_SUCCESS;
    }
    case HSA_AGENT_INFO_WORKGROUP_MAX_SIZE:
      *static_cast<uint32_t*>(value) = 1024;
      return HSA_STATUS_SUCCESS;
    case HSA_AGENT_INFO_GRID_MAX_DIM: {
      const hsa_dim3_t grid_size = {INT32_MAX, UINT16_MAX, UINT16_MAX};
      std::memcpy(value, &grid_size, sizeof(grid_size));
      return HSA_STATUS_SUCCESS;
    }
    case HSA_AGENT_INFO_GRID_MAX_SIZE:
      *static_cast<uint32_t*>(value) = UINT32_MAX;
      return HSA_STATUS_SUCCESS;
    case HSA_AGENT_INFO_FBARRIER_MAX_SIZE:
      *static_cast<uint32_t*>(value) = 32;
      return HSA_STATUS_SUCCESS;
    case HSA_AGENT_INFO_QUEUES_MAX:
      *static_cast<uint32_t*>(value) = 128;
      return HSA_STATUS_SUCCESS;
    case HSA_AGENT_INFO_QUEUE_MIN_SIZE:
      *static_cast<uint32_t*>(value) = 64;
      return HSA_STATUS_SUCCESS;
    case HSA_AGENT_INFO_QUEUE_MAX_SIZE:
      *static_cast<uint32_t*>(value) = 131072;
      return HSA_STATUS_SUCCESS;
    case HSA_AGENT_INFO_QUEUE_TYPE:
      *static_cast<hsa_queue_type32_t*>(value) = HSA_QUEUE_TYPE_MULTI;
      return HSA_STATUS_SUCCESS;
    case HSA_AGENT_INFO_NODE:
      *static_cast<uint32_t*>(value) = node_id();
      return HSA_STATUS_SUCCESS;
    case HSA_AGENT_INFO_CACHE_SIZE: {
      uint32_t cache_sizes[4] = {32 * 1024, 4 * 1024 * 1024, 0, 0};
      std::memcpy(value, cache_sizes, sizeof(cache_sizes));
      return HSA_STATUS_SUCCESS;
    }
    case HSA_AGENT_INFO_ISA:
      if (isa == nullptr) return HSA_STATUS_ERROR_INVALID_ISA;
      *static_cast<hsa_isa_t*>(value) = core::Isa::Handle(isa);
      return HSA_STATUS_SUCCESS;
    case HSA_AGENT_INFO_EXTENSIONS:
      std::memset(value, 0, sizeof(uint8_t) * 128);
      return HSA_STATUS_SUCCESS;
    case HSA_AGENT_INFO_VERSION_MAJOR:
      *static_cast<uint16_t*>(value) = 1;
      return HSA_STATUS_SUCCESS;
    case HSA_AGENT_INFO_VERSION_MINOR:
      *static_cast<uint16_t*>(value) = 1;
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_AGENT_INFO_CHIP_ID:
      *static_cast<uint32_t*>(value) = node_props_.DeviceId;
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_AGENT_INFO_CACHELINE_SIZE:
      *static_cast<uint32_t*>(value) = 64;
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT:
    case HSA_AMD_AGENT_INFO_COOPERATIVE_COMPUTE_UNIT_COUNT:
      *static_cast<uint32_t*>(value) = kDefaultCuCount;
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY:
      *static_cast<uint32_t*>(value) = 2500;
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_AGENT_INFO_DRIVER_NODE_ID:
      *static_cast<uint32_t*>(value) = node_id();
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_AGENT_INFO_PRODUCT_NAME:
      CopyHsaString(value, "AMD Radeon RX 9000 (amdgpu_lite)");
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU:
      *static_cast<uint32_t*>(value) = kDefaultSimdPerCu * kDefaultMaxWavesPerSimd;
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU:
      *static_cast<uint32_t*>(value) = kDefaultSimdPerCu;
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES:
      *static_cast<uint32_t*>(value) = 4;
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE:
      *static_cast<uint32_t*>(value) = 2;
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_AGENT_INFO_HDP_FLUSH:
      std::memset(value, 0, sizeof(hsa_amd_hdp_flush_t));
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_AGENT_INFO_SVM_DIRECT_HOST_ACCESS:
      *static_cast<bool*>(value) = true;
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_AGENT_INFO_MEMORY_AVAIL:
      *static_cast<uint64_t*>(value) = node_props_.LocalMemSize;
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_AGENT_INFO_TIMESTAMP_FREQUENCY:
      *static_cast<uint64_t*>(value) = 1000000000ull;
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_AGENT_INFO_NUM_SDMA_ENG:
      *static_cast<uint32_t*>(value) = 1;
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_AGENT_INFO_NUM_SDMA_XGMI_ENG:
      *static_cast<uint32_t*>(value) = 0;
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_AGENT_INFO_IOMMU_SUPPORT:
      *static_cast<hsa_amd_iommu_version_t*>(value) = HSA_IOMMU_SUPPORT_NONE;
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_AGENT_INFO_NUM_XCC:
      *static_cast<uint32_t*>(value) = 1;
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_AGENT_INFO_DRIVER_UID:
      *static_cast<uint32_t*>(value) = static_cast<uint32_t>(KfdGpuID());
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_AGENT_INFO_UUID:
      std::snprintf(static_cast<char*>(value), 21, "GPU-%016llx",
                    static_cast<unsigned long long>(
                        node_props_.UniqueID != 0 ? node_props_.UniqueID
                                                  : (0xA100000000000000ull | node_props_.DeviceId)));
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_AGENT_INFO_ASIC_REVISION:
      *static_cast<uint32_t*>(value) = node_props_.Capability.ui32.ASICRevision;
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_AGENT_INFO_MEMORY_PROPERTIES:
    case HSA_AMD_AGENT_INFO_AQL_EXTENSIONS:
      std::memset(value, 0, sizeof(uint8_t) * 8);
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_AGENT_INFO_SCRATCH_LIMIT_MAX:
    case HSA_AMD_AGENT_INFO_SCRATCH_LIMIT_CURRENT:
      *static_cast<uint64_t*>(value) = 0;
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_AGENT_INFO_PM4_EMULATION:
    case HSA_AMD_AGENT_INFO_HAS_EXPERT_SCHED_MODE:
      *static_cast<bool*>(value) = false;
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_AGENT_INFO_LUID:
      static_cast<hsa_luid_t*>(value)->low = 0;
      static_cast<hsa_luid_t*>(value)->high = 0;
      return HSA_STATUS_SUCCESS;
    case HSA_AMD_AGENT_INFO_CUID:
      std::memset(value, 0, 16);
      return HSA_STATUS_SUCCESS;
    default:
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
}

}  // namespace AMD
}  // namespace rocr

#endif  // defined(__linux__)
