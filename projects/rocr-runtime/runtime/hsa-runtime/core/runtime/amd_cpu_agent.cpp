////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#include "core/inc/amd_cpu_agent.h"

#include <algorithm>
#include <cstring>
#include <thread>

#include "core/inc/amd_memory_region.h"
#include "core/inc/driver.h"
#include "core/inc/host_queue.h"

#include "inc/hsa_ext_image.h"

namespace rocr {
namespace AMD {
CpuAgent::CpuAgent(HSAuint32 node, const HsaNodeProperties& node_props,
                   core::DriverType driver_type)
    : core::Agent(core::Runtime::runtime_singleton_->AgentDriver(driver_type), node, kAmdCpuDevice),
      properties_(node_props) {
  InitRegionList();

  InitCacheList();
}

CpuAgent::~CpuAgent() {
  std::for_each(regions_.begin(), regions_.end(), DeleteObject());
  regions_.clear();
}

void CpuAgent::InitRegionList() {
  const bool is_apu_node = (properties_.NumFComputeCores > 0);

  std::vector<HsaMemoryProperties> mem_props(properties_.NumMemoryBanks);
  if (HSA_STATUS_SUCCESS == driver().GetMemoryProperties(node_id(), mem_props)) {
    std::vector<HsaMemoryProperties>::iterator system_prop =
        std::find_if(mem_props.begin(), mem_props.end(), [](HsaMemoryProperties prop) -> bool {
          return (prop.SizeInBytes > 0 && prop.HeapType == HSA_HEAPTYPE_SYSTEM);
        });

    HsaMemoryProperties system_props;
    std::memset(&system_props, 0, sizeof(HsaMemoryProperties));
    system_props.HeapType = HSA_HEAPTYPE_SYSTEM;
    system_props.SizeInBytes = 0;
    system_props.VirtualBaseAddress = 0;

    if (system_prop != mem_props.end()) system_props = *system_prop;

    // Fine-Grain Memory
    regions_.push_back(new MemoryRegion(true, false, is_apu_node, false, true, this, system_props));

    // Ext-Fine-Grain Memory
    regions_.push_back(new MemoryRegion(false, false, is_apu_node, true, true, this, system_props));

    // Kernargs
    regions_.push_back(new MemoryRegion(true, true, is_apu_node, false, true, this, system_props));

    if (!is_apu_node) {
      // Coarse Grain
      regions_.push_back(new MemoryRegion(false, false, is_apu_node, false, true, this, system_props));
    }
  }
}

void CpuAgent::InitCacheList() {
  // Get CPU cache information.
  cache_props_.resize(properties_.NumCaches);
  if (HSA_STATUS_SUCCESS !=
      driver().GetCacheProperties(node_id(), properties_.CComputeIdLo, cache_props_)) {
    cache_props_.clear();
  } else {
    // Only store CPU D-cache.
    for (size_t cache_id = 0; cache_id < cache_props_.size(); ++cache_id) {
      const HsaCacheType type = cache_props_[cache_id].CacheType;
      if (type.ui32.CPU != 1 || type.ui32.Instruction == 1) {
        cache_props_.erase(cache_props_.begin() + cache_id);
        --cache_id;
      }
    }
  }

  // Update cache objects
  caches_.clear();
  caches_.resize(cache_props_.size());
  char name[64];
  GetInfo(HSA_AGENT_INFO_NAME, name);
  std::string deviceName = name;
  for (size_t i = 0; i < caches_.size(); i++)
    caches_[i].reset(new core::Cache(deviceName + " L" + std::to_string(cache_props_[i].CacheLevel),
                                     cache_props_[i].CacheLevel, cache_props_[i].CacheSize));
}

hsa_status_t CpuAgent::VisitRegion(bool include_peer,
                                   hsa_status_t (*callback)(hsa_region_t region,
                                                            void* data),
                                   void* data) const {
  if (!include_peer) {
    return VisitRegion(regions_, callback, data);
  }

  // Expose all system regions in the system.
  hsa_status_t stat = VisitRegion(
      core::Runtime::runtime_singleton_->system_regions_fine(), callback, data);
  if (stat != HSA_STATUS_SUCCESS) {
    return stat;
  }

  return VisitRegion(core::Runtime::runtime_singleton_->system_regions_coarse(),
                     callback, data);
}

hsa_status_t CpuAgent::VisitRegion(
    const std::vector<const core::MemoryRegion*>& regions,
    hsa_status_t (*callback)(hsa_region_t region, void* data),
    void* data) const {
  for (const core::MemoryRegion* region : regions) {
    if (!region->user_visible()) continue;
    hsa_region_t region_handle = core::MemoryRegion::Convert(region);
    hsa_status_t status = callback(region_handle, data);
    if (status != HSA_STATUS_SUCCESS) {
      return status;
    }
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t CpuAgent::IterateRegion(
    hsa_status_t (*callback)(hsa_region_t region, void* data),
    void* data) const {
  return VisitRegion(true, callback, data);
}

hsa_status_t CpuAgent::IterateCache(hsa_status_t (*callback)(hsa_cache_t cache, void* data),
                                    void* data) const {
  for (size_t i = 0; i < caches_.size(); i++) {
    hsa_status_t stat = callback(core::Cache::Convert(caches_[i].get()), data);
    if (stat != HSA_STATUS_SUCCESS) return stat;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t CpuAgent::IterateSupportedIsas(
                  hsa_status_t (*callback)(hsa_isa_t isa, void* data),
                                                          void* data) const {
  AMD::callback_t<decltype(callback)> call(callback);
  for (const auto& isa : supported_isas()) {
    hsa_status_t stat = call(core::Isa::Handle(isa), data);
    if (stat != HSA_STATUS_SUCCESS) return stat;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t CpuAgent::GetInfo(hsa_agent_info_t attribute, void* value) const {

  // agent, and vendor name size limit
  const size_t attribute_u = static_cast<size_t>(attribute);

  switch (attribute_u) {

    // The code copies HsaNodeProperties.MarketingName a Unicode string
    // which is encoded in UTF-16 as a 7-bit ASCII string. The value of
    // HsaNodeProperties.MarketingName is obtained from the "model name"
    // property of /proc/cpuinfo file
    case HSA_AGENT_INFO_NAME:
    case HSA_AMD_AGENT_INFO_PRODUCT_NAME: {
      std::memset(value, 0, HSA_PUBLIC_NAME_SIZE);
      char* temp = reinterpret_cast<char*>(value);
      for (uint32_t idx = 0;
           properties_.MarketingName[idx] != 0 && idx < HSA_PUBLIC_NAME_SIZE - 1; idx++) {
        temp[idx] = (uint8_t)properties_.MarketingName[idx];
      }
      break;
    }
    case HSA_AGENT_INFO_VENDOR_NAME:
      // TODO: hardcode for now, wait until SWDEV-88894 implemented
      std::memset(value, 0, HSA_PUBLIC_NAME_SIZE);
      std::memcpy(value, "CPU", sizeof("CPU"));
      break;
    case HSA_AGENT_INFO_FEATURE:
      *((hsa_agent_feature_t*)value) = static_cast<hsa_agent_feature_t>(0);
      break;
    case HSA_AGENT_INFO_MACHINE_MODEL:
#if defined(HSA_LARGE_MODEL)
      *((hsa_machine_model_t*)value) = HSA_MACHINE_MODEL_LARGE;
#else
      *((hsa_machine_model_t*)value) = HSA_MACHINE_MODEL_SMALL;
#endif
      break;
    case HSA_AGENT_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES:
    case HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE:
      // TODO: validate if this is true.
      *((hsa_default_float_rounding_mode_t*)value) =
          HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR;
      break;
    case HSA_AGENT_INFO_FAST_F16_OPERATION:
      // TODO: validate if this is true.
      *((bool*)value) = false;
      break;
    case HSA_AGENT_INFO_PROFILE:
      *((hsa_profile_t*)value) = HSA_PROFILE_FULL;
      break;
    case HSA_AGENT_INFO_WAVEFRONT_SIZE:
      *((uint32_t*)value) = 0;
      break;
    case HSA_AGENT_INFO_WORKGROUP_MAX_DIM:
      std::memset(value, 0, sizeof(uint16_t) * 3);
      break;
    case HSA_AGENT_INFO_WORKGROUP_MAX_SIZE:
      *((uint32_t*)value) = 0;
      break;
    case HSA_AGENT_INFO_GRID_MAX_DIM:
      std::memset(value, 0, sizeof(hsa_dim3_t));
      break;
    case HSA_AGENT_INFO_GRID_MAX_SIZE:
      *((uint32_t*)value) = 0;
      break;
    case HSA_AGENT_INFO_FBARRIER_MAX_SIZE:
      // TODO: ?
      *((uint32_t*)value) = 0;
      break;
    case HSA_AGENT_INFO_QUEUES_MAX:
      *((uint32_t*)value) = 0;
      break;
    case HSA_AGENT_INFO_QUEUE_MIN_SIZE:
      *((uint32_t*)value) = 0;
      break;
    case HSA_AGENT_INFO_QUEUE_MAX_SIZE:
      *((uint32_t*)value) = 0;
      break;
    case HSA_AGENT_INFO_QUEUE_TYPE:
      *((hsa_queue_type32_t*)value) = HSA_QUEUE_TYPE_MULTI;
      break;
    case HSA_AGENT_INFO_NODE:
      // TODO: associate with OS NUMA support (numactl / GetNumaProcessorNode).
      *((uint32_t*)value) = node_id();
      break;
    case HSA_AGENT_INFO_DEVICE:
      *((hsa_device_type_t*)value) = HSA_DEVICE_TYPE_CPU;
      break;
    case HSA_AGENT_INFO_CACHE_SIZE: {
      std::memset(value, 0, sizeof(uint32_t) * 4);
      const size_t num_cache = cache_props_.size();
      for (size_t i = 0; i < num_cache; ++i) {
        const uint32_t line_level = cache_props_[i].CacheLevel;
        ((uint32_t*)value)[line_level - 1] = cache_props_[i].CacheSize * 1024;
      }
    } break;
    case HSA_AGENT_INFO_ISA:
      ((hsa_isa_t*)value)->handle = 0;
      break;
    case HSA_AGENT_INFO_EXTENSIONS:
      memset(value, 0, sizeof(uint8_t) * 128);
      break;
    case HSA_AGENT_INFO_VERSION_MAJOR:
      *((uint16_t*)value) = 1;
      break;
    case HSA_AGENT_INFO_VERSION_MINOR:
      *((uint16_t*)value) = 1;
      break;
    case HSA_EXT_AGENT_INFO_IMAGE_1D_MAX_ELEMENTS:
    case HSA_EXT_AGENT_INFO_IMAGE_1DA_MAX_ELEMENTS:
    case HSA_EXT_AGENT_INFO_IMAGE_1DB_MAX_ELEMENTS:
      *((uint32_t*)value) = 0;
      break;
    case HSA_EXT_AGENT_INFO_IMAGE_2D_MAX_ELEMENTS:
    case HSA_EXT_AGENT_INFO_IMAGE_2DA_MAX_ELEMENTS:
    case HSA_EXT_AGENT_INFO_IMAGE_2DDEPTH_MAX_ELEMENTS:
    case HSA_EXT_AGENT_INFO_IMAGE_2DADEPTH_MAX_ELEMENTS:
      memset(value, 0, sizeof(uint32_t) * 2);
      break;
    case HSA_EXT_AGENT_INFO_IMAGE_3D_MAX_ELEMENTS:
      memset(value, 0, sizeof(uint32_t) * 3);
      break;
    case HSA_EXT_AGENT_INFO_IMAGE_ARRAY_MAX_LAYERS:
      *((uint32_t*)value) = 0;
      break;
    case HSA_EXT_AGENT_INFO_MAX_IMAGE_RD_HANDLES:
    case HSA_EXT_AGENT_INFO_MAX_IMAGE_RORW_HANDLES:
    case HSA_EXT_AGENT_INFO_MAX_SAMPLER_HANDLERS:
      *((uint32_t*)value) = 0;
      break;
    case HSA_AMD_AGENT_INFO_CHIP_ID:
      *((uint32_t*)value) = properties_.DeviceId;
      break;
    case HSA_AMD_AGENT_INFO_CACHELINE_SIZE:
      // TODO: hardcode for now.
      *((uint32_t*)value) = 64;
      break;
    case HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT:
      *((uint32_t*)value) = properties_.NumCPUCores;
      break;
    case HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY:
      *((uint32_t*)value) = properties_.MaxEngineClockMhzCCompute;
      break;
    case HSA_AMD_AGENT_INFO_DRIVER_NODE_ID:
      *((uint32_t*)value) = node_id();
      break;
    case HSA_AMD_AGENT_INFO_MAX_ADDRESS_WATCH_POINTS:
      *((uint32_t*)value) = static_cast<uint32_t>(
          1 << properties_.Capability.ui32.WatchPointsTotalBits);
      break;
    case HSA_AMD_AGENT_INFO_BDFID:
      *((uint32_t*)value) = static_cast<uint32_t>(properties_.LocationId);
      break;
    case HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU:
      *((uint32_t*)value) = static_cast<uint32_t>(
          properties_.NumSIMDPerCU * properties_.MaxWavesPerSIMD);
      break;
    case HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU:
      *((uint32_t*)value) = properties_.NumSIMDPerCU;
      break;
    case HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES:
      *((uint32_t*)value) = properties_.NumShaderBanks;
      break;
    case HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE:
      *((uint32_t*)value) = properties_.NumArrays;
      break;
    case HSA_AMD_AGENT_INFO_HDP_FLUSH:
      *((hsa_amd_hdp_flush_t*)value) = {nullptr, nullptr};
      break;
    case HSA_AMD_AGENT_INFO_DOMAIN:
      *((uint32_t*)value) = static_cast<uint32_t>(properties_.Domain);
      break;
    case HSA_AMD_AGENT_INFO_UUID: {
      // At this point CPU devices do not support UUID's.
      char uuid_tmp[] = "CPU-XX";
      snprintf((char*)value, sizeof(uuid_tmp), "%s", uuid_tmp);
      break;
    }
    case HSA_AMD_AGENT_INFO_ASIC_REVISION:
      *((uint32_t*)value) = static_cast<uint32_t>(properties_.Capability.ui32.ASICRevision);
      break;
    case HSA_AMD_AGENT_INFO_SVM_DIRECT_HOST_ACCESS:
      assert(regions_.size() != 0 && "No device local memory found!");
      *((bool*)value) = true;
      break;
    case HSA_AMD_AGENT_INFO_TIMESTAMP_FREQUENCY:
      return core::Runtime::runtime_singleton_->GetSystemInfo(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY,
                                                              value);
      break;
    case HSA_AMD_AGENT_INFO_ASIC_FAMILY_ID:
      *((uint32_t*)value) = static_cast<uint32_t>(properties_.FamilyID);
      break;
    case HSA_AMD_AGENT_INFO_UCODE_VERSION:
      *((uint32_t*)value) = 0;
      break;
    case HSA_AMD_AGENT_INFO_SDMA_UCODE_VERSION:
      *((uint32_t*)value) = 0;
      break;
    case HSA_AMD_AGENT_INFO_NUM_SDMA_ENG:
      *((uint32_t*)value) = 0;
      break;
    case HSA_AMD_AGENT_INFO_NUM_SDMA_XGMI_ENG:
      *((uint32_t*)value) = 0;
      break;
    case HSA_AMD_AGENT_INFO_IOMMU_SUPPORT:
      *((hsa_amd_iommu_version_t*)value) = HSA_IOMMU_SUPPORT_NONE;
      break;
    case HSA_AMD_AGENT_INFO_NUM_XCC:
      *((uint32_t*)value) = 0;
      break;
    case HSA_AMD_AGENT_INFO_DRIVER_UID:
      *((uint32_t*)value) = 0;
      break;
    case HSA_AMD_AGENT_INFO_NEAREST_CPU:
      ((hsa_agent_t*)value)->handle = 0;
      break;
    case HSA_AMD_AGENT_INFO_MEMORY_PROPERTIES:
      memset(value, 0, sizeof(uint8_t) * 8);
      break;
    case HSA_AMD_AGENT_INFO_AQL_EXTENSIONS:
      memset(value, 0, sizeof(uint8_t) * 8);
      break;
    case HSA_AMD_AGENT_INFO_SCRATCH_LIMIT_MAX:
    case HSA_AMD_AGENT_INFO_SCRATCH_LIMIT_CURRENT:
      *((uint64_t*)value) = 0;
      break;
    case HSA_AMD_AGENT_INFO_CLOCK_COUNTERS:
      memset(value, 0, sizeof(hsa_amd_clock_counters_t));
      break;
    default:
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
      break;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t CpuAgent::QueueCreate(size_t size, hsa_queue_type32_t queue_type, uint64_t flags,
                                   core::HsaEventCallback event_callback, void* data,
                                   uint32_t private_segment_size, uint32_t group_segment_size,
                                   core::Queue** queue) {
  // No HW AQL packet processor on CPU device.
  return HSA_STATUS_ERROR;
}

hsa_status_t CpuAgent::DmaCopy(void* dst, core::Agent& dst_agent, const void* src,
                               core::Agent& src_agent, size_t size,
                               std::vector<core::Signal*>& dep_signals, core::Signal& out_signal) {
  // For cpu to cpu, fire and forget a copy thread.
  const bool profiling_enabled = (dst_agent.profiling_enabled() || src_agent.profiling_enabled());
  if (profiling_enabled) out_signal.async_copy_agent(this);
  std::thread(
      [](void* dst, const void* src, size_t size, std::vector<core::Signal*> dep_signals,
         core::Signal* completion_signal, bool profiling_enabled) {
        for (core::Signal* dep : dep_signals) {
          dep->WaitRelaxed(HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
        }

        if (profiling_enabled) {
          core::Runtime::runtime_singleton_->GetSystemInfo(HSA_SYSTEM_INFO_TIMESTAMP,
                                                           &completion_signal->signal_.start_ts);
        }

        memcpy(dst, src, size);

        if (profiling_enabled) {
          core::Runtime::runtime_singleton_->GetSystemInfo(HSA_SYSTEM_INFO_TIMESTAMP,
                                                           &completion_signal->signal_.end_ts);
        }

        completion_signal->SubRelease(1);
      },
      dst, src, size, dep_signals, &out_signal, profiling_enabled)
      .detach();
  return HSA_STATUS_SUCCESS;
}

}  // namespace amd
}  // namespace rocr
