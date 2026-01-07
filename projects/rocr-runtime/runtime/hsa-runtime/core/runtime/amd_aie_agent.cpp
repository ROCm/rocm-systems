////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2022-2025, Advanced Micro Devices, Inc. All rights reserved.
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

#include "core/inc/amd_aie_agent.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <iterator>
#include <string_view>

#include "core/inc/amd_aie_aql_queue.h"
#include "core/inc/amd_memory_region.h"
#include "core/inc/amd_xdna_driver.h"
#include "core/inc/driver.h"
#include "core/inc/runtime.h"
#include "core/util/os.h"

namespace rocr {
namespace AMD {

AieAgent::AieAgent(uint32_t node, const HsaNodeProperties& node_props)
    : core::Agent(core::Runtime::runtime_singleton_->AgentDriver(core::DriverType::XDNA), node,
                  core::Agent::DeviceType::kAmdAieDevice),
      node_props_(node_props) {
  InitRegionList();
  InitAllocators();
}

AieAgent::~AieAgent() {
  regions_.clear();
}

hsa_status_t AieAgent::VisitRegion(bool include_peer,
                                   hsa_status_t (*callback)(hsa_region_t region,
                                                            void *data),
                                   void *data) const {
  AMD::callback_t<decltype(callback)> call(callback);
  for (const auto& r : regions_) {
    hsa_region_t region_handle(core::MemoryRegion::Convert(r.get()));
    hsa_status_t err = call(region_handle, data);
    if (err != HSA_STATUS_SUCCESS) {
      return err;
    }
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t AieAgent::IterateRegion(
    hsa_status_t (*callback)(hsa_region_t region, void *data),
    void *data) const {
  return VisitRegion(false, callback, data);
}

hsa_status_t AieAgent::IterateCache(hsa_status_t (*callback)(hsa_cache_t cache,
                                                             void *data),
                                    void *data) const {
  // AIE has no caches.
  return HSA_STATUS_ERROR_INVALID_CACHE;
}

hsa_status_t AieAgent::IterateSupportedIsas(
                    hsa_status_t (*callback)(hsa_isa_t isa, void* data),
                                                          void* data) const {
  AMD::callback_t<decltype(callback)> call(callback);
  for (const auto& isa : supported_isas()) {
    hsa_status_t err = call(core::Isa::Handle(isa), data);
    if (err != HSA_STATUS_SUCCESS) return err;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t AieAgent::GetInfo(hsa_agent_info_t attribute, void *value) const {
  const size_t attribute_ = static_cast<size_t>(attribute);

  switch (attribute_) {
    case HSA_AGENT_INFO_NAME:
      std::copy_n(node_props_.AMDName, HSA_PUBLIC_NAME_SIZE, static_cast<char*>(value));
      break;
    case HSA_AGENT_INFO_VENDOR_NAME: {
      constexpr std::string_view vendor_name_info("AMD");
      assert(vendor_name_info.size() < HSA_PUBLIC_NAME_SIZE);
      auto ptr = static_cast<char*>(value);
      std::copy(vendor_name_info.begin(), vendor_name_info.end(), ptr);
      std::fill(std::next(ptr, vendor_name_info.size()), std::next(ptr, HSA_PUBLIC_NAME_SIZE), 0);
      break;
    }
    case HSA_AGENT_INFO_FEATURE:
      *static_cast<hsa_agent_feature_t*>(value) = HSA_AGENT_FEATURE_AGENT_DISPATCH;
      break;
    case HSA_AGENT_INFO_MACHINE_MODEL:
      *static_cast<hsa_machine_model_t*>(value) = HSA_MACHINE_MODEL_LARGE;
      break;
    case HSA_AGENT_INFO_BASE_PROFILE_DEFAULT_FLOAT_ROUNDING_MODES:
    case HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE:
      // TODO: validate if this is true.
      *static_cast<hsa_default_float_rounding_mode_t*>(value) =
          HSA_DEFAULT_FLOAT_ROUNDING_MODE_NEAR;
      break;
    case HSA_AGENT_INFO_PROFILE:
      *static_cast<hsa_profile_t*>(value) = profile_;
      break;
    case HSA_AGENT_INFO_WAVEFRONT_SIZE:
      *static_cast<uint32_t*>(value) = 0;
      break;
    case HSA_AGENT_INFO_WORKGROUP_MAX_DIM:
      std::memset(value, 0, sizeof(uint16_t) * 3);
      break;
    case HSA_AGENT_INFO_WORKGROUP_MAX_SIZE:
      *static_cast<uint32_t*>(value) = 0;
      break;
    case HSA_AGENT_INFO_GRID_MAX_DIM:
      std::memset(value, 0, sizeof(uint16_t) * 3);
      break;
    case HSA_AGENT_INFO_GRID_MAX_SIZE:
      *static_cast<uint32_t*>(value) = 0;
      break;
    case HSA_AGENT_INFO_FBARRIER_MAX_SIZE:
      *static_cast<uint32_t*>(value) = 0;
      break;
    case HSA_AGENT_INFO_QUEUES_MAX:
      *static_cast<uint32_t*>(value) = max_queues_;
      break;
    case HSA_AGENT_INFO_QUEUE_MIN_SIZE:
      *static_cast<uint32_t*>(value) = min_aql_size_;
      break;
    case HSA_AGENT_INFO_QUEUE_MAX_SIZE:
      *static_cast<uint32_t*>(value) = max_aql_size_;
      break;
    case HSA_AGENT_INFO_QUEUE_TYPE:
      *static_cast<hsa_queue_type32_t*>(value) = HSA_QUEUE_TYPE_SINGLE;
      break;
    case HSA_AGENT_INFO_NODE:
      *static_cast<uint32_t*>(value) = node_id();
      break;
    case HSA_AGENT_INFO_DEVICE:
      *static_cast<hsa_device_type_t*>(value) = HSA_DEVICE_TYPE_AIE;
      break;
    case HSA_AGENT_INFO_CACHE_SIZE:
      *static_cast<uint32_t*>(value) = 0;
      break;
    case HSA_AGENT_INFO_VERSION_MAJOR:
      *static_cast<uint32_t*>(value) = 1;
      break;
    case HSA_AGENT_INFO_VERSION_MINOR:
      *static_cast<uint32_t*>(value) = 0;
      break;
    case HSA_AMD_AGENT_INFO_CHIP_ID:
      *static_cast<uint32_t*>(value) = 0;
      break;
    case HSA_AMD_AGENT_INFO_CACHELINE_SIZE:
      *static_cast<uint32_t*>(value) = 0;
      break;
    case HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT:
      *static_cast<uint32_t*>(value) = 0;
      break;
    case HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY:
      *static_cast<uint32_t*>(value) = 0;
      break;
    case HSA_AMD_AGENT_INFO_DRIVER_NODE_ID:
      *static_cast<uint32_t*>(value) = node_id();
      break;
    case HSA_AMD_AGENT_INFO_MAX_ADDRESS_WATCH_POINTS:
      *static_cast<uint32_t*>(value) = 0;
      break;
    case HSA_AMD_AGENT_INFO_BDFID:
      *static_cast<uint32_t*>(value) = 0;
      break;
    case HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU:
      *static_cast<uint32_t*>(value) = 0;
      break;
    case HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES:
      *static_cast<uint32_t*>(value) = 0;
      break;
    case HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE:
      *static_cast<uint32_t*>(value) = 0;
      break;
    case HSA_EXT_AGENT_INFO_IMAGE_1D_MAX_ELEMENTS:
    case HSA_EXT_AGENT_INFO_IMAGE_1DA_MAX_ELEMENTS:
    case HSA_EXT_AGENT_INFO_IMAGE_1DB_MAX_ELEMENTS:
    case HSA_EXT_AGENT_INFO_IMAGE_2D_MAX_ELEMENTS:
    case HSA_EXT_AGENT_INFO_IMAGE_2DA_MAX_ELEMENTS:
    case HSA_EXT_AGENT_INFO_IMAGE_2DDEPTH_MAX_ELEMENTS:
    case HSA_EXT_AGENT_INFO_IMAGE_2DADEPTH_MAX_ELEMENTS:
    case HSA_EXT_AGENT_INFO_IMAGE_3D_MAX_ELEMENTS:
    case HSA_EXT_AGENT_INFO_IMAGE_ARRAY_MAX_LAYERS:
      *static_cast<uint32_t*>(value) = 0;
      break;
    case HSA_AMD_AGENT_INFO_PRODUCT_NAME:
      // Copy MarketingName which is 7-bit ASCII stored in UTF-16 array
      std::copy_n(node_props_.MarketingName, HSA_PUBLIC_NAME_SIZE, static_cast<char*>(value));
      break;
    case HSA_AMD_AGENT_INFO_UUID: {
      // At this point AIE devices do not support UUID's.
      constexpr std::string_view uuid = "AIE-XX";
      auto ptr = static_cast<char*>(value);
      std::copy(uuid.begin(), uuid.end(), ptr);
      *std::next(ptr, uuid.size()) = '\0';
      break;
    }
    case HSA_AMD_AGENT_INFO_ASIC_REVISION:
      *static_cast<uint32_t*>(value) = 0;
      break;
    case HSA_AMD_AGENT_INFO_SVM_DIRECT_HOST_ACCESS:
      assert(regions_.size() != 0 && "No device local memory found!");
      *static_cast<bool*>(value) = true;
      break;
    case HSA_AMD_AGENT_INFO_MEMORY_PROPERTIES:
      std::memset(value, 0, sizeof(uint8_t) * 8);
      break;
    case HSA_AMD_AGENT_INFO_CLOCK_COUNTERS:
      std::memset(value, 0, sizeof(hsa_amd_clock_counters_t));
      break;
    default:
      *static_cast<uint32_t*>(value) = 0;
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t AieAgent::QueueCreate(size_t size, hsa_queue_type32_t queue_type, uint64_t flags,
                                   core::HsaEventCallback event_callback, void* data,
                                   uint32_t private_segment_size, uint32_t group_segment_size,
                                   core::Queue** queue) {
  if ((flags & HSA_AMD_QUEUE_CREATE_DEVICE_MEM_RING_BUF) != 0 ||
      (flags & HSA_AMD_QUEUE_CREATE_DEVICE_MEM_QUEUE_DESCRIPTOR) != 0) {
    // AIE agents do not currently support queue creation in device memory.
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  if (!IsPowerOfTwo(size)) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  if (size < min_aql_size_ || size > max_aql_size_) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  core::SharedQueue* shared_queue =
      static_cast<core::SharedQueue*>(core::Runtime::runtime_singleton_->system_allocator()(
          sizeof(core::SharedQueue), MemoryRegion::GetPageSize(), 0, node_id()));

  if (!shared_queue) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  auto aql_queue(new AieAqlQueue(shared_queue, this, size, node_id(), flags));
  if (aql_queue == nullptr) {
    core::Runtime::runtime_singleton_->system_deallocator()(shared_queue);
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  *queue = aql_queue;

  return HSA_STATUS_SUCCESS;
}

void AieAgent::InitRegionList() {
  /// AIE itself currently has no memory regions of its own, all memory is just the system DRAM.
  const uint64_t total_system_memory = os::HostTotalPhysicalMemory();

  /// For allocating kernel arguments or other objects that only need
  /// system memory.
  HsaMemoryProperties sys_mem_props = {};
  sys_mem_props.HeapType = HSA_HEAPTYPE_SYSTEM;
  sys_mem_props.SizeInBytes = total_system_memory;

  /// For any other allocation, e.g., buffers.
  HsaMemoryProperties other_mem_props = {};
  other_mem_props.HeapType = HSA_HEAPTYPE_SYSTEM;
  other_mem_props.SizeInBytes = total_system_memory;

  /// For allocating memory for programmable device image (PDI) files. These
  /// need to be mapped to the device so the hardware can access the PDIs.
  HsaMemoryProperties dev_mem_props = {};
  dev_mem_props.HeapType = HSA_HEAPTYPE_DEVICE_SVM;
  dev_mem_props.SizeInBytes = XdnaDriver::GetDevHeapByteSize();

  /// As of now the AIE devices support coarse-grain memory regions that require
  /// explicit sync operations.
  regions_.reserve(3);
  regions_.push_back(
    std::make_shared<MemoryRegion>(false, true, false, false, true, this, sys_mem_props));
  regions_.push_back(
    std::make_shared<MemoryRegion>(false, false, false, false, true, this, dev_mem_props));
  regions_.push_back(
    std::make_shared<MemoryRegion>(false, false, false, false, true, this, other_mem_props));
}

void AieAgent::InitAllocators() {
  for (const auto& region : regions()) {
    const MemoryRegion *amd_mem_region(
        static_cast<const MemoryRegion *>(region.get()));
    if (amd_mem_region->kernarg()) {
      const core::MemoryRegion* region_ptr = region.get();
      system_allocator_ =
          [region_ptr](size_t size, size_t align,
                   core::MemoryRegion::AllocateFlags alloc_flags) -> void * {
        void *mem(nullptr);
        return (core::Runtime::runtime_singleton_->AllocateMemory(
                    region_ptr, size, alloc_flags, &mem) == HSA_STATUS_SUCCESS)
                   ? mem
                   : nullptr;
      };

      system_deallocator_ = [](void* ptr) { core::Runtime::runtime_singleton_->FreeMemory(ptr); };
      break;
    }
  }
}

} // namespace AMD
} // namespace rocr
