////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2024, Advanced Micro Devices, Inc. All rights reserved.
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

#include "core/inc/amd_topology.h"

#include <algorithm>
#include <cstring>
#include <functional>

#ifndef NDEBUG
#include <iostream>
#endif

#include <array>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <link.h>

#include "core/inc/amd_aie_agent.h"
#include "core/inc/amd_available_drivers.h"
#include "core/inc/amd_cpu_agent.h"
#include "core/inc/amd_filter_device.h"
#include "core/inc/amd_gpu_agent.h"
#include "core/inc/amd_memory_region.h"
#include "core/inc/runtime.h"
#include "core/util/utils.h"

extern r_debug _amdgpu_r_debug;

namespace rocr {
namespace AMD {
// Anonymous namespace.
namespace {
#if _WIN32
constexpr size_t num_drivers = 0;
#elif __linux__
constexpr size_t num_drivers = 2;
#endif

const std::array<std::function<hsa_status_t(std::unique_ptr<core::Driver>&)>, num_drivers>
    discover_driver_funcs = {
#ifdef __linux__
        KfdDriver::DiscoverDriver, XdnaDriver::DiscoverDriver
#endif
};

void DiscoverDrivers() {
  for (const auto& discover_driver_fn : discover_driver_funcs) {
    std::unique_ptr<core::Driver> driver;
    hsa_status_t ret = discover_driver_fn(driver);

    if (ret != HSA_STATUS_SUCCESS) continue;

    core::Runtime::runtime_singleton_->RegisterDriver(std::move(driver));
  }
}

bool InitializeDriver(std::unique_ptr<core::Driver>& driver) {
  MAKE_NAMED_SCOPE_GUARD(driver_guard, [&]() { driver->Close(); });

  if (driver->Init() != HSA_STATUS_SUCCESS) {
    return false;
  }

  driver_guard.Dismiss();
  return true;
}

void DiscoverCpu(HSAuint32 node_id, HsaNodeProperties& node_prop) {
  CpuAgent* cpu = new CpuAgent(node_id, node_prop);
  cpu->Enable();
  core::Runtime::runtime_singleton_->RegisterAgent(cpu, true);
}

GpuAgent* DiscoverGpu(HSAuint32 node_id, HsaNodeProperties& node_prop, bool xnack_mode,
                      bool enabled) {
  GpuAgent* gpu = nullptr;
  if (node_prop.NumFComputeCores == 0) {
      // Ignore non GPUs.
      return nullptr;
  }
  try {
    gpu = new GpuAgent(node_id, node_prop, xnack_mode,
                       core::Runtime::runtime_singleton_->gpu_agents().size());

    const HsaVersionInfo& kfd_version = core::Runtime::runtime_singleton_->KfdVersion().version;

    // Check for sramecc incompatibility due to sramecc not being reported correctly in kfd before
    // 1.4.
    if (gpu->supported_isas()[0]->IsSrameccSupported() &&
         (kfd_version.KernelInterfaceMajorVersion <= 1 &&
              kfd_version.KernelInterfaceMinorVersion < 4)) {
      // gfx906 has both sramecc modes in use.  Suppress the device.
      if ((gpu->supported_isas()[0]->GetProcessorName() == "gfx906") &&
          core::Runtime::runtime_singleton_->flag().check_sramecc_validity()) {
        char name[64];
        gpu->GetInfo((hsa_agent_info_t)HSA_AMD_AGENT_INFO_PRODUCT_NAME, name);
        name[63] = '\0';
        fprintf(stderr,
                "HSA Error:  Incompatible kernel and userspace, %s disabled. Upgrade amdgpu.\n",
                name);
        delete gpu;
        return nullptr;
      }

      // gfx908 always has sramecc set to on in vbios.  Set mode bit to on and recreate the device.
      if (gpu->supported_isas()[0]->GetProcessorName() == "gfx908") {
        node_prop.Capability.ui32.SRAM_EDCSupport = 1;
        delete gpu;
        gpu = new GpuAgent(node_id, node_prop, xnack_mode,
                           core::Runtime::runtime_singleton_->gpu_agents().size());
      }
    }
  } catch (const hsa_exception& e) {
    if(e.error_code() == HSA_STATUS_ERROR_INVALID_ISA) {
      ifdebug {
        if (!strIsEmpty(e.what())) debug_print("Warning: %s\n", e.what());
      }
      // Ignore unrecognized GPUs.
      return nullptr;
    } else {
      // Rethrow remaining exceptions.
      throw;
    }
  }
  if (enabled) gpu->Enable();
  core::Runtime::runtime_singleton_->RegisterAgent(gpu, enabled);
  return gpu;
}

void DiscoverAie(uint32_t node_id, HsaNodeProperties& node_prop) {
  AieAgent* aie = new AieAgent(node_id);
  core::Runtime::runtime_singleton_->RegisterAgent(aie, true);
}

void RegisterLinkInfo(const std::unique_ptr<core::Driver>& driver, uint32_t node_id,
                      uint32_t num_link) {
  // Register connectivity links for this agent to the runtime.
  if (num_link == 0) {
    return;
  }

  std::vector<HsaIoLinkProperties> links(num_link);
  if (HSA_STATUS_SUCCESS != driver->GetEdgeProperties(links, node_id)) {
    return;
  }

  for (HsaIoLinkProperties io_link : links) {
    // Populate link info with thunk property.
    hsa_amd_memory_pool_link_info_t link_info = {0};

    switch (io_link.IoLinkType) {
      case HSA_IOLINKTYPE_HYPERTRANSPORT:
        link_info.link_type = HSA_AMD_LINK_INFO_TYPE_HYPERTRANSPORT;
        link_info.atomic_support_32bit = true;
        link_info.atomic_support_64bit = true;
        link_info.coherent_support = true;
        break;
      case HSA_IOLINKTYPE_PCIEXPRESS:
        link_info.link_type = HSA_AMD_LINK_INFO_TYPE_PCIE;
        link_info.atomic_support_32bit = true;
        link_info.atomic_support_64bit = true;
        link_info.coherent_support = true;
        break;
      case HSA_IOLINK_TYPE_QPI_1_1:
        link_info.link_type = HSA_AMD_LINK_INFO_TYPE_QPI;
        link_info.atomic_support_32bit = true;
        link_info.atomic_support_64bit = true;
        link_info.coherent_support = true;
        break;
      case HSA_IOLINK_TYPE_INFINIBAND:
        link_info.link_type = HSA_AMD_LINK_INFO_TYPE_INFINBAND;
        debug_print("IOLINK is missing atomic and coherency defaults.\n");
        break;
      case HSA_IOLINK_TYPE_XGMI:
        link_info.link_type = HSA_AMD_LINK_INFO_TYPE_XGMI;
        link_info.atomic_support_32bit = true;
        link_info.atomic_support_64bit = true;
        link_info.coherent_support = true;
        break;
      default:
        debug_print("Unrecognized IOLINK type.\n");
        break;
    }

    // KFD is reporting wrong override status for XGMI.  Disallow override for bringup.
    if (io_link.Flags.ui32.Override == 1) {
      if (io_link.Flags.ui32.NoPeerToPeerDMA == 1) {
        // Ignore this link since peer to peer is not allowed.
        continue;
      }
      link_info.atomic_support_32bit = (io_link.Flags.ui32.NoAtomics32bit == 0);
      link_info.atomic_support_64bit = (io_link.Flags.ui32.NoAtomics64bit == 0);
      link_info.coherent_support = (io_link.Flags.ui32.NonCoherent == 0);
    }

    link_info.max_bandwidth = io_link.MaximumBandwidth;
    link_info.max_latency = io_link.MaximumLatency;
    link_info.min_bandwidth = io_link.MinimumBandwidth;
    link_info.min_latency = io_link.MinimumLatency;
    link_info.numa_distance = io_link.Weight;

    core::Runtime::runtime_singleton_->RegisterLinkInfo(
        io_link.NodeFrom, io_link.NodeTo, io_link.Weight, io_link.RecSdmaEngIdMask, link_info);
  }
}

/**
 * Process the list of Gpus that are surfaced to user
 */
void SurfaceGpuList(std::vector<int32_t>& gpu_list, bool xnack_mode, bool enabled) {
  // Process user visible Gpu devices
  const int32_t invalidIdx = -1;
  int32_t list_sz = gpu_list.size();
  HsaNodeProperties node_prop = {0};
  const auto& gpu_driver = core::Runtime::runtime_singleton_->AgentDriver(core::DriverType::KFD);
  for (int32_t idx = 0; idx < list_sz; idx++) {
    if (gpu_list[idx] == invalidIdx) {
      break;
    }

    // Obtain properties of the node
    hsa_status_t ret = gpu_driver.GetNodeProperties(node_prop, gpu_list[idx]);
    assert(ret == HSA_STATUS_SUCCESS && "Error in getting Node Properties");

    // disable interrupt signal for DTIF platform
    if (core::Runtime::runtime_singleton_->flag().enable_dtif())
      core::g_use_interrupt_wait = false;

    // Instantiate a Gpu device. The IO links
    // of this node have already been registered
    assert((node_prop.NumFComputeCores != 0) && "Improper node used for GPU device discovery.");
    DiscoverGpu(gpu_list[idx], node_prop, xnack_mode, enabled);
  }
}

/// @brief Calls into the user-mode driver for each node to build the topology
/// of the system.
///
/// @details Topology information includes information about each node in the
/// topology graph, which includes agents, IO links, memory, and caches.
bool BuildTopology() {
  auto rt = core::Runtime::runtime_singleton_;
  std::unordered_map<core::DriverType, HsaSystemProperties> driver_sys_props;
  size_t link_count = 0;
  /// @todo Currently we can filter out GPU devices using the
  /// ROCR_VISIBLE_DEVICES environment variable. Eventually this
  /// should be updated to allow for filtering other agents like
  /// AIEs.
  RvdFilter rvdFilter;
  int32_t invalidIdx = -1;
  uint32_t visibleCnt = 0;
  std::vector<int32_t> gpu_usr_list;
  std::vector<int32_t> gpu_disabled;
  bool filter = RvdFilter::FilterDevices();

  // Get the system properties (i.e., node count) from each driver
  // then update the runtime's link count before traversing each
  // driver's individual nodes.
  for (const auto& driver : rt->AgentDrivers()) {
    driver->GetSystemProperties(driver_sys_props[driver->kernel_driver_type_]);

    if (!driver_sys_props[driver->kernel_driver_type_].NumNodes) continue;

    link_count += driver_sys_props[driver->kernel_driver_type_].NumNodes;
  }

  rt->SetLinkCount(link_count);

  // Traverse each driver's nodes and discover their agents.
  for (const auto& driver : core::Runtime::runtime_singleton_->AgentDrivers()) {
    if (driver_sys_props.find(driver->kernel_driver_type_) == driver_sys_props.end()) return false;

    const HsaSystemProperties& sys_props = driver_sys_props[driver->kernel_driver_type_];

    // Query if env ROCR_VISIBLE_DEVICES is defined. If defined
    // determine number and order of GPU devices to be surfaced.
    if (filter && driver->kernel_driver_type_ == core::DriverType::KFD) {
      rvdFilter.BuildRvdTokenList();
      rvdFilter.BuildDeviceUuidList(sys_props.NumNodes);
      visibleCnt = rvdFilter.BuildUsrDeviceList();
      for (int32_t idx = 0; idx < visibleCnt; idx++) {
        gpu_usr_list.push_back(invalidIdx);
      }
    }

    // Discover agents on every node in the platform.
    int32_t kfdIdx = 0;
    for (HSAuint32 node_id = 0; node_id < sys_props.NumNodes; node_id++) {
      HsaNodeProperties node_props = {0};
      if (driver->GetNodeProperties(node_props, node_id) != HSA_STATUS_SUCCESS) {
        return false;
      }

      if (node_props.NumCPUCores) {
        // Node has CPU cores so instantiate a CPU agent.
        DiscoverCpu(node_id, node_props);
      }

      if (node_props.NumNeuralCores) {
        // Node has AIE cores so instantiate an AIE agent.
        DiscoverAie(node_id, node_props);
      }

      // Current node is either a dGpu or Apu and might belong
      // to user visible list. Process node if present in usr
      // visible list, continue if not found
      if (node_props.NumFComputeCores != 0) {
        if (filter) {
          int32_t devRank = rvdFilter.GetUsrDeviceRank(kfdIdx);
          if (devRank != (-1)) {
            gpu_usr_list[devRank] = node_id;
          } else {
            gpu_disabled.push_back(node_id);
          }
        } else {
          gpu_usr_list.push_back(node_id);
        }
        kfdIdx++;
      }

      // Register IO links of node without regard to
      // it being visible to user or not. It is not
      // possible to access links of nodes that are
      // not visible
      RegisterLinkInfo(driver, node_id, node_props.NumIOLinks);
    }
  }

  // Instantiate ROCr objects to encapsulate Gpu devices
  SurfaceGpuList(gpu_usr_list, rt->XnackEnabled(), true);
  SurfaceGpuList(gpu_disabled, rt->XnackEnabled(), false);

  // Parse HSA_CU_MASK with GPU and CU count limits.
  uint32_t maxGpu = rt->gpu_agents().size();
  uint32_t maxCu = 0;
  uint32_t cus;
  for (auto& gpu : rt->gpu_agents()) {
    gpu->GetInfo((hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT, &cus);
    maxCu = Max(maxCu, cus);
  }
  const_cast<Flag&>(rt->flag()).parse_masks(maxGpu, maxCu);

  // Front load the rec_sdma_eng_id_mask to check whether needs to override old mask
  bool rec_sdma_engine_override = false;
  for (auto& src_gpu : rt->gpu_agents()) {
    uint32_t src_id = src_gpu->node_id();

    // Set RecSdmaEngOverride to true for all gpus
    if (rec_sdma_engine_override) {
      ((AMD::GpuAgent*)src_gpu)->SetRecSdmaEngOverride(rec_sdma_engine_override);
      continue;
    }

    // skip the pre-loop if NumSdmaXgmiEngines != 6
    if (((AMD::GpuAgent*)src_gpu)->properties().NumSdmaXgmiEngines != 6)
      break;

    for (auto& dst_gpu : rt->gpu_agents()) {
      uint32_t dst_id = dst_gpu->node_id();
      if (src_id != dst_id) {
        auto linfo = rt->GetLinkInfo(src_id, dst_id);
        if (IsPowerOfTwo(linfo.rec_sdma_eng_id_mask)) {
          rec_sdma_engine_override = true;
          ((AMD::GpuAgent*)src_gpu)->SetRecSdmaEngOverride(rec_sdma_engine_override);
          break;
        }
      }
    }
  }

  // Register destination agents that can SDMA gang copy for source agents
  for (auto& src_gpu : rt->gpu_agents()) {
    uint32_t src_id = src_gpu->node_id();
    for (auto& dst_gpu : rt->gpu_agents()) {
      uint32_t dst_id = dst_gpu->node_id();
      uint32_t gang_factor = 1, rec_sdma_eng_id_mask = 0;

      if (src_id != dst_id) {
        auto linfo = rt->GetLinkInfo(src_id, dst_id);
        // Ganging can only be done over xGMI and is either fixed or variable
        // based on topology information:
        // Weight of 13 - Intra-socket GPU link in multi-partition mode
        // Weigth of 15 - Direct GPU link in single partition mode
        // Weight of 41 - Inter-socket GPU link in multi-partition mode
        if (linfo.info.link_type == HSA_AMD_LINK_INFO_TYPE_XGMI) {
          // Temporary work-around, disable SDMA ganging on non-APUs in non-SPX modes
          // Check xGMI APU status
          const bool isXgmiApu = static_cast<AMD::GpuAgent*>(src_gpu)->is_xgmi_cpu_gpu();
          if (linfo.info.numa_distance == 13 || linfo.info.numa_distance == 41)
            gang_factor = isXgmiApu ? 2 : 1;
          else if (linfo.info.numa_distance == 15 && linfo.info.min_bandwidth)
            gang_factor = linfo.info.max_bandwidth/linfo.info.min_bandwidth;
          else gang_factor = 1;

          rec_sdma_eng_id_mask = linfo.rec_sdma_eng_id_mask;

          // Override the old mask if rec sdma eng verride is true
          // Using one pcie sdma for device to device copy with limited XGMI SDMA engine.
          // This will help improve all to all copy with limited XGMI SDMA engine.
          if (rec_sdma_engine_override) {
            uint32_t sdma_engine_mask = (1 << (((AMD::GpuAgent*)src_gpu)->properties().NumSdmaEngines - 1));
            rec_sdma_eng_id_mask = !IsPowerOfTwo(rec_sdma_eng_id_mask) ?
              sdma_engine_mask : rec_sdma_eng_id_mask;
          }
        }
      }

      // Register all GPUs regardless of connection type to take advantage of easy
      // key-value lookup later on.
      ((AMD::GpuAgent*)src_gpu)->RegisterGangPeer(*dst_gpu, gang_factor);
      ((AMD::GpuAgent*)src_gpu)->RegisterRecSdmaEngIdMaskPeer(*dst_gpu, rec_sdma_eng_id_mask);
    }
  }
  return true;
}
}  // Anonymous namespace

bool Load() {
  DiscoverDrivers();

  if (core::Runtime::runtime_singleton_->AgentDrivers().empty()) return false;

  for (auto& d : core::Runtime::runtime_singleton_->AgentDrivers()) {
    bool is_model_enabled = false;
    d->IsModelEnabled(&is_model_enabled);
    if (is_model_enabled) continue;
    if (!InitializeDriver(d)) return false;
  }

  return BuildTopology();
}

bool Unload() {
  for (auto& driver : core::Runtime::runtime_singleton_->AgentDrivers()) {
    hsa_status_t ret = driver->ShutDown();
    if (ret != HSA_STATUS_SUCCESS) return false;
  }

  return true;
}
}  // namespace amd
}  // namespace rocr
