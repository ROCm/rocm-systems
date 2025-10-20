/* Copyright (c) 2008 - 2025 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "cl.h"
#include "platform/program.hpp"
#include "platform/kernel.hpp"
#include "os/os.hpp"
#include "utils/debug.hpp"
#include "utils/flags.hpp"
#include "utils/options.hpp"
#include "utils/versions.hpp"
#include "thread/monitor.hpp"
#include "CL/cl_ext.h"

#include "vdi_common.hpp"
#include "device/comgrctx.hpp"
#include "device/devhostcall.hpp"
#include "device/rocm/rocdevice.hpp"
#include "device/rocm/rocblit.hpp"
#include "device/rocm/rocvirtual.hpp"
#include "device/rocm/rocprogram.hpp"
#include "device/rocm/rockernel.hpp"
#include "device/rocm/rocmemory.hpp"
#include "device/rocm/rocglinterop.hpp"
#include "device/rocm/rocsignal.hpp"
#include "platform/sampler.hpp"

#if defined(__clang__)
#if __has_feature(address_sanitizer)
#include "device/rocm/rocurilocator.hpp"
#endif
#endif

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#ifdef ROCCLR_SUPPORT_NUMA_POLICY
#include <numa.h>
#include <numaif.h>
#endif  // ROCCLR_SUPPORT_NUMA_POLICY
#include <sstream>
#include <vector>

#define OPENCL_VERSION_STR XSTR(OPENCL_MAJOR) "." XSTR(OPENCL_MINOR)
#define OPENCL_C_VERSION_STR XSTR(OPENCL_C_MAJOR) "." XSTR(OPENCL_C_MINOR)


static_assert(static_cast<uint32_t>(amd::Device::VmmAccess::kNone) ==
                  static_cast<uint32_t>(HSA_ACCESS_PERMISSION_NONE),
              "Vmm Access Flag None mismatch with ROC-runtime!");
static_assert(static_cast<uint32_t>(amd::Device::VmmAccess::kReadOnly) ==
                  static_cast<uint32_t>(HSA_ACCESS_PERMISSION_RO),
              "Vmm Access Flag Read mismatch with ROCr-runtime!");
static_assert(static_cast<uint32_t>(amd::Device::VmmAccess::kReadWrite) ==
                  static_cast<uint32_t>(HSA_ACCESS_PERMISSION_RW),
              "Vmm Access Flag Read Write mismatch with ROC-runtime!");


namespace amd::device {
extern const char* HipExtraSourceCode;
extern const char* HipExtraSourceCodeNoGWS;
}  // namespace amd::device

namespace amd::roc {
bool roc::Device::isHsaInitialized_ = false;
std::vector<hsa_agent_t> roc::Device::gpu_agents_;
std::vector<AgentInfo> roc::Device::cpu_agents_;

address Device::mg_sync_ = nullptr;

bool NullDevice::create(const amd::Isa& isa) {
  if (!isa.runtimeRocSupported()) {
    LogPrintfError("Offline HSA device %s is not supported", isa.targetId());
    return false;
  }

  online_ = false;
  // Mark the device as GPU type
  info_.type_ = CL_DEVICE_TYPE_GPU;
  info_.vendorId_ = 0x1002;

  roc::Settings* hsaSettings = new roc::Settings();
  settings_ = hsaSettings;
  if (!hsaSettings || !hsaSettings->create(false, isa, isa.xnack() == amd::Isa::Feature::Enabled)) {
    LogPrintfError("Error creating settings for offline HSA device %s", isa.targetId());
    return false;
  }

  if (!ValidateComgr()) {
    LogPrintfError("Code object manager initialization failed for offline HSA device %s",
                   isa.targetId());
    return false;
  }

  if (!amd::Device::create(isa)) {
    LogPrintfError("Unable to setup offline HSA device %s", isa.targetId());
    return false;
  }

  // Report the device name
  ::strncpy(info_.name_, isa.targetId(), sizeof(info_.name_) - 1);
  info_.extensions_ = getExtensionString();
  info_.maxWorkGroupSize_ = hsaSettings->maxWorkGroupSize_;
  ::strncpy(info_.vendor_, "Advanced Micro Devices, Inc.", sizeof(info_.vendor_) - 1);
  info_.oclcVersion_ = "OpenCL C " OPENCL_C_VERSION_STR " ";
  info_.spirVersions_ = "";
  std::stringstream ss;
  ss << AMD_BUILD_STRING " (HSA," << (settings().useLightning_ ? "LC" : "HSAIL");
  ss << ") [Offline]";
  ::strncpy(info_.driverVersion_, ss.str().c_str(), sizeof(info_.driverVersion_) - 1);
  info_.version_ = "OpenCL " OPENCL_VERSION_STR " ";
  return true;
}

Device::Device(hsa_agent_t bkendDevice)
    : mapCacheOps_(nullptr),
      mapCache_(nullptr),
      bkendDevice_(bkendDevice),
      pciDeviceId_(0),
      gpuvm_segment_max_alloc_(0),
      alloc_granularity_(0),
      xferQueue_(nullptr),
      freeMem_(0),
      vgpusAccess_(true) /* Virtual GPU List Ops Lock */
      ,
      hsa_exclusive_gpu_access_(false),
      queuePool_(QueuePriority::Total),
      coopHostcallBuffer_(nullptr),
      queueWithCUMaskPool_(QueuePriority::Total),
      numOfVgpus_(0),
      preferred_numa_node_(0),
      maxSdmaReadMask_(0),
      maxSdmaWriteMask_(0),
      cpu_agent_info_(nullptr) {
  group_segment_.handle = 0;
  gpuvm_segment_.handle = 0;
  gpu_fine_grained_segment_.handle = 0;
  gpu_ext_fine_grained_segment_.handle = 0;
  prefetch_signal_.handle = 0;
  isXgmi_ = false;
  cache_state_ = Device::CacheState::kCacheStateInvalid;
}

void Device::setupCpuAgent() {
  int32_t numaDistance = std::numeric_limits<int32_t>::max();
  uint32_t index = 0;  // 0 as default
  auto size = cpu_agents_.size();
  for (uint32_t i = 0; i < size; i++) {
    std::vector<amd::Device::LinkAttrType> link_attrs;
    link_attrs.push_back(std::make_pair(LinkAttribute::kLinkDistance, 0));
    if (findLinkInfo(cpu_agents_[i].fine_grain_pool, &link_attrs)) {
      if (link_attrs[0].second < numaDistance) {
        numaDistance = link_attrs[0].second;
        index = i;
      }
    }
  }
  std::vector<amd::Device::LinkAttrType> link_attrs;
  link_attrs.push_back(std::make_pair(LinkAttribute::kLinkLinkType, 0));
  if (findLinkInfo(cpu_agents_[0].fine_grain_pool, &link_attrs)) {
    isXgmi_ = (link_attrs[0].second == HSA_AMD_LINK_INFO_TYPE_XGMI);
  }

  preferred_numa_node_ = index;
  cpu_agent_info_ = &cpu_agents_[index];

  ClPrint(amd::LOG_INFO, amd::LOG_INIT,
          "Numa selects cpu agent[%zu]=0x%zx(fine=0x%zx,"
          "coarse=0x%zx) for gpu agent=0x%zx CPU<->GPU XGMI=%d",
          index, cpu_agent_info_->agent.handle, cpu_agent_info_->fine_grain_pool.handle,
          cpu_agent_info_->coarse_grain_pool.handle, bkendDevice_.handle, isXgmi_);
}

void Device::checkAtomicSupport() {
  std::vector<amd::Device::LinkAttrType> link_attrs;
  link_attrs.push_back(std::make_pair(LinkAttribute::kLinkAtomicSupport, 0));
  if (findLinkInfo(cpu_agent_info_->fine_grain_pool, &link_attrs)) {
    if (link_attrs[0].second == 1) {
      info_.pcie_atomics_ = true;
    }
  }
}

Device::~Device() {
  if (coopHostcallBuffer_) {
    amd::disableHostcalls(coopHostcallBuffer_);
    context().svmFree(coopHostcallBuffer_);
    coopHostcallBuffer_ = nullptr;
  }
  // Release cached map targets
  for (uint i = 0; mapCache_ != nullptr && i < mapCache_->size(); ++i) {
    if ((*mapCache_)[i] != nullptr) {
      (*mapCache_)[i]->release();
    }
  }
  delete mapCache_;
  delete mapCacheOps_;

  if (nullptr != p2p_stage_) {
    p2p_stage_->release();
    p2p_stage_ = nullptr;
  }
  if (nullptr != mg_sync_) {
    GlbCtx().svmFree(mg_sync_);
    mg_sync_ = nullptr;
  }
  if (glb_ctx_ != nullptr) {
    glb_ctx_->release();
    glb_ctx_ = nullptr;
  }

  for (auto& it : queuePool_) {
    for (auto qIter = it.begin(); qIter != it.end();) {
      hsa_queue_t* queue = qIter->first;
      auto& qInfo = qIter->second;
      if (qInfo.hostcallBuffer_) {
        ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_QUEUE,
                "Deleting hostcall buffer %p for hardware queue %p", qInfo.hostcallBuffer_,
                qIter->first->base_address);
        amd::disableHostcalls(qInfo.hostcallBuffer_);
        context().svmFree(qInfo.hostcallBuffer_);
      }
      ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_QUEUE, "Deleting hardware queue %p with refCount 0",
              queue->base_address);
      qIter = it.erase(qIter);
      Hsa::queue_destroy(queue);
    }
  }
  queuePool_.clear();

  // Destroy transfer queue
  delete xferQueue_;

  delete blitProgram_;

  if (context_ != nullptr) {
    context_->release();
  }

  delete[] p2p_agents_list_;

  if (0 != prefetch_signal_.handle) {
    Hsa::signal_destroy(prefetch_signal_);
  }
}

void NullDevice::tearDown() {}

bool NullDevice::init() {
  // Create offline devices for all ISAs not already associated with an online
  // device. This allows code objects to be compiled for all supported ISAs.
  std::vector<Device*> devices = getDevices(CL_DEVICE_TYPE_GPU, false);
  for (const amd::Isa* isa = amd::Isa::begin(); isa != amd::Isa::end(); isa++) {
    if (!isa->runtimeRocSupported()) {
      continue;
    }
    bool isOnline = false;
    // Check if the particular device is online
    for (size_t i = 0; i < devices.size(); i++) {
      if (&(devices[i]->isa()) == isa) {
        isOnline = true;
        break;
      }
    }
    if (isOnline) {
      continue;
    }
    std::unique_ptr<NullDevice> nullDevice(new NullDevice());
    if (!nullDevice) {
      LogPrintfError("Error allocating new instance of offline HSA device %s", isa->targetId());
      return false;
    }
    if (!nullDevice->create(*isa)) {
      LogPrintfError("Skipping creating new instance of offline HSA sevice %s", isa->targetId());
      continue;
    }
    nullDevice.release()->registerDevice();
  }
  return true;
}

NullDevice::~NullDevice() {}

hsa_status_t Device::iterateAgentCallback(hsa_agent_t agent, void* data) {
  hsa_device_type_t dev_type = HSA_DEVICE_TYPE_CPU;

  hsa_status_t stat = Hsa::agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &dev_type);

  if (stat != HSA_STATUS_SUCCESS) {
    LogPrintfError("HSA_AGENT_INFO_DEVICE failed with %x", stat);
    return stat;
  }

  if (dev_type == HSA_DEVICE_TYPE_CPU) {
    AgentInfo info = {agent, {0}, {0}, {0}};
    stat = Hsa::agent_iterate_memory_pools(agent, Device::iterateCpuMemoryPoolCallback,
                                           reinterpret_cast<void*>(&info));
    if (stat == HSA_STATUS_SUCCESS) {
      cpu_agents_.push_back(info);
    }
  } else if (dev_type == HSA_DEVICE_TYPE_GPU) {
    gpu_agents_.push_back(agent);
  }

  return stat;
}

hsa_ven_amd_loader_1_00_pfn_t Device::amd_loader_ext_table = {nullptr};

hsa_status_t Device::loaderQueryHostAddress(const void* device, const void** host) {
  return amd_loader_ext_table.hsa_ven_amd_loader_query_host_address
             ? amd_loader_ext_table.hsa_ven_amd_loader_query_host_address(device, host)
             : HSA_STATUS_ERROR;
}

// ================================================================================================
bool Device::init() {
  if (!Hsa::LoadLib()) {
    LogPrintfWarning("Failed to load rocr library!");
    return false;
  }

  hsa_status_t status = Hsa::init();

  // If there are no GPUs available, hsa_init will fail with HSA_STATUS_ERROR_OUT_OF_RESOURCES
  // but for NoGpu tests to pass, true needs to be returned
  constexpr bool kNoOfflineDevices = false;
  std::vector<amd::Device*> devices = getDevices(CL_DEVICE_TYPE_GPU, kNoOfflineDevices);
  if (status == HSA_STATUS_ERROR_OUT_OF_RESOURCES && devices.size() == 0) {
    return true;
  }

  if (status != HSA_STATUS_SUCCESS) {
    LogPrintfError("hsa_init failed with %x", status);
    return false;
  }

  Hsa::system_get_major_extension_table(HSA_EXTENSION_AMD_LOADER, 1, sizeof(amd_loader_ext_table),
                                        &amd_loader_ext_table);

  status = Hsa::iterate_agents(iterateAgentCallback, nullptr);
  if (status != HSA_STATUS_SUCCESS) {
    LogPrintfError("hsa_iterate_agents failed with %x", status);
    return false;
  }

  std::string ordinals =
      amd::IS_HIP ? ((HIP_VISIBLE_DEVICES[0] != '\0') ? HIP_VISIBLE_DEVICES : CUDA_VISIBLE_DEVICES)
                  : GPU_DEVICE_ORDINAL;
  if (ordinals[0] != '\0') {
    size_t pos = 0;
    std::vector<hsa_agent_t> valid_agents;
    std::set<size_t> valid_indexes;
    do {
      size_t end;
      bool deviceIdValid = true;
      end = ordinals.find_first_of(',', pos);
      if (end == std::string::npos) {
        end = ordinals.size();
      }
      std::string str_id = ordinals.substr(pos, end - pos);
      // If Uuid is specified, then convert it to index
      // Uuid is an Ascii string with a maximum of 21 chars including NULL
      // The string value is in the format GPU-<body>, <body> encodes UUID as a 16 chars hex
      if (str_id.find("GPU-") != std::string::npos) {
        for (int i = 0; i < gpu_agents_.size(); i++) {
          auto agent = gpu_agents_[i];
          char unique_id[32] = {0};
          if (HSA_STATUS_SUCCESS ==
              Hsa::agent_get_info(agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_UUID),
                                  unique_id)) {
            if (std::string(unique_id).find(str_id) != std::string::npos) {
              str_id = std::to_string(i);
              break;
            }
          }
        }
      }
      int index = atoi(str_id.c_str());
      if (index < 0 || static_cast<size_t>(index) >= gpu_agents_.size() ||
          str_id != std::to_string(index)) {
        deviceIdValid = false;
      }

      if (!deviceIdValid) {
        // Exit the loop as anything to the right of invalid deviceId
        // has to be discarded
        break;
      } else {
        if (valid_indexes.find(index) == valid_indexes.end()) {
          valid_agents.push_back(gpu_agents_[index]);
          valid_indexes.insert(index);
        }
      }
      pos = end + 1;
    } while (pos < ordinals.size());
    gpu_agents_ = valid_agents;
  }

  LogPrintfInfo("Initalizing runtime stack, Enumerated GPU agents = %lu", gpu_agents_.size());

  for (auto agent : gpu_agents_) {
    std::unique_ptr<Device> roc_device(new Device(agent));
    if (!roc_device) {
      LogError("Error creating new instance of Device on then heap.");
      continue;
    }

    if (!roc_device->create()) {
      LogError("Error creating new instance of Device.");
      continue;
    }

    // Setup System Memory to be Non-Coherent per user
    // request via environment variable. By default the
    // System Memory is setup to be Coherent
    if (roc_device->settings().enableNCMode_) {
      hsa_status_t err = Hsa::coherency_set_type(agent, HSA_AMD_COHERENCY_TYPE_NONCOHERENT);
      if (err != HSA_STATUS_SUCCESS) {
        LogError("Unable to set NC memory policy!");
        continue;
      }
    }

    // Check to see if a global CU mask is requested
    if (amd::IS_HIP && ROC_GLOBAL_CU_MASK[0] != '\0') {
      roc_device->getGlobalCUMask(ROC_GLOBAL_CU_MASK);
    }

    roc_device.release()->registerDevice();
  }

  // Query active devices only
  devices = getDevices(CL_DEVICE_TYPE_GPU, kNoOfflineDevices);
  if (devices.size() > 0) {
    bool p2p_available = false;
    // Loop through all available devices
    for (auto device1 : devices) {
      // Find all agents that can have access to the current device
      for (auto agent : static_cast<Device*>(device1)->p2pAgents()) {
        // Find cl_device_id associated with the current agent
        for (auto device2 : devices) {
          if (agent.handle == static_cast<Device*>(device2)->getBackendDevice().handle) {
            // Device2 can have access to device1
            device2->p2pDevices_.push_back(as_cl(device1));
            device1->p2p_access_devices_.push_back(device2);
            p2p_available = true;
          }
        }
      }
    }

    // Create a dummy context for internal memory allocations on all reported devices
    glb_ctx_ = new amd::Context(devices, amd::Context::Info());
    if (glb_ctx_ == nullptr) {
      LogError("glb_ctx failed");
      return false;
    }

    // Allocate a staging buffer for P2P emulation path
    if ((devices.size() >= 1) && !p2p_available) {
      amd::Buffer* buf =
          new (*glb_ctx_) amd::Buffer(*glb_ctx_, CL_MEM_ALLOC_HOST_PTR, kP2PStagingSize);
      if ((buf != nullptr) && buf->create()) {
        p2p_stage_ = buf;
      } else {
        delete buf;
        LogError("p2p stg buffer alloc failed");
        return false;
      }
    }

    // Allocate mgpu sync buffer for cooperative launches
    if (amd::IS_HIP) {
      mg_sync_ = reinterpret_cast<address>(
          glb_ctx_->svmAlloc(kMGInfoSizePerDevice * devices.size(), kMGInfoSizePerDevice,
                             (CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS)));
      if (mg_sync_ == nullptr) {
        LogError("mgpu sync buffer alloc failed");
        return false;
      }
    }
  }

  if (amd::IS_HIP) {
    RegisterBackendErrorCb();
  }

  return true;
}

extern const char* SchedulerSourceCode;

void Device::tearDown() {
  NullDevice::tearDown();
  Hsa::shut_down();
}

// ================================================================================================
bool Device::create() {
  char agent_name[64] = {0};
  if (HSA_STATUS_SUCCESS != Hsa::agent_get_info(bkendDevice_, HSA_AGENT_INFO_NAME, agent_name)) {
    LogError("Unable to get HSA device name");
    return false;
  }

  if (HSA_STATUS_SUCCESS != Hsa::agent_get_info(bkendDevice_,
                                                (hsa_agent_info_t)HSA_AMD_AGENT_INFO_CHIP_ID,
                                                &pciDeviceId_)) {
    LogPrintfError("Unable to get PCI ID of HSA device %s", agent_name);
    return false;
  }

  struct agent_isas_t {
    uint count;
    hsa_isa_t first_isa;
  } agent_isas = {0, {0}};
  if (HSA_STATUS_SUCCESS != Hsa::agent_iterate_isas(
                                bkendDevice_,
                                [](hsa_isa_t isa, void* data) {
                                  agent_isas_t* agent_isas = static_cast<agent_isas_t*>(data);
                                  if (agent_isas->count++ == 0) {
                                    agent_isas->first_isa = isa;
                                  }
                                  return HSA_STATUS_SUCCESS;
                                },
                                &agent_isas)) {
    LogPrintfError("Unable to iterate supported ISAs for HSA device %s (PCI ID %x)", agent_name,
                   pciDeviceId_);
    return false;
  }

  uint32_t isa_name_length = 0;
  if (HSA_STATUS_SUCCESS != Hsa::isa_get_info_alt(agent_isas.first_isa,
                                                  (hsa_isa_info_t)HSA_ISA_INFO_NAME_LENGTH,
                                                  &isa_name_length)) {
    LogPrintfError("Unable to get ISA name length for HSA device %s (PCI ID %x)", agent_name,
                   pciDeviceId_);
    return false;
  }

  std::vector<char> isa_name(isa_name_length + 1, '\0');
  if (HSA_STATUS_SUCCESS != Hsa::isa_get_info_alt(agent_isas.first_isa,
                                                  (hsa_isa_info_t)HSA_ISA_INFO_NAME,
                                                  isa_name.data())) {
    LogPrintfError("Unable to get ISA name for HSA device %s (PCI ID %x)", agent_name,
                   pciDeviceId_);
    return false;
  }

  const amd::Isa* isa = amd::Isa::findIsa(isa_name.data());
  if (!isa || !isa->runtimeRocSupported()) {
    LogPrintfError("Unsupported HSA device %s (PCI ID %x) for ISA %s", agent_name, pciDeviceId_,
                   isa_name.data());
    return false;
  }

  if (HSA_STATUS_SUCCESS !=
      Hsa::agent_get_info(bkendDevice_, HSA_AGENT_INFO_PROFILE, &agent_profile_)) {
    LogPrintfError("Unable to get profile for HSA device %s (PCI ID %x)", agent_name, pciDeviceId_);
    return false;
  }

  uint32_t coop_groups = 0;
  // Check cooperative groups for HIP only
  if (amd::IS_HIP &&
      (HSA_STATUS_SUCCESS !=
       Hsa::agent_get_info(bkendDevice_,
                           static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_COOPERATIVE_QUEUES),
                           &coop_groups))) {
    LogPrintfError(
        "Unable to determine if cooperative queues are supported for HSA device %s (PCI ID %x)",
        agent_name, pciDeviceId_);
    return false;
  }

  setupCpuAgent();

  // Get Agent HDP Flush Register Memory
  hsa_amd_hdp_flush_t hdpInfo;
  if (HSA_STATUS_SUCCESS !=
      Hsa::agent_get_info(bkendDevice_, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_HDP_FLUSH),
                          &hdpInfo)) {
    LogPrintfError("Unable to determine HDP flush info for HSA device %s", agent_name);
    return false;
  }

  info_.hdpMemFlushCntl = hdpInfo.HDP_MEM_FLUSH_CNTL;
  info_.hdpRegFlushCntl = hdpInfo.HDP_REG_FLUSH_CNTL;
  bool hasValidHDPFlush = (info_.hdpMemFlushCntl != nullptr) && (info_.hdpRegFlushCntl != nullptr);

  // Create HSA settings
  assert(!settings_);
  roc::Settings* hsaSettings = new roc::Settings();
  settings_ = hsaSettings;
  if (!hsaSettings || !hsaSettings->create((agent_profile_ == HSA_PROFILE_FULL), *isa,
                                           isa->xnack() == amd::Isa::Feature::Enabled, coop_groups,
                                           isXgmi_, hasValidHDPFlush)) {
    LogPrintfError("Unable to create settings for HSA device %s (PCI ID %x)", agent_name,
                   pciDeviceId_);
    return false;
  }

  if (!ValidateComgr()) {
    LogPrintfError("Code object manager initialization failed for HSA device %s (PCI ID %x)",
                   agent_name, pciDeviceId_);
    return false;
  }

  if (!amd::Device::create(*isa)) {
    LogPrintfError("Unable to setup device for HSA device %s (PCI ID %x)", agent_name,
                   pciDeviceId_);
    return false;
  }

  uint32_t hsa_bdf_id = 0;
  if (HSA_STATUS_SUCCESS !=
      Hsa::agent_get_info(bkendDevice_, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_BDFID),
                          &hsa_bdf_id)) {
    LogPrintfError("Unable to determine BFD ID for HSA device %s (PCI ID %x)", agent_name,
                   pciDeviceId_);
    return false;
  }

  info_.deviceTopology_.pcie.type = CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD;
  info_.deviceTopology_.pcie.bus = (hsa_bdf_id & (0xFF << 8)) >> 8;
  info_.deviceTopology_.pcie.device = (hsa_bdf_id & (0x1F << 3)) >> 3;
  info_.deviceTopology_.pcie.function = (hsa_bdf_id & 0x07);
  uint32_t pci_domain_id = 0;
  if (HSA_STATUS_SUCCESS !=
      Hsa::agent_get_info(bkendDevice_, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DOMAIN),
                          &pci_domain_id)) {
    LogPrintfError("Unable to determine domain ID for HSA device %s (PCI ID %x)", agent_name,
                   pciDeviceId_);
    return false;
  }
  info_.pciDomainID = pci_domain_id;

  if (populateOCLDeviceConstants() == false) {
    LogPrintfError("populateOCLDeviceConstants failed for HSA device %s (PCI ID %x)", agent_name,
                   pciDeviceId_);
    return false;
  }
  hsaSettings->limit_blit_wg_ = info().maxComputeUnits_;
  if (!flagIsDefault(DEBUG_CLR_LIMIT_BLIT_WG)) {
    hsaSettings->limit_blit_wg_ = std::max(DEBUG_CLR_LIMIT_BLIT_WG, 0x1U);
  }
  amd::Context::Info info = {0};
  std::vector<amd::Device*> devices;
  devices.push_back(this);

  // Create a dummy context
  context_ = new amd::Context(devices, info);
  if (context_ == nullptr) {
    return false;
  }

  // Map Cache Lock
  mapCacheOps_ = new amd::Monitor(true);
  if (nullptr == mapCacheOps_) {
    return false;
  }

  mapCache_ = new std::vector<amd::Memory*>();
  if (mapCache_ == nullptr) {
    return false;
  }
  // Use just 1 entry by default for the map cache
  mapCache_->push_back(nullptr);

  // Create signal for HMM prefetch operation on device
  if (HSA_STATUS_SUCCESS != Hsa::signal_create(kInitSignalValueOne, 0, nullptr, &prefetch_signal_)) {
    return false;
  }

  if (AMD_LOG_LEVEL >= LOG_EXTRA_DEBUG) {
    uint8_t logMask[8] = {0};
    hsa_flag_set64(logMask, HSA_AMD_LOG_FLAG_BLIT_KERNEL_PKTS);
    Hsa::enable_logging(logMask, outFile);
  }

  return true;
}

// ================================================================================================
device::Program* NullDevice::createProgram(amd::Program& owner, amd::option::Options* options) {
  device::Program* program;
  if (settings().useLightning_) {
    program = new LightningProgram(*this, owner);
  } else {
    program = new HSAILProgram(*this, owner);
  }

  if (program == nullptr) {
    LogError("Memory allocation has failed!");
  }

  return program;
}

bool Device::createBlitProgram() {
  bool result = true;
  std::string extraKernel;

#if defined(USE_COMGR_LIBRARY)
  if (settings().useLightning_) {
    if (amd::IS_HIP) {
      if (settings().gwsInitSupported_) {
        extraKernel = device::HipExtraSourceCode;
      } else {
        extraKernel = device::HipExtraSourceCodeNoGWS;
      }
    } else {
      extraKernel = SchedulerSourceCode;
    }
  }
#endif  // USE_COMGR_LIBRARY

  blitProgram_ = new BlitProgram(context_);
  // Create blit programs
  if (blitProgram_ == nullptr || !blitProgram_->create(this, extraKernel, "")) {
    delete blitProgram_;
    blitProgram_ = nullptr;
    LogError("Couldn't create blit kernels!");
    return false;
  }

  return result;
}

device::Program* Device::createProgram(amd::Program& owner, amd::option::Options* options) {
  device::Program* program;
  if (settings().useLightning_) {
    program = new LightningProgram(*this, owner);
  } else {
    program = new HSAILProgram(*this, owner);
  }

  if (program == nullptr) {
    LogError("Memory allocation has failed!");
  }

  return program;
}

hsa_status_t Device::iterateGpuMemoryPoolCallback(hsa_amd_memory_pool_t pool, void* data) {
  if (data == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  hsa_region_segment_t segment_type = (hsa_region_segment_t)0;
  hsa_status_t stat =
      Hsa::memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment_type);
  if (stat != HSA_STATUS_SUCCESS) {
    return stat;
  }

  // TODO: system and device local segment
  Device* dev = reinterpret_cast<Device*>(data);
  switch (segment_type) {
    case HSA_REGION_SEGMENT_GLOBAL: {
      if (dev->settings().enableLocalMemory_) {
        uint32_t global_flag = 0;
        hsa_status_t stat =
            Hsa::memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &global_flag);
        if (stat != HSA_STATUS_SUCCESS) {
          return stat;
        }

        // If the flag set is ext scoped fine grain, break the loop
        if ((global_flag & HSA_REGION_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED) != 0) {
          dev->gpu_ext_fine_grained_segment_ = pool;
          break;
        }

        if ((global_flag & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED) != 0) {
          dev->gpu_fine_grained_segment_ = pool;
        } else if ((global_flag & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) != 0) {
          dev->gpuvm_segment_ = pool;

          // If cpu agent cannot access this pool, the device does not support large bar.
          hsa_amd_memory_pool_access_t tmp{};
          Hsa::agent_memory_pool_get_info(dev->cpu_agent_info_->agent, pool,
                                          HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &tmp);

          if (tmp == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
            dev->info_.largeBar_ = false;
          } else {
            dev->info_.largeBar_ = ROC_ENABLE_LARGE_BAR;
          }

          // Query the recommended granularity for this pool.
          stat = Hsa::memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE,
                                           &(dev->info_.virtualMemAllocGranularity_));
          if (stat != HSA_STATUS_SUCCESS) {
            LogPrintfError(
                "Cannot query HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE info"
                "failed with hsa_status: %d \n",
                stat);
          }
        }

        if (dev->gpuvm_segment_.handle == 0) {
          dev->gpuvm_segment_ = pool;
        }
      }
      break;
    }
    case HSA_REGION_SEGMENT_GROUP:
      dev->group_segment_ = pool;
      break;
    default:
      break;
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t Device::iterateCpuMemoryPoolCallback(hsa_amd_memory_pool_t pool, void* data) {
  if (data == nullptr) {
    LogError("CpuMemoryPoolCallback invalid args");
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  hsa_region_segment_t segment_type = (hsa_region_segment_t)0;
  hsa_status_t stat =
      Hsa::memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment_type);
  if (stat != HSA_STATUS_SUCCESS) {
    LogPrintfError("HSA_AMD_MEMORY_POOL_INFO_SEGMENT query failed with %x", stat);
    return stat;
  }
  AgentInfo* agentInfo = reinterpret_cast<AgentInfo*>(data);

  switch (segment_type) {
    case HSA_REGION_SEGMENT_GLOBAL: {
      uint32_t global_flag = 0;
      stat =
          Hsa::memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &global_flag);
      if (stat != HSA_STATUS_SUCCESS) {
        LogPrintfError("HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS query failed with %x", stat);
        break;
      }

      // If the flag set is ext scoped fine grain, break the loop
      if ((global_flag & HSA_REGION_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED) != 0) {
        agentInfo->ext_fine_grain_pool = pool;
        break;
      }

      if ((global_flag & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED) != 0) {
        if (agentInfo->fine_grain_pool.handle == 0) {
          agentInfo->fine_grain_pool = pool;
        } else if ((global_flag & HSA_REGION_GLOBAL_FLAG_KERNARG) == 0) {
          // If the fine_grain_pool was already filled, but kern_args flag was not set over-write.
          // This means this is region-1(fine_grain only), so over-write this with memory pool set
          // from "fine_grain and kern_args".
          agentInfo->fine_grain_pool = pool;
        }
        guarantee(((global_flag & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) == 0),
                  "Memory Segment cannot be both coarse and fine grained");
      } else {
        // If the flag is not set to fine grained, then it is coarse_grained by default.
        agentInfo->coarse_grain_pool = pool;
        guarantee(((global_flag & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) != 0),
                  "Memory Segments that are not fine grained has to be coarse grained");
        guarantee(((global_flag & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED) == 0),
                  "Memory Segment cannot be both coarse and fine grained");
        guarantee(((global_flag & HSA_REGION_GLOBAL_FLAG_KERNARG) == 0),
                  "Coarse grained memory segment cannot have kern_args tag");
      }

      if ((global_flag & HSA_REGION_GLOBAL_FLAG_KERNARG) != 0) {
        agentInfo->kern_arg_pool = pool;
        guarantee(((global_flag & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) == 0),
                  "Coarse grained memory segment cannot have kern_args tag");
      }

      break;
    }
    default:
      break;
  }

  return stat;
}

bool Device::createSampler(const amd::Sampler& owner, device::Sampler** sampler) const {
  *sampler = nullptr;
  Sampler* gpuSampler = new Sampler(*this);
  if ((nullptr == gpuSampler) || !gpuSampler->create(owner)) {
    delete gpuSampler;
    return false;
  }
  *sampler = gpuSampler;
  return true;
}

void Sampler::fillSampleDescriptor(hsa_ext_sampler_descriptor_v2_t& samplerDescriptor,
                                   const amd::Sampler& sampler) const {
  samplerDescriptor.filter_mode = sampler.filterMode() == CL_FILTER_NEAREST
                                      ? HSA_EXT_SAMPLER_FILTER_MODE_NEAREST
                                      : HSA_EXT_SAMPLER_FILTER_MODE_LINEAR;
  samplerDescriptor.coordinate_mode = sampler.normalizedCoords()
                                          ? HSA_EXT_SAMPLER_COORDINATE_MODE_NORMALIZED
                                          : HSA_EXT_SAMPLER_COORDINATE_MODE_UNNORMALIZED;
  for (int i = 0; i < 3; i++) {
    switch (sampler.addressingMode(i)) {
      case CL_ADDRESS_CLAMP_TO_EDGE:
        samplerDescriptor.address_modes[i] = HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_EDGE;
        break;
      case CL_ADDRESS_REPEAT:
        samplerDescriptor.address_modes[i] = HSA_EXT_SAMPLER_ADDRESSING_MODE_REPEAT;
        break;
      case CL_ADDRESS_CLAMP:
        samplerDescriptor.address_modes[i] = HSA_EXT_SAMPLER_ADDRESSING_MODE_CLAMP_TO_BORDER;
        break;
      case CL_ADDRESS_MIRRORED_REPEAT:
        samplerDescriptor.address_modes[i] = HSA_EXT_SAMPLER_ADDRESSING_MODE_MIRRORED_REPEAT;
        break;
      case CL_ADDRESS_NONE:
        samplerDescriptor.address_modes[i] = HSA_EXT_SAMPLER_ADDRESSING_MODE_UNDEFINED;
        break;
      default:
        return;
    }
  }
}

bool Sampler::create(const amd::Sampler& owner) {
  hsa_ext_sampler_descriptor_v2_t samplerDescriptor;
  fillSampleDescriptor(samplerDescriptor, owner);

  hsa_status_t status =
      Hsa::sampler_create(dev_.getBackendDevice(), &samplerDescriptor, &hsa_sampler);

  if (HSA_STATUS_SUCCESS != status) {
    DevLogPrintfError("Sampler creation failed with status: %d \n", status);
    return false;
  }

  hwSrd_ = hsa_sampler.handle;
  hwState_ = reinterpret_cast<address>(hsa_sampler.handle);

  return true;
}

Sampler::~Sampler() { Hsa::sampler_destroy(dev_.getBackendDevice(), hsa_sampler); }

Memory* Device::getGpuMemory(amd::Memory* mem) const {
  return static_cast<roc::Memory*>(mem->getDeviceMemory(*this));
}

const bool Device::isFineGrainSupported() const {
  bool result = (info().svmCapabilities_ & CL_DEVICE_SVM_ATOMICS) != 0 ? true : false;
  if (result) {
    if (gpu_fine_grained_segment_.handle != 0) {
      return true;
    }
  }
  return false;
}
// ================================================================================================
bool Device::populateOCLDeviceConstants() {
  info_.available_ = true;

  ::strncpy(info_.name_, isa().targetId(), sizeof(info_.name_) - 1);
  char device_name[64] = {0};
  if (HSA_STATUS_SUCCESS == Hsa::agent_get_info(bkendDevice_,
                                                (hsa_agent_info_t)HSA_AMD_AGENT_INFO_PRODUCT_NAME,
                                                device_name)) {
    ::strncpy(info_.boardName_, device_name, sizeof(info_.boardName_) - 1);
  }

  char unique_id[32] = {0};
  if (HSA_STATUS_SUCCESS ==
      Hsa::agent_get_info(bkendDevice_, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_UUID),
                         unique_id)) {
    // ROCr gives the UUID info in the format GPU-XXXX with length 20 bytes
    // Strip the first 4 bytes and store only the 16 bytes representing UUID
    for (size_t i = 0; i < 16; i++) {
      info_.uuid_[i] = unique_id[i + 4];
    }
  }
  if (HSA_STATUS_SUCCESS !=
      Hsa::agent_get_info(bkendDevice_,
                          (amd::IS_HIP)
                              ? (hsa_agent_info_t)HSA_AMD_AGENT_INFO_COOPERATIVE_COMPUTE_UNIT_COUNT
                              : (hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT,
                          &info_.maxComputeUnits_)) {
    return false;
  }
  assert(info_.maxComputeUnits_ > 0);

  info_.maxComputeUnits_ =
      settings().enableWgpMode_ ? info_.maxComputeUnits_ / 2 : info_.maxComputeUnits_;

  if (HSA_STATUS_SUCCESS !=
      Hsa::agent_get_info(bkendDevice_, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT,
                          &info_.maxPhysicalComputeUnits_)) {
    return false;
  }
  assert(info_.maxPhysicalComputeUnits_ > 0);

  info_.maxPhysicalComputeUnits_ = settings().enableWgpMode_ ? info_.maxPhysicalComputeUnits_ / 2
                                                             : info_.maxPhysicalComputeUnits_;

  if (HSA_STATUS_SUCCESS != Hsa::agent_get_info(bkendDevice_,
                                                (hsa_agent_info_t)HSA_AMD_AGENT_INFO_CACHELINE_SIZE,
                                                &info_.globalMemCacheLineSize_)) {
    return false;
  }
  info_.globalMemCacheLineSize_ =
      (info_.globalMemCacheLineSize_ != 0) ? info_.globalMemCacheLineSize_ : 64;

  uint32_t cachesize[4] = {0};
  if (HSA_STATUS_SUCCESS !=
      Hsa::agent_get_info(bkendDevice_, HSA_AGENT_INFO_CACHE_SIZE, cachesize)) {
    return false;
  }
  assert(cachesize[0] > 0);
  info_.globalMemCacheSize_ = cachesize[0];

  info_.globalMemCacheType_ = CL_READ_WRITE_CACHE;

  info_.type_ = CL_DEVICE_TYPE_GPU;

  info_.extensions_ = getExtensionString();
  info_.nativeVectorWidthDouble_ = info_.preferredVectorWidthDouble_ =
      (settings().doublePrecision_) ? 1 : 0;

  if (HSA_STATUS_SUCCESS !=
      Hsa::agent_get_info(bkendDevice_, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY,
                          &info_.maxEngineClockFrequency_)) {
    return false;
  }

  if (!(isa().versionMajor() == 9 && isa().versionMinor() == 0 && isa().versionStepping() == 2)) {
    if (info_.maxEngineClockFrequency_ <= 0) {
      LogError("maxEngineClockFrequency_ is NOT positive!");
    }
  }

  if (HSA_STATUS_SUCCESS !=
      Hsa::agent_get_info(bkendDevice_, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MEMORY_MAX_FREQUENCY,
                          &info_.maxMemoryClockFrequency_)) {
    return false;
  }

  uint64_t wallClockFrequency = 0;  // in Hz
  if (HSA_STATUS_SUCCESS !=
      Hsa::agent_get_info(bkendDevice_, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_TIMESTAMP_FREQUENCY,
                          &wallClockFrequency)) {
    LogWarning("HSA_AMD_AGENT_INFO_TIMESTAMP_FREQUENCY cannot be queried. Ignored!");
  }
  info_.wallClockFrequency_ = static_cast<uint32_t>(wallClockFrequency / 1000);  // in KHz

  if (HSA_STATUS_SUCCESS !=
      Hsa::agent_get_info(bkendDevice_,
                          static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_DRIVER_NODE_ID),
                          &info_.driverNodeId_)) {
    return false;
  }

  if (HSA_STATUS_SUCCESS !=
      Hsa::agent_get_info(bkendDevice_,
                          static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_NUM_SDMA_ENG),
                          &info_.numSDMAengines_)) {
    return false;
  }

  uint64_t scratchLimitMax = 0;
  if (HSA_STATUS_SUCCESS !=
      Hsa::agent_get_info(bkendDevice_, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_SCRATCH_LIMIT_MAX,
                          &scratchLimitMax)) {
    LogWarning("HSA_AMD_AGENT_INFO_SCRATCH_LIMIT_MAX cannot be queried!");
    return false;
  }
  info_.scratchLimitMin = 0;
  info_.scratchLimitMax = scratchLimitMax;

  checkAtomicSupport();

  assert(cpu_agent_info_->fine_grain_pool.handle != 0);
  if (HSA_STATUS_SUCCESS != Hsa::agent_iterate_memory_pools(
                                bkendDevice_, Device::iterateGpuMemoryPoolCallback, this)) {
    return false;
  }

  assert(group_segment_.handle != 0);

  for (auto agent : gpu_agents_) {
    if (agent.handle != bkendDevice_.handle) {
      hsa_status_t err;
      // Can another GPU (agent) have access to the current GPU memory pool (gpuvm_segment_)?
      hsa_amd_memory_pool_access_t access;
      err = Hsa::agent_memory_pool_get_info(agent, gpuvm_segment_,
                                            HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access);
      if (err != HSA_STATUS_SUCCESS) {
        continue;
      }

      // Find accessible p2p agents - i.e != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED
      if (HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT == access ||
          HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT == access) {
        // Agent can have access to the current gpuvm_segment_
        p2p_agents_.push_back(agent);
      }
    }
  }

  // Keep track of all P2P Agents in a Array including current device handle for IPC
  p2p_agents_list_ = new hsa_agent_t[1 + p2p_agents_.size()];
  p2p_agents_list_[0] = getBackendDevice();
  for (size_t agent_idx = 0; agent_idx < p2p_agents_.size(); ++agent_idx) {
    p2p_agents_list_[1 + agent_idx] = p2p_agents_[agent_idx];
  }

  size_t group_segment_size = 0;
  if (HSA_STATUS_SUCCESS != Hsa::memory_pool_get_info(group_segment_,
                                                      HSA_AMD_MEMORY_POOL_INFO_SIZE,
                                                      &group_segment_size)) {
    return false;
  }
  assert(group_segment_size > 0);

  // Find SDMA read mask
  if (HSA_STATUS_SUCCESS !=
      Hsa::memory_copy_engine_status(getCpuAgent(), getBackendDevice(), &maxSdmaReadMask_)) {
    return false;
  }
  assert(maxSdmaReadMask_ > 0 && "No SDMA engines available for Read");

  // Find SDMA write mask
  if (HSA_STATUS_SUCCESS !=
      Hsa::memory_copy_engine_status(getBackendDevice(), getCpuAgent(), &maxSdmaWriteMask_)) {
    return false;
  }
  assert(maxSdmaWriteMask_ > 0 && "No SDMA engines available for Write");

  info_.localMemSizePerCU_ = group_segment_size;
  info_.localMemSize_ = group_segment_size;

  info_.maxWorkItemDimensions_ = 3;

  uint8_t memory_properties[8];
  // Get the memory property from ROCr.
  if (HSA_STATUS_SUCCESS !=
      Hsa::agent_get_info(bkendDevice_, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_MEMORY_PROPERTIES,
                         memory_properties)) {
    LogError("HSA_AGENT_INFO_AMD_MEMORY_PROPERTIES query failed");
  }

  // Check if the device is APU
  if (hsa_flag_isset64(memory_properties, HSA_AMD_MEMORY_PROPERTY_AGENT_IS_APU)) {
    info_.hostUnifiedMemory_ = 1;
  }

  if (settings().enableLocalMemory_ && gpuvm_segment_.handle != 0) {
    size_t global_segment_size = 0;
    if (HSA_STATUS_SUCCESS != Hsa::memory_pool_get_info(gpuvm_segment_,
                                                        HSA_AMD_MEMORY_POOL_INFO_SIZE,
                                                        &global_segment_size)) {
      return false;
    }

    assert(global_segment_size > 0);
    info_.globalMemSize_ = (static_cast<uint64_t>(std::min(GPU_MAX_HEAP_SIZE, 100u)) *
                            static_cast<uint64_t>(global_segment_size)) /
                           100u;

    // For APU with vram size <= 512MiB, use a smaller single alloc percentage
    if (info_.globalMemSize_ <= 536870912) {
      if (flagIsDefault(GPU_SINGLE_ALLOC_PERCENT)) {
        GPU_SINGLE_ALLOC_PERCENT = 75;
      }
    }

    gpuvm_segment_max_alloc_ =
        uint64_t(info_.globalMemSize_ * std::min(GPU_SINGLE_ALLOC_PERCENT, 100u) / 100u);
    assert(gpuvm_segment_max_alloc_ > 0);

    info_.maxMemAllocSize_ = static_cast<uint64_t>(gpuvm_segment_max_alloc_);

    if (HSA_STATUS_SUCCESS !=
        Hsa::memory_pool_get_info(gpuvm_segment_, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE,
                                  &alloc_granularity_)) {
      return false;
    }

    assert(alloc_granularity_ > 0);
  } else {
    // We suppose half of physical memory can be used by GPU in APU system
    info_.globalMemSize_ = amd::Os::hostTotalPhysicalMemory() / 2;
    info_.globalMemSize_ = std::max(info_.globalMemSize_, uint64_t(1 * Gi));
    info_.globalMemSize_ = (static_cast<uint64_t>(std::min(GPU_MAX_HEAP_SIZE, 100u)) *
                            static_cast<uint64_t>(info_.globalMemSize_)) /
                           100u;

    info_.maxMemAllocSize_ =
        uint64_t(info_.globalMemSize_ * std::min(GPU_SINGLE_ALLOC_PERCENT, 100u) / 100u);

    if (HSA_STATUS_SUCCESS !=
        Hsa::memory_pool_get_info(cpu_agent_info_->fine_grain_pool,
                                  HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE,
                                  &alloc_granularity_)) {
      return false;
    }
  }

  freeMem_ = info_.globalMemSize_;

  // Make sure the max allocation size is not larger than the available memory size.
  info_.maxMemAllocSize_ = std::min(info_.maxMemAllocSize_, info_.globalMemSize_);
  info_.maxMemAllocSize_ = amd::alignDown(info_.maxMemAllocSize_, sizeof(uint64_t));

  // Maximum system memory allocation size allowed
  info_.maxPhysicalMemAllocSize_ = amd::Os::getPhysicalMemSize();

  // make sure we don't run anything over 8 params for now
  info_.maxParameterSize_ = 1024;

  uint32_t max_work_group_size = 0;
  if (HSA_STATUS_SUCCESS !=
      Hsa::agent_get_info(bkendDevice_, HSA_AGENT_INFO_WORKGROUP_MAX_SIZE, &max_work_group_size)) {
    return false;
  }
  assert(max_work_group_size > 0);
  max_work_group_size =
      std::min(max_work_group_size, static_cast<uint32_t>(settings().maxWorkGroupSize_));
  info_.maxWorkGroupSize_ = max_work_group_size;

  uint16_t max_workgroup_size[3] = {0, 0, 0};
  if (HSA_STATUS_SUCCESS !=
      Hsa::agent_get_info(bkendDevice_, HSA_AGENT_INFO_WORKGROUP_MAX_DIM, &max_workgroup_size)) {
    return false;
  }
  assert(max_workgroup_size[0] != 0 && max_workgroup_size[1] != 0 && max_workgroup_size[2] != 0);

  uint16_t max_work_item_size = static_cast<uint16_t>(max_work_group_size);
  info_.maxWorkItemSizes_[0] = std::min(max_workgroup_size[0], max_work_item_size);
  info_.maxWorkItemSizes_[1] = std::min(max_workgroup_size[1], max_work_item_size);
  info_.maxWorkItemSizes_[2] = std::min(max_workgroup_size[2], max_work_item_size);
  info_.preferredWorkGroupSize_ = settings().preferredWorkGroupSize_;

  info_.nativeVectorWidthChar_ = info_.preferredVectorWidthChar_ = 4;
  info_.nativeVectorWidthShort_ = info_.preferredVectorWidthShort_ = 2;
  info_.nativeVectorWidthInt_ = info_.preferredVectorWidthInt_ = 1;
  info_.nativeVectorWidthLong_ = info_.preferredVectorWidthLong_ = 1;
  info_.nativeVectorWidthFloat_ = info_.preferredVectorWidthFloat_ = 1;

  if (agent_profile_ == HSA_PROFILE_FULL) {  // full-profile = participating in coherent memory,
                                             // base-profile = NUMA based non-coherent memory
    info_.hostUnifiedMemory_ = 1;
    info_.iommuv2_ = true;
  }
  info_.memBaseAddrAlign_ = 8 * (flagIsDefault(MEMOBJ_BASE_ADDR_ALIGN) ? sizeof(int64_t[16]) * 2
                                                                       : MEMOBJ_BASE_ADDR_ALIGN);
  info_.minDataTypeAlignSize_ = sizeof(int64_t[16]);

  info_.maxConstantArgs_ = 8;
  info_.preferredConstantBufferSize_ = 16 * Ki;
  info_.maxConstantBufferSize_ = info_.maxMemAllocSize_;
  info_.localMemType_ = CL_LOCAL;
  info_.errorCorrectionSupport_ = false;
  info_.profilingTimerResolution_ = 1;
  info_.littleEndian_ = true;
  info_.compilerAvailable_ = true;
  info_.executionCapabilities_ = CL_EXEC_KERNEL;
  info_.queueProperties_ = CL_QUEUE_PROFILING_ENABLE;
  info_.platform_ = AMD_PLATFORM;
  info_.profile_ = "FULL_PROFILE";
  ::strncpy(info_.vendor_, "Advanced Micro Devices, Inc.", sizeof(info_.vendor_) - 1);

  info_.addressBits_ = LP64_SWITCH(32, 64);
  info_.maxSamplers_ = 16;
  info_.bufferFromImageSupport_ = false;
  info_.oclcVersion_ = "OpenCL C " OPENCL_C_VERSION_STR " ";
  info_.spirVersions_ = "";

  uint16_t major, minor;
  if (Hsa::agent_get_info(bkendDevice_, HSA_AGENT_INFO_VERSION_MAJOR, &major) !=
          HSA_STATUS_SUCCESS ||
      Hsa::agent_get_info(bkendDevice_, HSA_AGENT_INFO_VERSION_MINOR, &minor) !=
          HSA_STATUS_SUCCESS) {
    return false;
  }
  std::stringstream ss;
  ss << AMD_BUILD_STRING " (HSA" << major << "." << minor << ","
     << (settings().useLightning_ ? "LC" : "HSAIL");
  ss << ")";

  ::strncpy(info_.driverVersion_, ss.str().c_str(), sizeof(info_.driverVersion_) - 1);

  if (isa().versionMajor() >= 9) {
    info_.version_ =
        "OpenCL " /*OPENCL_VERSION_STR*/
        "2.0"
        " ";
  } else {
    info_.version_ =
        "OpenCL " /*OPENCL_VERSION_STR*/
        "1.2"
        " ";
  }

  info_.builtInKernels_ = "";
  info_.linkerAvailable_ = true;
  info_.preferredInteropUserSync_ = true;
  info_.printfBufferSize_ = PrintfDbg::WorkitemDebugSize * info().maxWorkGroupSize_;
  info_.vendorId_ = 0x1002;  // AMD's PCIe vendor id

  info_.maxGlobalVariableSize_ = static_cast<size_t>(info_.maxMemAllocSize_);
  info_.globalVariablePreferredTotalSize_ = static_cast<size_t>(info_.globalMemSize_);

  // Populate the single config setting.
  info_.singleFPConfig_ =
      CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO | CL_FP_ROUND_TO_INF | CL_FP_INF_NAN | CL_FP_FMA;

  if (settings().doublePrecision_) {
    info_.doubleFPConfig_ = info_.singleFPConfig_ | CL_FP_DENORM;
    info_.singleFPConfig_ |= CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT;
  }

  if (settings().singleFpDenorm_) {
    info_.singleFPConfig_ |= CL_FP_DENORM;
  }

  if (settings().checkExtension(ClKhrFp16)) {
    info_.halfFPConfig_ = info_.singleFPConfig_;
  }

  info_.preferredPlatformAtomicAlignment_ = 0;
  info_.preferredGlobalAtomicAlignment_ = 0;
  info_.preferredLocalAtomicAlignment_ = 0;

  uint8_t hsa_extensions[128];
  if (HSA_STATUS_SUCCESS !=
      Hsa::agent_get_info(bkendDevice_, HSA_AGENT_INFO_EXTENSIONS, hsa_extensions)) {
    return false;
  }

  assert(HSA_EXTENSION_IMAGES < 8);
  const bool image_is_supported = ((hsa_extensions[0] & (1 << HSA_EXTENSION_IMAGES)) != 0);
  if (image_is_supported) {
    // Images
    if (HSA_STATUS_SUCCESS !=
        Hsa::agent_get_info(bkendDevice_,
                            static_cast<hsa_agent_info_t>(HSA_EXT_AGENT_INFO_MAX_SAMPLER_HANDLERS),
                            &info_.maxSamplers_)) {
      return false;
    }

    if (HSA_STATUS_SUCCESS !=
        Hsa::agent_get_info(bkendDevice_,
                            static_cast<hsa_agent_info_t>(HSA_EXT_AGENT_INFO_MAX_IMAGE_RD_HANDLES),
                            &info_.maxReadImageArgs_)) {
      return false;
    }

    // TODO: no attribute for write image.
    info_.maxWriteImageArgs_ = 8;

    if (HSA_STATUS_SUCCESS !=
        Hsa::agent_get_info(bkendDevice_,
                            static_cast<hsa_agent_info_t>(HSA_EXT_AGENT_INFO_MAX_IMAGE_RORW_HANDLES),
                            &info_.maxReadWriteImageArgs_)) {
      return false;
    }

    uint32_t image_max_dim[3];
    if (HSA_STATUS_SUCCESS !=
        Hsa::agent_get_info(bkendDevice_,
                            static_cast<hsa_agent_info_t>(HSA_EXT_AGENT_INFO_IMAGE_2D_MAX_ELEMENTS),
                            &image_max_dim)) {
      return false;
    }

    info_.image2DMaxWidth_ = image_max_dim[0];
    info_.image2DMaxHeight_ = image_max_dim[1];

    if (HSA_STATUS_SUCCESS !=
        Hsa::agent_get_info(bkendDevice_,
                            static_cast<hsa_agent_info_t>(HSA_EXT_AGENT_INFO_IMAGE_3D_MAX_ELEMENTS),
                            &image_max_dim)) {
      return false;
    }

    info_.image3DMaxWidth_ = image_max_dim[0];
    info_.image3DMaxHeight_ = image_max_dim[1];
    info_.image3DMaxDepth_ = image_max_dim[2];

    uint32_t max_array_size = 0;
    if (HSA_STATUS_SUCCESS !=
        Hsa::agent_get_info(bkendDevice_,
                            static_cast<hsa_agent_info_t>(HSA_EXT_AGENT_INFO_IMAGE_ARRAY_MAX_LAYERS),
                            &max_array_size)) {
      return false;
    }

    info_.imageMaxArraySize_ = max_array_size;

    uint32_t max_image1da_width = 0;
    if (HSA_STATUS_SUCCESS !=
        Hsa::agent_get_info(bkendDevice_,
                            static_cast<hsa_agent_info_t>(HSA_EXT_AGENT_INFO_IMAGE_1DA_MAX_ELEMENTS),
                            &max_image1da_width)) {
      return false;
    }

    info_.image1DAMaxWidth_ = max_image1da_width;

    uint32_t max_image2da_width[2] = {0, 0};
    if (HSA_STATUS_SUCCESS !=
        Hsa::agent_get_info(bkendDevice_,
                            static_cast<hsa_agent_info_t>(HSA_EXT_AGENT_INFO_IMAGE_2DA_MAX_ELEMENTS),
                            &max_image2da_width)) {
      return false;
    }

    info_.image2DAMaxWidth_[0] = max_image2da_width[0];
    info_.image2DAMaxWidth_[1] = max_image2da_width[1];

    uint32_t max_image1d_width = 0;
    if (HSA_STATUS_SUCCESS !=
        Hsa::agent_get_info(bkendDevice_,
                            static_cast<hsa_agent_info_t>(HSA_EXT_AGENT_INFO_IMAGE_1D_MAX_ELEMENTS),
                            &max_image1d_width)) {
      return false;
    }
    info_.image1DMaxWidth_ = max_image1d_width;

    if (HSA_STATUS_SUCCESS !=
        Hsa::agent_get_info(bkendDevice_,
                            static_cast<hsa_agent_info_t>(HSA_EXT_AGENT_INFO_IMAGE_1DB_MAX_ELEMENTS),
                            &image_max_dim)) {
      return false;
    }
    info_.imageMaxBufferSize_ = (amd::IS_HIP) ? image_max_dim[0] : (1 << 27);

    info_.imagePitchAlignment_ = 256;

    info_.imageBaseAddressAlignment_ = 256;

    info_.bufferFromImageSupport_ = false;

    info_.imageSupport_ = (info_.maxReadWriteImageArgs_ > 0) ? true : false;
  }

  // Enable SVM Capabilities of Hsa device. Ensure
  // user has not setup memory to be non-coherent
  info_.svmCapabilities_ = 0;
  if (!settings().enableNCMode_) {
    info_.svmCapabilities_ = CL_DEVICE_SVM_COARSE_GRAIN_BUFFER;
    info_.svmCapabilities_ |= CL_DEVICE_SVM_FINE_GRAIN_BUFFER;
    // Report fine-grain system only on full profile
    if (agent_profile_ == HSA_PROFILE_FULL) {
      info_.svmCapabilities_ |= CL_DEVICE_SVM_FINE_GRAIN_SYSTEM;
    }
    if (amd::IS_HIP) {
      if (info_.iommuv2_ || isa().versionMajor() >= 8) {
        info_.svmCapabilities_ |= CL_DEVICE_SVM_ATOMICS;
      }
    } else if (!settings().useLightning_) {
      if (info_.iommuv2_ || (isa().versionMajor() == 8)) {
        info_.svmCapabilities_ |= CL_DEVICE_SVM_ATOMICS;
      }
    }
  }

  if (settings().checkExtension(ClAmdDeviceAttributeQuery)) {
    info_.simdWidth_ = isa().simdWidth();
    info_.simdInstructionWidth_ = isa().simdInstructionWidth();
    if (HSA_STATUS_SUCCESS !=
        Hsa::agent_get_info(bkendDevice_, HSA_AGENT_INFO_WAVEFRONT_SIZE, &info_.wavefrontWidth_)) {
      return false;
    }

    if (HSA_STATUS_SUCCESS !=
        Hsa::agent_get_info(bkendDevice_,
                            static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_MEMORY_WIDTH),
                            &info_.vramBusBitWidth_)) {
      return false;
    }

    info_.globalMemChannels_ = info_.vramBusBitWidth_ / 32;

    if (HSA_STATUS_SUCCESS !=
        Hsa::agent_get_info(bkendDevice_,
                            static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU),
                            &info_.simdPerCU_)) {
      return false;
    }

    uint32_t max_waves_per_cu = 0;
    if (HSA_STATUS_SUCCESS !=
        Hsa::agent_get_info(bkendDevice_,
                            static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU),
                            &max_waves_per_cu)) {
      return false;
    }

    if (settings().enableWgpMode_) {
      info_.simdPerCU_ *= 2;
      max_waves_per_cu *= 2;
    }

    info_.maxThreadsPerCU_ = info_.wavefrontWidth_ * max_waves_per_cu;
    uint32_t cache_sizes[4];
    /* FIXIT [skudchad] -  Seems like hardcoded in HSA backend so 0*/
    if (HSA_STATUS_SUCCESS !=
        Hsa::agent_get_info(bkendDevice_, static_cast<hsa_agent_info_t>(HSA_AGENT_INFO_CACHE_SIZE),
                            cache_sizes)) {
      return false;
    }

    uint32_t asic_revision = 0;
    if (HSA_STATUS_SUCCESS !=
        Hsa::agent_get_info(bkendDevice_,
                            static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_ASIC_REVISION),
                            &asic_revision)) {
      return false;
    }
    info_.asicRevision_ = asic_revision;

    info_.l2CacheSize_ = cache_sizes[1];
    info_.timeStampFrequency_ = 1000000;
    info_.globalMemChannelBanks_ = 4;
    info_.globalMemChannelBankWidth_ = isa().memChannelBankWidth();
    info_.localMemSizePerCU_ = isa().localMemSizePerCU();
    info_.localMemBanks_ = isa().localMemBanks();
    info_.numAsyncQueues_ = kMaxAsyncQueues;
    info_.numRTQueues_ = info_.numAsyncQueues_;
    info_.numRTCUs_ = info_.maxComputeUnits_;

    // TODO: set to true once thread trace support is available
    info_.threadTraceEnable_ = false;
    info_.pcieDeviceId_ = pciDeviceId_;
    info_.cooperativeGroups_ = settings().enableCoopGroups_;
    info_.cooperativeMultiDeviceGroups_ = settings().enableCoopMultiDeviceGroups_;
    // Enable StreamWrite and StreamWait for all devices
    info_.aqlBarrierValue_ = true;
  }

  info_.maxPipePacketSize_ = info_.maxMemAllocSize_;
  info_.maxPipeActiveReservations_ = 16;
  info_.maxPipeArgs_ = 16;

  info_.queueOnDeviceProperties_ =
      CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE;
  info_.queueOnDevicePreferredSize_ = 256 * Ki;
  info_.queueOnDeviceMaxSize_ = 8 * Mi;
  info_.maxOnDeviceQueues_ = 1;
  info_.maxOnDeviceEvents_ = settings().numDeviceEvents_;

  std::string addressableNumVGPRs, totalNumVGPRs, vGPRAllocGranule;
  std::string isaName = isa().isaName();
  info_.availableVGPRs_ =
      amd::device::getValueFromIsaMeta(isaName, "AddressableNumVGPRs", addressableNumVGPRs)
      ? atoi(addressableNumVGPRs.c_str())
      : 0;
  info_.vgprsPerSimd_ = amd::device::getValueFromIsaMeta(isaName, "TotalNumVGPRs", totalNumVGPRs)
      ? atoi(totalNumVGPRs.c_str())
      : 0;
  info_.vgprAllocGranularity_ =
      amd::device::getValueFromIsaMeta(isaName, "VGPRAllocGranule", vGPRAllocGranule)
      ? atoi(vGPRAllocGranule.c_str())
      : 0;

  info_.availableRegistersPerCU_ = info_.vgprsPerSimd_ * info_.simdPerCU_ * info_.wavefrontWidth_;
  ClPrint(amd::LOG_INFO, amd::LOG_INIT,
          "addressableNumVGPRs=%u, totalNumVGPRs=%u, vGPRAllocGranule=%u,"
          " availableRegistersPerCU_=%u",
          info_.availableVGPRs_, info_.vgprsPerSimd_, info_.vgprAllocGranularity_,
          info_.availableRegistersPerCU_);

  std::string sgprValue;
  info_.availableSGPRs_ =
      (amd::device::getValueFromIsaMeta(isaName, "AddressableNumSGPRs", sgprValue))
      ? (atoi(sgprValue.c_str()))
      : 0;
  std::string imageSupport;
  if (amd::device::getValueFromIsaMeta(isaName, "ImageSupport", imageSupport)) {
    info_.imageSupport_ = atoi(imageSupport.c_str());
    ClPrint(amd::LOG_INFO, amd::LOG_INIT, "imageSupport=%u", info_.imageSupport_);
  } else {
    LogInfo("Can not get image support info from ISA meta");
  }

  // Generic support for HMM interfaces
  if (HSA_STATUS_SUCCESS !=
      Hsa::system_get_info(HSA_AMD_SYSTEM_INFO_SVM_SUPPORTED, &info_.hmmSupported_)) {
    LogError("HSA_AMD_SYSTEM_INFO_SVM_SUPPORTED query failed. HMM will be disabled");
  }

  // This capability should be available with xnack enabled
  if (HSA_STATUS_SUCCESS != Hsa::system_get_info(HSA_AMD_SYSTEM_INFO_SVM_ACCESSIBLE_BY_DEFAULT,
                                                 &info_.hmmCpuMemoryAccessible_)) {
    LogError("HSA_AMD_SYSTEM_INFO_SVM_ACCESSIBLE_BY_DEFAULT query failed.");
  }

  // HMM specific capability for CPU direct access to device memory
  if (HSA_STATUS_SUCCESS !=
      Hsa::agent_get_info(bkendDevice_,
                          static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_SVM_DIRECT_HOST_ACCESS),
                          &info_.hmmDirectHostAccess_)) {
    LogError("HSA_AMD_AGENT_INFO_SVM_DIRECT_HOST_ACCESS query failed.");
  }

  if (HSA_STATUS_SUCCESS !=
      Hsa::agent_get_info(bkendDevice_, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_NUM_XCC),
                          &info_.numberOfXccs_)) {
    LogError("HSA_AMD_AGENT_INFO_NUM_XCC query failed.");
  }

  ClPrint(amd::LOG_INFO, amd::LOG_INIT, "Gfx Major/Minor/Stepping: %d/%d/%d", isa().versionMajor(),
          isa().versionMinor(), isa().versionStepping());
  ClPrint(amd::LOG_INFO, amd::LOG_INIT, "HMM support: %d, XNACK: %d, Direct host access: %d",
          info_.hmmSupported_, info_.hmmCpuMemoryAccessible_, info_.hmmDirectHostAccess_);
  ClPrint(amd::LOG_INFO, amd::LOG_INIT, "Max SDMA Read Mask: 0x%x, Max SDMA Write Mask: 0x%x",
          maxSdmaReadMask_, maxSdmaWriteMask_);

  info_.globalCUMask_ = {};

  // Virtual memory Management Support, if set to true then the HW and SW Stack supports VMM.
  info_.virtualMemoryManagement_ = false;
  if (HIP_VMEM_MANAGE_SUPPORT) {
    if (HSA_STATUS_SUCCESS !=
        Hsa::system_get_info(
            static_cast<hsa_system_info_t>(HSA_AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED),
            &info_.virtualMemoryManagement_)) {
      LogError("HSA_AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED query failed ");
    }
  }
  HIP_MEM_POOL_USE_VM &= info_.virtualMemoryManagement_;

  if (isa().versionMajor() < 8) {
    info_.sgprsPerSimd_ = 512;
  } else if (isa().versionMajor() < 10) {
    info_.sgprsPerSimd_ = 800;
  } else {
    info_.sgprsPerSimd_ =
        std::numeric_limits<uint32_t>::max();  // gfx10+ does not share SGPRs between waves
  }

  return true;
}

// ================================================================================================
device::VirtualDevice* Device::createVirtualDevice(amd::CommandQueue* queue) {
  amd::ScopedLock lock(vgpusAccess());

  bool profiling = (queue != nullptr) && queue->properties().test(CL_QUEUE_PROFILING_ENABLE);
  bool cooperative = false;

  // If amd command queue is null, then it's an internal device queue
  if (queue == nullptr) {
    // In HIP mode the device queue will be allocated for the cooperative launches only
    cooperative = amd::IS_HIP && settings().enableCoopGroups_;
    profiling = amd::IS_HIP;
  }
  // Initialization of heap and other resources occur during the command
  // queue creation time.
  const std::vector<uint32_t> defaultCuMask = {};
  bool q = (queue != nullptr);
  VirtualGPU* virtualDevice =
      new VirtualGPU(*this, profiling, cooperative, q ? queue->cuMask() : defaultCuMask,
                     q ? queue->priority() : amd::CommandQueue::Priority::Normal);

  if (!virtualDevice->create()) {
    delete virtualDevice;
    return nullptr;
  }

  return virtualDevice;
}

bool Device::globalFreeMemory(size_t* freeMemory) const {
  const uint TotalFreeMemory = 0;
  const uint LargestFreeBlock = 1;
  uint64_t globalAvailMemory;
  // Queries memory available in bytes across all global pools owned by the agent
  if (HSA_STATUS_SUCCESS !=
      Hsa::agent_get_info(bkendDevice_,
                          static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_MEMORY_AVAIL),
                          &globalAvailMemory)) {
    LogError("HSA_AMD_AGENT_INFO_MEMORY_AVAIL query failed.");
    return false;
  }

  globalAvailMemory = globalAvailMemory / Ki;
  if (globalAvailMemory > HIP_HIDDEN_FREE_MEM * Ki) {
    globalAvailMemory -= HIP_HIDDEN_FREE_MEM * Ki;
  } else {
    globalAvailMemory = 0;
  }

  freeMemory[TotalFreeMemory] = globalAvailMemory;
  // since there is no memory heap on ROCm, the biggest free block is
  // equal to total free local memory
  freeMemory[LargestFreeBlock] = freeMemory[TotalFreeMemory];

  return true;
}

bool Device::amdFileRead(amd::Os::FileDesc handle, void* devicePtr, uint64_t size, int64_t file_offset,
                      uint64_t* size_copied, int32_t* status) {
  hsa_amd_ais_file_handle_t fh{};
#if defined(_WIN32)
  fh.handle = handle;
#else
  fh.fd = handle;
#endif
  hsa_status_t ret = Hsa::ais_file_read(fh,
                                        devicePtr, size, file_offset, size_copied, status);
  if (HSA_STATUS_SUCCESS != ret) {
    LogPrintfError("hsa_amd_ais_file_read operation failed with err 0x%xh", ret);
    return false;
  }
  return true;
}

bool Device::amdFileWrite(amd::Os::FileDesc handle, void* devicePtr, uint64_t size, int64_t file_offset,
                       uint64_t* size_copied, int32_t* status) {
  hsa_amd_ais_file_handle_t fh{};
#if defined(_WIN32)
  fh.handle = handle;
#else
  fh.fd = handle;
#endif
  hsa_status_t ret = Hsa::ais_file_write(fh,
                                         devicePtr, size, file_offset, size_copied, status);
  if (HSA_STATUS_SUCCESS != ret) {
    LogPrintfError("hsa_amd_ais_file_write operation failed with err 0x%xh", ret);
    return false;
  }
  return true;
}

bool Device::bindExternalDevice(uint flags, void* const gfxDevice[], void* gfxContext,
                                bool validateOnly) {
#if defined(_WIN32)
  return false;
#else
  if ((flags & amd::Context::GLDeviceKhr) == 0) return false;

  MesaInterop::MESA_INTEROP_KIND kind = MesaInterop::MESA_INTEROP_NONE;
  MesaInterop::DisplayHandle display;
  MesaInterop::ContextHandle context;

  if ((flags & amd::Context::EGLDeviceKhr) != 0) {
    kind = MesaInterop::MESA_INTEROP_EGL;
    display.eglDisplay = reinterpret_cast<EGLDisplay>(gfxDevice[amd::Context::GLDeviceKhrIdx]);
    context.eglContext = reinterpret_cast<EGLContext>(gfxContext);
  } else {
    kind = MesaInterop::MESA_INTEROP_GLX;
    display.glxDisplay = reinterpret_cast<Display*>(gfxDevice[amd::Context::GLDeviceKhrIdx]);
    context.glxContext = reinterpret_cast<GLXContext>(gfxContext);
  }

  mesa_glinterop_device_info info;
  info.version = MESA_GLINTEROP_DEVICE_INFO_VERSION;
  if (!MesaInterop::Init(kind)) {
    return false;
  }

  if (!MesaInterop::GetInfo(info, kind, display, context)) {
    return false;
  }

  return info_.deviceTopology_.pcie.bus == info.pci_bus &&
         info_.deviceTopology_.pcie.device == info.pci_device &&
         info_.deviceTopology_.pcie.function == info.pci_function &&
         info_.vendorId_ == info.vendor_id && pciDeviceId_ == info.device_id;

#endif
}

bool Device::unbindExternalDevice(uint flags, void* const gfxDevice[], void* gfxContext,
                                  bool validateOnly) {
#if defined(_WIN32)
  return false;
#else
  if ((flags & amd::Context::GLDeviceKhr) == 0) return false;
  return true;
#endif
}

amd::Memory* Device::findMapTarget(size_t size) const {
  // Must be serialised for access
  amd::ScopedLock lk(*mapCacheOps_);

  amd::Memory* map = nullptr;
  size_t minSize = 0;
  size_t maxSize = 0;
  uint mapId = mapCache_->size();
  uint releaseId = mapCache_->size();

  // Find if the list has a map target of appropriate size
  for (uint i = 0; i < mapCache_->size(); i++) {
    if ((*mapCache_)[i] != nullptr) {
      // Requested size is smaller than the entry size
      if (size < (*mapCache_)[i]->getSize()) {
        if ((minSize == 0) || (minSize > (*mapCache_)[i]->getSize())) {
          minSize = (*mapCache_)[i]->getSize();
          mapId = i;
        }
      }
      // Requeted size matches the entry size
      else if (size == (*mapCache_)[i]->getSize()) {
        mapId = i;
        break;
      } else {
        // Find the biggest map target in the list
        if (maxSize < (*mapCache_)[i]->getSize()) {
          maxSize = (*mapCache_)[i]->getSize();
          releaseId = i;
        }
      }
    }
  }

  // Check if we found any map target
  if (mapId < mapCache_->size()) {
    map = (*mapCache_)[mapId];
    (*mapCache_)[mapId] = nullptr;
  }
  // If cache is full, then release the biggest map target
  else if (releaseId < mapCache_->size()) {
    (*mapCache_)[releaseId]->release();
    (*mapCache_)[releaseId] = nullptr;
  }

  return map;
}

bool Device::addMapTarget(amd::Memory* memory) const {
  // Must be serialised for access
  amd::ScopedLock lk(*mapCacheOps_);

  // the svm memory shouldn't be cached
  if (!memory->canBeCached()) {
    return false;
  }
  // Find if the list has a map target of appropriate size
  for (uint i = 0; i < mapCache_->size(); ++i) {
    if ((*mapCache_)[i] == nullptr) {
      (*mapCache_)[i] = memory;
      return true;
    }
  }

  // Add a new entry
  mapCache_->push_back(memory);

  return true;
}

Memory* Device::getRocMemory(amd::Memory* mem) const {
  return static_cast<roc::Memory*>(mem->getDeviceMemory(*this));
}

// ================================================================================================
device::Memory* Device::createMemory(amd::Memory& owner) const {
  roc::Memory* memory = nullptr;
  if (owner.asBuffer()) {
    memory = new roc::Buffer(*this, owner);
  } else if (owner.asImage()) {
    memory = new roc::Image(*this, owner);
  } else {
    LogError("Unknown memory type");
  }

  if (memory == nullptr) {
    return nullptr;
  }

  bool result = memory->create();

  if (!result) {
    LogError("Failed creating memory");
    delete memory;
    return nullptr;
  }

  if (isP2pEnabled()) {
    memory->setAllowedPeerAccess(true);
  }
  // Initialize if the memory is a pipe object
  if (owner.getType() == CL_MEM_OBJECT_PIPE) {
    // Pipe initialize in order read_idx, write_idx, end_idx. Refer clk_pipe_t structure.
    // Init with 3 DWORDS for 32bit addressing and 6 DWORDS for 64bit
    size_t pipeInit[3] = {0, 0, owner.asPipe()->getMaxNumPackets()};
    xferMgr().writeBuffer(pipeInit, *memory, amd::Coord3D(0), amd::Coord3D(sizeof(pipeInit)));
  }

  // Transfer data only if OCL context has one device.
  // Cache coherency layer will update data for multiple devices
  if (!memory->isHostMemDirectAccess() && owner.asImage() && (owner.parent() == nullptr) &&
      (owner.getMemFlags() & CL_MEM_COPY_HOST_PTR) && (owner.getContext().devices().size() == 1)) {
    // To avoid recurssive call to Device::createMemory, we perform
    // data transfer to the view of the image
    amd::Image* imageView = owner.asImage()->createView(
        owner.getContext(), owner.asImage()->getImageFormat(), xferQueue());

    if (imageView == nullptr) {
      LogError("[OCL] Fail to allocate view of image object");
      return nullptr;
    }

    Image* devImageView = new roc::Image(static_cast<const Device&>(*this), *imageView);
    if (devImageView == nullptr) {
      LogError("[OCL] Fail to allocate device mem object for the view");
      imageView->release();
      return nullptr;
    }

    if (devImageView != nullptr && !devImageView->createView(static_cast<roc::Image&>(*memory))) {
      LogError("[OCL] Fail to create device mem object for the view");
      delete devImageView;
      imageView->release();
      return nullptr;
    }

    imageView->replaceDeviceMemory(this, devImageView);

    // Copy data with the original pitch values, since runtime doesn't perform
    // extra sysmem allocation for one device
    const auto image = owner.asImage();
    result = xferMgr().writeImage(owner.getHostMem(), *devImageView, amd::Coord3D(0, 0, 0),
                                  imageView->getRegion(), image->getRowPitch(),
                                  image->getSlicePitch(), true);

    // Release host memory, since runtime copied data
    owner.setHostMem(nullptr);

    imageView->release();
  }

  // Prepin sysmem buffer for possible data synchronization between CPU and GPU
  if (!memory->isHostMemDirectAccess() &&
      // Pin memory for the parent object only
      (owner.parent() == nullptr) && (owner.getHostMem() != nullptr) &&
      (owner.getSvmPtr() == nullptr)) {
    memory->pinSystemMemory(owner.getHostMem(), owner.getSize());
  }

  if (!result) {
    delete memory;
    DevLogError("Cannot Write Image \n");
    return nullptr;
  }

  return memory;
}

// ================================================================================================
device::Memory* Device::createMemory(size_t size, size_t alignment) const {
  auto buffer = new roc::Buffer(*this, size);
  static constexpr bool LocalAlloc = true;
  if ((buffer == nullptr) || !buffer->create(LocalAlloc)) {
    LogError("Couldn't allocate memory on device!");
    return nullptr;
  }
  return buffer;
}

// ================================================================================================
hsa_amd_memory_pool_t Device::getHostMemoryPool(MemorySegment mem_seg,
                                                const AgentInfo* agentInfo) const {
  if (agentInfo == nullptr) {
    agentInfo = cpu_agent_info_;
  }
  hsa_amd_memory_pool_t segment{0};
  switch (mem_seg) {
    case kKernArg: {
      if (settings().fgs_kernel_arg_) {
        segment = agentInfo->kern_arg_pool;
        break;
      }
      // Falls through on else case.
    }
    case kNoAtomics:
      // If runtime disables barrier, then all host allocations must have L2 disabled
      if (agentInfo->coarse_grain_pool.handle != 0) {
        segment = agentInfo->coarse_grain_pool;
        break;
      }
      // Falls through on else case.
    case kAtomics:
      segment = agentInfo->fine_grain_pool;
      break;
    case kUncachedAtomics:
      if (agentInfo->ext_fine_grain_pool.handle != 0) {
        ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_MEM,
                "Using extended fine grained access system memory pool");
        segment = agentInfo->ext_fine_grain_pool;
        break;
      }
    default:
      guarantee(false, "Invalid Memory Segment");
      break;
  }
  assert(segment.handle != 0);
  return segment;
}

// ================================================================================================
void* Device::hostAlloc(size_t size, size_t alignment, MemorySegment mem_seg,
                        const void* agentInfo) const {
  void* ptr = nullptr;
  uint32_t memFlags = 0;
  if (mem_seg == kKernArg) {
    memFlags |= HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG;
  }
  hsa_amd_memory_pool_t pool =
      getHostMemoryPool(mem_seg, static_cast<const amd::roc::AgentInfo*>(agentInfo));
  hsa_status_t stat = Hsa::memory_pool_allocate(pool, size, memFlags, &ptr);

  ClPrint(amd::LOG_DEBUG, amd::LOG_MEM,
          "Allocate hsa host memory %p, size 0x%zx,"
          " numa_node = %d, mem_seg = %d",
          ptr, size, preferred_numa_node_, static_cast<int>(mem_seg));
  if (stat != HSA_STATUS_SUCCESS) {
    LogPrintfError("Fail allocation host memory with err %d", stat);
    return nullptr;
  }

  stat = Hsa::agents_allow_access(gpu_agents_.size(), &gpu_agents_[0], nullptr, ptr);
  if (stat != HSA_STATUS_SUCCESS) {
    LogPrintfError("Fail hsa_amd_agents_allow_access with err %d", stat);
    hostFree(ptr, size);
    return nullptr;
  }

  return ptr;
}

// ================================================================================================
void* Device::hostNumaAlloc(size_t size, size_t alignment, MemorySegment mem_seg) const {
  void* ptr = nullptr;
#ifndef ROCCLR_SUPPORT_NUMA_POLICY
  ptr = hostAlloc(size, alignment, mem_seg, cpu_agent_info_);
#else
  int mode = MPOL_DEFAULT;
  int maxNodes = numa_num_possible_nodes();
  bitmask* nodeMask = numa_bitmask_alloc(maxNodes);
  auto cpuCount = cpu_agents_.size();

  long res = get_mempolicy(&mode, nodeMask->maskp, nodeMask->size, NULL, 0);
  if (res) {
    LogPrintfError("get_mempolicy failed with error %ld", res);
    return ptr;
  }
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_RESOURCE,
          "get_mempolicy() succeed with mode %d, nodeMask 0x%lx, cpuCount %zu", mode,
          *nodeMask->maskp, cpuCount);

  switch (mode) {
    // For details, see "man get_mempolicy".
    case MPOL_BIND:
    case MPOL_PREFERRED:
      // We only care about the first CPU node
      for (unsigned int i = 0; i < cpuCount; i++) {
        if ((1u << i) & *nodeMask->maskp) {
          ptr = hostAlloc(size, alignment, mem_seg, &cpu_agents_[i]);
          break;
        }
      }
      break;
    default:
      //  All other modes fall back to default mode
      ptr = hostAlloc(size, alignment, mem_seg, cpu_agent_info_);
  }
  numa_free_cpumask(nodeMask);
#endif  // ROCCLR_SUPPORT_NUMA_POLICY
  return ptr;
}

void* Device::hostLock(void* hostMem, size_t size, const MemorySegment memSegment) const {
  hsa_amd_memory_pool_t pool = getHostMemoryPool(memSegment);
  void* deviceMemory = nullptr;
  hsa_status_t status = Hsa::memory_lock_to_pool(
      hostMem, size, const_cast<hsa_agent_t*>(&bkendDevice_), 1, pool, 0, &deviceMemory);
  ClPrint(amd::LOG_DEBUG, amd::LOG_MEM,
          "Locking to pool %p, size 0x%zx, hostMem = %p,"
          " deviceMemory = %p, memSegment = %d",
          pool, size, hostMem, deviceMemory, static_cast<int>(memSegment));
  if (status != HSA_STATUS_SUCCESS) {
    DevLogPrintfError("Failed to lock memory to pool, failed with hsa_status: %d \n", status);
    deviceMemory = nullptr;
  }
  return deviceMemory;
}

void Device::hostFree(void* ptr, size_t size) const { memFree(ptr, size); }

bool Device::deviceAllowAccess(void* ptr) const {
  std::lock_guard<std::mutex> lock(lock_allow_access_);
  if (!p2pAgents().empty()) {
    hsa_status_t stat =
        Hsa::agents_allow_access(p2pAgents().size(), p2pAgents().data(), nullptr, ptr);
    if (stat != HSA_STATUS_SUCCESS) {
      LogPrintfError("Allow p2p access failed - hsa_amd_agents_allow_access with err %d", stat);
      return false;
    }
  }
  return true;
}

bool Device::allowPeerAccess(device::Memory* memory) const {
  if (memory == nullptr) {
    return false;
  }
  if (!p2pAgents().empty()) {
    void* ptr = reinterpret_cast<void*>(memory->virtualAddress());
    hsa_agent_t agent = getBackendDevice();
    hsa_status_t stat = Hsa::agents_allow_access(1, &agent, nullptr, ptr);
    if (stat != HSA_STATUS_SUCCESS) {
      LogPrintfError("Allow p2p access failed - hsa_amd_agents_allow_access with err: %d", stat);
      return false;
    }
  }
  return true;
}

uint64_t Device::deviceVmemAlloc(size_t size, uint64_t flags) const {
  hsa_amd_vmem_alloc_handle_t hsa_vmem_handle{};

  // We only allow pinned memory at this time.
  hsa_status_t hsa_status =
      Hsa::vmem_handle_create(gpuvm_segment_, size, MEMORY_TYPE_PINNED, flags, &hsa_vmem_handle);
  if (hsa_status != HSA_STATUS_SUCCESS) {
    LogPrintfError("Failed hsa_amd_vmem_handle_create! Failed with hsa status: %d \n", hsa_status);
  }

  return hsa_vmem_handle.handle;
}

void Device::deviceVmemRelease(uint64_t mem_handle) const {
  hsa_amd_vmem_alloc_handle_t hsa_vmem_handle{};
  hsa_vmem_handle.handle = mem_handle;

  hsa_status_t hsa_status = Hsa::vmem_handle_release(hsa_vmem_handle);
  if (hsa_status != HSA_STATUS_SUCCESS) {
    LogPrintfError("Failed hsa_amd_vmem_handle_release! Failed with hsa status: %d \n", hsa_status);
  }
}

void* Device::reserveMemory(size_t size, size_t alignment) const {
  void* ptr = nullptr;
  // Reserves non registered VA memory using HSA APIs.
  hsa_status_t status = Hsa::vmem_address_reserve_align(&ptr, size, 0, alignment,
                                                        HSA_AMD_VMEM_ADDRESS_NO_REGISTER);
  ClPrint(amd::LOG_DEBUG, amd::LOG_MEM, "Reserve hsa device memory %p, size 0x%zx", ptr, size);
  if (status != HSA_STATUS_SUCCESS) {
    LogError("Fail to reserve memory");
    return nullptr;
  }
  return ptr;
}

void Device::releaseMemory(void* ptr, size_t size) const {
  hsa_status_t hsa_status = Hsa::vmem_address_free(ptr, size);
  ClPrint(amd::LOG_DEBUG, amd::LOG_MEM, "Free hsa reserved memory %p", ptr);
  if (hsa_status != HSA_STATUS_SUCCESS) {
    LogError("hsa_amd_vmem_address_free failed \n");
  }
}

void* Device::deviceLocalAlloc(size_t size, const AllocationFlags& flags) const {
  const hsa_amd_memory_pool_t& pool =
      (flags.pseudo_fine_grain_ && gpu_ext_fine_grained_segment_.handle)
          ? gpu_ext_fine_grained_segment_
      : (flags.atomics_ && gpu_fine_grained_segment_.handle) ? gpu_fine_grained_segment_
                                                             : gpuvm_segment_;

  if (pool.handle == 0 || gpuvm_segment_max_alloc_ == 0) {
    DevLogPrintfError("Invalid argument, pool_handle: 0x%x , max_alloc: %u \n", pool.handle,
                      gpuvm_segment_max_alloc_);
    return nullptr;
  }

  uint32_t hsa_mem_flags = 0;
  if (flags.contiguous_) {
    hsa_mem_flags = HSA_AMD_MEMORY_POOL_CONTIGUOUS_FLAG;
  }
  if (flags.executable_) {
    hsa_mem_flags |= HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG;
  }

  void* ptr = nullptr;
  hsa_status_t stat = Hsa::memory_pool_allocate(pool, size, hsa_mem_flags, &ptr);
  ClPrint(amd::LOG_DEBUG, amd::LOG_MEM,
          "Allocate hsa device memory %p, size 0x%zx, hsa_mem_flags 0x%xh", ptr, size,
          hsa_mem_flags);
  if (stat != HSA_STATUS_SUCCESS) {
    LogError("Fail allocation local memory");
    return nullptr;
  }

  if (isP2pEnabled() && deviceAllowAccess(ptr) == false) {
    LogError("Allow p2p access for memory allocation");
    memFree(ptr, size);
    return nullptr;
  }
  return ptr;
}

void Device::memFree(void* ptr, size_t size) const {
  hsa_status_t stat = Hsa::memory_pool_free(ptr);
  ClPrint(amd::LOG_DEBUG, amd::LOG_MEM, "Free hsa memory %p", ptr);
  if (stat != HSA_STATUS_SUCCESS) {
    LogError("Fail freeing local memory");
  }
}

void Device::updateFreeMemory(size_t size, bool free) {
  if (free) {
    freeMem_ += size;
  } else {
    if (size > freeMem_) {
      // To avoid underflow of the freeMem_
      // This can happen if the free mem tracked is inaccurate, as some allocations can happen
      // directly via ROCr
      ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS,
              "Free memory set to zero on device 0x%lx, requested size = 0x%zx, freeMem_ = 0x%zx",
              this, size, freeMem_.load());
      freeMem_ = 0;
      return;
    }
    freeMem_ -= size;
  }
  ClPrint(amd::LOG_INFO, amd::LOG_MEM, "Device=0x%lx, freeMem_ = 0x%zx", this, freeMem_.load());
}

// ================================================================================================
void* Device::svmAlloc(amd::Context& context, size_t size, size_t alignment, cl_svm_mem_flags flags,
                       void* svmPtr) const {
  amd::Memory* mem = nullptr;
  void* svmPtrUsed = reinterpret_cast<void*>(amd::Memory::MemoryType::kSvmMemoryPtr);

  if (nullptr != svmPtr) {
    // Find the existing amd::mem object
    mem = amd::MemObjMap::FindMemObj(svmPtr);
    if (mem != nullptr) {
      return mem->getSvmPtr();
    }
    if (flags & CL_MEM_USE_HOST_PTR) {
      svmPtrUsed = svmPtr;
    } else {
      DevLogPrintfError("Cannot find svm_ptr: 0x%x \n", svmPtr);
      return nullptr;
    }
  }

  // create a hidden buffer, which will allocated on the device later
  mem = new (context) amd::Buffer(context, flags, size, svmPtrUsed);
  if (mem == nullptr) {
    LogError("failed to create a svm mem object!");
    return nullptr;
  }

  if (!mem->create(nullptr)) {
    LogError("failed to create a svm hidden buffer!");
    mem->release();
    return nullptr;
  }
  // if the device supports SVM FGS, return the committed CPU address directly.
  Memory* gpuMem = getRocMemory(mem);
  if (gpuMem == nullptr) {
    LogError("failed to create GPU memory from svm hidden buffer!");
    return nullptr;
  }

  // add the information to context so that we can use it later.
  if (mem->getSvmPtr() != nullptr) {
    amd::MemObjMap::AddMemObj(mem->getSvmPtr(), mem);
  }
  return mem->getSvmPtr();
}

void* Device::virtualAlloc(void* req_addr, size_t size, size_t alignment) {
  void* vptr = nullptr;
  // Reserves the address using HSA APIs, with requested address.
  // There is no guarantee that we will get the requested address.
  hsa_status_t hsa_status =
      Hsa::vmem_address_reserve(&vptr, size, reinterpret_cast<uint64_t>(req_addr), 0);
  if (hsa_status != HSA_STATUS_SUCCESS) {
    LogPrintfError("Failed hsa_amd_vmem_address_reserve. Failed with status: %d \n", hsa_status);
    return nullptr;
  }

  constexpr bool kParent = true;
  amd::Memory* mem = CreateVirtualBuffer(context(), vptr, size, -1, -1, kParent);
  if (mem == nullptr) {
    LogPrintfError("Cannot create Virtual Buffer for vptr: %p of size: %u", vptr, size);
  }

  return mem->getSvmPtr();
}

bool Device::virtualFree(void* addr) {
  amd::Memory* memObj = amd::MemObjMap::FindVirtualMemObj(addr);
  if (memObj == nullptr) {
    LogPrintfError("Cannot find the Virtual MemObj entry for this addr 0x%x", addr);
  }

  if (!memObj->getContext().devices()[0]->DestroyVirtualBuffer(memObj)) {
    return false;
  }

  hsa_status_t hsa_status = Hsa::vmem_address_free(memObj->getSvmPtr(), memObj->getSize());
  if (hsa_status != HSA_STATUS_SUCCESS) {
    LogPrintfError("Failed hsa_amd_vmem_address_free. Failed with status:%d \n", hsa_status);
    return false;
  }
  return true;
}

bool Device::SetMemAccess(void* va_addr, size_t va_size, VmmAccess access_flags,
                          VmmLocationType access_location) {
  hsa_status_t hsa_status = HSA_STATUS_SUCCESS;
  hsa_amd_memory_access_desc_t desc;
  desc.permissions = static_cast<hsa_access_permission_t>(access_flags);
  desc.agent_handle =
      access_location == VmmLocationType::kDevice ? getBackendDevice() : getCpuAgent();

  if ((hsa_status = Hsa::vmem_set_access(va_addr, va_size, &desc, 1)) != HSA_STATUS_SUCCESS) {
    LogPrintfError("Failed hsa_amd_vmem_set_access. Failed with status:%d \n", hsa_status);
    return false;
  }

  return true;
}

bool Device::GetMemAccess(void* va_addr, VmmAccess* access_flags_ptr) const {
  hsa_status_t hsa_status = HSA_STATUS_SUCCESS;
  hsa_access_permission_t perms;

  size_t discard_offset = 0;
  amd::Memory* va_mem_obj = amd::MemObjMap::FindMemObj(va_addr, &discard_offset);
  if (va_mem_obj == nullptr) {
    LogPrintfError("Failed to get Memory Object for va_addr: 0x%x", va_addr);
    return false;
  }

  if ((hsa_status = Hsa::vmem_get_access(va_mem_obj->getSvmPtr(), &perms, getBackendDevice())) !=
      HSA_STATUS_SUCCESS) {
    LogPrintfError("Failed hsa_amd_vmem_get_access. Failed with status:%d \n", hsa_status);
    return false;
  }

  *access_flags_ptr = static_cast<VmmAccess>(perms);

  return true;
}

// ================================================================================================
bool Device::ExportShareableVMMHandle(amd::Memory& amd_mem_obj, int flags, void* shareableHandle) {
  hsa_status_t hsa_status = HSA_STATUS_SUCCESS;
  hsa_amd_vmem_alloc_handle_t hsa_vmem_handle{};
  hsa_vmem_handle.handle = amd_mem_obj.getUserData().hsa_handle;
  int dmabuf_fd = 0;

  if (hsa_vmem_handle.handle == 0) {
    LogError("HSA Handle is not valid");
    return false;
  }

  if ((hsa_status = Hsa::vmem_export_shareable_handle(&dmabuf_fd, hsa_vmem_handle, flags)) !=
      HSA_STATUS_SUCCESS) {
    LogPrintfError("Failed hsa_vmem_export_shareable_handle with status: %d \n", hsa_status);
    return false;
  }

  *(reinterpret_cast<int*>(shareableHandle)) = dmabuf_fd;

  return true;
}

// ================================================================================================
bool Device::ImportShareableHSAHandle(void* osHandle, uint64_t* hsa_handle_ptr) const {
  hsa_status_t hsa_status = HSA_STATUS_SUCCESS;
  hsa_amd_vmem_alloc_handle_t hsa_vmem_handle{};

  if (hsa_handle_ptr == nullptr) {
    LogError("HSA Handle ptr is null");
    return false;
  }

  int dmabuf_fd = static_cast<int>(reinterpret_cast<uintptr_t>(osHandle));
  if ((hsa_status = Hsa::vmem_import_shareable_handle(dmabuf_fd, &hsa_vmem_handle)) !=
      HSA_STATUS_SUCCESS) {
    LogPrintfError("Failed hsa_amd_vmem_import_shareable_handle with status: %d \n", hsa_status);
    return false;
  }

  *hsa_handle_ptr = hsa_vmem_handle.handle;
  return true;
}

// ================================================================================================
amd::Memory* Device::ImportShareableVMMHandle(void* osHandle) {
  amd::Memory* amd_mem_obj = new (context())
      amd::Buffer(context(), ROCCLR_MEM_PHYMEM | ROCCLR_MEM_INTERPROCESS, 0, osHandle);
  if (amd_mem_obj == nullptr) {
    LogError("Cannot create memory object");
    return nullptr;
  }

  if (!amd_mem_obj->create(nullptr, false)) {
    LogError("Failed to create mem_obj from imported fd");
    amd_mem_obj->release();
    return nullptr;
  }

  return amd_mem_obj;
}

// ================================================================================================
bool Device::SetSvmAttributesInt(const void* dev_ptr, size_t count, amd::MemoryAdvice advice,
                                 bool first_alloc, bool use_cpu, int numa_id) const {
  if ((settings().hmmFlags_ & Settings::Hmm::EnableSvmTracking) && !first_alloc) {
    amd::Memory* svm_mem = amd::MemObjMap::FindMemObj(dev_ptr);
    if ((nullptr == svm_mem) || ((svm_mem->getMemFlags() & CL_MEM_ALLOC_HOST_PTR) == 0) ||
        // Validate the range of provided memory
        ((svm_mem->getSize() - (reinterpret_cast<const_address>(dev_ptr) -
                                reinterpret_cast<address>(svm_mem->getSvmPtr()))) < count)) {
      LogPrintfError("SetSvmAttributes received unknown memory for update: %p!", dev_ptr);
      return false;
    }
  }
  if (info().hmmSupported_) {
    std::vector<hsa_amd_svm_attribute_pair_t> attr;

    switch (advice) {
      case amd::MemoryAdvice::SetReadMostly:
        attr.push_back({HSA_AMD_SVM_ATTRIB_READ_MOSTLY, true});
        break;
      case amd::MemoryAdvice::UnsetReadMostly:
        attr.push_back({HSA_AMD_SVM_ATTRIB_READ_MOSTLY, false});
        break;
      case amd::MemoryAdvice::SetPreferredLocation:
        if (use_cpu) {
          attr.push_back({HSA_AMD_SVM_ATTRIB_PREFERRED_LOCATION, getCpuAgent(numa_id).handle});
        } else {
          attr.push_back({HSA_AMD_SVM_ATTRIB_PREFERRED_LOCATION, getBackendDevice().handle});
        }
        break;
      case amd::MemoryAdvice::UnsetPreferredLocation:
        // @note: 0 may cause a failure on old runtimes
        attr.push_back({HSA_AMD_SVM_ATTRIB_PREFERRED_LOCATION, 0});
        break;
      case amd::MemoryAdvice::SetAccessedBy: {
        const uint64_t attrib = (first_alloc) ? HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE
                                              : HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE_IN_PLACE;
        if (use_cpu) {
          attr.push_back({attrib, getCpuAgent().handle});
        } else {
          if (first_alloc) {
            // Provide access to all possible devices.
            //! @note: HMM should support automatic page table update with xnack enabled,
            //! but currently it doesn't and runtime explicitly enables access from all devices
            for (const auto dev : devices()) {
              // Skip null devices
              if (static_cast<Device*>(dev)->getBackendDevice().handle != 0) {
                attr.push_back({attrib, static_cast<Device*>(dev)->getBackendDevice().handle});
              }
            }
          } else {
            attr.push_back({attrib, getBackendDevice().handle});
          }
        }
        break;
      }
      case amd::MemoryAdvice::UnsetAccessedBy:
        // When unsetting we should use HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE for the agent
        attr.push_back({HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE, getBackendDevice().handle});
        break;
      case amd::MemoryAdvice::SetCoarseGrain:
        attr.push_back({HSA_AMD_SVM_ATTRIB_GLOBAL_FLAG, HSA_AMD_SVM_GLOBAL_FLAG_COARSE_GRAINED});
        break;
      case amd::MemoryAdvice::UnsetCoarseGrain:
        attr.push_back({HSA_AMD_SVM_ATTRIB_GLOBAL_FLAG, HSA_AMD_SVM_GLOBAL_FLAG_FINE_GRAINED});
        break;
      default:
        return false;
        break;
    }

    hsa_status_t status =
        Hsa::svm_attributes_set(const_cast<void*>(dev_ptr), count, attr.data(), attr.size());
    if (status != HSA_STATUS_SUCCESS) {
      LogPrintfError("hsa_amd_svm_attributes_set() failed. Advice: %d, status: %d", advice, status);
      return false;
    }
  } else {
    LogWarning("hsa_amd_svm_attributes_set() is ignored, because no HMM support");
  }
  return true;
}

// ================================================================================================
bool Device::SetSvmAttributes(const void* dev_ptr, size_t count, amd::MemoryAdvice advice,
                              bool use_cpu, int numa_id) const {
  constexpr bool kFirstAlloc = false;
  return SetSvmAttributesInt(dev_ptr, count, advice, kFirstAlloc, use_cpu);
}

// ================================================================================================
bool Device::GetSvmAttributes(void** data, size_t* data_sizes, int* attributes,
                              size_t num_attributes, const void* dev_ptr, size_t count) const {
  if (settings().hmmFlags_ & Settings::Hmm::EnableSvmTracking) {
    amd::Memory* svm_mem = amd::MemObjMap::FindMemObj(dev_ptr);
    if ((nullptr == svm_mem) || ((svm_mem->getMemFlags() & CL_MEM_ALLOC_HOST_PTR) == 0) ||
        // Validate the range of provided memory
        ((svm_mem->getSize() - (reinterpret_cast<const_address>(dev_ptr) -
                                reinterpret_cast<address>(svm_mem->getSvmPtr()))) < count)) {
      LogPrintfError("GetSvmAttributes received unknown memory %p for state!", dev_ptr);
      return false;
    }
  }

  hsa_amd_pointer_info_t ptr_info = {};
  for (size_t i = 0; i < num_attributes; ++i) {
    if (attributes[i] == amd::MemRangeAttribute::CoherencyMode) {
      ptr_info.size = sizeof(hsa_amd_pointer_info_t);
      // Query ptr type to see if it's a HMM allocation
      hsa_status_t status =
          Hsa::pointer_info(const_cast<void*>(dev_ptr), &ptr_info, nullptr, nullptr, nullptr);
      // The call should never fail in ROCR, but just check for an error and continue
      if (status != HSA_STATUS_SUCCESS) {
        LogError("hsa_amd_pointer_info() failed");
      }

      // Check if it's a legacy non-HMM allocation and update query
      *reinterpret_cast<uint32_t*>(data[i]) = HSA_AMD_SVM_GLOBAL_FLAG_INDETERMINATE;
      if (ptr_info.type == HSA_EXT_POINTER_TYPE_HSA ||
          ptr_info.type == HSA_EXT_POINTER_TYPE_LOCKED ||
          ptr_info.type == HSA_EXT_POINTER_TYPE_IPC) {
        if (ptr_info.global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) {
          *reinterpret_cast<uint32_t*>(data[i]) = HSA_AMD_SVM_GLOBAL_FLAG_COARSE_GRAINED;
        } else if (ptr_info.global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) {
          *reinterpret_cast<uint32_t*>(data[i]) = HSA_AMD_SVM_GLOBAL_FLAG_FINE_GRAINED;
        }
      }
    }
  }

  if (info().hmmSupported_) {
    uint32_t accessed_by = 0;
    std::vector<hsa_amd_svm_attribute_pair_t> attr;

    for (size_t i = 0; i < num_attributes; ++i) {
      switch (attributes[i]) {
        case amd::MemRangeAttribute::ReadMostly:
          attr.push_back({HSA_AMD_SVM_ATTRIB_READ_MOSTLY, 0});
          break;
        case amd::MemRangeAttribute::PreferredLocation:
          attr.push_back({HSA_AMD_SVM_ATTRIB_PREFERRED_LOCATION, 0});
          break;
        case amd::MemRangeAttribute::AccessedBy:
          accessed_by = attr.size();
          // Add all GPU devices into the query
          for (const auto agent : gpu_agents_) {
            attr.push_back({HSA_AMD_SVM_ATTRIB_ACCESS_QUERY, agent.handle});
          }
          // Add CPU devices
          for (const auto agent_info : cpu_agents_) {
            attr.push_back({HSA_AMD_SVM_ATTRIB_ACCESS_QUERY, agent_info.agent.handle});
          }
          accessed_by = attr.size() - accessed_by;
          break;
        case amd::MemRangeAttribute::LastPrefetchLocation:
          attr.push_back({HSA_AMD_SVM_ATTRIB_PREFETCH_LOCATION, 0});
          break;
        case amd::MemRangeAttribute::CoherencyMode:
          if (*reinterpret_cast<uint32_t*>(data[i]) == HSA_AMD_SVM_GLOBAL_FLAG_INDETERMINATE) {
            attr.push_back({HSA_AMD_SVM_ATTRIB_GLOBAL_FLAG, 0});
          }
          break;
        default:
          return false;
          break;
      }
    }

    hsa_status_t status =
        Hsa::svm_attributes_get(const_cast<void*>(dev_ptr), count, attr.data(), attr.size());
    if (status != HSA_STATUS_SUCCESS) {
      LogError("hsa_amd_svm_attributes_get() failed");
      return false;
    }

    uint32_t idx = 0;
    uint32_t rocr_attr = 0;
    for (size_t i = 0; i < num_attributes; ++i) {
      const auto& it = attr[rocr_attr];
      switch (attributes[i]) {
        case amd::MemRangeAttribute::ReadMostly:
          if (data_sizes[idx] != sizeof(uint32_t)) {
            return false;
          }
          // Cast ROCr value into the hip format
          *reinterpret_cast<uint32_t*>(data[idx]) =
              (static_cast<uint32_t>(it.value) > 0) ? true : false;
          ++rocr_attr;
          break;
        // The logic should be identical for the both queries
        case amd::MemRangeAttribute::PreferredLocation:
        case amd::MemRangeAttribute::LastPrefetchLocation:
          if (data_sizes[idx] != sizeof(uint32_t)) {
            return false;
          }
          *reinterpret_cast<int32_t*>(data[idx]) = static_cast<int32_t>(amd::InvalidDeviceId);
          // Find device agent returned by ROCr
          for (auto& device : devices()) {
            if (static_cast<Device*>(device)->getBackendDevice().handle == it.value) {
              *reinterpret_cast<uint32_t*>(data[idx]) = static_cast<uint32_t>(device->index());
            }
          }
          // Find CPU agent returned by ROCr
          for (auto& agent_info : cpu_agents_) {
            if (agent_info.agent.handle == it.value) {
              *reinterpret_cast<int32_t*>(data[idx]) = static_cast<int32_t>(amd::CpuDeviceId);
            }
          }
          ++rocr_attr;
          break;
        case amd::MemRangeAttribute::AccessedBy: {
          uint32_t entry = 0;
          uint32_t device_count = data_sizes[idx] / 4;
          // Make sure it's multiple of 4
          if (data_sizes[idx] % 4 != 0) {
            return false;
          }
          for (uint32_t att = 0; att < accessed_by; ++att) {
            const auto& it = attr[rocr_attr + att];
            if (entry >= device_count) {
              // The size of the array is less than the amount of available devices
              break;
            }
            switch (it.attribute) {
              case HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE:
              case HSA_AMD_SVM_ATTRIB_AGENT_NO_ACCESS:
                break;
              case HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE_IN_PLACE:
                reinterpret_cast<int32_t*>(data[idx])[entry] =
                    static_cast<int32_t>(amd::InvalidDeviceId);
                // Find device agent returned by ROCr
                for (auto& device : devices()) {
                  if (static_cast<Device*>(device)->getBackendDevice().handle == it.value) {
                    reinterpret_cast<uint32_t*>(data[idx])[entry] =
                        static_cast<uint32_t>(device->index());
                  }
                }
                // Find CPU agent returned by ROCr
                for (auto& agent_info : cpu_agents_) {
                  if (agent_info.agent.handle == it.value) {
                    reinterpret_cast<int32_t*>(data[idx])[entry] =
                        static_cast<int32_t>(amd::CpuDeviceId);
                  }
                }
                ++entry;
                break;
              default:
                LogWarning("Unexpected result from HSA_AMD_SVM_ATTRIB_ACCESS_QUERY");
                break;
            }
          }
          rocr_attr += accessed_by;
          for (uint32_t i = entry; i < device_count; ++i) {
            reinterpret_cast<int32_t*>(data[idx])[i] = static_cast<int32_t>(amd::InvalidDeviceId);
          }
          break;
        }
        case amd::MemRangeAttribute::CoherencyMode:
          if (data_sizes[idx] != sizeof(uint32_t)) {
            return false;
          }
          // if ptr is HMM alloc then overwrite the values
          if (*reinterpret_cast<uint32_t*>(data[idx]) == HSA_AMD_SVM_GLOBAL_FLAG_INDETERMINATE) {
            // Cast ROCr value into the hip format
            *reinterpret_cast<uint32_t*>(data[idx]) = static_cast<uint32_t>(it.value);
          }
          ++rocr_attr;
          break;
        default:
          return false;
          break;
      }
      // Find the next location in the query
      ++idx;
    }
  } else if (ptr_info.type == HSA_EXT_POINTER_TYPE_RESERVED_ADDR) {
    LogError("GetSvmAttributes() failed, because no HMM support");
    return false;
  }

  return true;
}

size_t Device::ScratchLimitCurrent() const {
  uint64_t scratchLimitCurrent = 0;
  hsa_status_t ret =
      Hsa::agent_get_info(bkendDevice_, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_SCRATCH_LIMIT_CURRENT,
                         &scratchLimitCurrent);
  if (HSA_STATUS_SUCCESS != ret) {
    LogPrintfError("HSA_AMD_AGENT_INFO_SCRATCH_LIMIT_CURRENT cannot be queried! Err: 0x%xh", ret);
    return 0;
  }
  return static_cast<size_t>(scratchLimitCurrent);
};

bool Device::UpdateScratchLimitCurrent(size_t limit) const {
  hsa_status_t ret = Hsa::agent_set_async_scratch_limit(bkendDevice_, limit);
  if (HSA_STATUS_SUCCESS != ret) {
    LogPrintfError("hsa_amd_agent_set_async_scratch_limit(%zu) failed with err 0x%xh", limit, ret);
    return false;
  }
  return true;
};

// ================================================================================================
bool Device::SvmAllocInit(void* memory, size_t size) const {
  amd::MemoryAdvice advice = amd::MemoryAdvice::SetAccessedBy;
  constexpr bool kFirstAlloc = true;
  if (!SetSvmAttributesInt(memory, size, advice, kFirstAlloc)) {
    return false;
  }

  if ((settings().hmmFlags_ & Settings::Hmm::EnableMallocPrefetch) == 0) {
    return true;
  }

  if (info().hmmSupported_) {
    // Initialize signal for the barrier
    Hsa::signal_store_relaxed(prefetch_signal_, kInitSignalValueOne);

    // Initiate a prefetch command which should force memory update in HMM
    hsa_status_t status =
        Hsa::svm_prefetch_async(memory, size, getBackendDevice(), 0, nullptr, prefetch_signal_);
    if (status != HSA_STATUS_SUCCESS) {
      LogError("hsa_amd_svm_prefetch_async() failed");
      return false;
    }

    // Wait for the prefetch
    if (!WaitForSignal(prefetch_signal_)) {
      LogError("Barrier packet submission failed");
      return false;
    }
  } else {
    LogWarning("Early prefetch failed, because no HMM support");
  }

  return true;
}

// ================================================================================================
void Device::svmFree(void* ptr) const {
  amd::Memory* svmMem = amd::MemObjMap::FindMemObj(ptr);
  if (nullptr != svmMem) {
    amd::MemObjMap::RemoveMemObj(svmMem->getSvmPtr());
    svmMem->release();
  }
}

// ================================================================================================
VirtualGPU* Device::xferQueue() const {
  if (!xferQueue_) {
    // Create virtual device for internal memory transfer
    Device* thisDevice = const_cast<Device*>(this);
    thisDevice->xferQueue_ = reinterpret_cast<VirtualGPU*>(thisDevice->createVirtualDevice());
    if (!xferQueue_) {
      LogError("Couldn't create the device transfer manager!");
      return nullptr;
    }
    if (xferQueue_->gpu_queue() == nullptr) {
      xferQueue_->set_gpu_queue(thisDevice->AcquireActiveNormalQueue());
    }
  }
  xferQueue_->enableSyncBlit();
  return xferQueue_;
}

// ================================================================================================
bool Device::SetClockMode(const cl_set_device_clock_mode_input_amd setClockModeInput,
                          cl_set_device_clock_mode_output_amd* pSetClockModeOutput) {
  bool result = true;
  return result;
}

// ================================================================================================
bool Device::IsHwEventReady(const amd::Event& event, bool wait, amd::SyncPolicy policy) const {
  void* hw_event =
      (event.NotifyEvent() != nullptr) ? event.NotifyEvent()->HwEvent() : event.HwEvent();
  if (hw_event == nullptr) {
    ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_SIG, "No HW event");
    return false;
  } else if (wait) {
    // hipEventBlockingSync
    // when set the CPU gives up host thread for other work
    // when not set the CPU enters a busy-wait on the event to occur
    constexpr int kHipEventBlockingSync = 0x1;
    bool active_wait =
        !((policy == amd::SyncPolicy::Blocking) & kHipEventBlockingSync) && ActiveWait();
    bool yield = (policy == amd::SyncPolicy::Yield);
    return WaitForSignal(reinterpret_cast<ProfilingSignal*>(hw_event)->signal_, active_wait, yield);
  }

  auto signal = reinterpret_cast<ProfilingSignal*>(hw_event)->signal_;
  ClPrint(amd::LOG_INFO, amd::LOG_SIG, "Check HW event = 0x%lx", signal.handle);

  return (Hsa::signal_load_relaxed(signal) == 0);
}

// ================================================================================================
void Device::getHwEventTime(const amd::Event& event, uint64_t* start, uint64_t* end) const {
  void* hw_event =
      (event.NotifyEvent() != nullptr) ? event.NotifyEvent()->HwEvent() : event.HwEvent();
  if (hw_event == nullptr) {
    ClPrint(amd::LOG_INFO, amd::LOG_SIG, "No HW event to read time");
    *start = *end = 0;
  } else {
    fetchSignalTime(reinterpret_cast<ProfilingSignal*>(hw_event)->signal_, getBackendDevice(),
                    start, end);
  }
}

// ================================================================================================
hsa_queue_t* Device::getQueueFromPool(const uint qIndex) {
  // Check if queue with refCount 0 is available to use
  if (queuePool_[qIndex].size() < GPU_MAX_HW_QUEUES) {
    for (auto& it : queuePool_[qIndex]) {
      if (it.second.refCount == 0) {
        it.second.refCount++;
        ClPrint(amd::LOG_INFO, amd::LOG_QUEUE, "Selected queue refCount: %p (%d)",
                it.first->base_address, it.second.refCount);
        return it.first;
      }
    }
  } else {
    if (qIndex < QueuePriority::Total && queuePool_[qIndex].size() > 0) {
      // Search through all available queues for the lowest counter.
      // Note: the map is sorted in the allocation order for possible round-robin selection
      typedef decltype(queuePool_)::value_type::const_reference PoolRef;
      auto lowest = std::min_element(
          queuePool_[qIndex].begin(), queuePool_[qIndex].end(),
          [](PoolRef A, PoolRef B) { return A.second.refCount < B.second.refCount; });
      lowest->second.refCount++;
      ClPrint(amd::LOG_INFO, amd::LOG_QUEUE, "Selected queue refCount: %p (%d)",
              lowest->first->base_address, lowest->second.refCount);
      return lowest->first;
    }
  }
  return nullptr;
}

// ================================================================================================
hsa_queue_t* Device::AcquireActiveNormalQueue() {
  uint32_t queue_size = ROC_AQL_QUEUE_SIZE;
  auto queue = acquireQueue(queue_size, false, std::vector<uint32_t>{},
                            amd::CommandQueue::Priority::Normal, true);
  return queue;
}

// ================================================================================================
hsa_queue_t* Device::acquireQueue(uint32_t queue_size_hint, bool coop_queue,
                                  const std::vector<uint32_t>& cuMask,
                                  amd::CommandQueue::Priority priority, bool managed) {
  amd::ScopedLock l(active_queue_access_);

  assert(queuePool_[QueuePriority::Low].size() <= GPU_MAX_HW_QUEUES ||
         queuePool_[QueuePriority::Normal].size() <= GPU_MAX_HW_QUEUES ||
         queuePool_[QueuePriority::High].size() <= GPU_MAX_HW_QUEUES);

  ClPrint(amd::LOG_INFO, amd::LOG_QUEUE,
          "Number of allocated hardware queues with low priority: %d,"
          " with normal priority: %d, with high priority: %d, maximum per priority is: %d",
          queuePool_[QueuePriority::Low].size(), queuePool_[QueuePriority::Normal].size(),
          queuePool_[QueuePriority::High].size(), GPU_MAX_HW_QUEUES);

  hsa_amd_queue_priority_t queue_priority;
  uint qIndex;
  switch (priority) {
    case amd::CommandQueue::Priority::Low:
      queue_priority = HSA_AMD_QUEUE_PRIORITY_LOW;
      qIndex = QueuePriority::Low;
      break;
    case amd::CommandQueue::Priority::High:
      queue_priority = HSA_AMD_QUEUE_PRIORITY_HIGH;
      qIndex = QueuePriority::High;
      break;
    case amd::CommandQueue::Priority::Normal:
    case amd::CommandQueue::Priority::Medium:
    default:
      queue_priority = HSA_AMD_QUEUE_PRIORITY_NORMAL;
      qIndex = QueuePriority::Normal;
      break;
  }

  // If we have reached the max number of queues, reuse an existing queue with the matching queue
  // priority, choosing the one with the least number of users. Note: Don't attempt to reuse the
  // cooperative queue, since it's single per device
  if (!coop_queue && (cuMask.size() == 0) &&
      ((queuePool_[qIndex].size() == GPU_MAX_HW_QUEUES) || queuePool_[qIndex].size() > 0)) {
    hsa_queue_t* queue = getQueueFromPool(qIndex);
    if (queue != nullptr) {
      if (!managed && (qIndex == QueuePriority::Normal)) {
        num_normal_queues_++;
      }
      return queue;
    }
  }

  // Else create a new queue. This also includes the initial state where there
  // is no queue.
  uint32_t queue_max_packets = 0;
  if (HSA_STATUS_SUCCESS !=
      Hsa::agent_get_info(bkendDevice_, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_max_packets)) {
    DevLogError("Cannot get hsa agent info \n");
    return nullptr;
  }
  auto queue_size = (queue_max_packets < queue_size_hint) ? queue_max_packets : queue_size_hint;

  hsa_queue_t* queue;
  auto queue_type = HSA_QUEUE_TYPE_MULTI;

  // Enable cooperative queue for the device queue
  if (coop_queue) {
    queue_type = HSA_QUEUE_TYPE_COOPERATIVE;
  }

  while (Hsa::queue_create(bkendDevice_, queue_size, queue_type, callbackQueue, this,
                           std::numeric_limits<uint>::max(), std::numeric_limits<uint>::max(),
                           &queue) != HSA_STATUS_SUCCESS) {
    queue_size >>= 1;
    if (queue_size < 64) {
      // if a queue with the same requested priority available from the pool, returns it here
      if (!coop_queue && (cuMask.size() == 0) && (queuePool_[qIndex].size() > 0)) {
        return getQueueFromPool(qIndex);
      }
      DevLogError("Device::acquireQueue: hsa_queue_create failed!");
      return nullptr;
    }
  }

  // default priority is normal so no need to set it again
  if (queue_priority != HSA_AMD_QUEUE_PRIORITY_NORMAL) {
    hsa_status_t st = Hsa::queue_set_priority(queue, queue_priority);
    if (st != HSA_STATUS_SUCCESS) {
      DevLogError("Device::acquireQueue: hsa_amd_queue_set_priority failed!");
      Hsa::queue_destroy(queue);
      return nullptr;
    }
  }

  ClPrint(amd::LOG_INFO, amd::LOG_QUEUE,
          "Created SWq=%p to map on HWq=%p with "
          "size %d with priority %d, cooperative: %i",
          queue, queue->base_address, queue_size, queue_priority, coop_queue);

  Hsa::profiling_set_profiler_enabled(queue, 1);
  if (cuMask.size() != 0 || info_.globalCUMask_.size() != 0) {
    std::stringstream ss;
    ss << std::hex;
    std::vector<uint32_t> mask = {};

    // handle scenarios where cuMask (custom-defined), globalCUMask_ or both are valid and
    // fill the final mask which will be appiled to the current queue
    if (cuMask.size() != 0 && info_.globalCUMask_.size() == 0) {
      mask = cuMask;
    } else if (cuMask.size() != 0 && info_.globalCUMask_.size() != 0) {
      for (unsigned int i = 0; i < std::min(cuMask.size(), info_.globalCUMask_.size()); i++) {
        mask.push_back(cuMask[i] & info_.globalCUMask_[i]);
      }
      // check to make sure after ANDing cuMask (custom-defined) with global
      // CU mask, we have non-zero mask, oterwise just apply global CU mask
      bool zeroCUMask = true;
      for (auto m : mask) {
        if (m != 0) {
          zeroCUMask = false;
          break;
        }
      }
      if (zeroCUMask) {
        mask = info_.globalCUMask_;
      }
    } else {
      mask = info_.globalCUMask_;
    }


    for (int i = mask.size() - 1; i >= 0; i--) {
      ss << std::setfill('0') << std::setw(8) << mask[i];
    }
    ClPrint(amd::LOG_INFO, amd::LOG_QUEUE, "Setting CU mask 0x%s for hardware queue %p",
            ss.str().c_str(), queue->base_address);

    std::vector<uint32_t> final_mask = {};
    // hsa_amd_queue_cu_set_mask expects each bit in cuMask to represent each CU
    // For wgp mode: Each wgp consists of 2 CUs and CUs must be adjacent pairwise enabled
    // Convert each bit in the cuMask from wgp to cu by duplicating it
    if (settings().enableWgpMode_) {
      final_mask.resize(mask.size() * 2, 0);

      for (int i = 0; i < mask.size(); i++) {
        for (int j = 0; j < 16; j++) {
          // Convert least significant 16 bits
          if (((mask[i] >> j) & 0x1) == 0x1) {
            final_mask[2 * i] |= (0x3 << (2 * j));
          }

          // Convert most significant 16 bits
          if (((mask[i] >> (16 + j)) & 0x1) == 0x1) {
            final_mask[2 * i + 1] |= (0x3 << (2 * j));
          }
        }
      }
    } else {
      final_mask = mask;
    }

    hsa_status_t status =
        Hsa::queue_cu_set_mask(queue, final_mask.size() * 32, final_mask.data());
    if (status != HSA_STATUS_SUCCESS) {
      DevLogError("Device::acquireQueue: hsa_amd_queue_cu_set_mask failed!");
      Hsa::queue_destroy(queue);
      return nullptr;
    }
    if (cuMask.size() != 0) {
      // add queues with custom CU mask into their special pool to keep track
      // of mapping of these queues to their associated queueInfo (i.e., hostcall buffers)
      auto result = queueWithCUMaskPool_[qIndex].emplace(std::make_pair(queue, QueueInfo()));
      assert(result.second && "QueueInfo already exists");
      auto& qInfo = result.first->second;
      qInfo.refCount = 1;

      return queue;
    }
  }

  if (coop_queue) {
    // Skip queue recycling for cooperative queues, since it should be just one
    // per device.
    return queue;
  }
  auto result = queuePool_[qIndex].emplace(std::make_pair(queue, QueueInfo()));
  assert(result.second && "QueueInfo already exists");
  auto& qInfo = result.first->second;
  qInfo.refCount = 1;
  ClPrint(amd::LOG_INFO, amd::LOG_QUEUE, "acquireQueue refCount: %p (%d)",
          result.first->first->base_address, result.first->second.refCount);
  if (!managed && (cuMask.size() == 0) && (qIndex = QueuePriority::Normal)) {
    num_normal_queues_++;
  }
  return queue;
}

// ================================================================================================
bool Device::ReleaseActiveNormalQueue(hsa_queue_t* queue) {
  // Release a queue if the total number of allocated queues exceeds the max possible
  if (num_normal_queues_.load() > GPU_MAX_HW_QUEUES) {
    releaseQueue(queue, std::vector<uint32_t>{}, false, true);
    return true;
  } else {
    return false;
  }
}

// ================================================================================================
void Device::releaseQueue(hsa_queue_t* queue, const std::vector<uint32_t>& cuMask, bool coop_queue,
                          bool managed) {
  amd::ScopedLock l(active_queue_access_);
  for (auto& it : cuMask.size() == 0 ? queuePool_ : queueWithCUMaskPool_) {
    auto qIter = it.find(queue);
    if (qIter != it.end()) {
      if (!managed && (cuMask.size() == 0) && (&it == &queuePool_[QueuePriority::Normal])) {
        num_normal_queues_--;
      }
      auto& qInfo = qIter->second;
      assert(qInfo.refCount > 0);
      qInfo.refCount--;
      ClPrint(amd::LOG_INFO, amd::LOG_QUEUE, "releaseQueue refCount:%p (%d)",
              qIter->first->base_address, qIter->second.refCount);
      // hsa queues with cumask set are not being reused. Hence, if the app uses multiple
      // such queues it can cause memory leak and those must be destroyed here once the
      // refcount reaches 0.
      if ((!cuMask.empty()) && (qInfo.refCount == 0)) {
        if (qInfo.hostcallBuffer_) {
          ClPrint(amd::LOG_INFO, amd::LOG_QUEUE,
                  "Deleting hostcall buffer %p for hardware queue %p", qInfo.hostcallBuffer_,
                  qIter->first->base_address);
          amd::disableHostcalls(qInfo.hostcallBuffer_);
          context().svmFree(qInfo.hostcallBuffer_);
        }
        ClPrint(amd::LOG_INFO, amd::LOG_QUEUE, "Deleting hardware queue %p with refCount 0",
                queue->base_address);
        qIter = it.erase(qIter);
        Hsa::queue_destroy(queue);
      }
    }
  }
  if (coop_queue) {  // cooperative queue
    ClPrint(amd::LOG_INFO, amd::LOG_QUEUE, "Deleting CG enabled hardware queue %p ",
            queue->base_address);
    Hsa::queue_destroy(queue);
  }
}

void* Device::getOrCreateHostcallBuffer(hsa_queue_t* queue, bool coop_queue,
                                        const std::vector<uint32_t>& cuMask) {
  decltype(queuePool_)::value_type::iterator qIter;
  bool found = false;
  if (!coop_queue) {
    for (auto& it : cuMask.size() == 0 ? queuePool_ : queueWithCUMaskPool_) {
      qIter = it.find(queue);
      if (qIter != it.end()) {
        found = true;
        break;
      }
    }
    assert(found && "Couldn't find queue");

    if (qIter->second.hostcallBuffer_) {
      return qIter->second.hostcallBuffer_;
    }
  } else {
    if (coopHostcallBuffer_) {
      return coopHostcallBuffer_;
    }
  }

  // The number of packets required in each buffer is at least equal to the
  // maximum number of waves supported by the device.
  auto wavesPerCu = info().maxThreadsPerCU_ / info().wavefrontWidth_;
  auto numPackets = info().maxComputeUnits_ * wavesPerCu;

  auto size = amd::getHostcallBufferSize(numPackets);
  auto align = amd::getHostcallBufferAlignment();

  void* buffer = context().svmAlloc(size, align, CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS);
  if (!buffer) {
    ClPrint(amd::LOG_ERROR, amd::LOG_QUEUE,
            "Failed to create hostcall buffer for hardware queue %p", queue->base_address);
    return nullptr;
  }
  ClPrint(amd::LOG_INFO, amd::LOG_QUEUE, "Created hostcall buffer %p for hardware queue %p", buffer,
          queue->base_address);
  if (!coop_queue) {
    qIter->second.hostcallBuffer_ = buffer;
  } else {
    coopHostcallBuffer_ = buffer;
  }
  if (!amd::enableHostcalls(*this, buffer, numPackets)) {
    ClPrint(amd::LOG_ERROR, amd::LOG_QUEUE, "Failed to register hostcall buffer %p with listener",
            buffer);
    return nullptr;
  }
  return buffer;
}

bool Device::findLinkInfo(const amd::Device& other_device, std::vector<LinkAttrType>* link_attrs) {
  return findLinkInfo((static_cast<const roc::Device*>(&other_device))->gpuvm_segment_, link_attrs);
}

bool Device::findLinkInfo(const hsa_amd_memory_pool_t& pool,
                          std::vector<LinkAttrType>* link_attrs) {
  if ((!pool.handle) || (link_attrs == nullptr)) {
    return false;
  }

  // Retrieve the hops between 2 devices.
  int32_t hops = 0;
  hsa_status_t hsa_status = Hsa::agent_memory_pool_get_info(
      bkendDevice_, pool, HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS, &hops);

  if (hsa_status != HSA_STATUS_SUCCESS) {
    DevLogPrintfError("Cannot get hops info, hsa failed with status: %d", hsa_status);
    return false;
  }

  if (hops < 0) {
    return false;
  }

  // The pool is on its agent
  if (hops == 0) {
    for (auto& link_attr : (*link_attrs)) {
      switch (link_attr.first) {
        case kLinkLinkType: {
          // No link, so type is meaningless,
          // caller should ignore it
          link_attr.second = -1;
          break;
        }
        case kLinkHopCount: {
          // no hop
          link_attr.second = 0;
          break;
        }
        case kLinkDistance: {
          // distance is zero, if no hops
          link_attr.second = 0;
          break;
        }
        case kLinkAtomicSupport: {
          // atomic support if its on the same agent
          link_attr.second = 1;
          break;
        }
        default: {
          DevLogPrintfError("Invalid LinkAttribute: %d ", link_attr.first);
          return false;
        }
      }
    }
    return true;
  }

  // Retrieve link info on the pool.
  std::vector<hsa_amd_memory_pool_link_info_t> link_info(hops);
  hsa_status = Hsa::agent_memory_pool_get_info(
      bkendDevice_, pool, HSA_AMD_AGENT_MEMORY_POOL_INFO_LINK_INFO, link_info.data());

  if (hsa_status != HSA_STATUS_SUCCESS) {
    DevLogPrintfError("Cannot retrieve link info, hsa failed with status: %d", hsa_status);
    return false;
  }

  for (auto& link_attr : (*link_attrs)) {
    switch (link_attr.first) {
      case kLinkLinkType: {
        link_attr.second = static_cast<int32_t>(link_info[0].link_type);
        break;
      }
      case kLinkHopCount: {
        uint32_t distance = 0;
        // Because of Rocrs limitation hops is set to 1 always between two different devices
        // If Rocr Changes the behaviour revisit this logic
        for (size_t hop_idx = 0; hop_idx < static_cast<size_t>(hops); ++hop_idx) {
          distance += link_info[hop_idx].numa_distance;
        }
        uint32_t oneHopDistance = (link_info[0].link_type == HSA_AMD_LINK_INFO_TYPE_XGMI) ? 13 : 20;
        link_attr.second = static_cast<int32_t>(distance / oneHopDistance);
        break;
      }
      case kLinkDistance: {
        uint32_t distance = 0;
        // Sum of distances between hops
        for (size_t hop_idx = 0; hop_idx < static_cast<size_t>(hops); ++hop_idx) {
          distance += link_info[hop_idx].numa_distance;
        }
        link_attr.second = static_cast<int32_t>(distance);
        break;
      }
      case kLinkAtomicSupport: {
        // if either of the atomic is supported
        link_attr.second = static_cast<int32_t>(link_info[0].atomic_support_64bit ||
                                                link_info[0].atomic_support_32bit);
        break;
      }
      default: {
        DevLogPrintfError("Invalid LinkAttribute: %d ", link_attr.first);
        return false;
      }
    }
  }

  return true;
}

// ================================================================================================
void Device::getGlobalCUMask(std::string cuMaskStr) {
  if (cuMaskStr.length() != 0) {
    std::string pre = cuMaskStr.substr(0, 2);
    if (pre.compare("0x") == 0 || pre.compare("0X") == 0) {
      cuMaskStr = cuMaskStr.substr(2, cuMaskStr.length());
    }

    int end = cuMaskStr.length();

    // the number of current physical CUs compressed in 4-bits
    size_t compPhysicalCUs = static_cast<size_t>(
        (settings().enableWgpMode_ ? info_.maxComputeUnits_ * 2 : info_.maxComputeUnits_) / 4);

    // the number of final available compute units after applying the requested CU mask
    uint32_t availCUs = 0;

    // read numCharToRead characters (8 or less) from the cuMask string each time, convert
    // it into hex, and store it into the globalCUMask_. If the length of the cuMask string
    // is more than the compressed physical available CUs, ignore the rest
    for (unsigned i = 0; i < std::min(cuMaskStr.length(), compPhysicalCUs); i += 8) {
      int numCharToRead = (i + 8 <= compPhysicalCUs) ? 8 : compPhysicalCUs - 8;
      std::string temp =
          cuMaskStr.substr(std::max(0, end - numCharToRead), std::min(numCharToRead, end));
      end -= numCharToRead;
      unsigned long ul = 0;
      try {
        ul = std::stoul(temp, 0, 16);
      } catch (const std::invalid_argument&) {
        info_.globalCUMask_ = {};
        availCUs = 0;
        break;
      }
      info_.globalCUMask_.push_back(static_cast<uint32_t>(ul));
      // count number of set bits in ul to find the number of active CUs
      // in each iteration
      while (ul) {
        ul &= (ul - 1);
        availCUs++;
      }
    }
    // update the maxComputeUnits_ based on the requested CU mask
    if (availCUs != 0 && availCUs < compPhysicalCUs * 4) {
      info_.maxComputeUnits_ = settings().enableWgpMode_ ? availCUs / 2 : availCUs;
    } else {
      info_.globalCUMask_ = {};
    }
  } else {
    info_.globalCUMask_ = {};
  }
}

// ================================================================================================
device::Signal* Device::createSignal() const { return new roc::Signal(); }

// ================================================================================================
hsa_status_t Device::BackendErrorCallBackHandler(const hsa_amd_event_t* event, void* data) {
  cl_int gpu_error = CL_SUCCESS;
  switch (event->event_type) {
    case HSA_AMD_GPU_MEMORY_FAULT_EVENT:
      gpu_error = CL_INVALID_MEM_OBJECT;
      LogError("Memory Fault Error");
      break;
    case HSA_AMD_GPU_HW_EXCEPTION_EVENT:
      gpu_error = CL_INVALID_OPERATION;
      LogError("HW Exception Error");
      break;
    case HSA_AMD_GPU_MEMORY_ERROR_EVENT:
      gpu_error = CL_DEVICE_NOT_AVAILABLE;
      LogError("GPU Memory Error");
      break;
    default:
      gpu_error = CL_DEVICE_NOT_AVAILABLE;
      LogError("Unknown Event Type ");
      break;
  }

  // Execute the default handler if a GPU core file should be generated ...
  if (amd::Os::DumpCoreFile() || !HIP_SKIP_ABORT_ON_GPU_ERROR) {
    return HSA_STATUS_ERROR;
  }

  gpu_error_ = gpu_error;
  return HSA_STATUS_SUCCESS;
}

// ================================================================================================
void Device::RegisterBackendErrorCb() {
  // Register ROCclr Error Callback
  hsa_status_t hsa_error = HSA_STATUS_SUCCESS;
  hsa_error = Hsa::register_system_event_handler(BackendErrorCallBackHandler, nullptr);
  if (hsa_error != HSA_STATUS_SUCCESS) {
    LogError("Cannot Register Call back event handler");
  }
}
// ================================================================================================
amd::Memory* Device::GetArenaMemObj(const void* ptr, size_t& offset, size_t size) {
  // Only create arena_mem_object if CPU memory is accessible from HMM
  // or if runtime received an interop from another ROCr's client
  // Disable arena for XNACK
  hsa_amd_pointer_info_t ptr_info = {};
  ptr_info.size = sizeof(hsa_amd_pointer_info_t);
  if (!IsValidAllocation(ptr, size, &ptr_info)) {
    return nullptr;
  }

  if (arena_mem_obj_ == nullptr) {
    arena_mem_obj_ = new (context()) amd::ArenaMemory(context());
    if ((arena_mem_obj_ != nullptr) && !arena_mem_obj_->create(nullptr)) {
      LogError("Arena Memory Creation failed!");
      arena_mem_obj_->release();
      arena_mem_obj_ = nullptr;
    }
    if (arena_mem_obj_ == nullptr) {
      return arena_mem_obj_;
    }
  }

  // Calculate the offset of the pointer.
  const void* dev_ptr = reinterpret_cast<void*>(
      arena_mem_obj_->getDeviceMemory(*arena_mem_obj_->getContext().devices()[0])
          ->virtualAddress());
  offset = reinterpret_cast<size_t>(ptr) - reinterpret_cast<size_t>(dev_ptr);

  return arena_mem_obj_;
}

// ================================================================================================
void Device::ReleaseGlobalSignal(void* signal) const {
  if (signal != nullptr) {
    reinterpret_cast<ProfilingSignal*>(signal)->release();
  }
}

// ================================================================================================
bool Device::CreateUserEvent(amd::UserEvent* event) const {
  std::unique_ptr<ProfilingSignal> signal(new ProfilingSignal());
  if ((signal == nullptr) ||
      (HSA_STATUS_SUCCESS != Hsa::signal_create(0, 0, nullptr, &signal->signal_))) {
    return false;
  }
  Hsa::signal_silent_store_relaxed(signal->signal_, kInitSignalValueOne);
  event->SetHwEvent(signal.release());
  return true;
}

// ================================================================================================
void Device::SetUserEvent(amd::UserEvent* event) const {
  auto signal = reinterpret_cast<ProfilingSignal*>(event->HwEvent());
  assert(signal != nullptr && "Can't have user event without hw event!");
  Hsa::signal_silent_store_relaxed(signal->signal_, 0);
}

// ================================================================================================
bool Device::IsValidAllocation(const void* dev_ptr, size_t size, hsa_amd_pointer_info_t* ptr_info) {
  // Query ptr type to see if it's a HMM allocation
  hsa_status_t status =
      Hsa::pointer_info(const_cast<void*>(dev_ptr), ptr_info, nullptr, nullptr, nullptr);
  // The call should never fail in ROCR, but just check for an error and continue
  if (status != HSA_STATUS_SUCCESS) {
    LogError("hsa_amd_pointer_info() failed");
  }

  if (ptr_info->type == HSA_EXT_POINTER_TYPE_RESERVED_ADDR) {
    return false;
  }

  // Return false for pinned memory. A true return may result in a race because
  // ROCclr may attempt to do a pin/copy/unpin underneath in a multithreaded environment
  if (ptr_info->type == HSA_EXT_POINTER_TYPE_LOCKED) {
    return false;
  }

  if (ptr_info->type != HSA_EXT_POINTER_TYPE_UNKNOWN) {
    if ((size != 0) && ((reinterpret_cast<const_address>(dev_ptr) -
                         reinterpret_cast<const_address>(ptr_info->agentBaseAddress)) > size)) {
      return false;
    }
    return true;
  }
  return false;
}

// ================================================================================================
void Device::HiddenHeapAlloc(const VirtualGPU& gpu) {
  auto HeapAllocOnly = [this, &gpu]() -> bool {
    // Allocate initial heap for device memory allocator
    static constexpr size_t HeapBufferSize = 128 * Ki;
    heap_buffer_ = createMemory(HeapBufferSize);
    if (initial_heap_size_ != 0) {
      initial_heap_size_ = amd::alignUp(initial_heap_size_, 2 * Mi);
      initial_heap_buffer_ = createMemory(initial_heap_size_);
    }
    if (heap_buffer_ == nullptr) {
      LogError("Heap buffer allocation failed!");
      return false;
    }
    return true;
  };
  std::call_once(heap_allocated_, HeapAllocOnly);
}

// ================================================================================================
void Device::HiddenHeapInit(const VirtualGPU& gpu) {
  auto HeapZeroOut = [this, &gpu]() -> bool {
    static constexpr size_t HeapBufferSize = 128 * Ki;
    bool result = static_cast<const KernelBlitManager&>(gpu.blitMgr())
                      .initHeap(heap_buffer_, initial_heap_buffer_, HeapBufferSize,
                                initial_heap_size_ / (2 * Mi));

    return result;
  };
  std::call_once(heap_initialized_, HeapZeroOut);
}

// ================================================================================================
void Device::getSdmaRWMasks(uint32_t* readMask, uint32_t* writeMask) const {
  *readMask = maxSdmaReadMask_;
  *writeMask = maxSdmaWriteMask_;
}

// ================================================================================================
void Device::AddKernel(Kernel& gpuKernel) const {
  amd::ScopedLock lock(vgpusAccess());
  kernel_map_.insert({gpuKernel.KernelCodeHandle(), gpuKernel});
}

// ================================================================================================
void Device::RemoveKernel(Kernel& gpuKernel) const {
  if (gpuKernel.KernelCodeHandle() != 0) {
    amd::ScopedLock lock(vgpusAccess());
    auto it = kernel_map_.find(gpuKernel.KernelCodeHandle());
    if (it != kernel_map_.end()) {
      // Remove the old mapping
      kernel_map_.erase(it);
    }
  }
}

// ================================================================================================
ProfilingSignal::~ProfilingSignal() {
  if (signal_.handle != 0) {
    if (Hsa::signal_load_relaxed(signal_) > 0
        && !(HIP_SKIP_ABORT_ON_GPU_ERROR && amd::Device::IsGPUInError())) {
      LogError("Runtime shouldn't destroy a signal that is still busy!");
      if (Hsa::signal_wait_scacquire(signal_, HSA_SIGNAL_CONDITION_LT, kInitSignalValueOne,
                                    kUnlimitedWait, HSA_WAIT_STATE_BLOCKED) != 0) {
      }
    }
    Hsa::signal_destroy(signal_);
  }
}

// ================================================================================================
cl_int ConvertHSAErrorIntoCLError(hsa_status_t hsa_status) {
  cl_int cl_error = CL_SUCCESS;
  switch (hsa_status) {
    case HSA_STATUS_ERROR_EXCEPTION:
      cl_error = CL_INVALID_OPERATION;
      break;
    case HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS:
      cl_error = CL_INVALID_ARG_VALUE;
      break;
    case HSA_STATUS_ERROR_INVALID_ALLOCATION:
      cl_error = CL_MEM_OBJECT_ALLOCATION_FAILURE;
      break;
    case HSA_STATUS_ERROR_INVALID_CODE_OBJECT:
      cl_error = CL_INVALID_PROGRAM;
      break;
    case HSA_STATUS_ERROR_INVALID_PACKET_FORMAT:
      cl_error = CL_INVALID_OPERATION;
      break;
    case HSA_STATUS_ERROR_INVALID_ARGUMENT:
      cl_error = CL_INVALID_ARG_VALUE;
      break;
    case HSA_STATUS_ERROR_INVALID_ISA:
      cl_error = CL_INVALID_KERNEL;
      break;
    case (hsa_status_t)HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION:
      cl_error = CL_BUILD_PROGRAM_FAILURE;
      break;
    case (hsa_status_t)HSA_STATUS_ERROR_MEMORY_FAULT:
      cl_error = CL_INVALID_MEM_OBJECT;
      break;
    case (hsa_status_t)HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION:
      cl_error = CL_INVALID_MEM_OBJECT;
      break;
    case HSA_STATUS_ERROR:
    default:
      cl_error = CL_DEVICE_NOT_AVAILABLE;
      break;
  }
  return cl_error;
}

// ================================================================================================
void callbackQueue(hsa_status_t status, hsa_queue_t* queue, void* data) {
  if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
    Device* dev = reinterpret_cast<Device*>(data);
    for (auto it : dev->vgpus()) {
      roc::VirtualGPU* vgpu = reinterpret_cast<roc::VirtualGPU*>(it);
      if (vgpu->gpu_queue() == queue) {
        vgpu->AnalyzeAqlQueue();
      }
    }
    // Abort on device exceptions.
    const char* errorMsg = 0;
    Hsa::status_string(status, &errorMsg);
    if (status == HSA_STATUS_ERROR_OUT_OF_RESOURCES) {
      size_t global_available_mem = 0;
      if (HSA_STATUS_SUCCESS !=
          Hsa::agent_get_info(dev->getBackendDevice(),
                             static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_MEMORY_AVAIL),
                             &global_available_mem)) {
        LogError("HSA_AMD_AGENT_INFO_MEMORY_AVAIL query failed.");
      }
      ClPrint(amd::LOG_NONE, amd::LOG_ALWAYS,
              "Callback: Queue %p Aborting with error : %s Code: 0x%x Available Free mem : %zu MB",
              queue->base_address, errorMsg, status, global_available_mem / Mi);
    } else {
      ClPrint(amd::LOG_NONE, amd::LOG_ALWAYS,
              "Callback: Queue %p aborting with error : %s code: 0x%x", queue->base_address,
              errorMsg, status);
    }

    if (amd::Os::DumpCoreFile() || !HIP_SKIP_ABORT_ON_GPU_ERROR) {
      abort();
    }
    amd::Device::gpu_error_ = ConvertHSAErrorIntoCLError(status);
  }
}

// ================================================================================================
#if defined(__clang__)
#if __has_feature(address_sanitizer)
device::UriLocator* Device::createUriLocator() const { return new roc::UriLocator(); }
#endif
#endif
}  // namespace amd::roc
