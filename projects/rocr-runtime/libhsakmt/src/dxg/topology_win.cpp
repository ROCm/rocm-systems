/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cctype>
#include <cmath>
#include <string>
#include <vector>
#include <assert.h>
#include "impl/wddm/types.h"
#include "impl/wddm/device.h"
#include "util/utils.h"
#include "util/os.h"
#include <bit>
#include <bitset>
#include "topology.hpp"

struct proc_cpu_info {
  uint32_t group;                         //!< id of group
  uint32_t proc_num;                      //!< id of the logical processor in the group
  uint32_t apicid;                        //!< extended apicid to support 256+ logical processors
  uint32_t numa_node;                     //!< numa node of this logical processor
  char model_name[HSA_PUBLIC_NAME_SIZE];  //!< model name
};

struct proc_numa_node_info {
  uint32_t numa_node;  //!< numa node number
  uint32_t ccompute_id_low;  //!< extended apicid of the logical processor of the lowest ID in
                             //!< the primary group of this node.
  uint32_t count;      //!< count of logical processors on this node
  bool operator<(const proc_numa_node_info& other) const {
    return numa_node < other.numa_node;
  }
};

HSAKMT_STATUS topology_map_node_id(uint32_t node_id,
                                   wsl::thunk::WDDMDevice *&device) {
  if (dxg_topology->wdevices_.empty() || node_id < dxg_topology->numa_node_count_ ||
      node_id >= dxg_topology->num_sysfs_nodes) {
    device = nullptr;
    return HSAKMT_STATUS_ERROR;
  }

  device = dxg_topology->wdevices_[node_id - dxg_topology->numa_node_count_];
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS topology_sysfs_get_system_props(HsaSystemProperties& props) {
  std::memset(&props, 0, sizeof(props));

  dxg_runtime->HeapFini();
  for (auto device : dxg_topology->wdevices_)
    delete device;
  dxg_topology->wdevices_.clear();

  WDDMCreateDevices(dxg_topology->wdevices_);
  const auto num_adapters = static_cast<uint32_t>(dxg_topology->wdevices_.size());
  if (num_adapters == 0) {
    pr_err("No WDDM adapters found.\n");
    return HSAKMT_STATUS_ERROR;
  }

  dxg_topology->num_sysfs_nodes = dxg_topology->numa_node_count_ + num_adapters;
  dxg_runtime->HeapInit();
  props.NumNodes = dxg_topology->num_sysfs_nodes;
  if (dxg_runtime->default_node > num_adapters)
    dxg_runtime->default_node = num_adapters;

  return HSAKMT_STATUS_SUCCESS;
}

static HSAKMT_STATUS topology_get_cpu_model_name(HsaNodeProperties& props,
                                                 const std::vector<proc_cpu_info>& cpu_info) {
  for (const auto& info : cpu_info) {
    if (info.apicid == props.CComputeIdLo) {
      strncpy(reinterpret_cast<char*>(props.AMDName), info.model_name, sizeof(props.AMDName));
      /* Convert from UTF8 to UTF16 */
      size_t j = 0;
      for (; info.model_name[j] != '\0' && j < (HSA_PUBLIC_NAME_SIZE - 1); j++) {
        props.MarketingName[j] = info.model_name[j];
      }
      props.MarketingName[j] = '\0';
      break;
    }
  }
  return HSAKMT_STATUS_SUCCESS;
}

static bool parse_cpu_model_name(char* cpu_model_name, size_t size) {
  constexpr const char* subKey = "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0";
  constexpr const char* value_name = "ProcessorNameString";

  HKEY hKey;
  if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, subKey, 0, KEY_READ, &hKey) != ERROR_SUCCESS) {
    pr_err("Failed to open registry key\n");
    return false;
  }

  char value[256];
  DWORD value_size = sizeof(value);
  const auto status = RegQueryValueExA(hKey, value_name, nullptr, nullptr,
                                       reinterpret_cast<LPBYTE>(value), &value_size);
  RegCloseKey(hKey);

  if (status != ERROR_SUCCESS) {
    pr_err("Failed to query registry value. Error code: %lu\n", status);
    return false;
  }

  strncpy(cpu_model_name, value, size);
  std::cout << "Processor Name: " << cpu_model_name << std::endl;
  return true;
}

static HSAKMT_STATUS topology_parse_cpu_info(std::vector<proc_cpu_info>& cpu_info) {
  rocr::os::cpuid_t cpuid{};
  rocr::os::ParseCpuID(&cpuid);
  dxg_topology->processor_vendor = topology_search_processor_vendor(cpuid.ManufacturerID);

  if (dxg_topology->processor_vendor < 0) {
    pr_err("Failed to get Processor Vendor. Setting to %s",
           supported_processor_vendor_name[GENUINE_INTEL]);
    dxg_topology->processor_vendor = GENUINE_INTEL;
  }
  dxg_topology->freq_max_ = static_cast<double>(rocr::os::SystemClockFrequency());

  // Get model name
  char model[HSA_PUBLIC_NAME_SIZE] = {0};
  if (!parse_cpu_model_name(model, sizeof(model))) {
    return HSAKMT_STATUS_BUFFER_TOO_SMALL;
  }

  // Get processor topology
  ULONG buffer_size = 0;
  GetSystemCpuSetInformation(nullptr, 0, &buffer_size, 0, 0);
  std::vector<BYTE> buffer(buffer_size);
  if (!GetSystemCpuSetInformation(reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buffer.data()),
                                  buffer_size, &buffer_size, 0, 0)) {
    pr_err("GetSystemCpuSetInformation(%lu) failed\n", buffer_size);
    return HSAKMT_STATUS_ERROR;
  }

  ULONG offset = 0;
  while (offset < buffer_size) {
    auto* info = reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buffer.data() + offset);
    if (info->Type == CpuSetInformation) {
      proc_cpu_info cpu;
      cpu.proc_num = info->CpuSet.LogicalProcessorIndex;
      cpu.group = info->CpuSet.Group;
      cpu.apicid = info->CpuSet.Id;  // x2APIC ID
      cpu.numa_node = info->CpuSet.NumaNodeIndex;
      strncpy(cpu.model_name, model, sizeof(cpu.model_name));
      cpu_info.push_back(cpu);
    }
    offset += info->Size;
  }
  return HSAKMT_STATUS_SUCCESS;
}

static HSAKMT_STATUS topology_parse_numa_node_info(
                                      std::vector<proc_numa_node_info>& numa_node_info,
                                      const std::vector<proc_cpu_info>& cpu_info) {
  DWORD length = 0;
  GetLogicalProcessorInformationEx(RelationNumaNodeEx, nullptr, &length);
  std::vector<char> buffer(length);
  auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());
  const auto* end = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data() + length);

  if (!GetLogicalProcessorInformationEx(RelationNumaNodeEx, info, &length)) {
    pr_err("GetLogicalProcessorInformationEx failed\n");
    return HSAKMT_STATUS_ERROR;
  }

  while (info < end) {
    if (info->Relationship == RelationNumaNode) {
      const auto& numa_node = info->NumaNode;
      const GROUP_AFFINITY& ga = numa_node.GroupMask;
      proc_numa_node_info node_info = {numa_node.NodeNumber, 0, 0};

      // Query apicid on primary group
      const auto expected_logical_id = std::countr_zero(ga.Mask);
      for (const auto& cpu : cpu_info) {
        if (cpu.group == ga.Group && cpu.proc_num == expected_logical_id) {
          node_info.ccompute_id_low = cpu.apicid;
          break;  // Should always find matched info
        }
      }

      // Query count of logical processors on the node
      auto group_count = numa_node.GroupCount;
      if (group_count == 0) {
        // Before Windows 20H2
        group_count = 1;
      }
      for (uint32_t j = 0; j < group_count; j++) {
        const GROUP_AFFINITY& group_affinity = numa_node.GroupMasks[j];
        node_info.count += std::bitset<sizeof(group_affinity.Mask) * 8>(group_affinity.Mask).count();
      }
      numa_node_info.push_back(node_info);
    }
    info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
                                                     reinterpret_cast<char*>(info) + info->Size);
  }

  std::sort(numa_node_info.begin(), numa_node_info.end());
  return HSAKMT_STATUS_SUCCESS;
}

static HSAKMT_STATUS topology_sysfs_get_node_props(uint32_t node_id,
                                                   HsaNodeProperties& props,
                                                   bool& p2p_links,
                                                   uint32_t& num_p2pLinks) {
  p2p_links = false;
  num_p2pLinks = 0;
  props.MaxEngineClockMhzCCompute = dxg_topology->freq_max_;

  if (node_id < dxg_topology->numa_node_count_) {
    return HSAKMT_STATUS_SUCCESS;
  }

  // GPU node
  wsl::thunk::WDDMDevice *device;
  if (const auto ret = topology_map_node_id(node_id, device); ret != HSAKMT_STATUS_SUCCESS) {
    return ret;
  }

  props.NumCPUCores = 0;
  props.NumFComputeCores = device->SimdPerCu() * device->ComputeUnitCount();
  props.NumMemoryBanks = 1;
  props.NumCaches = 3;
  props.NumIOLinks = 1;
  props.CComputeIdLo = 0;
  props.FComputeIdLo = 0;
  props.Capability.ui32.ASICRevision = device->AsicRevision();
  props.Capability.ui32.WatchPointsTotalBits =
      std::log2(device->WatchPointsNum());
  props.MaxWavesPerSIMD = device->WavePerCu() / device->SimdPerCu();
  props.LDSSizeInKB = device->LdsSize() / 1024;
  props.GDSSizeInKB = 0;
  props.WaveFrontSize = device->WavefrontSize();
  props.NumShaderBanks = device->NumShaderEngine();
  props.NumArrays = device->ShaderArrayPerShaderEngine();
  props.NumCUPerArray = device->ComputeUnitCount() / props.NumArrays;
  props.NumSIMDPerCU = device->SimdPerCu();
  props.MaxSlotsScratchCU = device->MaxScratchSlotsPerCu();
  props.VendorId = 0x1002;
  props.DeviceId = device->DeviceId();
  props.LocationId = device->PciBusAddr();
  props.LocalMemSize = 0;
  props.MaxEngineClockMhzFCompute = device->MaxEngineClockMhz();
  props.DrmRenderMinor = node_id;
  props.Capability2.ui32.AqlEmulationPm4_ = device->IsAqlSupported() ? 0 : 1;

  {
    const char* name = device->ProductName();
    size_t i = 0;
    for (; name[i] != 0 && i < HSA_PUBLIC_NAME_SIZE - 1; i++) {
      props.MarketingName[i] = name[i];
    }
    props.MarketingName[i] = '\0';
  }
  props.uCodeEngineVersions.uCodeSDMA = device->GetSdmaFwVersion();
  props.DebugProperties.Value = 0;
  props.HiveID = 0;
  props.NumSdmaEngines = device->NumSdmaEngine();
  props.NumSdmaXgmiEngines = 0;
  props.NumSdmaQueuesPerEngine = 6; // TODO
  props.NumCpQueues = device->GetNumCpQueues();
  props.NumGws = 0;
  /*
   * In Native Linux, if the asic is APU, this value will be set to 1,
   * if the asic is dGPU, this value will be set to 0. clr use this info
   * to set hostUnifiedMemory_, but for now wsl does not support this feature.
   * Therefore, fore vaule to 0 temporarily.
   */
  props.Integrated = 0;
  props.Domain = device->Domain();
  props.UniqueID = device->Uuid();
  props.NumXcc = device->NumXcc();
  props.KFDGpuID = device->DeviceId(); // TODO
  props.FamilyID = device->GfxFamily();
  props.LuidLowPart = device->GetLuid().LowPart;
  props.LuidHighPart = device->GetLuid().HighPart;

  props.EngineId.ui32.uCode = device->GetMecFwVersion();
  if (const char* envvar = getenv("HSA_OVERRIDE_GFX_VERSION"); envvar) {
    char dummy = '\0';
    uint32_t major = 0, minor = 0, step = 0;
    // HSA_OVERRIDE_GFX_VERSION=major.minor.stepping
    if ((sscanf(envvar, "%u.%u.%u%c", &major, &minor, &step, &dummy) != 3) ||
        (major > 63 || minor > 255 || step > 255)) {
      pr_err("HSA_OVERRIDE_GFX_VERSION %s is invalid\n", envvar);
      return HSAKMT_STATUS_ERROR;
    }
    props.OverrideEngineId.ui32.Major = major & 0x3f;
    props.OverrideEngineId.ui32.Minor = minor & 0xff;
    props.OverrideEngineId.ui32.Stepping = step & 0xff;
  } else {
    props.EngineId.ui32.Major = device->Major();
    props.EngineId.ui32.Minor = device->Minor();
    props.EngineId.ui32.Stepping = device->Stepping();
  }

  snprintf(reinterpret_cast<char*>(props.AMDName), sizeof(props.AMDName) - 1, "GFX%06x",
           HSA_GET_GFX_VERSION_FULL(props.EngineId.ui32));

  if (!dxg_runtime->is_svm_api_supported) {
    props.Capability.ui32.SVMAPISupported = 0;
  }
  props.Capability.ui32.DoorbellType = 2;

  // Get VGPR/SGPR size in byte per CU
  props.SGPRSizePerCU = SGPR_SIZE_PER_CU;
  props.VGPRSizePerCU = get_vgpr_size_per_cu(props.EngineId);

  if (props.NumFComputeCores) {
    assert(props.EngineId.ui32.Major && "HSA_OVERRIDE_GFX_VERSION may be needed");
  }

  return HSAKMT_STATUS_SUCCESS;
}

static HSAKMT_STATUS topology_sysfs_get_mem_props(uint32_t node_id,
                                                  uint32_t mem_id,
                                                  HsaMemoryProperties& props) {
  std::memset(&props, 0, sizeof(props));

  if (node_id < dxg_topology->numa_node_count_) {
    // CPU node
    props.HeapType = HSA_HEAPTYPE_SYSTEM;
    props.SizeInBytes = rocr::os::HostTotalPhysicalMemory();
    // props.SizeInBytes is the actual physical system
    // memory size. Reserve 1/16th for WSL system usage.
    dxg_runtime->max_single_alloc_size = props.SizeInBytes - (props.SizeInBytes >> 4);
    props.Flags.MemoryProperty = 0;
    // TODO: sudo dmidecode --type memory doesn't work on wsl
    props.Width = 64;
    props.MemoryClockMax = 2133;
    return HSAKMT_STATUS_SUCCESS;
  }

  wsl::thunk::WDDMDevice *device;
  if (const auto ret = topology_map_node_id(node_id, device); ret != HSAKMT_STATUS_SUCCESS) {
    return ret;
  }

  props.HeapType = HSA_HEAPTYPE_FRAME_BUFFER_PRIVATE;
  props.SizeInBytes = device->LocalHeapSize();
  props.Width = device->MemoryBusWidth();
  props.MemoryClockMax = device->MaxMemoryClockMhz();

  return HSAKMT_STATUS_SUCCESS;
}

static inline void map_cache(HsaCacheProperties& to, const CACHE_RELATIONSHIP& from) {
  to.ProcessorIdLow = 0;
  to.CacheLevel = from.Level;
  to.CacheSize = from.CacheSize;
  to.CacheLineSize = from.LineSize;
  to.CacheLinesPerTag = 0;  // Windows doesn't expose it
  to.CacheAssociativity = from.Associativity;
  to.CacheLatency = 0;      // Windows doesn't expose it
  to.CacheType.ui32.CPU = 1;

  switch (from.Type) {
    case CacheData:
      to.CacheType.ui32.Data = 1;
      break;
    case CacheInstruction:
      to.CacheType.ui32.Instruction = 1;
      break;
    case CacheUnified:
      to.CacheType.ui32.Data = 1;
      to.CacheType.ui32.Instruction = 1;
      break;
    default:
      break;
  }
}

static HSAKMT_STATUS topology_parse_cpu_cache_props(node_props_t* tbl,
                                                    const std::vector<proc_cpu_info>& cpu_info) {
  DWORD length = 0;
  GetLogicalProcessorInformationEx(RelationCache, nullptr, &length);
  std::vector<char> buffer(length);
  auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());
  const auto* end = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data() + length);

  if (!GetLogicalProcessorInformationEx(RelationCache, info, &length)) {
    pr_err("GetLogicalProcessorInformationEx failed\n");
    return HSAKMT_STATUS_ERROR;
  }

  while (info < end) {
    if (info->Relationship == RelationCache) {
      const auto& cache = info->Cache;
      const GROUP_AFFINITY& ga = cache.GroupMask;
      const auto expected_logical_id = std::countr_zero(ga.Mask);

      for (const auto& cpu : cpu_info) {
        if (cpu.group == ga.Group && cpu.proc_num == expected_logical_id) {
          HsaCacheProperties this_cache{};
          map_cache(this_cache, cache);
          this_cache.ProcessorIdLow = cpu.apicid;
          tbl[cpu.numa_node].cache.push_back(this_cache);
          tbl[cpu.numa_node].node.NumCaches++;
          break;  // Should always find matched
        }
      }
    }
    info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
                                                reinterpret_cast<char*>(info) + info->Size);
  }

  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS topology_take_snapshot(void) {
  HsaSystemProperties sys_props;
  std::vector<node_props_t>& temp_props = dxg_topology->g_props;
  HSAKMT_STATUS ret = HSAKMT_STATUS_SUCCESS;
  std::vector<proc_cpu_info> cpu_info;
  std::vector<proc_numa_node_info> numa_node_info;
  bool p2p_links = false;
  uint32_t num_p2pLinks = 0;

  ret = topology_parse_cpu_info(cpu_info);
  if (ret != HSAKMT_STATUS_SUCCESS) {
    return ret;
  }
  ret = topology_parse_numa_node_info(numa_node_info, cpu_info);
  if (ret != HSAKMT_STATUS_SUCCESS) {
    return ret;
  }

  dxg_topology->numa_node_count_ = numa_node_info.size();
  ret = topology_sysfs_get_system_props(sys_props);
  if (ret != HSAKMT_STATUS_SUCCESS) {
    return ret;
  }

  if (sys_props.NumNodes > 0) {
    temp_props.resize(sys_props.NumNodes);

    // The first dxg_topology->numa_node_count_ temp_props denote
    // Cpu numa nodes in ascending order.
    ret = topology_parse_cpu_cache_props(temp_props.data(), cpu_info);
    if (ret != HSAKMT_STATUS_SUCCESS) {
      return ret;
    }

    for (uint32_t node_id = 0; node_id < sys_props.NumNodes; node_id++) {
      auto& node_prop = temp_props[node_id];

      if (node_id < dxg_topology->numa_node_count_) {
        // CPU numa node
        node_prop.node.CComputeIdLo = numa_node_info[node_id].ccompute_id_low;
        node_prop.node.NumCPUCores = numa_node_info[node_id].count;
        node_prop.node.NumMemoryBanks = 1;
        node_prop.node.KFDGpuID = 0;
        node_prop.node.MaxEngineClockMhzCCompute = dxg_topology->freq_max_;
        topology_get_cpu_model_name(node_prop.node, cpu_info);
      } else {
        // GPU node
        ret = topology_sysfs_get_node_props(node_id, node_prop.node, p2p_links,
                                          num_p2pLinks);
        if (ret != HSAKMT_STATUS_SUCCESS) {
          return ret;
        }
      }

      topology_setup_is_dgpu_param(&node_prop.node);

      if (node_prop.node.NumMemoryBanks) {
        node_prop.mem.resize(node_prop.node.NumMemoryBanks);

        for (uint32_t mem_id = 0; mem_id < node_prop.node.NumMemoryBanks; mem_id++) {
          ret = topology_sysfs_get_mem_props(node_id, mem_id,
                                             node_prop.mem[mem_id]);
          if (ret != HSAKMT_STATUS_SUCCESS) {
            return ret;
          }
        }
      }

      if (node_prop.node.KFDGpuID && node_prop.node.NumCaches) {
        node_prop.cache.resize(node_prop.node.NumCaches);
        for (uint32_t j = 0; j < 3; j++) {
          node_prop.cache[j].CacheType.ui32.Data = 1;
          node_prop.cache[j].CacheType.ui32.HSACU = 1;
          node_prop.cache[j].CacheLevel = j + 1;
        }

        wsl::thunk::WDDMDevice* device = nullptr;
        ret = topology_map_node_id(node_id, device);
        if (ret != HSAKMT_STATUS_SUCCESS) {
          return ret;
        }

        node_prop.cache[0].CacheSize = device->GetL1CacheSize() / 1024;
        node_prop.cache[1].CacheSize = device->GetL2CacheSize() / 1024;
        node_prop.cache[2].CacheSize = device->GetL3CacheSize() / 1024;
      }

      // To simplify, allocate maximum needed memory for io_links for each node.
      // This removes the need for realloc when indirect and QPI links are added later.
      node_prop.link.resize(sys_props.NumNodes - 1);
      const uint32_t num_ioLinks = node_prop.node.NumIOLinks - num_p2pLinks;
      uint32_t link_id = 0;

      if (num_ioLinks) {
        uint32_t sys_link_id = 0;

        // Parse all the sysfs specified io links. Skip the ones where the
        // remote node (node_to) is not accessible.
        while (sys_link_id < num_ioLinks && link_id < sys_props.NumNodes - 1) {
          ret = topology_sysfs_get_iolink_props(
              node_id, sys_link_id++, node_prop.link[link_id], false);
          if (ret == HSAKMT_STATUS_NOT_SUPPORTED) {
            ret = HSAKMT_STATUS_SUCCESS;
            continue;
          } else if (ret != HSAKMT_STATUS_SUCCESS) {
            return ret;
          }
          link_id++;
        }
        // sysfs specifies all the io links. Limit the number to valid ones.
        node_prop.node.NumIOLinks = link_id;
      }

      if (num_p2pLinks) {
        uint32_t sys_link_id = 0;

        // Parse all the sysfs specified p2p links.
        while (sys_link_id < num_p2pLinks && link_id < sys_props.NumNodes - 1) {
          ret = topology_sysfs_get_iolink_props(
              node_id, sys_link_id++, node_prop.link[link_id], true);
          if (ret == HSAKMT_STATUS_NOT_SUPPORTED) {
            ret = HSAKMT_STATUS_SUCCESS;
            continue;
          } else if (ret != HSAKMT_STATUS_SUCCESS) {
            return ret;
          }
          link_id++;
        }
        node_prop.node.NumIOLinks = link_id;
      }
    }
  }

  if (!p2p_links) {
    // All direct IO links are created in the kernel. Here we need to
    // connect GPU<->GPU or GPU<->CPU indirect IO links.
    topology_create_indirect_gpu_links(sys_props, temp_props);
  }

  if (!dxg_topology->g_system) {
    dxg_topology->g_system = static_cast<HsaSystemProperties*>(malloc(sizeof(HsaSystemProperties)));
    if (!dxg_topology->g_system) {
      ret = HSAKMT_STATUS_NO_MEMORY;
      return ret;
    }
  }

  *dxg_topology->g_system = sys_props;
  // Update default GPU node to account CPU nodes
  dxg_runtime->default_node = dxg_topology->numa_node_count_;
  return ret;
}
