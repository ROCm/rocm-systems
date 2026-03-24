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
  pr_debug("Processor Name: %s\n", cpu_model_name);
  return true;
}

HSAKMT_STATUS topology_parse_cpu_info(std::vector<proc_cpu_info>& cpu_info) {
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

HSAKMT_STATUS topology_parse_numa_node_info(
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

HSAKMT_STATUS topology_parse_cpu_cache_props(node_props_t* tbl,
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