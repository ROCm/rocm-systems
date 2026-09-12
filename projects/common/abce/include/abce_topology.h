/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef ABCE_TOPOLOGY_H_
#define ABCE_TOPOLOGY_H_

#include <cstdint>

/// Whether the KFD sysfs topology reader below is available. It walks
/// /sys/class/kfd/kfd/topology/nodes with opendir/readdir, which only exists on
/// Linux; TopologyData itself is platform-neutral, so a client on another OS
/// populates it from its own source (see SdmaEnginePolicy::SetTopology).
#ifndef ABCE_HAS_KFD_TOPOLOGY
#if defined(__linux__)
#define ABCE_HAS_KFD_TOPOLOGY 1
#else
#define ABCE_HAS_KFD_TOPOLOGY 0
#endif
#endif  // ABCE_HAS_KFD_TOPOLOGY

#if ABCE_HAS_KFD_TOPOLOGY
#include <dirent.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <utility>
#endif

namespace abce {

constexpr uint32_t kMaxTopologyDevices = 16;

/// Source/destination topology data consumed by engine policies.
class TopologyData {
 public:
  TopologyData() { Reset(); }

  void Reset() {
    for (uint32_t device_id = 0; device_id < kMaxTopologyDevices; ++device_id) {
      xgmi_physical_id_[device_id] = -1;
      hive_id_[device_id] = 0;
      for (uint32_t peer_id = 0; peer_id < kMaxTopologyDevices; ++peer_id)
        recommended_mask_[device_id][peer_id] = 0;
    }
  }

  void SetDevice(uint32_t device_id, int xgmi_physical_id, uint64_t hive_id) {
    if (device_id >= kMaxTopologyDevices) return;
    xgmi_physical_id_[device_id] = xgmi_physical_id;
    hive_id_[device_id] = hive_id;
  }

  void SetRecommendedMask(uint32_t source_device_id, uint32_t destination_device_id,
                          uint64_t mask) {
    if (source_device_id >= kMaxTopologyDevices || destination_device_id >= kMaxTopologyDevices)
      return;
    recommended_mask_[source_device_id][destination_device_id] = mask;
  }

  int XgmiPhysicalId(uint32_t device_id) const {
    return device_id < kMaxTopologyDevices ? xgmi_physical_id_[device_id] : -1;
  }

  uint64_t HiveId(uint32_t device_id) const {
    return device_id < kMaxTopologyDevices ? hive_id_[device_id] : 0;
  }

  uint64_t RecommendedMask(uint32_t source_device_id, uint32_t destination_device_id) const {
    return source_device_id < kMaxTopologyDevices && destination_device_id < kMaxTopologyDevices
               ? recommended_mask_[source_device_id][destination_device_id]
               : 0;
  }

 private:
  int xgmi_physical_id_[kMaxTopologyDevices];
  uint64_t hive_id_[kMaxTopologyDevices];
  uint64_t recommended_mask_[kMaxTopologyDevices][kMaxTopologyDevices];
};

struct TopologyLoadResult {
  bool found_gpu = false;
  uint8_t gfx_minor = 0;
  uint8_t total_sdma = 0;

  /// KFD's num_sdma_engines: how many of the SDMA engines are *not* xGMI. The
  /// driver numbers the non-xGMI engines first, so hardware ids at or above this
  /// are the xGMI engines (MI300X reports 2 and 14, and its CPU io_link
  /// recommends engine mask 0x3, confirming engines 0-1 are the non-xGMI pair).
  /// This is the xGMI band, derived rather than declared by the client.
  uint8_t num_non_xgmi_sdma = 0;
};

#if ABCE_HAS_KFD_TOPOLOGY

/// Header-only Linux/KFD topology adapter. Parsing into a temporary
/// TopologyData makes policy reconfiguration reset-safe.
class LinuxTopologyProvider {
 public:
  explicit LinuxTopologyProvider(std::string base_path = "/sys/class/kfd/kfd/topology/nodes")
      : base_path_(std::move(base_path)) {}

  TopologyLoadResult Populate(TopologyData& topology) {
    TopologyData loaded;
    TopologyLoadResult result{};
    DIR* directory = ::opendir(base_path_.c_str());
    if (!directory) return result;
    for (struct dirent* entry = ::readdir(directory); entry; entry = ::readdir(directory)) {
      const uint32_t node_id = ParseNodeId(entry->d_name);
      if (node_id == UINT32_MAX || node_id >= kMaxTopologyDevices) continue;
      LoadNode(node_id, base_path_ + "/" + entry->d_name, loaded, result);
    }
    ::closedir(directory);
    if (result.found_gpu) topology = loaded;
    return result;
  }

 private:
  static uint32_t ParseNodeId(const char* name) {
    if (!name || name[0] < '0' || name[0] > '9') return UINT32_MAX;
    uint32_t value = 0;
    for (const char* cursor = name; *cursor; ++cursor) {
      if (*cursor < '0' || *cursor > '9') return UINT32_MAX;
      value = value * 10u + static_cast<uint32_t>(*cursor - '0');
    }
    return value;
  }

  static int ReadXgmiPhysicalId(const std::string& pci_bdf) {
    std::ifstream file("/sys/bus/pci/devices/" + pci_bdf + "/xgmi_physical_id");
    int physical_id = -1;
    return file.is_open() && (file >> physical_id) ? physical_id : -1;
  }

  static void LoadLinks(uint32_t source_node, const std::string& links_dir,
                        TopologyData& topology) {
    DIR* directory = ::opendir(links_dir.c_str());
    if (!directory) return;
    for (struct dirent* entry = ::readdir(directory); entry; entry = ::readdir(directory)) {
      if (ParseNodeId(entry->d_name) == UINT32_MAX) continue;
      std::ifstream properties(links_dir + "/" + entry->d_name + "/properties");
      if (!properties.is_open()) continue;
      uint64_t destination_node = UINT32_MAX;
      uint64_t recommended_mask = 0;
      std::string key;
      uint64_t value = 0;
      while (properties >> key >> value) {
        if (key == "node_to")
          destination_node = value;
        else if (key == "recommended_sdma_engine_id_mask")
          recommended_mask = value;
      }
      if (destination_node < kMaxTopologyDevices && recommended_mask)
        topology.SetRecommendedMask(source_node, static_cast<uint32_t>(destination_node),
                                    recommended_mask);
    }
    ::closedir(directory);
  }

  static void LoadNode(uint32_t node_id, const std::string& node_directory, TopologyData& topology,
                       TopologyLoadResult& result) {
    std::ifstream properties(node_directory + "/properties");
    if (!properties.is_open()) return;
    uint64_t simd_count = 0;
    uint64_t hive_id = 0;
    uint64_t gfx_target_version = 0;
    uint64_t location_id = UINT64_MAX;
    uint64_t domain = 0;
    uint64_t num_sdma_engines = 0;
    uint64_t num_sdma_xgmi_engines = 0;
    std::string key;
    uint64_t value = 0;
    while (properties >> key >> value) {
      if (key == "simd_count")
        simd_count = value;
      else if (key == "hive_id")
        hive_id = value;
      else if (key == "gfx_target_version")
        gfx_target_version = value;
      else if (key == "num_sdma_engines")
        num_sdma_engines = value;
      else if (key == "num_sdma_xgmi_engines")
        num_sdma_xgmi_engines = value;
      else if (key == "location_id")
        location_id = value;
      else if (key == "domain")
        domain = value;
    }
    if (simd_count == 0) return;

    int physical_id = -1;
    if (location_id != UINT64_MAX) {
      char pci_bdf[32];
      std::snprintf(pci_bdf, sizeof(pci_bdf), "%04llx:%02llx:%02llx.%01llx",
                    static_cast<unsigned long long>(domain),
                    static_cast<unsigned long long>((location_id >> 8) & 0xff),
                    static_cast<unsigned long long>((location_id >> 3) & 0x1f),
                    static_cast<unsigned long long>(location_id & 0x7));
      physical_id = ReadXgmiPhysicalId(pci_bdf);
    }
    topology.SetDevice(node_id, physical_id, hive_id);
    LoadLinks(node_id, node_directory + "/io_links", topology);
    LoadLinks(node_id, node_directory + "/p2p_links", topology);

    result.found_gpu = true;
    if (result.total_sdma == 0 && gfx_target_version != 0) {
      result.gfx_minor = static_cast<uint8_t>((gfx_target_version / 100u) % 100u);
      result.total_sdma = static_cast<uint8_t>(num_sdma_engines + num_sdma_xgmi_engines);
      result.num_non_xgmi_sdma = static_cast<uint8_t>(num_sdma_engines);
    }
  }

  std::string base_path_;
};

#endif  // ABCE_HAS_KFD_TOPOLOGY

}  // namespace abce

#endif  // ABCE_TOPOLOGY_H_
