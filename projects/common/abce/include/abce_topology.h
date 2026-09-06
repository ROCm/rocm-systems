/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef ABCE_TOPOLOGY_H_
#define ABCE_TOPOLOGY_H_

#include <cstdint>

/// Whether the KFD sysfs reader below is available. It walks
/// /sys/class/kfd/kfd/topology/nodes with opendir/readdir, which only exists on
/// Linux. TopologyData itself is platform-neutral, so a client elsewhere fills
/// one from whatever its OS exposes and hands it to the orchestrator.
#ifndef ABCE_HAS_KFD_TOPOLOGY
#if defined(__linux__)
#define ABCE_HAS_KFD_TOPOLOGY 1
#else
#define ABCE_HAS_KFD_TOPOLOGY 0
#endif
#endif  // ABCE_HAS_KFD_TOPOLOGY

#if ABCE_HAS_KFD_TOPOLOGY
#include <dirent.h>

#include <fstream>
#include <string>
#include <utility>
#endif

namespace abce {

constexpr uint32_t kMaxTopologyDevices = 16;

/// @brief The driver's per-link SDMA engine recommendations — the whole of what
/// ABCE wants from system topology.
///
/// KFD's recommended_sdma_engine_id_mask names the engines with affinity to one
/// link, indexed by the node ids a client already uses as ABCE device ids, so it
/// drops straight into the engine table. ABCE derives nothing further: no engine
/// class, no hive membership, no physical slot. All three used to live here, and
/// all three existed only to reconstruct a per-link engine choice the driver
/// reports directly.
class TopologyData {
 public:
  TopologyData() { Reset(); }

  void Reset() {
    for (auto& row : recommended_mask_)
      for (auto& mask : row) mask = 0;
  }

  void SetRecommendedMask(uint32_t source_device_id, uint32_t destination_device_id,
                          uint64_t mask) {
    if (source_device_id >= kMaxTopologyDevices || destination_device_id >= kMaxTopologyDevices)
      return;
    recommended_mask_[source_device_id][destination_device_id] = mask;
  }

  /// Hardware SDMA engine ids recommended for this ordered pair, or 0 when the
  /// driver said nothing — which the engine table reads as "no preference".
  uint64_t RecommendedMask(uint32_t source_device_id, uint32_t destination_device_id) const {
    return source_device_id < kMaxTopologyDevices && destination_device_id < kMaxTopologyDevices
               ? recommended_mask_[source_device_id][destination_device_id]
               : 0;
  }

 private:
  uint64_t recommended_mask_[kMaxTopologyDevices][kMaxTopologyDevices];
};

struct TopologyLoadResult {
  bool found_gpu = false;

  /// Total SDMA engines the device reports (non-xGMI plus xGMI). Keys the
  /// measured host-copy profile, since the same IP ships with different engine
  /// counts and genuinely different measured orders.
  uint8_t total_sdma = 0;
};

#if ABCE_HAS_KFD_TOPOLOGY

/// Header-only Linux/KFD adapter. Parsing into a temporary TopologyData makes
/// reloading reset-safe: a partial read never replaces good data.
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
    uint64_t num_sdma_engines = 0;
    uint64_t num_sdma_xgmi_engines = 0;
    std::string key;
    uint64_t value = 0;
    while (properties >> key >> value) {
      if (key == "simd_count")
        simd_count = value;
      else if (key == "num_sdma_engines")
        num_sdma_engines = value;
      else if (key == "num_sdma_xgmi_engines")
        num_sdma_xgmi_engines = value;
    }
    // CPU nodes report no SIMDs and own no SDMA engines; their links are still
    // picked up from the GPU side, which is the direction a copy is ranked in.
    if (simd_count == 0) return;

    LoadLinks(node_id, node_directory + "/io_links", topology);
    LoadLinks(node_id, node_directory + "/p2p_links", topology);

    result.found_gpu = true;
    if (result.total_sdma == 0)
      result.total_sdma = static_cast<uint8_t>(num_sdma_engines + num_sdma_xgmi_engines);
  }

  std::string base_path_;
};

#endif  // ABCE_HAS_KFD_TOPOLOGY

}  // namespace abce

#endif  // ABCE_TOPOLOGY_H_
