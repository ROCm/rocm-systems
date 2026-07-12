// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Internal: correlates per-plugin devices into canonical physical GPUs, grouped
// by socket, with partition lists, plus a routing table so a (canonical entity,
// plugin) pair maps back to that plugin's local device index + partition.
// Correlation is explicit and key-prioritized (BDF > KFD node id > UUID > render
// minor).

#ifndef GPUMETRICS_SRC_DEVICE_REGISTRY_H_
#define GPUMETRICS_SRC_DEVICE_REGISTRY_H_

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "gpumetrics/gpumetrics.h"
#include "gpumetrics/types.h"

namespace gpumetrics {

// A device as reported by one plugin (the raw input to correlation).
struct PluginDeviceRef {
  std::string plugin;             // plugin name
  gpum_device_identity identity;  // correlation keys
  std::string name;               // human name
};

// Where a canonical (gpu, partition) lives inside a specific plugin.
struct PluginHandle {
  std::string plugin;
  uint32_t plugin_local_index;
  int32_t partition;  // -1 whole device
};

// Result of correlation: canonical topology + routing.
class DeviceRegistry {
 public:
  // Build from every plugin's reported devices. `provider_priority` picks which
  // plugin's descriptive fields (name, etc.) win on disagreement; it does not
  // affect matching.
  static DeviceRegistry Build(const std::vector<PluginDeviceRef>& refs,
                              const std::vector<std::string>& provider_priority);

  const std::vector<Socket>& sockets() const { return sockets_; }
  const std::vector<Device>& devices() const { return devices_; }

  // Plugin handles that can serve a canonical entity. Partition entities get
  // matching-partition handles, falling back to whole-device handles from
  // plugins that don't model partitions.
  std::vector<PluginHandle> handlesFor(const gpum_entity_id& e) const;

  // Selector resolution helpers.
  const Device* gpuByIndex(uint32_t gpu) const;
  const Device* gpuByBdf(uint64_t bdf) const;
  const Device* gpuByUuid(const uint8_t uuid[16]) const;

 private:
  std::vector<Socket> sockets_;
  std::vector<Device> devices_;
  struct GpuRoute {
    struct PerPlugin {
      std::string plugin;
      uint32_t whole_local_index = 0;
      bool has_whole = false;
      std::map<int32_t, uint32_t> partition_local_index;  // partition -> local index
    };
    std::vector<PerPlugin> plugins;
  };
  std::vector<GpuRoute> routes_;  // indexed by canonical gpu ordinal
};

// Exposed for testing: do two identities refer to the same physical GPU? Uses
// key priority BDF > KFD node id > UUID > render minor; true only on a positive
// match of the highest-priority key both share.
bool SamePhysicalGpu(const gpum_device_identity& a, const gpum_device_identity& b);

}  // namespace gpumetrics

#endif  // GPUMETRICS_SRC_DEVICE_REGISTRY_H_
