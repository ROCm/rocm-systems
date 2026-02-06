/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

/* Number of memory banks added by thunk on top of topology
 * This only includes static heaps like LDS, scratch and SVM,
 * not for MMIO_REMAP heap. MMIO_REMAP memory bank is reported
 * dynamically based on whether mmio aperture was mapped
 * successfully on this node.
 */
#define NUM_OF_IGPU_HEAPS 3
#define NUM_OF_DGPU_HEAPS 3

typedef struct {
  HsaNodeProperties node;
  std::vector<HsaMemoryProperties> mem; /* node->NumBanks elements */
  std::vector<HsaCacheProperties> cache;
  std::vector<HsaIoLinkProperties> link;
} node_props_t;

struct _topology_props {
  HsaSystemProperties *g_system = nullptr;
  std::vector<node_props_t> g_props;
  std::vector<wsl::thunk::WDDMDevice *> wdevices_;
  uint32_t wdevice_num_ = 0;
  uint32_t num_sysfs_nodes = 0;
  uint32_t numa_node_count_ = 1;
  int processor_vendor = -1;
  double freq_max_ = 0.0;
};

/* Supported System Vendors */
enum SUPPORTED_PROCESSOR_VENDORS {
  GENUINE_INTEL = 0,
  AUTHENTIC_AMD,
  IBM_POWER
};

extern _topology_props* dxg_topology;
extern const char *supported_processor_vendor_name[];
HSAKMT_STATUS topology_take_snapshot(void);
int topology_search_processor_vendor(const std::string& processor_name);
void topology_setup_is_dgpu_param(HsaNodeProperties* props);
HSAKMT_STATUS topology_map_node_id(uint32_t node_id, wsl::thunk::WDDMDevice*& device);
HSAKMT_STATUS topology_sysfs_get_iolink_props(uint32_t node_id, uint32_t iolink_id,
                                              HsaIoLinkProperties& props, bool p2pLink);
void topology_create_indirect_gpu_links(const HsaSystemProperties& sys_props,
                                        std::vector<node_props_t>& node_props);