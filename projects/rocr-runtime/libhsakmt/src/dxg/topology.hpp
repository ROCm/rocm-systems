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
  uint32_t numa_node_count_ = 0;
  int processor_vendor = -1;
  double freq_max_ = 0.0;
};

/* Supported System Vendors */
enum SUPPORTED_PROCESSOR_VENDORS {
  GENUINE_INTEL = 0,
  AUTHENTIC_AMD,
  IBM_POWER
};

struct proc_cpu_info {
  uint32_t group;                        //!< id of group, ignored for linux
  uint32_t proc_num;                     //!< id of the logical processor (in the group for Windows)
  uint32_t apicid;                       //!< extended apicid to support 256+ logical processors
  uint32_t numa_node;                    //!< numa node of this logical processor, ignored for Wsl
  char model_name[HSA_PUBLIC_NAME_SIZE];  //!< model name
};

struct proc_numa_node_info {
  uint32_t numa_node;        //!< numa node number
  uint32_t ccompute_id_low;  //!< extended apicid of the logical processor of the lowest ID in
                             //!< the primary group of this node.
  uint32_t count;            //!< count of logical processors on this node
  bool operator<(const proc_numa_node_info& other) const { return numa_node < other.numa_node; }
};

extern _topology_props* dxg_topology;
extern const char *supported_processor_vendor_name[];
HSAKMT_STATUS topology_take_snapshot(void);
int topology_search_processor_vendor(const std::string& processor_name);
void topology_setup_is_dgpu_param(HsaNodeProperties* props);
HSAKMT_STATUS topology_sysfs_get_iolink_props(uint32_t node_id, uint32_t iolink_id,
                                              HsaIoLinkProperties& props, bool p2pLink);
void topology_create_indirect_gpu_links(const HsaSystemProperties& sys_props,
                                        std::vector<node_props_t>& node_props);
HSAKMT_STATUS topology_parse_cpu_info(std::vector<proc_cpu_info>& cpu_info);
HSAKMT_STATUS topology_parse_numa_node_info(std::vector<proc_numa_node_info>& numa_node_info,
                                            const std::vector<proc_cpu_info>& cpu_info);
HSAKMT_STATUS topology_parse_cpu_cache_props(node_props_t* tbl,
                                             const std::vector<proc_cpu_info>& cpu_info);