/*
 * Copyright © 2014-2025 Advanced Micro Devices, Inc.
 * Copyright 2016-2018 Raptor Engineering, LLC. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including
 * the next paragraph) shall be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
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