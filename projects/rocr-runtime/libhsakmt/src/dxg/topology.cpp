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

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <assert.h>
#if defined(__linux__)
#include <dirent.h>
#include <unistd.h>
#include <sys/sysinfo.h>
#endif
#include "impl/wddm/types.h"
#include "impl/wddm/device.h"
#include "util/utils.h"
#include "util/os.h"
#include <topology.hpp>

_topology_props* dxg_topology = new _topology_props();

/* Adding newline to make the search easier */
const char *supported_processor_vendor_name[] = {
  "GenuineIntel",
  "AuthenticAMD",
  "" // POWER requires a different search method
};

void topology_setup_is_dgpu_param(HsaNodeProperties *props) {
  /* if we found a dGPU node, then treat the whole system as dGPU */
  /* noted that some APUs are also treated as dGPU in runtime */
  if (!props->NumCPUCores && props->NumFComputeCores)
    dxg_runtime->hsakmt_is_dgpu = true;
}

int topology_search_processor_vendor(const std::string& processor_name) {
  for (unsigned int i = 0; i < ARRAY_LEN(supported_processor_vendor_name); i++) {
    if (processor_name == supported_processor_vendor_name[i])
      return i;
    if (processor_name == "POWER9, altivec supported")
      return IBM_POWER;
  }
  return -1;
}

/* For a give Node @node_id the function gets @iolink_id information i.e. parses
 * sysfs the following sysfs entry
 * ./nodes/@node_id/io_links/@iolink_id/properties. @node_id has to be valid
 * accessible node.
 *
 * If node_to specified by the @iolink_id is not accessible the function returns
 * HSAKMT_STATUS_NOT_SUPPORTED. If node_to is accessible, then node_to is mapped
 * from sysfs_node to user_node and returns HSAKMT_STATUS_SUCCESS.
 */
HSAKMT_STATUS topology_sysfs_get_iolink_props(uint32_t node_id,
                                              uint32_t iolink_id,
                                              HsaIoLinkProperties& props,
                                              bool p2pLink) {
  wsl::thunk::WDDMDevice* device = get_wddmdev(node_id);
  assert(device);

  std::memset(&props, 0, sizeof(props));
  props.IoLinkType = HSA_IOLINKTYPE_PCIEXPRESS;
  props.VersionMajor = props.VersionMinor = 0;
  props.NodeFrom = node_id;
  props.NodeTo = 0;
  props.Weight = 20;
  props.Flags.ui32.Override = 1;
  props.Flags.ui32.NonCoherent = 1;
  props.Flags.ui32.NoAtomics32bit = !(device->SupportPlatformAtomic());
  props.Flags.ui32.NoAtomics64bit = !(device->SupportPlatformAtomic());
  props.RecSdmaEngIdMask = 0;

  return HSAKMT_STATUS_SUCCESS;
}

/* topology_get_free_io_link_slot_for_node - For the given node_id, find the
 * next available free slot to add an io_link
 */
static HsaIoLinkProperties *
topology_get_free_io_link_slot_for_node(uint32_t node_id,
                                        const HsaSystemProperties& sys_props,
                                        std::vector<node_props_t>& node_props) {
  std::vector<HsaIoLinkProperties>& props = node_props[node_id].link;

  if (node_id >= sys_props.NumNodes) {
    pr_err("Invalid node [%d]\n", node_id);
    return NULL;
  }

  if (!props.size()) {
    pr_err("No io_link reported for Node [%d]\n", node_id);
    return NULL;
  }

  if (node_props[node_id].node.NumIOLinks >= sys_props.NumNodes - 1) {
    pr_err("No more space for io_link for Node [%d]\n", node_id);
    return NULL;
  }

  return &props[node_props[node_id].node.NumIOLinks];
}

/* topology_add_io_link_for_node - If a free slot is available,
 * add io_link for the given Node.
 * TODO: Add other members of HsaIoLinkProperties
 */
static HSAKMT_STATUS topology_add_io_link_for_node(
    uint32_t node_from, const HsaSystemProperties& sys_props,
    std::vector<node_props_t>& node_props, HSA_IOLINKTYPE IoLinkType, uint32_t node_to,
    uint32_t Weight) {
  HsaIoLinkProperties *props;

  props =
      topology_get_free_io_link_slot_for_node(node_from, sys_props, node_props);
  if (!props)
    return HSAKMT_STATUS_NO_MEMORY;

  props->IoLinkType = IoLinkType;
  props->NodeFrom = node_from;
  props->NodeTo = node_to;
  props->Weight = Weight;
  node_props[node_from].node.NumIOLinks++;

  return HSAKMT_STATUS_SUCCESS;
}

/* Find the CPU that this GPU (gpu_node) directly connects to */
static int32_t gpu_get_direct_link_cpu(uint32_t gpu_node,
                                       const std::vector<node_props_t>& node_props) {
  const std::vector<HsaIoLinkProperties>& props = node_props[gpu_node].link;
  uint32_t i;

  if (!node_props[gpu_node].node.KFDGpuID || props.empty() ||
      node_props[gpu_node].node.NumIOLinks == 0)
    return -1;

  for (i = 0; i < node_props[gpu_node].node.NumIOLinks; i++)
    if (props[i].IoLinkType == HSA_IOLINKTYPE_PCIEXPRESS &&
        props[i].Weight <= 20) /* >20 is GPU->CPU->GPU */
      return props[i].NodeTo;

  return -1;
}

/* Get node1->node2 IO link information. This should be a direct link that has
 * been created in the kernel.
 */
static HSAKMT_STATUS get_direct_iolink_info(uint32_t node1, uint32_t node2,
                                            const std::vector<node_props_t>& node_props,
                                            HSAuint32 *weight,
                                            HSA_IOLINKTYPE *type) {
  const std::vector<HsaIoLinkProperties>& props = node_props[node1].link;
  uint32_t i;

  if (!props.size())
    return HSAKMT_STATUS_INVALID_NODE_UNIT;

  for (i = 0; i < node_props[node1].node.NumIOLinks; i++)
    if (props[i].NodeTo == node2) {
      if (weight)
        *weight = props[i].Weight;
      if (type)
        *type = props[i].IoLinkType;
      return HSAKMT_STATUS_SUCCESS;
    }

  return HSAKMT_STATUS_INVALID_PARAMETER;
}

static HSAKMT_STATUS get_indirect_iolink_info(uint32_t node1, uint32_t node2,
                                              const std::vector<node_props_t>& node_props,
                                              HSAuint32 *weight,
                                              HSA_IOLINKTYPE *type) {
  int32_t dir_cpu1 = -1, dir_cpu2 = -1;
  HSAKMT_STATUS ret;
  uint32_t i;

  *weight = 0;
  *type = HSA_IOLINKTYPE_UNDEFINED;

  if (node1 == node2)
    return HSAKMT_STATUS_INVALID_PARAMETER;

  /* CPU->CPU is not an indirect link */
  if (!node_props[node1].node.KFDGpuID && !node_props[node2].node.KFDGpuID)
    return HSAKMT_STATUS_INVALID_NODE_UNIT;

  if (node_props[node1].node.HiveID && node_props[node2].node.HiveID &&
      node_props[node1].node.HiveID == node_props[node2].node.HiveID)
    return HSAKMT_STATUS_INVALID_PARAMETER;

  if (node_props[node1].node.KFDGpuID)
    dir_cpu1 = gpu_get_direct_link_cpu(node1, node_props);
  if (node_props[node2].node.KFDGpuID)
    dir_cpu2 = gpu_get_direct_link_cpu(node2, node_props);

  if (dir_cpu1 < 0 && dir_cpu2 < 0)
    return HSAKMT_STATUS_ERROR;

  /* if the node2(dst) is GPU , it need to be large bar for host access*/
  if (node_props[node2].node.KFDGpuID) {
    for (i = 0; i < node_props[node2].node.NumMemoryBanks; ++i)
      if (node_props[node2].mem[i].HeapType == HSA_HEAPTYPE_FRAME_BUFFER_PUBLIC)
        break;
    if (i >= node_props[node2].node.NumMemoryBanks)
      return HSAKMT_STATUS_ERROR;
  }
  /* Possible topology:
   *   GPU --(weight1) -- CPU -- (weight2) -- GPU
   *   GPU --(weight1) -- CPU -- (weight2) -- CPU -- (weight3) -- GPU
   *   GPU --(weight1) -- CPU -- (weight2) -- CPU
   *   CPU -- (weight2) -- CPU -- (weight3) -- GPU
   */
  HSAuint32 weight1 = 0, weight2 = 0, weight3 = 0;
  if (dir_cpu1 >= 0) { /* GPU->CPU ... */
    if (dir_cpu2 >= 0) {
      if (dir_cpu1 == dir_cpu2) /* GPU->CPU->GPU*/ {
        ret =
            get_direct_iolink_info(node1, dir_cpu1, node_props, &weight1, NULL);
        if (ret != HSAKMT_STATUS_SUCCESS)
          return ret;
        ret =
            get_direct_iolink_info(dir_cpu1, node2, node_props, &weight2, type);
      } else /* GPU->CPU->CPU->GPU*/ {
        ret =
            get_direct_iolink_info(node1, dir_cpu1, node_props, &weight1, NULL);
        if (ret != HSAKMT_STATUS_SUCCESS)
          return ret;
        ret = get_direct_iolink_info(dir_cpu1, dir_cpu2, node_props, &weight2,
                                     type);
        if (ret != HSAKMT_STATUS_SUCCESS)
          return ret;
        /* On QPI interconnection, GPUs can't access
         * each other if they are attached to different
         * CPU sockets. CPU<->CPU weight larger than 20
         * means the two CPUs are in different sockets.
         */
        if (*type == HSA_IOLINK_TYPE_QPI_1_1 && weight2 > 20)
          return HSAKMT_STATUS_NOT_SUPPORTED;
        ret =
            get_direct_iolink_info(dir_cpu2, node2, node_props, &weight3, NULL);
      }
    } else /* GPU->CPU->CPU */ {
      ret = get_direct_iolink_info(node1, dir_cpu1, node_props, &weight1, NULL);
      if (ret != HSAKMT_STATUS_SUCCESS)
        return ret;
      ret = get_direct_iolink_info(dir_cpu1, node2, node_props, &weight2, type);
    }
  } else { /* CPU->CPU->GPU */
    ret = get_direct_iolink_info(node1, dir_cpu2, node_props, &weight2, type);
    if (ret != HSAKMT_STATUS_SUCCESS)
      return ret;
    ret = get_direct_iolink_info(dir_cpu2, node2, node_props, &weight3, NULL);
  }

  if (ret != HSAKMT_STATUS_SUCCESS)
    return ret;

  *weight = weight1 + weight2 + weight3;
  return HSAKMT_STATUS_SUCCESS;
}

void topology_create_indirect_gpu_links(const HsaSystemProperties& sys_props,
                                        std::vector<node_props_t>& node_props) {
  uint32_t i, j;
  HSAuint32 weight;
  HSA_IOLINKTYPE type;

  for (i = 0; i < sys_props.NumNodes - 1; i++) {
    for (j = i + 1; j < sys_props.NumNodes; j++) {
      get_indirect_iolink_info(i, j, node_props, &weight, &type);
      if (!weight)
        goto try_alt_dir;
      if (topology_add_io_link_for_node(i, sys_props, node_props, type, j,
                                        weight) != HSAKMT_STATUS_SUCCESS)
        pr_err("Fail to add IO link %d->%d\n", i, j);
    try_alt_dir:
      get_indirect_iolink_info(j, i, node_props, &weight, &type);
      if (!weight)
        continue;
      if (topology_add_io_link_for_node(j, sys_props, node_props, type, i,
                                        weight) != HSAKMT_STATUS_SUCCESS)
        pr_err("Fail to add IO link %d->%d\n", j, i);
    }
  }
}

/* Drop the Snashot of the HSA topology information. Assume lock is held. */
void topology_drop_snapshot(void) {
  if (!!dxg_topology->g_system != !!dxg_topology->g_props.size())
    pr_warn("Probably inconsistency?\n");

  // Free heap GPU VA BEFORE deleting adapters
  // The GPU VA free requires adapters to be alive
  dxg_runtime->HeapFini();

  dxg_topology->g_props.clear();

  free(dxg_topology->g_system);
  dxg_topology->g_system = NULL;

  trim_suballocator();
  for (auto device : dxg_topology->wdevices_)
    delete device;
  dxg_topology->wdevices_.clear();
}

HSAKMT_STATUS validate_nodeid(uint32_t nodeid, uint32_t *gpu_id) {
  if (dxg_topology->g_props.empty() || !dxg_topology->g_system || dxg_topology->g_system->NumNodes <= nodeid)
    return HSAKMT_STATUS_INVALID_NODE_UNIT;
  if (gpu_id)
    *gpu_id = dxg_topology->g_props[nodeid].node.KFDGpuID;

  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS gpuid_to_nodeid(uint32_t gpu_id, uint32_t *node_id) {
  uint64_t node_idx;

  for (node_idx = 0; node_idx < dxg_topology->g_system->NumNodes; node_idx++) {
    if (dxg_topology->g_props[node_idx].node.KFDGpuID == gpu_id) {
      *node_id = node_idx;
      return HSAKMT_STATUS_SUCCESS;
    }
  }

  return HSAKMT_STATUS_INVALID_NODE_UNIT;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtAcquireSystemProperties(HsaSystemProperties *SystemProperties) {
  HSAKMT_STATUS err = HSAKMT_STATUS_SUCCESS;

  CHECK_DXG_OPEN();

  if (!SystemProperties)
    return HSAKMT_STATUS_INVALID_PARAMETER;

  std::lock_guard<std::recursive_mutex> lck(dxg_runtime->hsakmt_mutex);

  /* We already have a valid snapshot. Avoid double initialization that
   * would leak memory.
   */
  if (dxg_topology->g_system) {
    *SystemProperties = *dxg_topology->g_system;
    goto out;
  }

  err = topology_take_snapshot();
  if (err != HSAKMT_STATUS_SUCCESS)
    goto out;

  assert(dxg_topology->g_system);

  // err = fmm_init_process_apertures(dxg_topology->g_system->NumNodes);
  if (err != HSAKMT_STATUS_SUCCESS)
    goto init_process_apertures_failed;

  // err = init_process_doorbells(dxg_topology->g_system->NumNodes);
  if (err != HSAKMT_STATUS_SUCCESS)
    goto init_doorbells_failed;

  *SystemProperties = *dxg_topology->g_system;

  goto out;

init_doorbells_failed:
  // fmm_destroy_process_apertures();
init_process_apertures_failed:
  topology_drop_snapshot();

out:
  return err;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtReleaseSystemProperties(void) {
  std::lock_guard<std::recursive_mutex> lck(dxg_runtime->hsakmt_mutex);

  topology_drop_snapshot();

  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS topology_get_node_props(HSAuint32 NodeId,
                                      HsaNodeProperties *NodeProperties) {
  if (!dxg_topology->g_system || dxg_topology->g_props.empty() || NodeId >= dxg_topology->g_system->NumNodes)
    return HSAKMT_STATUS_ERROR;

  *NodeProperties = dxg_topology->g_props[NodeId].node;
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtGetNodeProperties(HSAuint32 NodeId, HsaNodeProperties *NodeProperties) {
  HSAKMT_STATUS err;
  uint32_t gpu_id;

  if (!NodeProperties)
    return HSAKMT_STATUS_INVALID_PARAMETER;

  CHECK_DXG_OPEN();
  std::lock_guard<std::recursive_mutex> lck(dxg_runtime->hsakmt_mutex);

  err = validate_nodeid(NodeId, &gpu_id);
  if (err != HSAKMT_STATUS_SUCCESS)
    goto out;

  err = topology_get_node_props(NodeId, NodeProperties);
  if (err != HSAKMT_STATUS_SUCCESS)
    goto out;
  /* For CPU only node don't add any additional GPU memory banks. */
  if (gpu_id) {
    uint64_t base, limit;
    if (!(NodeProperties->Integrated))
      NodeProperties->NumMemoryBanks += NUM_OF_DGPU_HEAPS;
    else
      NodeProperties->NumMemoryBanks += NUM_OF_IGPU_HEAPS;
    // TODO: for apu
    /*if (fmm_get_aperture_base_and_limit(FMM_MMIO, gpu_id, &base,
                    &limit) == HSAKMT_STATUS_SUCCESS)
            NodeProperties->NumMemoryBanks += 1;*/
  }

out:

  return err;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtGetNodeMemoryProperties(HSAuint32 NodeId, HSAuint32 NumBanks,
                              HsaMemoryProperties *MemoryProperties) {
  HSAKMT_STATUS err = HSAKMT_STATUS_SUCCESS;
  uint32_t i;

  if (!MemoryProperties)
    return HSAKMT_STATUS_INVALID_PARAMETER;

  CHECK_DXG_OPEN();
  std::lock_guard<std::recursive_mutex> lck(dxg_runtime->hsakmt_mutex);

  memset(MemoryProperties, 0, NumBanks * sizeof(HsaMemoryProperties));
  for (i = 0; i < rocr::Min(dxg_topology->g_props[NodeId].node.NumMemoryBanks, NumBanks); i++) {
    assert(dxg_topology->g_props[NodeId].mem.size());
    MemoryProperties[i] = dxg_topology->g_props[NodeId].mem[i];
  }

  /* The following memory banks does not apply to CPU only node */
  wsl::thunk::WDDMDevice *device_ = get_wddmdev(NodeId);
  if (device_ == nullptr)
    goto out;

  /*Add LDS*/
  if (i < NumBanks) {
    MemoryProperties[i].HeapType = HSA_HEAPTYPE_GPU_LDS;
    MemoryProperties[i].VirtualBaseAddress = device_->SharedApertureBase();
    MemoryProperties[i].SizeInBytes = dxg_topology->g_props[NodeId].node.LDSSizeInKB * 1024;
    i++;
  }

  /* Add SCRATCH */
  if (i < NumBanks) {
    MemoryProperties[i].HeapType = HSA_HEAPTYPE_GPU_SCRATCH;
    MemoryProperties[i].VirtualBaseAddress = device_->PrivateApertureBase();
    MemoryProperties[i].SizeInBytes = device_->PrivateApertureSize();
    i++;
  }

out:
  return err;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtGetNodeCacheProperties(
    HSAuint32 NodeId, HSAuint32 ProcessorId, HSAuint32 NumCaches,
    HsaCacheProperties *CacheProperties) {
  HSAKMT_STATUS err;
  uint32_t i;

  if (!CacheProperties)
    return HSAKMT_STATUS_INVALID_PARAMETER;

  CHECK_DXG_OPEN();
  std::lock_guard<std::recursive_mutex> lck(dxg_runtime->hsakmt_mutex);

  /* KFD ADD page 18, snapshot protocol violation */
  if (!dxg_topology->g_system || NodeId >= dxg_topology->g_system->NumNodes) {
    err = HSAKMT_STATUS_INVALID_NODE_UNIT;
    goto out;
  }

  if (NumCaches > dxg_topology->g_props[NodeId].node.NumCaches) {
    err = HSAKMT_STATUS_INVALID_PARAMETER;
    goto out;
  }

  for (i = 0; i < rocr::Min(dxg_topology->g_props[NodeId].node.NumCaches, NumCaches); i++) {
    assert(dxg_topology->g_props[NodeId].cache.size());
    CacheProperties[i] = dxg_topology->g_props[NodeId].cache[i];
  }

  err = HSAKMT_STATUS_SUCCESS;

out:

  return err;
}

HSAKMT_STATUS topology_get_iolink_props(HSAuint32 NodeId, HSAuint32 NumIoLinks,
                                        HsaIoLinkProperties *IoLinkProperties) {
  if (!dxg_topology->g_system || dxg_topology->g_props.empty() || NodeId >= dxg_topology->g_system->NumNodes)
    return HSAKMT_STATUS_ERROR;

  memcpy(IoLinkProperties, dxg_topology->g_props[NodeId].link.data(),
         NumIoLinks * sizeof(*IoLinkProperties));

  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtGetNodeIoLinkProperties(HSAuint32 NodeId, HSAuint32 NumIoLinks,
                              HsaIoLinkProperties *IoLinkProperties) {
  HSAKMT_STATUS err;

  if (!IoLinkProperties)
    return HSAKMT_STATUS_INVALID_PARAMETER;

  CHECK_DXG_OPEN();

  std::lock_guard<std::recursive_mutex> lck(dxg_runtime->hsakmt_mutex);

  /* KFD ADD page 18, snapshot protocol violation */
  if (!dxg_topology->g_system || NodeId >= dxg_topology->g_system->NumNodes) {
    err = HSAKMT_STATUS_INVALID_NODE_UNIT;
    goto out;
  }

  if (NumIoLinks > dxg_topology->g_props[NodeId].node.NumIOLinks) {
    err = HSAKMT_STATUS_INVALID_PARAMETER;
    goto out;
  }

  assert(dxg_topology->g_props[NodeId].link.size());
  err = topology_get_iolink_props(NodeId, NumIoLinks, IoLinkProperties);

out:
  return err;
}

uint16_t get_device_id_by_node_id(HSAuint32 node_id) {
  if (dxg_topology->g_props.empty() || !dxg_topology->g_system || dxg_topology->g_system->NumNodes <= node_id)
    return 0;

  return dxg_topology->g_props[node_id].node.DeviceId;
}

bool prefer_ats(HSAuint32 node_id) {
  return dxg_topology->g_props[node_id].node.Capability.ui32.HSAMMUPresent &&
         dxg_topology->g_props[node_id].node.NumCPUCores &&
         dxg_topology->g_props[node_id].node.NumFComputeCores;
}

uint16_t get_device_id_by_gpu_id(HSAuint32 gpu_id) {
  unsigned int i;

  if (dxg_topology->g_props.empty() || !dxg_topology->g_system)
    return 0;

  for (i = 0; i < dxg_topology->g_system->NumNodes; i++) {
    if (dxg_topology->g_props[i].node.KFDGpuID == gpu_id)
      return dxg_topology->g_props[i].node.DeviceId;
  }

  return 0;
}

uint32_t get_direct_link_cpu(uint32_t gpu_node) {
  HSAuint64 size = 0;
  int32_t cpu_id;
  HSAuint32 i;

  cpu_id = gpu_get_direct_link_cpu(gpu_node, dxg_topology->g_props);
  if (cpu_id == -1)
    return INVALID_NODEID;

  assert(dxg_topology->g_props[cpu_id].mem.size());

  for (i = 0; i < dxg_topology->g_props[cpu_id].node.NumMemoryBanks; i++)
    size += dxg_topology->g_props[cpu_id].mem[i].SizeInBytes;

  return size ? (uint32_t)cpu_id : INVALID_NODEID;
}

HSAKMT_STATUS validate_nodeid_array(uint32_t **gpu_id_array,
                                    uint32_t NumberOfNodes,
                                    uint32_t *NodeArray) {
  HSAKMT_STATUS ret;
  unsigned int i;

  if (NumberOfNodes == 0 || !NodeArray || !gpu_id_array)
    return HSAKMT_STATUS_INVALID_PARAMETER;

  /* Translate Node IDs to gpu_ids */
  *gpu_id_array = (uint32_t *)malloc(NumberOfNodes * sizeof(uint32_t));
  if (!(*gpu_id_array))
    return HSAKMT_STATUS_NO_MEMORY;
  for (i = 0; i < NumberOfNodes; i++) {
    ret = validate_nodeid(NodeArray[i], *gpu_id_array + i);
    if (ret != HSAKMT_STATUS_SUCCESS) {
      free(*gpu_id_array);
      break;
    }
  }

  return ret;
}

uint32_t get_num_sysfs_nodes(void) { return dxg_topology->num_sysfs_nodes; }

wsl::thunk::WDDMDevice *get_wddmdev(uint32_t node_id) {
  if ((!dxg_topology->wdevices_.size()) || (node_id < dxg_topology->numa_node_count_) ||
      (node_id >= dxg_topology->num_sysfs_nodes))
    return nullptr;

  return dxg_topology->wdevices_[node_id - dxg_topology->numa_node_count_];
}

int CpuNodes() { return dxg_topology->numa_node_count_; }

wsl::thunk::WDDMDevice* WddmDevice(uint32_t dev_id) {
  assert(dxg_topology->wdevices_.size() && "No GPU device!");
  return dxg_topology->wdevices_[dev_id];
}

uint32_t get_num_wddmdev() {
  return dxg_topology->wdevices_.size();
}

HSAKMT_STATUS topology_sysfs_get_system_props(HsaSystemProperties& props) {
  std::memset(&props, 0, sizeof(props));

  dxg_runtime->HeapFini();
  for (auto device : dxg_topology->wdevices_) {
    delete device;
  }
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
  // Update default GPU node to account CPU nodes
  dxg_runtime->default_node = dxg_topology->numa_node_count_;

  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS topology_sysfs_get_node_props(uint32_t node_id, HsaNodeProperties& props,
                                            bool& p2p_links, uint32_t& num_p2pLinks) {
  memset(&props, 0, sizeof(props));
  p2p_links = false;
  num_p2pLinks = 0;
  props.MaxEngineClockMhzCCompute = dxg_topology->freq_max_;

  if (node_id < dxg_topology->numa_node_count_) {
    return HSAKMT_STATUS_SUCCESS;
  }

  // GPU node
  wsl::thunk::WDDMDevice* device = get_wddmdev(node_id);
  assert(device);

  props.NumCPUCores = 0;
  props.NumFComputeCores = device->SimdPerCu() * device->ComputeUnitCount();
  props.NumMemoryBanks = 1;
  props.NumCaches = 3;
  props.NumIOLinks = 1;
  props.CComputeIdLo = 0;
  props.FComputeIdLo = 0;
  props.Capability.ui32.ASICRevision = device->AsicRevision();
  props.Capability.ui32.WatchPointsTotalBits = std::log2(device->WatchPointsNum());
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
  props.NumSdmaQueuesPerEngine = 6;  // TODO
  props.NumCpQueues = device->GetNumCpQueues();
  props.NumGws = 0;
  /*
   * In Native Linux, if the asic is APU, this value will be set to 1,
   * if the asic is dGPU, this value will be set to 0. clr use this info
   * to set hostUnifiedMemory_, but for now wsl does not support this feature.
   * Therefore, force vaule to 0 temporarily.
   */
  props.Integrated = 0;
  props.Domain = device->Domain();
  props.UniqueID = device->Uuid();
  props.NumXcc = device->NumXcc();
  props.KFDGpuID = device->DeviceId();  // TODO
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

HSAKMT_STATUS topology_sysfs_get_mem_props(uint32_t node_id, uint32_t mem_id,
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

  wsl::thunk::WDDMDevice* device = get_wddmdev(node_id);
  assert(device);

  props.HeapType = HSA_HEAPTYPE_FRAME_BUFFER_PRIVATE;
  props.SizeInBytes = device->LocalHeapSize();
  props.Width = device->MemoryBusWidth();
  props.MemoryClockMax = device->MaxMemoryClockMhz();

  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS topology_get_cpu_model_name(HsaNodeProperties& props,
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
        ret = topology_sysfs_get_node_props(node_id, node_prop.node, p2p_links, num_p2pLinks);
        if (ret != HSAKMT_STATUS_SUCCESS) {
          return ret;
        }
      }

      topology_setup_is_dgpu_param(&node_prop.node);

      if (node_prop.node.NumMemoryBanks) {
        node_prop.mem.resize(node_prop.node.NumMemoryBanks);

        for (uint32_t mem_id = 0; mem_id < node_prop.node.NumMemoryBanks; mem_id++) {
          ret = topology_sysfs_get_mem_props(node_id, mem_id, node_prop.mem[mem_id]);
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

        wsl::thunk::WDDMDevice* device = get_wddmdev(node_id);
        assert(device);

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
          ret = topology_sysfs_get_iolink_props(node_id, sys_link_id++, node_prop.link[link_id],
                                                false);
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
          ret = topology_sysfs_get_iolink_props(node_id, sys_link_id++, node_prop.link[link_id],
                                                true);
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
  return ret;
}