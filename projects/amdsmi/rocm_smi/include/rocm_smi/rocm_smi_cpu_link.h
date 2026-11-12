// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_ROCM_SMI_ROCM_SMI_CPU_LINK_H_
#define INCLUDE_ROCM_SMI_ROCM_SMI_CPU_LINK_H_

#include <cstdint>
#include <vector>

#include "rocm_smi/rocm_smi_io_link.h"

namespace amd::smi {

// One GPU<->CPU link found via a GPU's io_link KFD neighbors (any link
// type -- XGMI, PCIe, etc; the caller decides which types it cares about).
// numa_node_id is read from the io_link's node_to, which for a GPU's direct
// CPU link KFD sets straight from the GPU's pdev->dev.numa_node (see
// kfd_fill_gpu_direct_io_link_to_cpu in amdkfd/kfd_crat.c) -- the real Linux
// NUMA node id, not an index KFD or amd-smi invents.
struct GpuCpuLink {
  uint32_t numa_node_id;
  IO_LINK_TYPE type;
};

// Finds every CPU-destined io_link neighbor of the GPU at KFD node
// gpu_kfd_node_indx, regardless of link type (a GPU's p2p_links can
// additionally report *reachability* to a second, non-directly-connected
// CPU via the fabric -- that's not a real physical link, so only io_links
// are checked; a GPU has exactly one directly-connected CPU).
int DiscoverGpuCpuLinks(uint32_t gpu_kfd_node_indx, std::vector<GpuCpuLink>* out);

}  // namespace amd::smi

#endif  // INCLUDE_ROCM_SMI_ROCM_SMI_CPU_LINK_H_
