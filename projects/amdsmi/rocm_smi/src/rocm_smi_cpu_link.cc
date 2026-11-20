// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocm_smi/rocm_smi_cpu_link.h"

#include <cerrno>
#include <map>
#include <memory>

#include "rocm_smi/rocm_smi_kfd.h"

namespace amd::smi {

int DiscoverGpuCpuLinks(uint32_t gpu_kfd_node_indx, std::vector<GpuCpuLink>* out) {
  out->clear();

  std::map<uint32_t, std::shared_ptr<IOLink>> neighbors;
  int ret = DiscoverIOLinksPerNode(gpu_kfd_node_indx, &neighbors);
  if (ret != 0) {
    return ret;
  }

  for (const auto& kv : neighbors) {
    uint32_t node_to = kv.first;
    const std::shared_ptr<IOLink>& link = kv.second;

    // ReadKFDGpuId() is a plain sysfs read, unlike get_gpu_id(), which builds
    // a throwaway KFDNode and runs a full Initialize() per neighbor -- cheap
    // for a CPU neighbor but unnecessarily heavy (and exception-prone for a
    // non-CPU neighbor missing hive_id/io_links) for each of the peer GPUs.
    uint64_t node_to_gpu_id = 0;
    int gpu_id_ret = ReadKFDGpuId(node_to, &node_to_gpu_id);
    if (gpu_id_ret == ENOENT) {
      // No gpu_id file at all: node_to is a CPU, not a GPU.
    } else if (gpu_id_ret != 0) {
      return gpu_id_ret;
    } else if (node_to_gpu_id != 0) {
      continue;  // GPU peer, not a CPU node
    }

    out->push_back(GpuCpuLink{node_to, link->type()});
  }

  return 0;
}

}  // namespace amd::smi
