// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocm_smi/rocm_smi_cpu_link.h"

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

    uint64_t node_to_gpu_id = 0;
    if (get_gpu_id(node_to, &node_to_gpu_id) != 0 || node_to_gpu_id != 0) {
      continue;  // not a CPU node
    }

    out->push_back(GpuCpuLink{node_to, link->type()});
  }

  return 0;
}

}  // namespace amd::smi
