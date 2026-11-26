// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocm_smi/rocm_smi_cpu_link.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>

#include "rocm_smi/rocm_smi_kfd.h"

namespace amd::smi {

int DiscoverGpuCpuLinks(uint32_t gpu_kfd_node_indx, std::vector<GpuCpuLink>* out) {
  out->clear();

  std::map<uint32_t, std::shared_ptr<IOLink>> neighbors;
  int ret = DiscoverIOLinksPerNode(gpu_kfd_node_indx, &neighbors);
  if (ret != 0) {
    fprintf(stderr,
            "[CPU_LINK_DEBUG] DiscoverIOLinksPerNode(node=%u) failed: errno=%d (%s), "
            "neighbors_found_before_failure=%zu\n",
            gpu_kfd_node_indx, ret, strerror(ret), neighbors.size());
    return ret;
  }

  fprintf(stderr, "[CPU_LINK_DEBUG] node=%u has %zu io_link neighbor(s): [", gpu_kfd_node_indx,
          neighbors.size());
  for (const auto& kv : neighbors) {
    fprintf(stderr, "%u(type=%d) ", kv.first, static_cast<int>(kv.second->type()));
  }
  fprintf(stderr, "]\n");

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
    } else if (gpu_id_ret == ENXIO) {
      // gpu_id exists but didn't read as a clean number. The driver only
      // ever writes a plain decimal value here, so in practice this is a
      // neighbor this process can't see the identity of -- e.g. a peer GPU
      // whose device node isn't granted in a container/cgroup-restricted
      // environment, where the read is denied at the syscall level and
      // ReadKFDGpuId() observes that as empty content rather than a distinct
      // errno. Its identity is unknown, so skip it rather than guessing.
      continue;
    } else if (gpu_id_ret != 0) {
      fprintf(stderr,
              "[CPU_LINK_DEBUG] ReadKFDGpuId(node=%u, neighbor node_to=%u, link_type=%d) failed: "
              "errno=%d (%s)\n",
              gpu_kfd_node_indx, node_to, static_cast<int>(link->type()), gpu_id_ret,
              strerror(gpu_id_ret));
      return gpu_id_ret;
    } else if (node_to_gpu_id != 0) {
      continue;  // GPU peer, not a CPU node
    }

    out->push_back(GpuCpuLink{node_to, link->type()});
  }

  fprintf(stderr, "[CPU_LINK_DEBUG] node=%u resolved %zu CPU link(s)\n", gpu_kfd_node_indx,
          out->size());

  return 0;
}

}  // namespace amd::smi
