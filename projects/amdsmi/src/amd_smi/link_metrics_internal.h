// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMD_SMI_LINK_METRICS_INTERNAL_H_
#define AMD_SMI_LINK_METRICS_INTERNAL_H_

#include <cstring>
#include <limits>

#include "amd_smi/amdsmi.h"

namespace amd {
namespace smi {
namespace detail {

template <typename ResolveBdf>
static amdsmi_status_t populate_link_metrics(const amdsmi_gpu_metrics_t& metrics,
                                             const uint16_t* link_to_dst_node,
                                             uint32_t mapped_count, amdsmi_status_t map_status,
                                             amdsmi_link_metrics_t* result,
                                             ResolveBdf resolve_bdf) {
  result->num_links = 0;
  // Without a port map (older kernels, non-XGMI GPUs), counters stay but peers are unknown.
  const bool map_available = map_status == AMDSMI_STATUS_SUCCESS;
  if (!map_available && map_status != AMDSMI_STATUS_NOT_SUPPORTED &&
      map_status != AMDSMI_STATUS_FILE_ERROR) {
    return map_status;
  }
  if (map_available && mapped_count > AMDSMI_MAX_NUM_XGMI_LINKS) {
    return AMDSMI_STATUS_UNEXPECTED_SIZE;
  }
  for (uint32_t slot = 0; slot < AMDSMI_MAX_NUM_XGMI_LINKS; ++slot) {
    const uint16_t dst_node =
        map_available ? link_to_dst_node[slot] : std::numeric_limits<uint16_t>::max();
    if (dst_node == std::numeric_limits<uint16_t>::max() &&
        metrics.xgmi_read_data_acc[slot] == std::numeric_limits<uint64_t>::max() &&
        metrics.xgmi_write_data_acc[slot] == std::numeric_limits<uint64_t>::max()) {
      continue;
    }
    auto& link = result->links[result->num_links++];
    link.max_bandwidth = std::numeric_limits<uint32_t>::max();
    link.bit_rate = std::numeric_limits<uint32_t>::max();
    memset(&link.bdf, 0xFF, sizeof(link.bdf));
    if (dst_node != std::numeric_limits<uint16_t>::max()) {
      link.bdf = resolve_bdf(dst_node);
    }
    link.read = metrics.xgmi_read_data_acc[slot];
    link.write = metrics.xgmi_write_data_acc[slot];
    link.link_type = AMDSMI_LINK_TYPE_XGMI;
    if (metrics.xgmi_link_speed != std::numeric_limits<uint16_t>::max()) {
      link.bit_rate = metrics.xgmi_link_speed;
    }
    if (metrics.xgmi_link_speed != std::numeric_limits<uint16_t>::max() &&
        metrics.xgmi_link_width != std::numeric_limits<uint16_t>::max()) {
      link.max_bandwidth = static_cast<uint32_t>(metrics.xgmi_link_speed) * metrics.xgmi_link_width;
    }
  }
  return AMDSMI_STATUS_SUCCESS;
}

}  // namespace detail
}  // namespace smi
}  // namespace amd

#endif  // AMD_SMI_LINK_METRICS_INTERNAL_H_
