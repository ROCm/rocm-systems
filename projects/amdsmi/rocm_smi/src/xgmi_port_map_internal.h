// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCM_SMI_XGMI_PORT_MAP_INTERNAL_H_
#define ROCM_SMI_XGMI_PORT_MAP_INTERNAL_H_

#include <algorithm>
#include <charconv>
#include <limits>
#include <sstream>
#include <string>

#include "rocm_smi/rocm_smi.h"

namespace amd {
namespace smi {
namespace detail {

static inline rsmi_status_t parse_xgmi_port_map(const std::string& contents, uint32_t* count,
                                                uint16_t* link_to_dst_node) {
  std::istringstream input(contents);
  std::string line;
  *count = 0;
  std::fill_n(link_to_dst_node, RSMI_MAX_NUM_XGMI_LINKS, std::numeric_limits<uint16_t>::max());
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    std::istringstream fields(line);
    std::string src_token, arrow, dst_token;
    if (!(fields >> src_token >> arrow >> dst_token) || arrow != "->") {
      return RSMI_STATUS_UNEXPECTED_DATA;
    }
    const size_t src_colon = src_token.find(':');
    const size_t dst_colon = dst_token.find(':');
    if (src_colon == std::string::npos || dst_colon == std::string::npos) {
      return RSMI_STATUS_UNEXPECTED_DATA;
    }
    uint32_t src_link = 0;
    uint32_t dst_node = 0;
    const auto* src_end = src_token.data() + src_token.size();
    const auto* dst_end = dst_token.data() + dst_colon;
    const auto src_result = std::from_chars(src_token.data() + src_colon + 1, src_end, src_link);
    const auto dst_result = std::from_chars(dst_token.data(), dst_end, dst_node);
    if (src_result.ec != std::errc{} || src_result.ptr != src_end || dst_result.ec != std::errc{} ||
        dst_result.ptr != dst_end || dst_node >= std::numeric_limits<uint16_t>::max()) {
      return RSMI_STATUS_UNEXPECTED_DATA;
    }
    if (src_link >= RSMI_MAX_NUM_XGMI_LINKS) return RSMI_STATUS_UNEXPECTED_SIZE;
    if (link_to_dst_node[src_link] != std::numeric_limits<uint16_t>::max()) {
      return RSMI_STATUS_UNEXPECTED_DATA;
    }
    link_to_dst_node[src_link] = static_cast<uint16_t>(dst_node);
    (*count)++;
  }
  return RSMI_STATUS_SUCCESS;
}

}  // namespace detail
}  // namespace smi
}  // namespace amd

#endif  // ROCM_SMI_XGMI_PORT_MAP_INTERNAL_H_
