// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCM_SMI_INCLUDE_ROCM_SMI_ROCM_SMI_NPM_H_
#define ROCM_SMI_INCLUDE_ROCM_SMI_ROCM_SMI_NPM_H_

#include <string>

#include "rocm_smi/rocm_smi.h"

namespace amd::smi {

// NPM board status and limit queries
rsmi_status_t get_npm_board_status(const std::string& board_path, bool* enabled);
rsmi_status_t get_npm_board_limit(const std::string& board_path, uint64_t* limit);

// NPM balancing mode query/set (board/npm_mode: "1"/"2"). A missing or
// unreadable board/npm_mode file returns RSMI_STATUS_NOT_SUPPORTED, matching
// set_npm_board_mode() and every sibling accessor in this file.
rsmi_status_t get_npm_board_mode(const std::string& board_path, std::string* mode);
rsmi_status_t set_npm_board_mode(const std::string& board_path, const std::string& mode);

// Platform max node-level power limit query (board/max_node_power_limit), used to
// bound requests made via set_npm_board_limit()/rsmi_dev_npm_limit_set().
rsmi_status_t get_npm_board_max_limit(const std::string& board_path, uint64_t* limit);

// NPM board limit set (Set NPM Limit request to GPU PMFW via amdgpu driver)
rsmi_status_t set_npm_board_limit(const std::string& board_path, uint64_t limit);

// UBB (baseboard) power limit query
rsmi_status_t get_ubb_power_limit(const std::string& board_path, uint64_t* limit);

// Current node power query (board/node_power)
rsmi_status_t get_npm_node_power(const std::string& board_path, uint64_t* power);

}  // namespace amd::smi
#endif  // ROCM_SMI_INCLUDE_ROCM_SMI_ROCM_SMI_NPM_H_
