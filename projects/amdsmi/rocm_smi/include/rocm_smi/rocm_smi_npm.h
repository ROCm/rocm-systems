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

// UBB (baseboard) power limit query
rsmi_status_t get_ubb_power_limit(const std::string& board_path, uint64_t* limit);

}  // namespace amd::smi
#endif  // ROCM_SMI_INCLUDE_ROCM_SMI_ROCM_SMI_NPM_H_
