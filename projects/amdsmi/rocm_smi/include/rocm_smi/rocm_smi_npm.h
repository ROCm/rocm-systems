// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCM_SMI_INCLUDE_ROCM_SMI_ROCM_SMI_NPM_H_
#define ROCM_SMI_INCLUDE_ROCM_SMI_ROCM_SMI_NPM_H_

#include <filesystem>
#include <string>

#include "rocm_smi/rocm_smi.h"

namespace amd::smi {

namespace fs = std::filesystem;

// MI4xx exposes NPM files under board/npm/; MI350 exposes the same files
// directly under board/. Returns board_path/npm if it exists, else board_path.
fs::path resolve_npm_dir(const fs::path& board_path);

// NPM board status query (npm_status). Resolved via resolve_npm_dir(): under
// board/npm/ if present, else flat board/ (MI350 backward compatibility).
rsmi_status_t get_npm_board_status(const std::string& board_path, bool* enabled);
// Current node power limit query (cur_node_power_limit). Resolved via
// resolve_npm_dir(), same fallback as get_npm_board_status().
rsmi_status_t get_npm_board_limit(const std::string& board_path, uint64_t* limit);

// NPM balancing mode query/set (mode: "1"/"2"). Resolved via
// resolve_npm_dir(), same fallback as get_npm_board_status(). A missing or
// unreadable file returns RSMI_STATUS_NOT_SUPPORTED, matching
// set_npm_board_mode() and every sibling accessor in this file.
rsmi_status_t get_npm_board_mode(const std::string& board_path, std::string* mode);
rsmi_status_t set_npm_board_mode(const std::string& board_path, const std::string& mode);

// Platform max node-level power limit query (max_node_power_limit), used to
// bound requests made via set_npm_board_limit()/rsmi_dev_npm_limit_set().
// Resolved via resolve_npm_dir(), same fallback as get_npm_board_status().
rsmi_status_t get_npm_board_max_limit(const std::string& board_path, uint64_t* limit);

// NPM board limit set (cur_node_power_limit: Set NPM Limit request to GPU
// PMFW via amdgpu driver). Resolved via resolve_npm_dir(), same fallback as
// get_npm_board_status().
rsmi_status_t set_npm_board_limit(const std::string& board_path, uint64_t limit);

// NPM supported-balancing-modes bitmask query (supported_mode: ASCII hex,
// e.g. "0x6"). Resolved via resolve_npm_dir(), same fallback as
// get_npm_board_status(). Bit N corresponds to enum value N of
// rsmi_npm_balancing_mode_t. A missing or unreadable file returns
// RSMI_STATUS_NOT_SUPPORTED, matching get_npm_board_mode(). Not gated on NPM
// enablement.
rsmi_status_t get_npm_supported_modes(const std::string& board_path, uint64_t* bitmask);

// UBB (baseboard) power limit query (baseboard_power_limit). Resolved via
// resolve_npm_dir(), same fallback as get_npm_board_status().
rsmi_status_t get_ubb_power_limit(const std::string& board_path, uint64_t* limit);

// Current node power query (node_power). Resolved via resolve_npm_dir(),
// same fallback as get_npm_board_status().
rsmi_status_t get_npm_node_power(const std::string& board_path, uint64_t* power);

}  // namespace amd::smi
#endif  // ROCM_SMI_INCLUDE_ROCM_SMI_ROCM_SMI_NPM_H_
