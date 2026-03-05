/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 * See LICENSE file for full license text.
 */

#ifndef ROCM_SMI_INCLUDE_ROCM_SMI_ROCM_SMI_NPM_H_
#define ROCM_SMI_INCLUDE_ROCM_SMI_ROCM_SMI_NPM_H_

#include "rocm_smi/rocm_smi.h"
#include <string>

namespace amd::smi {

// NPM board status and limit queries
rsmi_status_t get_npm_board_status(const std::string &board_path, bool *enabled);
rsmi_status_t get_npm_board_limit(const std::string &board_path, uint64_t *limit);

// UBB (baseboard) power limit query
rsmi_status_t get_ubb_power_limit(const std::string &board_path, uint64_t *limit);

}  // namespace amd::smi
#endif  // ROCM_SMI_INCLUDE_ROCM_SMI_ROCM_SMI_NPM_H_
