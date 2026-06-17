// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/preset_registry.hpp"

#include <optional>
#include <string>

namespace rocprofsys
{
namespace common_utils
{

struct domain_flag_state
{
    preset_registry    registry;
    std::string        active_preset_name;
    bool               export_config_requested = false;
    std::string        export_config_file;
    bool               gpu_domain_enabled      = false;
    bool               rocm_domain_enabled     = false;
    bool               cpu_domain_enabled      = false;
    bool               parallel_domain_enabled = false;
    std::optional<int> early_exit;
};

}  // namespace common_utils
}  // namespace rocprofsys
