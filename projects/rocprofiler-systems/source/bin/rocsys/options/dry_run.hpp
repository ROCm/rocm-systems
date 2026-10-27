// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "options/option_spec.hpp"

namespace rocprofsys::rocsys::options
{

/// @brief `--dry-run`. The launcher prints the plan and does not run the tool.
inline constexpr option_spec k_dry_run{
    .name    = "dry-run",
    .kind    = value_kind::flag,
    .summary = "Show the launch plan and exit",
};

}  // namespace rocprofsys::rocsys::options
