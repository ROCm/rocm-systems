// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/env_op.hpp"
#include "options/option_spec.hpp"

#include <string>
#include <vector>

namespace rocprofsys::rocsys::options
{

/// @brief `--traces NAME...` for verbs that trace. Names replace the defaults:
///        listed names are on, and every other Phase 2 name is off.
extern const option_spec k_traces;

/// @brief Notes for runtimes `--traces` turns off that were on by library default
///        or by @p get_env. A null @p get_env means the environment is unset.
///        The caller prints each line after `note: `.
[[nodiscard]] std::vector<std::string>
disabled_runtime_notes(const env_plan& ops, char* (*get_env)(const char*) );

}  // namespace rocprofsys::rocsys::options
