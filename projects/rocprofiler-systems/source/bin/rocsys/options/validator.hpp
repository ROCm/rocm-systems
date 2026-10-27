// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "options/parser.hpp"
#include "result.hpp"

namespace rocprofsys::rocsys::options
{

/// @brief Check capabilities, required values, and per-binary legacy conflicts.
///        Returns @p command on success.
[[nodiscard]] result<parsed_command>
validate(parsed_command command);

/// @brief Call each option's resolve hook and fold ops that write the same key.
///        Different values are an error naming both origins. Equal ops become one.
[[nodiscard]] result<env_plan>
resolve(const parsed_command& command, const context& ctx);

}  // namespace rocprofsys::rocsys::options
