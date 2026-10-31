// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "options/option_spec.hpp"

#include <span>

namespace rocprofsys::rocsys::options
{

/// @brief Shipped rocsys options. A new option is one row in this table.
[[nodiscard]] std::span<const option_spec>
option_table() noexcept;

}  // namespace rocprofsys::rocsys::options
