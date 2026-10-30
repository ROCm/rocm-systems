// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "parser/schema.hpp"

namespace rocprofsys::parser
{
/// Built-in help catalog installed by initialize().
/// @throws std::runtime_error if a category or parameter id is duplicated.
[[nodiscard]] schema
get_default_schema();

}  // namespace rocprofsys::parser
