// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "parser/schema.hpp"

namespace rocprofsys::parser
{
/// Small catalog installed by initialize(). It is a demo, not the rocsys catalog.
/// @throws std::runtime_error if a parameter id is duplicated.
[[nodiscard]] schema
get_default_schema();

}  // namespace rocprofsys::parser
