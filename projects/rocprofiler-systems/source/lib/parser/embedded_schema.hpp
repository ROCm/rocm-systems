// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <string_view>

namespace rocprofsys::parser
{
/// JSON schema compiled into the library. Generated at build time.
[[nodiscard]] std::string_view
get_embedded_schema_json();

}  // namespace rocprofsys::parser
