// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>
#include <string>
#include <string_view>

namespace rocjitsu::cli {

std::string json_quoted(std::string_view text);

void write_json_string(std::ostream &out, std::string_view text);

void write_json_string_field(std::ostream &out, std::string_view key, std::string_view value,
                             bool comma = true);

} // namespace rocjitsu::cli
