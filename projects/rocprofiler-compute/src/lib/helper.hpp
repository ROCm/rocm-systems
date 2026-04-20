// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once
#include <regex>
#include <string>
#include <vector>

namespace helper_utils {

std::string truncate_name(std::string_view name);
std::string cxa_demangle(std::string_view _mangled_name, int *_status);
std::vector<std::string> split_by_regex(const std::string &s,
                                        const std::string &regex_pattern);

} // namespace helper_utils
