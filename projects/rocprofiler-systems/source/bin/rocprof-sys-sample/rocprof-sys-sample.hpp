// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <string_view>
#include <unordered_set>
#include <vector>

const std::unordered_set<std::string_view>&
get_updated_envs();

int
get_verbose_level();

std::vector<char*>
get_initial_environment();

std::vector<char*>
parse_args(int argc, char** argv, std::vector<char*>& envp);

void
add_torch_library_path(std::vector<char*>& envp, const std::vector<char*>& argv);
