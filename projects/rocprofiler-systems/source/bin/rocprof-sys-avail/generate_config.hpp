// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common.hpp"

#include <set>
#include <string>

void
generate_config(std::string _config_file, const std::set<std::string>& _config_fmts,
                const std::array<bool, TOTAL>&, const format_options&  fmt_opts);
