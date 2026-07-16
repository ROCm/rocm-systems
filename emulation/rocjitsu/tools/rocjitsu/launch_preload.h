// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

namespace rocjitsu::cli {

std::string find_loaded_asan_runtime();

std::string find_loaded_tsan_runtime();

void prepend_launch_preloads(const std::string &interposer_path);

} // namespace rocjitsu::cli
