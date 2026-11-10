// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/cpp/reader_types.hpp"

#include <string>

namespace profiler_hub::pytorch
{
bool
kineto_prefix_matches(const std::string& path);

reader_types::file_probe_result_t
probe_kineto(const std::string& path);
}  // namespace profiler_hub::pytorch
