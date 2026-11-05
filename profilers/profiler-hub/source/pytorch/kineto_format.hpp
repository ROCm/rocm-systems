// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <string_view>

namespace profiler_hub::pytorch
{
bool
looks_like_kineto(std::string_view text);
}  // namespace profiler_hub::pytorch
