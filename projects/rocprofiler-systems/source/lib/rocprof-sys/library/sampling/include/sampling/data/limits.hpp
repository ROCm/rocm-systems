// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>

namespace rocprofsys::sampling
{

inline constexpr size_t MAX_STACK_DEPTH      = 64;
inline constexpr size_t RING_BUFFER_CAPACITY = 2048;
inline constexpr size_t MAX_THREADS_DEFAULT  = 512;
inline constexpr size_t METRICS_COUNT        = 6;
inline constexpr size_t PAPI_EVENT_COUNT     = 12;

}  // namespace rocprofsys::sampling
