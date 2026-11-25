// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "util/intrusive_shared_ptr.h"

#include <atomic>
#include <cstdint>

namespace rocjitsu::amdgpu {

/// @brief Retained page-policy epoch with its value isolated from the ownership counter.
using VmMutationEpoch = util::RefCountedCacheAligned<std::atomic<uint64_t>>;

} // namespace rocjitsu::amdgpu
