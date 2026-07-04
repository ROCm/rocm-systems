/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <sanitizer/gpu_sanitizer.h>

#include <cstddef>
#include <cstdint>

namespace amd {

class Device;

/// Report one GPU ConcurrencySanitizer race detected on \p dev, dropping it if
/// it was seen before.
void reportGpuCSanRace(const amd::Device& dev, const __tsan_gpu_race& race);

}  // namespace amd
