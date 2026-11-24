// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "simdojo/components/retained_backing_range.h"

namespace rocjitsu::amdgpu {

/// Binds the retained storage to the GPU whose VM may admit the memory effect.
struct GpuBackingRange {
  uint32_t owner_gpu_id = 0;
  simdojo::RetainedBackingRange storage;
};

} // namespace rocjitsu::amdgpu
