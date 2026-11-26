// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Remaining legacy signaling comparisons commit their causes here.

#include "rocjitsu/vm/amdgpu/wavefront.h"

namespace rocjitsu::cdna1 {

// Accumulate guest causes and defer enabled-trap delivery to the wavefront.
inline void record_exceptions(amdgpu::Wavefront &wf, uint32_t causes) {
  wf.set_trapsts(wf.trapsts() | causes);
  wf.raise_alu_causes(causes);
}

} // namespace rocjitsu::cdna1
