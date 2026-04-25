// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/wavefront.h"

// Wavefront instances are created by IsaExecComputeUnit<Mode, Isa> in its constructor.
// No factory or standalone creation - wavefronts are permanently bound to
// their parent CU and slot index at construction time.

namespace rocjitsu {
namespace amdgpu {} // namespace amdgpu
} // namespace rocjitsu
