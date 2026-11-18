// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "lib/common/defines.hpp"

#include <cstdint>

namespace rocprofiler
{
namespace kfd
{
namespace capabilities
{
// Architecture decisions for direct KFD resources, within the SDK-supported GPU set.

// Wave32-capable GPUs use per-SIMD wave counts and 12-byte CWSR control entries.
constexpr bool
supports_wave32(uint32_t gfx_target_version)
{
    return ROCPROFILER_GFXIP_MAJOR(gfx_target_version) >= 10;
}

// gfx10 requires the fallback control-stack size to be capped at 0x7000 bytes.
constexpr bool
needs_cwsr_control_stack_cap(uint32_t gfx_target_version)
{
    return ROCPROFILER_GFXIP_MAJOR(gfx_target_version) == 10;
}

// KFD does not use an EOP buffer for AQL queues on gfx94x.
constexpr bool
needs_aql_eop_buffer(uint32_t gfx_target_version)
{
    return ROCPROFILER_GFXIP_MAJOR(gfx_target_version) != 9 ||
           ROCPROFILER_GFXIP_MINOR(gfx_target_version) != 4;
}

// Direct copies use the extended COPY count; older targets use the HSA backend.
constexpr bool
supports_extended_sdma_copy(uint32_t gfx_target_version)
{
    return gfx_target_version >= 90010;  // gfx90a
}

// gfx11.5/gfx12.5 and later minors use COPY scope fields instead of explicit GCR.
constexpr bool
has_sdma_copy_scope_fields(uint32_t gfx_target_version)
{
    const auto major = ROCPROFILER_GFXIP_MAJOR(gfx_target_version);
    const auto minor = ROCPROFILER_GFXIP_MINOR(gfx_target_version);
    return (major == 11 || major == 12) && minor >= 5;
}

constexpr bool
needs_sdma_gcr(uint32_t gfx_target_version)
{
    return ROCPROFILER_GFXIP_MAJOR(gfx_target_version) >= 10 &&
           !has_sdma_copy_scope_fields(gfx_target_version);
}

constexpr bool
needs_uncached_sdma_fence(uint32_t gfx_target_version)
{
    return ROCPROFILER_GFXIP_MAJOR(gfx_target_version) >= 10;
}

// FENCE's system-memory bit (and scope fields when available) are gfx12+ fields.
constexpr bool
has_sdma_fence_system_bit(uint32_t gfx_target_version)
{
    return ROCPROFILER_GFXIP_MAJOR(gfx_target_version) >= 12;
}
}  // namespace capabilities
}  // namespace kfd
}  // namespace rocprofiler
