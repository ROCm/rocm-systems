// MIT License
//
// Copyright (c) 2023-2025 ROCm Developer Tools
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

#include "lib/rocprofiler-sdk/pc_sampling/defines.hpp"

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0

#    include <rocprofiler-sdk/fwd.h>
#    include <rocprofiler-sdk/pc_sampling.h>

#    include <hsa/hsa.h>
#    include <hsa/hsa_ven_amd_pc_sampling.h>

#    include <cstdint>
#    include <optional>

namespace rocprofiler
{
namespace pc_sampling
{
namespace utils
{
hsa_ven_amd_pcs_method_kind_t
get_matching_hsa_pcs_method(rocprofiler_pc_sampling_method_t method);

hsa_ven_amd_pcs_units_t
get_matching_hsa_pcs_units(rocprofiler_pc_sampling_unit_t unit);

constexpr size_t
get_hsa_pcs_latency()
{
    // TODO: Check with David about the default value in the hsa-runtime
    return 1000;
}

constexpr size_t
get_hsa_pcs_buffer_size()
{
    // TODO: Find the minimum size of all buffers and use that.
    return 64 * 1024 * sizeof(perf_sample_hosttrap_v1_t);  // 4MB
}
}  // namespace utils
}  // namespace pc_sampling
}  // namespace rocprofiler

#endif
