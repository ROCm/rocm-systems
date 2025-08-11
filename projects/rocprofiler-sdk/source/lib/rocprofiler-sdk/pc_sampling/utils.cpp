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

#include "lib/rocprofiler-sdk/pc_sampling/utils.hpp"
#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/defines.hpp"

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0

#    include "lib/rocprofiler-sdk/agent.hpp"

#    include <hsa/hsa_ext_amd.h>
#    include <hsa/hsa_ven_amd_pc_sampling.h>

#    include <stdexcept>

namespace rocprofiler
{
namespace pc_sampling
{
namespace utils
{
hsa_ven_amd_pcs_method_kind_t
get_matching_hsa_pcs_method(rocprofiler_pc_sampling_method_t method)
{
    switch(method)
    {
        case ROCPROFILER_PC_SAMPLING_METHOD_NONE: break;
        case ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC: return HSA_VEN_AMD_PCS_METHOD_STOCHASTIC_V1;
        case ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP: return HSA_VEN_AMD_PCS_METHOD_HOSTTRAP_V1;
        case ROCPROFILER_PC_SAMPLING_METHOD_LAST: break;
    }

    ROCP_FATAL << "Illegal pc sampling method " << method;
}

hsa_ven_amd_pcs_units_t
get_matching_hsa_pcs_units(rocprofiler_pc_sampling_unit_t unit)
{
    switch(unit)
    {
        case ROCPROFILER_PC_SAMPLING_UNIT_NONE: break;
        case ROCPROFILER_PC_SAMPLING_UNIT_INSTRUCTIONS:
            return HSA_VEN_AMD_PCS_INTERVAL_UNITS_INSTRUCTIONS;
        case ROCPROFILER_PC_SAMPLING_UNIT_CYCLES:
            return HSA_VEN_AMD_PCS_INTERVAL_UNITS_CLOCK_CYCLES;
        case ROCPROFILER_PC_SAMPLING_UNIT_TIME: return HSA_VEN_AMD_PCS_INTERVAL_UNITS_MICRO_SECONDS;
        case ROCPROFILER_PC_SAMPLING_UNIT_LAST: break;
    }

    ROCP_FATAL << "Illegal pc sampling unit " << unit;
}
}  // namespace utils
}  // namespace pc_sampling
}  // namespace rocprofiler

#endif
