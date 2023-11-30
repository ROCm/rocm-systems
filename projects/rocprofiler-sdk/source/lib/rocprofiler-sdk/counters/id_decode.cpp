// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/counters/id_decode.hpp"

#include <string>
#include <unordered_map>

#include "lib/common/utility.hpp"

namespace rocprofiler
{
namespace counters
{
const std::unordered_map<rocprofiler_profile_counter_instance_types, std::string>&
dimension_map()
{
    static std::unordered_map<rocprofiler_profile_counter_instance_types, std::string> map = {
        {ROCPROFILER_DIMENSION_NONE, "DIMENSION_NONE"},
        {ROCPROFILER_DIMENSION_XCC, "DIMENSION_XCC"},
        {ROCPROFILER_DIMENSION_SHADER_ENGINE, "DIMENSION_SHADER_ENGINE"},
        {ROCPROFILER_DIMENSION_AGENT, "DIMENSION_AGENT"},
        {ROCPROFILER_DIMENSION_PMC_CHANNEL, "DIMENSION_PMC_CHANNEL"},
        {ROCPROFILER_DIMENSION_CU, "DIMENSION_CU"},
    };
    return map;
}

}  // namespace counters
}  // namespace rocprofiler
