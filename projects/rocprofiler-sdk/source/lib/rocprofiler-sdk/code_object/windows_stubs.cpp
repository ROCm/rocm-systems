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

// Windows stand-in for code_object.cpp, which needs ELF parsing to identify loaded code
// objects. The name table is what callback_tracing.cpp queries; an empty id list reports
// the tracing kind as having no operations, which is what a caller should see here.

// ROCm's amd_hip_vector_types.h spells its members std::int32_t but does not include
// <cstdint> itself, so under MSVC it only compiles if something already has.
#include <cstdint>

#include "lib/rocprofiler-sdk/code_object/code_object.hpp"

#include <vector>

namespace rocprofiler
{
namespace code_object
{
const char* name_by_id(uint32_t) { return nullptr; }

std::vector<uint32_t>
get_ids()
{
    return {};
}
}  // namespace code_object
}  // namespace rocprofiler
