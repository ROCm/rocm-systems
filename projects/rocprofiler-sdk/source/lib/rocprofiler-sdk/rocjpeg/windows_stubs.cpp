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

// Windows stand-in for rocjpeg.cpp and abi.cpp. rocJPEG has no Windows build, so its API
// table is never captured and the domain carries no operations.

// ROCm's amd_hip_vector_types.h spells its members std::int32_t but does not include
// <cstdint> itself, so under MSVC it only compiles if something already has.
#include <cstdint>

#include "lib/rocprofiler-sdk/rocjpeg/rocjpeg.hpp"

#include <rocprofiler-sdk/rocjpeg/table_id.h>

#include <vector>

namespace rocprofiler
{
namespace rocjpeg
{
template <size_t TableIdx>
const char* name_by_id(uint32_t)
{
    return nullptr;
}

template <size_t TableIdx>
std::vector<uint32_t>
get_ids()
{
    return {};
}

template const char* name_by_id<ROCPROFILER_ROCJPEG_TABLE_ID_CORE>(uint32_t);
template std::vector<uint32_t>
get_ids<ROCPROFILER_ROCJPEG_TABLE_ID_CORE>();
}  // namespace rocjpeg
}  // namespace rocprofiler
