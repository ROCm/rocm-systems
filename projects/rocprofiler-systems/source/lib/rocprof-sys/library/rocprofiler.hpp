// MIT License
//
// Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "core/defines.hpp"
#include "core/timemory.hpp"
#include "library/components/rocprofiler.hpp"

#include <timemory/backends/hardware_counters.hpp>
#include <timemory/macros.hpp>
#include <timemory/mpl/concepts.hpp>
#include <timemory/mpl/macros.hpp>

#include <array>
#include <atomic>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <list>
#include <map>
#include <string>
#include <string_view>
#include <tuple>
#include <unistd.h>
#include <utility>
#include <variant>
#include <vector>

namespace rocprofsys
{
namespace rocprofiler
{
std::map<uint32_t, std::vector<std::string_view>>
get_data_labels();

void
rocm_initialize();

void
rocm_cleanup();

bool&
is_setup();

void
post_process();

std::vector<component::rocm_info_entry>
rocm_metrics();

#if !defined(ROCPROFSYS_USE_ROCPROFILER) || ROCPROFSYS_USE_ROCPROFILER == 0
inline void
post_process()
{}

inline void
rocm_cleanup()
{}

inline std::vector<component::rocm_info_entry>
rocm_metrics()
{
    return std::vector<component::rocm_info_entry>{};
}
#endif

}  // namespace rocprofiler
}  // namespace rocprofsys
