// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/registration/iterate.hpp"
#include "lib/common/logging.hpp"

#include <dlfcn.h>
#include <vector>

namespace rocprofiler
{
namespace registration
{
namespace iterate
{
namespace
{
// Define minimal types for rocprofiler-register API to avoid build dependency
// These match the definitions in rocprofiler-register.h
struct rocprofiler_register_registration_info_t
{
    size_t      size;
    const char* common_name;
    uint32_t    lib_version;
    uint64_t    api_table_length;
};

using rocprofiler_register_registration_info_cb_t =
    int (*)(rocprofiler_register_registration_info_t*, void*);

using rocprofiler_register_iterate_registration_info_fn_t =
    int (*)(rocprofiler_register_registration_info_cb_t, void*);

// Callback to collect registered API table names
int
rocprofiler_register_iterate_registration_callback(rocprofiler_register_registration_info_t* info,
                                                   void*                                     data)
{
    constexpr auto unknown_name = "<unknown>";
    const auto*    _common_name = (info && info->common_name) ? info->common_name : unknown_name;
    auto*          _data = static_cast<std::vector<rocprofiler_runtime_registration_info_t>*>(data);
    _data->emplace_back(rocprofiler_runtime_registration_info_t{
        sizeof(rocprofiler_runtime_registration_info_t),
        _common_name,
        info->lib_version,
        info->api_table_length,
    });
    return 0;
}
}  // namespace

std::optional<runtime_registration_vec_t>
get_runtime_registrations()
{
    auto _data = std::vector<rocprofiler_runtime_registration_info_t>{};

    // Step 1: Get the rocprofiler_register_iterate_registration_info function
    // Use dlsym(nullptr, ...) since rocprofiler-register should already be loaded
    auto* iterate_info_fn = reinterpret_cast<rocprofiler_register_iterate_registration_info_fn_t>(
        dlsym(nullptr, "rocprofiler_register_iterate_registration_info"));

    if(!iterate_info_fn) return std::nullopt;

    // Step 2: Check if any dispatch tables have been registered
    // Collect the names of all registrations using iterate_registration_info
    iterate_info_fn(rocprofiler_register_iterate_registration_callback, &_data);

    return _data;
}
}  // namespace iterate
}  // namespace registration
}  // namespace rocprofiler
