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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <rocprofiler-sdk/rocprofiler.h>

#include <hip/hip_version.h>

#if HIP_VERSION_MAJOR < 6
#    include "lib/rocprofiler-sdk/hip/details/hip_api_trace.hpp"
#else
#    include <hip/amd_detail/hip_api_trace.hpp>
#endif

#include <cstdint>
#include <vector>

namespace rocprofiler
{
namespace hip
{
using hip_compiler_api_table_t = HipCompilerDispatchTable;
using hip_runtime_api_table_t  = HipDispatchTable;

struct HipApiTable
{
    hip_compiler_api_table_t* compiler = nullptr;
    hip_runtime_api_table_t*  runtime  = nullptr;
};

using hip_api_table_t = HipApiTable;

hip_api_table_t&
get_table();

template <size_t OpIdx>
struct hip_table_lookup;

template <size_t TableIdx>
struct hip_domain_info;

template <size_t TableIdx, size_t OpIdx>
struct hip_api_info;

template <size_t TableIdx, size_t OpIdx>
struct hip_api_impl : hip_domain_info<TableIdx>
{
    template <typename DataArgsT, typename... Args>
    static auto set_data_args(DataArgsT&, Args... args);

    template <typename FuncT, typename... Args>
    static auto exec(FuncT&&, Args&&... args);

    template <typename... Args>
    static auto functor(Args&&... args);
};

template <size_t TableIdx>
const char*
name_by_id(uint32_t id);

template <size_t TableIdx>
uint32_t
id_by_name(const char* name);

void
iterate_args(uint32_t                                           id,
             const rocprofiler_callback_tracing_hip_api_data_t& data,
             rocprofiler_callback_tracing_operation_args_cb_t   callback,
             void*                                              user_data);

template <size_t TableIdx>
std::vector<const char*>
get_names();

template <size_t TableIdx>
std::vector<uint32_t>
get_ids();

void
copy_table(hip_compiler_api_table_t* _orig);

void
copy_table(hip_runtime_api_table_t* _orig);

void
update_table(hip_compiler_api_table_t* _orig);

void
update_table(hip_runtime_api_table_t* _orig);
}  // namespace hip
}  // namespace rocprofiler
