// Copyright (c) 2018-2023 Advanced Micro Devices, Inc.
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

#include <rocprofiler/rocprofiler.h>

#include <cstdint>
#include <vector>

namespace rocprofiler
{
namespace hsa
{
using activity_functor_t = int (*)(rocprofiler_service_callback_tracing_kind_t domain,
                                   uint32_t                                    operation_id,
                                   void*                                       data);

using hsa_api_table_t = HsaApiTable;

hsa_api_table_t&
get_table();

template <size_t Idx>
struct hsa_table_lookup;

template <size_t Idx>
struct hsa_api_impl
{
    template <typename DataArgsT, typename... Args>
    static auto set_data_args(DataArgsT&, Args... args);

    template <typename FuncT, typename... Args>
    static auto exec(FuncT&&, Args&&... args);

    template <typename... Args>
    static auto functor(Args&&... args);
};

template <size_t Idx>
struct hsa_api_info;

const char*
name_by_id(uint32_t id);

uint32_t
id_by_name(const char* name);

void
iterate_args(uint32_t                                          id,
             const rocprofiler_hsa_api_callback_tracer_data_t& data,
             rocprofiler_callback_tracing_operation_args_cb_t  callback,
             void*                                             user_data);

std::vector<const char*>
get_names();

std::vector<uint32_t>
get_ids();

void
set_callback(activity_functor_t _func);

void
update_table(hsa_api_table_t* _orig);
}  // namespace hsa
}  // namespace rocprofiler
