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

#include "lib/common/defines.hpp"
#include "lib/rocprofiler/hsa/hsa-defines.hpp"
#include "lib/rocprofiler/hsa/hsa-ostream.hpp"
#include "lib/rocprofiler/hsa/hsa-types.h"
#include "lib/rocprofiler/hsa/hsa-utils.hpp"

#include <hsa/hsa_api_trace.h>
#include <rocprofiler/rocprofiler.h>

#include <cstdint>

namespace rocprofiler
{
namespace hsa
{
using activity_functor_t = int (*)(rocprofiler_tracer_activity_domain_t domain,
                                   uint32_t                             operation_id,
                                   void*                                data);

using hsa_api_table_t = HsaApiTable;

struct hsa_trace_data_t
{
    hsa_api_data_t api_data;
    uint64_t       phase_enter_timestamp;
    uint64_t       phase_data;

    void (*phase_enter)(hsa_api_id_t operation_id, hsa_trace_data_t* data);
    void (*phase_exit)(hsa_api_id_t operation_id, hsa_trace_data_t* data);
};

enum hsa_table_api_id_t
{
    HSA_API_TABLE_ID_CoreApi,
    HSA_API_TABLE_ID_AmdExt,
    HSA_API_TABLE_ID_ImageExt,
    HSA_API_TABLE_ID_NUMBER,
};

template <typename DataT, typename Tp>
void
set_data_retval(DataT&, Tp);

template <size_t Idx>
struct hsa_table_lookup;

template <size_t Idx>
struct hsa_api_impl
{
    template <typename DataT, typename DataArgsT, typename... Args>
    static auto phase_enter(DataT& _data, DataArgsT&, Args... args);

    template <typename DataT, typename... Args>
    static auto phase_exit(DataT& _data);

    template <typename DataT, typename FuncT, typename... Args>
    static auto exec(DataT& _data, FuncT&&, Args&&... args);

    template <typename... Args>
    static auto functor(Args&&... args);
};

template <size_t Idx>
struct hsa_api_info;

const char*
hsa_api_name(uint32_t id);

uint32_t
hsa_api_id_by_name(const char* name);

std::string
hsa_api_data_string(uint32_t id, const hsa_trace_data_t& _data);

std::string
hsa_api_named_data_string(uint32_t id, const hsa_trace_data_t& _data);

void
hsa_api_iterate_args(uint32_t                id,
                     const hsa_trace_data_t& _data,
                     int (*_func)(const char*, const char*));

std::vector<const char*>
hsa_api_get_names();

std::vector<uint32_t>
hsa_api_get_ids();

void
hsa_api_set_callback(activity_functor_t _func);
}  // namespace hsa
}  // namespace rocprofiler

extern "C" {
using on_load_t = bool (*)(HsaApiTable*, uint64_t, uint64_t, const char* const*);

bool
OnLoad(HsaApiTable*       table,
       uint64_t           runtime_version,
       uint64_t           failed_tool_count,
       const char* const* failed_tool_names) ROCPROFILER_PUBLIC_API;
}
