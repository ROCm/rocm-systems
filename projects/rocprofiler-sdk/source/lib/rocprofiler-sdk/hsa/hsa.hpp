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

#include <hsa/hsa_api_trace.h>

#include <cstdint>
#include <string_view>
#include <vector>

#if defined(HSA_PC_SAMPLING_API_TABLE_MAJOR_VERSION) &&                                            \
    HSA_PC_SAMPLING_API_TABLE_MAJOR_VERSION > 0x0
#    define ROCPROFILER_SDK_HSA_PC_SAMPLING 1
#else
#    define ROCPROFILER_SDK_HSA_PC_SAMPLING 0
#endif

// redundant check based on whether the pc sampling API header was found
#if defined __has_include
#    if __has_include(<hsa/hsa_ven_amd_pc_sampling.h>)
#        if ROCPROFILER_SDK_HSA_PC_SAMPLING == 0
#            error                                                                                 \
                "rocprofiler-sdk disabled the HSA PC sampling table even though the hsa_ven_amd_pc_sampling.h was found"
#        endif
#    else
#        if ROCPROFILER_SDK_HSA_PC_SAMPLING == 1
#            error                                                                                 \
                "rocprofiler-sdk enabled the HSA PC sampling table even though the hsa_ven_amd_pc_sampling.h was not found"
#        endif
#    endif
#endif

namespace rocprofiler
{
namespace hsa
{
struct tracing_table
{};

struct internal_table
{};

using hsa_api_table_t      = ::HsaApiTable;
using hsa_table_version_t  = ::ApiTableVersion;
using hsa_core_table_t     = ::CoreApiTable;
using hsa_amd_ext_table_t  = ::AmdExtTable;
using hsa_fini_ext_table_t = ::FinalizerExtTable;
using hsa_img_ext_table_t  = ::ImageExtTable;
using hsa_amd_tool_table_t = ::ToolsApiTable;
#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
using hsa_pc_sampling_ext_table_t = ::PcSamplingExtTable;
#endif

hsa_api_table_t&
get_table();

hsa_table_version_t
get_table_version();

hsa_core_table_t*
get_core_table();

hsa_amd_ext_table_t*
get_amd_ext_table();

hsa_fini_ext_table_t*
get_fini_ext_table();

hsa_img_ext_table_t*
get_img_ext_table();

hsa_amd_tool_table_t*
get_amd_tool_table();

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
hsa_pc_sampling_ext_table_t*
get_pc_sampling_ext_table();
#endif

hsa_core_table_t*
get_tracing_core_table();

hsa_amd_ext_table_t*
get_tracing_amd_ext_table();

hsa_fini_ext_table_t*
get_tracing_fini_ext_table();

hsa_img_ext_table_t*
get_tracing_img_ext_table();

hsa_amd_tool_table_t*
get_tracing_amd_tool_table();

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
hsa_pc_sampling_ext_table_t*
get_tracing_pc_sampling_ext_table();
#endif

template <size_t Idx>
struct hsa_table_lookup;

template <typename Tp>
struct hsa_table_id_lookup;

template <size_t TableIdx>
struct hsa_domain_info;

template <size_t TableIdx, size_t OpIdx>
struct hsa_api_info;

template <size_t TableIdx, size_t OpIdx>
struct hsa_api_meta;

template <typename Tp>
struct hsa_api_func;

template <typename RetT, typename... Args>
struct hsa_api_func<RetT (*)(Args...)>
{
    using return_type   = RetT;
    using args_type     = std::tuple<Args...>;
    using function_type = RetT (*)(Args...);
};

template <typename RetT, typename... Args>
struct hsa_api_func<RetT (*)(Args...) noexcept> : hsa_api_func<RetT (*)(Args...)>
{};

template <size_t TableIdx, size_t OpIdx>
struct hsa_api_impl
{
    template <typename DataArgsT, typename... Args>
    static auto set_data_args(DataArgsT&, Args... args);

    template <typename FuncT, typename... Args>
    static auto exec(FuncT&&, Args&&... args);

    template <typename RetT, typename... Args>
    static RetT functor(Args... args);
};

std::string_view
get_hsa_status_string(hsa_status_t _status);

template <size_t TableIdx>
const char*
name_by_id(uint32_t id);

template <size_t TableIdx>
uint32_t
id_by_name(const char* name);

template <size_t TableIdx>
std::vector<const char*>
get_names();

template <size_t TableIdx>
std::vector<uint32_t>
get_ids();

template <size_t TableIdx>
void
iterate_args(uint32_t                                           id,
             const rocprofiler_callback_tracing_hsa_api_data_t& data,
             rocprofiler_callback_tracing_operation_args_cb_t   callback,
             int32_t                                            max_deref,
             void*                                              user_data);

template <typename TableT>
void
copy_table(TableT* _orig, uint64_t _tbl_instance);

template <typename TableT>
void
update_table(TableT* _orig, uint64_t _tbl_instance);
}  // namespace hsa
}  // namespace rocprofiler
