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

#include <rocprofiler-sdk/rocjpeg/details/rocjpeg_headers.h>

#if ROCPROFILER_SDK_USE_SYSTEM_ROCJPEG > 0
#    include <rocjpeg/amd_detail/rocjpeg_api_trace.h>
#    include <rocjpeg/rocjpeg.h>
#else
#    include <rocprofiler-sdk/rocjpeg/details/rocjpeg.h>
#    include <rocprofiler-sdk/rocjpeg/details/rocjpeg_api_trace.h>
#endif

#include <rocprofiler-sdk/callback_tracing.h>

#include <cstdint>
#include <vector>

namespace rocprofiler
{
namespace rocjpeg
{
using rocjpeg_api_func_table_t = ::RocJpegDispatchTable;

struct RocJPEGAPITable
{
    rocjpeg_api_func_table_t* rocjpeg_api_table = nullptr;
};

using rocjpeg_api_table_t = RocJPEGAPITable;

rocjpeg_api_table_t&
get_table();

template <size_t OpIdx>
struct rocjpeg_table_lookup;

template <typename Tp>
struct rocjpeg_table_id_lookup;

template <size_t TableIdx>
struct rocjpeg_domain_info;

template <size_t TableIdx, size_t OpIdx>
struct rocjpeg_api_info;

template <size_t TableIdx, size_t OpIdx>
struct rocjpeg_api_impl : rocjpeg_domain_info<TableIdx>
{
    template <typename DataArgsT, typename... Args>
    static auto set_data_args(DataArgsT&, Args... args);

    template <typename FuncT, typename... Args>
    static auto exec(FuncT&&, Args&&... args);

    template <typename RetT, typename... Args>
    static RetT functor(Args... args);
};

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
iterate_args(uint32_t                                               id,
             const rocprofiler_callback_tracing_rocjpeg_api_data_t& data,
             rocprofiler_callback_tracing_operation_args_cb_t       callback,
             int32_t                                                max_deref,
             void*                                                  user_data);

template <typename TableT>
void
copy_table(TableT* _orig, uint64_t _tbl_instance);

template <typename TableT>
void
update_table(TableT* _orig);

}  // namespace rocjpeg
}  // namespace rocprofiler
