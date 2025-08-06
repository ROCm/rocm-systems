// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/common/demangle.hpp"
#include "lib/common/logging.hpp"

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/cxx/hash.hpp>
#include <rocprofiler-sdk/cxx/name_info.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>
#include <rocprofiler-sdk/cxx/serialization.hpp>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace tool
{
using rocprofiler_host_kernel_symbol_data_t =
    rocprofiler_callback_tracing_code_object_host_kernel_symbol_register_data_t;

struct host_function_info : rocprofiler_host_kernel_symbol_data_t
{
    using base_type = rocprofiler_host_kernel_symbol_data_t;

    template <typename FuncT>
    host_function_info(const base_type& _base, FuncT&& _formatter)
    : base_type{_base}
    , formatted_host_function_name{_formatter(CHECK_NOTNULL(_base.device_function))}
    , demangled_host_function_name{common::cxx_demangle(CHECK_NOTNULL(_base.device_function))}
    , truncated_host_function_name{common::truncate_name(demangled_host_function_name)}
    {}

    host_function_info();
    ~host_function_info()                             = default;
    host_function_info(const host_function_info&)     = default;
    host_function_info(host_function_info&&) noexcept = default;
    host_function_info& operator=(const host_function_info&) = default;
    host_function_info& operator=(host_function_info&&) noexcept = default;

    std::string formatted_host_function_name = {};
    std::string demangled_host_function_name = {};
    std::string truncated_host_function_name = {};
};

using host_function_data_vec_t = std::vector<host_function_info>;
using host_function_info_map_t = std::unordered_map<uint64_t, host_function_info>;
}  // namespace tool
}  // namespace rocprofiler

namespace cereal
{
#define SAVE_DATA_FIELD(FIELD) ar(make_nvp(#FIELD, data.FIELD))

template <typename ArchiveT>
void
save(ArchiveT& ar, const ::rocprofiler::tool::host_function_info& data)
{
    cereal::save(
        ar, static_cast<const ::rocprofiler::tool::rocprofiler_host_kernel_symbol_data_t&>(data));
    SAVE_DATA_FIELD(formatted_host_function_name);
    SAVE_DATA_FIELD(demangled_host_function_name);
    SAVE_DATA_FIELD(truncated_host_function_name);
}

#undef SAVE_DATA_FIELD
}  // namespace cereal
