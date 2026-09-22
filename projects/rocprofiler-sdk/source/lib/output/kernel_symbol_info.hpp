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
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace tool
{
using rocprofiler_code_object_info_t = rocprofiler_callback_tracing_code_object_load_data_t;
using code_object_info               = rocprofiler_code_object_info_t;
using code_object_data_vec_t         = std::vector<code_object_info>;
using code_object_data_map_t         = std::unordered_map<uint64_t, code_object_info>;

using rocprofiler_kernel_symbol_info_t =
    rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;

struct kernel_symbol_info : rocprofiler_kernel_symbol_info_t
{
    using base_type = rocprofiler_kernel_symbol_info_t;

    template <typename FuncT>
    kernel_symbol_info(const base_type& _base, FuncT&& _formatter)
    : base_type{_base}
    , formatted_kernel_name{_formatter(CHECK_NOTNULL(_base.kernel_name))}
    , demangled_kernel_name{common::cxx_demangle(CHECK_NOTNULL(_base.kernel_name))}
    , truncated_kernel_name{common::truncate_name(demangled_kernel_name)}
    {}

    kernel_symbol_info();
    ~kernel_symbol_info()                             = default;
    kernel_symbol_info(const kernel_symbol_info&)     = default;
    kernel_symbol_info(kernel_symbol_info&&) noexcept = default;
    kernel_symbol_info& operator=(const kernel_symbol_info&) = default;
    kernel_symbol_info& operator=(kernel_symbol_info&&) noexcept = default;

    std::string formatted_kernel_name = {};
    std::string demangled_kernel_name = {};
    std::string truncated_kernel_name = {};
};

using kernel_symbol_data_vec_t = std::vector<kernel_symbol_info>;
using kernel_symbol_data_map_t = std::unordered_map<rocprofiler_kernel_id_t, kernel_symbol_info>;
}  // namespace tool
}  // namespace rocprofiler

namespace cereal
{
#define SAVE_DATA_FIELD(FIELD) ar(make_nvp(#FIELD, data.FIELD))
#define LOAD_DATA_FIELD(FIELD) ar(make_nvp(#FIELD, data.FIELD))

template <typename ArchiveT>
void
save(ArchiveT& ar, const ::rocprofiler::tool::kernel_symbol_info& data)
{
    using base_type = ::rocprofiler::tool::kernel_symbol_info::base_type;

    cereal::save(ar, static_cast<const base_type&>(data));
    SAVE_DATA_FIELD(formatted_kernel_name);
    SAVE_DATA_FIELD(demangled_kernel_name);
    SAVE_DATA_FIELD(truncated_kernel_name);
}

template <typename ArchiveT>
void
load(ArchiveT& ar, ::rocprofiler::tool::kernel_symbol_info& data)
{
    using base_type = ::rocprofiler::tool::kernel_symbol_info::base_type;

    cereal::load(ar, static_cast<base_type&>(data));
    LOAD_DATA_FIELD(formatted_kernel_name);
    LOAD_DATA_FIELD(demangled_kernel_name);
    LOAD_DATA_FIELD(truncated_kernel_name);
}

#undef SAVE_DATA_FIELD
#undef LOAD_DATA_FIELD
}  // namespace cereal
