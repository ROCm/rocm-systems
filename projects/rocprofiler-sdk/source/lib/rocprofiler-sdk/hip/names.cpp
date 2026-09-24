// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/hip/names.hpp"

#include "lib/rocprofiler-sdk/hip/hip.hpp"

#include <rocprofiler-sdk/hip/table_id.h>

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocprofiler
{
namespace hip
{
namespace names_detail
{
template <size_t TableIdx, size_t OpIdx>
struct hip_api_name_info;
}  // namespace names_detail
}  // namespace hip
}  // namespace rocprofiler

// Pull in hip/defines.hpp ahead of hip.def.cpp so that its own #include of the header is a
// no-op and the macros below survive. The real definitions additionally build the
// interception functors and the argument stringifiers; the latter route through
// details/ostream.hpp, which is why hip.cpp cannot be compiled on Windows and why this
// translation unit re-expands the list keeping only the names.
#include "lib/rocprofiler-sdk/hip/defines.hpp"

#undef HIP_API_INFO_DEFINITION_0
#undef HIP_API_INFO_DEFINITION_V
#undef HIP_API_TABLE_LOOKUP_DEFINITION

#define HIP_API_NAME_INFO_DEFINITION(HIP_TABLE, HIP_API_ID, HIP_FUNC)                              \
    namespace rocprofiler                                                                          \
    {                                                                                              \
    namespace hip                                                                                  \
    {                                                                                              \
    namespace names_detail                                                                         \
    {                                                                                              \
    template <>                                                                                    \
    struct hip_api_name_info<HIP_TABLE, HIP_API_ID>                                                \
    {                                                                                              \
        static constexpr auto name = #HIP_FUNC;                                                    \
    };                                                                                             \
    }                                                                                              \
    }                                                                                              \
    }

#define HIP_API_INFO_DEFINITION_0(HIP_TABLE, HIP_API_ID, HIP_FUNC, HIP_FUNC_PTR)                   \
    HIP_API_NAME_INFO_DEFINITION(HIP_TABLE, HIP_API_ID, HIP_FUNC)

#define HIP_API_INFO_DEFINITION_V(HIP_TABLE, HIP_API_ID, HIP_FUNC, HIP_FUNC_PTR, ...)              \
    HIP_API_NAME_INFO_DEFINITION(HIP_TABLE, HIP_API_ID, HIP_FUNC)

#define HIP_API_TABLE_LOOKUP_DEFINITION(TABLE_ID, TYPE, MEMBER)

#define ROCPROFILER_LIB_ROCPROFILER_HIP_HIP_CPP_IMPL 1
#include "lib/rocprofiler-sdk/hip/hip.def.cpp"

namespace rocprofiler
{
namespace hip
{
namespace
{
template <size_t TableIdx, size_t... OpIdx>
std::vector<const char*> build_name_table(std::index_sequence<OpIdx...>)
{
    return std::vector<const char*>{names_detail::hip_api_name_info<TableIdx, OpIdx>::name...};
}

// indexed by operation id; dense because hip.def.cpp defines an entry for every enumerator
template <size_t TableIdx>
const std::vector<const char*>&
get_name_table()
{
    static const auto _v =
        build_name_table<TableIdx>(std::make_index_sequence<hip_domain_info<TableIdx>::last>{});
    return _v;
}

template <size_t TableIdx>
const std::unordered_map<std::string_view, uint32_t>&
get_id_table()
{
    static const auto _v = []() {
        const auto& _names = get_name_table<TableIdx>();
        auto        _data  = std::unordered_map<std::string_view, uint32_t>{};
        _data.reserve(_names.size());
        for(size_t i = 0; i < _names.size(); ++i)
            _data.emplace(std::string_view{_names.at(i)}, static_cast<uint32_t>(i));
        return _data;
    }();
    return _v;
}

template <size_t TableIdx>
const char*
name_by_id_tmpl(uint32_t id)
{
    const auto& _names = get_name_table<TableIdx>();
    if(id >= _names.size()) return nullptr;
    return _names.at(id);
}

template <size_t TableIdx>
uint32_t
id_by_name_tmpl(const char* name)
{
    if(name == nullptr) return hip_domain_info<TableIdx>::none;

    const auto& _ids = get_id_table<TableIdx>();
    auto        itr  = _ids.find(std::string_view{name});
    if(itr == _ids.end()) return hip_domain_info<TableIdx>::none;
    return itr->second;
}

template <size_t TableIdx>
std::vector<uint32_t>
get_ids_tmpl()
{
    const auto& _names = get_name_table<TableIdx>();
    auto        _data  = std::vector<uint32_t>{};
    _data.reserve(_names.size());
    for(size_t i = 0; i < _names.size(); ++i)
        _data.emplace_back(static_cast<uint32_t>(i));
    return _data;
}
}  // namespace

const char*
name_by_id_impl(uint32_t table_idx, uint32_t id)
{
    switch(table_idx)
    {
        case ROCPROFILER_HIP_TABLE_ID_Compiler:
            return name_by_id_tmpl<ROCPROFILER_HIP_TABLE_ID_Compiler>(id);
        case ROCPROFILER_HIP_TABLE_ID_Runtime:
            return name_by_id_tmpl<ROCPROFILER_HIP_TABLE_ID_Runtime>(id);
        default: return nullptr;
    }
}

uint32_t
id_by_name_impl(uint32_t table_idx, const char* name)
{
    switch(table_idx)
    {
        case ROCPROFILER_HIP_TABLE_ID_Compiler:
            return id_by_name_tmpl<ROCPROFILER_HIP_TABLE_ID_Compiler>(name);
        case ROCPROFILER_HIP_TABLE_ID_Runtime:
            return id_by_name_tmpl<ROCPROFILER_HIP_TABLE_ID_Runtime>(name);
        default: return ROCPROFILER_HIP_RUNTIME_API_ID_NONE;
    }
}

std::vector<const char*>
get_names_impl(uint32_t table_idx)
{
    switch(table_idx)
    {
        case ROCPROFILER_HIP_TABLE_ID_Compiler:
            return get_name_table<ROCPROFILER_HIP_TABLE_ID_Compiler>();
        case ROCPROFILER_HIP_TABLE_ID_Runtime:
            return get_name_table<ROCPROFILER_HIP_TABLE_ID_Runtime>();
        default: return {};
    }
}

std::vector<uint32_t>
get_ids_impl(uint32_t table_idx)
{
    switch(table_idx)
    {
        case ROCPROFILER_HIP_TABLE_ID_Compiler:
            return get_ids_tmpl<ROCPROFILER_HIP_TABLE_ID_Compiler>();
        case ROCPROFILER_HIP_TABLE_ID_Runtime:
            return get_ids_tmpl<ROCPROFILER_HIP_TABLE_ID_Runtime>();
        default: return {};
    }
}

#if defined(_WIN32)
template <>
const char*
name_by_id<ROCPROFILER_HIP_TABLE_ID_Runtime>(uint32_t id)
{
    return name_by_id_tmpl<ROCPROFILER_HIP_TABLE_ID_Runtime>(id);
}

template <>
const char*
name_by_id<ROCPROFILER_HIP_TABLE_ID_Compiler>(uint32_t id)
{
    return name_by_id_tmpl<ROCPROFILER_HIP_TABLE_ID_Compiler>(id);
}

template <>
uint32_t
id_by_name<ROCPROFILER_HIP_TABLE_ID_Runtime>(const char* name)
{
    return id_by_name_tmpl<ROCPROFILER_HIP_TABLE_ID_Runtime>(name);
}

template <>
uint32_t
id_by_name<ROCPROFILER_HIP_TABLE_ID_Compiler>(const char* name)
{
    return id_by_name_tmpl<ROCPROFILER_HIP_TABLE_ID_Compiler>(name);
}

template <>
std::vector<const char*>
get_names<ROCPROFILER_HIP_TABLE_ID_Runtime>()
{
    return get_name_table<ROCPROFILER_HIP_TABLE_ID_Runtime>();
}

template <>
std::vector<const char*>
get_names<ROCPROFILER_HIP_TABLE_ID_Compiler>()
{
    return get_name_table<ROCPROFILER_HIP_TABLE_ID_Compiler>();
}

template <>
std::vector<uint32_t>
get_ids<ROCPROFILER_HIP_TABLE_ID_Runtime>()
{
    return get_ids_tmpl<ROCPROFILER_HIP_TABLE_ID_Runtime>();
}

template <>
std::vector<uint32_t>
get_ids<ROCPROFILER_HIP_TABLE_ID_Compiler>()
{
    return get_ids_tmpl<ROCPROFILER_HIP_TABLE_ID_Compiler>();
}
#endif
}  // namespace hip
}  // namespace rocprofiler
