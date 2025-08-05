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

#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/utility.hpp"

#include <rocprofiler-sdk-rocpd/defines.h>
#include <rocprofiler-sdk-rocpd/rocpd.h>
#include <rocprofiler-sdk-rocpd/types.h>

#include <array>
#include <atomic>
#include <cassert>

namespace rocpd
{
namespace
{
#define ROCPD_STATUS_STRING(CODE, MSG)                                                             \
    template <>                                                                                    \
    struct status_string<CODE>                                                                     \
    {                                                                                              \
        static constexpr auto name  = #CODE;                                                       \
        static constexpr auto value = MSG;                                                         \
    };

template <size_t Idx>
struct status_string;

ROCPD_STATUS_STRING(ROCPD_STATUS_SUCCESS, "Success")
ROCPD_STATUS_STRING(ROCPD_STATUS_ERROR, "General error")
ROCPD_STATUS_STRING(ROCPD_STATUS_ERROR_INVALID_ARGUMENT, "Invalid function argument")
ROCPD_STATUS_STRING(ROCPD_STATUS_ERROR_SQL_ERROR, "General SQL error")
ROCPD_STATUS_STRING(ROCPD_STATUS_ERROR_SQL_INVALID_ENGINE, "Invalid SQL engine")
ROCPD_STATUS_STRING(ROCPD_STATUS_ERROR_SQL_INVALID_SCHEMA_KIND, "Invalid SQL schema kind")
ROCPD_STATUS_STRING(ROCPD_STATUS_ERROR_SQL_SCHEMA_NOT_FOUND, "SQL schema not found")
ROCPD_STATUS_STRING(ROCPD_STATUS_ERROR_SQL_SCHEMA_PERMISSION_DENIED, "SQL schema could not be read")

template <size_t Idx, size_t... Tail>
const char*
get_status_name(rocpd_status_t status, std::index_sequence<Idx, Tail...>)
{
    if(status == Idx) return status_string<Idx>::name;
    // recursion until tail empty
    if constexpr(sizeof...(Tail) > 0)
        return get_status_name(status, std::index_sequence<Tail...>{});
    return nullptr;
}

template <size_t Idx, size_t... Tail>
const char*
get_status_string(rocpd_status_t status, std::index_sequence<Idx, Tail...>)
{
    if(status == Idx) return status_string<Idx>::value;
    // recursion until tail empty
    if constexpr(sizeof...(Tail) > 0)
        return get_status_string(status, std::index_sequence<Tail...>{});
    return nullptr;
}
}  // namespace

// force initialization of logging on library load
bool _rocpd_init_logging = (rocprofiler::common::init_logging("ROCPD"), true);
}  // namespace rocpd

ROCPD_EXTERN_C_INIT

rocpd_status_t
rocpd_get_version(uint32_t* major, uint32_t* minor, uint32_t* patch)
{
    if(major) *major = ROCPD_VERSION_MAJOR;
    if(minor) *minor = ROCPD_VERSION_MINOR;
    if(patch) *patch = ROCPD_VERSION_PATCH;
    return ROCPD_STATUS_SUCCESS;
}

rocpd_status_t
rocpd_get_version_triplet(rocpd_version_triplet_t* info)
{
    *info = {
        .major = ROCPD_VERSION_MAJOR, .minor = ROCPD_VERSION_MINOR, .patch = ROCPD_VERSION_PATCH};
    return ROCPD_STATUS_SUCCESS;
}

const char*
rocpd_get_status_name(rocpd_status_t status)
{
    return rocpd::get_status_name(status, std::make_index_sequence<ROCPD_STATUS_LAST>{});
}

const char*
rocpd_get_status_string(rocpd_status_t status)
{
    return rocpd::get_status_string(status, std::make_index_sequence<ROCPD_STATUS_LAST>{});
}

ROCPD_EXTERN_C_FINI
