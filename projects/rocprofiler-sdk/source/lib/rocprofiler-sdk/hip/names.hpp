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

#pragma once

#include "lib/rocprofiler-sdk/hip/hip.hpp"

#include <rocprofiler-sdk/hip/table_id.h>

#include <cstdint>
#include <vector>

namespace rocprofiler
{
namespace hip
{
// Operation name <-> operation id lookup for the HIP API tables.
//
// hip.cpp provides the same mapping, but only as a side effect of instantiating the whole
// interception path, which does not compile on Windows: hip/defines.hpp routes argument
// stringification through details/ostream.hpp. These entry points re-expand the generated
// list in hip.def.cpp with name-only macros instead, so the tables are available wherever
// the ETW consumer needs to translate a traced function name into an operation id.
//
// The table index is a rocprofiler_hip_table_id_t. An unrecognized index yields nullptr /
// an empty vector; an unrecognized name or id yields nullptr / the domain's *_ID_NONE.
const char*
name_by_id_impl(uint32_t table_idx, uint32_t id);

uint32_t
id_by_name_impl(uint32_t table_idx, const char* name);

std::vector<const char*>
get_names_impl(uint32_t table_idx);

std::vector<uint32_t>
get_ids_impl(uint32_t table_idx);

#if defined(_WIN32)
// On Linux these are instantiated from hip.cpp. That translation unit is not built on
// Windows, so names.cpp defines them instead. Declared here so every call site sees the
// specialization before it is used.
template <>
const char*
name_by_id<ROCPROFILER_HIP_TABLE_ID_Runtime>(uint32_t id);

template <>
const char*
name_by_id<ROCPROFILER_HIP_TABLE_ID_Compiler>(uint32_t id);

template <>
uint32_t
id_by_name<ROCPROFILER_HIP_TABLE_ID_Runtime>(const char* name);

template <>
uint32_t
id_by_name<ROCPROFILER_HIP_TABLE_ID_Compiler>(const char* name);

template <>
std::vector<const char*>
get_names<ROCPROFILER_HIP_TABLE_ID_Runtime>();

template <>
std::vector<const char*>
get_names<ROCPROFILER_HIP_TABLE_ID_Compiler>();

template <>
std::vector<uint32_t>
get_ids<ROCPROFILER_HIP_TABLE_ID_Runtime>();

template <>
std::vector<uint32_t>
get_ids<ROCPROFILER_HIP_TABLE_ID_Compiler>();
#endif
}  // namespace hip
}  // namespace rocprofiler
