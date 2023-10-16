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

#include <rocprofiler/hsa.h>
#include <rocprofiler/version.h>

#include "lib/common/defines.hpp"

#ifndef ROCPROFILER_UNSAFE_NO_VERSION_CHECK
#    if defined(ROCPROFILER_CI) && ROCPROFILER_CI > 0
#        if HSA_API_TABLE_MAJOR_VERSION <= 0x01
namespace rocprofiler
{
namespace hsa
{
static_assert(HSA_CORE_API_TABLE_MAJOR_VERSION == 0x01,
              "Change in the major version of HSA core API table");
static_assert(HSA_AMD_EXT_API_TABLE_MAJOR_VERSION == 0x01,
              "Change in the major version of HSA amd-extended API table");
static_assert(HSA_FINALIZER_API_TABLE_MAJOR_VERSION == 0x01,
              "Change in the major version of HSA finalizer API table");
static_assert(HSA_IMAGE_API_TABLE_MAJOR_VERSION == 0x01,
              "Change in the major version of HSA image API table");

static_assert(HSA_CORE_API_TABLE_STEP_VERSION == 0x00,
              "Change in the major version of HSA core API table");
static_assert(HSA_AMD_EXT_API_TABLE_STEP_VERSION == 0x00,
              "Change in the major version of HSA amd-extended API table");
static_assert(HSA_FINALIZER_API_TABLE_STEP_VERSION == 0x00,
              "Change in the major version of HSA finalizer API table");
static_assert(HSA_IMAGE_API_TABLE_STEP_VERSION == 0x00,
              "Change in the major version of HSA image API table");

// this should always be updated to latest table size
template <size_t VersionCode>
struct table_size;

// latest version of hsa runtime that has been updated for support by rocprofiler
// and the current version of hsa runtime during this compilation
constexpr size_t latest_version  = ROCPROFILER_COMPUTE_VERSION(1, 11, 0);
constexpr size_t current_version = ROCPROFILER_HSA_RUNTIME_VERSION;

// aliases to the template specializations providing the table size info
using current_table_size_t = table_size<current_version>;
using latest_table_size_t  = table_size<latest_version>;

// specialization for v1.9
template <>
struct table_size<ROCPROFILER_COMPUTE_VERSION(1, 9, 0)>
{
    static constexpr size_t finalizer_ext = 64;
    static constexpr size_t image_ext     = 120;
    static constexpr size_t amd_ext       = 456;
    static constexpr size_t core_api_ext  = 1016;
};

// specialization for v1.10 - increased amd_ext by 10 functions
template <>
struct table_size<ROCPROFILER_COMPUTE_VERSION(1, 10, 0)>
: table_size<ROCPROFILER_COMPUTE_VERSION(1, 9, 0)>
{
    static constexpr size_t amd_ext = 552;
};

// version 1.11 is same as 1.10
template <>
struct table_size<ROCPROFILER_COMPUTE_VERSION(1, 11, 0)>
: table_size<ROCPROFILER_COMPUTE_VERSION(1, 10, 0)>
{};

// default static asserts to check against latest version
// e.g. v1.12 might have the same table sizes as v1.11 so
// we don't want to fail to compile if nothing has changed
template <size_t VersionCode>
struct table_size : latest_table_size_t
{};

// if you hit these static asserts, that means HSA added entries to the table but did not update the
// step numbers
static_assert(sizeof(FinalizerExtTable) == current_table_size_t::finalizer_ext,
              "HSA finalizer API table size changed or version not supported");
static_assert(sizeof(ImageExtTable) == current_table_size_t::image_ext,
              "HSA image-extended API table size changed or version not supported");
static_assert(sizeof(AmdExtTable) == current_table_size_t::amd_ext,
              "HSA amd-extended API table size changed or version not supported");
static_assert(sizeof(CoreApiTable) == current_table_size_t::core_api_ext,
              "HSA core API table size changed or version not supported");
}  // namespace hsa
}  // namespace rocprofiler
#        else
namespace rocprofiler
{
namespace hsa
{
static_assert(HSA_CORE_API_TABLE_MAJOR_VERSION == 0x02,
              "Change in the major version of HSA core API table");
static_assert(HSA_AMD_EXT_API_TABLE_MAJOR_VERSION == 0x02,
              "Change in the major version of HSA amd-extended API table");
static_assert(HSA_FINALIZER_API_TABLE_MAJOR_VERSION == 0x02,
              "Change in the major version of HSA finalizer API table");
static_assert(HSA_IMAGE_API_TABLE_MAJOR_VERSION == 0x02,
              "Change in the major version of HSA image API table");

static_assert(HSA_CORE_API_TABLE_STEP_VERSION == 0x00,
              "Change in the major version of HSA core API table");
static_assert(HSA_AMD_EXT_API_TABLE_STEP_VERSION == 0x00,
              "Change in the major version of HSA amd-extended API table");
static_assert(HSA_FINALIZER_API_TABLE_STEP_VERSION == 0x00,
              "Change in the major version of HSA finalizer API table");
static_assert(HSA_IMAGE_API_TABLE_STEP_VERSION == 0x00,
              "Change in the major version of HSA image API table");
static_assert(HSA_AQLPROFILE_API_TABLE_STEP_VERSION == 0x00,
              "Change in the major version of HSA aqlprofile API table");

// this should always be updated to latest table size
template <size_t VersionCode>
struct table_size;

// latest version of hsa runtime that has been updated for support by rocprofiler
// and the current version of hsa runtime during this compilation
constexpr size_t latest_version  = ROCPROFILER_COMPUTE_VERSION(1, 12, 0);
constexpr size_t current_version = ROCPROFILER_HSA_RUNTIME_VERSION;

// aliases to the template specializations providing the table size info
using current_table_size_t = table_size<current_version>;
using latest_table_size_t  = table_size<latest_version>;

// specialization for v1.12
template <>
struct table_size<ROCPROFILER_COMPUTE_VERSION(1, 12, 0)>
{
    static constexpr size_t finalizer_ext = 64;
    static constexpr size_t image_ext     = 120;
    static constexpr size_t amd_ext       = 552;
    static constexpr size_t core_api_ext  = 1016;
};

// default static asserts to check against latest version
// e.g. v1.12 might have the same table sizes as v1.11 so
// we don't want to fail to compile if nothing has changed
template <size_t VersionCode>
struct table_size : latest_table_size_t
{};

// if you hit these static asserts, that means HSA added entries to the table but did not update the
// step numbers
static_assert(sizeof(FinalizerExtTable) == current_table_size_t::finalizer_ext,
              "HSA finalizer API table size changed or version not supported");
static_assert(sizeof(ImageExtTable) == current_table_size_t::image_ext,
              "HSA image-extended API table size changed or version not supported");
static_assert(sizeof(AmdExtTable) == current_table_size_t::amd_ext,
              "HSA amd-extended API table size changed or version not supported");
static_assert(sizeof(CoreApiTable) == current_table_size_t::core_api_ext,
              "HSA core API table size changed or version not supported");
}  // namespace hsa
}  // namespace rocprofiler
#        endif
#    endif
#endif
