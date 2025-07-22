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

#include <rocprofiler-sdk/ext_version.h>

#include "lib/common/defines.hpp"

#include <cstddef>

namespace rocprofiler
{
namespace common
{
namespace abi
{
constexpr auto
compute_table_offset(size_t num_funcs)
{
    return (num_funcs * sizeof(void*)) + sizeof(size_t);
}
}  // namespace abi
}  // namespace common
}  // namespace rocprofiler

// ROCP_SDK_ENFORCE_ABI_VERSIONING will cause a compiler error if the size of the API table
// changed (most likely due to addition of new dispatch table entry) to make sure the developer is
// reminded to update the table versioning value before changing the value in
// ROCP_SDK_ENFORCE_ABI_VERSIONING to make this static assert pass.
//
// ROCP_SDK_ENFORCE_ABI will cause a compiler error if the order of the members in the API table
// change. Do not reorder member variables and change existing ROCP_SDK_ENFORCE_ABI values --
// always
//
// Please note: rocprofiler will do very strict compile time checks to make
// sure these versioning values are appropriately updated -- so commenting out this check, only
// updating the size field in ROCP_SDK_ENFORCE_ABI_VERSIONING, etc. will result in the
// rocprofiler-sdk failing to build and you will be forced to do the work anyway.
#if !defined(ROCPROFILER_UNSAFE_NO_VERSION_CHECK)
#    define ROCP_SDK_ENFORCE_ABI_VERSIONING(TABLE, NUM)                                            \
        static_assert(                                                                             \
            sizeof(TABLE) == ::rocprofiler::common::abi::compute_table_offset(NUM),                \
            "size of the API table struct has changed. Update the STEP_VERSION number (or "        \
            "in rare cases, the MAJOR_VERSION number)");

#    define ROCP_SDK_ENFORCE_ABI(TABLE, ENTRY, NUM)                                                \
        static_assert(                                                                             \
            offsetof(TABLE, ENTRY) == ::rocprofiler::common::abi::compute_table_offset(NUM),       \
            "ABI break for " #TABLE "." #ENTRY                                                     \
            ". Only add new function pointers to end of struct and do not rearrange them");
#else
#    define ROCP_SDK_ENFORCE_ABI_VERSIONING(TABLE, NUM)
#    define ROCP_SDK_ENFORCE_ABI(TABLE, ENTRY, NUM)
#endif

// These are guarded by ROCPROFILER_CI=1
#if !defined(ROCPROFILER_UNSAFE_NO_VERSION_CHECK) && (defined(ROCPROFILER_CI) && ROCPROFILER_CI > 0)
#    define INTERNAL_CI_ROCP_SDK_ENFORCE_ABI_VERSIONING(TABLE, NUM)                                \
        static_assert(                                                                             \
            sizeof(TABLE) == ::rocprofiler::common::abi::compute_table_offset(NUM),                \
            "size of the API table struct has changed. Update the STEP_VERSION number (or "        \
            "in rare cases, the MAJOR_VERSION number)");

#    define INTERNAL_CI_ROCP_SDK_ENFORCE_ABI(TABLE, ENTRY, NUM)                                    \
        static_assert(                                                                             \
            offsetof(TABLE, ENTRY) == ::rocprofiler::common::abi::compute_table_offset(NUM),       \
            "ABI break for " #TABLE "." #ENTRY                                                     \
            ". Only add new function pointers to end of struct and do not rearrange them");
#else
#    define INTERNAL_CI_ROCP_SDK_ENFORCE_ABI_VERSIONING(TABLE, NUM)
#    define INTERNAL_CI_ROCP_SDK_ENFORCE_ABI(TABLE, ENTRY, NUM)
#endif
