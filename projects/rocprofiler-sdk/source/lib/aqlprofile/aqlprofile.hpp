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

/**
 * @file source/lib/aqlprofile/aqlprofile.hpp
 *
 * @brief Handles the AQLProfile interface for when using internal vs. external AQLProfile
 */

#if !defined(ROCPROFILER_EXTERNAL_AQLPROFILE)
#    define ROCPROFILER_EXTERNAL_AQLPROFILE 0
#endif

#define ROCPROFILER_INTERNAL_AQLPROFILE_INCLUDE 1
#include "lib/aqlprofile/aql_profile_v2.h"
#undef ROCPROFILER_INTERNAL_AQLPROFILE_INCLUDE

#if ROCPROFILER_EXTERNAL_AQLPROFILE == 0
#    include "lib/aqlprofile/util/hsa_rsrc_factory.h"
#else

extern "C" struct HsaApiTable;

namespace rocprofiler
{
namespace aqlprofile
{
template <typename Tp>
inline void
hsa_rsrc_factory_init(Tp* /*hsa_api_table*/)
{
    // External AQLProfile library - no internal initialization
}
}  // namespace aqlprofile
}  // namespace rocprofiler

#endif
