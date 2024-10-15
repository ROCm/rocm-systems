// MIT License
//
// Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "core/defines.hpp"

#if defined(ROCPROFSYS_USE_ROCPROFILER) && ROCPROFSYS_USE_ROCPROFILER > 0
#    include <rocprofiler.h>
#endif

#include <cstdint>
#include <mutex>

namespace rocprofsys
{
namespace rocm
{
using lock_t = std::unique_lock<std::mutex>;

extern std::mutex rocm_mutex;
extern bool       is_loaded;
}  // namespace rocm
}  // namespace rocprofsys

extern "C"
{
    struct HsaApiTable;
    using on_load_t = bool (*)(HsaApiTable*, uint64_t, uint64_t, const char* const*);

    bool OnLoad(HsaApiTable* table, uint64_t runtime_version, uint64_t failed_tool_count,
                const char* const* failed_tool_names) ROCPROFSYS_PUBLIC_API;
    void OnUnload() ROCPROFSYS_PUBLIC_API;

#if defined(ROCPROFSYS_USE_ROCPROFILER) && ROCPROFSYS_USE_ROCPROFILER > 0
    void OnLoadToolProp(rocprofiler_settings_t* settings) ROCPROFSYS_PUBLIC_API;
    void OnUnloadTool() ROCPROFSYS_PUBLIC_API;
#endif
}
