// Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// with the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
//
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimers in the
// documentation and/or other materials provided with the distribution.
//
// * Neither the names of Advanced Micro Devices, Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this Software without specific prior written permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.

#pragma once

#include <string_view>

namespace rocprofsys
{
namespace benchmark
{
enum class category
{
    kernel_dispatch,
    db_entry_kernel_dispatch,
    memory_copy,
    db_entry_memory_copy,
    memory_allocate,
    db_entry_memory_allocate,
    perfetto_kernel_dispatch,
    sdk_tool_buffered_tracing,
    count
};

constexpr std::string_view
to_string(category cat)
{
    switch(cat)
    {
        case category::kernel_dispatch: return "kernel_dispatch";
        case category::db_entry_kernel_dispatch: return "db_entry_kernel_dispatch";
        case category::memory_copy: return "memory_copy";
        case category::memory_allocate: return "memory_allocate";
        case category::db_entry_memory_copy: return "db_entry_memory_copy";
        case category::db_entry_memory_allocate: return "db_entry_memory_allocate";
        case category::perfetto_kernel_dispatch: return "perfetto_kernel_dispatch";
        case category::sdk_tool_buffered_tracing: return "sdk_tool_buffered_tracing";
        default: return "unknown";
    }
}

}  // namespace benchmark
}  // namespace rocprofsys
