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

#include "lib/rocprofiler-sdk/code_object/hsa/code_object.hpp"

#include <hsa/hsa.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hsa.h>

#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ven_amd_loader.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <string_view>
#include <vector>

namespace rocprofiler
{
namespace code_object
{
namespace hsa
{
code_object::code_object(code_object&& rhs) noexcept { operator=(std::move(rhs)); }

code_object&
code_object::operator=(code_object&& rhs) noexcept
{
    if(this != &rhs)
    {
        beg_notified    = rhs.beg_notified;
        end_notified    = rhs.end_notified;
        uri             = rhs.uri;
        hsa_executable  = rhs.hsa_executable;
        hsa_code_object = rhs.hsa_code_object;
        rocp_data       = rhs.rocp_data;
        user_data       = std::move(rhs.user_data);
        rocp_data.uri   = (uri) ? uri->c_str() : nullptr;
        symbols         = std::move(rhs.symbols);
    }

    return *this;
}

bool
operator==(const code_object& lhs, const code_object& rhs)
{
    return std::tie(lhs.hsa_executable.handle, lhs.hsa_code_object.handle) ==
           std::tie(rhs.hsa_executable.handle, rhs.hsa_code_object.handle);
}
}  // namespace hsa
}  // namespace code_object
}  // namespace rocprofiler
