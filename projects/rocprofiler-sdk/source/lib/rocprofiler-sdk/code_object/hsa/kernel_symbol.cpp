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

#include "lib/rocprofiler-sdk/code_object/hsa/kernel_symbol.hpp"

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
kernel_symbol::kernel_symbol(kernel_symbol&& rhs) noexcept { operator=(std::move(rhs)); }

kernel_symbol&
kernel_symbol::operator=(kernel_symbol&& rhs) noexcept
{
    if(this != &rhs)
    {
        beg_notified          = rhs.beg_notified;
        end_notified          = rhs.end_notified;
        name                  = rhs.name;
        hsa_executable        = rhs.hsa_executable;
        hsa_agent             = rhs.hsa_agent;
        hsa_symbol            = rhs.hsa_symbol;
        rocp_data             = rhs.rocp_data;
        user_data             = std::move(rhs.user_data);
        rocp_data.kernel_name = (name) ? name->c_str() : nullptr;
    }

    return *this;
}

bool
operator==(const kernel_symbol& lhs, const kernel_symbol& rhs)
{
    return std::tie(lhs.hsa_executable.handle, lhs.hsa_agent.handle, lhs.hsa_symbol.handle) ==
           std::tie(rhs.hsa_executable.handle, rhs.hsa_agent.handle, rhs.hsa_symbol.handle);
}
}  // namespace hsa
}  // namespace code_object
}  // namespace rocprofiler
