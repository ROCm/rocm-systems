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

#pragma once

#include <rocprofiler-sdk/hsa.h>

namespace rocprofiler
{
namespace hsa
{
// pair of hsa signal and user data pointer for async handler
struct signal_t
{
    hsa_signal_t value = {.handle = 0};

    // signal_pool_fini() destroys the HSA signal but deliberately preserves this handle. Retired
    // pool storage can still be referenced by an async packet, and keeping the identity lets its
    // completion path recognize the pooled signal instead of destroying the same handle twice.
    bool matches(hsa_signal_t signal) const noexcept
    {
        return signal.handle != 0 && signal.handle == value.handle;
    }
};
}  // namespace hsa
}  // namespace rocprofiler
