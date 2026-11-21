// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/common/scope_destructor.hpp"
#include "lib/rocprofiler-sdk/thread_trace/shared_trace_resources.hpp"

#include <gtest/gtest.h>
#include <cstdlib>
#include <optional>
#include <string>

namespace rocprofiler
{
namespace thread_trace
{
namespace test
{
inline common::scope_destructor
force_hsa_backend()
{
    const auto* previous = ::getenv("ROCPROFILER_SQTT_FORCE_HSA");
    auto saved = previous ? std::optional<std::string>{previous} : std::optional<std::string>{};
    ::setenv("ROCPROFILER_SQTT_FORCE_HSA", "1", 1);
    return common::scope_destructor{[saved]() {
        if(saved)
            ::setenv("ROCPROFILER_SQTT_FORCE_HSA", saved->c_str(), 1);
        else
            ::unsetenv("ROCPROFILER_SQTT_FORCE_HSA");
    }};
}

inline void
register_resource_cleanup()
{
    class cleanup_listener : public ::testing::EmptyTestEventListener
    {
        void OnTestEnd(const ::testing::TestInfo&) override { free_shared_trace_resources(); }
    };
    static const bool registered = []() {
        ::testing::UnitTest::GetInstance()->listeners().Append(new cleanup_listener{});
        return true;
    }();
    (void) registered;
}
}  // namespace test
}  // namespace thread_trace
}  // namespace rocprofiler
