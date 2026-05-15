// Copyright (c) 2024 Advanced Micro Devices, Inc.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "details/platform/pe_parser.hpp"

#include <gtest/gtest.h>

namespace
{
// A code address inside this test executable. Take address of a function so
// it lands in the .text section, which is executable on every platform.
void
sample_function()
{
}
}  // namespace

#if defined(_WIN32)
TEST(pe_parser, get_memory_regions_returns_nonempty_on_windows)
{
    using namespace rocprofiler_register::platform;
    auto regions = get_memory_regions();
    EXPECT_GT(regions.size(), 0u);
}

TEST(pe_parser, code_address_is_executable_on_windows)
{
    using namespace rocprofiler_register::platform;
    const auto* address = reinterpret_cast<const void*>(&sample_function);
    EXPECT_TRUE(is_address_executable(address));
    EXPECT_TRUE(is_address_readable(address));
}

TEST(pe_parser, null_address_is_not_executable_on_windows)
{
    using namespace rocprofiler_register::platform;
    EXPECT_FALSE(is_address_executable(nullptr));
    EXPECT_FALSE(is_address_readable(nullptr));
    EXPECT_EQ(get_memory_protection(nullptr), 0u);
}

TEST(pe_parser, find_module_returns_nonnull_for_code_address_on_windows)
{
    using namespace rocprofiler_register::platform;
    const auto* address = reinterpret_cast<const void*>(&sample_function);
    auto* module  = find_module_for_address(address);
    EXPECT_NE(module, nullptr);
}

TEST(pe_parser, get_memory_protection_nonzero_for_code_on_windows)
{
    using namespace rocprofiler_register::platform;
    const auto* address = reinterpret_cast<const void*>(&sample_function);
    EXPECT_NE(get_memory_protection(address), 0u);
}
#else
// On Linux the pe_parser interface is a stub; verify it returns the
// documented conservative answers and does not crash.
TEST(pe_parser, returns_stub_results_on_linux)
{
    using namespace rocprofiler_register::platform;
    const auto* address = reinterpret_cast<const void*>(&sample_function);
    EXPECT_TRUE(get_memory_regions().empty());
    EXPECT_EQ(get_memory_protection(address), 0u);
    EXPECT_FALSE(is_address_executable(address));
    EXPECT_FALSE(is_address_readable(address));
    EXPECT_EQ(find_module_for_address(address), nullptr);
}
#endif
