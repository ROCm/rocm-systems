// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/common/string_entry.hpp"

#include <gtest/gtest.h>

#include <string>
#include <string_view>

namespace
{
namespace common = ::rocprofiler::common;
}  // namespace

// A null const char* must not be dereferenced; it is interned as the "(null)"
// sentinel so a buggy application passing nullptr stays visible in the trace
// rather than being masked as an empty string.
TEST(common, string_entry_null_pointer)
{
    const char* _null = nullptr;

    const auto* _entry = common::get_string_entry(_null);
    ASSERT_NE(_entry, nullptr);
    EXPECT_EQ(*_entry, "(null)");
}

// The null sentinel must remain distinct from a genuine empty string so the two
// cases can be told apart in trace output.
TEST(common, string_entry_null_distinct_from_empty)
{
    const auto* _null_entry  = common::get_string_entry(static_cast<const char*>(nullptr));
    const auto* _empty_entry = common::get_string_entry("");

    ASSERT_NE(_null_entry, nullptr);
    ASSERT_NE(_empty_entry, nullptr);
    EXPECT_EQ(*_null_entry, "(null)");
    EXPECT_EQ(*_empty_entry, "");
    EXPECT_NE(_null_entry, _empty_entry);
}

// Interning copies the argument into stable storage owned by the cache. This is
// the property the HIP tracer relies on: a const char* argument captured from a
// caller's temporary buffer must survive after that buffer is destroyed, since
// the buffered record is stringified later on the callback thread.
TEST(common, string_entry_outlives_source_buffer)
{
    const std::string* _entry = nullptr;
    {
        auto _tmp = std::string{"transient-kernel-name"};
        _entry    = common::get_string_entry(_tmp.c_str());
        ASSERT_NE(_entry, nullptr);
        EXPECT_EQ(*_entry, "transient-kernel-name");
    }

    // The source std::string is now destroyed; the interned copy must still be
    // valid and unchanged.
    EXPECT_EQ(*_entry, "transient-kernel-name");
}

// Identical contents intern to the same cache entry regardless of whether the
// lookup arrives as a const char* or a std::string_view.
TEST(common, string_entry_dedup)
{
    const auto* _from_cstr = common::get_string_entry("shared-entry");
    const auto* _from_view = common::get_string_entry(std::string_view{"shared-entry"});

    ASSERT_NE(_from_cstr, nullptr);
    ASSERT_NE(_from_view, nullptr);
    EXPECT_EQ(_from_cstr, _from_view);
}
