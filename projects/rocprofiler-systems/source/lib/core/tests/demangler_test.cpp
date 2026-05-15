// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "gtest/gtest.h"

#include "core/demangler.hpp"

#include <cstring>
#include <string>

struct mock_demangler_impl
{
    static inline int call_count = 0;

    static char* demangle(const char* _mangled_name, char*, size_t*, int* _status)
    {
        ++call_count;

        if(std::string(_mangled_name).find("FAIL") != std::string::npos)
        {
            *_status = -1;
            return nullptr;
        }

        std::string result = "demangled_" + std::string(_mangled_name);
        char*       output = static_cast<char*>(std::malloc(result.size() + 1));
        memcpy(output, result.c_str(), result.size() + 1);
        *_status = 0;
        return output;
    }
};

class demangler_test : public ::testing::Test
{
protected:
    void SetUp() override { mock_demangler_impl::call_count = 0; }
};

TEST_F(demangler_test, successful_demangle)
{
    rocprofsys::utility::demangler<mock_demangler_impl> demangler;
    std::string result = demangler.demangle("mangled");

    EXPECT_EQ(result, "demangled_mangled");
    EXPECT_EQ(mock_demangler_impl::call_count, 1);
}

TEST_F(demangler_test, failed_demangle_returns_original)
{
    rocprofsys::utility::demangler<mock_demangler_impl> demangler;
    std::string result = demangler.demangle("mangled_FAIL");

    EXPECT_EQ(result, "mangled_FAIL");
    EXPECT_EQ(mock_demangler_impl::call_count, 1);
}

TEST_F(demangler_test, empty_string_returns_empty)
{
    rocprofsys::utility::demangler<mock_demangler_impl> demangler;
    std::string                                         result = demangler.demangle("");

    EXPECT_TRUE(result.empty());
    EXPECT_EQ(mock_demangler_impl::call_count, 0);
}

TEST_F(demangler_test, caching_prevents_redundant_calls)
{
    rocprofsys::utility::demangler<mock_demangler_impl> demangler;

    std::string first  = demangler.demangle("mangled");
    std::string second = demangler.demangle("mangled");
    std::string third  = demangler.demangle("mangled");

    EXPECT_EQ(first, "demangled_mangled");
    EXPECT_EQ(first, second);
    EXPECT_EQ(second, third);
    EXPECT_EQ(mock_demangler_impl::call_count, 1);
}

TEST_F(demangler_test, different_inputs_cached_separately)
{
    rocprofsys::utility::demangler<mock_demangler_impl> demangler;

    std::string result1       = demangler.demangle("mangled1");
    std::string result2       = demangler.demangle("mangled2");
    std::string result1_again = demangler.demangle("mangled1");

    EXPECT_EQ(result1, "demangled_mangled1");
    EXPECT_EQ(result2, "demangled_mangled2");
    EXPECT_EQ(result1, result1_again);
    EXPECT_EQ(mock_demangler_impl::call_count, 2);
}

TEST_F(demangler_test, string_view_interface)
{
    rocprofsys::utility::demangler<mock_demangler_impl> demangler;
    std::string                                         str = "mangled";
    std::string_view                                    sv{ str };

    std::string result = demangler.demangle(sv);

    EXPECT_EQ(result, "demangled_mangled");
    EXPECT_EQ(mock_demangler_impl::call_count, 1);
}

TEST_F(demangler_test, template_demangle_interface)
{
    rocprofsys::utility::demangler<mock_demangler_impl> demangler;
    std::string result = demangler.demangle<int>();

    EXPECT_NE(result.find("demangled_"), std::string::npos);
    EXPECT_FALSE(result.empty());
}

TEST_F(demangler_test, cache_handles_failed_demangle)
{
    rocprofsys::utility::demangler<mock_demangler_impl> demangler;

    std::string first  = demangler.demangle("FAIL");
    std::string second = demangler.demangle("FAIL");

    EXPECT_EQ(first, "FAIL");
    EXPECT_EQ(first, second);
    EXPECT_EQ(mock_demangler_impl::call_count, 1);
}

TEST_F(demangler_test, multiple_instances_independent_caches)
{
    rocprofsys::utility::demangler<mock_demangler_impl> demangler1;
    rocprofsys::utility::demangler<mock_demangler_impl> demangler2;

    demangler1.demangle("mangled");
    demangler2.demangle("mangled");

    EXPECT_EQ(mock_demangler_impl::call_count, 2);
}
