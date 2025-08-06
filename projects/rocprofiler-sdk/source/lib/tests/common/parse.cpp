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

#include <rocprofiler-sdk/cxx/details/tokenize.hpp>

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>

namespace sdk = ::rocprofiler::sdk;

TEST(parse, tokenize_characters)
{
    auto tokenized = sdk::parse::tokenize(std::string_view{"0, GPU-DEADBEEFDEADBEEF,, -1"}, ", ");

    ASSERT_EQ(tokenized.size(), 3)
        << fmt::format("TOKENIZED: '{}'", fmt::join(tokenized.begin(), tokenized.end(), "' | '"));
    EXPECT_EQ(tokenized.at(0), "0");
    EXPECT_EQ(tokenized.at(1), "GPU-DEADBEEFDEADBEEF");
    EXPECT_EQ(tokenized.at(2), "-1");
}

TEST(parse, tokenize_strings)
{
    auto tokenized = sdk::parse::tokenize(std::string_view{"0, GPU-DEADBEEFDEADBEEF, -1, 8"},
                                          std::vector<std::string_view>{", "});

    ASSERT_EQ(tokenized.size(), 4)
        << fmt::format("TOKENIZED: '{}'", fmt::join(tokenized.begin(), tokenized.end(), "' | '"));
    EXPECT_EQ(tokenized.at(0), "0");
    EXPECT_EQ(tokenized.at(1), "GPU-DEADBEEFDEADBEEF");
    EXPECT_EQ(tokenized.at(2), "-1");
    EXPECT_EQ(tokenized.at(3), "8");
}

TEST(parse, strip)
{
    auto test_message     = std::string{" \t\v\f TEST MESSAGE\n\r\n "};
    auto stripped_message = sdk::parse::strip(std::string{test_message}, " \t\n\v\f\r");
    auto expected_message = std::string{"TEST MESSAGE"};

    EXPECT_EQ(stripped_message, expected_message)
        << fmt::format("Expected '{}' after stripping '{}'\nResult: '{}'",
                       expected_message,
                       test_message,
                       stripped_message);
}
