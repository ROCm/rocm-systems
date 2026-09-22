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

#include "lib/common/dl.hpp"
#include "lib/common/environment.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <cxxabi.h>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>

TEST(common, dl_get_linked_path)
{
    rocprofiler::common::set_env("ROCPROFILER_LINKED_PATH_USE_DLOPEN", "0", 1);

    auto real_lib  = std::string_view{"libstdc++.so"};
    auto real_path = rocprofiler::common::dl::get_linked_path(real_lib);
    ASSERT_TRUE(real_path.has_value());
    EXPECT_FALSE(real_path->empty());
    fmt::print("{} linked path: {}\n", real_lib, real_path.value());

    auto fake_lib  = std::string_view{"libthis_library_does_not_exist.so"};
    auto fake_path = rocprofiler::common::dl::get_linked_path(fake_lib);
    EXPECT_FALSE(fake_path.has_value())
        << fmt::format("fake_path: {}", fake_path.value_or(std::string{}));
    if(fake_path)
    {
        EXPECT_TRUE(fake_path->empty())
            << fmt::format("fake_path: {}", fake_path.value_or(std::string{}));
        fmt::print("{} linked path: {}\n", fake_lib, fake_path.value());
    }
}

TEST(common, dl_get_symbol_path)
{
    rocprofiler::common::set_env("ROCPROFILER_SYMBOL_PATH_USE_DLOPEN", "0", 1);

    auto real_lib  = std::string_view{"libstdc++.so"};
    auto real_path = rocprofiler::common::dl::get_symbol_path(
        {std::string{real_lib}, fmt::format("{}.6", real_lib)},
        "__cxa_demangle",
        reinterpret_cast<const void*>(abi::__cxa_demangle));
    ASSERT_TRUE(real_path.has_value());
    EXPECT_FALSE(real_path->empty());
    fmt::print("{} symbol path: {}\n", "__cxa_demangle", real_path.value());

    auto fake_lib = std::string_view{"libthis_library_does_not_exist.so"};
    auto fake_sym = std::string_view{"this_symbol_does_not_exist"};
    auto fake_path =
        rocprofiler::common::dl::get_symbol_path({std::string{fake_lib}}, fake_sym, nullptr);
    EXPECT_FALSE(fake_path.has_value())
        << fmt::format("fake_path: {}", fake_path.value_or(std::string{}));
    if(fake_path)
    {
        EXPECT_TRUE(fake_path->empty())
            << fmt::format("fake_path: {}", fake_path.value_or(std::string{}));
        fmt::print("{} symbol path: {}\n", fake_sym, fake_path.value());
    }
}

TEST(common, dl_get_linked_path_use_dlopen)
{
    rocprofiler::common::set_env("ROCPROFILER_LINKED_PATH_USE_DLOPEN", "1", 1);

    auto real_lib  = std::string_view{"libstdc++.so"};
    auto real_path = rocprofiler::common::dl::get_linked_path(real_lib);
    ASSERT_TRUE(real_path.has_value());
    EXPECT_FALSE(real_path->empty());
    fmt::print("{} linked path: {}\n", real_lib, real_path.value());

    auto fake_lib  = std::string_view{"libthis_library_does_not_exist.so"};
    auto fake_path = rocprofiler::common::dl::get_linked_path(fake_lib);
    EXPECT_FALSE(fake_path.has_value())
        << fmt::format("fake_path: {}", fake_path.value_or(std::string{}));
    if(fake_path)
    {
        EXPECT_TRUE(fake_path->empty())
            << fmt::format("fake_path: {}", fake_path.value_or(std::string{}));
        fmt::print("{} linked path: {}\n", fake_lib, fake_path.value());
    }

    rocprofiler::common::set_env("ROCPROFILER_LINKED_PATH_USE_DLOPEN", "0", 1);
}

TEST(common, dl_get_symbol_path_use_dlopen)
{
    rocprofiler::common::set_env("ROCPROFILER_SYMBOL_PATH_USE_DLOPEN", "1", 1);

    auto real_lib  = std::string_view{"libstdc++.so"};
    auto real_path = rocprofiler::common::dl::get_symbol_path(
        {std::string{real_lib}, fmt::format("{}.6", real_lib)},
        "__cxa_demangle",
        reinterpret_cast<const void*>(abi::__cxa_demangle));
    ASSERT_TRUE(real_path.has_value());
    EXPECT_FALSE(real_path->empty());
    fmt::print("{} symbol path: {}\n", "__cxa_demangle", real_path.value());

    auto fake_lib = std::string_view{"libthis_library_does_not_exist.so"};
    auto fake_sym = std::string_view{"this_symbol_does_not_exist"};
    auto fake_path =
        rocprofiler::common::dl::get_symbol_path({std::string{fake_lib}}, fake_sym, nullptr);
    EXPECT_FALSE(fake_path.has_value())
        << fmt::format("fake_path: {}", fake_path.value_or(std::string{}));
    if(fake_path)
    {
        EXPECT_TRUE(fake_path->empty())
            << fmt::format("fake_path: {}", fake_path.value_or(std::string{}));
        fmt::print("{} symbol path: {}\n", fake_sym, fake_path.value());
    }

    rocprofiler::common::set_env("ROCPROFILER_SYMBOL_PATH_USE_DLOPEN", "0", 1);
}
