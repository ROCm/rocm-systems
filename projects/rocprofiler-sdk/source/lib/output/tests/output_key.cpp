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

#include "lib/output/output_key.hpp"
#include "lib/common/environment.hpp"
#include "lib/common/logging.hpp"
#include "lib/output/format_path.hpp"

#include <gtest/gtest.h>

#include <string_view>
#include <utility>

namespace common = ::rocprofiler::common;
namespace fs     = ::rocprofiler::common::filesystem;

TEST(output, output_key_list)
{
    common::init_logging("TEST");

    auto output_keys = rocprofiler::tool::output_keys("test_tag");

    std::sort(output_keys.begin(), output_keys.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.key < rhs.key;
    });

    for(const auto& itr : output_keys)
    {
        ROCP_INFO << fmt::format("Key: {}, Value: {}, Description: {}, Is Multi-process Stable: {}",
                                 itr.key,
                                 itr.value,
                                 itr.description,
                                 itr.is_multiprocess_stable);

        EXPECT_FALSE(itr.key.empty());
        EXPECT_FALSE(itr.value.empty());
        EXPECT_FALSE(itr.description.empty());

        EXPECT_EQ(itr.key.find_first_of("{%"), 0)
            << fmt::format("Key '{}' does not start with '{}' or '%'", '{', itr.key);
    }
}

TEST(output, output_key_resolution)
{
    common::init_logging("TEST");

    auto verify = [](std::string inp, std::string_view expected) {
        ROCP_INFO << "";
        auto _ref   = inp;
        auto result = rocprofiler::tool::format_path(std::move(inp));
        EXPECT_EQ(result, expected)
            << fmt::format("[verify] Input: {}, Expected: {}, Got: {}", _ref, expected, result);
    };

    auto verify_mp = [](std::string inp, std::string_view expected) {
        ROCP_INFO << "";
        auto _ref   = inp;
        auto result = rocprofiler::tool::format_mp_stable_path(std::move(inp));
        EXPECT_EQ(result, expected)
            << fmt::format("[verify_mp] Input: {}, Expected: {}, Got: {}", _ref, expected, result);
    };

    verify("{pid}_output_{ppid}", fmt::format("{}_output_{}", getpid(), getppid()));

    verify("{ppid}_output_{pid}", fmt::format("{}_output_{}", getppid(), getpid()));

    verify("{pwd}/rocprofv3-output/$env{USER}.{pid}",
           fmt::format("{}/rocprofv3-output/{}.{}",
                       fs::canonical(common::get_env("PWD", fs::current_path().string())).string(),
                       common::get_env("USER", getlogin()),
                       getpid()));

    verify_mp("{pid}_output_{ppid}", "output");

    verify_mp("{ppid}_output_{pid}", "output");

    verify_mp("{pwd}/rocprofv3-output/$env{USER}.{pid}",
              fmt::format("{}/rocprofv3-output/{}",
                          common::get_env("PWD", fs::current_path().string()),
                          common::get_env("USER", getlogin())));

    verify_mp("hostname/{ppid}/{rank}.{pid}.{nid}.{launch_time}/$env{USER}",
              fmt::format("hostname/{}", common::get_env("USER", getlogin())));
}
