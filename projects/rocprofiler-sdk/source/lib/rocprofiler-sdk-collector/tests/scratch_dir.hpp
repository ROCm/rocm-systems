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

#include <gtest/gtest.h>

#include <stdlib.h>

#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace rocprofiler
{
namespace collector
{
namespace test
{
// A fresh, empty directory for one test; removed when the test ends.
struct scratch_dir
{
    std::filesystem::path path = {};

    scratch_dir()
    {
        auto templ = (std::filesystem::temp_directory_path() / "collector-test-XXXXXX").string();
        auto buf   = std::vector<char>{templ.begin(), templ.end()};
        buf.emplace_back('\0');
        EXPECT_NE(::mkdtemp(buf.data()), nullptr);
        path = buf.data();
    }

    ~scratch_dir()
    {
        auto ec = std::error_code{};
        std::filesystem::remove_all(path, ec);
    }

    scratch_dir(const scratch_dir&) = delete;
    scratch_dir(scratch_dir&&)      = delete;
    scratch_dir& operator=(const scratch_dir&) = delete;
    scratch_dir& operator=(scratch_dir&&) = delete;
};
}  // namespace test
}  // namespace collector
}  // namespace rocprofiler
