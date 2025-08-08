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

#include <rocprofiler-sdk/ext_version.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/version.h>

#include "lib/common/utility.hpp"

#include <gtest/gtest.h>

#include <tuple>

TEST(rocprofiler_lib, version)
{
    auto correct_version = std::tuple<uint32_t, uint32_t, uint32_t>(
        ROCPROFILER_VERSION_MAJOR, ROCPROFILER_VERSION_MINOR, ROCPROFILER_VERSION_PATCH);
    auto query_version      = std::tuple<uint32_t, uint32_t, uint32_t>(0, 0, 0);
    auto query_version_copy = std::tuple<uint32_t, uint32_t, uint32_t>(0, 0, 0);

    auto ret0 = rocprofiler_get_version(&std::get<0>(query_version), nullptr, nullptr);
    auto ret1 = rocprofiler_get_version(nullptr, &std::get<1>(query_version), nullptr);
    auto ret2 = rocprofiler_get_version(nullptr, nullptr, &std::get<2>(query_version));

    EXPECT_EQ(ret0, ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(ret1, ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(ret2, ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(query_version, correct_version);

    auto reta = rocprofiler_get_version(&std::get<0>(query_version_copy),
                                        &std::get<1>(query_version_copy),
                                        &std::get<2>(query_version_copy));
    EXPECT_EQ(reta, ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(query_version_copy, correct_version);
    EXPECT_EQ(query_version_copy, query_version);
}
