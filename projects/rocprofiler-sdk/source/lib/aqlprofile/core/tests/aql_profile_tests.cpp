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
#include "lib/aqlprofile/aqlprofile.hpp"

#include <gtest/gtest.h>

TEST(aqlprofile, version)
{
    auto correct_version = aqlprofile_version_t{.major = AQLPROFILE_VERSION_MAJOR,
                                                .minor = AQLPROFILE_VERSION_MINOR,
                                                .patch = AQLPROFILE_VERSION_PATCH};

    auto query_version = aqlprofile_version_t{};

    auto ret = aqlprofile_get_version(&query_version);

    EXPECT_EQ(ret, HSA_STATUS_SUCCESS);
    EXPECT_EQ(query_version.major, correct_version.major);
    EXPECT_EQ(query_version.minor, correct_version.minor);
    EXPECT_EQ(query_version.patch, correct_version.patch);

    EXPECT_EQ(aqlprofile_get_version(nullptr), HSA_STATUS_ERROR_INVALID_ARGUMENT);
}
