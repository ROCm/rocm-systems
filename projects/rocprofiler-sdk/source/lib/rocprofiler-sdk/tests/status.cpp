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

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <gtest/gtest.h>

#include <string_view>

TEST(rocprofiler_lib, error_string)
{
    for(size_t i = 0; i < static_cast<size_t>(ROCPROFILER_STATUS_LAST); ++i)
    {
        auto        status  = static_cast<rocprofiler_status_t>(i);
        const auto* name    = rocprofiler_get_status_name(status);
        const auto* message = rocprofiler_get_status_string(status);

        ASSERT_NE(name, nullptr) << "idx=" << i;
        ASSERT_NE(message, nullptr) << name << " (idx=" << i << ")";

        std::cout << std::setw(60) << name << " :: " << message << "\n";

        if(i == ROCPROFILER_STATUS_SUCCESS)
        {
            EXPECT_EQ(std::string_view{message}, std::string_view{"Success"});
        }
        else
        {
            EXPECT_GE(std::string_view{message}.length(), 8)
                << "status message for " << name << " (idx=" << i << ") is too short";
        }
    }
}
