// MIT License
//
// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <gtest/gtest.h>
#include <string>

#include "trace_decoder_api.h"

TEST(ApiStringTest, AllInfoStrings)
{
    EXPECT_STREQ(rocprof_trace_decoder_get_info_string(ROCPROFILER_THREAD_TRACE_DECODER_INFO_NONE), "NONE");
    EXPECT_NE(
        std::string(rocprof_trace_decoder_get_info_string(ROCPROFILER_THREAD_TRACE_DECODER_INFO_DATA_LOST))
            .find("Data Lost"),
        std::string::npos
    );
    EXPECT_NE(
        std::string(rocprof_trace_decoder_get_info_string(ROCPROFILER_THREAD_TRACE_DECODER_INFO_STITCH_INCOMPLETE))
            .find("Stitch"),
        std::string::npos
    );
    EXPECT_NE(
        std::string(rocprof_trace_decoder_get_info_string(ROCPROFILER_THREAD_TRACE_DECODER_INFO_WAVE_INCOMPLETE))
            .find("incomplete"),
        std::string::npos
    );
    EXPECT_STREQ(rocprof_trace_decoder_get_info_string(ROCPROFILER_THREAD_TRACE_DECODER_INFO_LAST), "INFO_LAST");
    EXPECT_STREQ(
        rocprof_trace_decoder_get_info_string(static_cast<rocprofiler_thread_trace_decoder_info_t>(9999)),
        "Unknown info parameter."
    );
}

TEST(ApiStringTest, AllStatusStrings)
{
    auto has = [](rocprofiler_thread_trace_decoder_status_t s, const char* sub)
    { return std::string(rocprof_trace_decoder_get_status_string(s)).find(sub) != std::string::npos; };
    EXPECT_TRUE(has(ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS, "SUCCESS"));
    EXPECT_TRUE(has(ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR, "ERROR"));
    EXPECT_TRUE(has(ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES, "OUT_OF_RESOURCES"));
    EXPECT_TRUE(has(ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT, "INVALID_ARGUMENT"));
    EXPECT_TRUE(has(ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_SHADER_DATA, "INVALID_SHADER_DATA"));
    EXPECT_TRUE(has(ROCPROFILER_THREAD_TRACE_DECODER_STATUS_LAST, "LAST"));
    EXPECT_STREQ(
        rocprof_trace_decoder_get_status_string(static_cast<rocprofiler_thread_trace_decoder_status_t>(9999)),
        "STATUS_UNKNOWN"
    );
}
