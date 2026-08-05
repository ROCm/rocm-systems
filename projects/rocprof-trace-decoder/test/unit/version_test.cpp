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

#include <gtest/gtest.h>

#include "rocprof_trace_decoder/rocprof_trace_decoder.h"

TEST(VersionTest, ReportsBuildVersion)
{
    rocprof_trace_decoder_version_t version{};

    EXPECT_EQ(rocprof_trace_decoder_get_version(&version), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS);
    EXPECT_EQ(version.size, sizeof(rocprof_trace_decoder_version_t));
    EXPECT_EQ(version.major, ROCPROF_TRACE_DECODER_VERSION_MAJOR);
    EXPECT_EQ(version.minor, ROCPROF_TRACE_DECODER_VERSION_MINOR);
    EXPECT_EQ(version.patch, ROCPROF_TRACE_DECODER_VERSION_PATCH);
}

TEST(VersionTest, RejectsNull)
{
    EXPECT_EQ(
        rocprof_trace_decoder_get_version(nullptr), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT
    );
}
