// Copyright (c) 2024 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "details/logging.hpp"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <thread>
#include <vector>

// Smoke-test that initialize() can be called repeatedly (and from multiple
// threads) without crashing. The first call performs glog initialization
// inside std::call_once; subsequent calls must be no-ops.
TEST(logging, initialize_is_idempotent)
{
    rocprofiler_register::logging::initialize();
    rocprofiler_register::logging::initialize();
    EXPECT_TRUE(google::IsGoogleLoggingInitialized());
}

TEST(logging, initialize_is_thread_safe)
{
    auto threads = std::vector<std::thread>{};
    threads.reserve(8);
    for(int i = 0; i < 8; ++i)
    {
        threads.emplace_back(
            []() { rocprofiler_register::logging::initialize(); });
    }
    for(auto& t : threads)
        t.join();
    EXPECT_TRUE(google::IsGoogleLoggingInitialized());
}

// Ensures the ERROR macro from <wingdi.h> (when transitively included via
// <windows.h>) does not poison glog severities. On Linux this is a vacuous
// check; on Windows it is the regression test for the NOGDI guard. Use the
// GLOG_-prefixed severity symbols which exist on both platforms (the
// abbreviated google::INFO/WARNING/ERROR/FATAL aliases are suppressed when
// GLOG_NO_ABBREVIATED_SEVERITIES is defined to avoid the wingdi collision).
TEST(logging, severity_symbols_resolve)
{
    EXPECT_EQ(google::GLOG_INFO, 0);
    EXPECT_EQ(google::GLOG_WARNING, 1);
    EXPECT_EQ(google::GLOG_ERROR, 2);
    EXPECT_EQ(google::GLOG_FATAL, 3);
}
