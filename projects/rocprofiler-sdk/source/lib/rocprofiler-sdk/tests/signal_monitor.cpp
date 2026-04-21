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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/rocprofiler-sdk/hsa/signal_monitor.hpp"
#include "lib/common/environment.hpp"

#include <gtest/gtest.h>

namespace rocprofiler::hsa::test
{
TEST(signal_monitor, evaluate_condition_eq)
{
    EXPECT_TRUE(evaluate_signal_condition(HSA_SIGNAL_CONDITION_EQ, 0, 0));
    EXPECT_FALSE(evaluate_signal_condition(HSA_SIGNAL_CONDITION_EQ, -1, 0));
}

TEST(signal_monitor, evaluate_condition_lt)
{
    EXPECT_TRUE(evaluate_signal_condition(HSA_SIGNAL_CONDITION_LT, 0, 1));
    EXPECT_FALSE(evaluate_signal_condition(HSA_SIGNAL_CONDITION_LT, 2, 1));
}

TEST(signal_monitor, parse_backend_env)
{
    {
        common::env_store env{{{"ROCPROF_SIGNAL_MONITOR_BACKEND", "poll", 1}}};
        env.push();
        EXPECT_EQ(parse_signal_monitor_backend_env(), SignalMonitorBackend::poll);
        env.pop();
    }

    {
        common::env_store env{{{"ROCPROF_SIGNAL_MONITOR_BACKEND", "ioctl", 1}}};
        env.push();
        EXPECT_EQ(parse_signal_monitor_backend_env(), SignalMonitorBackend::ioctl);
        env.pop();
    }

    {
        common::env_store env{{{"ROCPROF_SIGNAL_MONITOR_BACKEND", "invalid", 1}}};
        env.push();
        EXPECT_EQ(parse_signal_monitor_backend_env(), SignalMonitorBackend::auto_select);
        env.pop();
    }

    {
        common::env_store env{{{"ROCPROF_SIGNAL_MONITOR_BACKEND", "", 1}}};
        env.push();
        EXPECT_EQ(parse_signal_monitor_backend_env(), SignalMonitorBackend::auto_select);
        env.pop();
    }
}
}  // namespace rocprofiler::hsa::test
