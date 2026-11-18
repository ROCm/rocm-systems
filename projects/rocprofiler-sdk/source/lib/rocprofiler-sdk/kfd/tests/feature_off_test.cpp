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

// Flag-off inertness: with the feature off, the public KFD dispatch-log entry
// points reached during normal SDK operation must construct NEITHER the
// profiler_state NOR the reader_state singleton. The CTest layer pins
// ROCPROFILER_KFD_DISPATCH_LOG_SIGNAL_LESS=0 and runs this in its own process, so
// nothing else has constructed the singletons first.

#include "lib/rocprofiler-sdk/kfd/kfd_profiler.hpp"
#include "lib/rocprofiler-sdk/kfd/kfd_reader.hpp"

#include <gtest/gtest.h>

using namespace rocprofiler::kfd;

TEST(feature_off, static_objects_stay_unconstructed)
{
    // Clean start: nothing in this process has constructed either singleton yet.
    ASSERT_FALSE(kfd_profiler_state_constructed()) << "profiler_state must start unconstructed";
    ASSERT_FALSE(kfd_reader_state_constructed()) << "reader_state must start unconstructed";

    // The arm path (reached from a kernel-dispatch context start) must peek, not
    // construct, when the feature is off.
    arm_dispatch_log_sessions();
    EXPECT_FALSE(kfd_profiler_state_constructed()) << "arm must not construct profiler_state";
    EXPECT_FALSE(kfd_reader_state_constructed()) << "arm must not start the reader";

    // The atfork/disable path must peek, not construct.
    disable_kfd_dispatch_log();
    EXPECT_FALSE(kfd_profiler_state_constructed()) << "disable must not construct profiler_state";

    // Shutdown (called unconditionally from finalize) must return before
    // stop_kfd_reader()'s constructing accessor when nothing was ever constructed.
    shutdown_kfd_profiler();
    EXPECT_FALSE(kfd_profiler_state_constructed()) << "shutdown must not construct profiler_state";
    EXPECT_FALSE(kfd_reader_state_constructed()) << "shutdown must not construct reader_state";

    // The public availability/support queries are non-constructing and report off.
    EXPECT_FALSE(kfd_dispatch_log_available());
    EXPECT_FALSE(kfd_dispatch_log_supported());
    EXPECT_FALSE(kfd_profiler_state_constructed());
    EXPECT_FALSE(kfd_reader_state_constructed());
}
