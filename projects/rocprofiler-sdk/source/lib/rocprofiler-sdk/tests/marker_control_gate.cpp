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

#include "lib/rocprofiler-sdk/marker/marker.hpp"

#include <gtest/gtest.h>

#include <sys/wait.h>
#include <unistd.h>
#include <chrono>
#include <future>
#include <thread>

TEST(marker_control_gate, writer_waits_for_inflight_control_call)
{
    auto release_reader  = std::promise<void>{};
    auto reader_acquired = std::promise<void>{};
    auto writer_started  = std::promise<void>{};
    auto writer_acquired = std::promise<void>{};

    auto release_reader_future  = release_reader.get_future();
    auto reader_acquired_future = reader_acquired.get_future();
    auto writer_started_future  = writer_started.get_future();
    auto writer_acquired_future = writer_acquired.get_future();

    auto reader = std::thread{[&]() {
        auto lock = rocprofiler::marker::acquire_control_api_read_lock();
        reader_acquired.set_value();
        release_reader_future.wait();
    }};

    reader_acquired_future.wait();

    auto writer = std::thread{[&]() {
        writer_started.set_value();
        auto lock = rocprofiler::marker::acquire_control_api_write_lock();
        writer_acquired.set_value();
    }};

    writer_started_future.wait();
    EXPECT_EQ(writer_acquired_future.wait_for(std::chrono::milliseconds{10}),
              std::future_status::timeout);

    release_reader.set_value();
    EXPECT_EQ(writer_acquired_future.wait_for(std::chrono::seconds{1}), std::future_status::ready);

    reader.join();
    writer.join();
}

TEST(marker_control_gate, fork_child_uses_fresh_gate)
{
    auto parent_lock = rocprofiler::marker::acquire_control_api_read_lock();
    auto child_pid   = fork();
    ASSERT_GE(child_pid, 0);

    if(child_pid == 0)
    {
        alarm(2);
        auto child_lock = rocprofiler::marker::acquire_control_api_write_lock();
        _exit(0);
    }

    auto child_status = int{0};
    ASSERT_EQ(waitpid(child_pid, &child_status, 0), child_pid);
    EXPECT_TRUE(WIFEXITED(child_status));
    EXPECT_EQ(WEXITSTATUS(child_status), 0);
}
