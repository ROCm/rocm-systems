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

#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/common/units.hpp"
#include "lib/rocprofiler-sdk/internal_threading.hpp"

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>

#include <gtest/gtest.h>

#include <pthread.h>
#include <sys/wait.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <random>
#include <thread>
#include <typeinfo>
#include <vector>

TEST(rocprofiler_lib, buffer)
{
    namespace buffer = ::rocprofiler::buffer;
    namespace common = ::rocprofiler::common;

    ASSERT_EQ(buffer::get_buffers()->size(), 0)
        << "no buffers should have been created at this point";
    auto buffer_id = buffer::allocate_buffer();

    EXPECT_TRUE(buffer_id) << "failed to allocate buffer";
    EXPECT_GT(buffer_id->handle, 0);
    EXPECT_TRUE(buffer::is_valid_buffer_id(*buffer_id)) << "id=" << buffer_id->handle;
    ASSERT_EQ(buffer::get_buffers()->size(), 1) << "incorrect number of buffers created";

    // get pointer to buffer
    auto* buffer_v = buffer::get_buffer(*buffer_id);
    ASSERT_NE(buffer_v, nullptr) << "get_buffer returned a nullptr. id=" << buffer_id->handle;
    EXPECT_EQ(buffer_v->buffer_id, buffer_id->handle);

    buffer_v->watermark = common::units::get_page_size();
    EXPECT_EQ(buffer_v->get_internal_buffer().get_num_record_headers(), 0);

    EXPECT_TRUE(buffer_v->get_internal_buffer().allocate(sizeof(rocprofiler_buffer_id_t)));

    EXPECT_EQ(buffer_v->get_internal_buffer().capacity(), common::units::get_page_size());

    auto data = *buffer_id;
    buffer_v->emplace(1, 1, data);

    EXPECT_EQ(buffer_v->get_internal_buffer().get_num_record_headers(), 1);

    auto flush_status = buffer::flush(*buffer_id, true);
    EXPECT_EQ(flush_status, ROCPROFILER_STATUS_SUCCESS);

    auto destroy_status = rocprofiler_destroy_buffer(*buffer_id);
    EXPECT_EQ(destroy_status, ROCPROFILER_STATUS_SUCCESS);
}

TEST(rocprofiler_lib, completion_fence_waits_and_recovers_after_timeout)
{
    auto worker    = rocprofiler::internal_threading::task_group_t{1};
    auto started   = std::promise<void>{};
    auto released  = std::promise<void>{};
    auto begin     = started.get_future();
    auto release   = released.get_future();
    auto completed = std::atomic<bool>{false};

    EXPECT_FALSE(rocprofiler::internal_threading::is_task_group_thread());
    worker.async([&]() {
        started.set_value();
        release.wait();
        completed.store(true, std::memory_order_release);
    });
    EXPECT_EQ(begin.wait_for(std::chrono::seconds{2}), std::future_status::ready);
    EXPECT_FALSE(worker.wait_async_for(0));
    EXPECT_FALSE(worker.wait_async_for(1'000'000));
    EXPECT_FALSE(completed.load(std::memory_order_acquire));
    released.set_value();
    EXPECT_TRUE(worker.wait_async_for(2'000'000'000));
    EXPECT_TRUE(completed.load(std::memory_order_acquire));
    EXPECT_TRUE(worker.wait_async_for(0));
    worker.join();
}

TEST(rocprofiler_lib, completion_fence_rejects_worker_reentry)
{
    auto worker = rocprofiler::internal_threading::task_group_t{1};
    auto other  = rocprofiler::internal_threading::task_group_t{1};
    for(bool use_async : {false, true})
    {
        auto result = std::promise<bool>{};
        auto ready  = result.get_future();
        auto task   = [&]() {
            result.set_value(rocprofiler::internal_threading::is_task_group_thread() &&
                             !worker.wait_async_for(1'000'000'000) &&
                             !other.wait_async_for(1'000'000'000));
        };
        if(use_async)
            worker.async(task);
        else
            worker.exec(task);
        ASSERT_EQ(ready.wait_for(std::chrono::seconds{2}), std::future_status::ready);
        EXPECT_TRUE(ready.get());
        worker.join();
    }
    EXPECT_FALSE(rocprofiler::internal_threading::is_task_group_thread());
}

TEST(rocprofiler_lib, completion_fence_rejects_inherited_pool)
{
    rocprofiler::internal_threading::initialize();
    auto worker = rocprofiler::internal_threading::task_group_t{1};
    auto child  = fork();
    ASSERT_GE(child, 0);
    if(child == 0)
        _exit(rocprofiler::internal_threading::fork_stale() && !worker.wait_async_for(0)
                  ? EXIT_SUCCESS
                  : EXIT_FAILURE);

    int status = 0;
    ASSERT_EQ(waitpid(child, &status, 0), child);
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), EXIT_SUCCESS);
    EXPECT_TRUE(worker.wait_async_for(0));
}

TEST(rocprofiler_lib, completion_fence_supports_concurrent_callers)
{
    auto worker  = rocprofiler::internal_threading::task_group_t{1};
    auto written = std::atomic<unsigned>{0};
    for(unsigned index = 0; index < 64; ++index)
        worker.async([&]() { written.fetch_add(1, std::memory_order_release); });
    auto callers = std::vector<std::future<bool>>{};
    for(unsigned index = 0; index < 4; ++index)
        callers.emplace_back(std::async(std::launch::async, [&]() {
            return worker.wait_async_for(2'000'000'000) &&
                   written.load(std::memory_order_acquire) == 64;
        }));
    for(auto& caller : callers)
        EXPECT_TRUE(caller.get());
    worker.join();
}

TEST(rocprofiler_lib, completion_fence_waits_for_multiple_workers)
{
    auto worker    = rocprofiler::internal_threading::task_group_t{2};
    auto started   = std::promise<void>{};
    auto ready     = started.get_future();
    auto released  = std::promise<void>{};
    auto release   = released.get_future().share();
    auto running   = std::atomic<unsigned>{0};
    auto completed = std::atomic<unsigned>{0};
    for(unsigned index = 0; index < 2; ++index)
        worker.async([&]() {
            if(running.fetch_add(1) == 1) started.set_value();
            release.wait();
            completed.fetch_add(1, std::memory_order_release);
        });
    EXPECT_EQ(ready.wait_for(std::chrono::seconds{2}), std::future_status::ready);
    EXPECT_FALSE(worker.wait_async_for(0));
    released.set_value();
    EXPECT_TRUE(worker.wait_async_for(2'000'000'000));
    EXPECT_EQ(completed.load(std::memory_order_acquire), 2);
    worker.join();
}

TEST(rocprofiler_lib, completion_fence_waits_for_another_producer)
{
    auto worker          = rocprofiler::internal_threading::task_group_t{1};
    auto first_started   = std::promise<void>{};
    auto first_ready     = first_started.get_future();
    auto first_release   = std::promise<void>{};
    auto first_released  = first_release.get_future();
    auto tail_started    = std::promise<void>{};
    auto tail_ready      = tail_started.get_future();
    auto tail_release    = std::promise<void>{};
    auto tail_released   = tail_release.get_future();
    auto fence_started   = std::promise<void>{};
    auto fence_submitted = fence_started.get_future();

    auto producer = std::thread{[&]() {
        worker.async([&]() {
            first_started.set_value();
            first_released.wait();
        });
        worker.async([&]() {
            tail_started.set_value();
            tail_released.wait();
        });
    }};
    producer.join();
    EXPECT_EQ(first_ready.wait_for(std::chrono::seconds{2}), std::future_status::ready);
    auto fence = std::async(std::launch::async, [&]() {
        fence_started.set_value();
        return worker.wait_async_for(2'000'000'000);
    });
    EXPECT_EQ(fence_submitted.wait_for(std::chrono::seconds{2}), std::future_status::ready);
    first_release.set_value();
    EXPECT_EQ(tail_ready.wait_for(std::chrono::seconds{2}), std::future_status::ready);
    EXPECT_EQ(fence.wait_for(std::chrono::milliseconds{20}), std::future_status::timeout);
    tail_release.set_value();
    EXPECT_TRUE(fence.get());
    worker.join();
}
