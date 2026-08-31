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

#include "lib/rocprofiler-sdk/thread_trace/kfd_resource.hpp"
#include "lib/common/environment.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"

#include <gtest/gtest.h>

#include <hsa/hsa.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <thread>
#include <vector>

namespace rocprofiler
{
void
test_init();
}  // namespace rocprofiler

namespace
{
using namespace rocprofiler;

class kfd_resource_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
        test_init();
        ASSERT_FALSE(hsa::get_queue_controller()->get_supported_agents().empty());
    }
};

TEST_F(kfd_resource_test, memory_allocation)
{
    constexpr size_t allocation_size = 8192;
    constexpr size_t alignment       = 2 * 1024 * 1024;

    for(const auto& [_, agent] : hsa::get_queue_controller()->get_supported_agents())
    {
        auto memory = std::make_shared<thread_trace::kfd_memory_pool_t>(
            *CHECK_NOTNULL(agent.get_rocp_agent()));

        EXPECT_EQ(memory->allocate(0, thread_trace::kfd_memory_kind_t::host), nullptr);

        auto* host =
            memory->allocate(allocation_size, thread_trace::kfd_memory_kind_t::host, alignment);
        auto* device = memory->allocate(allocation_size, thread_trace::kfd_memory_kind_t::device);

        ASSERT_NE(host, nullptr);
        ASSERT_NE(device, nullptr);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(host) % alignment, 0);
        EXPECT_FALSE(memory->is_device_pointer(host));
        EXPECT_TRUE(memory->is_device_pointer(device));
        EXPECT_TRUE(memory->is_device_pointer(static_cast<char*>(device) + allocation_size - 1));
        EXPECT_FALSE(memory->is_device_pointer(static_cast<char*>(device) + allocation_size));

        std::memset(host, 0xA5, allocation_size);
        EXPECT_EQ(static_cast<unsigned char*>(host)[allocation_size - 1], 0xA5);

        memory->deallocate(device);
        memory->deallocate(host);
    }
}

TEST_F(kfd_resource_test, cp_copy_boundaries)
{
    constexpr size_t max_copy_size = (64 * 1024 * 1024) + 4096;

    for(const auto& [_, agent] : hsa::get_queue_controller()->get_supported_agents())
    {
        auto memory = std::make_shared<thread_trace::kfd_memory_pool_t>(
            *CHECK_NOTNULL(agent.get_rocp_agent()));
        auto queue  = thread_trace::kfd_aql_queue_t{memory, max_copy_size};
        auto signal = thread_trace::kfd_signal_t{memory};

        auto* src = memory->allocate(max_copy_size, thread_trace::kfd_memory_kind_t::host);
        auto* gpu = memory->allocate(max_copy_size, thread_trace::kfd_memory_kind_t::device);
        auto* dst = memory->allocate(max_copy_size, thread_trace::kfd_memory_kind_t::host);
        std::memset(src, 0xA5, max_copy_size);

        constexpr auto sizes = std::array<size_t, 3>{1, 4096, max_copy_size};
        for(const auto size : sizes)
        {
            std::memset(dst, 0x5A, size);
            queue.copy(gpu, src, size, signal);
            queue.copy(dst, gpu, size, signal);
            EXPECT_EQ(std::memcmp(src, dst, size), 0) << "copy size " << size;
        }

        memory->deallocate(dst);
        memory->deallocate(gpu);
        memory->deallocate(src);
    }
}

TEST_F(kfd_resource_test, preferred_copy_boundaries)
{
    constexpr size_t max_copy_size = (64 * 1024 * 1024) + 4096;

    for(const auto& [_, agent] : hsa::get_queue_controller()->get_supported_agents())
    {
        auto memory = std::make_shared<thread_trace::kfd_memory_pool_t>(
            *CHECK_NOTNULL(agent.get_rocp_agent()));
        auto queue = std::make_shared<thread_trace::kfd_copy_queue_t>(memory, max_copy_size);

        auto* src = memory->allocate(max_copy_size, thread_trace::kfd_memory_kind_t::host);
        auto* gpu = memory->allocate(max_copy_size, thread_trace::kfd_memory_kind_t::device);
        auto* dst = memory->allocate(max_copy_size, thread_trace::kfd_memory_kind_t::host);
        std::memset(src, 0x3C, max_copy_size);
        std::memset(dst, 0xC3, max_copy_size);

        constexpr auto sizes = std::array<size_t, 4>{1, 4096, (4 * 1024 * 1024) + 1, max_copy_size};
        for(const auto size : sizes)
        {
            queue->copy(gpu, src, size);
            queue->copy(dst, gpu, size);
            EXPECT_EQ(std::memcmp(src, dst, size), 0) << "copy size " << size;
        }

        memory->deallocate(dst);
        memory->deallocate(gpu);
        memory->deallocate(src);
    }
}

TEST_F(kfd_resource_test, forced_cp_copy)
{
    constexpr size_t  copy_size = 4096;
    common::env_store env({{"ROCPROFILER_SQTT_FORCE_CP_DMA", "1", 1}});
    ASSERT_TRUE(env.push());

    for(const auto& [_, agent] : hsa::get_queue_controller()->get_supported_agents())
    {
        auto memory = std::make_shared<thread_trace::kfd_memory_pool_t>(
            *CHECK_NOTNULL(agent.get_rocp_agent()));
        auto queue = std::make_shared<thread_trace::kfd_copy_queue_t>(memory, copy_size);

        auto* src = memory->allocate(copy_size, thread_trace::kfd_memory_kind_t::host);
        auto* gpu = memory->allocate(copy_size, thread_trace::kfd_memory_kind_t::device);
        auto* dst = memory->allocate(copy_size, thread_trace::kfd_memory_kind_t::host);
        std::memset(src, 0xC3, copy_size);

        queue->copy(gpu, src, copy_size);
        queue->copy(dst, gpu, copy_size);
        EXPECT_EQ(std::memcmp(src, dst, copy_size), 0);

        memory->deallocate(dst);
        memory->deallocate(gpu);
        memory->deallocate(src);
    }
}

TEST_F(kfd_resource_test, concurrent_copy_reuse)
{
    constexpr size_t copy_size  = 4096;
    constexpr size_t iterations = 192;
    constexpr size_t workers    = 4;

    for(const auto& [_, agent] : hsa::get_queue_controller()->get_supported_agents())
    {
        auto memory = std::make_shared<thread_trace::kfd_memory_pool_t>(
            *CHECK_NOTNULL(agent.get_rocp_agent()));

        struct copy_data_t
        {
            void* src{};
            void* gpu{};
            void* dst{};
        };
        auto data = std::array<copy_data_t, workers>{};

        for(size_t i = 0; i < workers; ++i)
        {
            data[i].src = memory->allocate(copy_size, thread_trace::kfd_memory_kind_t::host);
            data[i].gpu = memory->allocate(copy_size, thread_trace::kfd_memory_kind_t::device);
            data[i].dst = memory->allocate(copy_size, thread_trace::kfd_memory_kind_t::host);
            std::memset(data[i].src, static_cast<int>(i + 1), copy_size);
        }

        {
            auto queue  = thread_trace::kfd_aql_queue_t{memory, copy_size};
            auto signal = thread_trace::kfd_signal_t{memory};
            for(size_t i = 0; i < 600; ++i)
            {
                queue.copy(data[0].gpu, data[0].src, copy_size, signal);
                queue.copy(data[0].dst, data[0].gpu, copy_size, signal);
            }
            EXPECT_EQ(std::memcmp(data[0].src, data[0].dst, copy_size), 0);
        }

        auto queue   = std::make_shared<thread_trace::kfd_copy_queue_t>(memory, copy_size);
        auto threads = std::array<std::thread, workers>{};
        for(size_t i = 0; i < workers; ++i)
        {
            threads[i] = std::thread{[&, i]() {
                for(size_t j = 0; j < iterations; ++j)
                {
                    queue->copy(data[i].gpu, data[i].src, copy_size);
                    queue->copy(data[i].dst, data[i].gpu, copy_size);
                }
            }};
        }
        for(auto& thread : threads)
            thread.join();

        for(const auto& entry : data)
        {
            EXPECT_EQ(std::memcmp(entry.src, entry.dst, copy_size), 0);
            memory->deallocate(entry.dst);
            memory->deallocate(entry.gpu);
            memory->deallocate(entry.src);
        }
    }
}
}  // namespace
