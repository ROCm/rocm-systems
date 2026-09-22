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

#include "lib/common/filesystem.hpp"
#include "lib/common/scope_destructor.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/kfd/capabilities.hpp"
#include "lib/rocprofiler-sdk/kfd/resource.hpp"

#include <gtest/gtest.h>

#include <hsa/hsa.h>

#include <fcntl.h>
#include <unistd.h>
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

TEST(kfd_resource, copy_support)
{
    EXPECT_FALSE(kfd::kfd_copy_queue_t::is_supported(90000));  // gfx900
    EXPECT_FALSE(kfd::kfd_copy_queue_t::is_supported(90006));  // gfx906
    EXPECT_FALSE(kfd::kfd_copy_queue_t::is_supported(90008));  // gfx908
    EXPECT_TRUE(kfd::kfd_copy_queue_t::is_supported(90010));   // gfx90a
    EXPECT_TRUE(kfd::kfd_copy_queue_t::is_supported(90402));   // gfx942
    EXPECT_TRUE(kfd::kfd_copy_queue_t::is_supported(100300));  // gfx1030
    EXPECT_TRUE(kfd::kfd_copy_queue_t::is_supported(120001));  // gfx1201
}

TEST(kfd_resource, queue_capabilities)
{
    namespace caps = kfd::capabilities;
    EXPECT_FALSE(caps::supports_wave32(90402));
    EXPECT_TRUE(caps::supports_wave32(100300));
    EXPECT_TRUE(caps::supports_wave32(110000));

    EXPECT_FALSE(caps::needs_cwsr_control_stack_cap(90402));
    EXPECT_TRUE(caps::needs_cwsr_control_stack_cap(100300));
    EXPECT_FALSE(caps::needs_cwsr_control_stack_cap(110000));

    EXPECT_TRUE(caps::needs_aql_eop_buffer(90010));
    EXPECT_FALSE(caps::needs_aql_eop_buffer(90400));
    EXPECT_FALSE(caps::needs_aql_eop_buffer(90402));
    EXPECT_TRUE(caps::needs_aql_eop_buffer(90500));
    EXPECT_TRUE(caps::needs_aql_eop_buffer(100300));
}

TEST(kfd_resource, sdma_capabilities)
{
    namespace caps = kfd::capabilities;
    EXPECT_FALSE(caps::has_sdma_copy_scope_fields(90402));
    EXPECT_FALSE(caps::has_sdma_copy_scope_fields(110000));
    EXPECT_TRUE(caps::has_sdma_copy_scope_fields(110500));
    EXPECT_FALSE(caps::has_sdma_copy_scope_fields(120001));
    EXPECT_TRUE(caps::has_sdma_copy_scope_fields(120500));

    EXPECT_FALSE(caps::needs_sdma_gcr(90402));
    EXPECT_TRUE(caps::needs_sdma_gcr(100300));
    EXPECT_TRUE(caps::needs_sdma_gcr(110000));
    EXPECT_FALSE(caps::needs_sdma_gcr(110500));
    EXPECT_TRUE(caps::needs_sdma_gcr(120001));
    EXPECT_FALSE(caps::needs_sdma_gcr(120500));

    EXPECT_FALSE(caps::needs_uncached_sdma_fence(90402));
    EXPECT_TRUE(caps::needs_uncached_sdma_fence(100300));
    EXPECT_TRUE(caps::needs_uncached_sdma_fence(120001));

    EXPECT_FALSE(caps::has_sdma_fence_system_bit(110500));
    EXPECT_TRUE(caps::has_sdma_fence_system_bit(120001));
    EXPECT_TRUE(caps::has_sdma_fence_system_bit(120500));
}

TEST_F(kfd_resource_test, memory_allocation)
{
    constexpr size_t allocation_size = 8192;
    constexpr size_t alignment       = 2 * 1024 * 1024;

    for(const auto& [_, agent] : hsa::get_queue_controller()->get_supported_agents())
    {
        auto memory = kfd::kfd_memory_pool_t::create(*CHECK_NOTNULL(agent.get_rocp_agent()));
        ASSERT_NE(memory, nullptr);

        EXPECT_EQ(memory->allocate(0, kfd::kfd_memory_kind_t::host), nullptr);

        auto* host   = memory->allocate(allocation_size, kfd::kfd_memory_kind_t::host, alignment);
        auto* device = memory->allocate(allocation_size, kfd::kfd_memory_kind_t::device);

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

TEST_F(kfd_resource_test, sdma_copy_boundaries)
{
    constexpr size_t max_copy_size = (64 * 1024 * 1024) + 4096;

    for(const auto& [_, agent] : hsa::get_queue_controller()->get_supported_agents())
    {
        auto memory = kfd::kfd_memory_pool_t::create(*CHECK_NOTNULL(agent.get_rocp_agent()));
        ASSERT_NE(memory, nullptr);
        auto queue = kfd::kfd_copy_queue_t::create(memory, max_copy_size);
        if(!kfd::kfd_copy_queue_t::is_supported(memory->gfx_target_version()))
        {
            EXPECT_EQ(queue, nullptr);
            continue;
        }
        ASSERT_NE(queue, nullptr);

        auto* src = memory->allocate(max_copy_size, kfd::kfd_memory_kind_t::host);
        auto* gpu = memory->allocate(max_copy_size, kfd::kfd_memory_kind_t::device);
        auto* dst = memory->allocate(max_copy_size, kfd::kfd_memory_kind_t::host);
        std::memset(src, 0x3C, max_copy_size);
        std::memset(dst, 0xC3, max_copy_size);

        constexpr auto sizes = std::array<size_t, 4>{1, 4096, (4 * 1024 * 1024) + 1, max_copy_size};
        for(const auto size : sizes)
        {
            ASSERT_TRUE(queue->copy(gpu, src, size));
            ASSERT_TRUE(queue->copy(dst, gpu, size));
            EXPECT_EQ(std::memcmp(src, dst, size), 0) << "copy size " << size;
        }

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
        auto memory = kfd::kfd_memory_pool_t::create(*CHECK_NOTNULL(agent.get_rocp_agent()));
        ASSERT_NE(memory, nullptr);
        auto queue = kfd::kfd_copy_queue_t::create(memory, copy_size);
        if(!kfd::kfd_copy_queue_t::is_supported(memory->gfx_target_version()))
        {
            EXPECT_EQ(queue, nullptr);
            continue;
        }
        ASSERT_NE(queue, nullptr);

        struct copy_data_t
        {
            void* src{};
            void* gpu{};
            void* dst{};
        };
        auto data = std::array<copy_data_t, workers>{};

        for(size_t i = 0; i < workers; ++i)
        {
            data[i].src = memory->allocate(copy_size, kfd::kfd_memory_kind_t::host);
            data[i].gpu = memory->allocate(copy_size, kfd::kfd_memory_kind_t::device);
            data[i].dst = memory->allocate(copy_size, kfd::kfd_memory_kind_t::host);
            std::memset(data[i].src, static_cast<int>(i + 1), copy_size);
        }

        auto threads = std::array<std::thread, workers>{};
        for(size_t i = 0; i < workers; ++i)
        {
            threads[i] = std::thread{[&, i]() {
                for(size_t j = 0; j < iterations; ++j)
                {
                    ASSERT_TRUE(queue->copy(data[i].gpu, data[i].src, copy_size));
                    ASSERT_TRUE(queue->copy(data[i].dst, data[i].gpu, copy_size));
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

TEST_F(kfd_resource_test, invalid_arguments_and_closed_queue)
{
    const auto& agent  = hsa::get_queue_controller()->get_supported_agents().begin()->second;
    auto        memory = kfd::kfd_memory_pool_t::create(*agent.get_rocp_agent());
    ASSERT_NE(memory, nullptr);
    EXPECT_EQ(kfd::kfd_signal_t::create(nullptr), nullptr);
    EXPECT_EQ(kfd::kfd_copy_queue_t::create(nullptr, 4096), nullptr);
    EXPECT_EQ(kfd::kfd_copy_queue_t::create(memory, 0), nullptr);
    auto queue = kfd::kfd_copy_queue_t::create(memory, 4096);
    if(!kfd::kfd_copy_queue_t::is_supported(memory->gfx_target_version()))
    {
        EXPECT_EQ(queue, nullptr);
        return;
    }
    ASSERT_NE(queue, nullptr);
    EXPECT_TRUE(queue->copy(nullptr, nullptr, 0));
    EXPECT_FALSE(queue->copy(nullptr, nullptr, 1));
    EXPECT_TRUE(queue->close());
    EXPECT_TRUE(queue->close());
    hsa_ext_amd_aql_pm4_packet_t packet{};
    EXPECT_FALSE(queue->submit(packet, {}));
}

TEST(kfd_resource, unrelated_render_descriptor)
{
    std::vector<int> descriptors;
    auto             cleanup = common::scope_destructor{[&] {
        for(int fd : descriptors)
            ::close(fd);
    }};
    for(const auto& entry : common::filesystem::directory_iterator{"/dev/dri"})
    {
        if(entry.path().filename().string().find("renderD") != 0) continue;
        const int fd = ::open(entry.path().c_str(), O_RDWR | O_CLOEXEC);
        // Multi-XCD GPUs and container filtering leave some render nodes unusable.
        if(fd >= 0) descriptors.push_back(fd);
    }
    ASSERT_FALSE(descriptors.empty()) << "No accessible DRM render nodes";
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();
    ASSERT_FALSE(hsa::get_queue_controller()->get_supported_agents().empty());
    for(const auto& [_, agent] : hsa::get_queue_controller()->get_supported_agents())
    {
        auto memory = kfd::kfd_memory_pool_t::create(*agent.get_rocp_agent());
        ASSERT_NE(memory, nullptr);
        auto* host = memory->allocate(4096, kfd::kfd_memory_kind_t::host);
        ASSERT_NE(host, nullptr);
        memset(host, 0xA5, 4096);
        memory->deallocate(host);
    }
}

TEST_F(kfd_resource_test, copy_after_runtime_shutdown)
{
    const auto& agent = hsa::get_queue_controller()->get_supported_agents().begin()->second;
    if(!kfd::kfd_copy_queue_t::is_supported(agent.get_rocp_agent()->gfx_target_version))
        GTEST_SKIP() << "Direct KFD copies require gfx90a or newer";
    auto memory = kfd::kfd_memory_pool_t::create(*agent.get_rocp_agent());
    ASSERT_NE(memory, nullptr);
    auto queue = kfd::kfd_copy_queue_t::create(memory, 4096);
    ASSERT_NE(queue, nullptr);
    auto* source = memory->allocate(4096, kfd::kfd_memory_kind_t::host);
    auto* device = memory->allocate(4096, kfd::kfd_memory_kind_t::device);
    auto* target = memory->allocate(4096, kfd::kfd_memory_kind_t::host);
    ASSERT_NE(source, nullptr);
    ASSERT_NE(device, nullptr);
    ASSERT_NE(target, nullptr);
    memset(source, 0xBA, 4096);
    ASSERT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
    uint16_t version = 0;
    ASSERT_EQ(hsa_system_get_info(HSA_SYSTEM_INFO_VERSION_MAJOR, &version),
              HSA_STATUS_ERROR_NOT_INITIALIZED);
    ASSERT_TRUE(queue->copy(device, source, 4096));
    ASSERT_TRUE(queue->copy(target, device, 4096));
    EXPECT_EQ(memcmp(source, target, 4096), 0);
}
}  // namespace
