// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "lib/aqlprofile/core/pm4_factory.h"

#include <gtest/gtest.h>

#include <atomic>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

using namespace aql_profile;

namespace
{
aqlprofile_agent_info_v1_t
makeTestAgentInfo(uint32_t cu_num)
{
    aqlprofile_agent_info_v1_t info{};
    info.agent_gfxip          = "gfx900";
    info.cu_num               = cu_num;
    info.se_num               = 4;
    info.xcc_num              = 1;
    info.shader_arrays_per_se = 2;
    info.domain               = 0;
    info.location_id          = 0x1234;
    return info;
}

void
captureFirstException(std::exception_ptr  exception,
                      std::exception_ptr& first_exception,
                      std::mutex&         mutex)
{
    std::lock_guard<std::mutex> lock{mutex};
    if(first_exception == nullptr) first_exception = exception;
}

}  // namespace

// Regression test for concurrent handle-based factory creation. All threads start together to
// exercise the shared PM4 factory cache, then verify same-agent calls reuse the cached factory
// while different agent keys get distinct factories.
TEST(Pm4FactoryThreadSafetyTest, CreateFromHandlesIsThreadSafe)
{
    auto same_agent_info      = makeTestAgentInfo(64);
    auto different_agent_info = makeTestAgentInfo(65);

    const auto same_agent_handle      = RegisterAgent(&same_agent_info);
    const auto different_agent_handle = RegisterAgent(&different_agent_info);

    constexpr size_t thread_count = 16;
    constexpr size_t iterations   = 128;
    const size_t     result_count = thread_count * iterations;

    std::vector<Pm4Factory*> same_agent_factories(result_count, nullptr);
    std::vector<Pm4Factory*> different_agent_factories(result_count, nullptr);

    std::atomic<bool>        start{false};
    std::exception_ptr       first_exception = nullptr;
    std::mutex               exception_mutex;
    std::vector<std::thread> threads;
    threads.reserve(thread_count);

    for(size_t thread_idx = 0; thread_idx < thread_count; ++thread_idx)
    {
        threads.emplace_back([&, thread_idx]() {
            try
            {
                while(!start.load(std::memory_order_acquire))
                {
                    std::this_thread::yield();
                }

                for(size_t iteration = 0; iteration < iterations; ++iteration)
                {
                    const auto result_idx            = thread_idx * iterations + iteration;
                    same_agent_factories[result_idx] = Pm4Factory::Create(same_agent_handle, false);
                    different_agent_factories[result_idx] =
                        Pm4Factory::Create(different_agent_handle, false);
                }
            } catch(...)
            {
                captureFirstException(std::current_exception(), first_exception, exception_mutex);
            }
        });
    }

    start.store(true, std::memory_order_release);

    for(auto& thread : threads)
    {
        thread.join();
    }

    if(first_exception != nullptr) std::rethrow_exception(first_exception);

    ASSERT_NE(same_agent_factories.front(), nullptr);
    ASSERT_NE(different_agent_factories.front(), nullptr);
    EXPECT_NE(same_agent_factories.front(), different_agent_factories.front());

    for(size_t idx = 0; idx < result_count; ++idx)
    {
        EXPECT_EQ(same_agent_factories[idx], same_agent_factories.front());
        EXPECT_EQ(different_agent_factories[idx], different_agent_factories.front());
    }
}
