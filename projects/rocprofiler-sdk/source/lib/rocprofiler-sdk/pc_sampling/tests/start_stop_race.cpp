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

#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/pc_sampling.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <thread>
#include <vector>

namespace
{
constexpr size_t BUFFER_SIZE_BYTES = 8192;
constexpr size_t WATERMARK         = (BUFFER_SIZE_BYTES / 4);

struct race_data
{
    rocprofiler_context_id_t context_id = {0};
    rocprofiler_buffer_id_t  buffer_id  = {0};
    bool                     configured = false;
};

std::vector<rocprofiler_pc_sampling_configuration_t>
query_configs(rocprofiler_agent_id_t agent_id)
{
    auto cb = [](const rocprofiler_pc_sampling_configuration_t* configs,
                 size_t                                         num_config,
                 void*                                          user_data) {
        auto* out = static_cast<std::vector<rocprofiler_pc_sampling_configuration_t>*>(user_data);
        out->insert(out->end(), configs, configs + num_config);
        return ROCPROFILER_STATUS_SUCCESS;
    };

    auto configs = std::vector<rocprofiler_pc_sampling_configuration_t>{};
    if(rocprofiler_query_pc_sampling_agent_configurations(agent_id, cb, &configs) !=
       ROCPROFILER_STATUS_SUCCESS)
        configs.clear();
    return configs;
}

rocprofiler_status_t
find_pc_sampling_agents(rocprofiler_agent_version_t,
                        const void** agents,
                        size_t       num_agents,
                        void*        user_data)
{
    auto* out     = static_cast<std::vector<const rocprofiler_agent_t*>*>(user_data);
    auto* _agents = reinterpret_cast<const rocprofiler_agent_t**>(agents);
    for(size_t i = 0; i < num_agents; ++i)
    {
        if(_agents[i]->type == ROCPROFILER_AGENT_TYPE_GPU && !query_configs(_agents[i]->id).empty())
            out->push_back(_agents[i]);
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

void
drop_records(rocprofiler_context_id_t,
             rocprofiler_buffer_id_t,
             rocprofiler_record_header_t**,
             size_t,
             void*,
             uint64_t)
{}
}  // namespace

// start_context() releases its start-in-progress marker before it starts PC sampling. A stop that
// claims the context in that window tears it down first, and the start then enables PC sampling
// on a context that is no longer active, which the tool can no longer stop; or the stop's HSA
// teardown runs while the start is still inside its HSA start.
TEST(pc_sampling, concurrent_start_stop_does_not_leave_sampling_enabled)
{
    using init_func_t = int (*)(rocprofiler_client_finalize_t, void*);

    static init_func_t tool_init = [](rocprofiler_client_finalize_t, void* client_data) -> int {
        auto* data   = static_cast<race_data*>(client_data);
        auto  agents = std::vector<const rocprofiler_agent_t*>{};
        EXPECT_EQ(rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                                                     &find_pc_sampling_agents,
                                                     sizeof(rocprofiler_agent_t),
                                                     static_cast<void*>(&agents)),
                  ROCPROFILER_STATUS_SUCCESS);
        if(agents.empty()) return 0;

        auto config = query_configs(agents.front()->id).front();
        EXPECT_EQ(rocprofiler_create_context(&data->context_id), ROCPROFILER_STATUS_SUCCESS);
        EXPECT_EQ(rocprofiler_create_buffer(data->context_id,
                                            BUFFER_SIZE_BYTES,
                                            WATERMARK,
                                            ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                            drop_records,
                                            nullptr,
                                            &data->buffer_id),
                  ROCPROFILER_STATUS_SUCCESS);
        data->configured =
            rocprofiler_configure_pc_sampling_service(data->context_id,
                                                      agents.front()->id,
                                                      config.method,
                                                      config.unit,
                                                      config.min_interval,
                                                      data->buffer_id,
                                                      0) == ROCPROFILER_STATUS_SUCCESS;
        return 0;
    };

    static auto data       = race_data{};
    static auto cfg_result = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), tool_init, nullptr, static_cast<void*>(&data)};
    static rocprofiler_configure_func_t rocp_init =
        [](uint32_t,
           const char*,
           uint32_t,
           rocprofiler_client_id_t* client_id) -> rocprofiler_tool_configure_result_t* {
        client_id->name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
        return &cfg_result;
    };

    ASSERT_EQ(rocprofiler_force_configure(rocp_init), ROCPROFILER_STATUS_SUCCESS);
    if(!data.configured)
    {
        ROCP_ERROR << "PC sampling unavailable";
        return;
    }

    auto failed_starts = std::atomic<int>{0};
    auto failed_stops  = std::atomic<int>{0};
    auto last_failure  = std::atomic<int>{ROCPROFILER_STATUS_SUCCESS};
    auto go            = std::atomic<bool>{false};
    auto worker        = [&]() {
        while(!go.load(std::memory_order_acquire))
            std::this_thread::yield();

        for(int i = 0; i < 20000; ++i)
        {
            auto status = rocprofiler_start_context(data.context_id);
            if(status != ROCPROFILER_STATUS_SUCCESS)
            {
                failed_starts.fetch_add(1, std::memory_order_relaxed);
                last_failure.store(status, std::memory_order_relaxed);
            }

            status = rocprofiler_stop_context(data.context_id);
            if(status != ROCPROFILER_STATUS_SUCCESS &&
               status != ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND)
            {
                failed_stops.fetch_add(1, std::memory_order_relaxed);
                last_failure.store(status, std::memory_order_relaxed);
            }
        }
    };

    auto first  = std::thread{worker};
    auto second = std::thread{worker};
    go.store(true, std::memory_order_release);
    first.join();
    second.join();

    (void) rocprofiler_stop_context(data.context_id);

    int active = -1;
    EXPECT_EQ(rocprofiler_context_is_active(data.context_id, &active), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(active, 0);

    const auto* ctx = rocprofiler::context::get_registered_context(data.context_id);
    ASSERT_TRUE(ctx && ctx->pc_sampler);
    EXPECT_FALSE(ctx->pc_sampler->enabled.load())
        << "PC sampling left enabled on a stopped context";

    // A start that finds PC sampling already enabled on an inactive context fails with
    // ROCPROFILER_STATUS_ERROR from pc_sampling::start_service(): the visible symptom of a stop
    // having won the window.
    const auto last_status = static_cast<rocprofiler_status_t>(last_failure.load());
    EXPECT_EQ(failed_starts.load(std::memory_order_relaxed), 0)
        << "last failure: " << rocprofiler_get_status_string(last_status);
    EXPECT_EQ(failed_stops.load(std::memory_order_relaxed), 0)
        << "last failure: " << rocprofiler_get_status_string(last_status);
}
