// MIT License
//
// Copyright (c) 2023-2025 ROCm Developer Tools
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

#include "lib/common/utility.hpp"

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/internal_threading.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <gtest/gtest.h>
#include <cstddef>
#include <vector>

namespace
{
constexpr size_t BUFFER_SIZE_BYTES = 8192;
constexpr size_t WATERMARK         = (BUFFER_SIZE_BYTES / 4);

#define ROCPROFILER_CALL(ARG, MSG)                                                                 \
    {                                                                                              \
        auto _status = (ARG);                                                                      \
        EXPECT_EQ(_status, ROCPROFILER_STATUS_SUCCESS) << MSG << " :: " << #ARG;                   \
    }

struct context_info
{
    rocprofiler_context_id_t context_id = {0};
    rocprofiler_buffer_id_t  buffer_id  = {0};
};

struct test_data
{
    std::vector<const rocprofiler_agent_t*> gpu_pcs_agents = {};
    std::vector<context_info>               contexts       = {};
};

bool
is_pc_sampling_supported(rocprofiler_agent_id_t agent_id)
{
    auto cb = [](const rocprofiler_pc_sampling_configuration_t* configs,
                 size_t                                         num_config,
                 void*                                          user_data) {
        auto* avail_configs =
            static_cast<std::vector<rocprofiler_pc_sampling_configuration_t>*>(user_data);
        for(size_t i = 0; i < num_config; i++)
        {
            avail_configs->emplace_back(configs[i]);
        }
        return ROCPROFILER_STATUS_SUCCESS;
    };

    std::vector<rocprofiler_pc_sampling_configuration_t> configs;
    auto status = rocprofiler_query_pc_sampling_agent_configurations(agent_id, cb, &configs);

    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        return false;
    }
    else if(configs.size() > 0)
    {
        return true;
    }
    else
    {
        return false;
    }
}

rocprofiler_status_t
find_all_gpu_agents_supporting_pc_sampling_impl(rocprofiler_agent_version_t version,
                                                const void**                agents,
                                                size_t                      num_agents,
                                                void*                       user_data)
{
    EXPECT_EQ(version, ROCPROFILER_AGENT_INFO_VERSION_0);

    if(!user_data) return ROCPROFILER_STATUS_ERROR;

    auto* _out_agents = static_cast<std::vector<const rocprofiler_agent_t*>*>(user_data);
    auto* _agents     = reinterpret_cast<const rocprofiler_agent_t**>(agents);
    for(size_t i = 0; i < num_agents; i++)
    {
        if(_agents[i]->type == ROCPROFILER_AGENT_TYPE_GPU)
        {
            if(is_pc_sampling_supported(_agents[i]->id))
            {
                _out_agents->push_back(_agents[i]);
                printf("[multi_context] GPU agent %s (id=%zu) supports PC sampling\n",
                       _agents[i]->name,
                       _agents[i]->id.handle);
            }
        }
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_pc_sampling_configuration_t
extract_pc_sampling_config(rocprofiler_agent_id_t agent_id)
{
    auto cb = [](const rocprofiler_pc_sampling_configuration_t* configs,
                 size_t                                         num_config,
                 void*                                          user_data) {
        auto* avail_configs =
            static_cast<std::vector<rocprofiler_pc_sampling_configuration_t>*>(user_data);
        for(size_t i = 0; i < num_config; i++)
        {
            avail_configs->emplace_back(configs[i]);
        }
        return ROCPROFILER_STATUS_SUCCESS;
    };

    std::vector<rocprofiler_pc_sampling_configuration_t> configs;
    ROCPROFILER_CALL(rocprofiler_query_pc_sampling_agent_configurations(agent_id, cb, &configs),
                     "Failed to query available configurations");

    EXPECT_GT(configs.size(), 0) << "No PC sampling configurations available";

    // Return the first valid configuration
    for(const auto& cfg : configs)
    {
        if(cfg.method != ROCPROFILER_PC_SAMPLING_METHOD_NONE &&
           cfg.method != ROCPROFILER_PC_SAMPLING_METHOD_LAST)
        {
            return cfg;
        }
    }

    return configs[0];
}

void
rocprofiler_pc_sampling_callback(rocprofiler_context_id_t /*context_id*/,
                                 rocprofiler_buffer_id_t /*buffer_id*/,
                                 rocprofiler_record_header_t** /*headers*/,
                                 size_t /*num_headers*/,
                                 void* /*data*/,
                                 uint64_t /*drop_count*/)
{}

}  // namespace

TEST(pc_sampling, multiple_contexts_one_per_agent)
{
    // The following test ensures that multiple contexts with PC sampling can be configured
    // with the constraint that at most one context per agent is allowed.
    using init_func_t = int (*)(rocprofiler_client_finalize_t, void*);
    using fini_func_t = void (*)(void*);

    auto cmd_line = rocprofiler::common::read_command_line(getpid());
    ASSERT_FALSE(cmd_line.empty());

    static init_func_t tool_init = [](rocprofiler_client_finalize_t fini_func,
                                      void*                         client_data) -> int {
        auto* data = static_cast<test_data*>(client_data);
        (void) fini_func;

        // Find all GPU agents that support PC sampling
        ROCPROFILER_CALL(
            rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                                               &find_all_gpu_agents_supporting_pc_sampling_impl,
                                               sizeof(rocprofiler_agent_t),
                                               static_cast<void*>(&data->gpu_pcs_agents)),
            "Failed to find GPU agents");

        if(data->gpu_pcs_agents.size() == 0)
        {
            ROCP_ERROR << "PC sampling unavailable, skipping test\n";
            return 0;
        }

        printf("[multi_context] Found %zu GPU agent(s) supporting PC sampling\n",
               data->gpu_pcs_agents.size());

        // Create as many contexts and buffers as there are agents
        data->contexts.resize(data->gpu_pcs_agents.size());
        for(size_t i = 0; i < data->gpu_pcs_agents.size(); i++)
        {
            ROCPROFILER_CALL(rocprofiler_create_context(&data->contexts[i].context_id),
                             "Failed to create context");

            ROCPROFILER_CALL(rocprofiler_create_buffer(data->contexts[i].context_id,
                                                       BUFFER_SIZE_BYTES,
                                                       WATERMARK,
                                                       ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                                       rocprofiler_pc_sampling_callback,
                                                       nullptr,
                                                       &data->contexts[i].buffer_id),
                             "Buffer creation failed");

            printf("[multi_context] Created context %zu (id=%zu)\n",
                   i,
                   data->contexts[i].context_id.handle);
        }

        // Phase 1: Assign one unique agent to each context (all should succeed)
        printf("\n[multi_context] Phase 1: Assigning one unique agent to each context\n");
        for(size_t i = 0; i < data->gpu_pcs_agents.size(); i++)
        {
            const auto* agent      = data->gpu_pcs_agents[i];
            const auto  agent_id   = agent->id;
            const auto  pcs_config = extract_pc_sampling_config(agent_id);

            printf("[multi_context] Configuring context %zu with agent %zu (%s)... ",
                   i,
                   agent_id.handle,
                   agent->name);

            auto status = rocprofiler_configure_pc_sampling_service(data->contexts[i].context_id,
                                                                    agent_id,
                                                                    pcs_config.method,
                                                                    pcs_config.unit,
                                                                    pcs_config.min_interval,
                                                                    data->contexts[i].buffer_id,
                                                                    0);

            EXPECT_EQ(status, ROCPROFILER_STATUS_SUCCESS)
                << "Failed to configure PC sampling for context " << i << " with agent "
                << agent_id.handle;
        }

        // Phase 2: Try to add remaining agents to each context (all should fail)
        printf("\n[multi_context] Phase 2: Attempting to add already-configured agents to contexts "
               "(should all fail)\n");
        for(size_t ctx_idx = 0; ctx_idx < data->contexts.size(); ctx_idx++)
        {
            for(size_t agent_idx = 0; agent_idx < data->gpu_pcs_agents.size(); agent_idx++)
            {
                // Skip the agent that's already assigned to this context
                if(ctx_idx == agent_idx) continue;

                const auto* agent      = data->gpu_pcs_agents[agent_idx];
                const auto  agent_id   = agent->id;
                const auto  pcs_config = extract_pc_sampling_config(agent_id);

                printf("[multi_context] Trying to configure context %zu with agent %zu (%s) "
                       "(already owned by context %zu)... ",
                       ctx_idx,
                       agent_id.handle,
                       agent->name,
                       agent_idx);

                auto status =
                    rocprofiler_configure_pc_sampling_service(data->contexts[ctx_idx].context_id,
                                                              agent_id,
                                                              pcs_config.method,
                                                              pcs_config.unit,
                                                              pcs_config.min_interval,
                                                              data->contexts[ctx_idx].buffer_id,
                                                              0);

                EXPECT_EQ(status, ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED)
                    << "Expected ERROR_SERVICE_ALREADY_CONFIGURED when trying to configure agent "
                    << agent_id.handle << " (" << agent->name << ") in context "
                    << data->contexts[ctx_idx].context_id.handle << " (already owned by context "
                    << data->contexts[agent_idx].context_id.handle << ")";
            }
        }

        printf("\n[multi_context] Test completed\n");

        // Start all contexts
        for(size_t i = 0; i < data->contexts.size(); i++)
        {
            ROCPROFILER_CALL(rocprofiler_start_context(data->contexts[i].context_id),
                             "Failed to start context");
        }

        return 0;
    };

    static fini_func_t tool_fini = [](void* client_data) -> void {
        auto* data = static_cast<test_data*>(client_data);

        // Stop all contexts
        for(size_t i = 0; i < data->contexts.size(); i++)
        {
            ROCPROFILER_CALL(rocprofiler_stop_context(data->contexts[i].context_id),
                             "Failed to stop context");
        }
    };

    static auto test_data_instance = test_data{};

    static auto cfg_result =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            tool_init,
                                            tool_fini,
                                            static_cast<void*>(&test_data_instance)};

    static rocprofiler_configure_func_t rocp_init =
        [](uint32_t                 version,
           const char*              runtime_version,
           uint32_t                 prio,
           rocprofiler_client_id_t* client_id) -> rocprofiler_tool_configure_result_t* {
        auto expected_version = ROCPROFILER_VERSION;
        EXPECT_EQ(expected_version, version);
        EXPECT_EQ(std::string_view{runtime_version}, std::string_view{ROCPROFILER_VERSION_STRING});
        EXPECT_EQ(prio, 0);
        EXPECT_EQ(client_id->name, nullptr);
        client_id->name = ::testing::UnitTest::GetInstance()->current_test_info()->name();

        return &cfg_result;
    };

    EXPECT_EQ(rocprofiler_force_configure(rocp_init), ROCPROFILER_STATUS_SUCCESS);
}
