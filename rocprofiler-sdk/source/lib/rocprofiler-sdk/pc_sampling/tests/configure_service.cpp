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

#include "lib/common/utility.hpp"

#include <gtest/gtest.h>
#include <cstddef>

namespace
{
constexpr size_t BUFFER_SIZE_BYTES = 8192;
constexpr size_t WATERMARK         = (BUFFER_SIZE_BYTES / 4);

#define ROCPROFILER_CALL(ARG, MSG)                                                                 \
    {                                                                                              \
        auto _status = (ARG);                                                                      \
        EXPECT_EQ(_status, ROCPROFILER_STATUS_SUCCESS) << MSG << " :: " << #ARG;                   \
    }

struct callback_data
{
    rocprofiler_client_id_t*                    client_id             = nullptr;
    rocprofiler_client_finalize_t               client_fini_func      = nullptr;
    rocprofiler_context_id_t                    client_ctx            = {0};
    rocprofiler_buffer_id_t                     client_buffer         = {};
    rocprofiler_callback_thread_t               client_thread         = {};
    uint64_t                                    client_workflow_count = {};
    uint64_t                                    client_callback_count = {};
    int64_t                                     current_depth         = 0;
    int64_t                                     max_depth             = 0;
    std::map<uint64_t, rocprofiler_user_data_t> client_correlation    = {};
    std::vector<const rocprofiler_agent_t*>     gpu_pcs_agents        = {};
};

struct agent_data
{
    uint64_t                       agent_count = 0;
    std::vector<hsa_device_type_t> agents      = {};
};

bool
is_pc_sampling_supported(rocprofiler_agent_id_t agent_id)
{
    auto cb = [](const rocprofiler_pc_sampling_configuration_t* configs,
                 size_t                                         num_config,
                 void*                                          user_data) {
        auto* avail_configs =
            static_cast<std::vector<rocprofiler_pc_sampling_configuration_t>*>(user_data);
        // printf("The agent with the id: %lu supports the %lu configurations: \n",
        // agent_id_.handle, num_config);
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
        // PC sampling is not supported
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

    // user_data represent the pointer to the array where gpu_agent will be stored
    if(!user_data) return ROCPROFILER_STATUS_ERROR;

    auto* _out_agents = static_cast<std::vector<const rocprofiler_agent_t*>*>(user_data);
    auto* _agents     = reinterpret_cast<const rocprofiler_agent_t**>(agents);
    for(size_t i = 0; i < num_agents; i++)
    {
        if(_agents[i]->type == ROCPROFILER_AGENT_TYPE_GPU)
        {
            if(is_pc_sampling_supported(_agents[i]->id)) _out_agents->push_back(_agents[i]);

            printf("[%s] %s :: id=%zu, type=%i\n",
                   __FUNCTION__,
                   _agents[i]->name,
                   _agents[i]->id.handle,
                   _agents[i]->type);
        }
        else
        {
            printf("[%s] %s :: id=%zu, type=%i\n",
                   __FUNCTION__,
                   _agents[i]->name,
                   _agents[i]->id.handle,
                   _agents[i]->type);
        }
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_pc_sampling_configuration_t
extract_pc_sampling_config_prefer(rocprofiler_pc_sampling_method_t method,
                                  rocprofiler_agent_id_t           agent_id)
{
    auto cb = [](const rocprofiler_pc_sampling_configuration_t* configs,
                 size_t                                         num_config,
                 void*                                          user_data) {
        auto* avail_configs =
            static_cast<std::vector<rocprofiler_pc_sampling_configuration_t>*>(user_data);
        // printf("The agent with the id: %lu supports the %lu configurations: \n",
        // agent_id_.handle, num_config);
        for(size_t i = 0; i < num_config; i++)
        {
            avail_configs->emplace_back(configs[i]);
        }
        return ROCPROFILER_STATUS_SUCCESS;
    };
    std::vector<rocprofiler_pc_sampling_configuration_t> configs;
    ROCPROFILER_CALL(rocprofiler_query_pc_sampling_agent_configurations(agent_id, cb, &configs),
                     "Failed to query available configurations");

    const rocprofiler_pc_sampling_configuration_t* first_preferred_method_config = nullptr;
    const rocprofiler_pc_sampling_configuration_t* first_remained_method_config  = nullptr;
    // Search until encountering the prefered method configuration, if any.
    // Otherwise, use what remained.
    for(auto const& cfg : configs)
    {
        if(cfg.method == method)
        {
            first_preferred_method_config = &cfg;
            break;
        }
        else if(!first_remained_method_config &&
                cfg.method != ROCPROFILER_PC_SAMPLING_METHOD_NONE &&
                cfg.method != ROCPROFILER_PC_SAMPLING_METHOD_LAST)
        {
            first_remained_method_config = &cfg;
        }
    }

    // Check if the config with the preferred method is found. Use config with other method
    // otherwise.
    const rocprofiler_pc_sampling_configuration_t* picked_cfg =
        (first_preferred_method_config != nullptr) ? first_preferred_method_config
                                                   : first_remained_method_config;

    return *picked_cfg;
}

rocprofiler_pc_sampling_configuration_t
extract_pc_sampling_config_prefer_stochastic(rocprofiler_agent_id_t agent_id)
{
    return extract_pc_sampling_config_prefer(ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC, agent_id);
}

rocprofiler_pc_sampling_configuration_t
extract_pc_sampling_config_prefer_host_trap(rocprofiler_agent_id_t agent_id)
{
    return extract_pc_sampling_config_prefer(ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP, agent_id);
}

void
rocprofiler_pc_sampling_callback(rocprofiler_context_id_t /*context_id*/,
                                 rocprofiler_buffer_id_t /*buffer_id*/,
                                 rocprofiler_record_header_t** /*headers*/,
                                 size_t /*num_headers*/,
                                 void* /*data*/,
                                 uint64_t /*drop_count*/)
{}

void
test_fail_because_of_wrong_agent(const callback_data*                           cb_data,
                                 const rocprofiler_pc_sampling_configuration_t* pcs_config)
{
    auto not_existing_agent = rocprofiler_agent_id_t{0xDEADBEEF};

    EXPECT_EQ(rocprofiler_configure_pc_sampling_service(cb_data->client_ctx,
                                                        not_existing_agent,
                                                        pcs_config->method,
                                                        pcs_config->unit,
                                                        pcs_config->min_interval,
                                                        cb_data->client_buffer,
                                                        0),
              ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND);
}

void
test_fail_because_of_wrong_context(const callback_data*                           cb_data,
                                   rocprofiler_agent_id_t                         agent_id,
                                   const rocprofiler_pc_sampling_configuration_t* pcs_config)
{
    auto not_existing_ctx = rocprofiler_context_id_t{0xDEADBEEF};

    EXPECT_EQ(rocprofiler_configure_pc_sampling_service(not_existing_ctx,
                                                        agent_id,
                                                        pcs_config->method,
                                                        pcs_config->unit,
                                                        pcs_config->min_interval,
                                                        cb_data->client_buffer,
                                                        0),
              ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND);
}

void
test_fail_because_of_wrong_buffer(const callback_data*                           cb_data,
                                  rocprofiler_agent_id_t                         agent_id,
                                  const rocprofiler_pc_sampling_configuration_t* pcs_config)
{
    auto not_existing_buffer_id = rocprofiler_buffer_id_t{0xDEADBEEF};

    EXPECT_EQ(rocprofiler_configure_pc_sampling_service(cb_data->client_ctx,
                                                        agent_id,
                                                        pcs_config->method,
                                                        pcs_config->unit,
                                                        pcs_config->min_interval,
                                                        not_existing_buffer_id,
                                                        0),
              ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND);
}

void
test_fail_because_of_unsupported_configuration(
    const callback_data*                           cb_data,
    rocprofiler_agent_id_t                         agent_id,
    const rocprofiler_pc_sampling_configuration_t* pcs_config)
{
    auto less_than_min_interval    = pcs_config->min_interval - 1;
    auto greater_than_max_interval = pcs_config->max_interval + 1;
    auto wrong_method              = ROCPROFILER_PC_SAMPLING_METHOD_LAST;
    auto wrong_unit                = ROCPROFILER_PC_SAMPLING_UNIT_NONE;

    EXPECT_NE(rocprofiler_configure_pc_sampling_service(cb_data->client_ctx,
                                                        agent_id,
                                                        pcs_config->method,
                                                        pcs_config->unit,
                                                        less_than_min_interval,
                                                        cb_data->client_buffer,
                                                        0),
              ROCPROFILER_STATUS_SUCCESS);

    EXPECT_NE(rocprofiler_configure_pc_sampling_service(cb_data->client_ctx,
                                                        agent_id,
                                                        pcs_config->method,
                                                        pcs_config->unit,
                                                        greater_than_max_interval,
                                                        cb_data->client_buffer,
                                                        0),
              ROCPROFILER_STATUS_SUCCESS);

    EXPECT_NE(rocprofiler_configure_pc_sampling_service(cb_data->client_ctx,
                                                        agent_id,
                                                        wrong_method,
                                                        pcs_config->unit,
                                                        pcs_config->max_interval,
                                                        cb_data->client_buffer,
                                                        0),
              ROCPROFILER_STATUS_SUCCESS);

    EXPECT_NE(rocprofiler_configure_pc_sampling_service(cb_data->client_ctx,
                                                        agent_id,
                                                        pcs_config->method,
                                                        wrong_unit,
                                                        pcs_config->max_interval,
                                                        cb_data->client_buffer,
                                                        0),
              ROCPROFILER_STATUS_SUCCESS);
}

void
test_fail_because_service_is_already_configured(
    const callback_data*                           cb_data,
    rocprofiler_agent_id_t                         agent_id,
    const rocprofiler_pc_sampling_configuration_t* pcs_config)
{
    EXPECT_EQ(rocprofiler_configure_pc_sampling_service(cb_data->client_ctx,
                                                        agent_id,
                                                        pcs_config->method,
                                                        pcs_config->unit,
                                                        pcs_config->min_interval,
                                                        cb_data->client_buffer,
                                                        0),
              ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED);
}

/**
 * @brief Current limitation - Stochastic and Host-Trap PC sampling cannot coexist
 * on the same device simultaneously.
 */
void
test_fail_stochastic_vs_host_trap(const callback_data*                           cb_data,
                                  rocprofiler_agent_id_t                         agent_id,
                                  const rocprofiler_pc_sampling_configuration_t* picked_pcs_config)
{
    // Ensure that stochastic sampling has been configured on the device.
    if(picked_pcs_config->method == ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC)
    {
        // KFD is implemented in the way that if stochastic is configured,
        // no host-trap configuration will be returned (and vice-versa).
        // Thus, ensure that the following function, although prefers host-trap,
        // returns stochastic.
        auto still_stochastic_config = extract_pc_sampling_config_prefer_host_trap(agent_id);
        EXPECT_EQ(still_stochastic_config.method, ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC);

        constexpr uint64_t host_trap_interva_us = 1;
        // Now, ensure that a user cannot still force rocprofiler-sdk and configure host-trap
        // sampling on the device with configured stochastic sampling.
        // ensure that stochastic and host trap sampling cannot coexist on the same device.
        EXPECT_EQ(
            rocprofiler_configure_pc_sampling_service(cb_data->client_ctx,
                                                      agent_id,
                                                      ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP,
                                                      ROCPROFILER_PC_SAMPLING_UNIT_TIME,
                                                      host_trap_interva_us,
                                                      cb_data->client_buffer,
                                                      0),
            ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED);
    }
}

}  // namespace

TEST(pc_sampling, rocprofiler_configure_pc_sampling_service)
{
    using init_func_t = int (*)(rocprofiler_client_finalize_t, void*);
    using fini_func_t = void (*)(void*);

    // using hsa_iterate_agents_cb_t = hsa_status_t (*)(hsa_agent_t, void*);

    auto cmd_line = rocprofiler::common::read_command_line(getpid());
    ASSERT_FALSE(cmd_line.empty());

    static init_func_t tool_init = [](rocprofiler_client_finalize_t fini_func,
                                      void*                         client_data) -> int {
        auto* cb_data = static_cast<callback_data*>(client_data);

        cb_data->client_workflow_count++;
        cb_data->client_fini_func = fini_func;

        // This function returns the all gpu agents supporting some kind of PC sampling
        ROCPROFILER_CALL(
            rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                                               &find_all_gpu_agents_supporting_pc_sampling_impl,
                                               sizeof(rocprofiler_agent_t),
                                               static_cast<void*>(&cb_data->gpu_pcs_agents)),
            "Failed to find GPU agents");

        if(cb_data->gpu_pcs_agents.size() == 0)
        {
            ROCP_ERROR << "PC sampling unavailable\n";
            exit(0);
        }

        ROCPROFILER_CALL(rocprofiler_create_context(&cb_data->client_ctx),
                         "failed to create context");

        ROCPROFILER_CALL(rocprofiler_create_buffer(cb_data->client_ctx,
                                                   BUFFER_SIZE_BYTES,
                                                   WATERMARK,
                                                   ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                                   rocprofiler_pc_sampling_callback,
                                                   client_data,
                                                   &cb_data->client_buffer),
                         "buffer creation failed");

        // We will create another context and try configuring pc sampling inside it,
        // that is supposed to fail.
        rocprofiler_context_id_t another_ctx{0};
        ROCPROFILER_CALL(rocprofiler_create_context(&another_ctx), "failed to create context");
        rocprofiler_buffer_id_t another_buff;
        ROCPROFILER_CALL(rocprofiler_create_buffer(another_ctx,
                                                   4096,
                                                   2048,
                                                   ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                                   rocprofiler_pc_sampling_callback,
                                                   nullptr,
                                                   &another_buff),
                         "buffer creation failed");

        for(const auto* agent : cb_data->gpu_pcs_agents)
        {
            const auto agent_id   = agent->id;
            const auto pcs_config = extract_pc_sampling_config_prefer_stochastic(agent_id);

            test_fail_because_of_wrong_agent(cb_data, &pcs_config);
            test_fail_because_of_wrong_context(cb_data, agent_id, &pcs_config);
            test_fail_because_of_wrong_buffer(cb_data, agent_id, &pcs_config);
            test_fail_because_of_unsupported_configuration(cb_data, agent_id, &pcs_config);

            size_t interval = pcs_config.max_interval;

            // This calls succeeds
            ROCPROFILER_CALL(rocprofiler_configure_pc_sampling_service(cb_data->client_ctx,
                                                                       agent_id,
                                                                       pcs_config.method,
                                                                       pcs_config.unit,
                                                                       interval,
                                                                       cb_data->client_buffer,
                                                                       0),
                             "Failed to configure PC sampling service");

            test_fail_because_service_is_already_configured(cb_data, agent_id, &pcs_config);
            test_fail_stochastic_vs_host_trap(cb_data, agent_id, &pcs_config);

            // Cannot create PC sampling service in context different than the `cb_data->client_ctx`
            EXPECT_EQ(rocprofiler_configure_pc_sampling_service(another_ctx,
                                                                agent_id,
                                                                pcs_config.method,
                                                                pcs_config.unit,
                                                                interval,
                                                                another_buff,
                                                                0),
                      ROCPROFILER_STATUS_ERROR);
        }

        ROCPROFILER_CALL(rocprofiler_create_callback_thread(&cb_data->client_thread),
                         "failure creating callback thread");

        ROCPROFILER_CALL(
            rocprofiler_assign_callback_thread(cb_data->client_buffer, cb_data->client_thread),
            "failed to assign thread for buffer");

        int valid_ctx = 0;
        ROCPROFILER_CALL(rocprofiler_context_is_valid(cb_data->client_ctx, &valid_ctx),
                         "failure checking context validity");

        EXPECT_EQ(valid_ctx, 1);

        ROCPROFILER_CALL(rocprofiler_start_context(cb_data->client_ctx),
                         "rocprofiler context start failed");

        // no errors
        return 0;
    };

    static fini_func_t tool_fini = [](void* client_data) -> void {
        auto* cb_data = static_cast<callback_data*>(client_data);
        ROCPROFILER_CALL(rocprofiler_stop_context(cb_data->client_ctx),
                         "rocprofiler context stop failed");

        static_cast<callback_data*>(client_data)->client_workflow_count++;
    };

    static auto cb_data = callback_data{};

    static auto cfg_result =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            tool_init,
                                            tool_fini,
                                            static_cast<void*>(&cb_data)};

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
        cb_data.client_id       = client_id;
        cb_data.client_id->name = ::testing::UnitTest::GetInstance()->current_test_info()->name();

        return &cfg_result;
    };

    EXPECT_EQ(rocprofiler_force_configure(rocp_init), ROCPROFILER_STATUS_SUCCESS);

    // Further tests assumes the existence of at least one GPU agent supporting
    if(cb_data.gpu_pcs_agents.size() == 0) return;

    const auto* agent = cb_data.gpu_pcs_agents.at(0);
    EXPECT_EQ(rocprofiler_configure_pc_sampling_service(cb_data.client_ctx,
                                                        agent->id,
                                                        ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP,
                                                        ROCPROFILER_PC_SAMPLING_UNIT_TIME,
                                                        1,
                                                        cb_data.client_buffer,
                                                        0),
              ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED);
}
