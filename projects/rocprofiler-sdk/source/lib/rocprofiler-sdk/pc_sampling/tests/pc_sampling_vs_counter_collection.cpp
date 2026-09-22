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
#include <functional>

namespace
{
constexpr size_t BUFFER_SIZE_BYTES = 8192;
constexpr size_t WATERMARK         = (BUFFER_SIZE_BYTES / 4);

#define ROCPROFILER_CALL(ARG, MSG)                                                                 \
    {                                                                                              \
        auto _status = (ARG);                                                                      \
        EXPECT_EQ(_status, ROCPROFILER_STATUS_SUCCESS) << MSG << " :: " << #ARG;                   \
    }

using cc_setup_fn_t = std::function<void(rocprofiler_context_id_t, rocprofiler_agent_id_t)>;

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
    cc_setup_fn_t                               cc_setup_fn           = {};
};

struct agent_data
{
    uint64_t                       agent_count = 0;
    std::vector<hsa_device_type_t> agents      = {};
};

// =========================== Functions related to the PC sampling service

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

const rocprofiler_pc_sampling_configuration_t
extract_pc_sampling_config_prefer_stochastic(rocprofiler_agent_id_t agent_id)
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

    const rocprofiler_pc_sampling_configuration_t* first_host_trap_config  = nullptr;
    const rocprofiler_pc_sampling_configuration_t* first_stochastic_config = nullptr;
    // Search until encountering on the stochastic configuration, if any.
    // Otherwise, use the host trap config
    for(auto const& cfg : configs)
    {
        if(cfg.method == ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC)
        {
            first_stochastic_config = &cfg;
            break;
        }
        else if(!first_host_trap_config && cfg.method == ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP)
        {
            first_host_trap_config = &cfg;
        }
    }

    // Check if the stochastic config is found. Use host trap config otherwise.
    const rocprofiler_pc_sampling_configuration_t* picked_cfg =
        (first_stochastic_config != nullptr) ? first_stochastic_config : first_host_trap_config;

    return *picked_cfg;
}

void
rocprofiler_pc_sampling_callback(rocprofiler_context_id_t /*context_id*/,
                                 rocprofiler_buffer_id_t /*buffer_id*/,
                                 rocprofiler_record_header_t** /*headers*/,
                                 size_t /*num_headers*/,
                                 void* /*data*/,
                                 uint64_t /*drop_count*/)
{}

// =================== Functions related to the counter collection service
void
record_callback(rocprofiler_dispatch_counting_service_data_t /*dispatch_data*/,
                rocprofiler_record_counter_t* /*record_data*/,
                size_t /*record_count*/,
                rocprofiler_user_data_t /*user_data*/,
                void* /*callback_data_args*/)
{}

void
dispatch_callback(rocprofiler_dispatch_counting_service_data_t /*dispatch_data*/,
                  rocprofiler_counter_config_id_t* /*config*/,
                  rocprofiler_user_data_t* /*user_data*/,
                  void* /*callback_data_args*/)
{}

void
set_profile(rocprofiler_context_id_t /*context_id*/,
            rocprofiler_agent_id_t /*agent*/,
            rocprofiler_device_counting_agent_cb_t /*set_config*/,
            void*)
{}

void
rocprofiler_counter_collection_callback(rocprofiler_context_id_t /*context_id*/,
                                        rocprofiler_buffer_id_t /*buffer_id*/,
                                        rocprofiler_record_header_t** /*headers*/,
                                        size_t /*num_headers*/,
                                        void* /*data*/,
                                        uint64_t /*drop_count*/)
{}

using cc_setup_fn_t = std::function<void(rocprofiler_context_id_t, rocprofiler_agent_id_t)>;

void
pc_sampling_vs_counter_collection(cc_setup_fn_t cc_setup_fn)
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
        EXPECT_EQ(
            rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                                               &find_all_gpu_agents_supporting_pc_sampling_impl,
                                               sizeof(rocprofiler_agent_t),
                                               static_cast<void*>(&cb_data->gpu_pcs_agents)),
            ROCPROFILER_STATUS_SUCCESS);

        if(cb_data->gpu_pcs_agents.size() == 0)
        {
            ROCP_ERROR << "PC sampling unavailable\n";
            exit(0);
        }

        EXPECT_EQ(rocprofiler_create_context(&cb_data->client_ctx), ROCPROFILER_STATUS_SUCCESS);

        // Create PC sampling buffer
        EXPECT_EQ(rocprofiler_create_buffer(cb_data->client_ctx,
                                            BUFFER_SIZE_BYTES,
                                            WATERMARK,
                                            ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                            rocprofiler_pc_sampling_callback,
                                            client_data,
                                            &cb_data->client_buffer),
                  ROCPROFILER_STATUS_SUCCESS);

        // Configure counter collection service.
        // Agent counter collection service is configured on the first listed GPU device.
        cb_data->cc_setup_fn(cb_data->client_ctx, cb_data->gpu_pcs_agents.at(0)->id);

        // Configuring PC sampling service should fail
        for(const auto* agent : cb_data->gpu_pcs_agents)
        {
            const auto agent_id   = agent->id;
            const auto pcs_config = extract_pc_sampling_config_prefer_stochastic(agent_id);

            size_t interval = pcs_config.max_interval;

            // This calls succeeds
            EXPECT_EQ(rocprofiler_configure_pc_sampling_service(cb_data->client_ctx,
                                                                agent_id,
                                                                pcs_config.method,
                                                                pcs_config.unit,
                                                                interval,
                                                                cb_data->client_buffer,
                                                                0),
                      ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT);
        }

        // no errors
        return 0;
    };

    static fini_func_t tool_fini = [](void* client_data) -> void {
        auto* cb_data = static_cast<callback_data*>(client_data);
        EXPECT_EQ(rocprofiler_stop_context(cb_data->client_ctx), ROCPROFILER_STATUS_SUCCESS);

        static_cast<callback_data*>(client_data)->client_workflow_count++;
    };

    static auto cb_data = callback_data{};
    cb_data.cc_setup_fn = cc_setup_fn;

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
}

void
counter_collection_vs_pc_sampling(cc_setup_fn_t cc_setup_fn)
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
        EXPECT_EQ(
            rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                                               &find_all_gpu_agents_supporting_pc_sampling_impl,
                                               sizeof(rocprofiler_agent_t),
                                               static_cast<void*>(&cb_data->gpu_pcs_agents)),
            ROCPROFILER_STATUS_SUCCESS);

        if(cb_data->gpu_pcs_agents.size() == 0)
        {
            ROCP_ERROR << "PC sampling unavailable\n";
            exit(0);
        }

        EXPECT_EQ(rocprofiler_create_context(&cb_data->client_ctx), ROCPROFILER_STATUS_SUCCESS);

        // Create PC sampling buffer
        EXPECT_EQ(rocprofiler_create_buffer(cb_data->client_ctx,
                                            BUFFER_SIZE_BYTES,
                                            WATERMARK,
                                            ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                            rocprofiler_pc_sampling_callback,
                                            client_data,
                                            &cb_data->client_buffer),
                  ROCPROFILER_STATUS_SUCCESS);

        // Configuring PC sampling service first
        for(const auto* agent : cb_data->gpu_pcs_agents)
        {
            const auto agent_id   = agent->id;
            const auto pcs_config = extract_pc_sampling_config_prefer_stochastic(agent_id);

            size_t interval = pcs_config.max_interval;

            // This calls succeeds
            EXPECT_EQ(rocprofiler_configure_pc_sampling_service(cb_data->client_ctx,
                                                                agent_id,
                                                                pcs_config.method,
                                                                pcs_config.unit,
                                                                interval,
                                                                cb_data->client_buffer,
                                                                0),
                      ROCPROFILER_STATUS_SUCCESS);
        }

        // Configuring counter collection service on the first listed GPU agent should fail
        cb_data->cc_setup_fn(cb_data->client_ctx, cb_data->gpu_pcs_agents.at(0)->id);

        // no errors
        return 0;
    };

    static fini_func_t tool_fini = [](void* client_data) -> void {
        auto* cb_data = static_cast<callback_data*>(client_data);
        EXPECT_EQ(rocprofiler_stop_context(cb_data->client_ctx), ROCPROFILER_STATUS_SUCCESS);

        static_cast<callback_data*>(client_data)->client_workflow_count++;
    };

    static auto cb_data = callback_data{};
    cb_data.cc_setup_fn = cc_setup_fn;

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
}

}  // namespace

TEST(pc_sampling, pc_sampling_vs_dispatch_counter_collection)
{
    auto dispatch_counter_collection_setup_fn = [](rocprofiler_context_id_t context_id,
                                                   rocprofiler_agent_id_t /*agent_id*/) {
        // Configure dispatch counter collection service on all agents
        EXPECT_EQ(rocprofiler_configure_callback_dispatch_counting_service(
                      context_id, dispatch_callback, nullptr, record_callback, nullptr),
                  ROCPROFILER_STATUS_SUCCESS);
    };

    pc_sampling_vs_counter_collection(dispatch_counter_collection_setup_fn);
}

TEST(pc_sampling, pc_sampling_vs_device_counter_collection)
{
    auto device_counter_collection_setup_fn = [](rocprofiler_context_id_t context_id,
                                                 rocprofiler_agent_id_t   agent_id) {
        rocprofiler_buffer_id_t cc_buf_id{};
        // Create PC sampling buffer
        EXPECT_EQ(rocprofiler_create_buffer(context_id,
                                            BUFFER_SIZE_BYTES,
                                            WATERMARK,
                                            ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                            rocprofiler_counter_collection_callback,
                                            nullptr,
                                            &cc_buf_id),
                  ROCPROFILER_STATUS_SUCCESS);

        EXPECT_EQ(rocprofiler_configure_device_counting_service(
                      context_id, cc_buf_id, agent_id, set_profile, nullptr),
                  ROCPROFILER_STATUS_SUCCESS);
    };

    pc_sampling_vs_counter_collection(device_counter_collection_setup_fn);
}

TEST(pc_sampling, dispatch_counter_collection_vs_pc_sampling)
{
    auto dispatch_counter_collection_setup_fn = [](rocprofiler_context_id_t context_id,
                                                   rocprofiler_agent_id_t /*agent_id*/) {
        // Configure dispatch counter collection service on all agents
        EXPECT_EQ(rocprofiler_configure_callback_dispatch_counting_service(
                      context_id, dispatch_callback, nullptr, record_callback, nullptr),
                  ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT);
    };

    counter_collection_vs_pc_sampling(dispatch_counter_collection_setup_fn);
}

TEST(pc_sampling, device_counter_collection_vs_pc_sampling)
{
    auto device_counter_collection_setup_fn = [](rocprofiler_context_id_t context_id,
                                                 rocprofiler_agent_id_t   agent_id) {
        rocprofiler_buffer_id_t cc_buf_id{};
        // Create PC sampling buffer
        EXPECT_EQ(rocprofiler_create_buffer(context_id,
                                            BUFFER_SIZE_BYTES,
                                            WATERMARK,
                                            ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                            rocprofiler_counter_collection_callback,
                                            nullptr,
                                            &cc_buf_id),
                  ROCPROFILER_STATUS_SUCCESS);

        EXPECT_EQ(rocprofiler_configure_device_counting_service(
                      context_id, cc_buf_id, agent_id, set_profile, nullptr),
                  ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT);
    };

    counter_collection_vs_pc_sampling(device_counter_collection_setup_fn);
}
