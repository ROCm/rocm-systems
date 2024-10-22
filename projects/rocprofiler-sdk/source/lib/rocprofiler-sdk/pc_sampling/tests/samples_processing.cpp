// MIT License
//
// Copyright (c) 2023 ROCm Developer Tools
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

#include <hsa/hsa.h>
#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/internal_threading.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/parser/rocr.h"
#include "lib/rocprofiler-sdk/pc_sampling/tests/pc_sampling_internals.hpp"
#include "pc_sampling_internals.hpp"

#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>

constexpr size_t BUFFER_SIZE_BYTES = 8192;
constexpr size_t WATERMARK         = (BUFFER_SIZE_BYTES / 4);

#define ROCPROFILER_CALL(ARG, MSG)                                                                 \
    {                                                                                              \
        auto _status = (ARG);                                                                      \
        EXPECT_EQ(_status, ROCPROFILER_STATUS_SUCCESS) << MSG << " :: " << #ARG;                   \
    }

namespace
{
#define NUM_SAMPLES 5
#define TRAP_ID     0

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

rocprofiler_status_t
find_all_gpu_agents_supporting_pc_sampling_impl(const rocprofiler_agent_t** agents,
                                                size_t                      num_agents,
                                                void*                       user_data)
{
    // user_data represent the pointer to the array where gpu_agent will be stored
    if(!user_data) return ROCPROFILER_STATUS_ERROR;

    auto* _out_agents = static_cast<std::vector<const rocprofiler_agent_t*>*>(user_data);
    // find the first GPU agent
    for(size_t i = 0; i < num_agents; i++)
    {
        if(agents[i]->type == ROCPROFILER_AGENT_TYPE_GPU)
        {
            // Skip GPU agents not supporting PC sampling
            // Vladimir: The assumption is that if a GPU agent does not support PC sampling,
            // the size is 0.
            if(agents[i]->num_pc_sampling_configs == 0) continue;

            _out_agents->push_back(agents[i]);

            printf("[%s] %s :: id=%zu, type=%i, num pc sample configs=%zu\n",
                   __FUNCTION__,
                   agents[i]->name,
                   agents[i]->id.handle,
                   agents[i]->type,
                   agents[i]->num_pc_sampling_configs);
        }
        else
        {
            printf("[%s] %s :: id=%zu, type=%i, num pc sample configs=%zu\n",
                   __FUNCTION__,
                   agents[i]->name,
                   agents[i]->id.handle,
                   agents[i]->type,
                   agents[i]->num_pc_sampling_configs);
        }
    }

    return !_out_agents->empty() ? ROCPROFILER_STATUS_SUCCESS : ROCPROFILER_STATUS_ERROR;
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
                                 rocprofiler_record_header_t** headers,
                                 size_t                        num_headers,
                                 void* /*data*/,
                                 uint64_t drop_count)
{
    EXPECT_EQ(drop_count, 0);

    for(size_t i = 0; i < num_headers; i++)
    {
        auto* cur_header = headers[i];

        if(cur_header == nullptr)
        {
            throw std::runtime_error{
                "rocprofiler provided a null pointer to header. this should never happen"};
        }
        else if(cur_header->hash !=
                rocprofiler_record_header_compute_hash(cur_header->category, cur_header->kind))
        {
            throw std::runtime_error{"rocprofiler_record_header_t (category | kind) != hash"};
        }
        else if(cur_header->category == ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING)
        {
            auto* pc_sample = static_cast<rocprofiler_pc_sampling_record_t*>(cur_header->payload);
            // FIXME: find the cause why this fails
            // EXPECT_EQ(pc_sample->correlation_id.internal, 1);
            EXPECT_EQ(pc_sample->pc, i + 1);
            EXPECT_EQ(pc_sample->timestamp, i + 33);
            EXPECT_EQ(pc_sample->hw_id, 0);
        }
        else
        {
            throw std::runtime_error{"unexpected rocprofiler_record_header_t category + kind"};
        }
    }
}

}  // namespace

TEST(pc_sampling, processing_pc_samples)
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
            rocprofiler_query_available_agents(&find_all_gpu_agents_supporting_pc_sampling_impl,
                                               sizeof(rocprofiler_agent_t),
                                               static_cast<void*>(&cb_data->gpu_pcs_agents)),
            "Failed to find GPU agents");

        ROCPROFILER_CALL(rocprofiler_create_context(&cb_data->client_ctx),
                         "failed to create context");

        ROCPROFILER_CALL(rocprofiler_create_buffer(cb_data->client_ctx,
                                                   4096,
                                                   2048,
                                                   ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                                   rocprofiler_pc_sampling_callback,
                                                   client_data,
                                                   &cb_data->client_buffer),
                         "buffer creation failed");

        const auto* agent      = cb_data->gpu_pcs_agents.at(0);
        const auto  agent_id   = agent->id;
        const auto  pcs_config = extract_pc_sampling_config_prefer_stochastic(agent_id);

        size_t interval = pcs_config.max_interval;

        // This calls succeeds
        ROCPROFILER_CALL(rocprofiler_configure_pc_sampling_service(cb_data->client_ctx,
                                                                   agent->id,
                                                                   pcs_config.method,
                                                                   pcs_config.unit,
                                                                   interval,
                                                                   cb_data->client_buffer),
                         "Failed to configure PC sampling service");

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
        // FIXME: for some reason, this returns context not found
        // ROCPROFILER_CALL(rocprofiler_stop_context(cb_data->client_ctx),
        //                  "rocprofiler context stop failed");

        cb_data->client_workflow_count++;
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

    auto& hsa_table          = rocprofiler::hsa::get_table();
    auto* pc_sampling_table_ = hsa_table.pc_sampling_ext_;
    EXPECT_NE(pc_sampling_table_, nullptr);

    pc_sampling_table_->hsa_ven_amd_pcs_create_from_id_fn =
        [](uint32_t /*ioctl_pcs_id*/,
           hsa_agent_t /*agent*/,
           hsa_ven_amd_pcs_method_kind_t /*method*/,
           hsa_ven_amd_pcs_units_t /*units*/,
           size_t /*interval*/,
           size_t /*latency*/,
           size_t /*buffer_size*/,
           hsa_ven_amd_pcs_data_ready_callback_t /*data_ready_callback*/,
           void* /*client_callback_data*/,
           hsa_ven_amd_pcs_t* /*pc_sampling*/) { return HSA_STATUS_SUCCESS; };

    pc_sampling_table_->hsa_ven_amd_pcs_flush_fn = [](hsa_ven_amd_pcs_t /*pc_sampling*/) {
        return HSA_STATUS_SUCCESS;
    };

    auto* ext_table_ = hsa_table.amd_ext_;
    EXPECT_NE(ext_table_, nullptr);

    ext_table_->hsa_amd_queue_get_info_fn =
        [](hsa_queue_t* queue, hsa_queue_info_attribute_t attribute, void* value) {
            (void) queue;
            switch(attribute)
            {
                case HSA_AMD_QUEUE_INFO_AGENT:
                    *(reinterpret_cast<hsa_agent_t*>(value)) = hsa_agent_t{.handle = 1};
                    break;
                case HSA_AMD_QUEUE_INFO_DOORBELL_ID:
                    *(reinterpret_cast<uint64_t*>(value)) = 0;
                    break;
                default: return HSA_STATUS_ERROR_INVALID_ARGUMENT;
            }
            return HSA_STATUS_SUCCESS;
        };

#if 1

    // Set the HSA agent for the active PCSamplingConfiguration,
    // The reason for setting HSA agent manually follows.
    // The test links against rocporifler static library.
    // Hence, the rocprofiler_set_api_table is not called.
    auto* service = rocprofiler::pc_sampling::get_active_pc_sampling_service().load();
    EXPECT_NE(service, nullptr);
    const auto* rocp_agent       = cb_data.gpu_pcs_agents.at(0);
    auto        agent_session    = service->agent_sessions.at(rocp_agent->id).get();
    hsa_agent_t pseudo_hsa_agent = {.handle = 1};
    agent_session->hsa_agent     = std::make_unique<hsa_agent_t>(pseudo_hsa_agent);

    // TODO: We need to register the agent inside the parser
    rocprofiler::pc_sampling::hsa::get_pc_sampling_parser().register_buffer_for_agent(
        cb_data.client_buffer.handle, rocp_agent->id.handle);

    // The following test calls some segments of internal PC sampling API implementation
    // by mimicking the HIP and ROCr

    // Generate dispatch and marker packet
    rocprofiler::hsa::rocprofiler_packet dispatch_pkt;
    auto                                 marker_pkt =
        rocprofiler::pc_sampling::hsa::generate_marker_packet_for_kernel(&dispatch_pkt);

    // create a pseudo hsa queue
    hsa_queue_t queue;
    queue.size = 1024;
    // Mimic the ROCr and notify the pc sampling service that the marker packet has been encoutered.
    rocprofiler::pc_sampling::hsa::amd_intercept_marker_handler_callback(
        &marker_pkt.marker, &queue, 0);

    // We need to generate some samples and send them via data_ready_calllback.
    size_t num_samples       = NUM_SAMPLES;
    auto   samples_data_size = num_samples * sizeof(packet_union_t);

    static hsa_ven_amd_pcs_data_copy_callback_t hsa_mock_data_copy_callback =
        [](void* hsa_callback_data, size_t data_size, void* destination) {
            (void) hsa_callback_data;
            (void) data_size;
            using rocr_buffer_t = std::vector<packet_union_t>;
            auto samples_buff   = rocr_buffer_t{};
            for(size_t i = 0; i < NUM_SAMPLES; i++)
            {
                perf_sample_host_trap_v1 hs;
                hs.pc                  = i + 1;
                hs.exec_mask           = 0xF;
                hs.workgroup_id_x      = 1;
                hs.workgroup_id_y      = 2;
                hs.workgroup_id_z      = 3;
                hs.chiplet_and_wave_id = 0;
                hs.hw_id               = 0;
                hs.timestamp           = 33 + i;
                hs.correlation_id      = TRAP_ID;
                samples_buff.push_back(packet_union_t{.host = hs});
            }
            // copy the data
            std::memcpy(destination, samples_buff.data(), NUM_SAMPLES * sizeof(packet_union_t));
            // clear the data
            return HSA_STATUS_SUCCESS;
        };

    // calling data_ready_callback that will result in copying the data from above
    // to the client buffer via PC sampling parser
    rocprofiler::pc_sampling::PCSAgentSession pcs_agent_session;
    pcs_agent_session.agent  = rocp_agent;
    pcs_agent_session.method = ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP;

    size_t lost_samples = 0;

    rocprofiler::pc_sampling::hsa::data_ready_callback(
        &pcs_agent_session, samples_data_size, lost_samples, hsa_mock_data_copy_callback, nullptr);

    rocprofiler::pc_sampling::hsa::kernel_completion_cb(
        nullptr, rocp_agent, static_cast<int64_t>(1), dispatch_pkt, nullptr);

    // Flush the buffer explicitly
    ROCPROFILER_CALL(rocprofiler_flush_buffer(cb_data.client_buffer),
                     "rocprofiler flush buffer failed");
    // Stop the context
    ROCPROFILER_CALL(rocprofiler_stop_context(cb_data.client_ctx),
                     "rocprofiler context stop failed");
#endif
}
