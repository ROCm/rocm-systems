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

#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "lib/common/defines.hpp"
#include "lib/common/environment.hpp"
#include "lib/common/filesystem.hpp"
#include "lib/common/units.hpp"
#include "lib/common/utility.hpp"

#include <gtest/gtest.h>

#include <dlfcn.h>
#include <pthread.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <string_view>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define ROCPROFILER_CALL(ARG, MSG)                                                                 \
    {                                                                                              \
        auto _status = (ARG);                                                                      \
        EXPECT_EQ(_status, ROCPROFILER_STATUS_SUCCESS) << MSG << " :: " << #ARG;                   \
    }

namespace
{
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
};

struct agent_data
{
    uint64_t                       agent_count = 0;
    std::vector<hsa_device_type_t> agents      = {};
};

void
tool_tracing_callback(rocprofiler_callback_tracing_record_t record,
                      rocprofiler_user_data_t*              user_data,
                      void*                                 client_data)
{
    static auto mtx = std::mutex{};
    auto        lk  = std::unique_lock{mtx};

    auto*       cb_data          = static_cast<callback_data*>(client_data);
    auto        now              = rocprofiler::common::timestamp_ns();
    auto        internal_corr_id = record.correlation_id.internal;
    auto&       external_corr_id = record.correlation_id.external;
    static auto first_now        = now;

    ASSERT_NE(cb_data, nullptr);

    cb_data->client_callback_count++;

    static auto first = std::once_flag{};
    std::call_once(first,
                   [record]() { EXPECT_EQ(record.phase, ROCPROFILER_CALLBACK_PHASE_ENTER); });

    if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
    {
        EXPECT_EQ(cb_data->client_correlation.find(internal_corr_id),
                  cb_data->client_correlation.end())
            << "entry for internal correlation id " << internal_corr_id << " already exists";

        cb_data->client_correlation[internal_corr_id] = external_corr_id;

        user_data->value   = now;
        auto current_depth = cb_data->current_depth++;

        if(current_depth == 0)
        {
            uint64_t tid = 0;
            ROCPROFILER_CALL(rocprofiler_get_thread_id(&tid), "Failed to get thread id");
            EXPECT_EQ(external_corr_id.value, tid);
        }

        ROCPROFILER_CALL(rocprofiler_push_external_correlation_id(
                             record.context_id,
                             record.thread_id,
                             rocprofiler_user_data_t{.value = (internal_corr_id + 1) * 1000}),
                         "Failed to push new external correlation");
    }
    else
    {
        EXPECT_NE(cb_data->client_correlation.find(internal_corr_id),
                  cb_data->client_correlation.end())
            << "entry for internal correlation id " << internal_corr_id << " does not exist";

        EXPECT_EQ(external_corr_id.value, (internal_corr_id + 1) * 1000)
            << "external correlation id change was not retained";

        auto external_corr_data = rocprofiler_user_data_t{};
        ROCPROFILER_CALL(rocprofiler_pop_external_correlation_id(
                             record.context_id, record.thread_id, &external_corr_data),
                         "Failed to pop external correlation");

        EXPECT_EQ(external_corr_data.value, (internal_corr_id + 1) * 1000)
            << "external correlation pop did not return current external correlation";

        EXPECT_GT(user_data->value, 0) << "user data not set";
        EXPECT_GE(user_data->value, first_now) << "timestamp not monotonically increasing";
        EXPECT_LT(user_data->value, now) << "timestamp not monotonically increasing";
        EXPECT_GT(cb_data->current_depth, 0) << "depth should be > 0";

        cb_data->max_depth = std::max(cb_data->current_depth, cb_data->max_depth);
        cb_data->current_depth--;
    }
}

void
tool_tracing_buffered(rocprofiler_context_id_t      context,
                      rocprofiler_buffer_id_t       buffer_id,
                      rocprofiler_record_header_t** headers,
                      size_t                        num_headers,
                      void*                         buffer_data,
                      uint64_t                      drop_count)
{
    std::cout << __FUNCTION__ << "...\n" << std::endl;
    auto* cb_data = static_cast<callback_data*>(buffer_data);

    auto v_records = std::vector<rocprofiler_buffer_tracing_hsa_api_record_t*>{};
    v_records.reserve(num_headers);

    static const auto supported =
        std::unordered_set<uint32_t>{ROCPROFILER_BUFFER_TRACING_HSA_CORE_API,
                                     ROCPROFILER_BUFFER_TRACING_HSA_AMD_EXT_API,
                                     ROCPROFILER_BUFFER_TRACING_HSA_IMAGE_EXT_API,
                                     ROCPROFILER_BUFFER_TRACING_HSA_FINALIZE_EXT_API};

    for(size_t i = 0; i < num_headers; ++i)
    {
        auto* header = headers[i];

        ASSERT_TRUE(header != nullptr);
        auto hash = rocprofiler_record_header_compute_hash(header->category, header->kind);
        EXPECT_EQ(header->hash, hash);
        EXPECT_TRUE(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
                    supported.count(header->kind) != 0);

        v_records.emplace_back(
            static_cast<rocprofiler_buffer_tracing_hsa_api_record_t*>(header->payload));
    }

    std::sort(v_records.begin(), v_records.end(), [](auto lhs, auto rhs) {
        return (lhs->start_timestamp == rhs->start_timestamp)
                   ? (lhs->end_timestamp < rhs->end_timestamp)
                   : (lhs->start_timestamp < rhs->start_timestamp);
    });

    for(auto* record : v_records)
    {
        auto info = std::stringstream{};
        info << "tid=" << record->thread_id << ", context=" << context.handle
             << ", buffer_id=" << buffer_id.handle << ", cid=" << record->correlation_id.internal
             << ", kind=" << record->kind << ", operation=" << record->operation
             << ", drop_count=" << drop_count << ", start=" << record->start_timestamp
             << ", stop=" << record->end_timestamp;

        static int64_t last_corr_id = -1;
        auto           corr_id      = static_cast<int64_t>(record->correlation_id.internal);

        std::cout << info.str() << "\n" << std::flush;
        EXPECT_GE(context.handle, 0) << info.str();
        EXPECT_GT(record->thread_id, 0) << info.str();
        EXPECT_GT(record->kind, 0) << info.str();
        EXPECT_GT(corr_id, last_corr_id) << info.str();
        EXPECT_GT(record->start_timestamp, 0) << info.str();
        EXPECT_GT(record->end_timestamp, 0) << info.str();
        EXPECT_LE(record->start_timestamp, record->end_timestamp) << info.str();

        cb_data->client_callback_count++;
        last_corr_id = corr_id;
    }
}

void
thread_precreate(rocprofiler_runtime_library_t /*lib*/, void* tool_data)
{
    auto* cb_data = static_cast<callback_data*>(tool_data);
    cb_data->client_workflow_count++;
}

void
thread_postcreate(rocprofiler_runtime_library_t /*lib*/, void* tool_data)
{
    auto* cb_data = static_cast<callback_data*>(tool_data);
    cb_data->client_workflow_count++;
}
}  // namespace

TEST(rocprofiler_lib, callback_external_correlation)
{
    using init_func_t = int (*)(rocprofiler_client_finalize_t, void*);
    using fini_func_t = void (*)(void*);

    using hsa_iterate_agents_cb_t = hsa_status_t (*)(hsa_agent_t, void*);

    auto cmd_line = rocprofiler::common::read_command_line(getpid());
    ASSERT_FALSE(cmd_line.empty());

    static init_func_t tool_init = [](rocprofiler_client_finalize_t fini_func,
                                      void*                         client_data) -> int {
        auto* cb_data = static_cast<callback_data*>(client_data);

        cb_data->client_workflow_count++;
        cb_data->client_fini_func = fini_func;

        ROCPROFILER_CALL(rocprofiler_create_context(&cb_data->client_ctx),
                         "failed to create context");

        for(auto itr : {ROCPROFILER_CALLBACK_TRACING_HSA_CORE_API,
                        ROCPROFILER_CALLBACK_TRACING_HSA_AMD_EXT_API,
                        ROCPROFILER_CALLBACK_TRACING_HSA_IMAGE_EXT_API,
                        ROCPROFILER_CALLBACK_TRACING_HSA_FINALIZE_EXT_API})
        {
            ROCPROFILER_CALL(
                rocprofiler_configure_callback_tracing_service(
                    cb_data->client_ctx, itr, nullptr, 0, tool_tracing_callback, client_data),
                "callback tracing service failed to configure");
        }

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
        int   status  = 0;
        ROCPROFILER_CALL(rocprofiler_context_is_active(cb_data->client_ctx, &status),
                         "rocprofiler_context_is_active failed");
        EXPECT_EQ(status, 0);

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

    uint64_t tid = 0;
    ROCPROFILER_CALL(rocprofiler_get_thread_id(&tid), "failed to get thread id");

    ROCPROFILER_CALL(rocprofiler_push_external_correlation_id(
                         cb_data.client_ctx, tid, rocprofiler_user_data_t{.value = tid}),
                     "failed to push correlation id");

    hsa_iterate_agents_cb_t agent_cb = [](hsa_agent_t agent, void* data) {
        static_cast<agent_data*>(data)->agent_count++;

        auto status     = HSA_STATUS_SUCCESS;
        auto agent_type = hsa_device_type_t{};
        if((status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &agent_type)) ==
           HSA_STATUS_SUCCESS)
            static_cast<agent_data*>(data)->agents.emplace_back(agent_type);

        return status;
    };

    auto     _agent_data = agent_data{};
    uint64_t num_runs    = 0;
    hsa_init();
    auto run = [&agent_cb, &_agent_data, &num_runs]() {
        ++num_runs;
        uint64_t _tid = 0;
        ROCPROFILER_CALL(rocprofiler_get_thread_id(&_tid), "failed to get thread id");
        ROCPROFILER_CALL(rocprofiler_push_external_correlation_id(
                             cb_data.client_ctx, _tid, rocprofiler_user_data_t{.value = _tid}),
                         "failed to push correlation id");

        hsa_status_t itr_status = hsa_iterate_agents(agent_cb, static_cast<void*>(&_agent_data));
        EXPECT_EQ(itr_status, HSA_STATUS_SUCCESS);

        auto user_data = rocprofiler_user_data_t{};
        ROCPROFILER_CALL(
            rocprofiler_pop_external_correlation_id(cb_data.client_ctx, _tid, &user_data),
            "failed to push correlation id");
        EXPECT_EQ(user_data.value, _tid)
            << "callback modification to external correlation id should not be seen here";
    };

    run();
    std::thread{run}.join();
    std::thread{run}.join();

    EXPECT_GT(_agent_data.agent_count, 0);
    EXPECT_EQ(_agent_data.agent_count, _agent_data.agents.size());

    ASSERT_NE(cb_data.client_id, nullptr);
    ASSERT_NE(cb_data.client_fini_func, nullptr);

    cb_data.client_fini_func(*cb_data.client_id);

    // expected callback count is two for each hsa_iterate_agents and two callbacks for
    // hsa_agent_get_info for each agent.
    uint64_t expected_cb_count = (2 * num_runs) + (2 * _agent_data.agent_count);

    EXPECT_EQ(cb_data.client_workflow_count, 2);
    EXPECT_EQ(cb_data.client_callback_count, expected_cb_count);
    EXPECT_EQ(cb_data.client_correlation.size(), expected_cb_count / 2);
    EXPECT_EQ(cb_data.current_depth, 0);
    EXPECT_EQ(cb_data.max_depth, 2);
}

TEST(rocprofiler_lib, buffered_external_correlation)
{
    using init_func_t = int (*)(rocprofiler_client_finalize_t, void*);
    using fini_func_t = void (*)(void*);

    using hsa_iterate_agents_cb_t = hsa_status_t (*)(hsa_agent_t, void*);

    auto cmd_line = rocprofiler::common::read_command_line(getpid());
    ASSERT_FALSE(cmd_line.empty());

    static init_func_t tool_init = [](rocprofiler_client_finalize_t fini_func,
                                      void*                         client_data) -> int {
        auto* cb_data = static_cast<callback_data*>(client_data);

        cb_data->client_workflow_count++;
        cb_data->client_fini_func = fini_func;

        ROCPROFILER_CALL(rocprofiler_create_context(&cb_data->client_ctx),
                         "failed to create context");

        ROCPROFILER_CALL(rocprofiler_create_buffer(cb_data->client_ctx,
                                                   4096,
                                                   2048,
                                                   ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                                   tool_tracing_buffered,
                                                   client_data,
                                                   &cb_data->client_buffer),
                         "buffer creation failed");

        for(auto itr : {ROCPROFILER_BUFFER_TRACING_HSA_CORE_API,
                        ROCPROFILER_BUFFER_TRACING_HSA_AMD_EXT_API,
                        ROCPROFILER_BUFFER_TRACING_HSA_IMAGE_EXT_API,
                        ROCPROFILER_BUFFER_TRACING_HSA_FINALIZE_EXT_API})
        {
            ROCPROFILER_CALL(rocprofiler_configure_buffer_tracing_service(
                                 cb_data->client_ctx, itr, nullptr, 0, cb_data->client_buffer),
                             "buffer tracing service failed to configure");
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
        int   status  = 0;
        ROCPROFILER_CALL(rocprofiler_context_is_active(cb_data->client_ctx, &status),
                         "rocprofiler_context_is_active failed");
        EXPECT_EQ(status, 0);

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

        ROCPROFILER_CALL(rocprofiler_at_internal_thread_create(thread_precreate,
                                                               thread_postcreate,
                                                               ROCPROFILER_LIBRARY,
                                                               static_cast<void*>(&cb_data)),
                         "failed to register for thread creation notifications");

        return &cfg_result;
    };

    auto ctx = rocprofiler_context_id_t{0};
    EXPECT_NE(rocprofiler_create_context(&ctx), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(rocprofiler_force_configure(rocp_init), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_NE(rocprofiler_create_context(&ctx), ROCPROFILER_STATUS_SUCCESS);

    hsa_iterate_agents_cb_t agent_cb = [](hsa_agent_t agent, void* data) {
        static_cast<agent_data*>(data)->agent_count++;

        auto status     = HSA_STATUS_SUCCESS;
        auto agent_type = hsa_device_type_t{};
        if((status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &agent_type)) ==
           HSA_STATUS_SUCCESS)
            static_cast<agent_data*>(data)->agents.emplace_back(agent_type);

        return status;
    };

    auto _agent_data = agent_data{};
    hsa_init();
    hsa_status_t itr_status = hsa_iterate_agents(agent_cb, static_cast<void*>(&_agent_data));

    EXPECT_EQ(itr_status, HSA_STATUS_SUCCESS);
    EXPECT_GT(_agent_data.agent_count, 0);
    EXPECT_EQ(_agent_data.agent_count, _agent_data.agents.size());

    ASSERT_NE(cb_data.client_id, nullptr);
    ASSERT_NE(cb_data.client_fini_func, nullptr);

    EXPECT_EQ(rocprofiler_flush_buffer(cb_data.client_buffer), ROCPROFILER_STATUS_SUCCESS);

    cb_data.client_fini_func(*cb_data.client_id);

    // expected callback count is two for hsa_iterate_agents and two callbacks for
    // hsa_agent_get_info for each agent.
    uint64_t expected_cb_count = 1 + _agent_data.agent_count;
    // expect the tool init, tool fini, and two calls to thread_precreate and thread_postcreate each
    // (the main thread and the assigned thread for the buffer)
    uint64_t expected_workflow_count = 6;

    EXPECT_EQ(cb_data.client_workflow_count, expected_workflow_count);
    EXPECT_EQ(cb_data.client_callback_count, expected_cb_count);
    EXPECT_GT(cb_data.client_thread.handle, 0);
    EXPECT_EQ(cb_data.current_depth, 0);
    EXPECT_EQ(cb_data.max_depth, 0);
}
