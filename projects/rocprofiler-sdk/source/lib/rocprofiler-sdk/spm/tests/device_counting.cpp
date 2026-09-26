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

#include "lib/rocprofiler-sdk/spm/device_counting.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/spm/core.hpp"

#include <rocprofiler-sdk/experimental/spm.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <gtest/gtest.h>
#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ext_amd.h>

#include <cstdint>
#include <sstream>

using namespace rocprofiler::counters;
using namespace rocprofiler;

namespace
{
AmdExtTable&
get_ext_table()
{
    static auto _v = []() {
        auto val                                  = AmdExtTable{};
        val.version.major_id                      = HSA_AMD_EXT_API_TABLE_MAJOR_VERSION;
        val.version.minor_id                      = sizeof(AmdExtTable);
        val.version.step_id                       = HSA_AMD_EXT_API_TABLE_STEP_VERSION;
        val.hsa_amd_memory_pool_get_info_fn       = hsa_amd_memory_pool_get_info;
        val.hsa_amd_agent_iterate_memory_pools_fn = hsa_amd_agent_iterate_memory_pools;
        val.hsa_amd_memory_pool_allocate_fn       = hsa_amd_memory_pool_allocate;
        val.hsa_amd_memory_pool_free_fn           = hsa_amd_memory_pool_free;
        val.hsa_amd_agent_memory_pool_get_info_fn = hsa_amd_agent_memory_pool_get_info;
        val.hsa_amd_agents_allow_access_fn        = hsa_amd_agents_allow_access;
        val.hsa_amd_memory_fill_fn                = hsa_amd_memory_fill;
        val.hsa_amd_signal_create_fn              = hsa_amd_signal_create;
        val.hsa_amd_spm_acquire_fn                = hsa_amd_spm_acquire;
        val.hsa_amd_spm_release_fn                = hsa_amd_spm_release;
        val.hsa_amd_signal_async_handler_fn       = hsa_amd_signal_async_handler;
        return val;
    }();
    return _v;
}

CoreApiTable&
get_api_table()
{
    static auto _v = []() {
        auto val                          = CoreApiTable{};
        val.version.major_id              = HSA_CORE_API_TABLE_MAJOR_VERSION;
        val.version.minor_id              = sizeof(CoreApiTable);
        val.version.step_id               = HSA_CORE_API_TABLE_STEP_VERSION;
        val.hsa_iterate_agents_fn         = hsa_iterate_agents;
        val.hsa_agent_get_info_fn         = hsa_agent_get_info;
        val.hsa_queue_create_fn           = hsa_queue_create;
        val.hsa_queue_destroy_fn          = hsa_queue_destroy;
        val.hsa_signal_wait_relaxed_fn    = hsa_signal_wait_relaxed;
        val.hsa_memory_copy_fn            = hsa_memory_copy;
        val.hsa_signal_create_fn          = hsa_signal_create;
        val.hsa_signal_destroy_fn         = hsa_signal_destroy;
        val.hsa_signal_store_relaxed_fn   = hsa_signal_store_relaxed;
        val.hsa_signal_store_screlease_fn = hsa_signal_store_screlease;
        return val;
    }();
    return _v;
}

#define ROCPROFILER_CALL(result, msg)                                                              \
    {                                                                                              \
        rocprofiler_status_t CHECKSTATUS = result;                                                 \
        if(CHECKSTATUS != ROCPROFILER_STATUS_SUCCESS)                                              \
        {                                                                                          \
            std::string status_msg = rocprofiler_get_status_string(CHECKSTATUS);                   \
            std::cerr << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg            \
                      << " failed with error code " << CHECKSTATUS << ": " << status_msg           \
                      << std::endl;                                                                \
            std::stringstream errmsg{};                                                            \
            errmsg << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg " failure ("  \
                   << status_msg << ")";                                                           \
            ASSERT_EQ(CHECKSTATUS, ROCPROFILER_STATUS_SUCCESS) << errmsg.str();                    \
        }                                                                                          \
    }

bool
is_spm_supported_arch(const hsa::AgentCache& agent)
{
    auto rocp_agent = agent.get_rocp_agent();
    if(!rocp_agent) return false;
    auto v = rocp_agent->gfx_target_version;
    return (v >= 90400 && v <= 90402) || v == 90500;
}

void
test_init()
{
    HsaApiTable table;
    table.amd_ext_ = &get_ext_table();
    table.core_    = &get_api_table();
    rocprofiler::hsa::copy_table(table.core_, 0);
    rocprofiler::hsa::copy_table(table.amd_ext_, 0);
    agent::construct_agent_cache(&table);
    ASSERT_TRUE(hsa::get_queue_controller() != nullptr);
    hsa::get_queue_controller()->init(get_api_table(), get_ext_table());
}

rocprofiler_context_id_t&
get_client_ctx()
{
    static rocprofiler_context_id_t ctx{0};
    return ctx;
}

void
set_client_ctx(rocprofiler_context_id_t& ctx)
{
    ctx = rocprofiler_context_id_t{0};
}

void
null_buffered_callback(rocprofiler_context_id_t,
                       rocprofiler_buffer_id_t,
                       rocprofiler_record_header_t**,
                       size_t,
                       void*,
                       uint64_t)
{}

void
null_device_counting_cb(rocprofiler_context_id_t,
                        rocprofiler_agent_id_t,
                        rocprofiler_device_counting_agent_cb_t,
                        void*)
{}

rocprofiler_agent_id_t
find_first_spm_gpu_agent()
{
    auto agents = hsa::get_queue_controller()->get_supported_agents();
    for(const auto& [_, agent] : agents)
    {
        auto rocp_agent = agent.get_rocp_agent();
        if(rocp_agent && rocp_agent->runtime_visibility.hsa && rocp_agent->runtime_visibility.hip &&
           is_spm_supported_arch(agent))
        {
            return rocp_agent->id;
        }
    }
    return {.handle = 0};
}

}  // namespace

TEST(spm_device_counting, configure_service_basic)
{
    rocprofiler::common::set_env("ROCPROFILER_SPM_BETA_ENABLED", true);
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);

    ROCPROFILER_CALL(rocprofiler_create_context(&get_client_ctx()), "context creation failed");

    rocprofiler_buffer_id_t buf_id = {.handle = 0};
    ROCPROFILER_CALL(rocprofiler_create_buffer(get_client_ctx(),
                                               500 * sizeof(size_t),
                                               500 * sizeof(size_t),
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               null_buffered_callback,
                                               nullptr,
                                               &buf_id),
                     "buffer creation failed");

    auto agent_id = find_first_spm_gpu_agent();
    if(agent_id.handle == 0)
    {
        ROCP_ERROR << "SPM unavailable";
        registration::set_init_status(1);
        registration::finalize();
        context::pop_client(1);
        set_client_ctx(get_client_ctx());
        return;
    }

    auto status = rocprofiler_configure_spm_device_counting_service(
        get_client_ctx(), buf_id, agent_id, null_device_counting_cb, nullptr);
    EXPECT_EQ(status, ROCPROFILER_STATUS_SUCCESS);

    auto* ctx_p = context::get_mutable_registered_context(get_client_ctx());
    ASSERT_TRUE(ctx_p);
    ASSERT_TRUE(ctx_p->device_spm);
    EXPECT_EQ(ctx_p->device_spm->agent_data.size(), 1);
    EXPECT_EQ(ctx_p->device_spm->agent_data.back().agent_id.handle, agent_id.handle);
    EXPECT_EQ(ctx_p->device_spm->agent_data.back().cb, null_device_counting_cb);
    EXPECT_EQ(ctx_p->device_spm->agent_data.back().buffer.handle, buf_id.handle);
    EXPECT_EQ(ctx_p->device_spm->conf_agents.count(agent_id.handle), 1);

    rocprofiler_flush_buffer(buf_id);
    rocprofiler_destroy_buffer(buf_id);

    registration::set_init_status(1);
    registration::finalize();
    context::pop_client(1);
    set_client_ctx(get_client_ctx());
}

TEST(spm_device_counting, configure_service_duplicate_agent)
{
    rocprofiler::common::set_env("ROCPROFILER_SPM_BETA_ENABLED", true);
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);

    ROCPROFILER_CALL(rocprofiler_create_context(&get_client_ctx()), "context creation failed");

    rocprofiler_buffer_id_t buf_id = {.handle = 0};
    ROCPROFILER_CALL(rocprofiler_create_buffer(get_client_ctx(),
                                               500 * sizeof(size_t),
                                               500 * sizeof(size_t),
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               null_buffered_callback,
                                               nullptr,
                                               &buf_id),
                     "buffer creation failed");

    auto agent_id = find_first_spm_gpu_agent();
    if(agent_id.handle == 0)
    {
        ROCP_ERROR << "SPM unavailable";
        registration::set_init_status(1);
        registration::finalize();
        context::pop_client(1);
        set_client_ctx(get_client_ctx());
        return;
    }

    auto status = rocprofiler_configure_spm_device_counting_service(
        get_client_ctx(), buf_id, agent_id, null_device_counting_cb, nullptr);
    ASSERT_EQ(status, ROCPROFILER_STATUS_SUCCESS);

    status = rocprofiler_configure_spm_device_counting_service(
        get_client_ctx(), buf_id, agent_id, null_device_counting_cb, nullptr);
    EXPECT_EQ(status, ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);

    rocprofiler_flush_buffer(buf_id);
    rocprofiler_destroy_buffer(buf_id);

    registration::set_init_status(1);
    registration::finalize();
    context::pop_client(1);
    set_client_ctx(get_client_ctx());
}

TEST(spm_device_counting, configure_service_dispatch_spm_conflict)
{
    rocprofiler::common::set_env("ROCPROFILER_SPM_BETA_ENABLED", true);
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);
    registration::set_fini_status(0);

    ROCPROFILER_CALL(rocprofiler_create_context(&get_client_ctx()), "context creation failed");

    ROCPROFILER_CALL(rocprofiler_spm_configure_callback_dispatch_service(
                         get_client_ctx(),
                         [](const rocprofiler_spm_dispatch_counting_service_data_t*,
                            rocprofiler_counter_config_id_t*,
                            rocprofiler_user_data_t*,
                            void*) {},
                         nullptr,
                         [](const rocprofiler_spm_dispatch_counting_service_data_t*,
                            const rocprofiler_spm_counter_record_t**,
                            size_t,
                            rocprofiler_spm_record_flag_t,
                            rocprofiler_user_data_t,
                            void*) {},
                         nullptr),
                     "setup dispatch spm service");

    rocprofiler_buffer_id_t buf_id   = {.handle = 0};
    rocprofiler_agent_id_t  agent_id = {.handle = 1};

    auto status = rocprofiler_configure_spm_device_counting_service(
        get_client_ctx(), buf_id, agent_id, null_device_counting_cb, nullptr);
    EXPECT_EQ(status, ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT);

    registration::set_init_status(1);
    registration::finalize();
    context::pop_client(1);
    set_client_ctx(get_client_ctx());
}

TEST(spm_device_counting, configure_service_null_buffer_rejected)
{
    rocprofiler::common::set_env("ROCPROFILER_SPM_BETA_ENABLED", true, 1);
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);

    ROCPROFILER_CALL(rocprofiler_create_context(&get_client_ctx()), "context creation failed");

    auto agent_id = find_first_spm_gpu_agent();
    if(agent_id.handle == 0)
    {
        ROCP_ERROR << "SPM unavailable";
        registration::set_init_status(1);
        registration::finalize();
        context::pop_client(1);
        set_client_ctx(get_client_ctx());
        return;
    }

    rocprofiler_buffer_id_t null_buf = {.handle = 0};
    auto                    status   = rocprofiler_configure_spm_device_counting_service(
        get_client_ctx(), null_buf, agent_id, null_device_counting_cb, nullptr);
    EXPECT_EQ(status, ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND);

    registration::set_init_status(1);
    registration::finalize();
    context::pop_client(1);
    set_client_ctx(get_client_ctx());
}

TEST(spm_device_counting, configure_service_user_data_passthrough)
{
    rocprofiler::common::set_env("ROCPROFILER_SPM_BETA_ENABLED", true);
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);

    ROCPROFILER_CALL(rocprofiler_create_context(&get_client_ctx()), "context creation failed");

    rocprofiler_buffer_id_t buf_id = {.handle = 0};
    ROCPROFILER_CALL(rocprofiler_create_buffer(get_client_ctx(),
                                               500 * sizeof(size_t),
                                               500 * sizeof(size_t),
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               null_buffered_callback,
                                               nullptr,
                                               &buf_id),
                     "buffer creation failed");

    auto agent_id = find_first_spm_gpu_agent();
    if(agent_id.handle == 0)
    {
        ROCP_ERROR << "SPM unavailable";
        registration::set_init_status(1);
        registration::finalize();
        context::pop_client(1);
        set_client_ctx(get_client_ctx());
        return;
    }

    int  sentinel = 42;
    auto status   = rocprofiler_configure_spm_device_counting_service(
        get_client_ctx(), buf_id, agent_id, null_device_counting_cb, &sentinel);
    ASSERT_EQ(status, ROCPROFILER_STATUS_SUCCESS);

    auto* ctx_p = context::get_mutable_registered_context(get_client_ctx());
    ASSERT_TRUE(ctx_p);
    ASSERT_TRUE(ctx_p->device_spm);
    EXPECT_EQ(ctx_p->device_spm->agent_data.back().callback_data.ptr, &sentinel);

    rocprofiler_flush_buffer(buf_id);
    rocprofiler_destroy_buffer(buf_id);

    registration::set_init_status(1);
    registration::finalize();
    context::pop_client(1);
    set_client_ctx(get_client_ctx());
}

TEST(spm_device_counting, state_machine_initial_disabled)
{
    rocprofiler::common::set_env("ROCPROFILER_SPM_BETA_ENABLED", true);
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);

    ROCPROFILER_CALL(rocprofiler_create_context(&get_client_ctx()), "context creation failed");

    rocprofiler_buffer_id_t buf_id = {.handle = 0};
    ROCPROFILER_CALL(rocprofiler_create_buffer(get_client_ctx(),
                                               500 * sizeof(size_t),
                                               500 * sizeof(size_t),
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               null_buffered_callback,
                                               nullptr,
                                               &buf_id),
                     "buffer creation failed");

    auto agent_id = find_first_spm_gpu_agent();
    if(agent_id.handle == 0)
    {
        ROCP_ERROR << "SPM unavailable";
        registration::set_init_status(1);
        registration::finalize();
        context::pop_client(1);
        set_client_ctx(get_client_ctx());
        return;
    }

    ROCPROFILER_CALL(rocprofiler_configure_spm_device_counting_service(
                         get_client_ctx(), buf_id, agent_id, null_device_counting_cb, nullptr),
                     "configure spm device counting");

    auto* ctx_p = context::get_mutable_registered_context(get_client_ctx());
    ASSERT_TRUE(ctx_p);
    ASSERT_TRUE(ctx_p->device_spm);

    EXPECT_EQ(ctx_p->device_spm->status.load(),
              context::spm_device_counting_service::state::DISABLED);

    rocprofiler_flush_buffer(buf_id);
    rocprofiler_destroy_buffer(buf_id);

    registration::set_init_status(1);
    registration::finalize();
    context::pop_client(1);
    set_client_ctx(get_client_ctx());
}

TEST(spm_device_counting, finalize_transitions_to_exit)
{
    rocprofiler::common::set_env("ROCPROFILER_SPM_BETA_ENABLED", true);
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);

    ROCPROFILER_CALL(rocprofiler_create_context(&get_client_ctx()), "context creation failed");

    rocprofiler_buffer_id_t buf_id = {.handle = 0};
    ROCPROFILER_CALL(rocprofiler_create_buffer(get_client_ctx(),
                                               500 * sizeof(size_t),
                                               500 * sizeof(size_t),
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               null_buffered_callback,
                                               nullptr,
                                               &buf_id),
                     "buffer creation failed");

    auto agent_id = find_first_spm_gpu_agent();
    if(agent_id.handle == 0)
    {
        ROCP_ERROR << "SPM unavailable";
        registration::set_init_status(1);
        registration::finalize();
        context::pop_client(1);
        set_client_ctx(get_client_ctx());
        return;
    }

    ROCPROFILER_CALL(rocprofiler_configure_spm_device_counting_service(
                         get_client_ctx(), buf_id, agent_id, null_device_counting_cb, nullptr),
                     "configure spm device counting");

    auto* ctx_p = context::get_mutable_registered_context(get_client_ctx());
    ASSERT_TRUE(ctx_p);
    ASSERT_TRUE(ctx_p->device_spm);
    EXPECT_EQ(ctx_p->device_spm->status.load(),
              context::spm_device_counting_service::state::DISABLED);

    auto finalize_status = SPM::spm_device_counting_service_finalize();
    EXPECT_EQ(finalize_status, ROCPROFILER_STATUS_SUCCESS);

    EXPECT_EQ(ctx_p->device_spm->status.load(), context::spm_device_counting_service::state::EXIT);

    rocprofiler_flush_buffer(buf_id);
    rocprofiler_destroy_buffer(buf_id);

    registration::set_init_status(1);
    registration::finalize();
    context::pop_client(1);
    set_client_ctx(get_client_ctx());
}

TEST(spm_device_counting, callback_data_fields)
{
    rocprofiler::common::set_env("ROCPROFILER_SPM_BETA_ENABLED", true, 1);
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);

    ROCPROFILER_CALL(rocprofiler_create_context(&get_client_ctx()), "context creation failed");

    rocprofiler_buffer_id_t buf_id = {.handle = 0};
    ROCPROFILER_CALL(rocprofiler_create_buffer(get_client_ctx(),
                                               500 * sizeof(size_t),
                                               500 * sizeof(size_t),
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               null_buffered_callback,
                                               nullptr,
                                               &buf_id),
                     "buffer creation failed");

    auto agent_id = find_first_spm_gpu_agent();
    if(agent_id.handle == 0)
    {
        ROCP_ERROR << "SPM unavailable";
        registration::set_init_status(1);
        registration::finalize();
        context::pop_client(1);
        set_client_ctx(get_client_ctx());
        return;
    }

    ROCPROFILER_CALL(rocprofiler_configure_spm_device_counting_service(
                         get_client_ctx(), buf_id, agent_id, null_device_counting_cb, nullptr),
                     "configure spm device counting");

    auto* ctx_p = context::get_mutable_registered_context(get_client_ctx());
    ASSERT_TRUE(ctx_p);
    ASSERT_TRUE(ctx_p->device_spm);
    ASSERT_EQ(ctx_p->device_spm->agent_data.size(), 1);

    auto& data = ctx_p->device_spm->agent_data.back();
    EXPECT_EQ(data.agent_id.handle, agent_id.handle);
    EXPECT_EQ(data.cb, null_device_counting_cb);
    EXPECT_EQ(data.buffer.handle, buf_id.handle);
    EXPECT_EQ(data.set_profile, false);
    EXPECT_EQ(data.queue, nullptr);
    EXPECT_EQ(data.start_signal.handle, 0u);
    EXPECT_EQ(data.stop_signal.handle, 0u);
    EXPECT_FALSE(data.packet);
    EXPECT_FALSE(data.profile);

    rocprofiler_flush_buffer(buf_id);
    rocprofiler_destroy_buffer(buf_id);

    registration::set_init_status(1);
    registration::finalize();
    context::pop_client(1);
    set_client_ctx(get_client_ctx());
}
