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

// Pins the should_batch_packets predicate WriteInterceptor uses:
//   !counters::is_any_active() && !thread_trace::is_any_active()

#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/counters/queue_hooks.hpp"
#include "lib/rocprofiler-sdk/counters/tests/hsa_tables.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/thread_trace/queue_hooks.hpp"

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/dispatch_counting_service.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <gtest/gtest.h>
#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>

#include <sstream>

using namespace rocprofiler::counters::test_constants;
using namespace rocprofiler;

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

namespace
{
rocprofiler_context_id_t&
get_client_ctx()
{
    static rocprofiler_context_id_t ctx{0};
    return ctx;
}

void
test_init()
{
    HsaApiTable table;
    table.amd_ext_ = &get_ext_table();
    table.core_    = &get_api_table();
    agent::construct_agent_cache(&table);
    ASSERT_TRUE(hsa::get_queue_controller() != nullptr);
    hsa::get_queue_controller()->init(get_api_table(), get_ext_table());
}

void
null_dispatch_callback(rocprofiler_dispatch_counting_service_data_t,
                       rocprofiler_counter_config_id_t*,
                       rocprofiler_user_data_t*,
                       void*)
{}

void
null_buffered_callback(rocprofiler_context_id_t,
                       rocprofiler_buffer_id_t,
                       rocprofiler_record_header_t**,
                       size_t,
                       void*,
                       uint64_t)
{}
}  // namespace

TEST(ShouldBatchPackets, BatchesPacketsWhenNoSubsystemActive)
{
    EXPECT_FALSE(rocprofiler::counters::is_any_active());
    EXPECT_FALSE(rocprofiler::thread_trace::is_any_active());

    bool should_batch_packets = !rocprofiler::counters::is_any_active() &&
                                !rocprofiler::thread_trace::is_any_active();
    EXPECT_TRUE(should_batch_packets);
}

// Uses the full registered-context activation pattern (configure + start_context)
// rather than direct queue_cb invocation; is_any_active() walks the active-context
// list and would not see a context that bypasses registration.
TEST(ShouldBatchPackets, RequiresPerPacketWhenCountersActive)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);
    ROCPROFILER_CALL(rocprofiler_create_context(&get_client_ctx()), "context creation failed");

    rocprofiler_buffer_id_t opt_buff_id = {.handle = 0};
    ROCPROFILER_CALL(rocprofiler_create_buffer(get_client_ctx(),
                                               500 * sizeof(size_t),
                                               500 * sizeof(size_t),
                                               ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                               null_buffered_callback,
                                               nullptr,
                                               &opt_buff_id),
                     "Could not create buffer");

    ROCPROFILER_CALL(rocprofiler_configure_buffer_dispatch_counting_service(
                         get_client_ctx(), opt_buff_id, null_dispatch_callback, (void*) 0x12345),
                     "Could not setup buffered service");
    ROCPROFILER_CALL(rocprofiler_start_context(get_client_ctx()), "start context");

    EXPECT_TRUE(rocprofiler::counters::is_any_active());

    bool should_batch_packets = !rocprofiler::counters::is_any_active() &&
                                !rocprofiler::thread_trace::is_any_active();
    EXPECT_FALSE(should_batch_packets);

    ROCPROFILER_CALL(rocprofiler_stop_context(get_client_ctx()), "stop context");
    EXPECT_FALSE(rocprofiler::counters::is_any_active());

    rocprofiler_flush_buffer(opt_buff_id);
    rocprofiler_destroy_buffer(opt_buff_id);

    registration::set_init_status(1);
    registration::finalize();
}

// ATT-active coverage requires GPU/KFD plumbing not available in unit tests;
// the per-packet path is exercised by the rocprofv3 --att smoke instead.
TEST(ShouldBatchPackets, RequiresPerPacketWhenATTActive)
{
    GTEST_SKIP() << "DispatchThreadTracer activation requires GPU integration smoke";
}
