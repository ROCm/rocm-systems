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

// Kernel-replay single-subscriber rule: A second context configuring it must be rejected with
// ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED. This exercises only the configuration path
// (no GPU / HSA), so it runs unconditionally.

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <gtest/gtest.h>

namespace
{
void
tracing_noop(rocprofiler_callback_tracing_record_t, rocprofiler_user_data_t*, void*)
{}

rocprofiler_status_t
configure_replay(rocprofiler_context_id_t ctx)
{
    return rocprofiler_configure_callback_tracing_service(
        ctx, ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY, nullptr, 0, tracing_noop, nullptr);
}
}  // namespace

// Only one context may own kernel replay; a second is rejected, and the rejection is specific to
// the KERNEL_REPLAY domain (other callback-tracing services on the second context still configure).
TEST(kernel_replay_configure, single_subscriber_only)
{
    using init_func_t = int (*)(rocprofiler_client_finalize_t, void*);
    using fini_func_t = void (*)(void*);

    // The configuration-path assertions run inside tool_init (the configuration window), matching
    // the pc-sampling configure_service test.
    static init_func_t tool_init = [](rocprofiler_client_finalize_t, void*) -> int {
        rocprofiler_context_id_t ctx0{0};
        EXPECT_EQ(rocprofiler_create_context(&ctx0), ROCPROFILER_STATUS_SUCCESS);

        // First subscriber succeeds.
        EXPECT_EQ(configure_replay(ctx0), ROCPROFILER_STATUS_SUCCESS);

        // Same context, same service again: rejected by the per-context/per-kind check.
        EXPECT_EQ(configure_replay(ctx0), ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED);

        // A second, distinct context: rejected because replay allows a single subscriber
        // process-wide (P2-16 / C10).
        rocprofiler_context_id_t ctx1{0};
        EXPECT_EQ(rocprofiler_create_context(&ctx1), ROCPROFILER_STATUS_SUCCESS);
        EXPECT_EQ(configure_replay(ctx1), ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED);

        // The rejection is specific to KERNEL_REPLAY: a different callback-tracing service on that
        // same second context still configures fine.
        EXPECT_EQ(rocprofiler_configure_callback_tracing_service(
                      ctx1,
                      ROCPROFILER_CALLBACK_TRACING_HSA_AMD_EXT_API,
                      nullptr,
                      0,
                      tracing_noop,
                      nullptr),
                  ROCPROFILER_STATUS_SUCCESS);

        return 0;
    };

    static fini_func_t tool_fini = [](void*) -> void {};

    static auto cfg_result = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), tool_init, tool_fini, nullptr};

    static rocprofiler_configure_func_t rocp_init =
        [](uint32_t,
           const char*,
           uint32_t,
           rocprofiler_client_id_t* client_id) -> rocprofiler_tool_configure_result_t* {
        client_id->name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
        return &cfg_result;
    };

    EXPECT_EQ(rocprofiler_force_configure(rocp_init), ROCPROFILER_STATUS_SUCCESS);
}
