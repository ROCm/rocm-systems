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

// Range-replay configuration path, mirroring kernel_replay/tests/replay_configure.cpp.
//
// Covers the single-subscriber rule, the domain's operation table as the SDK reports it, and the
// errors rocprofiler_range_replay_begin/end return when no service owns range replay. None of this
// touches a GPU or the HSA runtime, so it runs unconditionally.

#include <rocprofiler-sdk/experimental/range_replay.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace
{
void
tracing_noop(rocprofiler_callback_tracing_record_t, rocprofiler_user_data_t*, void*)
{}

rocprofiler_status_t
configure_range_replay(rocprofiler_context_id_t ctx)
{
    return rocprofiler_configure_callback_tracing_service(
        ctx, ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY, nullptr, 0, tracing_noop, nullptr);
}
}  // namespace

// Only one context may own range replay: a range runs a single plan, so a second subscriber would
// make pass_count_cb last-writer-wins and deliver one tool's user_data to another. The rejection
// must also be specific to the RANGE_REPLAY domain, and it must not consume the claim -- a rejected
// attempt that released the owner's claim would silently disable the first tool's replay.
TEST(range_replay_configure, single_subscriber_only)
{
    using init_func_t = int (*)(rocprofiler_client_finalize_t, void*);
    using fini_func_t = void (*)(void*);

    // The configuration-path assertions run inside tool_init (the configuration window), matching
    // kernel-replay-configure-test.
    static init_func_t tool_init = [](rocprofiler_client_finalize_t, void*) -> int {
        // No context owns range replay yet, so a range cannot be opened: the tool would get no
        // CONFIG callback and nothing would ever report the range's outcome.
        EXPECT_EQ(rocprofiler_range_replay_begin(1), ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND);

        rocprofiler_context_id_t ctx0{0};
        EXPECT_EQ(rocprofiler_create_context(&ctx0), ROCPROFILER_STATUS_SUCCESS);

        // First subscriber succeeds.
        EXPECT_EQ(configure_range_replay(ctx0), ROCPROFILER_STATUS_SUCCESS);

        // Registered but not started. The two errors are distinct on purpose: "you never
        // configured the service" and "you configured it but the context is stopped" are different
        // tool bugs, and the second is the one a tool hits by forgetting rocprofiler_start_context.
        EXPECT_EQ(rocprofiler_range_replay_begin(1), ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_STARTED);

        // Same context, same service again: rejected by the per-context/per-kind check.
        EXPECT_EQ(configure_range_replay(ctx0),
                  ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED);

        // A second, distinct context: rejected because range replay allows a single subscriber
        // process-wide.
        rocprofiler_context_id_t ctx1{0};
        EXPECT_EQ(rocprofiler_create_context(&ctx1), ROCPROFILER_STATUS_SUCCESS);
        EXPECT_EQ(configure_range_replay(ctx1),
                  ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED);

        // The rejection is specific to RANGE_REPLAY: a different callback-tracing service on that
        // same second context still configures fine.
        EXPECT_EQ(rocprofiler_configure_callback_tracing_service(
                      ctx1,
                      ROCPROFILER_CALLBACK_TRACING_HSA_AMD_EXT_API,
                      nullptr,
                      0,
                      tracing_noop,
                      nullptr),
                  ROCPROFILER_STATUS_SUCCESS);

        // Range replay and kernel replay are separate services with separate claims, so owning one
        // must not block the other. A tool that drives both configures them on one context.
        EXPECT_EQ(rocprofiler_configure_callback_tracing_service(
                      ctx0,
                      ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY,
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

// The SDK reports a domain's operations through
// rocprofiler_iterate_callback_tracing_kind_operations and names them through
// rocprofiler_query_callback_tracing_kind_operation_name. A domain added without wiring both ends
// up reporting an empty operation set, which tools use to decide what to subscribe to. This checks
// the RANGE_REPLAY entry is wired in both directions.
TEST(range_replay_configure, domain_reports_its_operations)
{
    auto operations = std::vector<int>{};

    const auto status = rocprofiler_iterate_callback_tracing_kind_operations(
        ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY,
        [](rocprofiler_callback_tracing_kind_t, int operation, void* user_data) -> int {
            static_cast<std::vector<int>*>(user_data)->emplace_back(operation);
            return 0;
        },
        &operations);

    ASSERT_EQ(status, ROCPROFILER_STATUS_SUCCESS);

    // The SDK reports every value below LAST, NONE included, matching the other domains.
    ASSERT_EQ(operations.size(), static_cast<size_t>(ROCPROFILER_RANGE_REPLAY_LAST));

    EXPECT_EQ(operations.at(0), static_cast<int>(ROCPROFILER_RANGE_REPLAY_NONE));
    EXPECT_EQ(operations.at(1), static_cast<int>(ROCPROFILER_RANGE_REPLAY_CONFIG));
    EXPECT_EQ(operations.at(2), static_cast<int>(ROCPROFILER_RANGE_REPLAY_PASS));
    EXPECT_EQ(operations.at(3), static_cast<int>(ROCPROFILER_RANGE_REPLAY_CLOSE));
}

// Operation names are the enum suffix, as they are for every other domain: a tool that formats a
// trace record by name gets "RANGE_REPLAY_PASS", not "PASS".
TEST(range_replay_configure, operations_have_names)
{
    const auto expected = std::vector<std::pair<rocprofiler_range_replay_operation_t, std::string>>{
        {ROCPROFILER_RANGE_REPLAY_NONE, "RANGE_REPLAY_NONE"},
        {ROCPROFILER_RANGE_REPLAY_CONFIG, "RANGE_REPLAY_CONFIG"},
        {ROCPROFILER_RANGE_REPLAY_PASS, "RANGE_REPLAY_PASS"},
        {ROCPROFILER_RANGE_REPLAY_CLOSE, "RANGE_REPLAY_CLOSE"}};

    for(const auto& [operation, name] : expected)
    {
        const char* actual = nullptr;
        uint64_t    length = 0;

        ASSERT_EQ(rocprofiler_query_callback_tracing_kind_operation_name(
                      ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY, operation, &actual, &length),
                  ROCPROFILER_STATUS_SUCCESS)
            << "operation " << static_cast<int>(operation);
        ASSERT_NE(actual, nullptr);
        EXPECT_EQ(std::string{actual}, name);
        EXPECT_EQ(length, name.size());
    }
}

TEST(range_replay_configure, domain_has_a_name)
{
    const char* name   = nullptr;
    uint64_t    length = 0;

    ASSERT_EQ(rocprofiler_query_callback_tracing_kind_name(
                  ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY, &name, &length),
              ROCPROFILER_STATUS_SUCCESS);
    ASSERT_NE(name, nullptr);
    EXPECT_EQ(std::string{name}, "RANGE_REPLAY");
}

// Out-of-range operation IDs must be rejected rather than indexing past the name table.
TEST(range_replay_configure, unknown_operations_are_rejected)
{
    const char* name   = nullptr;
    uint64_t    length = 0;

    EXPECT_EQ(rocprofiler_query_callback_tracing_kind_operation_name(
                  ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY,
                  ROCPROFILER_RANGE_REPLAY_LAST,
                  &name,
                  &length),
              ROCPROFILER_STATUS_ERROR_OPERATION_NOT_FOUND);
    EXPECT_EQ(name, nullptr);
    EXPECT_EQ(length, 0u);
}

// end() without a matching begin() is a tool bug, not a decline: there is no range to report a
// status for, so it fails rather than delivering a CLOSE callback for a range that never opened.
// Ranges are thread-local, so this holds regardless of what any other context has configured.
TEST(range_replay_configure, end_without_begin_is_rejected)
{
    EXPECT_EQ(rocprofiler_range_replay_end(), ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);
}
