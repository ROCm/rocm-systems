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

#include "lib/rocprofiler-sdk/thread_trace/tests/fake_trace_decoder.hpp"

#include "lib/rocprofiler-sdk/registration.hpp"

#include <rocprofiler-sdk/experimental/thread-trace/trace_decoder.h>

#include <gtest/gtest.h>

#include <dlfcn.h>
#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>

namespace
{
constexpr uint64_t HIDDEN_LATENCY = ROCPROFILER_THREAD_TRACE_DECODER_ANALYSIS_HIDDEN_LATENCY;

/// Loads a fake decoder and gives the test access to the call log inside it.
class FakeDecoder
{
public:
    explicit FakeDecoder(const char* directory)
    {
        rocprofiler::registration::init_logging();

        status = rocprofiler_thread_trace_decoder_create(&id, directory);
        if(status != ROCPROFILER_STATUS_SUCCESS) return;

        // The SDK loaded the library RTLD_LOCAL, so reach the log by opening the same path,
        // which returns the mapping already in place rather than a second copy.
        auto path = std::string{directory} + "/librocprof-trace-decoder.so";
        library   = dlopen(path.c_str(), RTLD_LAZY | RTLD_LOCAL);
        if(library == nullptr) return;

        auto* accessor = reinterpret_cast<decltype(&fake_trace_decoder_log)>(
            dlsym(library, "fake_trace_decoder_log"));
        fail_at = reinterpret_cast<decltype(&fake_trace_decoder_fail_at)>(
            dlsym(library, "fake_trace_decoder_fail_at"));
        if(accessor == nullptr || fail_at == nullptr) return;

        log  = accessor();
        *log = fake_trace_decoder_log_t{};
        fail_at(FAKE_TRACE_DECODER_STEP_NONE);
    }

    ~FakeDecoder()
    {
        if(fail_at != nullptr) fail_at(FAKE_TRACE_DECODER_STEP_NONE);
        if(status == ROCPROFILER_STATUS_SUCCESS) rocprofiler_thread_trace_decoder_destroy(id);
        if(library != nullptr) dlclose(library);
    }

    FakeDecoder(const FakeDecoder&) = delete;
    FakeDecoder(FakeDecoder&&)      = delete;
    FakeDecoder& operator=(const FakeDecoder&) = delete;
    FakeDecoder& operator=(FakeDecoder&&) = delete;

    rocprofiler_thread_trace_decoder_id_t id{};
    rocprofiler_status_t                  status{ROCPROFILER_STATUS_ERROR};
    void*                                 library{nullptr};
    fake_trace_decoder_log_t*             log{nullptr};
    decltype(&fake_trace_decoder_fail_at) fail_at{nullptr};
};

using RecordCounts = std::unordered_map<int, uint64_t>;

void
count_records(rocprofiler_thread_trace_decoder_record_type_t record_type_id,
              void*                                          trace_events,
              uint64_t                                       trace_size,
              void*                                          userdata)
{
    (void) trace_events;
    (*static_cast<RecordCounts*>(userdata))[record_type_id] += trace_size;
}

/// Decodes a throwaway payload, which the fake reports the size of rather than parsing.
rocprofiler_status_t
decode(const FakeDecoder& decoder, RecordCounts* counts)
{
    static std::array<uint8_t, 8> payload{};
    return rocprofiler_trace_decode(
        decoder.id, count_records, payload.data(), payload.size(), counts);
}

TEST(decoder_analysis, rejects_unknown_flags)
{
    auto decoder = FakeDecoder{FAKE_TRACE_DECODER_V2_DIR};
    ASSERT_EQ(decoder.status, ROCPROFILER_STATUS_SUCCESS);

    EXPECT_EQ(rocprofiler_thread_trace_decoder_set_analysis(decoder.id, HIDDEN_LATENCY << 1),
              ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(rocprofiler_thread_trace_decoder_set_analysis(decoder.id, ~uint64_t{0}),
              ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);

    EXPECT_EQ(rocprofiler_thread_trace_decoder_set_analysis(decoder.id, HIDDEN_LATENCY),
              ROCPROFILER_STATUS_SUCCESS);
}

TEST(decoder_analysis, rejects_unknown_decoder)
{
    auto unknown = rocprofiler_thread_trace_decoder_id_t{.handle = 0};

    EXPECT_EQ(rocprofiler_thread_trace_decoder_set_analysis(unknown, HIDDEN_LATENCY),
              ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);
}

TEST(decoder_analysis, defaults_to_parse_data)
{
    auto decoder = FakeDecoder{FAKE_TRACE_DECODER_V2_DIR};
    ASSERT_EQ(decoder.status, ROCPROFILER_STATUS_SUCCESS);
    ASSERT_NE(decoder.log, nullptr);

    auto counts = RecordCounts{};
    EXPECT_EQ(decode(decoder, &counts), ROCPROFILER_STATUS_SUCCESS);

    EXPECT_EQ(decoder.log->parse_data_calls, 1u);
    EXPECT_EQ(decoder.log->create_handle_calls, 0u);
    EXPECT_EQ(decoder.log->set_analysis_calls, 0u);
    EXPECT_EQ(decoder.log->se_data_bytes, 8u);

    EXPECT_EQ(counts[ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE], 1u);
    EXPECT_EQ(counts[ROCPROFILER_THREAD_TRACE_DECODER_RECORD_HIDDEN_LATENCY], 0u);
}

TEST(decoder_analysis, analysis_takes_the_handle_api)
{
    auto decoder = FakeDecoder{FAKE_TRACE_DECODER_V2_DIR};
    ASSERT_EQ(decoder.status, ROCPROFILER_STATUS_SUCCESS);
    ASSERT_NE(decoder.log, nullptr);

    ASSERT_EQ(rocprofiler_thread_trace_decoder_set_analysis(decoder.id, HIDDEN_LATENCY),
              ROCPROFILER_STATUS_SUCCESS);

    auto counts = RecordCounts{};
    EXPECT_EQ(decode(decoder, &counts), ROCPROFILER_STATUS_SUCCESS);

    EXPECT_EQ(decoder.log->parse_data_calls, 0u);
    EXPECT_EQ(decoder.log->create_handle_calls, 1u);
    EXPECT_EQ(decoder.log->set_isa_callback_calls, 1u);
    EXPECT_EQ(decoder.log->set_se_data_callback_calls, 1u);
    EXPECT_EQ(decoder.log->set_analysis_calls, 1u);
    EXPECT_EQ(decoder.log->handle_parse_calls, 1u);
    EXPECT_EQ(decoder.log->destroy_handle_calls, 1u);

    EXPECT_EQ(decoder.log->analysis_flags, HIDDEN_LATENCY);
    EXPECT_EQ(decoder.log->se_data_bytes, 8u);

    EXPECT_EQ(counts[ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE], 1u);
    EXPECT_EQ(counts[ROCPROFILER_THREAD_TRACE_DECODER_RECORD_HIDDEN_LATENCY], 1u);
}

TEST(decoder_analysis, analysis_is_reversible)
{
    auto decoder = FakeDecoder{FAKE_TRACE_DECODER_V2_DIR};
    ASSERT_EQ(decoder.status, ROCPROFILER_STATUS_SUCCESS);
    ASSERT_NE(decoder.log, nullptr);

    ASSERT_EQ(rocprofiler_thread_trace_decoder_set_analysis(decoder.id, HIDDEN_LATENCY),
              ROCPROFILER_STATUS_SUCCESS);
    ASSERT_EQ(rocprofiler_thread_trace_decoder_set_analysis(
                  decoder.id, ROCPROFILER_THREAD_TRACE_DECODER_ANALYSIS_NONE),
              ROCPROFILER_STATUS_SUCCESS);

    auto counts = RecordCounts{};
    EXPECT_EQ(decode(decoder, &counts), ROCPROFILER_STATUS_SUCCESS);

    EXPECT_EQ(decoder.log->parse_data_calls, 1u);
    EXPECT_EQ(decoder.log->create_handle_calls, 0u);
    EXPECT_EQ(counts[ROCPROFILER_THREAD_TRACE_DECODER_RECORD_HIDDEN_LATENCY], 0u);
}

TEST(decoder_analysis, handle_is_not_destroyed_when_never_created)
{
    auto decoder = FakeDecoder{FAKE_TRACE_DECODER_V2_DIR};
    ASSERT_EQ(decoder.status, ROCPROFILER_STATUS_SUCCESS);
    ASSERT_NE(decoder.log, nullptr);

    ASSERT_EQ(rocprofiler_thread_trace_decoder_set_analysis(decoder.id, HIDDEN_LATENCY),
              ROCPROFILER_STATUS_SUCCESS);
    decoder.fail_at(FAKE_TRACE_DECODER_STEP_CREATE_HANDLE);

    auto counts = RecordCounts{};
    EXPECT_EQ(decode(decoder, &counts), ROCPROFILER_STATUS_ERROR);

    EXPECT_EQ(decoder.log->create_handle_calls, 1u);
    EXPECT_EQ(decoder.log->destroy_handle_calls, 0u);
    EXPECT_EQ(decoder.log->set_isa_callback_calls, 0u);
    EXPECT_EQ(decoder.log->handle_parse_calls, 0u);
}

TEST(decoder_analysis, failed_setup_still_destroys_the_handle)
{
    auto decoder = FakeDecoder{FAKE_TRACE_DECODER_V2_DIR};
    ASSERT_EQ(decoder.status, ROCPROFILER_STATUS_SUCCESS);
    ASSERT_NE(decoder.log, nullptr);

    ASSERT_EQ(rocprofiler_thread_trace_decoder_set_analysis(decoder.id, HIDDEN_LATENCY),
              ROCPROFILER_STATUS_SUCCESS);
    decoder.fail_at(FAKE_TRACE_DECODER_STEP_SET_ANALYSIS);

    auto counts = RecordCounts{};
    EXPECT_EQ(decode(decoder, &counts), ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);

    EXPECT_EQ(decoder.log->create_handle_calls, 1u);
    EXPECT_EQ(decoder.log->destroy_handle_calls, 1u);

    // Setup stops at the first failure rather than carrying on into the parse.
    EXPECT_EQ(decoder.log->handle_parse_calls, 0u);
    EXPECT_EQ(counts[ROCPROFILER_THREAD_TRACE_DECODER_RECORD_HIDDEN_LATENCY], 0u);
}

TEST(decoder_analysis, failed_parse_still_destroys_the_handle)
{
    auto decoder = FakeDecoder{FAKE_TRACE_DECODER_V2_DIR};
    ASSERT_EQ(decoder.status, ROCPROFILER_STATUS_SUCCESS);
    ASSERT_NE(decoder.log, nullptr);

    ASSERT_EQ(rocprofiler_thread_trace_decoder_set_analysis(decoder.id, HIDDEN_LATENCY),
              ROCPROFILER_STATUS_SUCCESS);
    decoder.fail_at(FAKE_TRACE_DECODER_STEP_PARSE);

    auto counts = RecordCounts{};
    EXPECT_EQ(decode(decoder, &counts), ROCPROFILER_STATUS_ERROR_AGENT_ARCH_NOT_SUPPORTED);

    EXPECT_EQ(decoder.log->handle_parse_calls, 1u);
    EXPECT_EQ(decoder.log->destroy_handle_calls, 1u);
}

TEST(decoder_analysis, older_decoder_refuses_analysis)
{
    auto decoder = FakeDecoder{FAKE_TRACE_DECODER_V1_DIR};
    ASSERT_EQ(decoder.status, ROCPROFILER_STATUS_SUCCESS);

    EXPECT_EQ(rocprofiler_thread_trace_decoder_set_analysis(decoder.id, HIDDEN_LATENCY),
              ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI);

    // Turning analyses off asks nothing of the decoder, so it stays available.
    EXPECT_EQ(rocprofiler_thread_trace_decoder_set_analysis(
                  decoder.id, ROCPROFILER_THREAD_TRACE_DECODER_ANALYSIS_NONE),
              ROCPROFILER_STATUS_SUCCESS);
}

TEST(decoder_analysis, older_decoder_still_decodes)
{
    auto decoder = FakeDecoder{FAKE_TRACE_DECODER_V1_DIR};
    ASSERT_EQ(decoder.status, ROCPROFILER_STATUS_SUCCESS);
    ASSERT_NE(decoder.log, nullptr);

    ASSERT_EQ(rocprofiler_thread_trace_decoder_set_analysis(decoder.id, HIDDEN_LATENCY),
              ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI);

    auto counts = RecordCounts{};
    EXPECT_EQ(decode(decoder, &counts), ROCPROFILER_STATUS_SUCCESS);

    EXPECT_EQ(decoder.log->parse_data_calls, 1u);
    EXPECT_EQ(counts[ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE], 1u);
    EXPECT_EQ(counts[ROCPROFILER_THREAD_TRACE_DECODER_RECORD_HIDDEN_LATENCY], 0u);
}
}  // namespace
