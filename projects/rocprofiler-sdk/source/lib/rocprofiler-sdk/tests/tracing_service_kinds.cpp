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

// A tool built against a newer rocprofiler-sdk (e.g. one attached to a process running an older
// rocprofiler-sdk) may request tracing kinds the loaded library does not know. Configuring such a
// kind must return ROCPROFILER_STATUS_ERROR_KIND_NOT_FOUND, not throw.

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

namespace
{
struct tool_data
{
    rocprofiler_client_id_t*      client_id        = nullptr;
    rocprofiler_client_finalize_t client_fini_func = nullptr;
    rocprofiler_context_id_t      context          = {0};
    rocprofiler_buffer_id_t       buffer           = {0};
    int                           init_count       = 0;
};

void
tool_callback(rocprofiler_callback_tracing_record_t, rocprofiler_user_data_t*, void*)
{}

void
tool_buffered_callback(rocprofiler_context_id_t,
                       rocprofiler_buffer_id_t,
                       rocprofiler_record_header_t**,
                       size_t,
                       void*,
                       uint64_t)
{}

template <typename Tp>
auto
unknown_kinds(Tp none, Tp last)
{
    return std::array<Tp, 5>{
        none, last, static_cast<Tp>(last + 1), static_cast<Tp>(last + 1000), static_cast<Tp>(-1)};
}

int
tool_init(rocprofiler_client_finalize_t fini_func, void* data)
{
    auto* _data             = static_cast<tool_data*>(data);
    _data->client_fini_func = fini_func;
    _data->init_count++;

    EXPECT_EQ(rocprofiler_create_context(&_data->context), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(rocprofiler_create_buffer(_data->context,
                                        4096,
                                        2048,
                                        ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                        tool_buffered_callback,
                                        nullptr,
                                        &_data->buffer),
              ROCPROFILER_STATUS_SUCCESS);

    for(auto kind :
        unknown_kinds(ROCPROFILER_CALLBACK_TRACING_NONE, ROCPROFILER_CALLBACK_TRACING_LAST))
    {
        EXPECT_EQ(rocprofiler_configure_callback_tracing_service(
                      _data->context, kind, nullptr, 0, tool_callback, nullptr),
                  ROCPROFILER_STATUS_ERROR_KIND_NOT_FOUND)
            << "callback tracing kind=" << static_cast<int64_t>(kind);
    }

    for(auto kind : unknown_kinds(ROCPROFILER_BUFFER_TRACING_NONE, ROCPROFILER_BUFFER_TRACING_LAST))
    {
        EXPECT_EQ(rocprofiler_configure_buffer_tracing_service(
                      _data->context, kind, nullptr, 0, _data->buffer),
                  ROCPROFILER_STATUS_ERROR_KIND_NOT_FOUND)
            << "buffer tracing kind=" << static_cast<int64_t>(kind);
    }

    // rejected kinds must leave the context usable for known kinds
    EXPECT_EQ(
        rocprofiler_configure_callback_tracing_service(_data->context,
                                                       ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                       nullptr,
                                                       0,
                                                       tool_callback,
                                                       nullptr),
        ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(
        rocprofiler_configure_buffer_tracing_service(
            _data->context, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, nullptr, 0, _data->buffer),
        ROCPROFILER_STATUS_SUCCESS);

    int valid_ctx = 0;
    EXPECT_EQ(rocprofiler_context_is_valid(_data->context, &valid_ctx), ROCPROFILER_STATUS_SUCCESS);
    EXPECT_EQ(valid_ctx, 1);

    return 0;
}

void
tool_fini(void* data)
{
    static_cast<tool_data*>(data)->init_count++;
}
}  // namespace

TEST(rocprofiler_lib, tracing_service_unknown_kinds)
{
    static auto _data = tool_data{};

    static auto cfg_result = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), tool_init, tool_fini, &_data};

    static rocprofiler_configure_func_t rocp_init =
        [](uint32_t,
           const char*,
           uint32_t,
           rocprofiler_client_id_t* client_id) -> rocprofiler_tool_configure_result_t* {
        _data.client_id       = client_id;
        _data.client_id->name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
        return &cfg_result;
    };

    EXPECT_EQ(rocprofiler_force_configure(rocp_init), ROCPROFILER_STATUS_SUCCESS);

    ASSERT_NE(_data.client_id, nullptr);
    ASSERT_NE(_data.client_fini_func, nullptr);
    EXPECT_EQ(_data.init_count, 1);

    _data.client_fini_func(*_data.client_id);

    EXPECT_EQ(_data.init_count, 2);
}
