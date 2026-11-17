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

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/experimental/registration.h>
#include <rocprofiler-sdk/registration.h>

#include <stdexcept>

namespace
{
rocprofiler_context_id_t context         = {};
int                      attach_count    = 0;
int                      detach_count    = 0;
int                      failure         = (ROCPROFILER_TEST_CLIENT_ID == 2) ? 1 : 0;
bool                     throw_on_detach = false;

void
check(rocprofiler_status_t status)
{
    if(status != ROCPROFILER_STATUS_SUCCESS)
        throw std::runtime_error("fixture context operation failed");
}

void
callback(rocprofiler_callback_tracing_record_t, rocprofiler_user_data_t*, void*)
{}

int
initialize(rocprofiler_client_finalize_t, void*)
{
    check(rocprofiler_create_context(&context));
    check(rocprofiler_configure_callback_tracing_service(
        context, ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API, nullptr, 0, callback, nullptr));
    // Client 3 represents an unrelated, already active tracing client.
    if(ROCPROFILER_TEST_CLIENT_ID == 3) check(rocprofiler_start_context(context));
    return 0;
}

int
attach(rocprofiler_client_detach_t, rocprofiler_context_id_t*, uint64_t, void*)
{
    check(rocprofiler_start_context(context));
    ++attach_count;
    if(failure == 2) throw std::runtime_error("fixture attach exception");
    return failure ? -1 : 0;
}

void
detach(void*)
{
    int active = -1;
    check(rocprofiler_context_is_active(context, &active));
    // Count only callbacks reached after the SDK stopped this client's context.
    if(active != 0) throw std::runtime_error("detach callback ran before context stop");
    ++detach_count;
    if(throw_on_detach) throw std::runtime_error("fixture detach exception");
}
}  // namespace

extern "C"
{
rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t*)
{
    static auto result = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), initialize, nullptr, nullptr};
    return &result;
}

rocprofiler_tool_configure_attach_result_t*
rocprofiler_configure_attach(uint32_t, const char*, uint32_t, rocprofiler_client_id_t*)
{
    static auto result = rocprofiler_tool_configure_attach_result_t{
        sizeof(rocprofiler_tool_configure_attach_result_t), attach, detach, nullptr};
    return ROCPROFILER_TEST_CLIENT_ID == 3 ? nullptr : &result;
}

ROCPROFILER_PUBLIC_API int
fixture_active()
{
    int active = -1;
    check(rocprofiler_context_is_active(context, &active));
    return active;
}
ROCPROFILER_PUBLIC_API int
fixture_attaches()
{
    return attach_count;
}
ROCPROFILER_PUBLIC_API int
fixture_detaches()
{
    return detach_count;
}
ROCPROFILER_PUBLIC_API void
fixture_failure(int value)
{
    failure = value;
}
ROCPROFILER_PUBLIC_API void
fixture_throw_detach(int value)
{
    throw_on_detach = value;
}
}
