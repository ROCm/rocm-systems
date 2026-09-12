// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// A minimal rocprofiler-sdk client used to exercise the SDK's OMPT
// keep-or-defer role decision (it stands in for a real SDK client such as
// rocprofv3). Its OMPT appetite is toggled by an environment variable:
//
//   COEXIST_OMPT set    -> the client registers an OMPT tracing service, so
//                          ompt_service_requested() is true and the SDK KEEPS
//                          the OMPT tool role.
//   COEXIST_OMPT unset  -> the client registers ONLY kernel-dispatch tracing
//                          (no OMPT), so ompt_service_requested() is false and
//                          the SDK DEFERS the OMPT tool role to another tool.
//
// In both cases the client always traces kernel dispatches, which lets the
// validator prove the SDK keeps doing its non-OMPT work even while it defers
// OMPT. On teardown the client writes a tiny JSON summary (record counts) to
// $COEXIST_CLIENT_OUTPUT for pytest.

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

#define COEXIST_CALL(result, msg)                                                                  \
    {                                                                                              \
        rocprofiler_status_t CHECKSTATUS = (result);                                               \
        if(CHECKSTATUS != ROCPROFILER_STATUS_SUCCESS)                                              \
        {                                                                                          \
            const char* status_msg = rocprofiler_get_status_string(CHECKSTATUS);                   \
            std::cerr << "[coexist-client] " << __FILE__ << ":" << __LINE__ << " " << (msg)        \
                      << " failed with error code " << CHECKSTATUS << ": " << status_msg           \
                      << std::endl;                                                                \
            std::abort();                                                                          \
        }                                                                                          \
    }

namespace
{
rocprofiler_client_id_t*      client_id        = nullptr;
rocprofiler_client_finalize_t client_fini_func = nullptr;
rocprofiler_context_id_t      kernel_dispatch_ctx{0};
rocprofiler_context_id_t      ompt_ctx{0};

std::atomic<uint64_t> n_ompt_records{0};
std::atomic<uint64_t> n_kernel_dispatch_records{0};

bool
ompt_requested()
{
    const char* v = ::getenv("COEXIST_OMPT");
    return v != nullptr && v[0] != '\0' && std::string{v} != "0";
}

void
tool_tracing_callback(rocprofiler_callback_tracing_record_t record,
                      rocprofiler_user_data_t* /*user_data*/,
                      void* /*callback_data*/)
{
    if(record.kind == ROCPROFILER_CALLBACK_TRACING_OMPT)
        ++n_ompt_records;
    else if(record.kind == ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH)
        ++n_kernel_dispatch_records;
}

void
write_summary()
{
    const char* path = ::getenv("COEXIST_CLIENT_OUTPUT");
    FILE*       os   = path ? ::fopen(path, "w") : stderr;
    if(os == nullptr) os = stderr;

    fprintf(os,
            "{\n"
            "  \"coexist-client\": {\n"
            "    \"ompt_requested\": %s,\n"
            "    \"ompt_records\": %llu,\n"
            "    \"kernel_dispatch_records\": %llu\n"
            "  }\n"
            "}\n",
            ompt_requested() ? "true" : "false",
            static_cast<unsigned long long>(n_ompt_records.load()),
            static_cast<unsigned long long>(n_kernel_dispatch_records.load()));
    if(os != stderr) ::fclose(os);
}

int
tool_init(rocprofiler_client_finalize_t fini_func, void* /*tool_data*/)
{
    client_fini_func = fini_func;

    // Always trace kernel dispatches: this is the SDK's non-OMPT work that must
    // continue whether the SDK keeps or defers the OMPT tool role.
    COEXIST_CALL(rocprofiler_create_context(&kernel_dispatch_ctx),
                 "create kernel-dispatch context");
    COEXIST_CALL(
        rocprofiler_configure_callback_tracing_service(kernel_dispatch_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH,
                                                       nullptr,
                                                       0,
                                                       tool_tracing_callback,
                                                       nullptr),
        "configure kernel-dispatch callback tracing");
    COEXIST_CALL(rocprofiler_start_context(kernel_dispatch_ctx), "start kernel-dispatch context");

    // Only register OMPT when asked: this is what flips ompt_service_requested()
    // and therefore the SDK's keep-vs-defer decision.
    if(ompt_requested())
    {
        COEXIST_CALL(rocprofiler_create_context(&ompt_ctx), "create OMPT context");
        COEXIST_CALL(
            rocprofiler_configure_callback_tracing_service(ompt_ctx,
                                                           ROCPROFILER_CALLBACK_TRACING_OMPT,
                                                           nullptr,
                                                           0,
                                                           tool_tracing_callback,
                                                           nullptr),
            "configure OMPT callback tracing");
        COEXIST_CALL(rocprofiler_start_context(ompt_ctx), "start OMPT context");
    }

    return 0;
}

void
tool_fini(void* /*tool_data*/)
{
    write_summary();
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
    id->name  = "coexist-client";
    client_id = id;

    auto info = std::stringstream{};
    info << id->name << " (priority=" << priority << ") using rocprofiler-sdk v" << version << " ("
         << runtime_version << "), OMPT " << (ompt_requested() ? "requested" : "not requested");
    std::clog << info.str() << std::endl;

    std::atexit([]() {
        if(client_fini_func) client_fini_func(*client_id);
    });

    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}
