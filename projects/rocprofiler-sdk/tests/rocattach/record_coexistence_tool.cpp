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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "../common/defines.hpp"

#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>

namespace
{
auto dispatch_count = std::atomic<uint64_t>{0};
auto context_id     = rocprofiler_context_id_t{};
auto buffer_id      = rocprofiler_buffer_id_t{};
char output_path[4096];

uint64_t
flush_and_get_count()
{
    if(buffer_id.handle != 0 && rocprofiler_flush_buffer(buffer_id) != ROCPROFILER_STATUS_SUCCESS)
        std::fprintf(stderr, "Record coexistence tool FAILED to flush its buffer\n");
    return dispatch_count.load();
}

void
buffer_callback(rocprofiler_context_id_t,
                rocprofiler_buffer_id_t,
                rocprofiler_record_header_t** headers,
                size_t                        num_headers,
                void*,
                uint64_t)
{
    for(size_t i = 0; i < num_headers; ++i)
    {
        if(headers[i] != nullptr && headers[i]->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
           headers[i]->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
            ++dispatch_count;
    }
}

int
tool_initialize(rocprofiler_client_finalize_t, void*)
{
    if(rocprofiler_create_context(&context_id) != ROCPROFILER_STATUS_SUCCESS ||
       rocprofiler_create_buffer(context_id,
                                 4096,
                                 2048,
                                 ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                 buffer_callback,
                                 nullptr,
                                 &buffer_id) != ROCPROFILER_STATUS_SUCCESS ||
       rocprofiler_configure_buffer_tracing_service(
           context_id, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, nullptr, 0, buffer_id) !=
           ROCPROFILER_STATUS_SUCCESS ||
       rocprofiler_start_context(context_id) != ROCPROFILER_STATUS_SUCCESS)
    {
        std::fprintf(stderr, "Record coexistence tool FAILED to initialize\n");
        return -1;
    }
    return 0;
}

void
tool_finalize(void*)
{
    const auto count = flush_and_get_count();
    std::printf("Record coexistence tool writing %" PRIu64 " records to %s\n", count, output_path);
    auto* output = std::fopen(output_path, "w");
    if(output == nullptr)
    {
        std::fprintf(stderr, "Record coexistence tool FAILED to open %s\n", output_path);
        return;
    }
    std::fprintf(output, "%" PRIu64 "\n", count);
    std::fclose(output);
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t* client_id)
{
    client_id->name  = "record-coexistence-tool";
    const auto* path = std::getenv("ROCPROFILER_RECORD_TOOL_OUTPUT");
    std::snprintf(output_path,
                  sizeof(output_path),
                  "%s",
                  (path != nullptr) ? path : "record-coexistence-tool.txt");

    static auto result = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t),
        tool_initialize,
        tool_finalize,
        nullptr,
    };
    return &result;
}

extern "C" uint64_t
rocprofiler_test_record_count() ROCPROFILER_TEST_PUBLIC_API;

extern "C" uint64_t
rocprofiler_test_record_count()
{
    return flush_and_get_count();
}
