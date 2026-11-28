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

// Tool for the memory-allocation-op-filter integration test. It traces
// MEMORY_ALLOCATION in several contexts, each with a different operation filter, in
// both callback and buffered mode, and writes the number of ALLOCATE and FREE records
// each context received to a JSON file at finalization. validate.py checks that a
// filtered context receives exactly the records of the selected operation that the
// unfiltered context of the same mode received, and none of the other operations.

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <vector>

#define ROCPROFILER_CALL(result, msg)                                                              \
    {                                                                                              \
        rocprofiler_status_t CHECKSTATUS = result;                                                 \
        if(CHECKSTATUS != ROCPROFILER_STATUS_SUCCESS)                                              \
        {                                                                                          \
            std::fprintf(stderr,                                                                   \
                         "[memory-allocation-op-filter] %s failed: %s\n",                          \
                         msg,                                                                      \
                         rocprofiler_get_status_string(CHECKSTATUS));                              \
            std::abort();                                                                          \
        }                                                                                          \
    }

namespace
{
struct context_counts
{
    std::string_view         name     = {};
    bool                     buffered = false;
    std::atomic<uint64_t>    allocate = {0};
    std::atomic<uint64_t>    free     = {0};
    std::atomic<uint64_t>    other    = {0};
    rocprofiler_context_id_t context  = {.handle = 0};
    rocprofiler_buffer_id_t  buffer   = {.handle = 0};
};

auto g_contexts = std::array<context_counts, 6>{};

void
count_operation(context_counts* counts, uint32_t operation)
{
    if(operation == ROCPROFILER_MEMORY_ALLOCATION_ALLOCATE)
        ++counts->allocate;
    else if(operation == ROCPROFILER_MEMORY_ALLOCATION_FREE)
        ++counts->free;
    else
        ++counts->other;
}

void
callback_record(rocprofiler_callback_tracing_record_t record, rocprofiler_user_data_t*, void* data)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_MEMORY_ALLOCATION) return;
    if(record.phase != ROCPROFILER_CALLBACK_PHASE_EXIT) return;
    count_operation(static_cast<context_counts*>(data), record.operation);
}

void
buffered_records(rocprofiler_context_id_t,
                 rocprofiler_buffer_id_t,
                 rocprofiler_record_header_t** headers,
                 size_t                        num_headers,
                 void*                         data,
                 uint64_t)
{
    for(size_t i = 0; i < num_headers; ++i)
    {
        const auto* header = headers[i];
        if(header->category != ROCPROFILER_BUFFER_CATEGORY_TRACING ||
           header->kind != ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION)
            continue;
        const auto* record =
            static_cast<const rocprofiler_buffer_tracing_memory_allocation_record_t*>(
                header->payload);
        count_operation(static_cast<context_counts*>(data), record->operation);
    }
}

void
configure_context(context_counts&                              counts,
                  std::string_view                             name,
                  bool                                         buffered,
                  std::vector<rocprofiler_tracing_operation_t> operations)
{
    counts.name     = name;
    counts.buffered = buffered;
    ROCPROFILER_CALL(rocprofiler_create_context(&counts.context), "context creation");

    auto* ops     = operations.empty() ? nullptr : operations.data();
    auto  num_ops = operations.size();
    if(buffered)
    {
        ROCPROFILER_CALL(rocprofiler_create_buffer(counts.context,
                                                   4096,
                                                   2048,
                                                   ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                                   buffered_records,
                                                   &counts,
                                                   &counts.buffer),
                         "buffer creation");
        ROCPROFILER_CALL(rocprofiler_configure_buffer_tracing_service(
                             counts.context,
                             ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION,
                             ops,
                             num_ops,
                             counts.buffer),
                         "buffer tracing service configuration");
    }
    else
    {
        ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
                             counts.context,
                             ROCPROFILER_CALLBACK_TRACING_MEMORY_ALLOCATION,
                             ops,
                             num_ops,
                             callback_record,
                             &counts),
                         "callback tracing service configuration");
    }
    ROCPROFILER_CALL(rocprofiler_start_context(counts.context), "context start");
}

int
tool_init(rocprofiler_client_finalize_t, void*)
{
    configure_context(g_contexts.at(0), "callback_all", false, {});
    configure_context(
        g_contexts.at(1), "callback_allocate", false, {ROCPROFILER_MEMORY_ALLOCATION_ALLOCATE});
    configure_context(
        g_contexts.at(2), "callback_free", false, {ROCPROFILER_MEMORY_ALLOCATION_FREE});
    configure_context(g_contexts.at(3), "buffered_all", true, {});
    configure_context(
        g_contexts.at(4), "buffered_allocate", true, {ROCPROFILER_MEMORY_ALLOCATION_ALLOCATE});
    configure_context(
        g_contexts.at(5), "buffered_free", true, {ROCPROFILER_MEMORY_ALLOCATION_FREE});
    return 0;
}

void
tool_fini(void*)
{
    for(auto& itr : g_contexts)
    {
        if(itr.buffer.handle != 0)
            ROCPROFILER_CALL(rocprofiler_flush_buffer(itr.buffer), "buffer flush");
    }

    const auto* filename = std::getenv("ROCPROFILER_TOOL_OUTPUT_FILE");
    if(filename == nullptr) filename = "memory-allocation-op-filter.json";

    auto* ofs = std::fopen(filename, "w");
    if(ofs == nullptr)
    {
        std::fprintf(stderr, "[memory-allocation-op-filter] cannot open %s\n", filename);
        std::abort();
    }

    std::fprintf(ofs, "{\n  \"contexts\": {\n");
    for(size_t i = 0; i < g_contexts.size(); ++i)
    {
        const auto& itr = g_contexts.at(i);
        std::fprintf(ofs,
                     "    \"%.*s\": {\"buffered\": %s, \"allocate\": %lu, \"free\": %lu, "
                     "\"other\": %lu}%s\n",
                     static_cast<int>(itr.name.size()),
                     itr.name.data(),
                     itr.buffered ? "true" : "false",
                     static_cast<unsigned long>(itr.allocate.load()),
                     static_cast<unsigned long>(itr.free.load()),
                     static_cast<unsigned long>(itr.other.load()),
                     (i + 1 < g_contexts.size()) ? "," : "");
    }
    std::fprintf(ofs, "  }\n}\n");
    std::fclose(ofs);
    std::printf("[memory-allocation-op-filter] wrote %s\n", filename);
}
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t* id)
{
    id->name = "memory-allocation-op-filter";

    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}
