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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// rocattach tool which, like rocprofiler-systems, finalizes itself from tool_detach by calling
// the client finalize function it received in initialize. The client must stay registered so the
// process can be attached again.

#include <rocprofiler-sdk/experimental/registration.h>
#include <rocprofiler-sdk/registration.h>

#include <atomic>
#include <cstdio>
#include <optional>

namespace
{
rocprofiler_client_finalize_t          client_finalizer = nullptr;
std::optional<rocprofiler_client_id_t> client_id        = {};
std::atomic<unsigned>                  sessions{0};

int
tool_init(rocprofiler_client_finalize_t fini_func, void*)
{
    client_finalizer = fini_func;
    return 0;
}

void
tool_fini(void*)
{
    std::printf("reattach-tool: tool_fini after %u session(s)\n", sessions.load());
    std::fflush(stdout);
}

int
tool_attach(rocprofiler_client_detach_t, rocprofiler_context_id_t*, uint64_t, void*)
{
    std::printf("reattach-tool: tool_attach session %u\n", ++sessions);
    std::fflush(stdout);
    return 0;
}

void
tool_detach(void*)
{
    std::printf("reattach-tool: tool_detach session %u\n", sessions.load());
    std::fflush(stdout);
    if(client_finalizer && client_id) client_finalizer(*client_id);
}
}  // namespace

extern "C" {
rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t* identifier)
{
    identifier->name = "rocattach-reattach-tool";
    client_id.emplace(*identifier);
    static auto result = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), tool_init, tool_fini, nullptr};
    return &result;
}

rocprofiler_tool_configure_attach_result_t*
rocprofiler_configure_attach(uint32_t, const char*, uint32_t, rocprofiler_client_id_t*)
{
    static auto result = rocprofiler_tool_configure_attach_result_t{
        sizeof(rocprofiler_tool_configure_attach_result_t), tool_attach, tool_detach, nullptr};
    return &result;
}
}
