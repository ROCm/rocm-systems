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

#include <rocprofiler-sdk/experimental/registration.h>
#include <rocprofiler-sdk/registration.h>

#include <stdio.h>

static int
tool_initialize(rocprofiler_client_finalize_t finalize_func, void* tool_data)
{
    (void) finalize_func;
    (void) tool_data;
    return 0;
}

static void
tool_finalize(void* tool_data)
{
    (void) tool_data;
}

static int
tool_attach(rocprofiler_client_detach_t detach_func,
            rocprofiler_context_id_t*   context_ids,
            uint64_t                    context_count,
            void*                       tool_data)
{
    (void) detach_func;
    (void) context_ids;
    (void) context_count;
    (void) tool_data;
    return 0;
}

static void
tool_detach(void* tool_data)
{
    (void) tool_data;
}

rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* client_id)
{
    client_id->name = "Test C attachment tool";
    printf("Test C attachment tool (priority=%u) is using rocprofiler-sdk v%u (%s)\n",
           priority,
           version,
           runtime_version);

    static rocprofiler_tool_configure_result_t result = {
        sizeof(rocprofiler_tool_configure_result_t),
        tool_initialize,
        tool_finalize,
        NULL,
    };
    return &result;
}

rocprofiler_tool_configure_attach_result_t*
rocprofiler_configure_attach(uint32_t                 version,
                             const char*              runtime_version,
                             uint32_t                 priority,
                             rocprofiler_client_id_t* client_id)
{
    (void) version;
    (void) runtime_version;
    (void) priority;
    (void) client_id;

    static rocprofiler_tool_configure_attach_result_t result = {
        sizeof(rocprofiler_tool_configure_attach_result_t),
        tool_attach,
        tool_detach,
        NULL,
    };
    return &result;
}
