// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

// undefine NDEBUG so asserts are implemented
#ifdef NDEBUG
#    undef NDEBUG
#endif

/**
 * @file tests/tools/c-tool.c
 *
 * @brief Example rocprofiler client (tool) written in C
 */

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <stdio.h>
#include <stdlib.h>

static void
thread_precreate(rocprofiler_runtime_library_t lib, void* tool_data)
{
    uint32_t* priority = (uint32_t*) tool_data;

    if(priority && *priority == 0)
    {
        fprintf(
            stderr,
            "Internal thread for rocprofiler-sdk should not be created when all tools return NULL "
            "from rocprofiler_configure\n");
        fflush(stderr);
        abort();
    }

    (void) lib;
}

static uint32_t tool_priority = 0;

rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
    // set the client name
    id->name = "Test C tool";

    tool_priority = priority;

    // compute major/minor/patch version info
    uint32_t major = version / 10000;
    uint32_t minor = (version % 10000) / 100;
    uint32_t patch = version % 100;

    // generate info string
    printf("%s (priority=%u) is using rocprofiler-sdk v%i.%i.%i (%s)\n",
           id->name,
           priority,
           major,
           minor,
           patch,
           runtime_version);

    rocprofiler_at_internal_thread_create(
        thread_precreate, NULL, ROCPROFILER_LIBRARY, &tool_priority);

    // return pointer to configure data
    return NULL;
}
