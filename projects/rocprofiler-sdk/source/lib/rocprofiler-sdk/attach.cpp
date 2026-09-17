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

#include "lib/rocprofiler-sdk/registration.hpp"

#include <rocprofiler-sdk/defines.h>
#include <rocprofiler-sdk/fwd.h>

ROCPROFILER_EXTERN_C_INIT

int
rocprofiler_is_current_client_attachment(void) ROCPROFILER_API;

rocprofiler_status_t
rocprofiler_load_attachment_tool(const char*) ROCPROFILER_API;

rocprofiler_status_t
rocprofiler_attach(void) ROCPROFILER_API;

rocprofiler_status_t
rocprofiler_detach(void) ROCPROFILER_API;

int
rocprofiler_is_current_client_attachment(void)
{
    return rocprofiler::registration::is_initializing_attachment_client() ? 1 : 0;
}

rocprofiler_status_t
rocprofiler_load_attachment_tool(const char* tool_path)
{
    return rocprofiler::registration::load_attachment_tool(tool_path);
}

rocprofiler_status_t
rocprofiler_attach(void)
{
    return rocprofiler::registration::attach();
}

rocprofiler_status_t
rocprofiler_detach(void)
{
    return rocprofiler::registration::detach();
}

ROCPROFILER_EXTERN_C_FINI
