// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#if defined(hip_api_etw_tracing_client_EXPORTS)
#    define CLIENT_API __declspec(dllexport)
#else
#    define CLIENT_API __declspec(dllimport)
#endif

namespace client
{
// Nothing loads a tool implicitly on Windows: the HIP runtime reports through ETW instead of
// rocprofiler-register, so no dispatch-table handshake ever reaches rocprofiler-sdk. The
// application drives configuration explicitly instead.
CLIENT_API void
setup();

// Flushes the buffer and finalizes, which is what makes the collected records available to
// the validation in tool_fini.
CLIENT_API void
shutdown();

// Records the name of a HIP function the application deliberately called, so that tool_fini
// can assert the ETW op_name survived the round trip back into a
// ROCPROFILER_HIP_RUNTIME_API_ID_* operation.
CLIENT_API void
expect_operation(const char* name);
}  // namespace client
