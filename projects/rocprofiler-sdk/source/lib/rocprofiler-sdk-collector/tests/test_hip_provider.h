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

// A stand-in for the HIP runtime's probes (provider rocm_hip), with the event and field
// names of two of them, so the tests can read HIP API records without a GPU. Its schema is
// test_schema/rocm_hip.json.

#undef LTTNG_UST_TRACEPOINT_PROVIDER
#define LTTNG_UST_TRACEPOINT_PROVIDER rocm_hip

#undef LTTNG_UST_TRACEPOINT_INCLUDE
#define LTTNG_UST_TRACEPOINT_INCLUDE "./test_hip_provider.h"

#if !defined(ROCPROFILER_COLLECTOR_TEST_HIP_PROVIDER_H) ||                                         \
    defined(LTTNG_UST_TRACEPOINT_HEADER_MULTI_READ)
#    define ROCPROFILER_COLLECTOR_TEST_HIP_PROVIDER_H

#    include <lttng/tracepoint.h>

#    include <stdint.h>
#    include <string.h>

LTTNG_UST_TRACEPOINT_EVENT(
    rocm_hip,
    hipMemcpyAsync_enter,
    LTTNG_UST_TP_ARGS(uint64_t,
                      dst,
                      uint64_t,
                      src,
                      uint64_t,
                      size,
                      int32_t,
                      kind,
                      uint64_t,
                      stream),
    LTTNG_UST_TP_FIELDS(lttng_ust_field_integer_hex(uint64_t, dst, dst)
                            lttng_ust_field_integer_hex(uint64_t, src, src)
                                lttng_ust_field_integer(uint64_t, sizeBytes, size)
                                    lttng_ust_field_integer(int32_t, kind, kind)
                                        lttng_ust_field_integer_hex(uint64_t, stream, stream)))

LTTNG_UST_TRACEPOINT_EVENT(rocm_hip,
                           hipMemcpyAsync_exit,
                           LTTNG_UST_TP_ARGS(int32_t, result),
                           LTTNG_UST_TP_FIELDS(lttng_ust_field_integer(int32_t, result, result)))

LTTNG_UST_TRACEPOINT_EVENT(
    rocm_hip,
    hipModuleGetFunction_enter,
    LTTNG_UST_TP_ARGS(uint64_t, function, uint64_t, module, const char*, kname),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_integer_hex(uint64_t, function, function)
            lttng_ust_field_integer_hex(uint64_t, module, module)
                lttng_ust_field_sequence_text(char, kname, kname, uint16_t, strlen(kname))))

LTTNG_UST_TRACEPOINT_EVENT(rocm_hip,
                           hipModuleGetFunction_exit,
                           LTTNG_UST_TP_ARGS(int32_t, result),
                           LTTNG_UST_TP_FIELDS(lttng_ust_field_integer(int32_t, result, result)))

#endif

#include <lttng/tracepoint-event.h>
