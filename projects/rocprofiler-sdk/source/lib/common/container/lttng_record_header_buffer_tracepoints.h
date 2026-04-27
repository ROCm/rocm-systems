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

#undef LTTNG_UST_TRACEPOINT_PROVIDER
#define LTTNG_UST_TRACEPOINT_PROVIDER rocprofiler_sdk_buffer

#undef LTTNG_UST_TRACEPOINT_INCLUDE
#define LTTNG_UST_TRACEPOINT_INCLUDE                                                        \
    "lib/common/container/lttng_record_header_buffer_tracepoints.h"

#if !defined(ROCPROFILER_SDK_LTTNG_RECORD_HEADER_BUFFER_TRACEPOINTS_H) ||                  \
    defined(LTTNG_UST_TRACEPOINT_HEADER_MULTI_READ)
#    define ROCPROFILER_SDK_LTTNG_RECORD_HEADER_BUFFER_TRACEPOINTS_H

#    include <lttng/tracepoint.h>

LTTNG_UST_TRACEPOINT_EVENT(
    rocprofiler_sdk_buffer,
    record,
    LTTNG_UST_TP_ARGS(uint32_t,
                      category,
                      uint32_t,
                      kind,
                      uint64_t,
                      hash,
                      uint64_t,
                      payload_size,
                      uint64_t,
                      payload_alignment,
                      uint64_t,
                      sequence),
    LTTNG_UST_TP_FIELDS(lttng_ust_field_integer(uint32_t, category, category)
                            lttng_ust_field_integer(uint32_t, kind, kind)
                                lttng_ust_field_integer(uint64_t, hash, hash)
                                    lttng_ust_field_integer(uint64_t, payload_size, payload_size)
                                        lttng_ust_field_integer(
                                            uint64_t, payload_alignment, payload_alignment)
                                            lttng_ust_field_integer(uint64_t, sequence, sequence)))

#endif

#include <lttng/tracepoint-event.h>
