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

// Test tracepoint provider: the events the collector unit tests record.

#undef LTTNG_UST_TRACEPOINT_PROVIDER
#define LTTNG_UST_TRACEPOINT_PROVIDER rocprofiler_collector_test

#undef LTTNG_UST_TRACEPOINT_INCLUDE
#define LTTNG_UST_TRACEPOINT_INCLUDE "./test_provider.h"

#if !defined(ROCPROFILER_COLLECTOR_TEST_PROVIDER_H) ||                                             \
    defined(LTTNG_UST_TRACEPOINT_HEADER_MULTI_READ)
#    define ROCPROFILER_COLLECTOR_TEST_PROVIDER_H

#    include <lttng/tracepoint.h>

LTTNG_UST_TRACEPOINT_EVENT(rocprofiler_collector_test,
                           ping,
                           LTTNG_UST_TP_ARGS(int, value),
                           LTTNG_UST_TP_FIELDS(lttng_ust_field_integer(int, value, value)))

LTTNG_UST_TRACEPOINT_ENUM(rocprofiler_collector_test,
                          kind,
                          LTTNG_UST_TP_ENUM_VALUES(lttng_ust_field_enum_value("even", 0)
                                                       lttng_ust_field_enum_value("odd", 1)))

// An API call "outer" that makes a nested call "inner", as <api>_enter / <api>_exit pairs.
LTTNG_UST_TRACEPOINT_EVENT(
    rocprofiler_collector_test,
    outer_enter,
    LTTNG_UST_TP_ARGS(int, index, const char*, label, double, scale),
    LTTNG_UST_TP_FIELDS(
        lttng_ust_field_integer(int, index, index)
            lttng_ust_field_integer(int64_t, negative, -(int64_t) index) lttng_ust_field_string(
                label,
                label) lttng_ust_field_sequence_text(char, text, label, size_t, strlen(label))
                lttng_ust_field_float(double, scale, scale)
                    lttng_ust_field_enum(rocprofiler_collector_test, kind, int, parity, index % 2)))

LTTNG_UST_TRACEPOINT_EVENT(rocprofiler_collector_test,
                           outer_exit,
                           LTTNG_UST_TP_ARGS(int, result),
                           LTTNG_UST_TP_FIELDS(lttng_ust_field_integer(int, result, result)))

LTTNG_UST_TRACEPOINT_EVENT(
    rocprofiler_collector_test,
    inner_enter,
    LTTNG_UST_TP_ARGS(uint64_t, pointer),
    LTTNG_UST_TP_FIELDS(lttng_ust_field_integer_hex(uint64_t, pointer, pointer)))

LTTNG_UST_TRACEPOINT_EVENT(rocprofiler_collector_test,
                           inner_exit,
                           LTTNG_UST_TP_ARGS(int, result),
                           LTTNG_UST_TP_FIELDS(lttng_ust_field_integer(int, result, result)))

#endif

#include <lttng/tracepoint-event.h>
