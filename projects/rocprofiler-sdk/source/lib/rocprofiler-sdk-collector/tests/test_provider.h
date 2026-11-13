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

#endif

#include <lttng/tracepoint-event.h>
