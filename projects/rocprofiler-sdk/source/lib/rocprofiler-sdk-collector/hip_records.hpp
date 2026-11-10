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

#pragma once

#include "lib/rocprofiler-sdk-collector/clock.hpp"
#include "lib/rocprofiler-sdk-collector/join.hpp"
#include "lib/rocprofiler-sdk-collector/schema.hpp"

#include <rocprofiler-sdk/buffer_tracing.h>

#include <cstdint>
#include <string>
#include <unordered_set>

namespace rocprofiler
{
namespace collector
{
// Text the records point to (string arguments, string results). Each distinct string is
// kept once, at a stable address, for the life of the store.
struct string_store
{
    std::unordered_set<std::string> values = {};
};

const char*
intern_string(string_store* store, const std::string& value);

// rocprofiler-sdk's buffer tracing kind and operation for a HIP API, from the HIP APIs
// rocprofiler-sdk traces (lib/rocprofiler-sdk/hip/hip.def.cpp). False if it traces none of
// that name.
bool
find_hip_operation(const std::string&                 api,
                   rocprofiler_buffer_tracing_kind_t* kind,
                   rocprofiler_tracing_operation_t*   operation);

struct hip_record_status
{
    bool     known_api    = false;  // rocprofiler-sdk traces the API
    uint32_t missing_args = 0;      // arguments left zero: no matching probe field
};

// A HIP API buffer record, as rocprofiler-sdk writes in process, for a joined call: kind,
// operation, correlation ids (the call's, and its parent's as the ancestor), thread,
// start and end converted to CLOCK_BOOTTIME, and the args and retval unions rebuilt from
// the probe's fields. Pointers keep the target's addresses (they are only printed); string
// arguments point into strings. enter and exit are the probe schema's events of the API
// (exit may be null). A call without an end gets end = start.
hip_record_status
make_hip_api_record(const api_call&                                  call,
                    const schema_event&                              enter,
                    const schema_event*                              exit,
                    const clock_conversion&                          clocks,
                    string_store*                                    strings,
                    rocprofiler_buffer_tracing_hip_api_ext_record_t* record);
}  // namespace collector
}  // namespace rocprofiler
