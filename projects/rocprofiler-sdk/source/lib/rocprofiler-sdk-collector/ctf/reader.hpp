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

#include "lib/rocprofiler-sdk-collector/ctf/types.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace rocprofiler
{
namespace collector
{
namespace ctf
{
// One CTF trace: a directory with a metadata file and its stream files. With per-process
// buffers LTTng writes one per process (ust/pid/<name>-<pid>-<date>/).
struct trace
{
    std::string       path        = {};
    trace_description description = {};
};

using trace_ptr = std::shared_ptr<const trace>;

// One decoded event. Structures are flattened: a nested member is named "outer.inner".
// Sequence length fields are folded into the sequence; text arrays and sequences are
// strings.
struct event_record
{
    const trace*             source      = nullptr;
    const event_description* description = nullptr;
    uint64_t                 timestamp   = 0;   // clock value, in ns (cycles scaled by freq)
    std::vector<field_value> context     = {};  // stream event context (vpid, vtid, ...)
    std::vector<field_value> fields      = {};
};

// Events a stream discarded (its buffers were full) between two packets.
struct discarded_record
{
    const trace* source = nullptr;
    std::string  stream = {};  // the stream file
    uint64_t     count  = 0;
    uint64_t     begin  = 0;  // clock value (ns) of the previous packet's end
    uint64_t     end    = 0;  // clock value (ns) of the packet's end
};

struct read_statistics
{
    uint64_t traces    = 0;
    uint64_t streams   = 0;
    uint64_t packets   = 0;
    uint64_t events    = 0;
    uint64_t discarded = 0;
};

using event_callback     = std::function<void(const event_record&)>;
using discarded_callback = std::function<void(const discarded_record&)>;

// The directories under root holding a CTF trace (a "metadata" file), sorted.
std::vector<std::string>
find_traces(const std::string& root);

// Read and parse a trace directory's metadata.
bool
load_trace(const std::string& path, trace* result, std::string* error);

// Decode every event of the traces, in timestamp order across all their streams (ties
// keep stream order), and report discarded events. Stops at the first decoding error.
bool
read_events(const std::vector<trace_ptr>& traces,
            const event_callback&         on_event,
            const discarded_callback&     on_discarded,
            read_statistics*              statistics,
            std::string*                  error);

// Report the discarded events of the traces from their packet headers alone (no event is
// decoded), so that a reader of the events can know every gap beforehand.
bool
scan_discarded(const std::vector<trace_ptr>& traces,
               const discarded_callback&     on_discarded,
               std::string*                  error);

// The value of a field by (unescaped) name, or nullptr.
const field_value*
find_field(const std::vector<field_value>& values, const std::string& name);
}  // namespace ctf
}  // namespace collector
}  // namespace rocprofiler
