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

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// The CTF 1.8 subset LTTng-UST 2.13 writes: field types, decoded values, and the trace
// description parsed from a trace's metadata.
namespace rocprofiler
{
namespace collector
{
namespace ctf
{
enum class type_kind
{
    integer,
    floating_point,
    string,
    structure,
    variant,
    array,
    sequence,
    enumeration,
};

struct type;
using type_ptr = std::shared_ptr<const type>;

// One named member of a struct or variant.
struct member
{
    std::string name = {};  // as declared; a leading '_' is the TSDL escape (see field_name)
    type_ptr    type = {};
};

// An enumeration label: values lo..hi (inclusive) map to it.
struct enumerator
{
    std::string label = {};
    int64_t     lo    = 0;
    int64_t     hi    = 0;
};

struct type
{
    type_kind kind      = type_kind::integer;
    uint32_t  alignment = 8;  // bits

    // integer, and the container of an enumeration
    uint32_t size       = 0;  // bits
    bool     is_signed  = false;
    uint32_t base       = 10;
    bool     is_text    = false;  // encoding = UTF8 / ASCII (a char in a string array)
    bool     maps_clock = false;  // map = clock.<name>.value: a timestamp
    // floating_point
    uint32_t exp_dig  = 0;
    uint32_t mant_dig = 0;
    // structure, variant
    std::vector<member> members = {};
    std::string         tag     = {};  // variant: the enumeration field that selects
    // array, sequence
    type_ptr    element     = {};
    uint64_t    length      = 0;   // array
    std::string length_name = {};  // sequence: the integer field holding the length
    // enumeration
    type_ptr                container   = {};
    std::vector<enumerator> enumerators = {};
};

// The event field name for a declared name: TSDL escapes names that could be keywords
// with a leading underscore, which readers strip (as babeltrace2 does).
std::string
field_name(const std::string& declared);

struct clock_description
{
    std::string name      = {};
    uint64_t    frequency = 1000000000;  // Hz
    int64_t     offset_s  = 0;
    int64_t     offset    = 0;  // cycles, added to offset_s
};

struct event_description
{
    std::string name      = {};  // "<provider>:<event>" as the tracer wrote it
    uint64_t    id        = 0;
    uint64_t    stream_id = 0;
    int64_t     loglevel  = -1;
    type_ptr    context   = {};  // event-specific context, may be null
    type_ptr    fields    = {};  // payload, a structure
};

struct stream_description
{
    uint64_t id             = 0;
    type_ptr packet_context = {};
    type_ptr event_header   = {};
    type_ptr event_context  = {};  // stream event context (vpid, vtid), may be null
};

struct trace_description
{
    uint32_t                               major         = 0;
    uint32_t                               minor         = 0;
    bool                                   little_endian = true;
    type_ptr                               packet_header = {};
    std::map<std::string, std::string>     env           = {};
    clock_description                      clock         = {};
    std::map<uint64_t, stream_description> streams       = {};
    std::map<uint64_t, event_description>  events        = {};  // by event id (one stream)
};

// A decoded scalar or string field of an event (structures are flattened, see reader).
enum class value_kind
{
    signed_integer,
    unsigned_integer,
    floating_point,
    string,
    unsigned_array,  // arrays and sequences of integers (not text)
};

struct field_value
{
    std::string           name       = {};
    value_kind            kind       = value_kind::unsigned_integer;
    int64_t               i64        = 0;
    uint64_t              u64        = 0;
    double                f64        = 0.0;
    std::string           str        = {};
    std::vector<uint64_t> elements   = {};
    std::string           enum_label = {};  // the label, for an enumeration field
};
}  // namespace ctf
}  // namespace collector
}  // namespace rocprofiler
