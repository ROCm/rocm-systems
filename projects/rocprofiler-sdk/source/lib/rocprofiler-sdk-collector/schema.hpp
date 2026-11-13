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
#include <string>
#include <vector>

namespace rocprofiler
{
namespace collector
{
// The probe schema a runtime installs with its probes (share/rocm-trace/<provider>.json,
// written by the rocm-trace generator): for each event, what its fields are.
struct schema_field
{
    std::string              name    = {};  // the event field
    std::string              role    = {};  // argument, result, output, exception
    std::string              param   = {};  // the API parameter it comes from
    std::vector<std::string> path    = {};  // member path in a flattened struct (indices as text)
    std::string              capture = {};  // lowest capture level that records it
};

struct schema_event
{
    std::string               api    = {};
    std::vector<schema_field> fields = {};  // in event order
    // The API's parameters with role "argument", in order (each once).
    std::vector<std::string> params = {};
};

struct probe_schema
{
    std::string                         provider = {};
    std::string                         version  = {};
    std::map<std::string, schema_event> events   = {};  // by event name ("hipMalloc_enter")
};

bool
load_probe_schema(const std::string& path, probe_schema* schema, std::string* error);

// The field of an event's schema by name, or nullptr.
const schema_field*
find_schema_field(const schema_event& event, const std::string& name);
}  // namespace collector
}  // namespace rocprofiler
