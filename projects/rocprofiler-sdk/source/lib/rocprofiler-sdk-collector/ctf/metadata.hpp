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

#include <string>
#include <string_view>

namespace rocprofiler
{
namespace collector
{
namespace ctf
{
// The TSDL text of a metadata file: LTTng writes it in packets (a header with magic
// 0x75D11D57, then text), or as plain text. Returns false with *error set on a malformed
// packet.
bool
metadata_text(std::string_view file_contents, std::string* text, std::string* error);

// Parse TSDL (CTF 1.8) as LTTng-UST 2.13 writes it. Constructs outside that subset, a
// big-endian trace, or a trace with more than one clock are errors, never guessed.
bool
parse_metadata(std::string_view text, trace_description* trace, std::string* error);
}  // namespace ctf
}  // namespace collector
}  // namespace rocprofiler
