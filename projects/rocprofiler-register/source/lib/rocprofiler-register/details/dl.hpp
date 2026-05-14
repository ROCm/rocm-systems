// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofiler_register
{
namespace binary
{
// open_modes_vec_t is a vestigial Linux concept (vector of dlopen flags)
// preserved here so the get_linked_path overload signature stays stable for
// callers. The platform::loader implementation does not consult these flags
// on Windows (PE/COFF has no RTLD_LAZY/RTLD_NOLOAD analogue).
using open_modes_vec_t = std::vector<int>;

struct address_range
{
    std::uintptr_t start = 0;
    std::uintptr_t last  = 0;
};

struct segment_address_ranges
{
    std::string                filepath = {};
    std::vector<address_range> ranges   = {};
};

// Enumerate the address ranges of every module loaded into this process.
// Linux: parses /proc/self/maps. Windows: EnumProcessModulesEx +
// GetModuleInformation. Each returned segment has filepath set and one or
// more ranges (Linux: one per ELF segment; Windows: one covering SizeOfImage).
std::vector<segment_address_ranges>
get_segment_addresses();

// Translate a generic library name to its absolute filesystem path, if
// loadable. Linux: dlopen(name, RTLD_LAZY | RTLD_NOLOAD) then dlinfo. Windows:
// GetModuleHandleW(name) (preferred) then a transient LoadLibraryW(name) +
// GetModuleFileNameW + FreeLibrary fallback. Returns nullopt if the library
// cannot be located. The open_modes argument is preserved for source
// compatibility with prior Linux code; it is unused on Windows.
std::optional<std::string>
get_linked_path(std::string_view, open_modes_vec_t&& = {});

}  // namespace binary
}  // namespace rocprofiler_register
