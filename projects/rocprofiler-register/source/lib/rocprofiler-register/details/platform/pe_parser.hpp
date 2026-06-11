// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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
#include <string>
#include <vector>

namespace rocprofiler_register
{
namespace platform
{
// Per-page memory region as reported by VirtualQuery on Windows. Provides a
// rough analogue of /proc/self/maps entries for use by the secure-mode
// validator in rocprofiler_register.cpp.
//
// On Linux this header still exists and is included by callers; the Linux
// implementation reports an empty vector and conservative (false/zero)
// answers — secure-mode validation on Linux uses get_segment_addresses
// directly and does not rely on per-page protection metadata.
struct memory_region
{
    std::uintptr_t base_address  = 0;
    std::size_t    size          = 0;
    std::uint32_t  protection    = 0;
    std::uint32_t  state         = 0;
    std::uint32_t  type          = 0;
    std::string    module_path   = {};
    bool           is_executable = false;
    bool           is_readable   = false;
    bool           is_writable   = false;
};

// Enumerate every committed memory region in the current process. On Linux
// this returns an empty vector.
std::vector<memory_region>
get_memory_regions();

// Return the protection flags (PAGE_EXECUTE_READ, etc.) for the page that
// contains the given address. Returns 0 if the address is not committed.
// On Linux, returns 0 unconditionally.
std::uint32_t
get_memory_protection(const void* address);

// True if the page containing the address is committed and executable.
bool
is_address_executable(const void* address);

// True if the page containing the address is committed and readable.
bool
is_address_readable(const void* address);

// Return the base address (HMODULE) of the loaded module that contains
// the given address, or nullptr if the address is not within any loaded
// image. On Linux, returns nullptr.
void*
find_module_for_address(const void* address);

}  // namespace platform
}  // namespace rocprofiler_register
