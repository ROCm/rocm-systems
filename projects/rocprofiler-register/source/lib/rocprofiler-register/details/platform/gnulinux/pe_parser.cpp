// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
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

// pe_parser is a Windows-only concept (per-page protection / PE images). On
// Linux, callers obtain the equivalent information from get_segment_addresses
// (which reads /proc/self/maps). The functions defined here are stubs that
// keep the header link-compatible across platforms.

#include "details/platform/pe_parser.hpp"

namespace rocprofiler_register
{
namespace platform
{
std::vector<memory_region>
get_memory_regions()
{
    return std::vector<memory_region>{};
}

std::uint32_t
get_memory_protection(const void* /*address*/)
{
    return 0;
}

bool
is_address_executable(const void* /*address*/)
{
    return false;
}

bool
is_address_readable(const void* /*address*/)
{
    return false;
}

void*
find_module_for_address(const void* /*address*/)
{
    return nullptr;
}

}  // namespace platform
}  // namespace rocprofiler_register
