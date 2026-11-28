// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include "lib/rocprofiler-sdk/code_object/hsa/code_object.hpp"
#include "lib/rocprofiler-sdk/code_object/hsa/kernel_symbol.hpp"

#include "lib/rocprofiler-sdk-attach/table.h"

#include <hsa/hsa_api_trace.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofiler
{
namespace code_object
{
using code_object_array_t    = std::vector<std::unique_ptr<hsa::code_object>>;
using code_object_iterator_t = std::function<void(const hsa::code_object&)>;

const char*
name_by_id(uint32_t id);

uint32_t
id_by_name(const char* name);

std::vector<const char*>
get_names();

std::vector<uint32_t>
get_ids();

uint64_t
get_kernel_id(uint64_t kernel_object);

// A range of a kernarg segment, in bytes from its start.
struct kernarg_region_t
{
    uint32_t offset{0};
    uint32_t size{0};

    friend bool operator==(const kernarg_region_t& lhs, const kernarg_region_t& rhs) noexcept
    {
        return lhs.offset == rhs.offset && lhs.size == rhs.size;
    }
};

// A loaded kernel, found by the kernel_object of a dispatch packet.
struct kernel_locator_t
{
    uint64_t           kernel_id{0};
    size_t             segment_size{0};  // kernarg segment size the load reported
    const std::string* name{nullptr};    // interned in the process-lifetime string table
    // The ranges of the kernarg segment that can hold a pointer, read from the kernel's amdhsa
    // metadata when its code object was loaded; null when that metadata was not readable.
    std::shared_ptr<const std::vector<kernarg_region_t>> pointer_regions{};
};

// The kernel whose descriptor handle is `kernel_object`, or nullopt when no loaded kernel carries
// it.
std::optional<kernel_locator_t>
locate_kernel(uint64_t kernel_object);

// Layout of the AMDHSA kernel descriptor fields read for kernel symbol register counts.
struct kernel_descriptor_t
{
    uint8_t  reserved0[16];
    int64_t  kernel_code_entry_byte_offset = 0;
    uint8_t  reserved1[20];
    uint32_t compute_pgm_rsrc3      = 0;
    uint32_t compute_pgm_rsrc1      = 0;
    uint32_t compute_pgm_rsrc2      = 0;
    uint16_t kernel_code_properties = 0;
    uint8_t  reserved2[6];
};

// Architected VGPRs a kernel allocates on the agent named `name` (e.g. "gfx90a").
uint32_t
arch_vgpr_count(std::string_view name, kernel_descriptor_t kernel_code);

// Accumulation VGPRs (AGPRs) a kernel allocates on the agent named `name`; 0 where the
// architecture has none or is not supported.
uint32_t
accum_vgpr_count(std::string_view name, kernel_descriptor_t kernel_code);

void
iterate_loaded_code_objects(code_object_iterator_t&& func);

void
initialize(HsaApiTable* table);

void
initialize(HipCompilerDispatchTable* table);

void
finalize();

void
initialize(RocAttachDispatchTable* table);

}  // namespace code_object
}  // namespace rocprofiler
