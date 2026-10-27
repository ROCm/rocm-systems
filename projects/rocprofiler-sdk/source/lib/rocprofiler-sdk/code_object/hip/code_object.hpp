// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/common/logging.hpp"
#include "lib/common/scope_destructor.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/string_entry.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/code_object/code_object.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hsa.h>

#include <amd_comgr/amd_comgr.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocprofiler
{
namespace code_object
{
namespace hip
{
using host_symbol_data_t =
    rocprofiler_callback_tracing_code_object_host_kernel_symbol_register_data_t;
using hip_host_function_map_t        = std::unordered_map<std::string, host_symbol_data_t>;
using isa_names_t                    = std::vector<const std::string*>;
using kernel_symbol_hip_device_map_t = std::unordered_map<std::string, std::string>;
using comgr_code_object_vec_t        = std::vector<amd_comgr_code_object_info_t>;

// Where the arguments a kernel declares itself end inside its kernarg segment. The compiler
// appends its hidden argument block only to kernels that use it, so this boundary is a property of
// each kernel and is not derivable from the segment size. `segment_size` is the size the same
// metadata entry reports, kept so a reader can tell whether it describes the kernel it is asked
// about.
struct kernarg_layout_t
{
    size_t explicit_size{0};
    size_t segment_size{0};

    // A layout describes one kernel of one load, so it only answers for a request reporting the same
    // segment size. Any other size means this is not the kernel the layout was read from -- another
    // kernel sharing the name, or the same source rebuilt with a different argument list -- and the
    // caller is told the boundary is unknown instead of being handed one it never measured.
    std::optional<size_t>
    explicit_size_for(size_t segment) const noexcept
    {
        if(segment != segment_size) return std::nullopt;
        return explicit_size;
    }
};

using kernel_symbol_kernarg_map_t = std::unordered_map<std::string, kernarg_layout_t>;

// What one code object's amdhsa metadata says, read in a single pass over its kernel list. Kernels
// are keyed by both their symbol name and their metadata name, which differ between code object
// versions ("<symbol>.kd" versus the function name).
struct code_object_metadata_t
{
    kernel_symbol_hip_device_map_t kernel_symbol_device_map = {};
    kernel_symbol_kernarg_map_t    kernarg_layout_map       = {};
};

constexpr unsigned HIP_FAT_MAGIC      = 0x48495046;  // HIPF
constexpr unsigned HIP_FAT_MAGIC_HIPK = 0x4B504948;  // HIPK

struct hip_register_data
{
    const void*                    fat_binary               = nullptr;
    void**                         fat_binary_modules       = nullptr;
    hip_host_function_map_t        host_function_map        = {};
    kernel_symbol_hip_device_map_t kernel_symbol_device_map = {};
    // Kernarg layouts by the id of the load they were read from, so that reloading a code object
    // (or two loads that define kernels of the same name) cannot supply another load's boundary.
    std::unordered_map<uint64_t, kernel_symbol_kernarg_map_t> kernel_kernarg_layout_map = {};
};

struct hip_fat_binary_wrapper
{
    unsigned int magic   = 0;
    unsigned int version = 0;
    void*        binary  = nullptr;
    void*        dummy1  = nullptr;
};

constexpr bool
is_valid_fat_binary_wrapper(const hip_fat_binary_wrapper& fbwrapper)
{
    return (fbwrapper.magic == HIP_FAT_MAGIC || fbwrapper.magic == HIP_FAT_MAGIC_HIPK) &&
           fbwrapper.version == 1;
}

constexpr bool
has_hip_fat_binary_payload(const hip_fat_binary_wrapper& fbwrapper)
{
    return fbwrapper.magic == HIP_FAT_MAGIC && fbwrapper.version == 1;
}

comgr_code_object_vec_t
get_isa_offsets(hsa_agent_t hsa_agent, const void* fat_bin);

kernel_symbol_hip_device_map_t
get_kernel_symbol_device_name_map(const amd_comgr_code_object_info_t& isa_offset,
                                  const void*                         fat_bin);

kernel_symbol_hip_device_map_t
get_kernel_symbol_device_name_map_from_executable(const void* executable, size_t executable_size);

// Both metadata maps of an ELF code object image (a loaded .hsaco, not a fat binary). The image is
// only read here: whatever is derived from it has to outlive the call, because the loader reports
// the application's own buffer and the application may free it once the executable is frozen.
code_object_metadata_t
get_metadata_from_executable(const void* executable, size_t executable_size);

}  // namespace hip
}  // namespace code_object
}  // namespace rocprofiler
