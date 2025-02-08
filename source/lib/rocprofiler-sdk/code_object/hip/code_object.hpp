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

constexpr unsigned HIP_FAT_MAGIC = 0x48495046;  // HIPF

struct hip_register_data
{
    const void*                    fat_binary               = nullptr;
    hip_host_function_map_t        host_function_map        = {};
    kernel_symbol_hip_device_map_t kernel_symbol_device_map = {};
};

struct hip_fat_binary_wrapper
{
    unsigned int magic   = 0;
    unsigned int version = 0;
    void*        binary  = nullptr;
    void*        dummy1  = nullptr;
};

comgr_code_object_vec_t
get_isa_offsets(hsa_agent_t hsa_agent, const void* fat_bin);

kernel_symbol_hip_device_map_t
get_kernel_symbol_device_name_map(const amd_comgr_code_object_info_t& isa_offset,
                                  const void*                         fat_bin);

}  // namespace hip
}  // namespace code_object
}  // namespace rocprofiler
