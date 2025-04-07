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

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <string>

#include "lib/common/filesystem.hpp"

namespace rocprofiler
{
namespace counters
{
namespace testing
{
struct CodeObject
{
    hsa_file_t               file         = 0;
    hsa_code_object_reader_t code_obj_rdr = {};
    hsa_executable_t         executable   = {};
};

hsa_status_t
load_code_object(const std::string& filename, hsa_agent_t agent, CodeObject& code_object);
struct Kernel
{
    uint64_t handle        = 0;
    uint32_t scratch       = 0;
    uint32_t group         = 0;
    uint32_t kernarg_size  = 0;
    uint32_t kernarg_align = 0;
};

hsa_status_t
get_kernel(const CodeObject&  code_object,
           const std::string& kernel,
           hsa_agent_t        agent,
           Kernel&            kern);

void
search_hasco(const common::filesystem::path& directory, std::string& filename);
}  // namespace testing
}  // namespace counters
}  // namespace rocprofiler
