// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include <cstddef>
#include <cstdint>
#include <string>

#if ROCPROFSYS_USE_ROCM > 0
#    include <amd_smi/amdsmi.h>
#    include <rocprofiler-sdk/agent.h>
#endif

namespace rocprofsys
{

enum class agent_type : uint8_t
{
    CPU,  ///< Agent type is a CPU
    GPU   ///< Agent type is a GPU
};

struct agent
{
    agent_type  type;
    uint64_t    id;
    uint32_t    node_id;
    int32_t     logical_node_id;
    int32_t     logical_node_type_id;
    std::string name;
    std::string model_name;
    std::string vendor_name;
    std::string product_name;

    size_t device_id{ 0 };
    size_t base_id{ 0 };
#if ROCPROFSYS_USE_ROCM > 0
    amdsmi_processor_handle smi_handle = nullptr;
#endif
};

}  // namespace rocprofsys
