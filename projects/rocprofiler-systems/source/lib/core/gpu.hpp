// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#if ROCPROFSYS_USE_ROCM > 0
#    include <amd_smi/amdsmi.h>
#endif

namespace rocprofsys
{
namespace gpu
{
#if ROCPROFSYS_USE_ROCM > 0
void
get_processor_handles();

uint32_t
get_processor_count();

amdsmi_processor_handle
get_handle_from_id(uint32_t dev_id);

bool
vcn_is_device_level_only(uint32_t dev_id);

bool
jpeg_is_device_level_only(uint32_t dev_id);

bool
is_vcn_busy_supported(uint32_t dev_id);

bool
is_jpeg_busy_supported(uint32_t dev_id);

bool
is_xgmi_supported(uint32_t dev_id);

bool
is_pcie_supported(uint32_t dev_id);

struct processors
{
    static uint32_t                             total_processor_count;
    static std::vector<amdsmi_processor_handle> processors_list;
    static std::vector<bool>                    vcn_device_level_only;
    static std::vector<bool>                    jpeg_device_level_only;
    static std::vector<bool>                    vcn_busy_supported;
    static std::vector<bool>                    jpeg_busy_supported;
    static std::vector<bool>                    xgmi_supported;
    static std::vector<bool>                    pcie_supported;

private:
    friend void                    rocprofsys::gpu::get_processor_handles();
    friend uint32_t                rocprofsys::gpu::get_processor_count();
    friend amdsmi_processor_handle rocprofsys::gpu::get_handle_from_id(uint32_t dev_id);
    friend bool rocprofsys::gpu::vcn_is_device_level_only(uint32_t dev_id);
    friend bool rocprofsys::gpu::jpeg_is_device_level_only(uint32_t dev_id);
    friend bool rocprofsys::gpu::is_vcn_busy_supported(uint32_t dev_id);
    friend bool rocprofsys::gpu::is_jpeg_busy_supported(uint32_t dev_id);
    friend bool rocprofsys::gpu::is_xgmi_supported(uint32_t dev_id);
    friend bool rocprofsys::gpu::is_pcie_supported(uint32_t dev_id);
};
#endif

int
device_count();

bool
initialize_amdsmi();

void
add_device_metadata();
}  // namespace gpu
}  // namespace rocprofsys
