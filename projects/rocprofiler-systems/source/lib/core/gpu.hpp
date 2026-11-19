// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/amd_smi/ainic_feature.hpp"  // defines AINIC_SUPPORTED (consumed by gpu.cpp)

#include <amd_smi/amdsmi.h>
#include <cstdint>
#include <vector>

namespace rocprofsys::gpu
{
void
get_processor_handles();

std::uint32_t
get_processor_count();

bool
vcn_is_device_level_only(std::uint32_t dev_id);

bool
jpeg_is_device_level_only(std::uint32_t dev_id);

bool
is_vcn_busy_supported(std::uint32_t dev_id);

bool
is_jpeg_busy_supported(std::uint32_t dev_id);

bool
is_xgmi_supported(std::uint32_t dev_id);

bool
is_pcie_supported(std::uint32_t dev_id);

struct processors
{
    static std::uint32_t                        total_processor_count;
    static std::vector<amdsmi_processor_handle> processors_list;
    static std::vector<bool>                    vcn_device_level_only;
    static std::vector<bool>                    jpeg_device_level_only;
    static std::vector<bool>                    vcn_busy_supported;
    static std::vector<bool>                    jpeg_busy_supported;
    static std::vector<bool>                    xgmi_supported;
    static std::vector<bool>                    pcie_supported;

private:
    friend void          rocprofsys::gpu::get_processor_handles();
    friend std::uint32_t rocprofsys::gpu::get_processor_count();
    friend bool          rocprofsys::gpu::vcn_is_device_level_only(std::uint32_t dev_id);
    friend bool          rocprofsys::gpu::jpeg_is_device_level_only(std::uint32_t dev_id);
    friend bool          rocprofsys::gpu::is_vcn_busy_supported(std::uint32_t dev_id);
    friend bool          rocprofsys::gpu::is_jpeg_busy_supported(std::uint32_t dev_id);
    friend bool          rocprofsys::gpu::is_xgmi_supported(std::uint32_t dev_id);
    friend bool          rocprofsys::gpu::is_pcie_supported(std::uint32_t dev_id);
};

int
device_count();

bool
initialize_amdsmi();

void
add_device_metadata();
}  // namespace rocprofsys::gpu
