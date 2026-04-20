// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <amd_smi/amdsmi.h>

namespace rocprofsys
{
namespace gpu
{
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
    static uint32_t                             total_ainic_count;
    static std::vector<amdsmi_processor_handle> ainic_list;

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

int
device_count();

bool
initialize_amdsmi();

bool
reinitialize_amdsmi();

void
add_device_metadata();
}  // namespace gpu
}  // namespace rocprofsys
