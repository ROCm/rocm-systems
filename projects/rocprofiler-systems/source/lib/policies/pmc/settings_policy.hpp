// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

namespace rocprofsys::policies::pmc
{
/** The settings the CPU traits read. */
template <typename Settings>
concept cpu_settings = requires {
    Settings::get_cpu_device_filter();
    Settings::get_cpu_enabled_metrics();
};

/** The settings the GPU (AMD SMI) traits read. */
template <typename Settings>
concept gpu_smi_settings = requires {
    Settings::get_gpu_device_filter();
    Settings::get_visible_gpu_bdfs();
    Settings::get_enabled_metrics();
};

/** The settings the NIC traits read. */
template <typename Settings>
concept nic_settings = requires {
    Settings::get_nic_device_filter();
    Settings::get_nic_enabled_metrics();
};

/** The settings the hipFile traits read. */
template <typename Settings>
concept hipfile_settings = requires {
    Settings::get_gpu_device_filter();
    Settings::get_visible_gpu_type_indices();
    Settings::get_hipfile_enabled_metrics();
};
}  // namespace rocprofsys::policies::pmc
