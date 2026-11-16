// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <vector>

namespace rocprofsys::gpu
{
/// GPU metrics data structure for VCN, JPEG, XGMI, and PCIe metrics
struct gpu_metrics_t
{
    // VCN metrics
    std::vector<std::uint16_t> vcn_activity;  // Device-level VCN (when supported)
    std::vector<std::vector<std::uint16_t>>
        vcn_busy;  // XCP-level VCN (per-XCP organization)

    // JPEG metrics
    std::vector<std::uint16_t> jpeg_activity;  // Device-level JPEG (when supported)
    std::vector<std::vector<std::uint16_t>>
        jpeg_busy;  // XCP-level JPEG (per-XCP organization)

    // XGMI metrics
    std::uint16_t              xgmi_link_width = 0;
    std::uint16_t              xgmi_link_speed = 0;
    std::vector<std::uint64_t> xgmi_read_data_acc;
    std::vector<std::uint64_t> xgmi_write_data_acc;

    // PCIe metrics
    std::uint16_t pcie_link_width     = 0;
    std::uint16_t pcie_link_speed     = 0;
    std::uint64_t pcie_bandwidth_acc  = 0;
    std::uint64_t pcie_bandwidth_inst = 0;
};

/// Settings structure for controlling which metrics are serialized
struct gpu_metrics_settings_t
{
    bool vcn_activity  = true;
    bool jpeg_activity = true;
    bool xgmi          = true;
    bool pcie          = true;
};

/// GPU metrics capabilities structure with bitfield flags
struct gpu_metrics_capabilities_t
{
    struct flags_t
    {
        std::uint8_t vcn_is_device_level_only : 1;  ///< VCN is device-level (vs per-XCP)
        std::uint8_t
            jpeg_is_device_level_only : 1;  ///< JPEG is device-level (vs per-XCP)
        std::uint8_t reserved         : 6;  ///< Reserved for future use
    };

    union
    {
        flags_t      flags;
        std::uint8_t value;  ///< Raw byte value for easy serialization
    };

    /// Default constructor - initializes all flags to zero
    gpu_metrics_capabilities_t()
    : value(0)
    {}
};

/**
 * @brief Deserializes GPU metrics from binary format
 *
 * @param serialized_data Binary data to deserialize
 * @param result Output GPU metrics structure
 * @param is_vcn_enabled Whether to deserialize VCN data
 * @param is_jpeg_enabled Whether to deserialize JPEG data
 * @param is_xgmi_enabled Whether to deserialize XGMI data
 * @param is_pcie_enabled Whether to deserialize PCIe data
 * @param capabilities Output: capability flags (vcn/jpeg device-level status)
 * @throws std::runtime_error if serialized data is invalid
 */
void
deserialize_gpu_metrics(const std::vector<std::uint8_t>& serialized_data,
                        gpu_metrics_t& result, bool is_vcn_enabled, bool is_jpeg_enabled,
                        bool is_xgmi_enabled, bool is_pcie_enabled,
                        gpu_metrics_capabilities_t& capabilities);

}  // namespace rocprofsys::gpu
