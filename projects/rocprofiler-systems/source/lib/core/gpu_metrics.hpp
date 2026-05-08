// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <vector>

namespace rocprofsys
{
namespace gpu
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
 * @brief Serializes GPU metrics into a compact binary format
 *
 * Serialization format:
 * 1. Support flags byte (1 byte):
 *      - bit 0: vcn_is_device_level_only (device-level vs per-XCP)
 *      - bit 1: jpeg_is_device_level_only (device-level vs per-XCP)
 *      - bits 2-7: reserved
 * 2. Data element counts (6 bytes):
 *      - vcn_count (1 byte): total VCN values (flattened across all XCPs)
 *      - jpeg_count (1 byte): total JPEG values (flattened across all XCPs)
 *      - vcn_xcp_count (1 byte): number of XCPs with VCN data
 *      - jpeg_xcp_count (1 byte): number of XCPs with JPEG data
 *      - xgmi_read_count (1 byte): number of XGMI read data values
 *      - xgmi_write_count (1 byte): number of XGMI write data values
 * 3. Per-XCP size arrays (variable):
 *      - vcn_xcp_sizes[0..vcn_xcp_count-1]: size of each XCP's VCN data (1 byte each)
 *      - jpeg_xcp_sizes[0..jpeg_xcp_count-1]: size of each XCP's JPEG data (1 byte each)
 * 4. Flattened data arrays (conditionally serialized based on settings):
 *      - VCN data (if vcn_activity setting enabled): flattened uint16 values
 *      - JPEG data (if jpeg_activity setting enabled): flattened uint16 values
 *      - XGMI data (if xgmi setting enabled):
 *          link_width (uint16), link_speed (uint16)
 *          xgmi_read_data array (uint64[xgmi_read_count])
 *          xgmi_write_data array (uint64[xgmi_write_count])
 *      - PCIe data (if pcie setting enabled):
 *          link_width (uint16), link_speed (uint16)
 *          bandwidth_acc (uint64), bandwidth_inst (uint64)
 *
 * @param metrics GPU metrics to serialize
 * @param capabilities Capability flags (vcn/jpeg device-level status)
 * @param settings Controls which metrics to include in serialization
 * @return Binary serialized data
 */
std::vector<std::uint8_t>
serialize_gpu_metrics(const gpu_metrics_t&              metrics,
                      const gpu_metrics_capabilities_t& capabilities,
                      const gpu_metrics_settings_t&     settings);

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

}  // namespace gpu
}  // namespace rocprofsys
