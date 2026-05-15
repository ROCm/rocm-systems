// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gpu_metrics.hpp"
#include <cstdint>

#include <stdexcept>

namespace rocprofsys
{
namespace gpu
{
namespace
{
// Helper functions for serialization
void
serialize_uint8(std::vector<std::uint8_t>& data, std::uint8_t val)
{
    data.push_back(val);
}

void
serialize_uint16(std::vector<std::uint8_t>& data, std::uint16_t val)
{
    data.push_back(static_cast<std::uint8_t>(val & 0xFF));
    data.push_back(static_cast<std::uint8_t>((val >> 8) & 0xFF));
}

void
serialize_uint16_vector(std::vector<std::uint8_t>&        data,
                        const std::vector<std::uint16_t>& vec, std::uint8_t count)
{
    for(std::uint8_t i = 0; i < count; ++i)
    {
        data.push_back(static_cast<std::uint8_t>(vec[i] & 0xFF));
        data.push_back(static_cast<std::uint8_t>((vec[i] >> 8) & 0xFF));
    }
}

void
serialize_uint64(std::vector<std::uint8_t>& data, std::uint64_t val)
{
    for(int i = 0; i < 8; ++i)
        data.push_back(static_cast<std::uint8_t>((val >> (i * 8)) & 0xFF));
}

void
serialize_uint64_vector(std::vector<std::uint8_t>&        data,
                        const std::vector<std::uint64_t>& vec, std::uint8_t count)
{
    for(std::uint8_t i = 0; i < count; ++i)
    {
        for(int j = 0; j < 8; ++j)
            data.push_back(static_cast<std::uint8_t>((vec[i] >> (j * 8)) & 0xFF));
    }
}

// Helper functions for deserialization
std::uint8_t
deserialize_uint8(const std::vector<std::uint8_t>& data, size_t& offset)
{
    if(offset >= data.size())
        throw std::runtime_error("Invalid serialized data: unexpected end");
    return data[offset++];
}

std::uint16_t
deserialize_uint16(const std::vector<std::uint8_t>& data, size_t& offset)
{
    if(offset + 1 >= data.size())
        throw std::runtime_error("Invalid serialized data: unexpected end");
    std::uint16_t value = static_cast<std::uint16_t>(data[offset]) |
                          (static_cast<std::uint16_t>(data[offset + 1]) << 8);
    offset += 2;
    return value;
}

std::uint64_t
deserialize_uint64(const std::vector<std::uint8_t>& data, size_t& offset)
{
    if(offset + 7 >= data.size())
        throw std::runtime_error("Invalid serialized data: unexpected end");
    std::uint64_t value = 0;
    for(int i = 0; i < 8; ++i)
        value |= (static_cast<std::uint64_t>(data[offset + i]) << (i * 8));
    offset += 8;
    return value;
}

std::vector<std::uint16_t>
deserialize_uint16_vector(const std::vector<std::uint8_t>& data, size_t& offset,
                          std::uint8_t count)
{
    std::vector<std::uint16_t> values;
    values.reserve(count);
    for(std::uint8_t i = 0; i < count; ++i)
        values.push_back(deserialize_uint16(data, offset));
    return values;
}

std::vector<std::uint64_t>
deserialize_uint64_vector(const std::vector<std::uint8_t>& data, size_t& offset,
                          std::uint8_t count)
{
    std::vector<std::uint64_t> values;
    values.reserve(count);
    for(std::uint8_t i = 0; i < count; ++i)
        values.push_back(deserialize_uint64(data, offset));
    return values;
}
}  // namespace

std::vector<std::uint8_t>
serialize_gpu_metrics(const gpu_metrics_t&              metrics,
                      const gpu_metrics_capabilities_t& capabilities,
                      const gpu_metrics_settings_t&     settings)
{
    // Flatten XCP data if needed and pre-calculate counts
    // Example:
    // XCP 0: [10, 20, 30]        (3 values)
    // XCP 1: [15, 25]            (2 values)
    // XCP 2: [5, 10, 15, 20]     (4 values)
    // vcn_xcp_count: 3
    // vcn_xcp_sizes: [3, 2, 4]
    // vcn_data_flat: [10, 20, 30, 15, 25, 5, 10, 15, 20]
    std::vector<std::uint16_t> vcn_data_flat;
    std::vector<std::uint16_t> jpeg_data_flat;
    std::vector<std::uint8_t>  vcn_xcp_sizes;   // Size of each XCP's VCN data
    std::vector<std::uint8_t>  jpeg_xcp_sizes;  // Size of each XCP's JPEG data

    if(capabilities.flags.vcn_is_device_level_only)
    {
        vcn_data_flat = metrics.vcn_activity;
    }
    else
    {
        // Flatten per-XCP VCN data and record sizes
        for(const auto& xcp_data : metrics.vcn_busy)
        {
            vcn_xcp_sizes.push_back(static_cast<std::uint8_t>(xcp_data.size()));
            vcn_data_flat.insert(vcn_data_flat.end(), xcp_data.begin(), xcp_data.end());
        }
    }

    if(capabilities.flags.jpeg_is_device_level_only)
    {
        jpeg_data_flat = metrics.jpeg_activity;
    }
    else
    {
        // Flatten per-XCP JPEG data and record sizes
        for(const auto& xcp_data : metrics.jpeg_busy)
        {
            jpeg_xcp_sizes.push_back(static_cast<std::uint8_t>(xcp_data.size()));
            jpeg_data_flat.insert(jpeg_data_flat.end(), xcp_data.begin(), xcp_data.end());
        }
    }

    std::uint8_t vcn_count      = static_cast<std::uint8_t>(vcn_data_flat.size());
    std::uint8_t jpeg_count     = static_cast<std::uint8_t>(jpeg_data_flat.size());
    std::uint8_t vcn_xcp_count  = static_cast<std::uint8_t>(vcn_xcp_sizes.size());
    std::uint8_t jpeg_xcp_count = static_cast<std::uint8_t>(jpeg_xcp_sizes.size());
    std::uint8_t xgmi_read_count =
        static_cast<std::uint8_t>(metrics.xgmi_read_data_acc.size());
    std::uint8_t xgmi_write_count =
        static_cast<std::uint8_t>(metrics.xgmi_write_data_acc.size());

    std::vector<std::uint8_t> result;

    // Serialize capability flags (1 byte)
    // These flags determine how the activity information is provided in the data
    // Current flags:
    //      - bit 0 (0x01): vcn_is_device_level_only (device-level vs per-XCP)
    //      - bit 1 (0x02): jpeg_is_device_level_only (device-level vs per-XCP)
    //      - bits 2-7: Reserved for future use
    //
    serialize_uint8(result, capabilities.value);

    // Serialize counts
    serialize_uint8(result, vcn_count);
    serialize_uint8(result, jpeg_count);
    serialize_uint8(result, vcn_xcp_count);
    serialize_uint8(result, jpeg_xcp_count);
    serialize_uint8(result, xgmi_read_count);
    serialize_uint8(result, xgmi_write_count);

    // Serialize per-XCP sizes
    for(std::uint8_t size : vcn_xcp_sizes)
        serialize_uint8(result, size);

    for(std::uint8_t size : jpeg_xcp_sizes)
        serialize_uint8(result, size);

    // Serialize the flattened data
    if(settings.vcn_activity && vcn_count > 0)
        serialize_uint16_vector(result, vcn_data_flat, vcn_count);
    if(settings.jpeg_activity && jpeg_count > 0)
        serialize_uint16_vector(result, jpeg_data_flat, jpeg_count);
    if(settings.xgmi)
    {
        serialize_uint16(result, metrics.xgmi_link_width);
        serialize_uint16(result, metrics.xgmi_link_speed);
        serialize_uint64_vector(result, metrics.xgmi_read_data_acc, xgmi_read_count);
        serialize_uint64_vector(result, metrics.xgmi_write_data_acc, xgmi_write_count);
    }
    if(settings.pcie)
    {
        serialize_uint16(result, metrics.pcie_link_width);
        serialize_uint16(result, metrics.pcie_link_speed);
        serialize_uint64(result, metrics.pcie_bandwidth_acc);
        serialize_uint64(result, metrics.pcie_bandwidth_inst);
    }

    return result;
}

void
deserialize_gpu_metrics(const std::vector<std::uint8_t>& serialized_data,
                        gpu_metrics_t& result, bool is_vcn_enabled, bool is_jpeg_enabled,
                        bool is_xgmi_enabled, bool is_pcie_enabled,
                        gpu_metrics_capabilities_t& capabilities)
{
    if(serialized_data.empty())
    {
        throw std::runtime_error("Invalid serialized data: insufficient header size");
    }
    size_t offset = 0;

    // Deserialize capability flags (1 byte)
    // Extract capability flags from packed byte.
    // See serialize_gpu_metrics() for flag definitions.
    capabilities.value = deserialize_uint8(serialized_data, offset);

    // Deserialize counts
    std::uint8_t vcn_count        = deserialize_uint8(serialized_data, offset);
    std::uint8_t jpeg_count       = deserialize_uint8(serialized_data, offset);
    std::uint8_t vcn_xcp_count    = deserialize_uint8(serialized_data, offset);
    std::uint8_t jpeg_xcp_count   = deserialize_uint8(serialized_data, offset);
    std::uint8_t xgmi_read_count  = deserialize_uint8(serialized_data, offset);
    std::uint8_t xgmi_write_count = deserialize_uint8(serialized_data, offset);

    // Deserialize per-XCP sizes
    std::vector<std::uint8_t> vcn_xcp_sizes;
    std::vector<std::uint8_t> jpeg_xcp_sizes;
    for(std::uint8_t i = 0; i < vcn_xcp_count; ++i)
        vcn_xcp_sizes.push_back(deserialize_uint8(serialized_data, offset));
    for(std::uint8_t i = 0; i < jpeg_xcp_count; ++i)
        jpeg_xcp_sizes.push_back(deserialize_uint8(serialized_data, offset));

    // Deserialize VCN data and reconstruct structure
    if(is_vcn_enabled && vcn_count > 0)
    {
        auto flat_data = deserialize_uint16_vector(serialized_data, offset, vcn_count);
        if(capabilities.flags.vcn_is_device_level_only)
        {
            result.vcn_activity = flat_data;
        }
        else
        {
            // Per-XCP: split flat data according to XCP sizes into vcn_busy
            size_t flat_offset = 0;
            for(std::uint8_t xcp_size : vcn_xcp_sizes)
            {
                std::vector<std::uint16_t> xcp_data(flat_data.begin() + flat_offset,
                                                    flat_data.begin() + flat_offset +
                                                        xcp_size);
                result.vcn_busy.push_back(xcp_data);
                flat_offset += xcp_size;
            }
        }
    }

    // Deserialize JPEG data and reconstruct structure
    if(is_jpeg_enabled && jpeg_count > 0)
    {
        auto flat_data = deserialize_uint16_vector(serialized_data, offset, jpeg_count);
        if(capabilities.flags.jpeg_is_device_level_only)
        {
            result.jpeg_activity = flat_data;
        }
        else
        {
            // Per-XCP: split flat data according to XCP sizes into jpeg_busy
            size_t flat_offset = 0;
            for(std::uint8_t xcp_size : jpeg_xcp_sizes)
            {
                std::vector<std::uint16_t> xcp_data(flat_data.begin() + flat_offset,
                                                    flat_data.begin() + flat_offset +
                                                        xcp_size);
                result.jpeg_busy.push_back(xcp_data);
                flat_offset += xcp_size;
            }
        }
    }

    // Deserialize XGMI data
    if(is_xgmi_enabled)
    {
        result.xgmi_link_width = deserialize_uint16(serialized_data, offset);
        result.xgmi_link_speed = deserialize_uint16(serialized_data, offset);
        result.xgmi_read_data_acc =
            deserialize_uint64_vector(serialized_data, offset, xgmi_read_count);
        result.xgmi_write_data_acc =
            deserialize_uint64_vector(serialized_data, offset, xgmi_write_count);
    }

    // Deserialize PCIe data
    if(is_pcie_enabled)
    {
        result.pcie_link_width     = deserialize_uint16(serialized_data, offset);
        result.pcie_link_speed     = deserialize_uint16(serialized_data, offset);
        result.pcie_bandwidth_acc  = deserialize_uint64(serialized_data, offset);
        result.pcie_bandwidth_inst = deserialize_uint64(serialized_data, offset);
    }
}

}  // namespace gpu
}  // namespace rocprofsys
