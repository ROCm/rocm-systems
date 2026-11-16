// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gpu_metrics.hpp"
#include <cstdint>

#include <stdexcept>

namespace rocprofsys::gpu
{
namespace
{
// Helper functions for deserialization
std::uint8_t
deserialize_uint8(const std::vector<std::uint8_t>& data, size_t& offset)
{
    if(offset >= data.size())
    {
        throw std::runtime_error("Invalid serialized data: unexpected end");
    }
    return data[offset++];
}

std::uint16_t
deserialize_uint16(const std::vector<std::uint8_t>& data, size_t& offset)
{
    if(offset + 1 >= data.size())
    {
        throw std::runtime_error("Invalid serialized data: unexpected end");
    }
    const std::uint16_t value = static_cast<std::uint16_t>(data[offset]) |
                                (static_cast<std::uint16_t>(data[offset + 1]) << 8);
    offset += 2;
    return value;
}

std::uint64_t
deserialize_uint64(const std::vector<std::uint8_t>& data, size_t& offset)
{
    if(offset + 7 >= data.size())
    {
        throw std::runtime_error("Invalid serialized data: unexpected end");
    }
    std::uint64_t value = 0;
    for(int i = 0; i < 8; ++i)
    {
        value |= (static_cast<std::uint64_t>(data[offset + i]) << (i * 8));
    }
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
    {
        values.push_back(deserialize_uint16(data, offset));
    }
    return values;
}

std::vector<std::uint64_t>
deserialize_uint64_vector(const std::vector<std::uint8_t>& data, size_t& offset,
                          std::uint8_t count)
{
    std::vector<std::uint64_t> values;
    values.reserve(count);
    for(std::uint8_t i = 0; i < count; ++i)
    {
        values.push_back(deserialize_uint64(data, offset));
    }
    return values;
}
}  // namespace

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
    capabilities.value = deserialize_uint8(serialized_data, offset);

    // Deserialize counts
    const std::uint8_t vcn_count        = deserialize_uint8(serialized_data, offset);
    const std::uint8_t jpeg_count       = deserialize_uint8(serialized_data, offset);
    const std::uint8_t vcn_xcp_count    = deserialize_uint8(serialized_data, offset);
    const std::uint8_t jpeg_xcp_count   = deserialize_uint8(serialized_data, offset);
    const std::uint8_t xgmi_read_count  = deserialize_uint8(serialized_data, offset);
    const std::uint8_t xgmi_write_count = deserialize_uint8(serialized_data, offset);

    // Deserialize per-XCP sizes
    std::vector<std::uint8_t> vcn_xcp_sizes;
    std::vector<std::uint8_t> jpeg_xcp_sizes;
    vcn_xcp_sizes.reserve(vcn_xcp_count);
    for(std::uint8_t i = 0; i < vcn_xcp_count; ++i)
    {
        vcn_xcp_sizes.push_back(deserialize_uint8(serialized_data, offset));
    }
    jpeg_xcp_sizes.reserve(jpeg_xcp_count);
    for(std::uint8_t i = 0; i < jpeg_xcp_count; ++i)
    {
        jpeg_xcp_sizes.push_back(deserialize_uint8(serialized_data, offset));
    }

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
            for(const std::uint8_t xcp_size : vcn_xcp_sizes)
            {
                const std::vector<std::uint16_t> xcp_data(flat_data.begin() + flat_offset,
                                                          flat_data.begin() +
                                                              flat_offset + xcp_size);
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
            for(const std::uint8_t xcp_size : jpeg_xcp_sizes)
            {
                const std::vector<std::uint16_t> xcp_data(flat_data.begin() + flat_offset,
                                                          flat_data.begin() +
                                                              flat_offset + xcp_size);
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

}  // namespace rocprofsys::gpu
