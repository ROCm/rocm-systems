// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

namespace rocprofsys::inline common
{
// File exists to decode AMD-SMI and rocprofiler-sdk PCIe BDF strings
// into a format that can be compared.

// Format a PCI BDF string from its components. Use directly with information from
// AMD SMI's amdsmi_get_gpu_device_bdf().
// For rocprofiler-SDK, use format_pci_bdf_from_location_id() instead
[[nodiscard]] inline std::string
format_pci_bdf(std::uint64_t domain, std::uint64_t bus, std::uint64_t device,
               std::uint64_t function)
{
    char buffer[32] = {};
    std::snprintf(buffer, sizeof(buffer), "%04x:%02x:%02x.%x",
                  static_cast<unsigned>(domain), static_cast<unsigned>(bus),
                  static_cast<unsigned>(device), static_cast<unsigned>(function));
    return std::string{ buffer };
}

// Decode a KFD/rocprofiler-sdk PCIe location_id (rocprofiler_agent_v0_t::location_id)
// plus PCI domain into the canonical BDF string. The location_id bit layout matches
// the KFD topology encoding: function=bits[0:2], device=bits[3:7], bus=bits[8:15].
[[nodiscard]] inline std::string
format_pci_bdf_from_location_id(std::uint32_t domain, std::uint32_t location_id)
{
    const std::uint32_t function = location_id & 0x7U;
    const std::uint32_t device   = (location_id >> 3U) & 0x1FU;
    const std::uint32_t bus      = (location_id >> 8U) & 0xFFU;
    return format_pci_bdf(domain, bus, device, function);
}
}  // namespace rocprofsys::inline common
