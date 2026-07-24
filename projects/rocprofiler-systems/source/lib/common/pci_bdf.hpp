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

// The KFD / rocminfo "BDFID": bus/device/function packed as
// (bus << 8) | (device << 3) | function, with the PCI domain excluded. This is the
// value `rocminfo` prints as "BDFID" (HSA_AMD_AGENT_INFO_BDFID) and equals the low 16
// bits of a KFD/rocprofiler-sdk location_id.
[[nodiscard]] inline std::uint16_t
pci_bdfid(std::uint64_t bus, std::uint64_t device, std::uint64_t function)
{
    return static_cast<std::uint16_t>(((bus & 0xFFULL) << 8U) |
                                      ((device & 0x1FULL) << 3U) | (function & 0x7ULL));
}

// Parse a canonical BDF string ("domain:bus:device.function", as produced by
// format_pci_bdf) and return its rocminfo-style BDFID. Returns 0 if the string is not in
// the expected format.
[[nodiscard]] inline std::uint16_t
pci_bdfid_from_string(const std::string& bdf)
{
    unsigned domain = 0U, bus = 0U, device = 0U, function = 0U;
    if(std::sscanf(bdf.c_str(), "%x:%x:%x.%x", &domain, &bus, &device, &function) != 4)
    {
        return 0U;
    }
    return pci_bdfid(bus, device, function);
}
}  // namespace rocprofsys::inline common
