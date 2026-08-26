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

// PCI BDF bit layout
// function=bits[0:2], device=bits[3:7], bus=bits[8:15], domain in the upper 16 bits.
inline constexpr std::uint32_t PCI_BDF_FUNCTION_MASK = 0x7U;
inline constexpr std::uint32_t PCI_BDF_DEVICE_MASK   = 0x1FU;
inline constexpr std::uint32_t PCI_BDF_BUS_MASK      = 0xFFU;
inline constexpr std::uint32_t PCI_BDF_DOMAIN_MASK   = 0xFFFFU;
inline constexpr std::uint32_t PCI_BDF_DEVICE_SHIFT  = 3U;
inline constexpr std::uint32_t PCI_BDF_BUS_SHIFT     = 8U;
inline constexpr std::uint32_t PCI_BDF_DOMAIN_SHIFT  = 16U;

// Format a PCI BDF string from its components. Use directly with information from
// AMD SMI's amdsmi_get_gpu_device_bdf(), whose amdsmi_bdf_t sizes the fields as
// function:3, device:5, bus:8 and domain:48
// For rocprofiler-SDK, use format_pci_bdf_from_location_id() instead
[[nodiscard]] inline std::string
format_pci_bdf(std::uint64_t domain, std::uint16_t bus, std::uint16_t device,
               std::uint16_t function)
{
    char buffer[32] = {};
    std::snprintf(buffer, sizeof(buffer), "%04llx:%02x:%02x.%x",
                  static_cast<unsigned long long>(domain), static_cast<unsigned>(bus),
                  static_cast<unsigned>(device), static_cast<unsigned>(function));
    return std::string{ buffer };
}

// Decode a KFD/rocprofiler-sdk PCIe location_id (rocprofiler_agent_v0_t::location_id)
// plus PCI domain into the canonical BDF string.
[[nodiscard]] inline std::string
format_pci_bdf_from_location_id(std::uint32_t domain, std::uint32_t location_id)
{
    const auto function = static_cast<std::uint16_t>(location_id & PCI_BDF_FUNCTION_MASK);
    const auto device = static_cast<std::uint16_t>((location_id >> PCI_BDF_DEVICE_SHIFT) &
                                                   PCI_BDF_DEVICE_MASK);
    const auto bus =
        static_cast<std::uint16_t>((location_id >> PCI_BDF_BUS_SHIFT) & PCI_BDF_BUS_MASK);
    return format_pci_bdf(domain, bus, device, function);
}

// The rocminfo "BDFID": (domain << 16) | (bus << 8) | (device << 3) | function, matching
// ROCR's std::uint32_t HSA_AMD_AGENT_INFO_BDFID (domain in the upper 16 bits). The domain
// keeps otherwise-identical devices distinct across PCI domains; single-domain values are
// the 16-bit bus/device/function.
[[nodiscard]] inline std::uint32_t
pci_bdfid(std::uint64_t domain, std::uint16_t bus, std::uint16_t device,
          std::uint16_t function) noexcept
{
    return static_cast<std::uint32_t>(
        ((domain & PCI_BDF_DOMAIN_MASK) << PCI_BDF_DOMAIN_SHIFT) |
        ((bus & PCI_BDF_BUS_MASK) << PCI_BDF_BUS_SHIFT) |
        ((device & PCI_BDF_DEVICE_MASK) << PCI_BDF_DEVICE_SHIFT) |
        (function & PCI_BDF_FUNCTION_MASK));
}

// Parse a canonical BDF string ("domain:bus:device.function", as produced by
// format_pci_bdf) and return its rocminfo-style BDFID, domain included.
// Returns 0 unless the whole string parses, so a trailing suffix is rejected rather than
// ignored.
// NOTE: 0 is also the BDFID of "0000:00:00.0", so a caller cannot tell a parse failure
// from that address; safe here because it is the root complex, never a GPU.
[[nodiscard]] inline std::uint32_t
pci_bdfid_from_string(const std::string& bdf)
{
    unsigned domain = 0U, bus = 0U, device = 0U, function = 0U;
    int      consumed = 0;
    if(std::sscanf(bdf.c_str(), "%x:%x:%x.%x%n", &domain, &bus, &device, &function,
                   &consumed) != 4 ||
       consumed != static_cast<int>(bdf.size()))
    {
        return 0U;
    }
    return pci_bdfid(domain, static_cast<std::uint16_t>(bus),
                     static_cast<std::uint16_t>(device),
                     static_cast<std::uint16_t>(function));
}
}  // namespace rocprofsys::inline common
