// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/pci_bdf.hpp"

#include <gtest/gtest.h>

#include <cstdint>

using namespace rocprofsys::common;

// The canonical form must match what `amd-smi list` reports and what
// amdsmi_get_gpu_device_bdf yields, since it is the correlation key between an AMD SMI
// device and a rocprofiler-sdk agent.
TEST(pci_bdf_test, canonical_form)
{
    EXPECT_EQ(format_pci_bdf(0x0000, 0x03, 0x00, 0x0), "0000:03:00.0");
    EXPECT_EQ(format_pci_bdf(0x0000, 0xC1, 0x00, 0x0), "0000:c1:00.0");
    EXPECT_EQ(format_pci_bdf(0x0001, 0x0A, 0x1F, 0x7), "0001:0a:1f.7");
}

TEST(pci_bdf_test, zero_padding)
{
    // Domain is 4 hex digits, bus/device 2 hex digits, function 1 hex digit.
    EXPECT_EQ(format_pci_bdf(0, 0, 0, 0), "0000:00:00.0");
    EXPECT_EQ(format_pci_bdf(0xABCD, 0x0F, 0x02, 0x1), "abcd:0f:02.1");
}

// location_id encoding used by KFD / rocprofiler_agent_v0_t::location_id:
//   function = bits[0:2], device = bits[3:7], bus = bits[8:15]
TEST(pci_bdf_test, decode_location_id)
{
    // bus=0x03, device=0x00, function=0x0 => location_id 0x0300
    EXPECT_EQ(format_pci_bdf_from_location_id(0x0000, 0x0300), "0000:03:00.0");
    // bus=0xC1, device=0x00, function=0x0 => location_id 0xC100
    EXPECT_EQ(format_pci_bdf_from_location_id(0x0000, 0xC100), "0000:c1:00.0");
    // bus=0x0A, device=0x1F, function=0x7 => (0x0A<<8)|(0x1F<<3)|0x7 = 0x0AFF
    EXPECT_EQ(format_pci_bdf_from_location_id(0x0001, 0x0AFF), "0001:0a:1f.7");
}

// The two producers of the correlation key (AMD SMI decode vs agent location_id decode)
// must agree for the same physical device.
TEST(pci_bdf_test, location_id_matches_component_form)
{
    const std::uint32_t domain   = 0x0000;
    const std::uint32_t bus      = 0x03;
    const std::uint32_t device   = 0x00;
    const std::uint32_t function = 0x0;

    const std::uint32_t location_id = (bus << 8U) | (device << 3U) | function;

    EXPECT_EQ(format_pci_bdf(domain, bus, device, function),
              format_pci_bdf_from_location_id(domain, location_id));
}
