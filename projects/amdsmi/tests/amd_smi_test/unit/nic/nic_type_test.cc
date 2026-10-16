// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * Contract of the NIC type and the public layout kept from the previous release.
 * amdsmi_nic_asic_info_t keeps its previous size, because a caller built against the old header
 * passes a buffer of that size and the library fills it; the type and capability live behind their
 * own getters instead.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/nic/amd_smi_ainic_device.h"
#include "nic_unit_fixture.h"

namespace {

// Pinned for AMDSMI_MAX_STRING_LENGTH == 256; changing any of these is an ABI decision.
constexpr auto kMaxStringLength = size_t{256};
constexpr auto kAsicInfoSizeBytes = size_t{1290};
constexpr auto kPortSizeBytes = size_t{2328};
constexpr auto kPortInfoSizeBytes = size_t{74504};
constexpr auto kAsicVendorNameOffsetBytes = size_t{1033};
constexpr auto kPortIfindexOffsetBytes = size_t{780};
constexpr auto kPortMacAddressOffsetBytes = size_t{781};
constexpr auto kPortLinkSpeedOffsetBytes = size_t{1296};

// FillByte(kAsic) repeated across the 32-bit capability member.
constexpr auto kAsicFillCapability = uint32_t{0xA2A2A2A2};

// Distinct from each other and from the fill byte, so a swapped source cannot match.
constexpr auto kDistinctType = uint8_t{2};
constexpr auto kDistinctCapability = uint32_t{3};
constexpr auto kDistinctBdf = uint64_t{0x300};

static_assert(AMDSMI_MAX_STRING_LENGTH == kMaxStringLength,
              "the pinned layout below assumes this string length");

}  // namespace

TEST_F(NicUnit, NicPublicLayoutAsicInfoMatchPreviousRelease) {
  EXPECT_EQ(sizeof(amdsmi_nic_asic_info_t), kAsicInfoSizeBytes);
}

TEST_F(NicUnit, NicPublicLayoutOffsetsMatchDevelop) {
  EXPECT_EQ(offsetof(amdsmi_nic_asic_info_t, vendor_name), kAsicVendorNameOffsetBytes);
  EXPECT_EQ(offsetof(amdsmi_nic_port_t, ifindex), kPortIfindexOffsetBytes);
  EXPECT_EQ(offsetof(amdsmi_nic_port_t, mac_address), kPortMacAddressOffsetBytes);
  EXPECT_EQ(offsetof(amdsmi_nic_port_t, link_speed), kPortLinkSpeedOffsetBytes);
}

TEST_F(NicUnit, NicPublicLayoutPortMatchPreviousRelease) {
  EXPECT_EQ(sizeof(amdsmi_nic_port_t), kPortSizeBytes);
  EXPECT_EQ(sizeof(amdsmi_nic_port_info_t), kPortInfoSizeBytes);
  EXPECT_EQ(sizeof(amdsmi_nic_port_t::ifindex), size_t{1});
}

// A binary built against the previous release calls the symbol with these four arguments; a fifth
// shifts them.
TEST_F(NicUnit, NicPublicLayoutVendorStatisticsSignatureMatchPreviousRelease) {
  using DevelopSignature_t =
      amdsmi_status_t (*)(amdsmi_processor_handle, uint32_t, uint32_t*, amdsmi_nic_stat_t*);

  EXPECT_TRUE((std::is_same_v<decltype(&amdsmi_get_nic_vendor_statistics), DevelopSignature_t>));
}

TEST_F(NicUnit, NicTypeEnumValuesAreStable) {
  EXPECT_EQ(static_cast<int>(AMDSMI_NIC_TYPE_UNKNOWN), 0);
  EXPECT_EQ(static_cast<int>(AMDSMI_NIC_TYPE_AINIC), 1);
  EXPECT_EQ(static_cast<int>(AMDSMI_NIC_TYPE_UALOE), 2);
  EXPECT_EQ(static_cast<int>(AMDSMI_NIC_TYPE_OTHER), 3);
}

// A binary built against the previous release passes these numeric values to
// amdsmi_get_processor_handles_by_type().
TEST_F(NicUnit, NicPublicLayoutBrcmProcessorTypesKeepTheirValues) {
  EXPECT_EQ(static_cast<int>(AMDSMI_PROCESSOR_TYPE_AMD_NIC), 7);
  EXPECT_EQ(static_cast<int>(AMDSMI_PROCESSOR_TYPE_BRCM_NIC), 8);
  EXPECT_EQ(static_cast<int>(AMDSMI_PROCESSOR_TYPE_BRCM_SWITCH), 9);
}

// The stub getter fills the shim asic struct with FillByte(kAsic); type and capability are its
// last two members.
TEST_F(NicUnit, PopulateKeepsAsicTypeAndCapabilityOnTheDevice) {
  ASSERT_EQ(Populate(), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(info_.type, FillByte(kAsic));
  EXPECT_EQ(info_.capability, kAsicFillCapability);
}

smi_nic_status_t DistinctAsicStub(smi_nic_ctx_t, uint64_t, smi_nic_asic_info_t* info) {
  std::memset(info, 0, sizeof(*info));
  info->type = kDistinctType;
  info->capability = kDistinctCapability;
  return SMI_NIC_STATUS_SUCCESS;
}

// The uniform fill above cannot tell type from capability or from a neighbouring member.
TEST_F(NicUnit, PopulateTakesTypeAndCapabilityFromTheirOwnShimMembers) {
  auto getters = kStubGetters;
  getters.asic = DistinctAsicStub;
  amd::smi::nic_set_info_getters_for_testing(&getters);

  ASSERT_EQ(Populate(), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(info_.type, kDistinctType);
  EXPECT_EQ(info_.capability, kDistinctCapability);
}

// The public getters read these without copying the whole info struct, which is over 600 KB.
TEST_F(NicUnit, DeviceReportsItsTypeCapabilityAndBdf) {
  info_.type = kDistinctType;
  info_.capability = kDistinctCapability;
  info_.bus.bdf.as_uint = kDistinctBdf;
  const amd::smi::AMDSmiAINICDevice device(info_);

  EXPECT_EQ(device.nic_type(), kDistinctType);
  EXPECT_EQ(device.capability(), kDistinctCapability);
  EXPECT_EQ(device.bdf(), kDistinctBdf);
}
