// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * Contract of the NIC type: the public enum values, and the one-byte ``type`` field that sits in
 * the alignment padding of amdsmi_nic_asic_info_t. The struct must keep its size and every
 * existing offset, because a caller built against the old header passes a buffer of the old size
 * and the library fills it.
 */

#include <gtest/gtest.h>

#include <cstddef>

#include "amd_smi/amdsmi.h"
#include "nic_unit_fixture.h"

namespace {

// Pinned for AMDSMI_MAX_STRING_LENGTH == 256; changing any of these is an ABI decision.
constexpr auto kMaxStringLength = size_t{256};
constexpr auto kAsicInfoSizeBytes = size_t{1296};
constexpr auto kCapabilityOffsetBytes = size_t{1292};
constexpr auto kTypeOffsetBytes = size_t{1289};

static_assert(AMDSMI_MAX_STRING_LENGTH == kMaxStringLength,
              "the pinned layout below assumes this string length");

}  // namespace

TEST_F(NicUnit, NicAsicInfoKeepsItsSize) {
  EXPECT_EQ(sizeof(amdsmi_nic_asic_info_t), kAsicInfoSizeBytes);
}

TEST_F(NicUnit, NicAsicInfoCapabilityKeepsItsOffset) {
  EXPECT_EQ(offsetof(amdsmi_nic_asic_info_t, capability), kCapabilityOffsetBytes);
}

TEST_F(NicUnit, NicAsicInfoTypeSitsInTheExistingPadding) {
  EXPECT_EQ(offsetof(amdsmi_nic_asic_info_t, type), kTypeOffsetBytes);
  EXPECT_EQ(offsetof(amdsmi_nic_asic_info_t, type), (offsetof(amdsmi_nic_asic_info_t, vendor_name) +
                                                     sizeof(amdsmi_nic_asic_info_t::vendor_name)));
}

TEST_F(NicUnit, NicAsicInfoTypeIsOneByte) {
  // An enum-typed member is 4 bytes and would not fit in the padding.
  EXPECT_EQ(sizeof(amdsmi_nic_asic_info_t::type), size_t{1});
}

TEST_F(NicUnit, NicTypeEnumValuesAreStable) {
  EXPECT_EQ(static_cast<int>(AMDSMI_NIC_TYPE_UNKNOWN), 0);
  EXPECT_EQ(static_cast<int>(AMDSMI_NIC_TYPE_AINIC), 1);
  EXPECT_EQ(static_cast<int>(AMDSMI_NIC_TYPE_UALOE), 2);
  EXPECT_EQ(static_cast<int>(AMDSMI_NIC_TYPE_OTHER), 3);
}

TEST_F(NicUnit, ShimAndPublicAsicInfoAgreeOnTypeAndCapability) {
  // populate_amd_ainic_device() overlays the shim struct on the public one.
  EXPECT_EQ(sizeof(smi_nic_asic_info_t), sizeof(amdsmi_nic_asic_info_t));
  EXPECT_EQ(offsetof(smi_nic_asic_info_t, type), offsetof(amdsmi_nic_asic_info_t, type));
  EXPECT_EQ(offsetof(smi_nic_asic_info_t, capability),
            offsetof(amdsmi_nic_asic_info_t, capability));
}
