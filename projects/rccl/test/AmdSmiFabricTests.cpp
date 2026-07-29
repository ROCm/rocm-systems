/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "amdsmi_wrap.h"

#include <cstddef>
#include <gtest/gtest.h>

namespace RcclUnitTesting
{

// ---------------------------------------------------------------------------
// Fabric info version acceptance
//
// amdsmi_get_gpu_fabric_info() returns SUCCESS with a fully populated payload
// but never writes the version field, leaving it at the sentinel amd_smi uses
// for anything it could not source from sysfs. Rejecting that sentinel made
// RCCL_USE_AMD_SMI_LIB=1 discard working UALoE fabric on every device.
// ---------------------------------------------------------------------------

TEST(AmdSmiFabricVersion, AcceptsUnreportedSentinel)
{
    EXPECT_TRUE(amdSmiFabricVersionUsable(kAmdSmiFabricVersionUnreported));
}

TEST(AmdSmiFabricVersion, AcceptsCurrentVersion)
{
    EXPECT_TRUE(amdSmiFabricVersionUsable(AMDSMI_FABRIC_INFO_CURRENT_VERSION));
}

// A version the library actually claims, but whose union layout we do not
// implement, must still be refused rather than misparsed as v1.
TEST(AmdSmiFabricVersion, RejectsUnimplementedFutureVersion)
{
    EXPECT_FALSE(amdSmiFabricVersionUsable(AMDSMI_FABRIC_INFO_CURRENT_VERSION + 1));
}

// Distinct from the sentinel: zero means the library returned without writing
// the field at all, which contradicts a SUCCESS status and is worth surfacing.
TEST(AmdSmiFabricVersion, RejectsZero)
{
    EXPECT_FALSE(amdSmiFabricVersionUsable(0));
}

// ---------------------------------------------------------------------------
// Fabric type / accelerator state eligibility
//
// This is the decision the version check was gating: only a link type we drive
// combined with a state whose vPoD can carry traffic counts as usable.
// ---------------------------------------------------------------------------

TEST(AmdSmiFabricState, UaloeReadyIsUsable)
{
    EXPECT_TRUE(amdSmiFabricStateUsable(AMDSMI_FABRIC_TYPE_UALOE,
                                        AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_READY));
}

TEST(AmdSmiFabricState, UaloeActiveIsUsable)
{
    EXPECT_TRUE(amdSmiFabricStateUsable(AMDSMI_FABRIC_TYPE_UALOE,
                                        AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_ACTIVE));
}

TEST(AmdSmiFabricState, UalinkReadyIsUsable)
{
    EXPECT_TRUE(amdSmiFabricStateUsable(AMDSMI_FABRIC_TYPE_UALLINK,
                                        AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_READY));
}

TEST(AmdSmiFabricState, ConfiguredButNotReadyIsNotUsable)
{
    EXPECT_FALSE(amdSmiFabricStateUsable(AMDSMI_FABRIC_TYPE_UALOE,
                                         AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_CONFIGURED));
}

TEST(AmdSmiFabricState, UnconfiguredIsNotUsable)
{
    EXPECT_FALSE(amdSmiFabricStateUsable(AMDSMI_FABRIC_TYPE_UALOE,
                                         AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_UNCONFIGURED));
}

TEST(AmdSmiFabricState, ErrorStateIsNotUsable)
{
    EXPECT_FALSE(amdSmiFabricStateUsable(AMDSMI_FABRIC_TYPE_UALOE,
                                         AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_ERROR));
}

TEST(AmdSmiFabricState, UnknownLinkTypeIsNotUsable)
{
    EXPECT_FALSE(amdSmiFabricStateUsable(AMDSMI_FABRIC_TYPE_UNKNOWN,
                                         AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_ACTIVE));
}

// The combination this node actually reports, and that the version check used
// to throw away: UALoE + READY reached through the full struct.
TEST(AmdSmiFabricState, PayloadFromSysfsBackedDeviceIsUsable)
{
    amdsmi_fabric_info_v1_t v1{};
    v1.fabric_type = AMDSMI_FABRIC_TYPE_UALOE;
    v1.accel_state = AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_READY;
    v1.accelerator_id = 3;
    v1.ppod_size = 8;
    v1.vpod_id = 1;
    v1.vpod_size = 4;

    EXPECT_TRUE(amdSmiFabricVersionUsable(kAmdSmiFabricVersionUnreported));
    EXPECT_TRUE(amdSmiFabricStateUsable(v1.fabric_type, v1.accel_state));
}

// ---------------------------------------------------------------------------
// Struct layout
//
// The library writes these structs through dlopen using its own layout, so a
// declaration that disagrees overflows the caller's object and shifts the
// fields we read. amdsmi_wrap.h pins this at compile time; assert the field
// offsets too, since those are what a short local_accelerators[] would move.
// ---------------------------------------------------------------------------

TEST(AmdSmiFabricLayout, MatchesShippedLibraryAbi)
{
    EXPECT_EQ(sizeof(amdsmi_fabric_info_v1_t), 244u);
    EXPECT_EQ(sizeof(amdsmi_fabric_info_t), 320u);
    EXPECT_EQ(offsetof(amdsmi_fabric_info_t, reserved), 256u);
}

TEST(AmdSmiFabricLayout, TrailingFieldsAreNotShifted)
{
    EXPECT_EQ(offsetof(amdsmi_fabric_info_v1_t, addr_mode), 236u);
    EXPECT_EQ(offsetof(amdsmi_fabric_info_v1_t, accel_state), 240u);
}

} // namespace RcclUnitTesting
