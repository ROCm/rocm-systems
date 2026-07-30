/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "amdsmi_wrap.h"

#include <cstddef>
#include <cstring>
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

// ---------------------------------------------------------------------------
// Both amdsmi_fabric_info_t shapes
//
// amd_smi 27.0 flattened the struct, so RCCL has to read a version and a v1
// payload out of either shape. Only one of them exists in any given build, so
// stand both up as mocks here: that is the only way to compile and exercise the
// branch the installed header does not select.
// ---------------------------------------------------------------------------

namespace
{
union MockFabricPayload
{
    amdsmi_fabric_info_v1_t v1;
};

// Pre-27: version and payload sit inside fabric_info, where the union carries
// the fabric_version name.
struct MockNestedFabricVer
{
    uint32_t          version;
    MockFabricPayload fabric_version;
};

struct MockNestedFabricInfo
{
    amdsmi_bdf_t        bdf;
    MockNestedFabricVer fabric_info;
    uint32_t            reserved[15];
};

// 27.0 and later: version hoisted to the top level, payload reachable directly.
struct MockFlatFabricInfo
{
    amdsmi_bdf_t      bdf;
    uint32_t          fabric_version;
    MockFabricPayload fabric_info;
    uint32_t          reserved[15];
};

// A top-level fabric_version that is not the version number, which is what the
// name means in the nested shape. Detection keys on the type, not the name.
struct MockHoistedUnionFabricInfo
{
    amdsmi_bdf_t      bdf;
    MockFabricPayload fabric_version;
};
} // namespace

TEST(AmdSmiFabricLayoutCompat, DetectsEachShape)
{
    EXPECT_FALSE(amdSmiFabricInfoIsFlat<MockNestedFabricInfo>::value);
    EXPECT_TRUE(amdSmiFabricInfoIsFlat<MockFlatFabricInfo>::value);
    EXPECT_FALSE(amdSmiFabricInfoIsFlat<MockHoistedUnionFabricInfo>::value);
}

TEST(AmdSmiFabricLayoutCompat, ReadsVersionFromNestedShape)
{
    MockNestedFabricInfo info{};
    info.fabric_info.version = kAmdSmiFabricVersionUnreported;

    EXPECT_EQ(amdSmiFabricInfoVersion(info), kAmdSmiFabricVersionUnreported);
    EXPECT_TRUE(amdSmiFabricVersionUsable(amdSmiFabricInfoVersion(info)));
}

TEST(AmdSmiFabricLayoutCompat, ReadsVersionFromFlatShape)
{
    MockFlatFabricInfo info{};
    info.fabric_version = kAmdSmiFabricVersionUnreported;

    EXPECT_EQ(amdSmiFabricInfoVersion(info), kAmdSmiFabricVersionUnreported);
    EXPECT_TRUE(amdSmiFabricVersionUsable(amdSmiFabricInfoVersion(info)));
}

TEST(AmdSmiFabricLayoutCompat, ReadsPayloadFromNestedShape)
{
    MockNestedFabricInfo info{};
    info.fabric_info.fabric_version.v1.fabric_type  = AMDSMI_FABRIC_TYPE_UALOE;
    info.fabric_info.fabric_version.v1.accel_state  = AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_READY;
    info.fabric_info.fabric_version.v1.accelerator_id = 3;

    const amdsmi_fabric_info_v1_t* v1 = amdSmiFabricInfoV1(info);
    ASSERT_NE(v1, nullptr);
    EXPECT_EQ(v1->accelerator_id, 3u);
    EXPECT_TRUE(amdSmiFabricStateUsable(v1->fabric_type, v1->accel_state));
}

TEST(AmdSmiFabricLayoutCompat, ReadsPayloadFromFlatShape)
{
    MockFlatFabricInfo info{};
    info.fabric_info.v1.fabric_type    = AMDSMI_FABRIC_TYPE_UALOE;
    info.fabric_info.v1.accel_state    = AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_READY;
    info.fabric_info.v1.accelerator_id = 3;

    const amdsmi_fabric_info_v1_t* v1 = amdSmiFabricInfoV1(info);
    ASSERT_NE(v1, nullptr);
    EXPECT_EQ(v1->accelerator_id, 3u);
    EXPECT_TRUE(amdSmiFabricStateUsable(v1->fabric_type, v1->accel_state));
}

// The rename did not move anything, which is why one set of ABI asserts covers
// both headers. If a future layout change breaks this, the accessors are no
// longer enough on their own.
TEST(AmdSmiFabricLayoutCompat, BothShapesShareOneAbi)
{
    EXPECT_EQ(sizeof(MockNestedFabricInfo), sizeof(MockFlatFabricInfo));
    EXPECT_EQ(sizeof(MockFlatFabricInfo), sizeof(amdsmi_fabric_info_t));

    EXPECT_EQ(offsetof(MockNestedFabricInfo, fabric_info) + offsetof(MockNestedFabricVer, version),
              offsetof(MockFlatFabricInfo, fabric_version));
    EXPECT_EQ(offsetof(MockNestedFabricInfo, fabric_info) + offsetof(MockNestedFabricVer, fabric_version),
              offsetof(MockFlatFabricInfo, fabric_info));
    EXPECT_EQ(offsetof(MockNestedFabricInfo, reserved), offsetof(MockFlatFabricInfo, reserved));
}

// ---------------------------------------------------------------------------
// accel_state runtime offset
//
// local_accelerators[] in amdsmi_fabric_info_v1_t grew from 8 to 16 entries in
// 27.0, moving accel_state 32 bytes later. We compile against the 16-entry
// layout but reach the library through dlopen, so a <=26.x runtime writes
// accel_state 32 bytes before where our struct places it. The accessor has to
// follow the loaded runtime's major version, not our compiled offset.
// ---------------------------------------------------------------------------

// Offset accel_state occupies in the older 8-entry layout, derived from our
// compiled 16-entry struct so the test tracks the real field automatically.
static size_t Pre27AccelStateOffset()
{
    return offsetof(amdsmi_fabric_info_v1_t, accel_state) -
           static_cast<size_t>(AMDSMI_FABRIC_MAX_LOCAL_GPUS - kAmdSmiFabricLocalGpusPre27) * sizeof(uint32_t);
}

TEST(AmdSmiFabricAccelState, ReadsCurrentLayoutForModernRuntime)
{
    amdsmi_fabric_info_v1_t v1{};
    v1.accel_state = AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_ACTIVE;
    EXPECT_EQ(amdSmiFabricAccelState(v1, kAmdSmiFabricLocal16Major), AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_ACTIVE);
}

TEST(AmdSmiFabricAccelState, UnknownRuntimeUsesCurrentLayout)
{
    amdsmi_fabric_info_v1_t v1{};
    v1.accel_state = AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_READY;
    EXPECT_EQ(amdSmiFabricAccelState(v1, kAmdSmiLibVersionUnknown), AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_READY);
}

TEST(AmdSmiFabricAccelState, ReadsShiftedOffsetForPre27Runtime)
{
    // Emulate what a 26.x library writes into our 16-entry buffer: accel_state
    // lands 32 bytes earlier, and the tail of local_accelerators[] where our
    // compiled field sits is left as some other value.
    amdsmi_fabric_info_v1_t v1{};
    const auto pre27State = AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_ACTIVE;
    std::memcpy(reinterpret_cast<uint8_t*>(&v1) + Pre27AccelStateOffset(), &pre27State, sizeof(pre27State));
    v1.accel_state = AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_ERROR; // decoy at the 16-entry offset

    EXPECT_EQ(amdSmiFabricAccelState(v1, 26), AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_ACTIVE);
    // And the modern read would have picked up the decoy, proving the offset matters.
    EXPECT_EQ(amdSmiFabricAccelState(v1, kAmdSmiFabricLocal16Major),
              AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_ERROR);
}

// ---------------------------------------------------------------------------
// Telemetry name call convention
//
// amdsmi_fabric_telem_id_to_string swapped its return value for an out-parameter
// in 27.0, and we reach it through dlopen, so the version the loaded runtime
// reports decides how it may be called.
// ---------------------------------------------------------------------------

TEST(AmdSmiFabricTelemAbi, PreflattenRuntimeUsesReturnValue)
{
    EXPECT_FALSE(amdSmiTelemIdUsesOutParam(26));
}

TEST(AmdSmiFabricTelemAbi, FlattenedRuntimeUsesOutParam)
{
    EXPECT_TRUE(amdSmiTelemIdUsesOutParam(kAmdSmiTelemOutParamMajor));
    EXPECT_TRUE(amdSmiTelemIdUsesOutParam(kAmdSmiTelemOutParamMajor + 1));
}

// Guessing wrong is not symmetric: the out-parameter form called on an older
// runtime just leaves the name unwritten, while the older form called on a newer
// runtime hands it an uninitialized pointer to write through.
TEST(AmdSmiFabricTelemAbi, UnknownVersionTakesTheSaferConvention)
{
    EXPECT_TRUE(amdSmiTelemIdUsesOutParam(kAmdSmiLibVersionUnknown));
}

} // namespace RcclUnitTesting
