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
// fields we read. amdsmi_wrap.h accepts the coherent 8-GPU layout used by some
// pre-release ROCm 7.14 snapshots, the 16-GPU layout used by final ROCm 7.14 and
// ROCm 7.15, and the extended union amd_smi 27.x added, but rejects any other
// combination.
// ---------------------------------------------------------------------------

TEST(AmdSmiFabricLayout, MatchesShippedLibraryAbi)
{
    const int recognized = (amdSmiFabricLayoutIs8Gpu ? 1 : 0) + (amdSmiFabricLayoutIs16Gpu ? 1 : 0) +
                           (amdSmiFabricLayoutIsExtendedUnion ? 1 : 0);
    EXPECT_EQ(recognized, 1) << "exactly one layout must match the header in use";
}

// The extended union grew the outer struct without moving the v1 window, so the
// 16-GPU offsets are expected for it too.
TEST(AmdSmiFabricLayout, TrailingFieldsAreNotShifted)
{
    const size_t expectedAddrMode = amdSmiFabricLayoutIs8Gpu ? 204u : 236u;
    EXPECT_EQ(offsetof(amdsmi_fabric_info_v1_t, addr_mode), expectedAddrMode);
    EXPECT_EQ(offsetof(amdsmi_fabric_info_v1_t, accel_state), expectedAddrMode + sizeof(uint32_t));
}

// Every detector test derives its write extent from the constant it is pinning, so a mutated
// boundary moves the input with it and the split stays self-consistent. Assert the values here so
// narrowing a window, or moving a boundary, is caught by something.
TEST(AmdSmiFabricRuntimeLayout, WindowBoundariesHaveTheirShippedValues)
{
    EXPECT_EQ(kAmdSmiFabricV1PayloadBegin, 12u);
    EXPECT_EQ(kAmdSmiFabricV1PayloadEnd, 256u);
    EXPECT_EQ(kAmdSmiFabricInfo8GpuSize, 288u);
    EXPECT_EQ(kAmdSmiFabricInfo16GpuSize, 320u);
    EXPECT_GE(kAmdSmiFabricInfoBufferSize, kAmdSmiFabricInfo16GpuSize);
}

TEST(AmdSmiFabricRuntimeLayout, DetectsEightGpuWriter)
{
    amdSmiFabricInfoBuffer buffer;
    amdSmiPrepareFabricInfoBuffer(buffer);
    memset(buffer.bytes, 0, kAmdSmiFabricInfo8GpuSize);

    EXPECT_EQ(amdSmiDetectFabricRuntimeLayout(buffer), amdSmiFabricRuntimeLayout::EightGpu);
}

TEST(AmdSmiFabricRuntimeLayout, DetectsSixteenGpuWriter)
{
    amdSmiFabricInfoBuffer buffer;
    amdSmiPrepareFabricInfoBuffer(buffer);
    memset(buffer.bytes, 0, kAmdSmiFabricInfo16GpuSize);

    EXPECT_EQ(amdSmiDetectFabricRuntimeLayout(buffer), amdSmiFabricRuntimeLayout::SixteenGpu);
}

// amd_smi 27.x assigns field-wise and stops at the end of the v1 payload, so the
// canary survives everywhere above it. This is the extent the real runtime writes.
TEST(AmdSmiFabricRuntimeLayout, DetectsExtendedUnionWriter)
{
    amdSmiFabricInfoBuffer buffer;
    amdSmiPrepareFabricInfoBuffer(buffer);
    memset(buffer.bytes, 0, kAmdSmiFabricV1PayloadEnd);

    EXPECT_EQ(amdSmiDetectFabricRuntimeLayout(buffer), amdSmiFabricRuntimeLayout::ExtendedUnion);
}

// An untouched v1 payload means the call wrote nothing, which says nothing about
// the layout and must not be read as one.
TEST(AmdSmiFabricRuntimeLayout, RejectsUnknownWriter)
{
    amdSmiFabricInfoBuffer buffer;
    amdSmiPrepareFabricInfoBuffer(buffer);
    buffer.bytes[kAmdSmiFabricV1PayloadEnd - 1] = 0;

    EXPECT_EQ(amdSmiDetectFabricRuntimeLayout(buffer), amdSmiFabricRuntimeLayout::Unknown);
}

// Pins the upper edge of the ExtendedUnion window at 288. The dirty byte sits at
// 288, outside [256,288), so the classification must not change; widening that
// window to 320 would swallow the byte and reclassify.
TEST(AmdSmiFabricRuntimeLayout, DirtyByteAtTheEightGpuBoundaryStaysExtendedUnion)
{
    amdSmiFabricInfoBuffer buffer;
    amdSmiPrepareFabricInfoBuffer(buffer);
    memset(buffer.bytes, 0, kAmdSmiFabricV1PayloadEnd);
    buffer.bytes[kAmdSmiFabricInfo8GpuSize] = 0;

    EXPECT_EQ(amdSmiDetectFabricRuntimeLayout(buffer), amdSmiFabricRuntimeLayout::ExtendedUnion);
}

// A writer that stops partway through the 16-GPU tail matches no shipped layout,
// so it must not be concluded to be 16-GPU by elimination.
TEST(AmdSmiFabricRuntimeLayout, PartialSixteenGpuWriteIsUnknown)
{
    amdSmiFabricInfoBuffer buffer;
    amdSmiPrepareFabricInfoBuffer(buffer);
    memset(buffer.bytes, 0, kAmdSmiFabricInfo16GpuSize - 1);

    EXPECT_EQ(amdSmiDetectFabricRuntimeLayout(buffer), amdSmiFabricRuntimeLayout::Unknown);
}

// A real fabric payload is not all-zero. Every other case here fills with zeros,
// so without this one a detector that tested for zero rather than for the absence
// of the canary would pass the whole suite.
TEST(AmdSmiFabricRuntimeLayout, ClassifiesNonZeroWrites)
{
    amdSmiFabricInfoBuffer buffer;

    amdSmiPrepareFabricInfoBuffer(buffer);
    memset(buffer.bytes, 0xFF, kAmdSmiFabricV1PayloadEnd);
    EXPECT_EQ(amdSmiDetectFabricRuntimeLayout(buffer), amdSmiFabricRuntimeLayout::ExtendedUnion);

    amdSmiPrepareFabricInfoBuffer(buffer);
    memset(buffer.bytes, 0xFF, kAmdSmiFabricInfo8GpuSize);
    EXPECT_EQ(amdSmiDetectFabricRuntimeLayout(buffer), amdSmiFabricRuntimeLayout::EightGpu);

    amdSmiPrepareFabricInfoBuffer(buffer);
    memset(buffer.bytes, 0xFF, kAmdSmiFabricInfo16GpuSize);
    EXPECT_EQ(amdSmiDetectFabricRuntimeLayout(buffer), amdSmiFabricRuntimeLayout::SixteenGpu);
}

// A sparse write touches the v1 payload without filling it, so no layout is
// confirmed. Reached only by concluding a layout from the windows above it.
TEST(AmdSmiFabricRuntimeLayout, SparseWriteIsUnknown)
{
    amdSmiFabricInfoBuffer buffer;
    amdSmiPrepareFabricInfoBuffer(buffer);
    buffer.bytes[kAmdSmiFabricV1PayloadBegin] = 0;
    buffer.bytes[kAmdSmiFabricV1PayloadEnd] = 0;
    memset(buffer.bytes + kAmdSmiFabricInfo8GpuSize, 0,
           kAmdSmiFabricInfo16GpuSize - kAmdSmiFabricInfo8GpuSize);

    EXPECT_EQ(amdSmiDetectFabricRuntimeLayout(buffer), amdSmiFabricRuntimeLayout::Unknown);
}

// The probe must send the same request the per-device call sends: a zeroed bdf and
// fabric_version, canary from the v1 payload on. Spelled with literals so a change
// to kAmdSmiFabricV1PayloadBegin cannot move the boundary and the test with it.
TEST(AmdSmiFabricRuntimeLayout, PreparedBufferZeroesTheRequestHeader)
{
    amdSmiFabricInfoBuffer buffer;
    amdSmiPrepareFabricInfoBuffer(buffer);

    for (size_t i = 0; i < 12; ++i) {
        EXPECT_EQ(buffer.bytes[i], 0) << "request header byte " << i << " must be zeroed";
    }
    EXPECT_EQ(buffer.bytes[12], kAmdSmiFabricBufferCanary) << "the v1 payload must start as canary";
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
// longer enough on their own. The mocks carry a v1-only payload union, so they
// stay at the 16-GPU size while an extended-union header is larger; the buffer is
// what has to cover the real struct, and asserting that holds in every config.
TEST(AmdSmiFabricLayoutCompat, BothShapesShareOneAbi)
{
    EXPECT_EQ(sizeof(MockNestedFabricInfo), sizeof(MockFlatFabricInfo));
    EXPECT_GE(sizeof(amdSmiFabricInfoBuffer), sizeof(amdsmi_fabric_info_t))
        << "the probe buffer must be able to name the struct it is cast to";
    if (!kAmdSmiFabricHeaderIsExtended) {
        // Only the non-extended headers are the shape the mocks model, but there the mocks
        // must track the real struct: nothing else catches them drifting apart.
        EXPECT_EQ(sizeof(MockFlatFabricInfo), sizeof(amdsmi_fabric_info_t));
    }

    EXPECT_EQ(offsetof(MockNestedFabricInfo, fabric_info) + offsetof(MockNestedFabricVer, version),
              offsetof(MockFlatFabricInfo, fabric_version));
    EXPECT_EQ(offsetof(MockNestedFabricInfo, fabric_info) + offsetof(MockNestedFabricVer, fabric_version),
              offsetof(MockFlatFabricInfo, fabric_info));
    EXPECT_EQ(offsetof(MockNestedFabricInfo, reserved), offsetof(MockFlatFabricInfo, reserved));
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
