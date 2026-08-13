/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_DDA_FABRIC_FOOTPRINTS_HPP
#define RCCL_TEST_DDA_FABRIC_FOOTPRINTS_HPP

// DDA fabric scratch footprints, mirrored from the launcher headers so tests can
// compute expected sizes without pulling the device kernel headers everywhere.
// Every mirror below (all eight per-tier caps, all seven slot strides, the two
// sizeof values and the data-elems count) is pinned to its real constant by a
// static_assert in DdaFabricEpochTests.cpp, which builds in the Release-capable
// rccl-UnitTestsFixtures target -- so a production stride/cap change fails that
// build rather than silently passing. What is NOT pinnable that way: the literal
// 2 bank factor, the footprint formula shape (2*nRanks*stride*elemSize), and the
// ddaLL128ArSlotLines identity -- all eight ddaLL*ScratchSize functions are
// static-inline in .cu TUs, so those are pinned only behaviorally by the
// exact-boundary tests in DdaFabricScratchTests.cpp (Debug target). Kept
// lightweight (only <cstddef>) so most TUs need not pull the kernel headers.
// Mirror sources:
//   kLLTierMaxBytes    : kDdaLLArMaxBytes / kDdaLLAgMaxPerRankBytes /
//                        kDdaLLA2AMaxPerChunkBytes / kDdaLLRsMaxBytes  (all 128 KiB)
//   kLL128TierMaxBytes : kDdaLL128AgMaxPerRankBytes / kDdaLL128A2AMaxPerChunkBytes /
//                        kDdaLL128RsMaxBytes  (all 512 KiB)
//   kLL128DataElems    : meta::comms::kDdaLL128DataElems
//   kLLPacketBytes     : sizeof(meta::comms::LLPacket16)
//   kLL128LineBytes    : sizeof(meta::comms::LLLine128)

#include <cstddef>

namespace RcclUnitTesting
{

constexpr size_t kLLTierMaxBytes    = 131072;     // 128 KiB LL per-tier cap
constexpr size_t kLL128TierMaxBytes = 524288;     // 512 KiB LL128 AG/A2A/RS per-tier cap
constexpr size_t kLL128ArMaxBytes   = 1073741824; // 1 GiB LL128 AllReduce cap (message-dependent tier)
constexpr size_t kLL128DataElems    = 15;     // payload words per 128B line
constexpr size_t kLLPacketBytes     = 16;     // sizeof(LLPacket16)
constexpr size_t kLL128LineBytes    = 128;    // sizeof(LLLine128)

// Payload words -> 128B lines. Production uses `bytes >> 3` (truncating); every
// caller here passes a multiple of 8 (the predicates reject bytes % 8 != 0), so
// truncation and ceil agree. bytes/8 matches production literally.
constexpr size_t ddaLL128LinesForBytes(size_t bytes)
{
    // precondition: bytes % 8 == 0 (predicates reject otherwise), so bytes/8
    // matches production's `bytes >> 3`. constexpr so it can pin the slot stride.
    return (bytes / 8 + kLL128DataElems - 1) / kLL128DataElems;
}

// Fixed LL footprint (all four LL tiers): 2 banks * nRanks slots *
// (kLLTierMaxBytes/8) pkts * 16B = 512 KiB * nRanks, independent of message size.
inline size_t ddaLLFixedFootprint(int nRanks)
{
    return static_cast<size_t>(2) * static_cast<size_t>(nRanks) * (kLLTierMaxBytes / 8) * kLLPacketBytes;
}

// Fixed LL128 AG/A2A/RS footprint: 2 banks * nRanks slots * ceil(cap/8/15) lines * 128B.
inline size_t ddaLL128FixedFootprint(int nRanks)
{
    return static_cast<size_t>(2) * static_cast<size_t>(nRanks) * ddaLL128LinesForBytes(kLL128TierMaxBytes)
         * kLL128LineBytes;
}

// LL128 AllReduce footprint tracks the actual message size (compact slot stride).
inline size_t ddaLL128ArFootprintForBytes(int nRanks, size_t bytes)
{
    return static_cast<size_t>(2) * static_cast<size_t>(nRanks) * ddaLL128LinesForBytes(bytes) * kLL128LineBytes;
}

} // namespace RcclUnitTesting

#endif // RCCL_TEST_DDA_FABRIC_FOOTPRINTS_HPP
