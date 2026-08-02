/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

// DDA fabric scratch footprints, mirrored from the launcher headers so tests can
// compute expected sizes without pulling the device kernel headers everywhere.
// The mirrors are pinned to the real constants by static_assert in
// DdaFabricEpochTests.cpp (which builds in the Release fixtures target), so a
// production stride/cap change fails the build rather than silently passing.
// Kept lightweight (only <cstddef>) so the Release epoch TU need not include the
// kernel headers except at the one static_assert site. Mirror sources:
//   kLLTierMaxBytes    : kDdaLLArMaxBytes / kDdaLLAgMaxPerRankBytes /
//                        kDdaLLA2AMaxPerChunkBytes / kDdaLLRsMaxBytes  (all 128 KiB)
//   kLL128TierMaxBytes : kDdaLL128AgMaxPerRankBytes / kDdaLL128A2AMaxPerChunkBytes /
//                        kDdaLL128RsMaxBytes  (all 512 KiB)
//   kLL128DataElems    : meta::comms::kDdaLL128DataElems
//   kLLPacketBytes     : sizeof(meta::comms::LLPacket16)
//   kLL128LineBytes    : sizeof(meta::comms::LLLine128)

#include <cstddef>

namespace RcclUnitTesting {

constexpr size_t kLLTierMaxBytes    = 131072; // 128 KiB LL per-tier cap
constexpr size_t kLL128TierMaxBytes = 524288; // 512 KiB LL128 AG/A2A/RS per-tier cap
constexpr size_t kLL128DataElems    = 15;     // payload words per 128B line
constexpr size_t kLLPacketBytes     = 16;     // sizeof(LLPacket16)
constexpr size_t kLL128LineBytes    = 128;    // sizeof(LLLine128)

// Payload words -> 128B lines. Production uses `bytes >> 3` (truncating); every
// caller here passes a multiple of 8 (the predicates reject bytes % 8 != 0), so
// truncation and ceil agree. bytes/8 matches production literally.
inline size_t ddaLL128LinesForBytes(size_t bytes) {
    const size_t words = bytes / 8; // precondition: bytes % 8 == 0
    return (words + kLL128DataElems - 1) / kLL128DataElems;
}

// Fixed LL footprint (all four LL tiers): 2 banks * nRanks slots *
// (kLLTierMaxBytes/8) pkts * 16B = 512 KiB * nRanks, independent of message size.
inline size_t ddaLLFixedFootprint(int nRanks) {
    return static_cast<size_t>(2) * static_cast<size_t>(nRanks) * (kLLTierMaxBytes / 8) * kLLPacketBytes;
}

// Fixed LL128 AG/A2A/RS footprint: 2 banks * nRanks slots * ceil(cap/8/15) lines * 128B.
inline size_t ddaLL128FixedFootprint(int nRanks) {
    return static_cast<size_t>(2) * static_cast<size_t>(nRanks) * ddaLL128LinesForBytes(kLL128TierMaxBytes)
         * kLL128LineBytes;
}

// LL128 AllReduce footprint tracks the actual message size (compact slot stride).
inline size_t ddaLL128ArFootprintForBytes(int nRanks, size_t bytes) {
    return static_cast<size_t>(2) * static_cast<size_t>(nRanks) * ddaLL128LinesForBytes(bytes) * kLL128LineBytes;
}

} // namespace RcclUnitTesting
