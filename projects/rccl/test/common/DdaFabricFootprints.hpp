/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

// DDA fabric scratch footprints, mirrored once from the launcher headers so
// tests track the source (exact-boundary tests go red if a mirror drifts, since
// the real predicate uses the real constant). Kept dependency-free (no comm.h /
// gtest) so it can be included by both the Debug-only scratch tests and the
// Release+Debug epoch tests. Sources:
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

constexpr size_t kLLTierMaxBytes    = 131072; // 128 KiB LL per-tier cap
constexpr size_t kLL128TierMaxBytes = 524288; // 512 KiB LL128 AG/A2A/RS per-tier cap
constexpr size_t kLL128DataElems    = 15;     // payload words per 128B line
constexpr size_t kLLPacketBytes     = 16;     // sizeof(LLPacket16)
constexpr size_t kLL128LineBytes    = 128;    // sizeof(LLLine128)

inline size_t ddaLL128LinesForBytes(size_t bytes)
{
    const size_t words = (bytes + 7) / 8;
    return (words + kLL128DataElems - 1) / kLL128DataElems;
}

// Fixed LL footprint (all four LL tiers): 2 banks * nRanks slots *
// (kLLTierMaxBytes/8) pkts * 16B = 512 KiB * nRanks, independent of message size.
inline size_t ddaLLFixedFootprint(int nRanks)
{
    return (size_t)2 * (size_t)nRanks * (kLLTierMaxBytes / 8) * kLLPacketBytes;
}

// Fixed LL128 AG/A2A/RS footprint: 2 banks * nRanks slots * ceil(cap/8/15) lines * 128B.
inline size_t ddaLL128FixedFootprint(int nRanks)
{
    return (size_t)2 * (size_t)nRanks * ddaLL128LinesForBytes(kLL128TierMaxBytes) * kLL128LineBytes;
}

// LL128 AllReduce footprint tracks the actual message size (compact slot stride).
inline size_t ddaLL128ArFootprintForBytes(int nRanks, size_t bytes)
{
    return (size_t)2 * (size_t)nRanks * ddaLL128LinesForBytes(bytes) * kLL128LineBytes;
}

} // namespace RcclUnitTesting
