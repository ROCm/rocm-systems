/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host unit tests for the pure GIN Anvil-SDMA put-segmentation math in
// projects/rccl/src/include/nccl_device/gin/anvil_sdma/gin_anvil_sdma_put_policy.h.
// These validate the exact loop that ncclGinApi_Put<NCCL_NET_DEVICE_GIN_ANVIL_SDMA>
// drives on device -- the header is compiled here as plain host C++
// (GIN_SDMA_HOST_ONLY drops the __host__ __device__ attributes), and the same
// ginPutSegmentCount / ginPutSegmentAt functions are called by the SDMA backend
// Put, so exercising them here exercises the real segmentation code with no GPU.
//
// Invariants asserted for every transfer size:
//   * no segment exceeds the 128 MiB single-copy cap (kGinPutSegBytes),
//   * the segments tile [0, bytes) contiguously and sum back to bytes,
//   * exactly one segment is flagged final (it alone carries the SignalInc),
//   * the final flag lands on the last segment.

#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#define GIN_SDMA_HOST_ONLY 1
#include "nccl_device/gin/anvil_sdma/gin_anvil_sdma_put_policy.h"

using namespace gin_sdma;

namespace {

constexpr size_t kMiB = 1024ull * 1024;
constexpr size_t kGiB = 1024ull * kMiB;

// The clamp the backend Put actually uses: 128 MiB (min of the reliability and
// correctness limits). Kept in lockstep with the header so a future change to
// either constant is caught here.
TEST(GinPutSeg, SegLimitIsMinOfBothCaps) {
  EXPECT_EQ(kGinPutSegBytes, kGinSdmaSafeCopyBytes);
  EXPECT_LE(kGinPutSegBytes, kGinPutMaxBytes);
  EXPECT_EQ(kGinPutSegBytes, 128ull * kMiB);
  // The 128 MiB cap must divide 1 GiB so a 1 GiB descriptor limit is a whole
  // number of safe copies, and be 32 B aligned for the SDMA copy descriptor.
  EXPECT_EQ(kGinPutMaxBytes % kGinPutSegBytes, 0u);
  EXPECT_EQ(kGinPutSegBytes % 32u, 0u);
}

// Re-derive the whole segmentation for `bytes` and assert every invariant.
void checkSegmentation(size_t bytes) {
  const size_t maxSeg = kGinPutSegBytes;
  const size_t n = ginPutSegmentCount(bytes, maxSeg);

  // Non-empty transfers => ceil(bytes/maxSeg); a zero transfer still issues one
  // (empty) put so its trailing signal fires.
  const size_t expectedN = (bytes == 0) ? 1 : (bytes + maxSeg - 1) / maxSeg;
  ASSERT_EQ(n, expectedN) << "bytes=" << bytes;
  ASSERT_GE(n, 1u) << "bytes=" << bytes;

  size_t sum = 0;
  size_t expectedOffset = 0;
  int finalCount = 0;
  for (size_t i = 0; i < n; ++i) {
    const PutSegment s = ginPutSegmentAt(bytes, maxSeg, i);

    // No segment exceeds the cap.
    EXPECT_LE(s.bytes, maxSeg) << "bytes=" << bytes << " i=" << i;

    // Segments tile [0, bytes) contiguously with no gaps or overlaps.
    EXPECT_EQ(s.offset, expectedOffset) << "bytes=" << bytes << " i=" << i;
    expectedOffset += s.bytes;

    // Only the last segment is final.
    if (s.isFinal) ++finalCount;
    EXPECT_EQ(s.isFinal, (i == n - 1)) << "bytes=" << bytes << " i=" << i;

    // Interior segments are exactly full (only the tail may be short).
    if (i + 1 < n) {
      EXPECT_EQ(s.bytes, maxSeg) << "bytes=" << bytes << " i=" << i;
    }

    sum += s.bytes;
  }

  // Exactly one signal is carried, and the segments reconstruct the transfer.
  EXPECT_EQ(finalCount, 1) << "bytes=" << bytes;
  EXPECT_EQ(sum, bytes) << "bytes=" << bytes;
  EXPECT_EQ(expectedOffset, bytes) << "bytes=" << bytes;
}

// The sizes called out in review: the 128 MiB cap boundary, the 1 GiB 30-bit
// truncation boundary, and the 2 GiB total hang repro (256 MiB/peer at 8 ranks).
TEST(GinPutSeg, CoversBoundarySizes) {
  const size_t sizes[] = {
      0,
      1,
      128 * kMiB,          // exactly one segment (at the cap)
      128 * kMiB + 1,      // just over: 2 segments (1 B tail)
      256 * kMiB,          // 2 segments; the size that HUNG as a single put
      1 * kGiB,            // 8 segments; the 30-bit single-descriptor max
      1 * kGiB + 1,        // 9 segments; just past the correctness boundary
      2 * kGiB,            // 16 segments; the original 2 GiB-total hang repro
  };
  for (size_t b : sizes) checkSegmentation(b);
}

// Segment counts for the documented boundaries (128 MiB cap).
TEST(GinPutSeg, SegmentCounts) {
  EXPECT_EQ(ginPutSegmentCount(0, kGinPutSegBytes), 1u);
  EXPECT_EQ(ginPutSegmentCount(1, kGinPutSegBytes), 1u);
  EXPECT_EQ(ginPutSegmentCount(128 * kMiB, kGinPutSegBytes), 1u);
  EXPECT_EQ(ginPutSegmentCount(128 * kMiB + 1, kGinPutSegBytes), 2u);
  EXPECT_EQ(ginPutSegmentCount(256 * kMiB, kGinPutSegBytes), 2u);
  EXPECT_EQ(ginPutSegmentCount(1 * kGiB, kGinPutSegBytes), 8u);
  EXPECT_EQ(ginPutSegmentCount(2 * kGiB, kGinPutSegBytes), 16u);
}

// A dense scan around each cap multiple catches off-by-one tiling errors.
TEST(GinPutSeg, DenseScanAroundCapMultiples) {
  const size_t maxSeg = kGinPutSegBytes;
  for (size_t k = 0; k <= 18; ++k) {
    const size_t base = k * maxSeg;
    for (long d = -2; d <= 2; ++d) {
      if (base == 0 && d < 0) continue;  // no negative sizes
      checkSegmentation(base + (size_t)d);
    }
  }
}

// maxSeg == 0 is a caller error; guard returns 0 (no puts) rather than dividing.
TEST(GinPutSeg, ZeroCapIsGuarded) {
  EXPECT_EQ(ginPutSegmentCount(1 * kGiB, 0), 0u);
}

}  // namespace
