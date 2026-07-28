// Known-defect tests for bounds/overflow bugs in RCCL transport layer.
//
// These tests reproduce buggy code patterns in isolation using local types.
// They document and assert the presence of known defects — a FAILING test
// confirms the bug pattern still exists in the codebase. They do NOT compile
// production RCCL source, so they will not automatically pass when a fix
// lands in production code. When a fix is applied, update or remove the
// corresponding test.
//
// Compile with g++; no GPU or network hardware needed.

#include "gtest/gtest.h"
#include <climits>
#include <cstddef>
#include <cstdint>

namespace RcclUnitTesting {

// =========================================================================
// CollNet size guard: unsigned comparison fails to catch underflow
// src/transport/coll_net.cc:340
//
//   if ((resources->maxCollBytes <= 0) || (... > MAX))
//
// maxCollBytes is size_t. Unsigned underflow wraps to SIZE_MAX which
// bypasses the <= 0 guard entirely.
// =========================================================================

TEST(CollNetSizeGuardTest, UnderflowDetectedByGuard) {
  // Simulate: a subtraction underflows size_t
  size_t maxCollBytes = static_cast<size_t>(5) - static_cast<size_t>(10);
  // Wraps to SIZE_MAX - 4

  bool guardTriggered = (maxCollBytes <= 0);
  EXPECT_TRUE(guardTriggered)
    << "size_t underflow produced " << maxCollBytes
    << " which bypassed the <= 0 guard (unsigned can never be < 0)";
}

TEST(CollNetSizeGuardTest, ExplicitNegativeOneDetected) {
  size_t maxCollBytes = static_cast<size_t>(-1);

  bool guardTriggered = (maxCollBytes <= 0);
  EXPECT_TRUE(guardTriggered)
    << "(size_t)-1 = " << maxCollBytes << " should trigger the size guard";
}

// =========================================================================
// PAT connect: signed int mask overflows on large rank counts
// src/transport/generic.cc:72
//
//   for (int mask = 1; mask < comm->nRanks; mask <<= 1)
//
// Signed left-shift overflow is undefined behavior in C/C++.
// mask = 2^30 << 1 = 2^31 overflows signed 32-bit int.
// =========================================================================

TEST(PatConnectMaskTest, MaskStaysPositiveThroughLoop) {
  int nRanks = INT_MAX;
  int mask = 1;

  while (mask < nRanks) {
    int prev = mask;
    mask <<= 1;
    EXPECT_GT(mask, 0)
      << "mask overflowed to " << mask << " (was " << prev << " before shift); "
         "signed int left-shift overflow is undefined behavior";
    if (mask <= 0) break;
  }
}

// =========================================================================
// Transport constants: TRANSPORT_PROFILER index vs NTRANSPORTS array size
// src/include/transport.h:16-22
//
//   #define NTRANSPORTS 4
//   #define TRANSPORT_PROFILER 4   // not a valid index into [NTRANSPORTS]
// =========================================================================

TEST(TransportConstantsTest, AllTransportIndicesWithinArrayBounds) {
  constexpr int NTRANSPORTS = 4;
  constexpr int TRANSPORT_P2P = 0;
  constexpr int TRANSPORT_SHM = 1;
  constexpr int TRANSPORT_NET = 2;
  constexpr int TRANSPORT_COLLNET = 3;
  constexpr int TRANSPORT_PROFILER = 4;

  int indices[] = {TRANSPORT_P2P, TRANSPORT_SHM, TRANSPORT_NET,
                   TRANSPORT_COLLNET, TRANSPORT_PROFILER};

  for (int idx : indices) {
    EXPECT_LT(idx, NTRANSPORTS)
      << "Transport index " << idx << " overflows array of size " << NTRANSPORTS;
  }
}

} // namespace RcclUnitTesting
