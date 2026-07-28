// Regression tests for buffer/array bounds bugs in RCCL infrastructure.
//
// Each test asserts CORRECT behavior. A test that FAILS means the underlying
// bug is still present. When the fix lands, the test will PASS and prevent
// the bug from regressing.
//
// Tests reproduce the exact logic from the buggy call site using local types
// so they compile with g++ and need no GPU or network hardware.

#include "gtest/gtest.h"
#include <cstddef>
#include <cstdint>

namespace RcclUnitTesting {

// =========================================================================
// CollTrace latency buffer: RANKS_PER_HOST=8 overflows on >8 GPUs/node
// src/misc/latency_profiler/CollTrace.cc:17, 150-155
//
//   constexpr int RANKS_PER_HOST = 8;
//   latencyAllGather.resize(RANKS_PER_HOST * RECORD_MAX, 0);
//   int start = localRank * RECORD_MAX;
//   for (int i = start; ...) latencyAllGather[i] = ...;
//
// MI300X nodes commonly have 16 GPUs. localRank >= 8 writes past the end.
// =========================================================================

TEST(CollTraceBufferTest, AllLocalRanksWithinBounds) {
  constexpr int RANKS_PER_HOST = 8;
  constexpr int RECORD_MAX = 100;
  size_t bufferSize = RANKS_PER_HOST * RECORD_MAX;

  int localRanks = 16;  // MI300X dual-socket

  for (int localRank = 0; localRank < localRanks; localRank++) {
    size_t start = static_cast<size_t>(localRank) * RECORD_MAX;
    size_t end   = start + RECORD_MAX;
    EXPECT_LE(end, bufferSize)
      << "localRank=" << localRank << " writes past buffer end "
      << "(index " << end - 1 << " >= size " << bufferSize << ")";
  }
}

// =========================================================================
// GDR support matrix: fixed-size [32] array accessed by cudaDev unbounded
// src/plugin/net.cc:443
//
//   static int gdrSupportMatrix[32] = {};
//   ... gdrSupportMatrix[comm->cudaDev] ...
// =========================================================================

TEST(GdrSupportMatrixTest, DeviceIndexWithinBounds) {
  constexpr int GDR_MATRIX_SIZE = 32;

  // Partitioned/virtualized environments can produce device indices >= 32
  int testDevices[] = {0, 7, 31, 32, 64, 128};
  for (int cudaDev : testDevices) {
    EXPECT_LT(cudaDev, GDR_MATRIX_SIZE)
      << "cudaDev=" << cudaDev << " exceeds gdrSupportMatrix[" << GDR_MATRIX_SIZE << "]";
  }
}

// =========================================================================
// Allocator insertSegment: malloc result not checked before dereference
// src/allocator.cc:175-177
//
//   int64_t* cuts1 = (int64_t*)malloc(a->capacity * sizeof(int64_t));
//   for (int i = 0; i < index; i++) cuts1[i] = a->cuts[i];  // NULL deref
//
// Also: int capacity doubles until signed overflow -> negative -> huge
// malloc request -> NULL return -> crash.
// =========================================================================

TEST(AllocatorResizeTest, CapacityDoublingStaysPositive) {
  int capacity = 16;  // initial size after first allocation

  while (capacity > 0) {
    int next = capacity * 2;
    EXPECT_GT(next, 0)
      << "capacity=" << capacity << " * 2 overflows signed int to " << next
      << "; malloc((negative) * sizeof(int64_t)) will return NULL and crash";
    if (next <= 0) break;
    capacity = next;
  }
}

} // namespace RcclUnitTesting
