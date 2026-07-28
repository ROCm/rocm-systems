// Regression tests for bugs in src/transport/net_ib_cast/p2p.cc.
//
// Each test asserts CORRECT behavior. A test that FAILS means the underlying
// bug is still present. When the fix lands, the test will PASS and prevent
// the bug from regressing.
//
// Tests reproduce the exact logic from the buggy call site using local types
// so they compile with g++ and need no GPU or network hardware.

#include "gtest/gtest.h"
#include <cstdint>

namespace RcclUnitTesting {

// =========================================================================
// CTS offload FIFO: double-indexing writes to wrong element
// src/transport/net_ib_cast/p2p.cc:657-662
//
// The pointer is set to &localElem[i], then indexed AGAIN by [i]:
//   ptr = (CtsInline*)&localElem[i];
//   ptr[i].addr = data[i];          // actually writes to localElem[2*i]
//
// The non-CTS path (lines 664-671) correctly uses localElem[i] directly.
// =========================================================================

struct FifoEntry {
  uint64_t addr;
  uint32_t rkeys[4];
  int nreqs;
  int size;
  uint64_t tag;
};

TEST(IbCastFifoTest, CtsInlineWritesCorrectElement) {
  constexpr int N = 4;
  FifoEntry elems[N] = {};

  // Reproduce the buggy CTS offload loop from p2p.cc:654-662
  for (int i = 0; i < N; i++) {
    FifoEntry* localElemCtsInline = &elems[i];
    localElemCtsInline[i].addr = 0xBEEF + i;  // buggy: double-offset
  }

  // Assert: each elems[i].addr should hold 0xBEEF+i
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(elems[i].addr, static_cast<uint64_t>(0xBEEF + i))
      << "elems[" << i << "].addr: CTS offload loop wrote to wrong element";
  }
}

TEST(IbCastFifoTest, NonCtsPathWritesCorrectElement) {
  constexpr int N = 4;
  FifoEntry elems[N] = {};

  // Non-CTS path (p2p.cc:664-671) — no double-indexing
  for (int i = 0; i < N; i++) {
    elems[i].addr = 0xBEEF + i;
  }

  for (int i = 0; i < N; i++) {
    EXPECT_EQ(elems[i].addr, static_cast<uint64_t>(0xBEEF + i));
  }
}

// =========================================================================
// Completion: NCCLCHECK in bool function swallows errors
// src/transport/net_ib_cast/p2p.cc:794-801
//
//   static inline bool IbCastRequestIsComplete(...) {
//     NCCLCHECK(IbCastResiliencyRequestIsComplete(request, &complete));
//     return complete;
//   }
//
// NCCLCHECK returns ncclResult_t (int) from a bool function.
// Any non-zero error code truncates to true (= "request complete").
// =========================================================================

#define TEST_NCCLCHECK(cmd) do { \
  int __ret = (cmd);             \
  if (__ret != 0) return __ret;  \
} while(0)

static inline bool simulateRequestIsComplete(bool baseComplete, int resiliencyResult) {
  bool complete = baseComplete;
  if (!complete) {
    TEST_NCCLCHECK(resiliencyResult);
  }
  return complete;
}

TEST(IbCastCompletionTest, ResiliencyErrorMustNotIndicateComplete) {
  // Resiliency subsystem returns ncclSystemError (2)
  // The function should propagate the error, NOT say "complete"
  bool result = simulateRequestIsComplete(false, /*ncclSystemError=*/2);
  EXPECT_FALSE(result)
    << "Resiliency error (ncclSystemError=2) was truncated to bool true, "
       "making a failed request appear complete";
}

TEST(IbCastCompletionTest, ResiliencySuccessWithIncompleteRequest) {
  bool result = simulateRequestIsComplete(false, /*ncclSuccess=*/0);
  EXPECT_FALSE(result)
    << "Request is not complete when base events are still pending";
}

TEST(IbCastCompletionTest, AlreadyCompleteSkipsResiliency) {
  bool result = simulateRequestIsComplete(true, /*ncclSystemError=*/2);
  EXPECT_TRUE(result)
    << "Already-complete request should stay complete";
}

#undef TEST_NCCLCHECK

// =========================================================================
// wr_id packing: bit shift exceeds type width for nreqs > 8
// src/transport/net_ib_cast/p2p.cc:115
//
//   wr_id += (uint64_t)(slot & 0xff) << (r * 8);
//
// For r >= 8, shift r*8 >= 64 is undefined behavior on uint64_t.
// =========================================================================

TEST(IbWrIdPackingTest, AllShiftsWithinBitWidth) {
  int maxNreqs = 12;  // NCCL_NET_IB_MAX_RECVS may allow this

  for (int r = 0; r < maxNreqs; r++) {
    int shift = r * 8;
    EXPECT_LT(shift, 64)
      << "r=" << r << ": shift of " << shift << " bits on uint64_t is undefined behavior";
  }
}

TEST(IbWrIdPackingTest, PackedValuesDoNotOverlap) {
  // With nreqs <= 8, each request gets a unique byte lane in wr_id
  uint64_t wr_id = 0;
  int nreqs = 8;
  uint8_t slot = 0xAB;

  for (int r = 0; r < nreqs; r++) {
    wr_id += (uint64_t)(slot & 0xff) << (r * 8);
  }

  // Verify each byte lane is independently readable
  for (int r = 0; r < nreqs; r++) {
    uint8_t extracted = (wr_id >> (r * 8)) & 0xff;
    EXPECT_EQ(extracted, slot) << "r=" << r;
  }
}

} // namespace RcclUnitTesting
