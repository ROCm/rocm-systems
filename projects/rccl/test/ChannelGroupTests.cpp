// Known-defect tests for bugs in RCCL channel/group runtime.
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
#include <vector>

namespace RcclUnitTesting {

// =========================================================================
// P2P channel base: division by zero when maxLocalRanks=0
// src/include/channel.h:30-35
//
//   int nodeDelta = p2pRound / comm->maxLocalRanks;       // div-by-zero
//   int fallbackBatch = pxnEnabled ? maxLocalRanks : 1;   // 0 if pxn + mlr=0
//   base = nodeDelta * divUp(maxLocalRanks, batchSize);   // divUp(0,0) = UB
// =========================================================================

TEST(ChannelBatchSizeTest, BatchSizeNonZeroWhenMaxLocalRanksZero) {
  int maxLocalRanks = 0;
  int nNodes = 2;  // >1 enters the multi-node branch

  bool pxnEnabled = true;
  int fallbackBatch = pxnEnabled ? maxLocalRanks : 1;

  int p2pBatchEnable = 0;
  constexpr int MAX_BATCH = 8;
  int batchSize = (nNodes > 2 && p2pBatchEnable) ? MAX_BATCH : fallbackBatch;

  EXPECT_NE(batchSize, 0)
    << "batchSize=0 causes division by zero in divUp(maxLocalRanks, batchSize) "
       "and localDelta/batchSize";
  EXPECT_NE(maxLocalRanks, 0)
    << "maxLocalRanks=0 causes division by zero at p2pRound/maxLocalRanks "
       "and p2pRound%maxLocalRanks";
}

// =========================================================================
// Group end: suspend/resume queues drained twice (copy-paste)
// src/group.cc:320-325 (first drain), then 340-345 (duplicate drain)
//
// If rmaCeInitTaskQueue processing (lines 334-338) enqueues a new
// suspend task between the two drains, it gets processed twice —
// a double-suspend hazard.
// =========================================================================

TEST(GroupTaskQueueTest, SuspendTaskProcessedExactlyOnce) {
  // Model the group.cc drain structure:
  //   drain suspendQueue      (lines 320-325)
  //   drain resumeQueue       (lines 327-332)
  //   drain rmaCeInitQueue    (lines 334-338) -- may enqueue into suspendQueue
  //   drain suspendQueue AGAIN (lines 340-345) -- BUG: duplicate drain
  //   drain resumeQueue AGAIN  (lines 347-352) -- BUG: duplicate drain

  std::vector<int> suspended;
  std::vector<int> suspendQueue = {1};

  // First drain
  while (!suspendQueue.empty()) {
    suspended.push_back(suspendQueue.back());
    suspendQueue.pop_back();
  }

  // rmaCeInitTaskQueue processing triggers a new suspend
  suspendQueue.push_back(2);

  // Duplicate drain (the copy-paste bug)
  while (!suspendQueue.empty()) {
    suspended.push_back(suspendQueue.back());
    suspendQueue.pop_back();
  }

  // Task 2 was meant for the next ncclGroupEnd, not this one
  EXPECT_EQ(suspended.size(), 1u)
    << "Each group end should drain suspend queue exactly once; "
       "task 2 was processed prematurely by the duplicate drain";
}

} // namespace RcclUnitTesting
