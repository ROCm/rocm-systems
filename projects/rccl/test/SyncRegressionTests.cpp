// Known-defect tests for bugs introduced or exposed by the NCCL 2.30.7 sync.
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
#include <cstdint>
#include <vector>

namespace RcclUnitTesting {

// =========================================================================
// Switch fallthrough in printIbWcStatusHint
// src/transport/net_ib/common.h:628-646 (sync branch)
//
//   switch (status) {
//   case IBV_WC_LOC_PROT_ERR:
//     INFO("ACS hint...");
//   case IBV_WC_WR_FLUSH_ERR:      // falls through — no break
//     INFO("NIC hint...");
//   case IBV_WC_RETRY_EXC_ERR:     // falls through — no break
//     INFO("TIMEOUT hint...");
//   default: break;
//   }
//
// Each case prints its own hints PLUS all subsequent cases' hints.
// IBV_WC_LOC_PROT_ERR prints ACS + NIC + TIMEOUT hints (wrong).
// =========================================================================

enum TestIbWcStatus {
  TEST_IBV_WC_SUCCESS = 0,
  TEST_IBV_WC_LOC_PROT_ERR = 5,
  TEST_IBV_WC_WR_FLUSH_ERR = 6,
  TEST_IBV_WC_RETRY_EXC_ERR = 7,
};

static std::vector<int> simulatePrintIbWcStatusHint(int status) {
  std::vector<int> hintsEmitted;

  // Exact reproduction of the buggy switch from common.h
  switch (status) {
  case TEST_IBV_WC_LOC_PROT_ERR:
    hintsEmitted.push_back(TEST_IBV_WC_LOC_PROT_ERR);  // ACS hint
    // MISSING: break;
  case TEST_IBV_WC_WR_FLUSH_ERR:
    hintsEmitted.push_back(TEST_IBV_WC_WR_FLUSH_ERR);  // NIC hint
    // MISSING: break;
  case TEST_IBV_WC_RETRY_EXC_ERR:
    hintsEmitted.push_back(TEST_IBV_WC_RETRY_EXC_ERR); // TIMEOUT hint
    // MISSING: break;
  default:
    break;
  }
  return hintsEmitted;
}

TEST(IbWcStatusHintTest, LocProtErrShowsOnlyAcsHint) {
  auto hints = simulatePrintIbWcStatusHint(TEST_IBV_WC_LOC_PROT_ERR);
  EXPECT_EQ(hints.size(), 1u)
    << "IBV_WC_LOC_PROT_ERR should emit 1 hint (ACS), but emitted "
    << hints.size() << " due to switch fallthrough";
}

TEST(IbWcStatusHintTest, FlushErrShowsOnlyNicHint) {
  auto hints = simulatePrintIbWcStatusHint(TEST_IBV_WC_WR_FLUSH_ERR);
  EXPECT_EQ(hints.size(), 1u)
    << "IBV_WC_WR_FLUSH_ERR should emit 1 hint (NIC), but emitted "
    << hints.size() << " due to switch fallthrough";
}

TEST(IbWcStatusHintTest, RetryErrShowsOnlyTimeoutHint) {
  auto hints = simulatePrintIbWcStatusHint(TEST_IBV_WC_RETRY_EXC_ERR);
  // This case is the last before default, so it doesn't fall through
  EXPECT_EQ(hints.size(), 1u)
    << "IBV_WC_RETRY_EXC_ERR should emit 1 hint (TIMEOUT)";
}

// =========================================================================
// PAT preconnect guard reverted by NCCL 2.30 sync
// src/group.cc:232 and :656
//
// develop branch (correct for ROCm):
//   if ((comm->cuMemSupport || algoNeedConnect[NCCL_ALGO_PAT]) && needConnect)
//
// sync branch (reverted to upstream NCCL):
//   if (comm->cuMemSupport && needConnect)
//
// On ROCm, cuMemSupport is typically false. The develop branch added the
// algoNeedConnect[NCCL_ALGO_PAT] clause so PAT QPs can be preconnected
// without cuMem. Removing it means PAT collectives silently skip
// preconnect and fail at runtime.
// =========================================================================

TEST(PatPreconnectTest, PatPreconnectEnteredWithoutCuMemSupport) {
  bool cuMemSupport = false;  // ROCm default
  bool needConnect = true;
  bool algoNeedConnect_PAT = true;  // PAT algorithm selected

  // sync branch logic (reverted):
  bool syncBranchEntersPreconnect = (cuMemSupport && needConnect);

  // develop branch logic (ROCm fix):
  bool developBranchEntersPreconnect =
    ((cuMemSupport || algoNeedConnect_PAT) && needConnect);

  EXPECT_EQ(syncBranchEntersPreconnect, developBranchEntersPreconnect)
    << "PAT preconnect skipped on ROCm: cuMemSupport=false causes "
       "sync branch to skip preconnect even when PAT algorithm needs it; "
       "develop branch correctly enters via algoNeedConnect[NCCL_ALGO_PAT]";
}

TEST(PatPreconnectTest, NonPatAlgoWithoutCuMemSkipsPreconnect) {
  bool cuMemSupport = false;
  bool needConnect = true;
  bool algoNeedConnect_PAT = false;  // non-PAT algorithm

  // Both branches should skip preconnect when neither cuMem nor PAT is needed
  bool syncResult = (cuMemSupport && needConnect);
  bool developResult = ((cuMemSupport || algoNeedConnect_PAT) && needConnect);

  EXPECT_EQ(syncResult, developResult)
    << "Non-PAT algorithm without cuMem: both branches should agree";
}

// =========================================================================
// Socket magic mismatch: ncclSocketDefaultMagic() vs NCCL_SOCKET_MAGIC
// src/transport/net_ib/connect.cc:616 vs :1858
//
//   handle->magic = ncclSocketDefaultMagic();  // line 616 — may be overridden by env
//   ...
//   if (ibHandle->magic != NCCL_SOCKET_MAGIC)  // line 1858 — hardcoded constant
//     return ncclInvalidArgument;
//
// ncclSocketDefaultMagic() returns NCCL_SOCKET_MAGIC by default, but can
// be overridden via the NCCL_SOCKET_MAGIC env var. If overridden,
// rcclNetP2pPolicy rejects valid handles.
// =========================================================================

TEST(SocketMagicTest, OverriddenMagicPassesValidation) {
  constexpr uint64_t NCCL_SOCKET_MAGIC = 0x564ab9f2fc4b9d05;
  uint64_t customMagic = 0xDEADBEEF12345678;  // simulates env override

  // Simulate ncclSocketDefaultMagic() returning custom value
  uint64_t handleMagic = customMagic;

  // Validation in rcclNetP2pPolicy compares against hardcoded constant
  bool validationPasses = (handleMagic == NCCL_SOCKET_MAGIC);

  EXPECT_TRUE(validationPasses)
    << "Handle magic 0x" << std::hex << handleMagic
    << " rejected by hardcoded check against 0x" << NCCL_SOCKET_MAGIC
    << "; rcclNetP2pPolicy should use ncclSocketDefaultMagic() for comparison";
}

TEST(SocketMagicTest, DefaultMagicPassesValidation) {
  constexpr uint64_t NCCL_SOCKET_MAGIC = 0x564ab9f2fc4b9d05;

  // Default case: ncclSocketDefaultMagic() returns the constant
  uint64_t handleMagic = NCCL_SOCKET_MAGIC;

  bool validationPasses = (handleMagic == NCCL_SOCKET_MAGIC);
  EXPECT_TRUE(validationPasses)
    << "Default magic should always pass validation";
}

} // namespace RcclUnitTesting
