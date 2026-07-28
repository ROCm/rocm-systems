// Known-defect tests for PIPE_READ macro in test infrastructure.
//
// These tests reproduce buggy code patterns in isolation using local types.
// They document and assert the presence of known defects — a FAILING test
// confirms the bug pattern still exists in the codebase. They do NOT compile
// production RCCL source, so they will not automatically pass when a fix
// lands in production code. When a fix is applied, update or remove the
// corresponding test.

#include "gtest/gtest.h"
#include <cstddef>
#include <sys/types.h>

namespace RcclUnitTesting {

// =========================================================================
// PIPE_READ macro: partial read check uses sizeof(int) not sizeof(val)
// test/common/TestBed.cpp:29
//
//   ssize_t retval = read(fd, &val, sizeof(val));
//   if (retval < sizeof(int)) { FAIL; }     // should be sizeof(val)
//
// For types > int (ncclUniqueId = 128 bytes), partial reads are undetected.
// =========================================================================

TEST(PipeReadTest, PartialReadDetectedForLargeTypes) {
  struct LargePayload { char data[128]; };
  ssize_t bytesRead = 64;  // partial: got 64 of 128 bytes

  // Buggy check: sizeof(int) = 4
  bool buggyCheck = (bytesRead < static_cast<ssize_t>(sizeof(int)));
  // Correct check: sizeof(LargePayload) = 128
  bool correctCheck = (bytesRead < static_cast<ssize_t>(sizeof(LargePayload)));

  EXPECT_EQ(buggyCheck, correctCheck)
    << "PIPE_READ partial-read check uses sizeof(int)=" << sizeof(int)
    << " instead of sizeof(val)=" << sizeof(LargePayload)
    << "; 64-byte partial read goes undetected";
}

TEST(PipeReadTest, IntSizedTypeIsMasked) {
  int val;
  ssize_t bytesRead = 2;  // partial read of int

  bool buggyCheck = (bytesRead < static_cast<ssize_t>(sizeof(int)));
  bool correctCheck = (bytesRead < static_cast<ssize_t>(sizeof(val)));

  EXPECT_EQ(buggyCheck, correctCheck)
    << "For int-sized values, both checks should agree";
}

} // namespace RcclUnitTesting
