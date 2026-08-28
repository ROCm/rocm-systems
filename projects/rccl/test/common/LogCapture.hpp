/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <utility>

// Declared here rather than including debug.h, which drags in the whole RCCL
// logging layer; the definitions are supplied by whatever the test links.
extern int ncclDebugLevel;
extern uint64_t ncclDebugMask;

namespace RcclUnitTesting {

/**
 * @brief Run @p body and return everything it wrote to stderr.
 *
 * RCCL's WARN/INFO macros go to stderr, so a test that needs to assert on what
 * a function *reported* — rather than only on what it returned — has to capture
 * the stream. gtest exposes that as CaptureStderr/GetCapturedStderr under
 * testing::internal; this wrapper keeps the dependency on an internal API in
 * one place instead of spreading it across test files.
 *
 * Captures are not nestable: gtest keeps a single stderr capture slot, so a
 * second CaptureLog inside @p body would abort.
 *
 * @par Example:
 * @code
 * ncclResult_t res = ncclSuccess;
 * const std::string log = CaptureLog([&]() { res = doSomething(); });
 * EXPECT_EQ(res, ncclInvalidUsage);
 * EXPECT_NE(log.find("expected warning"), std::string::npos);
 * @endcode
 */
template <typename Fn>
std::string CaptureLog(Fn&& body) {
  testing::internal::CaptureStderr();
  std::forward<Fn>(body)();
  return testing::internal::GetCapturedStderr();
}

/**
 * @brief True if @p log contains @p needle.
 *
 * The microtest targets link GTest::GTest only, so ::testing::HasSubstr is
 * unavailable; this is the substring check CaptureLog's callers need.
 */
inline bool LogHas(const std::string& log, const char* needle) {
  return log.find(needle) != std::string::npos;
}

/**
 * @brief Raise ncclDebugLevel/ncclDebugMask for a scope, then restore both.
 *
 * INFO is gated on ncclDebugLevel, which the fake logging layer pins to
 * NCCL_LOG_NONE, so an INFO-emitting path writes nothing and CaptureLog returns
 * an empty string. WARN and VERSION are ungated and need no guard. Construct
 * this before CaptureLog. Values are passed in so the caller keeps the
 * dependency on the NCCL_LOG_* / NCCL_ALL constants.
 */
class ScopedDebugLogging {
 public:
  ScopedDebugLogging(int level, uint64_t mask) : level_(ncclDebugLevel), mask_(ncclDebugMask) {
    ncclDebugLevel = level;
    ncclDebugMask = mask;
  }
  ~ScopedDebugLogging() {
    ncclDebugLevel = level_;
    ncclDebugMask = mask_;
  }
  ScopedDebugLogging(const ScopedDebugLogging&) = delete;
  ScopedDebugLogging& operator=(const ScopedDebugLogging&) = delete;

 private:
  int level_;
  uint64_t mask_;
};

}  // namespace RcclUnitTesting
