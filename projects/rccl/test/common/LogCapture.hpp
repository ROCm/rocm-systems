/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include <gtest/gtest.h>

#include <string>
#include <utility>

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

}  // namespace RcclUnitTesting
