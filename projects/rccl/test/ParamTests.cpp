// Modification Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "param.h"
#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include "common/ProcessIsolatedTestRunner.hpp"

namespace RcclUnitTesting {
TEST(ParamTests, initEnv_ParseValidConfFile) {
  // Skip the test if NCCL_CONF_FILE is not set
  const char *value = getenv("NCCL_CONF_FILE");

  if (!value) {
    GTEST_SKIP() << "SKIPPING TEST. Set environment variable NCCL_CONF_FILE.\n"
                 << "A sample config file has been provided at: "
                    "rccl/test/ParamTestsConfFile.txt\n"
                 << "Set NCCL_CONF_FILE to the absolute path of this file to "
                    "run the test.\n";
  }
  RUN_ISOLATED_TEST(
      "initEnv_ParseValidConfFile",
      []()
      {
          // This function call reads and opens the conf file from the path
          // which is set using env. variable NCCL_CONF_FILE
          initEnv();

          ASSERT_EQ(getenv("TEST_VAR_WITH_NO_VALUE"), nullptr);
          ASSERT_STREQ(getenv("TEST_VAR"), "12345");

          // Clean up
          unsetenv("TEST_VAR_WITH_NO_VALUE");
          unsetenv("TEST_VAR");
      }
  );
}

TEST(ParamTests, ncclLoadParam_InvalidParam) {
  RUN_ISOLATED_TEST(
      "ncclLoadParam_InvalidParam",
      []()
      {
          int64_t cache = -1;
          const int64_t defaultVal = 12345; // Dummy input value

          // Force overflow: value exceeds int64_t max (9223372036854775807)
          setenv("TEST_INVALID_PARAM", "99999999999999999999",
                 1); // Dummy variable and value
          ncclLoadParam("TEST_INVALID_PARAM", defaultVal, -1, &cache);
          unsetenv("TEST_INVALID_PARAM");

          ASSERT_EQ(cache, defaultVal); // Cache should be set to default value
      }
  );
}
// ---------------------------------------------------------------------------
// NCCL_SOCKET_POLL_TIMEOUT_MSEC parsing tests (NCCL v2.29.2-1).
//
// These tests pin down the parsing behaviour of the new NCCL_PARAM
//   NCCL_PARAM(PollTimeOut, "SOCKET_POLL_TIMEOUT_MSEC", 0);
// declared in src/misc/socket.cc. Because the param symbol is not yet
// present in RCCL, this translation unit will fail to *link* against the
// current develop branch -- which is the intentional "fails today" signal.
//
// Test list is organised using James Grenning's ZOMBIES mnemonic:
//   Z  - Zero / unset                          (default value path)
//   O  - One                                   (smallest non-zero)
//   M  - Many / typical value                  (1000 ms, the docs example)
//   B  - Boundary: negative, > INT64_MAX       (sign + overflow handling)
//   I  - Interface: env var name spelling      (covered implicitly via
//                                              ncclParamPollTimeOut())
//   E  - Exceptional input: non-numeric, empty (must fall back to default)
//   S  - Simple scenarios first
// ---------------------------------------------------------------------------
extern "C" int64_t ncclParamPollTimeOut();

namespace {
constexpr const char* kEnv = "NCCL_SOCKET_POLL_TIMEOUT_MSEC";
} // namespace

TEST(ParamTests, SocketPollTimeoutMsec_Unset_DefaultsToZero) {
  RUN_ISOLATED_TEST(
      "SocketPollTimeoutMsec_Unset_DefaultsToZero",
      []() {
        unsetenv(kEnv);
        EXPECT_EQ(ncclParamPollTimeOut(), 0);
      });
}

TEST(ParamTests, SocketPollTimeoutMsec_ExplicitZero) {
  RUN_ISOLATED_TEST(
      "SocketPollTimeoutMsec_ExplicitZero",
      []() {
        setenv(kEnv, "0", 1);
        EXPECT_EQ(ncclParamPollTimeOut(), 0);
        unsetenv(kEnv);
      });
}

TEST(ParamTests, SocketPollTimeoutMsec_One) {
  RUN_ISOLATED_TEST(
      "SocketPollTimeoutMsec_One",
      []() {
        setenv(kEnv, "1", 1);
        EXPECT_EQ(ncclParamPollTimeOut(), 1);
        unsetenv(kEnv);
      });
}

TEST(ParamTests, SocketPollTimeoutMsec_TypicalValue) {
  RUN_ISOLATED_TEST(
      "SocketPollTimeoutMsec_TypicalValue",
      []() {
        setenv(kEnv, "1000", 1);
        EXPECT_EQ(ncclParamPollTimeOut(), 1000);
        unsetenv(kEnv);
      });
}

// Boundary: a negative value parses successfully into the int64_t cache.
// This pins current behaviour and serves as a flag that downstream callers
// (socketWait -> poll(.., timeout)) would receive a negative timeout, which
// poll(2) interprets as "block indefinitely". If product policy decides this
// should be rejected instead, update both the param implementation and this
// test together.
TEST(ParamTests, SocketPollTimeoutMsec_Negative_AcceptedAsIs) {
  RUN_ISOLATED_TEST(
      "SocketPollTimeoutMsec_Negative_AcceptedAsIs",
      []() {
        setenv(kEnv, "-1", 1);
        EXPECT_EQ(ncclParamPollTimeOut(), -1);
        unsetenv(kEnv);
      });
}

// Boundary high: a value that overflows int64_t must fall back to the
// declared default (0) -- same contract as ncclLoadParam_InvalidParam above.
TEST(ParamTests, SocketPollTimeoutMsec_OverflowFallsBackToDefault) {
  RUN_ISOLATED_TEST(
      "SocketPollTimeoutMsec_OverflowFallsBackToDefault",
      []() {
        setenv(kEnv, "99999999999999999999", 1);
        EXPECT_EQ(ncclParamPollTimeOut(), 0);
        unsetenv(kEnv);
      });
}

TEST(ParamTests, SocketPollTimeoutMsec_NonNumericFallsBackToDefault) {
  RUN_ISOLATED_TEST(
      "SocketPollTimeoutMsec_NonNumericFallsBackToDefault",
      []() {
        setenv(kEnv, "abc", 1);
        EXPECT_EQ(ncclParamPollTimeOut(), 0);
        unsetenv(kEnv);
      });
}

TEST(ParamTests, SocketPollTimeoutMsec_EmptyStringFallsBackToDefault) {
  RUN_ISOLATED_TEST(
      "SocketPollTimeoutMsec_EmptyStringFallsBackToDefault",
      []() {
        setenv(kEnv, "", 1);
        EXPECT_EQ(ncclParamPollTimeOut(), 0);
        unsetenv(kEnv);
      });
}

} // namespace RcclUnitTesting
