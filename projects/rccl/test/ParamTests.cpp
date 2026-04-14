// Modification Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "param.h"
#include <gtest/gtest.h>
#include <rccl/rccl.h>

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
  // This function call reads and opens the conf file from the path
  // which is set using env. variable NCCL_CONF_FILE
  initEnv();

  EXPECT_EQ(getenv("TEST_VAR_WITH_NO_VALUE"), nullptr);
  EXPECT_STREQ(getenv("TEST_VAR"), "12345");

  // Clean up
  unsetenv("TEST_VAR_WITH_NO_VALUE");
  unsetenv("TEST_VAR");
}

TEST(ParamTests, ncclLoadParam_InvalidParam) {
  int64_t cache = -1;
  const int64_t defaultVal = 12345; // Dummy input value

  // Force overflow: value exceeds int64_t max (9223372036854775807)
  setenv("TEST_INVALID_PARAM", "99999999999999999999",
         1); // Dummy variable and value
  ncclLoadParam("TEST_INVALID_PARAM", defaultVal, -1, &cache);
  unsetenv("TEST_INVALID_PARAM");

  EXPECT_EQ(cache, defaultVal); // Cache should be set to default value
}
} // namespace RcclUnitTesting
