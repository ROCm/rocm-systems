// Modification Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "param.h"
#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include "common/ProcessIsolatedTestRunner.hpp"
#include "graph/topo.h"

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
          int8_t noCache = -1; // uninitialized sentinel, matches NCCL_PARAM macro
          const int64_t defaultVal = 12345; // Dummy input value

          // Force overflow: value exceeds int64_t max (9223372036854775807)
          setenv("TEST_INVALID_PARAM", "99999999999999999999",
                 1); // Dummy variable and value
          ncclLoadParam("TEST_INVALID_PARAM", defaultVal, -1, &cache, &noCache);
          unsetenv("TEST_INVALID_PARAM");

          ASSERT_EQ(cache, defaultVal); // Cache should be set to default value
      }
  );
}

TEST(ParamTests, ncclPxnC2cParam_DefaultOff) {
  RUN_ISOLATED_TEST(
      "ncclPxnC2cParam_DefaultOff",
      []()
      {
          initEnv();
          // No-op on xGMI; default must stay off.
          unsetenv("NCCL_PXN_C2C");
          ASSERT_EQ(ncclParamPxnC2c(), 0);
      }
  );
}
} // namespace RcclUnitTesting
