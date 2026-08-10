// Modification Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "param.h"
#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include "common/ProcessIsolatedTestRunner.hpp"
#include "graph/topo.h"

// Declared in src/include/param/utils.h (new param subsystem) as an
// extern "C" exported symbol. Forward-declared here so the test can call it
// without pulling in the new param headers / include paths.
extern "C" bool ncclParamIsCacheDisabled(const char* key);

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

// Key used by the NCCL_NO_CACHE tests below. The actual name is irrelevant as
// long as it is not a real NCCL parameter; ncclLoadParam reads it via getenv.
static constexpr const char* kNoCacheTestKey = "NCCL_TEST_NOCACHE_KEY";

// Baseline: when NCCL_NO_CACHE is unset, ncclLoadParam caches the first value
// it reads. A subsequent change to the environment must be ignored and the
// originally cached value returned.
TEST(ParamTests, NoCache_DefaultCachesValue) {
  RUN_ISOLATED_TEST(
      "NoCache_DefaultCachesValue",
      []()
      {
          constexpr int64_t uninitialized = INT64_MIN;
          int64_t cache = uninitialized;
          int8_t  noCache = -1; // uninitialized sentinel, matches NCCL_PARAM macro

          unsetenv("NCCL_NO_CACHE");

          setenv(kNoCacheTestKey, "100", 1);
          int64_t v1 = ncclLoadParam(kNoCacheTestKey, /*deftVal*/ 999, uninitialized, &cache, &noCache);
          ASSERT_EQ(v1, 100);

          // The environment changed, but the cached value must win.
          setenv(kNoCacheTestKey, "200", 1);
          int64_t v2 = ncclLoadParam(kNoCacheTestKey, /*deftVal*/ 999, uninitialized, &cache, &noCache);
          ASSERT_EQ(v2, 100) << "value should stay cached when NCCL_NO_CACHE is unset";

          unsetenv(kNoCacheTestKey);
      }
  );
}

// NCCL_NO_CACHE listing the specific key disables caching for it, so a later
// environment change is observed on the next load.
TEST(ParamTests, NoCache_PerKeyForcesReRead) {
  RUN_ISOLATED_TEST(
      "NoCache_PerKeyForcesReRead",
      []()
      {
          constexpr int64_t uninitialized = INT64_MIN;
          int64_t cache = uninitialized;
          int8_t  noCache = -1;

          // Must be set before the first ncclLoadParam call: ncclParamIsCacheDisabled
          // parses NCCL_NO_CACHE exactly once (std::call_once).
          setenv("NCCL_NO_CACHE", kNoCacheTestKey, 1);

          setenv(kNoCacheTestKey, "100", 1);
          int64_t v1 = ncclLoadParam(kNoCacheTestKey, /*deftVal*/ 999, uninitialized, &cache, &noCache);
          ASSERT_EQ(v1, 100);

          setenv(kNoCacheTestKey, "200", 1);
          int64_t v2 = ncclLoadParam(kNoCacheTestKey, /*deftVal*/ 999, uninitialized, &cache, &noCache);
          ASSERT_EQ(v2, 200) << "value should be re-read when the key is listed in NCCL_NO_CACHE";

          unsetenv(kNoCacheTestKey);
      }
  );
}

// NCCL_NO_CACHE=ALL disables caching for every key.
TEST(ParamTests, NoCache_AllForcesReRead) {
  RUN_ISOLATED_TEST(
      "NoCache_AllForcesReRead",
      []()
      {
          constexpr int64_t uninitialized = INT64_MIN;
          int64_t cache = uninitialized;
          int8_t  noCache = -1;

          setenv("NCCL_NO_CACHE", "ALL", 1);

          setenv(kNoCacheTestKey, "100", 1);
          int64_t v1 = ncclLoadParam(kNoCacheTestKey, /*deftVal*/ 999, uninitialized, &cache, &noCache);
          ASSERT_EQ(v1, 100);

          setenv(kNoCacheTestKey, "200", 1);
          int64_t v2 = ncclLoadParam(kNoCacheTestKey, /*deftVal*/ 999, uninitialized, &cache, &noCache);
          ASSERT_EQ(v2, 200) << "NCCL_NO_CACHE=ALL should disable caching for every key";

          unsetenv(kNoCacheTestKey);
      }
  );
}

// NCCL_NO_CACHE must never disable caching for itself (would be a circular
// dependency while resolving the list), even when set to ALL. Other keys are
// still reported as cache-disabled.
TEST(ParamTests, NoCache_SelfIsNeverDisabled) {
  RUN_ISOLATED_TEST(
      "NoCache_SelfIsNeverDisabled",
      []()
      {
          setenv("NCCL_NO_CACHE", "ALL", 1);
          ASSERT_FALSE(ncclParamIsCacheDisabled("NCCL_NO_CACHE"))
              << "NCCL_NO_CACHE must never disable caching for itself";
          ASSERT_TRUE(ncclParamIsCacheDisabled("NCCL_SOME_OTHER_KEY"))
              << "ALL should still disable caching for arbitrary keys";
      }
  );
}
} // namespace RcclUnitTesting
