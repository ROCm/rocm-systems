// Modification Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "param.h"
#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include "common/ProcessIsolatedTestRunner.hpp"

// Forward-declare rcclParamLazyKernelInit() / rcclParamKernelInitPrefetch() in
// the global namespace so the regression tests below can call into the same
// symbols that init.cc defines.
RCCL_PARAM_DECLARE(LazyKernelInit);
RCCL_PARAM_DECLARE(KernelInitPrefetch);

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

// Regression test for the lazy kernel-init env-var alias: the original PR
// declared the param with RCCL_PARAM, which only honored RCCL_LAZY_KERNEL_INIT
// even though the commit message advertised NCCL_LAZY_KERNEL_INIT as well.
// The fix switches to RCCL_PARAM_NCCL_ALIAS; this test pins that contract.
TEST(ParamTests, LazyKernelInit_NcclAlias_IsRead) {
  // ncclLoadParam reads via getenv; since rcclParamLazyKernelInit caches its
  // result in a function-local static, we run inside an isolated subprocess
  // so each variant gets a clean cache.
  RUN_ISOLATED_TEST(
      "LazyKernelInit_NcclAlias_OnlyNcclSet",
      []()
      {
          unsetenv("RCCL_LAZY_KERNEL_INIT");
          setenv("NCCL_LAZY_KERNEL_INIT", "1", 1);
          ASSERT_EQ(rcclParamLazyKernelInit(), 1);
          unsetenv("NCCL_LAZY_KERNEL_INIT");
      });
  RUN_ISOLATED_TEST(
      "LazyKernelInit_NcclAlias_OnlyRcclSet",
      []()
      {
          unsetenv("NCCL_LAZY_KERNEL_INIT");
          setenv("RCCL_LAZY_KERNEL_INIT", "1", 1);
          ASSERT_EQ(rcclParamLazyKernelInit(), 1);
          unsetenv("RCCL_LAZY_KERNEL_INIT");
      });
  RUN_ISOLATED_TEST(
      "LazyKernelInit_NcclAlias_RcclWinsOverNccl",
      []()
      {
          setenv("RCCL_LAZY_KERNEL_INIT", "0", 1);
          setenv("NCCL_LAZY_KERNEL_INIT", "1", 1);
          ASSERT_EQ(rcclParamLazyKernelInit(), 0);
          unsetenv("RCCL_LAZY_KERNEL_INIT");
          unsetenv("NCCL_LAZY_KERNEL_INIT");
      });
  RUN_ISOLATED_TEST(
      "LazyKernelInit_NcclAlias_DefaultIsZero",
      []()
      {
          unsetenv("RCCL_LAZY_KERNEL_INIT");
          unsetenv("NCCL_LAZY_KERNEL_INIT");
          ASSERT_EQ(rcclParamLazyKernelInit(), 0);
      });
}

// Same alias contract for the async prefetch param: RCCL_KERNEL_INIT_PREFETCH
// and NCCL_KERNEL_INIT_PREFETCH must both be honored, RCCL_ taking precedence,
// default 0. Mirrors LazyKernelInit_NcclAlias_IsRead.
TEST(ParamTests, KernelInitPrefetch_NcclAlias_IsRead) {
  RUN_ISOLATED_TEST(
      "KernelInitPrefetch_NcclAlias_OnlyNcclSet",
      []()
      {
          unsetenv("RCCL_KERNEL_INIT_PREFETCH");
          setenv("NCCL_KERNEL_INIT_PREFETCH", "1", 1);
          ASSERT_EQ(rcclParamKernelInitPrefetch(), 1);
          unsetenv("NCCL_KERNEL_INIT_PREFETCH");
      });
  RUN_ISOLATED_TEST(
      "KernelInitPrefetch_NcclAlias_OnlyRcclSet",
      []()
      {
          unsetenv("NCCL_KERNEL_INIT_PREFETCH");
          setenv("RCCL_KERNEL_INIT_PREFETCH", "1", 1);
          ASSERT_EQ(rcclParamKernelInitPrefetch(), 1);
          unsetenv("RCCL_KERNEL_INIT_PREFETCH");
      });
  RUN_ISOLATED_TEST(
      "KernelInitPrefetch_NcclAlias_RcclWinsOverNccl",
      []()
      {
          setenv("RCCL_KERNEL_INIT_PREFETCH", "0", 1);
          setenv("NCCL_KERNEL_INIT_PREFETCH", "1", 1);
          ASSERT_EQ(rcclParamKernelInitPrefetch(), 0);
          unsetenv("RCCL_KERNEL_INIT_PREFETCH");
          unsetenv("NCCL_KERNEL_INIT_PREFETCH");
      });
  RUN_ISOLATED_TEST(
      "KernelInitPrefetch_NcclAlias_DefaultIsZero",
      []()
      {
          unsetenv("RCCL_KERNEL_INIT_PREFETCH");
          unsetenv("NCCL_KERNEL_INIT_PREFETCH");
          ASSERT_EQ(rcclParamKernelInitPrefetch(), 0);
      });
}
} // namespace RcclUnitTesting
