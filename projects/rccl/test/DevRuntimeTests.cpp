/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only micro-tests for src/dev_runtime.cc.
 *
 * This translation unit #includes the (hipified) dev_runtime.cc source
 * directly so it can reach the file-static symMemory* helpers. It links no
 * librccl.so; every dependency the source references is satisfied by no-op
 * stubs in DevRuntimeTestsStubs.cc. See the rccl-microtest conventions.
 *************************************************************************/

#include DEV_RUNTIME_CC_PATH

#include <gtest/gtest.h>

// First milestone: just prove the source compiles/links into a host binary.
TEST(DevRuntime, TranslationUnitBuilds) {
  SUCCEED();
}
