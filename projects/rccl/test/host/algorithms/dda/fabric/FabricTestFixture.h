/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Shared fixture for the src/algorithms/dda/fabric/ tests.

#ifndef RCCL_TEST_HOST_ALGORITHMS_DDA_FABRIC_FABRICTESTFIXTURE_H_
#define RCCL_TEST_HOST_ALGORITHMS_DDA_FABRIC_FABRICTESTFIXTURE_H_

#include <gtest/gtest.h>

#include "HipVmmLedger.h"
#include "fakes/bootstrap_stubs.h"
#include "fakes/env_fakes.h"
#include "fakes/hip_fakes.h"
#include "fakes/nccl_fakes.h"

#include "alloc.h"

namespace dda_fabric_test {

constexpr int kNRanks = 4;
constexpr int kRank = 1;
inline void* const kBootstrap = reinterpret_cast<void*>(0xB007);

class FabricLedgerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    SetMicroEnvAbsent("NCCL_CUMEM_SKIP_FREE");
    ledger_.Install();
    ASSERT_FALSE(rcclSkipCuMemFree());  // alloc.h caches this once per process
  }

  // Call after destroying the unit under test.
  void TearDown() override {
    EXPECT_TRUE(LedgerClean());
    ResetBootstrapStubs();
    ResetHipFakes();
    ResetNcclFakes();
    ResetEnvFakes();
  }

  ::testing::AssertionResult LedgerClean() const {
    if (ledger_.Clean()) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << ledger_.Summary();
  }

  HipVmmLedger ledger_;
};

}  // namespace dda_fabric_test

#endif  // RCCL_TEST_HOST_ALGORITHMS_DDA_FABRIC_FABRICTESTFIXTURE_H_
