/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Common ground for the src/algorithms/dda/fabric/ microtests: the comm shape
// they share, device memory tracked through HipVmmLedger, and the fake resets
// every fixture here needs. What differs per unit -- how its allgather is
// modelled, what it records -- stays in its own fixture.

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

// A 4-rank clique seen from rank 1: neither first nor last.
constexpr int kNRanks = 4;
constexpr int kRank = 1;
inline void* const kBootstrap = reinterpret_cast<void*>(0xB007);

class FabricLedgerTest : public ::testing::Test {
 protected:
  // Derived fixtures call this first and return if it failed fatally.
  void SetUp() override {
    SetMicroEnvAbsent("NCCL_CUMEM_SKIP_FREE");
    ledger_.Install();
    // alloc.h memoises once per process whether ncclCuMemFreeAddr skips peer
    // unmaps. Latch it now, under this env and the emulator's gfx900, so no
    // test can find it already latched the other way.
    ASSERT_FALSE(rcclSkipCuMemFree());
  }

  // Derived fixtures destroy their unit first, so its frees still go through
  // the ledger, then call this.
  void TearDown() override {
    EXPECT_TRUE(LedgerClean());
    ResetBootstrapStubs();
    ResetHipFakes();
    ResetNcclFakes();
    ResetEnvFakes();
  }

  // Nothing device-side left allocated, and no call the ledger refused.
  ::testing::AssertionResult LedgerClean() const {
    if (ledger_.Clean()) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure() << ledger_.reserved.size() << " reservations, " << ledger_.liveHandles.size()
                                         << " handles, " << ledger_.liveBuffers.size() << " buffers live; "
                                         << ledger_.rejected.size() << " calls refused";
  }

  HipVmmLedger ledger_;
};

}  // namespace dda_fabric_test

#endif  // RCCL_TEST_HOST_ALGORITHMS_DDA_FABRIC_FABRICTESTFIXTURE_H_
