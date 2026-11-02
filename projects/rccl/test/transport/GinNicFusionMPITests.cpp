/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fusing NICs into vNICs must not take GIN or the host RMA proxy away from the
// communicator. Off the symmetric-memory path, so these need no cuMem.

#include "MPITestBase.hpp"
#include "TestChecks.hpp"

#include "graph.h"        // ncclTopoCheckNicFused()
#include "nccl_device.h"  // ncclCommQueryProperties(), ncclCommProperties_t

#include <cstdlib>
#include <cstring>
#include <string>

#ifdef MPI_TESTS_ENABLED

using namespace MPITestConstants;

namespace {

// GTEST_SKIP() returns out of the body, so preconditions are reduced across ranks
// first, through mpiAnyRank / mpiAllRanks in TestChecks.hpp.

bool EnvIsZero(const char* name) {
  const char* e = std::getenv(name);
  return e && std::strcmp(e, "0") == 0;
}

// With GIN off every assertion below would be testing the disable switch instead.
bool GinDisabledByEnv() { return EnvIsZero("NCCL_GIN_ENABLE"); }

}  // namespace

class GinNicFusionMPITest : public MPITestBase {
 protected:
  // Brings up a communicator and establishes that this run really fused NICs somewhere.
  bool setUpFusedComm(ncclCommProperties_t* props) {
    skipReason_.clear();

    if (mpiAnyRank(GinDisabledByEnv())) {
      skipReason_ = "GIN disabled by environment (NCCL_GIN_ENABLE=0)";
      return false;
    }

    // hostRmaSupport has terms beyond fusion; clearing one is an unjudgeable
    // configuration, not a regression, so skip rather than report red.
    if (mpiAnyRank(EnvIsZero("NCCL_CROSS_NIC"))) {
      skipReason_ = "NCCL_CROSS_NIC=0 clears globalCrossNicSupport, which hostRmaSupport also needs";
      return false;
    }
    if (mpiAnyRank(EnvIsZero("NCCL_NUM_RMA_CTX"))) {
      skipReason_ = "NCCL_NUM_RMA_CTX=0 clears hostRmaSupport independently of NIC fusion";
      return false;
    }
#ifndef RCCL_HAS_RMA_IB_PROXY
    skipReason_ = "Built without the IB RMA proxy, so hostRmaSupport is false regardless of fusion";
    return false;
#endif

    // Reduced like the skips: a lone early return hangs the peers and buries the cause.
    if (mpiAnyRank(createTestCommunicator() != ncclSuccess)) {
      ADD_FAILURE() << "createTestCommunicator failed on this rank or a peer";
      return false;
    }
    ncclComm_t comm = getActiveCommunicator();

    *props = NCCL_COMM_PROPERTIES_INITIALIZER;
    if (mpiAnyRank(ncclCommQueryProperties(comm, props) != ncclSuccess)) {
      ADD_FAILURE() << "ncclCommQueryProperties failed on this rank or a peer";
      return false;
    }

    // A single LSA team makes hostRmaSupport true regardless of what the proxy reports.
    if (!mpiAllRanks(props->nLsaTeams > 1)) {
      skipReason_ = "Needs ranks spread over more than one LSA team (run on >=2 nodes)";
      return false;
    }

    // Per-rank device table; the derivation ORs it across ranks, so match that here.
    bool localFused = false;
    if (mpiAnyRank(ncclTopoCheckNicFused(comm, &localFused) != ncclSuccess)) {
      ADD_FAILURE() << "ncclTopoCheckNicFused failed on this rank or a peer";
      return false;
    }
    if (!mpiAnyRank(localFused)) {
      skipReason_ =
        "No fused vNIC on any rank. Needs >=2 NICs plus NCCL_IB_MERGE_NICS=1 and "
        "NCCL_NET_FORCE_MERGE (or NCCL_NET_MERGE_LEVEL) naming NICs present on every node";
      return false;
    }
    return true;
  }

  std::string skipReason_;
};

// skipReason_ set means a precondition miss; empty means ADD_FAILURE already fired.
#define RETURN_OR_SKIP()                                   \
  do {                                                     \
    if (!skipReason_.empty()) GTEST_SKIP() << skipReason_; \
    return;                                                \
  } while (0)

// The proxy used to be switched off whenever any rank had fused NICs, leaving
// multi-node GIN without its host RMA path. Fusion does not make the proxy unusable.
TEST_F(GinNicFusionMPITest, HostRmaSurvivesNicFusion) {
  SKIP_UNLESS_MPI_PREREQS(2, kNoProcessLimit, kNoPowerOfTwoRequired, /*min_nodes=*/2);

  ncclCommProperties_t props{};
  if (!setUpFusedComm(&props)) RETURN_OR_SKIP();

  EXPECT_TRUE(props.hostRmaSupport)
    << "NIC fusion is active and the communicator spans " << props.nLsaTeams
    << " LSA teams, so the host RMA proxy must still be available to GIN";
}

// The other half: fusion must not drop the communicator to ginType NONE.
TEST_F(GinNicFusionMPITest, GinTypeSurvivesNicFusion) {
  SKIP_UNLESS_MPI_PREREQS(2, kNoProcessLimit, kNoPowerOfTwoRequired, /*min_nodes=*/2);

  ncclCommProperties_t props{};
  if (!setUpFusedComm(&props)) RETURN_OR_SKIP();

  EXPECT_NE(NCCL_GIN_TYPE_NONE, props.ginType)
    << "NIC fusion is active, but the communicator reports no usable GIN backend";
}

#endif  // MPI_TESTS_ENABLED
