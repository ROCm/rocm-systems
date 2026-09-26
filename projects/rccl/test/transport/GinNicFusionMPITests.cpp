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

// GTEST_SKIP() returns out of the body, so preconditions are reduced across ranks first.
bool AllRanks(bool local) {
  int v = local ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  return v != 0;
}

bool AnyRank(bool local) {
  int v = local ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  return v != 0;
}

// With GIN off every assertion below would be testing the disable switch instead.
bool GinDisabledByEnv() {
  const char* e = std::getenv("NCCL_GIN_ENABLE");
  return e && std::strcmp(e, "0") == 0;
}

}  // namespace

class GinNicFusionMPITest : public MPITestBase {
 protected:
  // Brings up a communicator and establishes that this run really fused NICs somewhere.
  bool setUpFusedComm(ncclCommProperties_t* props) {
    skipReason_.clear();

    if (AnyRank(GinDisabledByEnv())) {
      skipReason_ = "GIN disabled by environment (NCCL_GIN_ENABLE=0)";
      return false;
    }

    // Reduced like the skips: a lone early return hangs the peers and buries the cause.
    if (AnyRank(createTestCommunicator() != ncclSuccess)) {
      ADD_FAILURE() << "createTestCommunicator failed on this rank or a peer";
      return false;
    }
    ncclComm_t comm = getActiveCommunicator();

    *props = NCCL_COMM_PROPERTIES_INITIALIZER;
    if (AnyRank(ncclCommQueryProperties(comm, props) != ncclSuccess)) {
      ADD_FAILURE() << "ncclCommQueryProperties failed on this rank or a peer";
      return false;
    }

    // A single LSA team makes hostRmaSupport true regardless of what the proxy reports.
    if (!AllRanks(props->nLsaTeams > 1)) {
      skipReason_ = "Needs ranks spread over more than one LSA team (run on >=2 nodes)";
      return false;
    }

    // Per-rank device table; the derivation ORs it across ranks, so match that here.
    bool localFused = false;
    if (AnyRank(ncclTopoCheckNicFused(comm, &localFused) != ncclSuccess)) {
      ADD_FAILURE() << "ncclTopoCheckNicFused failed on this rank or a peer";
      return false;
    }
    if (!AnyRank(localFused)) {
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
  ASSERT_MPI_TRUE(validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired, 2));

  ncclCommProperties_t props{};
  if (!setUpFusedComm(&props)) RETURN_OR_SKIP();

  EXPECT_TRUE(props.hostRmaSupport)
    << "NIC fusion is active and the communicator spans " << props.nLsaTeams
    << " LSA teams, so the host RMA proxy must still be available to GIN";
}

// The other half: fusion must not drop the communicator to ginType NONE.
TEST_F(GinNicFusionMPITest, GinTypeSurvivesNicFusion) {
  ASSERT_MPI_TRUE(validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired, 2));

  ncclCommProperties_t props{};
  if (!setUpFusedComm(&props)) RETURN_OR_SKIP();

  EXPECT_NE(NCCL_GIN_TYPE_NONE, props.ginType)
    << "NIC fusion is active, but the communicator reports no usable GIN backend";
}

#endif  // MPI_TESTS_ENABLED
