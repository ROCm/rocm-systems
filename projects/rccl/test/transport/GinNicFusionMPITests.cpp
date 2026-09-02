/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Communicator-level MPI tests for the GIN <-> NIC fusion interaction: fusing
// NICs into vNICs (NCCL_IB_MERGE_NICS plus NCCL_NET_FORCE_MERGE or
// NCCL_NET_MERGE_LEVEL) must not take GIN or the host RMA proxy away from the
// communicator.
//
// These are deliberately not device-GIN tests. A gin.put() kernel needs a
// symmetric window, so it needs NCCL_CUMEM_ENABLE, which is unavailable on
// architectures where cuMem does not auto-enable. The capability derivation
// guarded here runs on every architecture, so keeping the tests off the
// symmetric-memory path is what makes them runnable at all on most clusters.

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

// A rank that skips alone leaves its peers blocked at their next collective,
// because GTEST_SKIP() returns out of the test body. Every precondition below
// is therefore reduced across the world before anyone acts on it.
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

// NCCL_GIN_ENABLE=0 forces ginType to NONE for the whole process, so every
// assertion about GIN under fusion would be testing the disable switch instead.
bool GinDisabledByEnv() {
  const char* e = std::getenv("NCCL_GIN_ENABLE");
  return e && std::strcmp(e, "0") == 0;
}

}  // namespace

class GinNicFusionMPITest : public MPITestBase {
 protected:
  // Shared preamble: bring up a communicator, read back its capabilities, and
  // establish that this run actually fused NICs somewhere. Returns false when
  // the caller should stop; skipReason_ is then set and identical on all ranks.
  bool setUpFusedComm(ncclCommProperties_t* props) {
    skipReason_.clear();

    if (AnyRank(GinDisabledByEnv())) {
      skipReason_ = "GIN disabled by environment (NCCL_GIN_ENABLE=0)";
      return false;
    }

    // Errors are reduced for the same reason the skips are: a rank that returns
    // on its own leaves the others inside the next collective until the suite
    // times out, which buries the error that started it.
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

    // With a single LSA team every rank reaches every peer through P2P, and
    // hostRmaSupport is then true no matter what the RMA proxy reports. Only a
    // multi-team communicator makes the flag depend on the proxy at all.
    if (!AllRanks(props->nLsaTeams > 1)) {
      skipReason_ = "Needs ranks spread over more than one LSA team (run on >=2 nodes)";
      return false;
    }

    // ncclTopoCheckNicFused reports this rank's own device table; globalNicFused
    // in the capability derivation is the OR across ranks, so match that here.
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

// setUpFusedComm reports a precondition miss through skipReason_ and a genuine
// failure through ADD_FAILURE with skipReason_ empty; only the former skips.
#define RETURN_OR_SKIP()                                   \
  do {                                                     \
    if (!skipReason_.empty()) GTEST_SKIP() << skipReason_; \
    return;                                                \
  } while (0)

// The RMA proxy backend used to be switched off whenever any rank had fused
// NICs, which left multi-node GIN without its host RMA path. Nothing about
// merging two NICs into one vNIC makes the proxy unusable, so hostRmaSupport
// must read the same as it would on unfused NICs.
TEST_F(GinNicFusionMPITest, HostRmaSurvivesNicFusion) {
  ASSERT_MPI_TRUE(validateTestPrerequisites(2, kNoProcessLimit, kNoPowerOfTwoRequired, 2));

  ncclCommProperties_t props{};
  if (!setUpFusedComm(&props)) RETURN_OR_SKIP();

  EXPECT_TRUE(props.hostRmaSupport)
    << "NIC fusion is active and the communicator spans " << props.nLsaTeams
    << " LSA teams, so the host RMA proxy must still be available to GIN";
}

#endif  // MPI_TESTS_ENABLED
