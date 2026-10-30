/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include "net_ib_cast_inspect.h"

#if defined(MPI_TESTS_ENABLED)

// =============================================================================
// Test: CastGinErrorPath
//
// White-box, no NIC. Covers the four net_ib error-path fixes ported into
// net_ib_cast:
//   - IGet has a name in IbCastReqTypeStr (p2p.cc)
//   - an error CQE that is not the first in the poll uses devBase, not
//     devBases[completion] (gin.cc IbCastRmaIbProxyTest)
//   - wr_id == NET_IB_MAX_REQUESTS is outside the receiver range
//     (p2p_resiliency.cc IbCastResiliencyHandleCompletionErrorReceiver)
//   - GIN allgather polls the receive while isend is still waiting for CTS
//     (gin.cc IbCastGinIbAllGather)
// Requires: no setup.
// =============================================================================
TEST(NetIbCast, CastGinErrorPath) {
  EXPECT_STREQ(ncclIbCastTestGinIgetName(), "IGet");
  EXPECT_EQ(ncclIbCastTestRmaProxyErrorCompletion(), ncclRemoteError);
  EXPECT_EQ(ncclIbCastTestRecvWrIdBound(), ncclInternalError);

  int recvPolledBeforeSend = 0;
  EXPECT_EQ(ncclIbCastTestGinAllGatherPollsRecv(&recvPolledBeforeSend), ncclSuccess);
  EXPECT_EQ(recvPolledBeforeSend, 1);
}

#endif  // MPI_TESTS_ENABLED
