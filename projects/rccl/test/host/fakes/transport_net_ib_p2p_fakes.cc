/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Symbols owned by src/transport/net_ib/p2p.cc.

#include <cstdint>

#include "fail_loud.h"
#include "fakes/nccl_fakes.h"
#include "transport_net_ib_p2p_fakes.h"
#include "transport/net_ib/common.h"

int64_t ncclParamIbArThreshold() { return g_loadParam("IB_AR_THRESHOLD", -2); }
int64_t ncclIbArThreshold = 8192;
int64_t ncclParamIbReceiverSideMatchingScheme() {
  return g_loadParam("IB_RECEIVER_SIDE_MATCHING_SCHEME", -2);
}

ncclResult_t ncclIbIsend(void*, void*, size_t, int, void*, void*, void**) {
  FailLoudUnfaked("transport_net_ib_p2p_fakes", "ncclIbIsend");
}
ncclResult_t ncclIbIrecv(void*, int, void**, size_t*, int*, void**, void**, void**) {
  FailLoudUnfaked("transport_net_ib_p2p_fakes", "ncclIbIrecv");
}
ncclResult_t ncclIbIflush(void*, int, void**, int*, void**, void**) {
  FailLoudUnfaked("transport_net_ib_p2p_fakes", "ncclIbIflush");
}
ncclResult_t ncclIbTest(void*, int*, int*) {
  FailLoudUnfaked("transport_net_ib_p2p_fakes", "ncclIbTest");
}

void ResetTransportNetIbP2pFakes() { ncclIbArThreshold = 8192; }
