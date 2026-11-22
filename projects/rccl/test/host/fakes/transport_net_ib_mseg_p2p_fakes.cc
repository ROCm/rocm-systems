/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Symbols owned by src/transport/net_ib_multiseg/p2p.cc.

#include <cstdint>

#include "fail_loud.h"
#include "fakes/nccl_fakes.h"
#include "transport_net_ib_mseg_p2p_fakes.h"
#include "transport/net_ib_multiseg/common_mseg.h"

int64_t ncclParamIbArThreshold() { return g_loadParam("IB_AR_THRESHOLD", -2); }
int64_t IbMsegArThreshold = 8192;
int64_t ncclParamIbReceiverSideMatchingScheme() {
  return g_loadParam("IB_RECEIVER_SIDE_MATCHING_SCHEME", -2);
}
int64_t rcclParamIbGdrFlushGpuMemNoRelaxedOrdering() {
  return g_loadParam("RCCL_GDR_FLUSH_GPU_MEM_NO_RELAXED_ORDERING", 1);
}

ncclResult_t IbMsegIsend(void*, void*, size_t, int, void*, void*, void**) {
  FailLoudUnfaked("transport_net_ib_mseg_p2p_fakes", "IbMsegIsend");
}
ncclResult_t IbMsegIrecv(void*, int, void**, size_t*, int*, void**, void**, void**) {
  FailLoudUnfaked("transport_net_ib_mseg_p2p_fakes", "IbMsegIrecv");
}
ncclResult_t IbMsegIflush(void*, int, void**, int*, void**, void**) {
  FailLoudUnfaked("transport_net_ib_mseg_p2p_fakes", "IbMsegIflush");
}
ncclResult_t IbMsegTest(void*, int*, int*) {
  FailLoudUnfaked("transport_net_ib_mseg_p2p_fakes", "IbMsegTest");
}

// Owned by src/transport/net_ib_multiseg/p2p_multiseg.cc; IbMsegCloseSend frees the sender's layout cache.
ncclResult_t rcclIbSendLayoutCacheFree(struct IbMsegSendComm*) { return ncclSuccess; }

void ResetTransportNetIbP2pFakes() { IbMsegArThreshold = 8192; }
