/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seams whose definitions live in transport_stubs.cc. Declared in a header, not
// re-declared by each consumer, so a signature change is a compile error at
// every use rather than a silent link-time mismatch.
//

#ifndef RCCL_TEST_HOST_TRANSPORT_STUBS_H_
#define RCCL_TEST_HOST_TRANSPORT_STUBS_H_

#include <functional>

#include "nccl.h"

struct ncclChannel;
struct ncclComm;
struct ncclConnect;
struct ncclPeerInfo;
struct ncclProxyConnector;
struct ncclReg;
struct ncclTopoGraph;

extern std::function<ncclResult_t(struct ncclComm*, struct ncclTopoGraph*, int, bool*)>
    g_ncclTransportP2pSetup;

// src/transport.cc, and ncclTransportInitRankMap from src/transport/generic.cc. Fail-loud by default,
// except ncclTransportInitRankMap, which succeeds.
extern std::function<ncclResult_t(struct ncclComm*, int /*channelId*/, int /*nrecv*/, int* /*peerRecv*/,
                                  int /*nsend*/, int* /*peerSend*/, int /*connIndex*/)>
    g_ncclTransportP2pConnect;
extern std::function<ncclResult_t(struct ncclComm*, int /*nHeads*/, const int* /*heads*/)> g_ncclTransportInitRankMap;
extern std::function<bool(struct ncclComm*, struct ncclTopoGraph*, struct ncclChannel*, int /*masterRank*/,
                          int /*masterPeer*/, int /*collNetGraphChannelId*/, int /*type*/, struct ncclConnect*)>
    g_ncclTransportCollNetSetup;
extern std::function<ncclResult_t(struct ncclComm*, int /*collNetSetupFail*/)> g_ncclTransportCollNetCheck;
extern std::function<ncclResult_t(struct ncclComm*)> g_ncclTransportCollNetFree;

// rcclUseAinic (src/transport/net.cc:343) queries whether an AINIC is present.
// A host-only binary has no device, so `false` is the honest answer rather than
// a steering choice; override it to exercise the AINIC arm.
extern bool g_rcclUseAinicValue;
extern std::function<bool()> g_useAinic;

// ncclPxnDisable (src/graph/paths.cc:754). The default mirrors an initialized
// communicator with PXN enabled; override for direct-selection tests.
extern std::function<int(struct ncclComm*)> g_pxnDisable;

// ncclProxyStop (src/proxy.cc): comm teardown stops the proxy through this.
extern std::function<ncclResult_t(struct ncclComm*)> g_ncclProxyStop;

// ncclGpuGdrSupport (src/plugin/net.cc): initTransportsRank's GDR probe.
extern int g_gdrSupportValue;
extern int g_gdrSupportCalls;

// src/transport/nvls.cc. :1618 uses bare NCCLCHECK, not NCCLCHECKGOTO, so a failure in ncclNvlsInit
// returns WITHOUT running exit: -- the counter on ncclOsCpuCount is what makes that bypass observable.
extern ncclResult_t g_ncclNvlsInitResult;
extern int g_ncclNvlsInitCalls;
// init.cc:2185, gated on comm->nvlsSupport surviving the :2182 fold.
extern ncclResult_t g_ncclNvlsTuningResult;
extern int g_ncclNvlsTuningCalls;

// Buffer deregistration in src/transport/net.cc, coll_net.cc and nvls_ub.cc. Defaults fail loudly.
extern std::function<ncclResult_t(struct ncclComm*, struct ncclProxyConnector*, void*)> g_ncclNetDeregBuffer;
extern std::function<ncclResult_t(struct ncclComm*, struct ncclProxyConnector*, void*)> g_ncclCollnetDeregBuffer;
extern std::function<ncclResult_t(struct ncclComm*, struct ncclReg*)> g_ncclNvlsUbDeregister;

// canConnect of both ncclTransports slots (src/transport.cc). Default: every peer is reachable.
extern std::function<ncclResult_t(int*, struct ncclComm*, struct ncclTopoGraph*, struct ncclPeerInfo*,
                                  struct ncclPeerInfo*)>
    g_transportCanConnect;

void ResetTransportStubs();

#endif  // RCCL_TEST_HOST_TRANSPORT_STUBS_H_
