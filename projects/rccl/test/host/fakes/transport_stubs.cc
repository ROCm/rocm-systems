/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fail-loud stub floor for the transport subsystem (transport/proxy/NVLS/
// CollNet/PXN), shared by host-only microtests. These satisfy a unit-under-
// test's link-time symbol closure; the shallower tests never call them
// (abort-on-call, except benign teardown returning ncclSuccess). A test that
// needs to drive one of these replaces that individual entry with a real fake.

#include "transport_stubs.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iterator>

#include "channel.h"
#include "fail_loud.h"
#include "graph.h"
#include "nccl.h"
#include "nccl_fakes.h"  // g_loadParam, for the NCCL_PARAM default this floor stands in for
#include "nvls_ub.h"
#include "signature-drift.h"
#include "transport.h"
#include "transport_reg_stubs.h"

ASSERT_HOOK_MATCHES_PROD(g_useAinic, rcclUseAinic);
ASSERT_HOOK_MATCHES_PROD(g_pxnDisable, ncclPxnDisable);
ASSERT_HOOK_MATCHES_PROD(g_ncclProxyStop, ncclProxyStop);
ASSERT_HOOK_MATCHES_PROD(g_ncclTransportP2pSetup, ncclTransportP2pSetup);
ASSERT_HOOK_MATCHES_PROD(g_ncclNetDeregBuffer, ncclNetDeregBuffer);
ASSERT_HOOK_MATCHES_PROD(g_ncclCollnetDeregBuffer, ncclCollnetDeregBuffer);
ASSERT_HOOK_MATCHES_PROD(g_ncclNvlsUbDeregister, ncclNvlsUbDeregister);
ASSERT_HOOK_MATCHES_PROD(g_ncclNetLocalRegisterBuffer, ncclNetLocalRegisterBuffer);
ASSERT_HOOK_MATCHES_PROD(g_ncclNetGraphRegisterBuffer, ncclNetGraphRegisterBuffer);
ASSERT_HOOK_MATCHES_PROD(g_ncclCollnetLocalRegisterBuffer, ncclCollnetLocalRegisterBuffer);
ASSERT_HOOK_MATCHES_PROD(g_ncclCollnetGraphRegisterBuffer, ncclCollnetGraphRegisterBuffer);
ASSERT_HOOK_MATCHES_PROD(g_ncclNvlsLocalRegisterBuffer, ncclNvlsLocalRegisterBuffer);
ASSERT_HOOK_MATCHES_PROD(g_ncclNvlsGraphRegisterBuffer, ncclNvlsGraphRegisterBuffer);
ASSERT_HOOK_MATCHES_PROD(g_ncclNvlsRegResourcesQuery, ncclNvlsRegResourcesQuery);
#undef ASSERT_HOOK_MATCHES_PROD

struct ncclComm;
struct ncclTopoGraph;

// src/transport/nvls.cc:159. Referenced by init.cc but not declared inside it, so the redirected
// NCCL_PARAM does not cover it.
int64_t ncclParamNvlsEnable() { return g_loadParam("NVLS_ENABLE", 2); }

// src/plugin/net.cc: initTransportsRank's GDR probe. Omitted for rccl-UnitTestsMicroPlugin, which compiles net.cc.
int g_gdrSupportValue = 0;
int g_gdrSupportCalls = 0;
#ifndef RCCL_TRANSPORT_STUBS_OMIT_ncclGpuGdrSupport
ncclResult_t ncclGpuGdrSupport(struct ncclComm*, int* gdrSupport) {
  ++g_gdrSupportCalls;
  if (gdrSupport) *gdrSupport = g_gdrSupportValue;
  return ncclSuccess;
}
#endif

// src/transport/net.cc:343. Was a fail-loud stub in nccl_stubs.cc, which forced
// the enqueue target to omit it via a macro and supply its own; a seam here
// serves both. `false` means "no AINIC", which is what a host-only binary with
// no device actually has, so no test is silently steered by the default.
bool g_rcclUseAinicValue = false;
static bool DefaultUseAinic() { return g_rcclUseAinicValue; }
std::function<bool()> g_useAinic = DefaultUseAinic;
bool rcclUseAinic() { return g_useAinic(); }

// src/transport/net.cc:212, read by net_ib_cast/common.cc.
int64_t ncclParamNetOptionalRecvCompletion() { return g_loadParam("NET_OPTIONAL_RECV_COMPLETION", 1); }

static int DefaultPxnDisable(struct ncclComm*) { return 0; }
std::function<int(struct ncclComm*)> g_pxnDisable = DefaultPxnDisable;
#ifndef RCCL_TRANSPORT_STUBS_OMIT_ncclPxnDisable
int ncclPxnDisable(struct ncclComm* comm) { return g_pxnDisable(comm); }
#endif

// src/proxy.cc: comm teardown stops the proxy through this, so it needs a working default, not a fail-loud one.
static ncclResult_t DefaultNcclProxyStop(struct ncclComm*) { return ncclSuccess; }
std::function<ncclResult_t(struct ncclComm*)> g_ncclProxyStop = DefaultNcclProxyStop;

static ncclResult_t DefaultTransportP2pSetup(struct ncclComm*, struct ncclTopoGraph*, int, bool*) {
  FailLoudUnfaked("transport_stubs", "ncclTransportP2pSetup");
}
std::function<ncclResult_t(struct ncclComm*, struct ncclTopoGraph*, int, bool*)>
    g_ncclTransportP2pSetup = DefaultTransportP2pSetup;

// src/transport.cc's table. ncclTopoComputePaths asks P2P, then SHM, whether a peer is reachable; both say yes.
static ncclResult_t ReachableCanConnect(int* ret, struct ncclComm*, struct ncclTopoGraph*, struct ncclPeerInfo*,
                                        struct ncclPeerInfo*) {
  *ret = 1;
  return ncclSuccess;
}
static struct ncclTransport reachableTransport = {"fake", ReachableCanConnect};
struct ncclTransport* ncclTransports[NTRANSPORTS] = {&reachableTransport, &reachableTransport};

static ncclResult_t DefaultNetDeregBuffer(struct ncclComm*, struct ncclProxyConnector*, void*) {
  FailLoudUnfaked("transport_stubs", "ncclNetDeregBuffer");
}
std::function<ncclResult_t(struct ncclComm*, struct ncclProxyConnector*, void*)> g_ncclNetDeregBuffer =
  DefaultNetDeregBuffer;
ncclResult_t ncclNetDeregBuffer(struct ncclComm* comm, struct ncclProxyConnector* proxyConn, void* handle) {
  return g_ncclNetDeregBuffer(comm, proxyConn, handle);
}

static ncclResult_t DefaultCollnetDeregBuffer(struct ncclComm*, struct ncclProxyConnector*, void*) {
  FailLoudUnfaked("transport_stubs", "ncclCollnetDeregBuffer");
}
std::function<ncclResult_t(struct ncclComm*, struct ncclProxyConnector*, void*)> g_ncclCollnetDeregBuffer =
  DefaultCollnetDeregBuffer;
ncclResult_t ncclCollnetDeregBuffer(struct ncclComm* comm, struct ncclProxyConnector* proxyconn, void* handle) {
  return g_ncclCollnetDeregBuffer(comm, proxyconn, handle);
}

static ncclResult_t DefaultNvlsUbDeregister(struct ncclComm*, struct ncclReg*) {
  FailLoudUnfaked("transport_stubs", "ncclNvlsUbDeregister");
}
std::function<ncclResult_t(struct ncclComm*, struct ncclReg*)> g_ncclNvlsUbDeregister = DefaultNvlsUbDeregister;
ncclResult_t ncclNvlsUbDeregister(struct ncclComm* comm, struct ncclReg* reg) {
  return g_ncclNvlsUbDeregister(comm, reg);
}

static ncclResult_t DefaultNetLocalRegisterBuffer(ncclComm*, const void*, size_t, struct ncclConnector**, int, int*,
                                                  void**) {
  FailLoudUnfaked("transport_stubs", "ncclNetLocalRegisterBuffer");
}
std::function<ncclResult_t(ncclComm*, const void*, size_t, struct ncclConnector**, int, int*, void**)>
  g_ncclNetLocalRegisterBuffer = DefaultNetLocalRegisterBuffer;
ncclResult_t ncclNetLocalRegisterBuffer(ncclComm* comm, const void* userbuff, size_t buffSize,
                                        struct ncclConnector** peerConns, int nPeers, int* outRegBufFlag,
                                        void** outHandle) {
  return g_ncclNetLocalRegisterBuffer(comm, userbuff, buffSize, peerConns, nPeers, outRegBufFlag, outHandle);
}

static ncclResult_t DefaultNetGraphRegisterBuffer(ncclComm*, const void*, size_t, struct ncclConnector**, int, int*,
                                                  void**, ncclRegCleanupQueue*, int*) {
  FailLoudUnfaked("transport_stubs", "ncclNetGraphRegisterBuffer");
}
std::function<ncclResult_t(ncclComm*, const void*, size_t, struct ncclConnector**, int, int*, void**,
                           ncclRegCleanupQueue*, int*)>
  g_ncclNetGraphRegisterBuffer = DefaultNetGraphRegisterBuffer;
ncclResult_t ncclNetGraphRegisterBuffer(ncclComm* comm, const void* userbuff, size_t buffSize,
                                        struct ncclConnector** peerConns, int nPeers, int* outRegBufFlag,
                                        void** outHandle, ncclRegCleanupQueue* cleanupQueue,
                                        int* nCleanupQueueElts) {
  return g_ncclNetGraphRegisterBuffer(comm, userbuff, buffSize, peerConns, nPeers, outRegBufFlag, outHandle,
                                      cleanupQueue, nCleanupQueueElts);
}

static ncclResult_t DefaultCollnetLocalRegisterBuffer(struct ncclComm*, const void*, size_t, int, int*, void**) {
  FailLoudUnfaked("transport_stubs", "ncclCollnetLocalRegisterBuffer");
}
std::function<ncclResult_t(struct ncclComm*, const void*, size_t, int, int*, void**)>
  g_ncclCollnetLocalRegisterBuffer = DefaultCollnetLocalRegisterBuffer;
ncclResult_t ncclCollnetLocalRegisterBuffer(struct ncclComm* comm, const void* userbuff, size_t buffSize, int type,
                                            int* outRegBufUsed, void** outHandle) {
  return g_ncclCollnetLocalRegisterBuffer(comm, userbuff, buffSize, type, outRegBufUsed, outHandle);
}

static ncclResult_t DefaultCollnetGraphRegisterBuffer(struct ncclComm*, const void*, size_t, int, int*, void**,
                                                      ncclRegCleanupQueue*, int*) {
  FailLoudUnfaked("transport_stubs", "ncclCollnetGraphRegisterBuffer");
}
std::function<ncclResult_t(struct ncclComm*, const void*, size_t, int, int*, void**, ncclRegCleanupQueue*, int*)>
  g_ncclCollnetGraphRegisterBuffer = DefaultCollnetGraphRegisterBuffer;
ncclResult_t ncclCollnetGraphRegisterBuffer(struct ncclComm* comm, const void* userbuff, size_t buffSize, int type,
                                            int* outRegBufFlag, void** outHandle, ncclRegCleanupQueue* cleanupQueue,
                                            int* nCleanupQueueElts) {
  return g_ncclCollnetGraphRegisterBuffer(comm, userbuff, buffSize, type, outRegBufFlag, outHandle, cleanupQueue,
                                          nCleanupQueueElts);
}

static ncclResult_t DefaultNvlsLocalRegisterBuffer(struct ncclComm*, const void*, void*, size_t, size_t, int*, void**,
                                                   void**) {
  FailLoudUnfaked("transport_stubs", "ncclNvlsLocalRegisterBuffer");
}
std::function<ncclResult_t(struct ncclComm*, const void*, void*, size_t, size_t, int*, void**, void**)>
  g_ncclNvlsLocalRegisterBuffer = DefaultNvlsLocalRegisterBuffer;
ncclResult_t ncclNvlsLocalRegisterBuffer(struct ncclComm* comm, const void* sendbuff, void* recvbuff,
                                         size_t sendbuffSize, size_t recvbuffSize, int* outRegBufUsed,
                                         void** outRegBufSend, void** outRegBufRecv) {
  return g_ncclNvlsLocalRegisterBuffer(comm, sendbuff, recvbuff, sendbuffSize, recvbuffSize, outRegBufUsed,
                                       outRegBufSend, outRegBufRecv);
}

static ncclResult_t DefaultNvlsGraphRegisterBuffer(struct ncclComm*, const void*, void*, size_t, size_t, int*, void**,
                                                   void**, ncclRegCleanupQueue*, int*) {
  FailLoudUnfaked("transport_stubs", "ncclNvlsGraphRegisterBuffer");
}
std::function<ncclResult_t(struct ncclComm*, const void*, void*, size_t, size_t, int*, void**, void**,
                           ncclRegCleanupQueue*, int*)>
  g_ncclNvlsGraphRegisterBuffer = DefaultNvlsGraphRegisterBuffer;
ncclResult_t ncclNvlsGraphRegisterBuffer(struct ncclComm* comm, const void* sendbuff, void* recvbuff,
                                         size_t sendbuffSize, size_t recvbuffSize, int* outRegBufUsed,
                                         void** outRegBufSend, void** outRegBufRecv,
                                         ncclRegCleanupQueue* cleanupQueue, int* nCleanupQueueElts) {
  return g_ncclNvlsGraphRegisterBuffer(comm, sendbuff, recvbuff, sendbuffSize, recvbuffSize, outRegBufUsed,
                                       outRegBufSend, outRegBufRecv, cleanupQueue, nCleanupQueueElts);
}

static ncclResult_t DefaultNvlsRegResourcesQuery(struct ncclComm*, ncclFunc_t, int*) {
  FailLoudUnfaked("transport_stubs", "ncclNvlsRegResourcesQuery");
}
std::function<ncclResult_t(struct ncclComm*, ncclFunc_t, int*)> g_ncclNvlsRegResourcesQuery =
  DefaultNvlsRegResourcesQuery;
ncclResult_t ncclNvlsRegResourcesQuery(struct ncclComm* comm, ncclFunc_t func, int* recChannels) {
  return g_ncclNvlsRegResourcesQuery(comm, func, recChannels);
}

void ResetTransportStubs() {
  g_rcclUseAinicValue = false;
  g_useAinic = DefaultUseAinic;
  g_pxnDisable = DefaultPxnDisable;
  g_ncclProxyStop = DefaultNcclProxyStop;
  g_gdrSupportValue = 0;
  g_gdrSupportCalls = 0;
  g_ncclNvlsInitResult = ncclSuccess;
  g_ncclNvlsInitCalls = 0;
  g_ncclNvlsTuningResult = ncclSuccess;
  g_ncclNvlsTuningCalls = 0;
  g_ncclTransportP2pSetup = DefaultTransportP2pSetup;
  g_ncclNetDeregBuffer = DefaultNetDeregBuffer;
  g_ncclCollnetDeregBuffer = DefaultCollnetDeregBuffer;
  g_ncclNvlsUbDeregister = DefaultNvlsUbDeregister;
  g_ncclNetLocalRegisterBuffer = DefaultNetLocalRegisterBuffer;
  g_ncclNetGraphRegisterBuffer = DefaultNetGraphRegisterBuffer;
  g_ncclCollnetLocalRegisterBuffer = DefaultCollnetLocalRegisterBuffer;
  g_ncclCollnetGraphRegisterBuffer = DefaultCollnetGraphRegisterBuffer;
  g_ncclNvlsLocalRegisterBuffer = DefaultNvlsLocalRegisterBuffer;
  g_ncclNvlsGraphRegisterBuffer = DefaultNvlsGraphRegisterBuffer;
  g_ncclNvlsRegResourcesQuery = DefaultNvlsRegResourcesQuery;
  std::fill(std::begin(ncclTransports), std::end(ncclTransports), &reachableTransport);
}

#ifndef RCCL_TRANSPORT_STUBS_OMIT_COLLECTIVE_FLOOR
ncclResult_t ncclCollNetChainBufferSetup(ncclComm_t comm) { ::abort(); }
ncclResult_t ncclCollNetDirectBufferSetup(ncclComm_t comm) { ::abort(); }
#endif
ncclResult_t ncclCollNetSetup(ncclComm_t comm, ncclComm_t parent, struct ncclTopoGraph* graphs[]) { ::abort(); }
// ncclGetUserP2pLevel (src/graph/paths.cc): topo_stubs.cc, with the rest of that TU's seams.
#ifndef RCCL_TRANSPORT_STUBS_OMIT_COLLECTIVE_FLOOR
ncclResult_t ncclNvlsBufferSetup(struct ncclComm* comm) { ::abort(); }
#endif
// Controllable (was fail-loud). :1618 uses bare NCCLCHECK, not NCCLCHECKGOTO, so a failure here returns
// WITHOUT running exit: -- the counter on ncclOsCpuCount is what makes that bypass observable.
ncclResult_t g_ncclNvlsInitResult = ncclSuccess;
int g_ncclNvlsInitCalls = 0;
ncclResult_t ncclNvlsInit(struct ncclComm* comm) {
  g_ncclNvlsInitCalls++;
  return g_ncclNvlsInitResult;
}
ncclResult_t ncclNvlsSetup(struct ncclComm* comm, struct ncclComm* parent) { ::abort(); }
#ifndef RCCL_TRANSPORT_STUBS_OMIT_COLLECTIVE_FLOOR
ncclResult_t ncclNvlsTreeConnect(struct ncclComm* comm) { ::abort(); }
#endif
// Controllable (was fail-loud). init.cc:2185, gated on comm->nvlsSupport surviving the :2182 fold.
ncclResult_t g_ncclNvlsTuningResult = ncclSuccess;
int g_ncclNvlsTuningCalls = 0;
ncclResult_t ncclNvlsTuning(struct ncclComm* comm) {
  g_ncclNvlsTuningCalls++;
  return g_ncclNvlsTuningResult;
}
ncclResult_t ncclProxyCreate(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclProxyDestroy(struct ncclComm* comm) { return ncclSuccess; }
ncclResult_t ncclProxyShmUnlink(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclProxyStop(struct ncclComm* comm) { return g_ncclProxyStop(comm); }
#ifndef RCCL_TRANSPORT_STUBS_OMIT_COLLECTIVE_FLOOR
ncclResult_t ncclTransportPatConnect(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclTransportRingConnect(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclTransportTreeConnect(struct ncclComm* comm) { ::abort(); }
#endif
ncclResult_t ncclTransportInitRankMap(struct ncclComm* comm, int, const int*) { return ncclSuccess; }
ncclResult_t ncclTuningInit(struct ncclComm* comm) { return ncclSuccess; }
// ncclTreeBasePostset (src/graph/connect.cc): topo_stubs.cc, next to ncclTopoPreset/ncclTopoPostset.
ncclResult_t ncclTransportCheckP2pType(struct ncclComm*, bool*, bool*, bool*) { ::abort(); }
ncclResult_t ncclTransportP2pConnect(struct ncclComm*, int, int, int*, int, int*, int) { ::abort(); }
ncclResult_t ncclTransportP2pSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, int connIndex,
                                   bool* needsProxy) {
  return g_ncclTransportP2pSetup(comm, graph, connIndex, needsProxy);
}

// ppc64le TOC references survive --gc-sections, so netTransport must have
// a definition even though the code path that uses it is dead.
#include "transport.h"
struct ncclTransport netTransport = {};

