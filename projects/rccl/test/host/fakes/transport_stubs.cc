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

#include <cstdint>
#include <cstdlib>
#include <functional>

#include "nccl.h"
#include "nccl_fakes.h"  // g_loadParam, for the NCCL_PARAM default this floor stands in for

struct ncclComm;
struct ncclTopoGraph;

// src/transport/nvls.cc:159. Referenced by init.cc but not declared inside it, so the redirected
// NCCL_PARAM does not cover it.
int64_t ncclParamNvlsEnable() { return g_loadParam("NVLS_ENABLE", 2); }

// src/plugin/net.cc: initTransportsRank's GDR probe.
int g_gdrSupportValue = 0;
int g_gdrSupportCalls = 0;
ncclResult_t ncclGpuGdrSupport(struct ncclComm*, int* gdrSupport) {
  ++g_gdrSupportCalls;
  if (gdrSupport) *gdrSupport = g_gdrSupportValue;
  return ncclSuccess;
}

// src/transport/net.cc:343. Was a fail-loud stub in nccl_stubs.cc, which forced
// the enqueue target to omit it via a macro and supply its own; a seam here
// serves both. `false` means "no AINIC", which is what a host-only binary with
// no device actually has, so no test is silently steered by the default.
bool g_rcclUseAinic = false;
bool rcclUseAinic() { return g_rcclUseAinic; }

// src/proxy.cc: comm teardown stops the proxy through this, so it needs a working default, not a fail-loud one.
static ncclResult_t DefaultNcclProxyStop(struct ncclComm*) { return ncclSuccess; }
std::function<ncclResult_t(struct ncclComm*)> g_ncclProxyStop = DefaultNcclProxyStop;

void ResetTransportStubs() {
  g_rcclUseAinic = false;
  g_ncclProxyStop = DefaultNcclProxyStop;
  g_gdrSupportValue = 0;
  g_gdrSupportCalls = 0;
  g_ncclNvlsInitResult = ncclSuccess;
  g_ncclNvlsInitCalls = 0;
  g_ncclNvlsTuningResult = ncclSuccess;
  g_ncclNvlsTuningCalls = 0;
}

ncclResult_t ncclCollNetChainBufferSetup(ncclComm_t comm) { ::abort(); }
ncclResult_t ncclCollNetDirectBufferSetup(ncclComm_t comm) { ::abort(); }
ncclResult_t ncclCollNetSetup(ncclComm_t comm, ncclComm_t parent, struct ncclTopoGraph* graphs[]) { ::abort(); }
// ncclGetUserP2pLevel (src/graph/paths.cc): topo_stubs.cc, with the rest of that TU's seams.
ncclResult_t ncclNvlsBufferSetup(struct ncclComm* comm) { ::abort(); }
// Controllable (was fail-loud). :1618 uses bare NCCLCHECK, not NCCLCHECKGOTO, so a failure here returns
// WITHOUT running exit: -- the counter on ncclOsCpuCount is what makes that bypass observable.
ncclResult_t g_ncclNvlsInitResult = ncclSuccess;
int g_ncclNvlsInitCalls = 0;
ncclResult_t ncclNvlsInit(struct ncclComm* comm) {
  g_ncclNvlsInitCalls++;
  return g_ncclNvlsInitResult;
}
ncclResult_t ncclNvlsSetup(struct ncclComm* comm, struct ncclComm* parent) { ::abort(); }
ncclResult_t ncclNvlsTreeConnect(struct ncclComm* comm) { ::abort(); }
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
int ncclPxnDisable(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclTransportPatConnect(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclTransportRingConnect(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclTransportTreeConnect(struct ncclComm* comm) { ::abort(); }
// ncclTreeBasePostset (src/graph/connect.cc): topo_stubs.cc, next to ncclTopoPreset/ncclTopoPostset.
ncclResult_t ncclTransportCheckP2pType(struct ncclComm*, bool*, bool*, bool*) { ::abort(); }
ncclResult_t ncclTransportP2pConnect(struct ncclComm*, int, int, int*, int, int*, int) { ::abort(); }
ncclResult_t ncclTransportP2pSetup(struct ncclComm*, struct ncclTopoGraph*, int, bool*) { ::abort(); }
