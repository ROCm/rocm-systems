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

#include <cstdlib>

#include "nccl.h"

struct ncclComm;
struct ncclTopoGraph;

ncclResult_t ncclCollNetChainBufferSetup(ncclComm_t comm) { ::abort(); }
ncclResult_t ncclCollNetDirectBufferSetup(ncclComm_t comm) { ::abort(); }
ncclResult_t ncclCollNetSetup(ncclComm_t comm, ncclComm_t parent, struct ncclTopoGraph* graphs[]) { ::abort(); }
ncclResult_t ncclGetUserP2pLevel(int* level) { ::abort(); }
ncclResult_t ncclNvlsBufferSetup(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclNvlsInit(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclNvlsSetup(struct ncclComm* comm, struct ncclComm* parent) { ::abort(); }
ncclResult_t ncclNvlsTreeConnect(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclNvlsTuning(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclProxyCreate(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclProxyDestroy(struct ncclComm* comm) { return ncclSuccess; }
ncclResult_t ncclProxyShmUnlink(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclProxyStop(struct ncclComm* comm) { ::abort(); }
int ncclPxnDisable(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclTransportPatConnect(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclTransportRingConnect(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclTransportTreeConnect(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclTreeBasePostset(struct ncclComm* comm, struct ncclTopoGraph* treeGraph) { ::abort(); }
ncclResult_t ncclTransportCheckP2pType(struct ncclComm*, bool*, bool*, bool*) { ::abort(); }
ncclResult_t ncclTransportP2pConnect(struct ncclComm*, int, int, int*, int, int*, int) { ::abort(); }
ncclResult_t ncclTransportP2pSetup(struct ncclComm*, struct ncclTopoGraph*, int, bool*) { ::abort(); }
