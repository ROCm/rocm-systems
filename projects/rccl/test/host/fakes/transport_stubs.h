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

struct ncclComm;

// rcclUseAinic (src/transport/net.cc:343) queries whether an AINIC is present.
// A host-only binary has no device, so `false` is the honest answer rather than
// a steering choice; override it to exercise the AINIC arm.
extern bool g_rcclUseAinic;

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

void ResetTransportStubs();

#endif  // RCCL_TEST_HOST_TRANSPORT_STUBS_H_
