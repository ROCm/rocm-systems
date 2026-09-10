/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Seams whose definitions live in transport_stubs.cc. Declared in a header, not
// re-declared by each consumer, so a signature change is a compile error at
// every use rather than a silent link-time mismatch.
//
// SCOPE: this rule covers what transport_stubs.cc OWNS. It does not retroactively
// cover the seams it merely CONSUMES -- g_ncclGetUserP2pLevel,
// g_ncclNvlsInitResult and g_ncclNvlsInitCalls are defined in init_fakes.cc and
// still hand-declared extern below, as they are in topo_stubs.cc, nccl_stubs.cc
// and bootstrap_stubs.cc. Giving those a header means moving them to a fakes
// file named after their owning production TU, which is a separate change.

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

void ResetTransportStubs();

#endif  // RCCL_TEST_HOST_TRANSPORT_STUBS_H_
