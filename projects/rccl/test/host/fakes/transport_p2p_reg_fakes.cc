/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See transport_p2p_reg_fakes.h.

#include "transport_p2p_reg_fakes.h"

#include "fail_loud.h"
#include "signature-drift.h"
#include "transport.h"

ASSERT_HOOK_MATCHES_PROD(g_ncclIpcLocalRegisterBuffer, ncclIpcLocalRegisterBuffer);
ASSERT_HOOK_MATCHES_PROD(g_ncclIpcGraphRegisterBuffer, ncclIpcGraphRegisterBuffer);
ASSERT_HOOK_MATCHES_PROD(g_ncclIpcDeregBuffer, ncclIpcDeregBuffer);
ASSERT_HOOK_MATCHES_PROD(g_ncclP2pUsesMemcpy, ncclP2pUsesMemcpy);
#undef ASSERT_HOOK_MATCHES_PROD

static ncclResult_t DefaultIpcLocalRegisterBuffer(ncclComm*, const void*, size_t, int*, int, ncclIpcRegType, int*,
                                                  uintptr_t*, uintptr_t**) {
  FailLoudUnfaked("transport_p2p_reg_fakes", "ncclIpcLocalRegisterBuffer");
}
std::function<ncclResult_t(ncclComm*, const void*, size_t, int*, int, ncclIpcRegType, int*, uintptr_t*, uintptr_t**)>
  g_ncclIpcLocalRegisterBuffer = DefaultIpcLocalRegisterBuffer;

static ncclResult_t DefaultIpcGraphRegisterBuffer(ncclComm*, const void*, size_t, int*, int, ncclIpcRegType, int*,
                                                  uintptr_t*, uintptr_t**, void*, int*) {
  FailLoudUnfaked("transport_p2p_reg_fakes", "ncclIpcGraphRegisterBuffer");
}
std::function<ncclResult_t(ncclComm*, const void*, size_t, int*, int, ncclIpcRegType, int*, uintptr_t*, uintptr_t**,
                           void*, int*)>
  g_ncclIpcGraphRegisterBuffer = DefaultIpcGraphRegisterBuffer;

static ncclResult_t DefaultIpcDeregBuffer(struct ncclComm*, struct ncclIpcRegInfo*) {
  FailLoudUnfaked("transport_p2p_reg_fakes", "ncclIpcDeregBuffer");
}
std::function<ncclResult_t(struct ncclComm*, struct ncclIpcRegInfo*)> g_ncclIpcDeregBuffer = DefaultIpcDeregBuffer;

static bool DefaultP2pUsesMemcpy() { return false; }
std::function<bool()> g_ncclP2pUsesMemcpy = DefaultP2pUsesMemcpy;

void ResetTransportP2pRegFakes() {
  g_ncclIpcLocalRegisterBuffer = DefaultIpcLocalRegisterBuffer;
  g_ncclIpcGraphRegisterBuffer = DefaultIpcGraphRegisterBuffer;
  g_ncclIpcDeregBuffer = DefaultIpcDeregBuffer;
  g_ncclP2pUsesMemcpy = DefaultP2pUsesMemcpy;
}

ncclResult_t micro_ncclIpcLocalRegisterBuffer(ncclComm* comm, const void* userbuff, size_t buffSize, int* peerRanks,
                                              int nPeers, ncclIpcRegType type, int* regBufFlag, uintptr_t* offsetOut,
                                              uintptr_t** peerRmtAddrsOut) {
  return g_ncclIpcLocalRegisterBuffer(comm, userbuff, buffSize, peerRanks, nPeers, type, regBufFlag, offsetOut,
                                      peerRmtAddrsOut);
}
ncclResult_t micro_ncclIpcGraphRegisterBuffer(ncclComm* comm, const void* userbuff, size_t buffSize, int* peerRanks,
                                              int nPeers, ncclIpcRegType type, int* regBufFlag, uintptr_t* offsetOut,
                                              uintptr_t** peerRmtAddrsOut, void* cleanupQueuePtr,
                                              int* nCleanupQueueElts) {
  return g_ncclIpcGraphRegisterBuffer(comm, userbuff, buffSize, peerRanks, nPeers, type, regBufFlag, offsetOut,
                                      peerRmtAddrsOut, cleanupQueuePtr, nCleanupQueueElts);
}
ncclResult_t micro_ncclIpcDeregBuffer(struct ncclComm* comm, struct ncclIpcRegInfo* regInfo) {
  return g_ncclIpcDeregBuffer(comm, regInfo);
}
bool micro_ncclP2pUsesMemcpy() { return g_ncclP2pUsesMemcpy(); }
