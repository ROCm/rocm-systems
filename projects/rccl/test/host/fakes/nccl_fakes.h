/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Reusable fakes for NCCL (`nccl*`) symbols that the micro-test binary
// links against instead of pulling in librccl.so.
//
// Several of the symbols below are "controllable seams": a std::function
// hook whose default preserves the historical stub behaviour, plus a thin
// `nccl*` wrapper that dispatches through the hook. Tests install per-test
// behaviour by overwriting a hook in a fixture's SetUp() and
// ResetNcclFakes() (called from ResetP2pFakes()) restores the defaults so
// tests don't contaminate each other.

#ifndef RCCL_TEST_HOST_NCCL_FAKES_H_
#define RCCL_TEST_HOST_NCCL_FAKES_H_

#include <cstddef>
#include <cstdint>
#include <functional>

#include "nccl.h"
#include "strongstream.h"
#include "proxy.h"
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

struct ncclReg;

// ncclStrongStreamAcquire: by default returns ncclSuccess with *stream=nullptr
// (matching the stub's prior behaviour). Tests that need to exercise the
// strong-stream block's failure paths can install a hook that returns an
// error code; tests that want to count entries can install a counting hook.
extern std::function<ncclResult_t(struct ncclCudaGraph,
                                  struct ncclStrongStream*,
                                  bool,
                                  hipStream_t*)>
    g_strongStreamAcquire;

// ncclProxyConnect / ncclProxyCallBlocking: fresh-registration arm of
// ipcRegisterBuffer routes the per-peer IPC handshake through these. The
// default Connect returns ncclSystemError (so tests that don't expect to
// reach the proxy fail loudly); the default CallBlocking also returns
// ncclSystemError. Happy-path tests install a hook that returns success
// and writes a canned rmtRegAddr into respBuff for ncclProxyMsgRegister.
extern std::function<ncclResult_t(struct ncclComm*, int /*transport*/,
                                  int /*send*/, int /*proxyRank*/,
                                  struct ncclProxyConnector*)>
    g_proxyConnect;
extern std::function<ncclResult_t(struct ncclComm*, struct ncclProxyConnector*,
                                  int /*type*/,
                                  void* /*reqBuff*/, int /*reqSize*/,
                                  void* /*respBuff*/, int /*respSize*/)>
    g_proxyCallBlocking;

// ncclCuMemEnable: gates the cuMem*-export arm of ipcRegisterBuffer
// against the legacy-IPC arm. Default returns 0 so existing tests stay
// on the legacy arm. Tests for the cuMem* arm install a hook returning 1.
extern std::function<int()> g_cuMemEnable;

// ncclProxyClientQueryFdBlocking: the cuMem*-export POSIX_FD arm of
// ipcRegisterBuffer calls this to register the exported fd with the
// remote proxy and receive back an imported fd handle. Default returns
// ncclSystemError so unexpected call sites fail loudly; happy-path tests
// install a hook that succeeds and writes a canned imported-fd value.
extern std::function<ncclResult_t(struct ncclComm*,
                                  struct ncclProxyConnector*,
                                  int /*localFd*/, int* /*rmtFd*/)>
    g_proxyClientQueryFdBlocking;

// ncclProxyClientBatchQueryFdBlocking: the POSIX_FD, cross-process arm of
// ipcHandleMultiSegmentRegistration ships every exported segment fd to the
// remote proxy and receives an imported fd per segment. Default returns
// ncclSystemError so unexpected calls fail loudly; tests driving that arm
// install a hook that succeeds and fills the imported-fd array.
extern std::function<ncclResult_t(struct ncclComm*,
                                  struct ncclProxyConnector*,
                                  int* /*localFds*/, int* /*rmtFds*/,
                                  int /*numSegments*/)>
    g_proxyClientBatchQueryFdBlocking;

// ncclDynMemMarkExportToPeer / ncclMemTrackImportFromPeer: the memory-manager
// bookkeeping the cuMem arms of ncclP2pAllocateShareableBuffer /
// ncclP2pImportShareableBuffer drive. Neither leaves observable public state
// here, so the shareable-buffer tests assert mock-style that the arm reached
// (or skipped) them. Defaults succeed.
extern std::function<ncclResult_t(struct ncclMemManager*, void* /*ptr*/,
                                  int /*peerRank*/)>
    g_dynMemMarkExportToPeer;
extern std::function<ncclResult_t(struct ncclMemManager*, void* /*ptr*/,
                                  size_t /*size*/,
                                  hipMemGenericAllocationHandle_t /*handle*/,
                                  hipMemAllocationHandleType /*handleType*/,
                                  ncclMemType_t /*memType*/, int /*ownerRank*/,
                                  int /*ownerDev*/, void* /*ownerPtr*/)>
    g_memTrackImportFromPeer;

// NCCL_PARAM redirector: p2p-test.cc replaces the body of every
// NCCL_PARAM(name, env, deftVal) generator in the #included p2p.cc with a
// thin trampoline that calls g_loadParam(env, deftVal) on every invocation
// (no caching, unlike the real NCCL_PARAM). Default returns deftVal so
// callers see their compile-time defaults. Tests that need to flip a
// specific param (e.g. force ncclParamLegacyCudaRegister() == 1 to enter
// the legacy-export arm) install a hook that dispatches on the env string.
//
// Because the redirection happens at macro-expansion time, this only
// affects NCCL_PARAM bodies inside the #included p2p.cc -- not any
// already-compiled TUs.
extern std::function<int64_t(const char* /*env*/, int64_t /*deftVal*/)>
    g_loadParam;

// The two device ids and the isXGMI out-parameter; the topo system and the inter-GPU knobs are dropped.
// maxInter is carried because init.cc:1834 passes an explicit 1 where graph.h:90 defaults to MAX_XGMI_INTER_GPUS.
extern std::function<ncclResult_t(int /*cudaDev1*/, int /*cudaDev2*/, bool* /*isXGMI*/, int /*maxInter*/)>
    g_ncclTopoGetLinkType;
extern int g_ncclTopoGetLinkTypeCalls;

// Controllable seams for the topology eligibility checks p2pCanConnect drives.
// (rank1, rank2, p2p, read, intermediateRank, cudaP2p, isCrossClique) -- the
// topo system pointer and comm are dropped; only the rank pair and out-params
// matter. isCrossClique is defaulted to 0 by the wrapper before the hook runs,
// so a hook that ignores it keeps the common case; a hook that sets it drives
// the cross-clique arm that skips the ncclTopoCheckNet block (p2p.cc).
extern std::function<ncclResult_t(int /*rank1*/, int /*rank2*/, int* /*p2p*/,
                                  int* /*read*/, int* /*intermediateRank*/,
                                  int* /*cudaP2p*/, int* /*isCrossClique*/)>
    g_ncclTopoCheckP2p;
extern std::function<ncclResult_t(int /*rank1*/, int /*rank2*/, int* /*net*/)>
    g_ncclTopoCheckNet;

// ncclRegLocalIsValid / ncclCommGraphRegister / ncclCommGraphDeregister:
// the register-family wrappers (ncclIpcLocalRegisterBuffer /
// ncclIpcGraphRegisterBuffer / cleanupIpc) drive these. Defaults preserve
// the old stubs (isValid=false, graph-register fails, graph-deregister
// succeeds); tests install hooks to reach the delegate/enqueue/cleanup arms.
extern std::function<ncclResult_t(struct ncclReg*, bool* /*isValid*/)>
    g_regLocalIsValid;
extern std::function<ncclResult_t(struct ncclComm*, void* /*buff*/,
                                  size_t /*size*/, void** /*handle*/)>
    g_commGraphRegister;
extern std::function<ncclResult_t(struct ncclComm*, struct ncclReg* /*reg*/)>
    g_commGraphDeregister;

// ncclShmAllocateShareableBuffer: the CE-memcpy arm of p2pSendProxySetup
// allocates its peer SHM segment through this. Default fails so unexpected
// call sites surface loudly; the CE proxy-setup test installs a hook that
// succeeds and hands back backing storage for the host/device SHM pointers.
extern std::function<ncclResult_t(size_t /*size*/, bool /*legacy*/,
                                  void* /*desc*/,
                                  void** /*hptr*/, void** /*dptr*/)>
    g_shmAllocateShareableBuffer;

// Restore every NCCL controllable seam in this header to its default.
// Called by ResetP2pFakes(); exposed for tests that only touch NCCL hooks.
// Hands back an fd the caller must close. Defaults to ncclSystemError (the
// historical fixed return); the dev_runtime suite installs one that opens
// /dev/null so its symmetric-memory export path can run.
extern std::function<ncclResult_t(struct ncclComm*, int, void*, int*)>
    g_ncclProxyClientGetFdBlocking;

void ResetNcclFakes();

#endif  // RCCL_TEST_HOST_NCCL_FAKES_H_
