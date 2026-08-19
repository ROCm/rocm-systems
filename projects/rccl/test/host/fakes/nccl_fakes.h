/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Reusable fakes for NCCL (`nccl*`) symbols that the micro-test binary
// links against instead of pulling in librccl.so.
//
// These fakes are not p2p-specific -- any micro-test that #includes a
// production TU referencing these `nccl*` functions can link against
// nccl_fakes.cc. p2p-specific fakes (the alloc-macro emulators, arch/topo
// helpers) live in p2p_fakes.{h,cc}; HIP runtime seams live in
// hip_fakes.{h,cc}.
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

// Restore every NCCL controllable seam in this header to its default.
// Called by ResetP2pFakes(); exposed for tests that only touch NCCL hooks.
void ResetNcclFakes();

#endif  // RCCL_TEST_HOST_NCCL_FAKES_H_
