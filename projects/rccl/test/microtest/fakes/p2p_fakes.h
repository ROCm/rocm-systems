/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Per-test controllable seams for the fakes layer.
//
// See README.md, "Adding more controllable seams". Tests install per-test
// behaviour by overwriting one of these std::function hooks in a fixture's
// SetUp(), and ResetP2pFakes() in TearDown() restores defaults so tests
// don't contaminate each other.

#pragma once

#include <cstddef>
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

// FakeCudaCallocAsync / FakeCudaMemcpyAsync: targets of the macro shims that
// p2p-test.cc installs over ncclCudaCallocAsync / ncclCudaMemcpyAsync so the
// header-only templates in alloc.h don't reach real HIP runtime. Default
// behaviour is what an honest GPU emulator would do: calloc heap memory and
// memcpy bytes between host pointers. Tests that need to inject failure or
// observe the calls override these hooks; ResetP2pFakes() frees any
// outstanding fake allocations.
extern std::function<ncclResult_t(void** ptr, std::size_t nbytes, hipStream_t)>
    g_fakeCudaCallocAsync;
extern std::function<ncclResult_t(void* dst, void* src, std::size_t nbytes, hipStream_t)>
    g_fakeCudaMemcpyAsync;

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

// hipMemGetAddressRange / hipIpcGetMemHandle: real HIP runtime entry points
// reached from ipcRegisterBuffer's fresh-registration arm. The microtest
// binary links hip::host, so these symbols resolve at link time -- but at
// runtime they need a real GPU. The macro shims in p2p-test.cc route the
// p2p.cc call sites through these hooks instead.
//
// Defaults return hipErrorInvalidValue so any test that *doesn't* opt in
// surfaces the unexpected call as ncclUnhandledCudaError via CUCHECKGOTO.
extern std::function<hipError_t(hipDeviceptr_t* /*pbase*/, std::size_t* /*psize*/,
                                hipDeviceptr_t /*dptr*/)>
    g_hipMemGetAddressRange;
extern std::function<hipError_t(hipIpcMemHandle_t* /*handle*/, void* /*devPtr*/)>
    g_hipIpcGetMemHandle;

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

// Restore every hook in this header to its default. Call from fixture
// TearDown().
void ResetP2pFakes();
