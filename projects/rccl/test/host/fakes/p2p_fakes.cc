/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Minimal stubs for the p2p-specific symbols p2p.cc references but doesn't define
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "nccl.h"
#include "alloc.h"        // allocationTracker, ncclCuMemEnable
#include "rocmwrap.h"     // ncclCuMemHandleType
#include "archinfo.h"     // IsArchMatch
#include "utils.h"        // busIdToInt64
#include "graph.h"        // getBusId

#include "nccl_fakes.h"    // reusable nccl* fakes + their reset
#include "p2p_fakes.h"     // controllable seam hooks
#include "hip_fakes.h"     // ResetHipFakes

#include <type_traits>

// ---------------------------------------------------------------------------
// Signature-drift watchdog for the HIP seams: the nccl* asserts live in
// nccl_fakes.cc; here anchor each HIP hook to its production declaration
// (templates + macro live in fakes/signature-drift.h).
#include "signature-drift.h"

ASSERT_HOOK_MATCHES_PROD(g_hipMemGetAddressRange,     hipMemGetAddressRange);
ASSERT_HOOK_MATCHES_PROD(g_hipIpcGetMemHandle,        hipIpcGetMemHandle);
ASSERT_HOOK_MATCHES_PROD(g_hipMemRetainAllocationHandle,  hipMemRetainAllocationHandle);
ASSERT_HOOK_MATCHES_PROD(g_hipMemExportToShareableHandle, hipMemExportToShareableHandle);
ASSERT_HOOK_MATCHES_PROD(g_hipMemRelease,             hipMemRelease);
ASSERT_HOOK_MATCHES_PROD(g_hipPointerGetAttribute,    hipPointerGetAttribute);

#undef ASSERT_HOOK_MATCHES_PROD

// ---------------------------------------------------------------------------
// Trivial globals
// ---------------------------------------------------------------------------

// allocTracker is an array of per-device counters in alloc.h; size it to
// the same MAX_ALLOC_TRACK_NGPU the header uses. Zero-initialised.
struct allocationTracker allocTracker[32 /* MAX_ALLOC_TRACK_NGPU */] = {};

// ---------------------------------------------------------------------------
// Arch / topology / busId helpers
// ---------------------------------------------------------------------------

// IsArchMatch is NOT faked here: wrap-test.cc (also part of this binary,
// since rccl_wrap.cc's tests need the real prefix-match behaviour for
// rcclIsArchSupportedForFunc et al.) compiles the real archinfo.cc, which
// defines it. p2p.cc's own calls (transport/p2p.cc's IsArchMatch(gcn, ...)
// checks) are unaffected: no p2p-test.cc fixture ever sets
// comm->topo->nodes[GPU].nodes[0].gpu.gcn, so it stays at its zero-init
// empty string, which the real strncmp-based match also reads as "no match"
// -- identical to this file's former hardcoded-false stub.

ncclResult_t busIdToInt64(const char* /*busId*/, int64_t* id)
{
    if (id) *id = 0;
    return ncclSuccess;
}

ncclResult_t getBusId(int /*cudaDev*/, int64_t* busId)
{
    if (busId) *busId = 0;
    return ncclSuccess;
}

// ---------------------------------------------------------------------------
// Controllable seams: ncclCudaCallocAsync / ncclCudaMemcpyAsync
//
// Substitutes for the header-only function templates in alloc.h. The shim
// macros in p2p-test.cc route the ncclCudaCallocAsync / ncclCudaMemcpyAsync
// macros through these, type-erased to (void*, nbytes), so the test binary
// never reaches real HIP runtime.
//
// Defaults behave like an honest emulator: heap-allocate zeroed memory and
// memcpy bytes between host pointers. ResetP2pFakes() frees any allocations
// the default hook handed out so individual tests don't have to. Tests that
// install their own hook also take responsibility for any memory they hand
// out.
// ---------------------------------------------------------------------------
static std::vector<void*> g_fakeAllocations;

static ncclResult_t DefaultFakeCudaCallocAsync(void** ptr, std::size_t nbytes,
                                               hipStream_t /*stream*/)
{
    if (ptr == nullptr) return ncclInvalidArgument;
    void* p = std::calloc(1, nbytes);
    if (p == nullptr && nbytes > 0) return ncclSystemError;
    g_fakeAllocations.push_back(p);
    *ptr = p;
    return ncclSuccess;
}

static ncclResult_t DefaultFakeCudaMemcpyAsync(void* dst, void* src,
                                               std::size_t nbytes,
                                               hipStream_t /*stream*/)
{
    if (nbytes > 0 && (dst == nullptr || src == nullptr)) return ncclInvalidArgument;
    if (nbytes > 0) std::memcpy(dst, src, nbytes);
    return ncclSuccess;
}

std::function<ncclResult_t(void**, std::size_t, hipStream_t)>
    g_fakeCudaCallocAsync = DefaultFakeCudaCallocAsync;
std::function<ncclResult_t(void*, void*, std::size_t, hipStream_t)>
    g_fakeCudaMemcpyAsync = DefaultFakeCudaMemcpyAsync;

void ResetP2pFakes()
{
    g_fakeCudaCallocAsync    = DefaultFakeCudaCallocAsync;
    g_fakeCudaMemcpyAsync    = DefaultFakeCudaMemcpyAsync;
    ResetNcclFakes();  // restore the nccl* hooks owned by nccl_fakes.cc
    ResetHipFakes();   // restore the HIP hooks owned by hip_fakes.cc
    for (void* p : g_fakeAllocations) std::free(p);
    g_fakeAllocations.clear();
}
