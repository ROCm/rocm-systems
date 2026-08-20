/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "nccl.h"
#include "alloc.h"        // ncclCuMemEnable
#include "rocmwrap.h"     // ncclCuMemHandleType

#include "nccl_fakes.h"    // reusable nccl* fakes + their reset
#include "nccl_cuda_fakes.h" // controllable seam hooks
#include "hip_fakes.h"     // ResetHipFakes

#include <type_traits>

// ---------------------------------------------------------------------------
// Controllable seams: ncclCudaCallocAsync / ncclCudaMemcpyAsync
//
// Substitutes for the header-only function templates in alloc.h. The shim
// macros in p2p-test.cc route the ncclCudaCallocAsync / ncclCudaMemcpyAsync
// macros through these, type-erased to (void*, nbytes), so the test binary
// never reaches real HIP runtime.
//
// Defaults behave like an honest emulator: heap-allocate zeroed memory and
// memcpy bytes between host pointers. ResetNcclCudaFakes() frees any allocations
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

void ResetNcclCudaFakes()
{
    g_fakeCudaCallocAsync    = DefaultFakeCudaCallocAsync;
    g_fakeCudaMemcpyAsync    = DefaultFakeCudaMemcpyAsync;
    for (void* p : g_fakeAllocations) std::free(p);
    g_fakeAllocations.clear();
}
