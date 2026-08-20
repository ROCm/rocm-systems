/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_NCCL_CUDA_FAKES_H_
#define RCCL_TEST_HOST_NCCL_CUDA_FAKES_H_

#include <cstddef>
#include <functional>

#include "nccl.h"
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

#include "nccl_fakes.h"
#include "hip_fakes.h"

extern std::function<ncclResult_t(void** ptr, std::size_t nbytes, hipStream_t)>
    g_fakeCudaCallocAsync;
extern std::function<ncclResult_t(void* dst, void* src, std::size_t nbytes, hipStream_t)>
    g_fakeCudaMemcpyAsync;

// Restore every hook owned by this file (and, transitively, the nccl* and
// HIP hooks) to its default. Call from fixture TearDown().
void ResetNcclCudaFakes();

// ===========================================================================
// Macro shims for the header-only ncclCuda* allocation templates.
//
// Pull in alloc.h NOW so its macros (ncclCudaCallocAsync etc.) are visible
// to be #undef'd and to avoid code under test from including alloc.h and it's
// original definitions.
//
// The shims replace the header-only function templates ncclCudaCallocAsync
// and ncclCudaMemcpyAsync from alloc.h with thin trampolines that route
// through the hookable fakes above.
// ===========================================================================
#include "alloc.h"

#undef ncclCudaCallocAsync
#undef ncclCudaMemcpyAsync
// Variadic: production's ncclCudaCallocAsync now takes trailing
// manager/memType args (e.g. comm->memManager). We ignore them here --
// the fake is type-erased to (void**, nbytes, stream) -- but the macro
// must still swallow the extra arguments so the call site expands.
#define ncclCudaCallocAsync(ptr, nelem, stream, ...) \
    g_fakeCudaCallocAsync(reinterpret_cast<void**>(ptr), \
                          (nelem) * sizeof(**(ptr)), (stream))
#define ncclCudaMemcpyAsync(dst, src, nelem, stream) \
    g_fakeCudaMemcpyAsync(reinterpret_cast<void*>(dst), \
                          reinterpret_cast<void*>(src), \
                          (nelem) * sizeof(*(dst)), (stream))

#endif  // RCCL_TEST_HOST_NCCL_CUDA_FAKES_H_
