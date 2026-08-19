/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * DevRuntimeTests-specific host stubs.
 *
 * dev_runtime.cc is #included whole into DevRuntimeTests.cpp, which leaves
 * undefined references to everything the translation unit calls but does not
 * define. Most of those dependencies are satisfied by the shared fakes layer
 * this target now links:
 *   - generic nccl / bootstrap / group symbols -> test/host/fakes/nccl_fakes.cc
 *   - devr / gin / rma / team / shadow-pool     -> test/host/fakes/devr_fakes.cc
 * Only the pieces that remain genuinely specific to this binary live here: the
 * thread-local group globals, and the host-memory-backed HIP VMM / stream fakes.
 * Real headers are included so every signature is checked against the real
 * declaration rather than hand-transcribed.
 *************************************************************************/

#include "comm.h"
#include "group.h"
#include "cudawrap.h"

#include <cassert>
#include <cstdarg>
#include <cstdlib>
#include <sys/mman.h>

// ---------------------------------------------------------------------------
// Globals the translation unit references.
//
// ncclDebugLevel/Mask/NoWarn, ncclCuMemHandleType, ncclDebugLog,
// ncclLoadParam and ncclProxyClientGetFdBlocking are shared with (and defined
// identically by) test/host/fakes/nccl_fakes.cc, which this target now links.
// They are intentionally NOT redefined here to avoid duplicate symbols.
// nccl_fakes.cc also uses POSIX-FD handles for ncclCuMemHandleType, so the
// single-rank success path still takes the no-export / reuse-local branch in
// symMemory{Export,ImportAndMap}SegmentHandle (no real shareable-handle
// export/import needed).
// ---------------------------------------------------------------------------
thread_local int             ncclGroupDepth = 0;
thread_local ncclResult_t    ncclGroupError = ncclSuccess;
thread_local struct ncclComm* ncclGroupCommHead[ncclGroupTaskTypeNum] = {};
thread_local int             ncclGroupBlocking = 0;

// ---------------------------------------------------------------------------
// Fake HIP VMM driver API, backed by ordinary host memory.
//
// dev_runtime.cc drives the CUDA/HIP driver VMM API (hipMemAddressReserve /
// hipMemMap / ...) which needs a real GPU. Here we replace just those calls
// with host-memory equivalents so symMemoryObtain runs to completion on a plain
// CPU. HIDDEN visibility is essential: it satisfies dev_runtime's references
// without exporting these names, so libamdhip64's own internal calls still bind
// to the real driver (no process-wide interposition).
// ---------------------------------------------------------------------------
#define HIP_FAKE /* default visibility: this binary does not link librccl */

// A reserved VA range: mirror the real cuMemAddressReserve semantics with an
// uncommitted anonymous mapping (MAP_NORESERVE). This makes multi-GB flat-VA
// reservations cheap and never dereferenced (all cuMemMap/SetAccess are no-ops),
// so no physical memory is committed.
HIP_FAKE hipError_t hipMemAddressReserve(void** ptr, size_t size, size_t, void*, unsigned long long) {
  // The real driver rejects a zero-size reservation; a zero here would be a bug
  // in the code under test, so surface it rather than silently substituting one.
  assert(size != 0);
  void* p = mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (p == MAP_FAILED) return hipErrorOutOfMemory;
  *ptr = p;
  return hipSuccess;
}
HIP_FAKE hipError_t hipMemAddressFree(void* devPtr, size_t size) {
  assert(size != 0);
  munmap(devPtr, size);
  return hipSuccess;
}
HIP_FAKE hipError_t hipMemCreate(hipMemGenericAllocationHandle_t* handle, size_t, const hipMemAllocationProp*,
                                 unsigned long long) {
  if (handle) *handle = reinterpret_cast<hipMemGenericAllocationHandle_t>(0x1);
  return hipSuccess;
}
HIP_FAKE hipError_t hipMemGetAllocationGranularity(size_t* granularity, const hipMemAllocationProp*,
                                                   hipMemAllocationGranularity_flags) {
  if (granularity) *granularity = 4096;
  return hipSuccess;
}
HIP_FAKE hipError_t hipMemGetAllocationPropertiesFromHandle(hipMemAllocationProp* prop,
                                                           hipMemGenericAllocationHandle_t) {
  if (prop) {
    *prop = hipMemAllocationProp{};
    prop->location.type = hipMemLocationTypeDevice;
  }
  return hipSuccess;
}
HIP_FAKE hipError_t hipMemExportToShareableHandle(void*, hipMemGenericAllocationHandle_t,
                                                  hipMemAllocationHandleType, unsigned long long) {
  return hipSuccess;
}
HIP_FAKE hipError_t hipMemImportFromShareableHandle(hipMemGenericAllocationHandle_t* handle, void*,
                                                    hipMemAllocationHandleType) {
  if (handle) *handle = reinterpret_cast<hipMemGenericAllocationHandle_t>(0x1);
  return hipSuccess;
}
HIP_FAKE hipError_t hipMemMap(void*, size_t, size_t, hipMemGenericAllocationHandle_t, unsigned long long) {
  return hipSuccess;
}
HIP_FAKE hipError_t hipMemSetAccess(void*, size_t, const hipMemAccessDesc*, size_t) { return hipSuccess; }
HIP_FAKE hipError_t hipMemUnmap(void*, size_t) { return hipSuccess; }
HIP_FAKE hipError_t hipMemRelease(hipMemGenericAllocationHandle_t) { return hipSuccess; }
HIP_FAKE hipError_t hipMemRetainAllocationHandle(hipMemGenericAllocationHandle_t* handle, void*) {
  if (handle) *handle = reinterpret_cast<hipMemGenericAllocationHandle_t>(0x1);
  return hipSuccess;
}
HIP_FAKE hipError_t hipMemGetAddressRange(hipDeviceptr_t* pbase, size_t* psize, hipDeviceptr_t dptr) {
  if (pbase) *pbase = dptr;
  if (psize) *psize = 0;
  return hipSuccess;
}

// ---------------------------------------------------------------------------
// Fake HIP runtime stream API. ncclDevrFinalize creates/synchronizes/destroys
// throwaway streams for its teardown bookkeeping; none carry real work on the
// host, so a non-null opaque handle and success returns are sufficient.
// ---------------------------------------------------------------------------
HIP_FAKE hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int) {
  if (stream) *stream = reinterpret_cast<hipStream_t>(0x1);
  return hipSuccess;
}
HIP_FAKE hipError_t hipStreamSynchronize(hipStream_t) { return hipSuccess; }
HIP_FAKE hipError_t hipStreamDestroy(hipStream_t) { return hipSuccess; }
HIP_FAKE hipError_t hipThreadExchangeStreamCaptureMode(hipStreamCaptureMode* mode) {
  if (mode) *mode = hipStreamCaptureModeRelaxed;
  return hipSuccess;
}
