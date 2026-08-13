/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * No-op host stubs for the DevRuntimeTests micro-test binary.
 *
 * dev_runtime.cc is #included whole into DevRuntimeTests.cpp, which leaves
 * undefined references to everything the translation unit calls but does not
 * define. These inert host-side definitions let the binary link without
 * librccl.so or a GPU. Real headers are included so every signature is checked
 * against the real declaration rather than hand-transcribed.
 *************************************************************************/

#include "comm.h"
#include "bootstrap.h"
#include "argcheck.h"
#include "group.h"
#include "param.h"
#include "proxy.h"
#include "sym_kernels.h"
#include "allocator.h"
#include "utils.h"
#include "cudawrap.h"
#include "dev_runtime_internal.h"
#include "gin/gin_host.h"
#include "rma/rma_proxy.h"
#include "nccl_device/core_tmp.h"
#include "nccl_device/lsa_barrier.h"
#include "nccl_device/gin_barrier.h"

#include <cstdarg>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Globals the translation unit references.
// ---------------------------------------------------------------------------
int                          ncclDebugLevel = 0;
uint64_t                     ncclDebugMask  = 0;
thread_local int             ncclDebugNoWarn = 0;
hipMemAllocationHandleType   ncclCuMemHandleType = hipMemHandleTypeNone;

thread_local int             ncclGroupDepth = 0;
thread_local ncclResult_t    ncclGroupError = ncclSuccess;
thread_local struct ncclComm* ncclGroupCommHead[ncclGroupTaskTypeNum] = {};
thread_local int             ncclGroupBlocking = 0;

// devcomm compat tables (defined in devcomm/devcomm_v*.cc in the real build).
struct ncclDevCommCompat ncclDevCommCompat_v22902 = {};
struct ncclDevCommCompat ncclDevCommCompat_v22907 = {};
struct ncclDevCommCompat ncclDevCommCompat_v23000 = {};

// ---------------------------------------------------------------------------
// Debug / error.
// ---------------------------------------------------------------------------
void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char*, int, const char*, ...) {}
const char* ncclGetErrorString(ncclResult_t) { return "ncclSuccess"; }

// ---------------------------------------------------------------------------
// Bootstrap.
// ---------------------------------------------------------------------------
ncclResult_t bootstrapAllGather(void*, void*, int) { return ncclSuccess; }
ncclResult_t bootstrapBarrier(void*, int, int, int) { return ncclSuccess; }
ncclResult_t bootstrapIntraNodeBarrier(void*, int*, int, int, int) { return ncclSuccess; }
ncclResult_t bootstrapIntraNodeAllGather(void*, int*, int, int, void*, int) { return ncclSuccess; }

// ---------------------------------------------------------------------------
// Arg checks / comm readiness.
// ---------------------------------------------------------------------------
ncclResult_t PtrCheck(const void*, const char*, const char*) { return ncclSuccess; }
ncclResult_t CommCheck(struct ncclComm*, const char*, const char*) { return ncclSuccess; }
ncclResult_t ncclCommEnsureReady(ncclComm_t) { return ncclSuccess; }

// ---------------------------------------------------------------------------
// Public registration API.
// ---------------------------------------------------------------------------
ncclResult_t ncclCommRegister(const ncclComm_t, void*, size_t, void**) { return ncclSuccess; }
ncclResult_t ncclCommDeregister(const ncclComm_t, void*) { return ncclSuccess; }
ncclResult_t ncclCommWindowDeregister(ncclComm_t, ncclWindow_t) { return ncclSuccess; }

// ---------------------------------------------------------------------------
// Group state machine.
// ---------------------------------------------------------------------------
ncclResult_t ncclGroupStartInternal() { return ncclSuccess; }
ncclResult_t ncclGroupEndInternal(ncclSimInfo_t*) { return ncclSuccess; }

// ---------------------------------------------------------------------------
// Param loader.
// ---------------------------------------------------------------------------
int64_t ncclLoadParam(char const*, int64_t deftVal, int64_t, int64_t* cache, int8_t* noCache) {
  if (cache) *cache = deftVal;
  if (noCache) *noCache = 0;
  return deftVal;
}

// ---------------------------------------------------------------------------
// Proxy.
// ---------------------------------------------------------------------------
ncclResult_t ncclProxyClientGetFdBlocking(struct ncclComm*, int, void*, int*) { return ncclSuccess; }

// ---------------------------------------------------------------------------
// Symmetric kernels.
// ---------------------------------------------------------------------------
ncclResult_t ncclSymkInitOnce(struct ncclComm*) { return ncclSuccess; }

// ---------------------------------------------------------------------------
// Space allocator.
// ---------------------------------------------------------------------------
void         ncclSpaceConstruct(struct ncclSpace*) {}
void         ncclSpaceDestruct(struct ncclSpace*) {}
ncclResult_t ncclSpaceAlloc(struct ncclSpace*, int64_t, int64_t, int, int64_t* outOffset) {
  if (outOffset) *outOffset = 0;
  return ncclSuccess;
}
ncclResult_t ncclSpaceFree(struct ncclSpace*, int64_t, int64_t) { return ncclSuccess; }

// ---------------------------------------------------------------------------
// Shadow pool.
// ---------------------------------------------------------------------------
void         ncclShadowPoolConstruct(struct ncclShadowPool*) {}
ncclResult_t ncclShadowPoolDestruct(struct ncclShadowPool*, hipStream_t) { return ncclSuccess; }
ncclResult_t ncclShadowPoolAlloc(struct ncclShadowPool*, size_t, void** outDevObj, void** outHostObj, hipStream_t) {
  if (outDevObj) *outDevObj = nullptr;
  if (outHostObj) *outHostObj = nullptr;
  return ncclSuccess;
}
ncclResult_t ncclShadowPoolFree(struct ncclShadowPool*, void*, hipStream_t) { return ncclSuccess; }
ncclResult_t ncclShadowPoolToHost(struct ncclShadowPool*, void*, void** outHostObj) {
  if (outHostObj) *outHostObj = nullptr;
  return ncclSuccess;
}

// ---------------------------------------------------------------------------
// Intrusive address map.
// ---------------------------------------------------------------------------
ncclResult_t ncclIntruAddressMapInsert_untyped(struct ncclIntruAddressMap_untyped*, int, int, int, uintptr_t, void*) {
  return ncclSuccess;
}
ncclResult_t ncclIntruAddressMapFind_untyped(struct ncclIntruAddressMap_untyped*, int, int, int, uintptr_t, void** out) {
  if (out) *out = nullptr;
  return ncclSuccess;
}
ncclResult_t ncclIntruAddressMapRemove_untyped(struct ncclIntruAddressMap_untyped*, int, int, int, uintptr_t) {
  return ncclSuccess;
}

// ---------------------------------------------------------------------------
// Memory stack spill (must return real memory to avoid a crash if ever hit).
// ---------------------------------------------------------------------------
void* ncclMemoryStack::allocateSpilled(struct ncclMemoryStack*, size_t size, size_t align) {
  void* p = nullptr;
  if (align < sizeof(void*)) align = sizeof(void*);
  if (posix_memalign(&p, align, size) != 0) return nullptr;
  return p;  // intentionally leaked; process is short-lived
}

// ---------------------------------------------------------------------------
// GIN host.
// ---------------------------------------------------------------------------
ncclResult_t ncclGetGinType(struct ncclComm*, ncclGinType_t* ginType) {
  if (ginType) *ginType = NCCL_GIN_TYPE_NONE;
  return ncclSuccess;
}
ncclResult_t ncclGetRailedGinType(struct ncclComm*, ncclGinType_t* ginType) {
  if (ginType) *ginType = NCCL_GIN_TYPE_NONE;
  return ncclSuccess;
}
ncclResult_t ncclGinConnectOnce(struct ncclComm*) { return ncclSuccess; }
ncclResult_t ncclGinDevCommSetup(struct ncclComm*, struct ncclDevCommRequirements const*, struct ncclDevComm*) {
  return ncclSuccess;
}
ncclResult_t ncclGinDevCommFree(struct ncclComm*, struct ncclDevComm const*) { return ncclSuccess; }
ncclResult_t ncclGinRegister(struct ncclComm*, void*, size_t, void*[NCCL_GIN_MAX_CONNECTIONS],
                             ncclGinWindow_t[NCCL_GIN_MAX_CONNECTIONS], int, bool, int) {
  return ncclSuccess;
}
ncclResult_t ncclGinDeregister(struct ncclComm*, void*[NCCL_GIN_MAX_CONNECTIONS]) { return ncclSuccess; }

// ---------------------------------------------------------------------------
// RMA proxy.
// ---------------------------------------------------------------------------
ncclResult_t ncclRmaProxyConnectOnce(struct ncclComm*) { return ncclSuccess; }
ncclResult_t ncclRmaProxyRegister(struct ncclComm*, void*, size_t, void*[NCCL_GIN_MAX_CONNECTIONS]) {
  return ncclSuccess;
}
ncclResult_t ncclRmaProxyDeregister(struct ncclComm*, void*[NCCL_GIN_MAX_CONNECTIONS]) { return ncclSuccess; }

// ---------------------------------------------------------------------------
// devr internal helpers (defined elsewhere in the real build).
// ---------------------------------------------------------------------------
ncclResult_t ncclDevrPopulateSegmentSizes(struct ncclDevrMemory*, int) { return ncclSuccess; }
ncclResult_t ncclDevrAllocAndPopulateSegmentWindows(struct ncclDevrState*, struct ncclDevrMemory*, hipStream_t,
                                                    struct ncclSegmentWindow** out) {
  if (out) *out = nullptr;
  return ncclSuccess;
}

// ---------------------------------------------------------------------------
// Team accessors (host variants).
// ---------------------------------------------------------------------------
extern "C" ncclTeam_t ncclTeamWorld(ncclComm_t) { return ncclTeam_t{}; }
extern "C" ncclTeam_t ncclTeamLsa(ncclComm_t) { return ncclTeam_t{}; }
extern "C" ncclTeam_t ncclTeamRail(ncclComm_t) { return ncclTeam_t{}; }

// ---------------------------------------------------------------------------
// Barrier requirement builders (host variants).
// ---------------------------------------------------------------------------
extern "C" ncclResult_t ncclLsaBarrierCreateRequirement(ncclTeam_t, int, ncclLsaBarrierHandle_t*,
                                                        ncclDevResourceRequirements_t*) {
  return ncclSuccess;
}
extern "C" ncclResult_t ncclGinBarrierCreateRequirement(ncclComm_t, ncclTeam_t, int, ncclGinBarrierHandle_t*,
                                                        ncclDevResourceRequirements_t*) {
  return ncclSuccess;
}
