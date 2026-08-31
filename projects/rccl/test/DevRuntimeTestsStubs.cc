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
#include "enqueue.h"
#include "enqueue/mgmt_task_enq.h"
#include "gin/gin_host.h"
#include "rma/rma.h"
#include "rma/rma_proxy.h"
#include "rma/rma_ce.h"
#include "nccl_device/core_tmp.h"
#include "nccl_device/core.h"
#include "nccl_device/cft_barrier.h"
#include "nccl_device/lsa_barrier.h"
#include "nccl_device/gin_barrier.h"

#include <cassert>
#include <cstdarg>
#include <cstdlib>
#include <sys/mman.h>

// ---------------------------------------------------------------------------
// Globals the translation unit references.
// ---------------------------------------------------------------------------
int                          ncclDebugLevel = 0;
uint64_t                     ncclDebugMask  = 0;
thread_local int             ncclDebugNoWarn = 0;
// Use POSIX-FD handles so the single-rank success path takes the no-export /
// reuse-local branch in symMemory{Export,ImportAndMap}SegmentHandle (no real
// shareable-handle export/import needed).
hipMemAllocationHandleType   ncclCuMemHandleType = hipMemHandleTypePosixFileDescriptor;

thread_local int             ncclGroupDepth = 0;
thread_local ncclResult_t    ncclGroupError = ncclSuccess;
thread_local struct ncclComm* ncclGroupCommHead[ncclGroupTaskTypeNum] = {};
thread_local int             ncclGroupBlocking = 0;

// devcomm compat tables (defined in devcomm/devcomm_v*.cc in the real build).
struct ncclDevCommCompat ncclDevCommCompat_v22902 = {};
struct ncclDevCommCompat ncclDevCommCompat_v22907 = {};
struct ncclDevCommCompat ncclDevCommCompat_v23000 = {};
struct ncclDevCommCompat ncclDevCommCompat_v23100 = {};

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
// Param loader / enqueue rearch gate.
// ---------------------------------------------------------------------------
int64_t ncclLoadParam(char const*, int64_t deftVal, int64_t, int64_t* cache, int8_t* noCache) {
  if (cache) *cache = deftVal;
  if (noCache) *noCache = 0;
  return deftVal;
}
int64_t ncclParamEnqueueRearchEnable() { return 0; }

// ---------------------------------------------------------------------------
// Proxy / mgmt task enqueue.
// ---------------------------------------------------------------------------
ncclResult_t ncclProxyClientGetFdBlocking(struct ncclComm*, int, void*, int*) { return ncclSuccess; }
ncclResult_t ncclMgmtTaskEnqueue(struct ncclAsyncJob*, ncclResult_t (*)(struct ncclAsyncJob*), void (*)(void*),
                                 struct ncclComm*) {
  return ncclSuccess;
}

// ---------------------------------------------------------------------------
// Symmetric kernels / RMA.
// ---------------------------------------------------------------------------
ncclResult_t ncclSymkInitOnce(struct ncclComm*) { return ncclSuccess; }
bool ncclRmaProxyEnabled(struct ncclComm*) { return false; }
ncclResult_t ncclRmaCeInit(struct ncclComm*) { return ncclSuccess; }

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
ncclResult_t ncclGinDevCommSetup(struct ncclComm*, struct ncclDevCommRequirements const*, struct ncclDevComm*,
                                 uint32_t) {
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
ncclResult_t ncclDevrVerifySegmentLayouts(struct ncclDevrMemory*, struct ncclComm*) { return ncclSuccess; }
ncclResult_t ncclDevrBuildGinSegmentInfos(struct ncclDevrMemory*) { return ncclSuccess; }

// ---------------------------------------------------------------------------
// CFT / LE helpers pulled in via #include of hipified dev_runtime.cc.
// ---------------------------------------------------------------------------
int computeCftSize(struct ncclComm*) { return 1; }
int computeCftMcSize(struct ncclComm*) { return 1; }
ncclResult_t symBindTeamLe(struct ncclComm*, struct ncclDevrMemory*, ncclCftLeId) { return ncclSuccess; }
ncclResult_t symUnbindTeamLe(struct ncclComm*, struct ncclDevrMemory*, ncclCftLeId) { return ncclSuccess; }
ncclResult_t symTeamObtainUcLe(struct ncclComm*, struct ncclDevrTeam*, struct ncclDevrState*, bool* needBarrier) {
  if (needBarrier) *needBarrier = false;
  return ncclSuccess;
}
ncclResult_t symTeamObtainMcLe(struct ncclComm*, struct ncclDevrTeam*, struct ncclDevrState*, bool* needBarrier) {
  if (needBarrier) *needBarrier = false;
  return ncclSuccess;
}

// ---------------------------------------------------------------------------
// Team accessors (host variants).
// ---------------------------------------------------------------------------
extern "C" ncclTeam_t ncclTeamWorld(ncclComm_t) { return ncclTeam_t{}; }
extern "C" ncclTeam_t ncclTeamLsa(ncclComm_t) { return ncclTeam_t{}; }
extern "C" ncclTeam_t ncclTeamRail(ncclComm_t) { return ncclTeam_t{}; }
extern "C" ncclTeam_t ncclTeamCft(ncclComm_t, ncclCftTeamMode_t) { return ncclTeam_t{}; }
extern "C" ncclTeam_t ncclTeamCftMultimem(ncclComm_t) { return ncclTeam_t{}; }

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
extern "C" ncclResult_t ncclCftBarrierCreateRequirement(ncclTeam_t, int, ncclCftBarrierHandle_t*,
                                                        ncclDevResourceRequirements_t*) {
  return ncclSuccess;
}

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
