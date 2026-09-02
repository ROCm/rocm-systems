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

#include "DevRuntimeTestsStubs.h"

#include <cassert>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <set>
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

// ---------------------------------------------------------------------------
// Debug / error.
// ---------------------------------------------------------------------------
void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char*, int, const char*, ...) {}
const char* ncclGetErrorString(ncclResult_t) { return "ncclSuccess"; }

// ---------------------------------------------------------------------------
// Bootstrap.
// ---------------------------------------------------------------------------
static ncclResult_t DefaultAllGather(void*, void*, int) { return ncclSuccess; }
std::function<ncclResult_t(void*, void*, int)> g_bootstrapAllGather = DefaultAllGather;

ncclResult_t bootstrapAllGather(void* bs, void* buf, int bytes) { return g_bootstrapAllGather(bs, buf, bytes); }
ncclResult_t bootstrapBarrier(void*, int, int, int) { return ncclSuccess; }
// Seams: symMemoryMapLsaTeam NCCLCHECKGOTOs both of these, so their failure
// arms are only reachable by driving them.
static ncclResult_t DefaultIntraNodeBarrier(void*, int*, int, int, int) { return ncclSuccess; }
std::function<ncclResult_t(void*, int*, int, int, int)> g_bootstrapIntraNodeBarrier = DefaultIntraNodeBarrier;

ncclResult_t bootstrapIntraNodeBarrier(void* bs, int* ranks, int self, int size, int tag) {
  return g_bootstrapIntraNodeBarrier(bs, ranks, self, size, tag);
}

static ncclResult_t DefaultIntraNodeAllGather(void*, int*, int, int, void*, int) { return ncclSuccess; }
std::function<ncclResult_t(void*, int*, int, int, void*, int)> g_bootstrapIntraNodeAllGather =
    DefaultIntraNodeAllGather;

ncclResult_t bootstrapIntraNodeAllGather(void* bs, int* ranks, int self, int size, void* buf, int bytes) {
  return g_bootstrapIntraNodeAllGather(bs, ranks, self, size, buf, bytes);
}

// ---------------------------------------------------------------------------
// Arg checks / comm readiness.
// ---------------------------------------------------------------------------
ncclResult_t PtrCheck(const void*, const char*, const char*) { return ncclSuccess; }
ncclResult_t CommCheck(struct ncclComm*, const char*, const char*) { return ncclSuccess; }
ncclResult_t ncclCommEnsureReady(ncclComm_t) { return ncclSuccess; }

// ---------------------------------------------------------------------------
// Public registration API.
// ---------------------------------------------------------------------------
// Seams: ncclDevrWindowRegisterInGroup takes a local registration up front and
// releases it on every failure path, which is only observable by counting.
static ncclResult_t DefaultCommRegister(const ncclComm_t, void*, size_t, void** handle) {
  if (handle) *handle = reinterpret_cast<void*>(0x1234);
  return ncclSuccess;
}
std::function<ncclResult_t(const ncclComm_t, void*, size_t, void**)> g_ncclCommRegister = DefaultCommRegister;

ncclResult_t ncclCommRegister(const ncclComm_t comm, void* ptr, size_t size, void** handle) {
  return g_ncclCommRegister(comm, ptr, size, handle);
}

static ncclResult_t DefaultCommDeregister(const ncclComm_t, void*) { return ncclSuccess; }
std::function<ncclResult_t(const ncclComm_t, void*)> g_ncclCommDeregister = DefaultCommDeregister;

ncclResult_t ncclCommDeregister(const ncclComm_t comm, void* handle) {
  return g_ncclCommDeregister(comm, handle);
}
// Seam: ncclDevCommDestroy releases its resource window through the public
// wrapper, so the branch is only observable by counting.
static ncclResult_t DefaultCommWindowDeregister(ncclComm_t, ncclWindow_t) { return ncclSuccess; }
std::function<ncclResult_t(ncclComm_t, ncclWindow_t)> g_ncclCommWindowDeregister = DefaultCommWindowDeregister;

ncclResult_t ncclCommWindowDeregister(ncclComm_t comm, ncclWindow_t win) {
  return g_ncclCommWindowDeregister(comm, win);
}

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
// The real proxy hands back an fd the caller owns and closes. Returning -1
// would make the SYSCHECK on close() in symMemoryImportAndMapSegmentHandle fail
// and read like an import bug, so hand out a real descriptor the code can close.
static ncclResult_t DefaultProxyClientGetFdBlocking(struct ncclComm*, int, void*, int* fd) {
  if (fd) {
    *fd = open("/dev/null", O_RDONLY);
    if (*fd < 0) return ncclSystemError;
  }
  return ncclSuccess;
}
std::function<ncclResult_t(struct ncclComm*, int, void*, int*)> g_proxyClientGetFdBlocking =
    DefaultProxyClientGetFdBlocking;

ncclResult_t ncclProxyClientGetFdBlocking(struct ncclComm* comm, int rank, void* handle, int* fd) {
  return g_proxyClientGetFdBlocking(comm, rank, handle, fd);
}

// ---------------------------------------------------------------------------
// Symmetric kernels.
// ---------------------------------------------------------------------------
ncclResult_t ncclSymkInitOnce(struct ncclComm*) { return ncclSuccess; }

// ---------------------------------------------------------------------------
// Space allocator.
// ---------------------------------------------------------------------------
void         ncclSpaceConstruct(struct ncclSpace*) {}
void         ncclSpaceDestruct(struct ncclSpace*) {}
// Seams: symMemoryObtain NCCLCHECKGOTOs the alloc, and its rollback is only
// observable through the matching free.
static ncclResult_t DefaultSpaceAlloc(struct ncclSpace*, int64_t, int64_t, int, int64_t* outOffset) {
  if (outOffset) *outOffset = 0;
  return ncclSuccess;
}
std::function<ncclResult_t(struct ncclSpace*, int64_t, int64_t, int, int64_t*)> g_spaceAlloc = DefaultSpaceAlloc;

ncclResult_t ncclSpaceAlloc(struct ncclSpace* sp, int64_t total, int64_t size, int align, int64_t* outOffset) {
  return g_spaceAlloc(sp, total, size, align, outOffset);
}

static ncclResult_t DefaultSpaceFree(struct ncclSpace*, int64_t, int64_t) { return ncclSuccess; }
std::function<ncclResult_t(struct ncclSpace*, int64_t, int64_t)> g_spaceFree = DefaultSpaceFree;

ncclResult_t ncclSpaceFree(struct ncclSpace* sp, int64_t offset, int64_t size) {
  return g_spaceFree(sp, offset, size);
}

// ---------------------------------------------------------------------------
// Shadow pool.
// ---------------------------------------------------------------------------
void         ncclShadowPoolConstruct(struct ncclShadowPool*) {}
ncclResult_t ncclShadowPoolDestruct(struct ncclShadowPool*, hipStream_t) { return ncclSuccess; }
// The real pool hands out a device object plus a host shadow of it. Here one
// zeroed host buffer stands in for both, so ToHost is the identity: callers
// that write through the host pointer (allocAndPopulateSegmentWindows) get real
// storage rather than the nullptr the previous stub returned.
// Live allocations, so a freed handle stops resolving. Without this the fake
// hands a freed pointer back from ToHost and callers that decode a stale handle
// read through it -- the real pool drops the mapping on free.
static std::set<void*>& ShadowPoolLive() {
  static std::set<void*> live;
  return live;
}

static ncclResult_t DefaultShadowPoolAlloc(struct ncclShadowPool*, size_t size, void** outDevObj, void** outHostObj,
                                           hipStream_t) {
  void* p = calloc(1, size != 0 ? size : 1);
  if (p == nullptr) return ncclSystemError;
  ShadowPoolLive().insert(p);
  if (outDevObj) *outDevObj = p;
  if (outHostObj) *outHostObj = p;
  return ncclSuccess;
}
std::function<ncclResult_t(struct ncclShadowPool*, size_t, void**, void**, hipStream_t)> g_shadowPoolAlloc =
    DefaultShadowPoolAlloc;

ncclResult_t ncclShadowPoolAlloc(struct ncclShadowPool* pool, size_t size, void** outDevObj, void** outHostObj,
                                 hipStream_t stream) {
  return g_shadowPoolAlloc(pool, size, outDevObj, outHostObj, stream);
}

static ncclResult_t DefaultShadowPoolFree(struct ncclShadowPool*, void* devObj, hipStream_t) {
  ShadowPoolLive().erase(devObj);
  free(devObj);
  return ncclSuccess;
}
std::function<ncclResult_t(struct ncclShadowPool*, void*, hipStream_t)> g_shadowPoolFree = DefaultShadowPoolFree;

ncclResult_t ncclShadowPoolFree(struct ncclShadowPool* pool, void* devObj, hipStream_t stream) {
  return g_shadowPoolFree(pool, devObj, stream);
}

static ncclResult_t DefaultShadowPoolToHost(struct ncclShadowPool*, void* devObj, void** outHostObj) {
  if (ShadowPoolLive().count(devObj) == 0) return ncclInvalidArgument;  // freed or never ours
  if (outHostObj) *outHostObj = devObj;  // same buffer, see above
  return ncclSuccess;
}
std::function<ncclResult_t(struct ncclShadowPool*, void*, void**)> g_shadowPoolToHost = DefaultShadowPoolToHost;

ncclResult_t ncclShadowPoolToHost(struct ncclShadowPool* pool, void* devObj, void** outHostObj) {
  return g_shadowPoolToHost(pool, devObj, outHostObj);
}

// ---------------------------------------------------------------------------
// Intrusive address map.
// ---------------------------------------------------------------------------
ncclResult_t ncclIntruAddressMapInsert_untyped(struct ncclIntruAddressMap_untyped*, int, int, int, uintptr_t, void*) {
  return ncclSuccess;
}
// Seam: findCommAndHostWindowFromDeviceWindow resolves a device window through
// this map, so every public pointer accessor needs it to answer.
static ncclResult_t DefaultIntruAddressMapFind(struct ncclIntruAddressMap_untyped*, int, int, int, uintptr_t,
                                               void** out) {
  if (out) *out = nullptr;
  return ncclSuccess;
}
std::function<ncclResult_t(struct ncclIntruAddressMap_untyped*, int, int, int, uintptr_t, void**)>
    g_intruAddressMapFind = DefaultIntruAddressMapFind;

ncclResult_t ncclIntruAddressMapFind_untyped(struct ncclIntruAddressMap_untyped* map, int a, int b, int c,
                                             uintptr_t key, void** out) {
  return g_intruAddressMapFind(map, a, b, c, key, out);
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
// Seams: symMemoryRegisterGin NCCLCHECKs both, and its rollback path is only
// observable through the deregister count.
static ncclResult_t DefaultGinRegister(struct ncclComm*, void*, size_t, void*[NCCL_GIN_MAX_CONNECTIONS],
                                       ncclGinWindow_t[NCCL_GIN_MAX_CONNECTIONS], int, bool, int) {
  return ncclSuccess;
}
std::function<ncclResult_t(struct ncclComm*, void*, size_t, void*[NCCL_GIN_MAX_CONNECTIONS],
                           ncclGinWindow_t[NCCL_GIN_MAX_CONNECTIONS], int, bool, int)>
    g_ginRegister = DefaultGinRegister;

ncclResult_t ncclGinRegister(struct ncclComm* comm, void* addr, size_t size,
                             void* hostWins[NCCL_GIN_MAX_CONNECTIONS],
                             ncclGinWindow_t devWins[NCCL_GIN_MAX_CONNECTIONS], int winFlags, bool multiSegment,
                             int memType) {
  return g_ginRegister(comm, addr, size, hostWins, devWins, winFlags, multiSegment, memType);
}

static ncclResult_t DefaultGinDeregister(struct ncclComm*, void*[NCCL_GIN_MAX_CONNECTIONS]) { return ncclSuccess; }
std::function<ncclResult_t(struct ncclComm*, void*[NCCL_GIN_MAX_CONNECTIONS])> g_ginDeregister =
    DefaultGinDeregister;

ncclResult_t ncclGinDeregister(struct ncclComm* comm, void* hostWins[NCCL_GIN_MAX_CONNECTIONS]) {
  return g_ginDeregister(comm, hostWins);
}

// ---------------------------------------------------------------------------
// RMA proxy.
// ---------------------------------------------------------------------------
// Seams: symMemoryRegisterRma NCCLCHECKs both.
static ncclResult_t DefaultRmaProxyConnectOnce(struct ncclComm*) { return ncclSuccess; }
std::function<ncclResult_t(struct ncclComm*)> g_rmaProxyConnectOnce = DefaultRmaProxyConnectOnce;

ncclResult_t ncclRmaProxyConnectOnce(struct ncclComm* comm) { return g_rmaProxyConnectOnce(comm); }

static ncclResult_t DefaultRmaProxyRegister(struct ncclComm*, void*, size_t, void*[NCCL_GIN_MAX_CONNECTIONS]) {
  return ncclSuccess;
}
std::function<ncclResult_t(struct ncclComm*, void*, size_t, void*[NCCL_GIN_MAX_CONNECTIONS])> g_rmaProxyRegister =
    DefaultRmaProxyRegister;

ncclResult_t ncclRmaProxyRegister(struct ncclComm* comm, void* addr, size_t size,
                                  void* hostWins[NCCL_GIN_MAX_CONNECTIONS]) {
  return g_rmaProxyRegister(comm, addr, size, hostWins);
}

static ncclResult_t DefaultRmaProxyDeregister(struct ncclComm*, void*[NCCL_GIN_MAX_CONNECTIONS]) {
  return ncclSuccess;
}
std::function<ncclResult_t(struct ncclComm*, void*[NCCL_GIN_MAX_CONNECTIONS])> g_rmaProxyDeregister =
    DefaultRmaProxyDeregister;

ncclResult_t ncclRmaProxyDeregister(struct ncclComm* comm, void* wins[NCCL_GIN_MAX_CONNECTIONS]) {
  return g_rmaProxyDeregister(comm, wins);
}

// ---------------------------------------------------------------------------
// devr internal helpers (defined elsewhere in the real build).
// ---------------------------------------------------------------------------
static ncclResult_t DefaultPopulateSegmentSizes(struct ncclDevrMemory*, int) { return ncclSuccess; }
std::function<ncclResult_t(struct ncclDevrMemory*, int)> g_devrPopulateSegmentSizes = DefaultPopulateSegmentSizes;

ncclResult_t ncclDevrPopulateSegmentSizes(struct ncclDevrMemory* mem, int numSegments) {
  return g_devrPopulateSegmentSizes(mem, numSegments);
}
static ncclResult_t DefaultDevrAllocAndPopulateSegmentWindows(struct ncclDevrState*, struct ncclDevrMemory*,
                                                              hipStream_t, struct ncclSegmentWindow** out) {
  if (out) *out = nullptr;
  return ncclSuccess;
}
std::function<ncclResult_t(struct ncclDevrState*, struct ncclDevrMemory*, hipStream_t, struct ncclSegmentWindow**)>
    g_devrAllocAndPopulateSegmentWindows = DefaultDevrAllocAndPopulateSegmentWindows;

ncclResult_t ncclDevrAllocAndPopulateSegmentWindows(struct ncclDevrState* devr, struct ncclDevrMemory* mem,
                                                    hipStream_t stream, struct ncclSegmentWindow** out) {
  return g_devrAllocAndPopulateSegmentWindows(devr, mem, stream, out);
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
static hipError_t DefaultMemAddressReserve(void** ptr, size_t size, size_t, void*, unsigned long long) {
  // The real driver rejects a zero-size reservation; a zero here would be a bug
  // in the code under test, so surface it rather than silently substituting one.
  assert(size != 0);
  void* p = mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (p == MAP_FAILED) return hipErrorOutOfMemory;
  *ptr = p;
  return hipSuccess;
}
std::function<hipError_t(void**, size_t, size_t, void*, unsigned long long)> g_hipMemAddressReserve =
    DefaultMemAddressReserve;

HIP_FAKE hipError_t hipMemAddressReserve(void** ptr, size_t size, size_t align, void* addr,
                                         unsigned long long flags) {
  return g_hipMemAddressReserve(ptr, size, align, addr, flags);
}
static hipError_t DefaultMemAddressFree(void* devPtr, size_t size) {
  assert(size != 0);
  munmap(devPtr, size);
  return hipSuccess;
}
std::function<hipError_t(void*, size_t)> g_hipMemAddressFree = DefaultMemAddressFree;

HIP_FAKE hipError_t hipMemAddressFree(void* devPtr, size_t size) {
  return g_hipMemAddressFree(devPtr, size);
}
HIP_FAKE hipError_t hipMemCreate(hipMemGenericAllocationHandle_t* handle, size_t, const hipMemAllocationProp*,
                                 unsigned long long) {
  if (handle) *handle = reinterpret_cast<hipMemGenericAllocationHandle_t>(0x1);
  return hipSuccess;
}
static hipError_t DefaultMemGetAllocationGranularity(size_t* granularity, const hipMemAllocationProp*,
                                                     hipMemAllocationGranularity_flags) {
  if (granularity) *granularity = 4096;
  return hipSuccess;
}
std::function<hipError_t(size_t*, const hipMemAllocationProp*, hipMemAllocationGranularity_flags)>
    g_hipMemGetAllocationGranularity = DefaultMemGetAllocationGranularity;

HIP_FAKE hipError_t hipMemGetAllocationGranularity(size_t* granularity, const hipMemAllocationProp* prop,
                                                   hipMemAllocationGranularity_flags flags) {
  return g_hipMemGetAllocationGranularity(granularity, prop, flags);
}
// Seams, not plain stubs: these three sit on error paths a test needs to drive.
// Each default lives in a named function so the hook's initialiser and
// ResetDevRuntimeFakes() share one definition instead of drifting copies.
static hipError_t DefaultMemGetAllocationPropertiesFromHandle(hipMemAllocationProp* prop,
                                                              hipMemGenericAllocationHandle_t) {
  if (prop) {
    *prop = hipMemAllocationProp{};
    prop->location.type = hipMemLocationTypeDevice;
  }
  return hipSuccess;
}
std::function<hipError_t(hipMemAllocationProp*, hipMemGenericAllocationHandle_t)>
    g_hipMemGetAllocationPropertiesFromHandle = DefaultMemGetAllocationPropertiesFromHandle;

HIP_FAKE hipError_t hipMemGetAllocationPropertiesFromHandle(hipMemAllocationProp* prop,
                                                           hipMemGenericAllocationHandle_t handle) {
  return g_hipMemGetAllocationPropertiesFromHandle(prop, handle);
}

static hipError_t DefaultMemExportToShareableHandle(void*, hipMemGenericAllocationHandle_t,
                                                    hipMemAllocationHandleType, unsigned long long) {
  return hipSuccess;
}
std::function<hipError_t(void*, hipMemGenericAllocationHandle_t, hipMemAllocationHandleType, unsigned long long)>
    g_hipMemExportToShareableHandle = DefaultMemExportToShareableHandle;

HIP_FAKE hipError_t hipMemExportToShareableHandle(void* shareable, hipMemGenericAllocationHandle_t handle,
                                                  hipMemAllocationHandleType type, unsigned long long flags) {
  return g_hipMemExportToShareableHandle(shareable, handle, type, flags);
}
static hipError_t DefaultMemImportFromShareableHandle(hipMemGenericAllocationHandle_t* handle, void*,
                                                      hipMemAllocationHandleType) {
  if (handle) *handle = reinterpret_cast<hipMemGenericAllocationHandle_t>(0x1);
  return hipSuccess;
}
std::function<hipError_t(hipMemGenericAllocationHandle_t*, void*, hipMemAllocationHandleType)>
    g_hipMemImportFromShareableHandle = DefaultMemImportFromShareableHandle;

HIP_FAKE hipError_t hipMemImportFromShareableHandle(hipMemGenericAllocationHandle_t* handle, void* shareable,
                                                    hipMemAllocationHandleType type) {
  return g_hipMemImportFromShareableHandle(handle, shareable, type);
}

static hipError_t DefaultMemMap(void*, size_t, size_t, hipMemGenericAllocationHandle_t, unsigned long long) {
  return hipSuccess;
}
std::function<hipError_t(void*, size_t, size_t, hipMemGenericAllocationHandle_t, unsigned long long)> g_hipMemMap =
    DefaultMemMap;

HIP_FAKE hipError_t hipMemMap(void* ptr, size_t size, size_t offset, hipMemGenericAllocationHandle_t handle,
                              unsigned long long flags) {
  return g_hipMemMap(ptr, size, offset, handle, flags);
}
static hipError_t DefaultMemSetAccess(void*, size_t, const hipMemAccessDesc*, size_t) { return hipSuccess; }
std::function<hipError_t(void*, size_t, const hipMemAccessDesc*, size_t)> g_hipMemSetAccess = DefaultMemSetAccess;

HIP_FAKE hipError_t hipMemSetAccess(void* ptr, size_t size, const hipMemAccessDesc* desc, size_t count) {
  return g_hipMemSetAccess(ptr, size, desc, count);
}

static hipError_t DefaultMemUnmap(void*, size_t) { return hipSuccess; }
std::function<hipError_t(void*, size_t)> g_hipMemUnmap = DefaultMemUnmap;

HIP_FAKE hipError_t hipMemUnmap(void* ptr, size_t size) { return g_hipMemUnmap(ptr, size); }

// Not faked until now, and its absence did not show as a link error: this
// target links the HIP runtime, so an unfaked call reaches the real driver and
// fails at run time without a GPU. The shadow-pool fake gives the same buffer
// for device and host, so a self-copy is skipped.
static hipError_t DefaultMemcpyAsync(void* dst, const void* src, size_t n, hipMemcpyKind, hipStream_t) {
  if (dst != nullptr && src != nullptr && dst != src) memcpy(dst, src, n);
  return hipSuccess;
}
std::function<hipError_t(void*, const void*, size_t, hipMemcpyKind, hipStream_t)> g_hipMemcpyAsync =
    DefaultMemcpyAsync;

HIP_FAKE hipError_t hipMemcpyAsync(void* dst, const void* src, size_t n, hipMemcpyKind kind, hipStream_t stream) {
  return g_hipMemcpyAsync(dst, src, n, kind, stream);
}

// Same reasoning as hipMemcpyAsync above: unfaked, this reaches the real driver
// rather than failing to link.
static hipError_t DefaultMemsetAsync(void* dst, int value, size_t n, hipStream_t) {
  if (dst != nullptr) memset(dst, value, n);
  return hipSuccess;
}
std::function<hipError_t(void*, int, size_t, hipStream_t)> g_hipMemsetAsync = DefaultMemsetAsync;

HIP_FAKE hipError_t hipMemsetAsync(void* dst, int value, size_t n, hipStream_t stream) {
  return g_hipMemsetAsync(dst, value, n, stream);
}

// IPC handles for the non-symmetric intra-node path. Unfaked these reach the
// real driver, as hipMemcpyAsync did.
static hipError_t DefaultIpcGetMemHandle(hipIpcMemHandle_t* handle, void*) {
  if (handle) *handle = hipIpcMemHandle_t{};
  return hipSuccess;
}
std::function<hipError_t(hipIpcMemHandle_t*, void*)> g_hipIpcGetMemHandle = DefaultIpcGetMemHandle;

HIP_FAKE hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* ptr) {
  return g_hipIpcGetMemHandle(handle, ptr);
}

static hipError_t DefaultIpcOpenMemHandle(void** ptr, hipIpcMemHandle_t, unsigned int) {
  if (ptr) *ptr = reinterpret_cast<void*>(0x9000);
  return hipSuccess;
}
std::function<hipError_t(void**, hipIpcMemHandle_t, unsigned int)> g_hipIpcOpenMemHandle =
    DefaultIpcOpenMemHandle;

HIP_FAKE hipError_t hipIpcOpenMemHandle(void** ptr, hipIpcMemHandle_t handle, unsigned int flags) {
  return g_hipIpcOpenMemHandle(ptr, handle, flags);
}

static hipError_t DefaultIpcCloseMemHandle(void*) { return hipSuccess; }
std::function<hipError_t(void*)> g_hipIpcCloseMemHandle = DefaultIpcCloseMemHandle;

HIP_FAKE hipError_t hipIpcCloseMemHandle(void* ptr) { return g_hipIpcCloseMemHandle(ptr); }

// Current-device get/set, used by the public window API to scope its work to
// the comm's device. Unfaked they reach the real driver, as the memcpy/memset
// pair did.
static hipError_t DefaultGetDevice(int* dev) {
  if (dev) *dev = 0;
  return hipSuccess;
}
std::function<hipError_t(int*)> g_hipGetDevice = DefaultGetDevice;

HIP_FAKE hipError_t hipGetDevice(int* dev) { return g_hipGetDevice(dev); }

static hipError_t DefaultSetDevice(int) { return hipSuccess; }
std::function<hipError_t(int)> g_hipSetDevice = DefaultSetDevice;

HIP_FAKE hipError_t hipSetDevice(int dev) { return g_hipSetDevice(dev); }

// Synchronous copy, used by the devcomm dump helpers. Unfaked it reaches the
// real driver; the default reports failure so a dump of a fake device handle
// prints nothing rather than reading through it.
static hipError_t DefaultMemcpy(void*, const void*, size_t, hipMemcpyKind) { return hipErrorInvalidValue; }
std::function<hipError_t(void*, const void*, size_t, hipMemcpyKind)> g_hipMemcpy = DefaultMemcpy;

HIP_FAKE hipError_t hipMemcpy(void* dst, const void* src, size_t n, hipMemcpyKind kind) {
  return g_hipMemcpy(dst, src, n, kind);
}
static hipError_t DefaultMemRelease(hipMemGenericAllocationHandle_t) { return hipSuccess; }
std::function<hipError_t(hipMemGenericAllocationHandle_t)> g_hipMemRelease = DefaultMemRelease;

HIP_FAKE hipError_t hipMemRelease(hipMemGenericAllocationHandle_t handle) { return g_hipMemRelease(handle); }
static hipError_t DefaultMemRetainAllocationHandle(hipMemGenericAllocationHandle_t* handle, void*) {
  if (handle) *handle = reinterpret_cast<hipMemGenericAllocationHandle_t>(0x1);
  return hipSuccess;
}
std::function<hipError_t(hipMemGenericAllocationHandle_t*, void*)> g_hipMemRetainAllocationHandle =
    DefaultMemRetainAllocationHandle;

HIP_FAKE hipError_t hipMemRetainAllocationHandle(hipMemGenericAllocationHandle_t* handle, void* ptr) {
  return g_hipMemRetainAllocationHandle(handle, ptr);
}
static hipError_t DefaultMemGetAddressRange(hipDeviceptr_t* pbase, size_t* psize, hipDeviceptr_t dptr) {
  if (pbase) *pbase = dptr;
  if (psize) *psize = 0;
  return hipSuccess;
}
std::function<hipError_t(hipDeviceptr_t*, size_t*, hipDeviceptr_t)> g_hipMemGetAddressRange =
    DefaultMemGetAddressRange;

HIP_FAKE hipError_t hipMemGetAddressRange(hipDeviceptr_t* pbase, size_t* psize, hipDeviceptr_t dptr) {
  return g_hipMemGetAddressRange(pbase, psize, dptr);
}

// ---------------------------------------------------------------------------
// Fake HIP runtime stream API. ncclDevrFinalize creates/synchronizes/destroys
// throwaway streams for its teardown bookkeeping; none carry real work on the
// host, so a non-null opaque handle and success returns are sufficient.
// ---------------------------------------------------------------------------
// Seams: ncclDevrFinalize wraps each of these in CUDACHECKIGNORE/CUDASUCCESS,
// so the failure arms are only reachable by driving them.
static hipError_t DefaultStreamCreateWithFlags(hipStream_t* stream, unsigned int) {
  if (stream) *stream = reinterpret_cast<hipStream_t>(0x1);
  return hipSuccess;
}
std::function<hipError_t(hipStream_t*, unsigned int)> g_hipStreamCreateWithFlags =
    DefaultStreamCreateWithFlags;

HIP_FAKE hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags) {
  return g_hipStreamCreateWithFlags(stream, flags);
}

static hipError_t DefaultStreamSynchronize(hipStream_t) { return hipSuccess; }
std::function<hipError_t(hipStream_t)> g_hipStreamSynchronize = DefaultStreamSynchronize;

HIP_FAKE hipError_t hipStreamSynchronize(hipStream_t stream) { return g_hipStreamSynchronize(stream); }

static hipError_t DefaultStreamDestroy(hipStream_t) { return hipSuccess; }
std::function<hipError_t(hipStream_t)> g_hipStreamDestroy = DefaultStreamDestroy;

HIP_FAKE hipError_t hipStreamDestroy(hipStream_t stream) { return g_hipStreamDestroy(stream); }

static hipError_t DefaultThreadExchangeStreamCaptureMode(hipStreamCaptureMode* mode) {
  if (mode) *mode = hipStreamCaptureModeRelaxed;
  return hipSuccess;
}
std::function<hipError_t(hipStreamCaptureMode*)> g_hipThreadExchangeStreamCaptureMode =
    DefaultThreadExchangeStreamCaptureMode;

HIP_FAKE hipError_t hipThreadExchangeStreamCaptureMode(hipStreamCaptureMode* mode) {
  return g_hipThreadExchangeStreamCaptureMode(mode);
}

// Not called from dev_runtime.cc's own text, which is why an audit of that file
// misses them -- they come from the CHECK macro bodies that expand into this
// same TU (rocmwrap.h:133,143,155 and checks.h:36,37,43). WARN is
// unconditional, so hipGetErrorString's argument is evaluated on every checked
// failure, which is most of the failure-path tests in this suite. Deliberately
// not seams: no test asserts on the text of an error message, and a fixed
// string keeps the WARN output readable.
HIP_FAKE const char* hipGetErrorString(hipError_t err) {
  return err == hipSuccess ? "hipSuccess" : "hipError (DevRuntimeTests fake)";
}

HIP_FAKE hipError_t hipGetLastError(void) { return hipSuccess; }

// ---------------------------------------------------------------------------
// Params are not cached here (see DevRuntimeTestsStubs.h), so the default just
// hands back the value the NCCL_PARAM declaration was written with.
static int64_t DefaultLoadParam(const char*, int64_t deftVal) { return deftVal; }
std::function<int64_t(const char*, int64_t)> g_loadParam = DefaultLoadParam;

void ResetDevRuntimeFakes() {
  g_hipMemGetAllocationPropertiesFromHandle = DefaultMemGetAllocationPropertiesFromHandle;
  g_hipMemExportToShareableHandle           = DefaultMemExportToShareableHandle;
  g_hipMemSetAccess                         = DefaultMemSetAccess;
  g_hipMemGetAllocationGranularity          = DefaultMemGetAllocationGranularity;
  g_hipStreamCreateWithFlags                = DefaultStreamCreateWithFlags;
  g_hipStreamSynchronize                    = DefaultStreamSynchronize;
  g_hipStreamDestroy                        = DefaultStreamDestroy;
  g_hipThreadExchangeStreamCaptureMode      = DefaultThreadExchangeStreamCaptureMode;
  g_hipMemImportFromShareableHandle         = DefaultMemImportFromShareableHandle;
  g_hipMemMap                               = DefaultMemMap;
  g_hipMemRelease                           = DefaultMemRelease;
  g_proxyClientGetFdBlocking                = DefaultProxyClientGetFdBlocking;
  g_hipMemAddressReserve                    = DefaultMemAddressReserve;
  g_bootstrapIntraNodeBarrier               = DefaultIntraNodeBarrier;
  g_bootstrapIntraNodeAllGather             = DefaultIntraNodeAllGather;
  g_hipMemAddressFree                       = DefaultMemAddressFree;
  g_hipMemUnmap                             = DefaultMemUnmap;
  g_bootstrapAllGather                      = DefaultAllGather;
  g_ginRegister                             = DefaultGinRegister;
  g_ginDeregister                           = DefaultGinDeregister;
  g_rmaProxyConnectOnce                     = DefaultRmaProxyConnectOnce;
  g_rmaProxyRegister                        = DefaultRmaProxyRegister;
  g_spaceAlloc                              = DefaultSpaceAlloc;
  g_spaceFree                               = DefaultSpaceFree;
  g_devrPopulateSegmentSizes                = DefaultPopulateSegmentSizes;
  g_shadowPoolAlloc                         = DefaultShadowPoolAlloc;
  g_shadowPoolFree                          = DefaultShadowPoolFree;
  g_shadowPoolToHost                        = DefaultShadowPoolToHost;
  g_intruAddressMapFind                     = DefaultIntruAddressMapFind;
  g_hipGetDevice                            = DefaultGetDevice;
  g_hipSetDevice                            = DefaultSetDevice;
  g_hipMemcpy                               = DefaultMemcpy;
  g_ncclCommWindowDeregister                = DefaultCommWindowDeregister;
  g_hipMemcpyAsync                          = DefaultMemcpyAsync;
  g_hipMemsetAsync                          = DefaultMemsetAsync;
  g_hipIpcGetMemHandle                      = DefaultIpcGetMemHandle;
  g_hipIpcOpenMemHandle                     = DefaultIpcOpenMemHandle;
  g_hipIpcCloseMemHandle                    = DefaultIpcCloseMemHandle;
  g_hipMemGetAddressRange                   = DefaultMemGetAddressRange;
  g_hipMemRetainAllocationHandle            = DefaultMemRetainAllocationHandle;
  g_ncclCommRegister                        = DefaultCommRegister;
  g_ncclCommDeregister                      = DefaultCommDeregister;
  g_rmaProxyDeregister                      = DefaultRmaProxyDeregister;
  g_devrAllocAndPopulateSegmentWindows      = DefaultDevrAllocAndPopulateSegmentWindows;
  g_loadParam                               = DefaultLoadParam;

  // The liveness set is state, not a hook, but it is just as capable of
  // outliving a test: anything that installs a non-freeing g_shadowPoolFree
  // leaves its entries behind, and DefaultShadowPoolToHost keeps honouring
  // them for the rest of the process. Clearing it keeps the header's "a test
  // cannot leak behaviour into the next one" true rather than nearly true.
  //
  // Cleared, not freed. Several fixtures free shadow-pool buffers themselves
  // in TearDown (they hold the only pointer to them once the window is gone),
  // so freeing here as well is a double free. What leaks is a bounded number
  // of small per-test buffers; what is fixed is the stale oracle.
  ShadowPoolLive().clear();
}
