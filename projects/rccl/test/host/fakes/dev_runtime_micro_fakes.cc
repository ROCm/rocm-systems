/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * No-op host stubs for the dev_runtime micro-test binary.
 *
 * dev_runtime.cc is #included whole into dev-runtime-test.cc, which leaves
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
#ifdef ENABLE_ROCSHMEM_GIN
#include "gin/gin_host_anvil_sdma.h"
#endif
#include "rma/rma_proxy.h"
#include "nccl_device/core_tmp.h"
#include "nccl_device/lsa_barrier.h"
#include "nccl_device/gin_barrier.h"

#include "fakes/dev_runtime_micro_fakes.h"
#include "fakes/hip_fakes.h"   // shared HIP seams + InstallHipVmmEmulator()
#include "fakes/nccl_fakes.h"  // g_ncclProxyClientGetFdBlocking, ResetNcclFakes()

#include <fcntl.h>
#include <sys/mman.h>

#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <set>

// ---------------------------------------------------------------------------
// Globals the translation unit references.
// ---------------------------------------------------------------------------
// Use POSIX-FD handles so the single-rank success path takes the no-export /
// reuse-local branch in symMemory{Export,ImportAndMap}SegmentHandle (no real
// shareable-handle export/import needed).


// devcomm compat tables (defined in devcomm/devcomm_v*.cc in the real build).
// devcomm-test.cc supplies v22902 and v22907 by compiling their sources; the
// rest only need to exist for getNcclVersionCompat's table to link.
struct ncclDevCommCompat ncclDevCommCompat_v23000 = {};
struct ncclDevCommCompat ncclDevCommCompat_v23100 = {};

// ---------------------------------------------------------------------------
// Debug / error.
// ---------------------------------------------------------------------------
const char* ncclGetErrorString(ncclResult_t) { return "ncclSuccess"; }

// ---------------------------------------------------------------------------
// Bootstrap.
// ---------------------------------------------------------------------------
static ncclResult_t DefaultAllGather(void*, void*, int) { return ncclSuccess; }
std::function<ncclResult_t(void*, void*, int)> g_devrBootstrapAllGather = DefaultAllGather;

ncclResult_t bootstrapAllGather(void* bs, void* buf, int bytes) { return g_devrBootstrapAllGather(bs, buf, bytes); }
// Seam: ncclDevrWindowRegisterInGroup and ncclDevrCommCreateInternal both
// NCCLCHECKGOTO this, so their closing-barrier failure arms are only reachable
// by driving it.
static ncclResult_t DefaultBootstrapBarrier(void*, int, int, int) { return ncclSuccess; }
std::function<ncclResult_t(void*, int, int, int)> g_devrBootstrapBarrier = DefaultBootstrapBarrier;

ncclResult_t bootstrapBarrier(void* b, int rank, int nRanks, int tag) {
  return g_devrBootstrapBarrier(b, rank, nRanks, tag);
}
// Seams: symMemoryMapLsaTeam NCCLCHECKGOTOs both of these, so their failure
// arms are only reachable by driving them.
static ncclResult_t DefaultIntraNodeBarrier(void*, int*, int, int, int) { return ncclSuccess; }
std::function<ncclResult_t(void*, int*, int, int, int)> g_devrBootstrapIntraNodeBarrier = DefaultIntraNodeBarrier;

ncclResult_t bootstrapIntraNodeBarrier(void* bs, int* ranks, int self, int size, int tag) {
  return g_devrBootstrapIntraNodeBarrier(bs, ranks, self, size, tag);
}

static ncclResult_t DefaultIntraNodeAllGather(void*, int*, int, int, void*, int) { return ncclSuccess; }
std::function<ncclResult_t(void*, int*, int, int, void*, int)> g_devrBootstrapIntraNodeAllGather =
    DefaultIntraNodeAllGather;

ncclResult_t bootstrapIntraNodeAllGather(void* bs, int* ranks, int self, int size, void* buf, int bytes) {
  return g_devrBootstrapIntraNodeAllGather(bs, ranks, self, size, buf, bytes);
}

// ---------------------------------------------------------------------------
// Arg checks / comm readiness.
// ---------------------------------------------------------------------------
ncclResult_t PtrCheck(const void*, const char*, const char*) { return ncclSuccess; }
ncclResult_t CommCheck(struct ncclComm*, const char*, const char*) { return ncclSuccess; }
// Seam: NCCLCHECK'd on entry to ncclCommWindowRegister_impl, so the
// not-ready rejection is only reachable by driving it.


// ---------------------------------------------------------------------------
// Public registration API.
// ---------------------------------------------------------------------------
// Seams: ncclDevrWindowRegisterInGroup takes a local registration up front and
// releases it on every failure path, which is only observable by counting.
static ncclResult_t DefaultCommRegister(const ncclComm_t, void*, size_t, void** handle) {
  if (handle) *handle = reinterpret_cast<void*>(0x1234);
  return ncclSuccess;
}
std::function<ncclResult_t(const ncclComm_t, void*, size_t, void**)> g_devrNcclCommRegister = DefaultCommRegister;

ncclResult_t ncclCommRegister(const ncclComm_t comm, void* ptr, size_t size, void** handle) {
  return g_devrNcclCommRegister(comm, ptr, size, handle);
}

static ncclResult_t DefaultCommDeregister(const ncclComm_t, void*) { return ncclSuccess; }
std::function<ncclResult_t(const ncclComm_t, void*)> g_devrNcclCommDeregister = DefaultCommDeregister;

ncclResult_t ncclCommDeregister(const ncclComm_t comm, void* handle) {
  return g_devrNcclCommDeregister(comm, handle);
}
// Seam: ncclDevCommDestroy releases its resource window through the public
// wrapper, so the branch is only observable by counting.
static ncclResult_t DefaultCommWindowDeregister(ncclComm_t, ncclWindow_t) { return ncclSuccess; }
std::function<ncclResult_t(ncclComm_t, ncclWindow_t)> g_devrNcclCommWindowDeregister = DefaultCommWindowDeregister;

ncclResult_t ncclCommWindowDeregister(ncclComm_t comm, ncclWindow_t win) {
  return g_devrNcclCommWindowDeregister(comm, win);
}

// ---------------------------------------------------------------------------
// Group state machine.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Param loader.
// ---------------------------------------------------------------------------
int64_t ncclLoadParam(char const*, int64_t deftVal, int64_t, int64_t* cache, int8_t* noCache) {
  if (cache) *cache = deftVal;
  if (noCache) *noCache = 0;
  return deftVal;
}

// Emitted by init.cc, which this binary does not link, so the redirected
// NCCL_PARAM in dev_runtime.cc does not cover it. Routed through g_loadParam
// like the params dev_runtime.cc owns, so a test can drive the NCCL_WIN_ENABLE=0
// opt-out in ncclCommWindowRegister_impl.
int64_t ncclParamWinEnable() { return g_loadParam("WIN_ENABLE", 1); }

// ---------------------------------------------------------------------------
// Proxy.
// ---------------------------------------------------------------------------
// The real proxy hands back an fd the caller owns and closes. Returning -1
// would make the SYSCHECK on close() in symMemoryImportAndMapSegmentHandle fail
// and read like an import bug, so hand out a real descriptor the code can close.


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
std::function<ncclResult_t(struct ncclSpace*, int64_t, int64_t, int, int64_t*)> g_devrSpaceAlloc = DefaultSpaceAlloc;

ncclResult_t ncclSpaceAlloc(struct ncclSpace* sp, int64_t total, int64_t size, int align, int64_t* outOffset) {
  return g_devrSpaceAlloc(sp, total, size, align, outOffset);
}

static ncclResult_t DefaultSpaceFree(struct ncclSpace*, int64_t, int64_t) { return ncclSuccess; }
std::function<ncclResult_t(struct ncclSpace*, int64_t, int64_t)> g_devrSpaceFree = DefaultSpaceFree;

ncclResult_t ncclSpaceFree(struct ncclSpace* sp, int64_t offset, int64_t size) {
  return g_devrSpaceFree(sp, offset, size);
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
std::function<ncclResult_t(struct ncclShadowPool*, size_t, void**, void**, hipStream_t)> g_devrShadowPoolAlloc =
    DefaultShadowPoolAlloc;

ncclResult_t ncclShadowPoolAlloc(struct ncclShadowPool* pool, size_t size, void** outDevObj, void** outHostObj,
                                 hipStream_t stream) {
  return g_devrShadowPoolAlloc(pool, size, outDevObj, outHostObj, stream);
}

static ncclResult_t DefaultShadowPoolFree(struct ncclShadowPool*, void* devObj, hipStream_t) {
  ShadowPoolLive().erase(devObj);
  free(devObj);
  return ncclSuccess;
}
std::function<ncclResult_t(struct ncclShadowPool*, void*, hipStream_t)> g_devrShadowPoolFree = DefaultShadowPoolFree;

ncclResult_t ncclShadowPoolFree(struct ncclShadowPool* pool, void* devObj, hipStream_t stream) {
  return g_devrShadowPoolFree(pool, devObj, stream);
}

static ncclResult_t DefaultShadowPoolToHost(struct ncclShadowPool*, void* devObj, void** outHostObj) {
  if (ShadowPoolLive().count(devObj) == 0) return ncclInvalidArgument;  // freed or never ours
  if (outHostObj) *outHostObj = devObj;  // same buffer, see above
  return ncclSuccess;
}
std::function<ncclResult_t(struct ncclShadowPool*, void*, void**)> g_devrShadowPoolToHost = DefaultShadowPoolToHost;

ncclResult_t ncclShadowPoolToHost(struct ncclShadowPool* pool, void* devObj, void** outHostObj) {
  return g_devrShadowPoolToHost(pool, devObj, outHostObj);
}

// ---------------------------------------------------------------------------
// Intrusive address map.
// ---------------------------------------------------------------------------
// Seam: the window-map insert is NCCLCHECKGOTO'd at the deepest rollback in
// ncclDevrWindowRegisterInGroup, so that unwind needs this to be drivable.
static ncclResult_t DefaultIntruAddressMapInsert(struct ncclIntruAddressMap_untyped*, int, int, int, uintptr_t,
                                                void*) {
  return ncclSuccess;
}
std::function<ncclResult_t(struct ncclIntruAddressMap_untyped*, int, int, int, uintptr_t, void*)>
    g_devrIntruAddressMapInsert = DefaultIntruAddressMapInsert;

ncclResult_t ncclIntruAddressMapInsert_untyped(struct ncclIntruAddressMap_untyped* m, int a, int b, int c,
                                               uintptr_t k, void* v) {
  return g_devrIntruAddressMapInsert(m, a, b, c, k, v);
}
// Seam: findCommAndHostWindowFromDeviceWindow resolves a device window through
// this map, so every public pointer accessor needs it to answer.
static ncclResult_t DefaultIntruAddressMapFind(struct ncclIntruAddressMap_untyped*, int, int, int, uintptr_t,
                                               void** out) {
  if (out) *out = nullptr;
  return ncclSuccess;
}
std::function<ncclResult_t(struct ncclIntruAddressMap_untyped*, int, int, int, uintptr_t, void**)>
    g_devrIntruAddressMapFind = DefaultIntruAddressMapFind;

ncclResult_t ncclIntruAddressMapFind_untyped(struct ncclIntruAddressMap_untyped* map, int a, int b, int c,
                                             uintptr_t key, void** out) {
  return g_devrIntruAddressMapFind(map, a, b, c, key, out);
}
ncclResult_t ncclIntruAddressMapRemove_untyped(struct ncclIntruAddressMap_untyped*, int, int, int, uintptr_t) {
  return ncclSuccess;
}

// ---------------------------------------------------------------------------
// Memory stack spill lives in utils_fakes.cc, which owns src/misc/utils.cc's
// symbols and now links into this binary alongside this file. That version
// builds a reclaimable hunk chain rather than leaking each spill, so
// ncclMemoryStackDestruct (also there) actually frees them.
// ---------------------------------------------------------------------------

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
#ifdef ENABLE_ROCSHMEM_GIN
// Only referenced from ncclDevrCommCreateInternal's SDMA signal-binding block,
// which is itself behind ENABLE_ROCSHMEM_GIN. Copied from develop's
// test/DevRuntimeTestsStubs.cc (#10388), which still backs rccl-UnitTestsDevRuntime.
ncclResult_t ncclGinAnvilBindResourceWindowSignals(struct ncclComm*, void*, size_t, int, int) {
  return ncclSuccess;
}
#endif
// Seams: symMemoryRegisterGin NCCLCHECKs both, and its rollback path is only
// observable through the deregister count.
static ncclResult_t DefaultGinRegister(struct ncclComm*, void*, size_t, void*[NCCL_GIN_MAX_CONNECTIONS],
                                       ncclGinWindow_t[NCCL_GIN_MAX_CONNECTIONS], int, bool, int) {
  return ncclSuccess;
}
std::function<ncclResult_t(struct ncclComm*, void*, size_t, void*[NCCL_GIN_MAX_CONNECTIONS],
                           ncclGinWindow_t[NCCL_GIN_MAX_CONNECTIONS], int, bool, int)>
    g_devrGinRegister = DefaultGinRegister;

ncclResult_t ncclGinRegister(struct ncclComm* comm, void* addr, size_t size,
                             void* hostWins[NCCL_GIN_MAX_CONNECTIONS],
                             ncclGinWindow_t devWins[NCCL_GIN_MAX_CONNECTIONS], int winFlags, bool multiSegment,
                             int memType) {
  return g_devrGinRegister(comm, addr, size, hostWins, devWins, winFlags, multiSegment, memType);
}

static ncclResult_t DefaultGinDeregister(struct ncclComm*, void*[NCCL_GIN_MAX_CONNECTIONS]) { return ncclSuccess; }
std::function<ncclResult_t(struct ncclComm*, void*[NCCL_GIN_MAX_CONNECTIONS])> g_devrGinDeregister =
    DefaultGinDeregister;

ncclResult_t ncclGinDeregister(struct ncclComm* comm, void* hostWins[NCCL_GIN_MAX_CONNECTIONS]) {
  return g_devrGinDeregister(comm, hostWins);
}

// ---------------------------------------------------------------------------
// RMA proxy.
// ---------------------------------------------------------------------------
// Seams: symMemoryRegisterRma NCCLCHECKs both.
static ncclResult_t DefaultRmaProxyConnectOnce(struct ncclComm*) { return ncclSuccess; }
std::function<ncclResult_t(struct ncclComm*)> g_devrRmaProxyConnectOnce = DefaultRmaProxyConnectOnce;

ncclResult_t ncclRmaProxyConnectOnce(struct ncclComm* comm) { return g_devrRmaProxyConnectOnce(comm); }

static ncclResult_t DefaultRmaProxyRegister(struct ncclComm*, void*, size_t, void*[NCCL_GIN_MAX_CONNECTIONS]) {
  return ncclSuccess;
}
std::function<ncclResult_t(struct ncclComm*, void*, size_t, void*[NCCL_GIN_MAX_CONNECTIONS])> g_devrRmaProxyRegister =
    DefaultRmaProxyRegister;

ncclResult_t ncclRmaProxyRegister(struct ncclComm* comm, void* addr, size_t size,
                                  void* hostWins[NCCL_GIN_MAX_CONNECTIONS]) {
  return g_devrRmaProxyRegister(comm, addr, size, hostWins);
}

static ncclResult_t DefaultRmaProxyDeregister(struct ncclComm*, void*[NCCL_GIN_MAX_CONNECTIONS]) {
  return ncclSuccess;
}
std::function<ncclResult_t(struct ncclComm*, void*[NCCL_GIN_MAX_CONNECTIONS])> g_devrRmaProxyDeregister =
    DefaultRmaProxyDeregister;

ncclResult_t ncclRmaProxyDeregister(struct ncclComm* comm, void* wins[NCCL_GIN_MAX_CONNECTIONS]) {
  return g_devrRmaProxyDeregister(comm, wins);
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

// The segment cross-rank layout check and the GIN segment-info build, both
// reached from symMemoryRegisterGin and both real in dev_runtime_segments.cc.
// dev-runtime-test.cc compiles that source and routes these seams to it, so
// the no-op defaults here only serve the other suites in this binary, which
// never enable GIN and so never reach either call.
static ncclResult_t DefaultVerifySegmentLayouts(struct ncclDevrMemory*, struct ncclComm*) { return ncclSuccess; }
std::function<ncclResult_t(struct ncclDevrMemory*, struct ncclComm*)> g_devrVerifySegmentLayouts =
    DefaultVerifySegmentLayouts;

ncclResult_t ncclDevrVerifySegmentLayouts(struct ncclDevrMemory* mem, struct ncclComm* comm) {
  return g_devrVerifySegmentLayouts(mem, comm);
}

static ncclResult_t DefaultBuildGinSegmentInfos(struct ncclDevrMemory*) { return ncclSuccess; }
std::function<ncclResult_t(struct ncclDevrMemory*)> g_devrBuildGinSegmentInfos = DefaultBuildGinSegmentInfos;

ncclResult_t ncclDevrBuildGinSegmentInfos(struct ncclDevrMemory* mem) { return g_devrBuildGinSegmentInfos(mem); }

// ---------------------------------------------------------------------------
// Team accessors (host variants).
// ---------------------------------------------------------------------------
// Seams rather than fixed returns: ncclTeamRankIsMember/ncclTeamRankToTeam are
// real inline code that divides by the team's stride, so a zero-initialised
// ncclTeam_t here would SIGFPE the moment a test reached the symmetric arm of
// ncclDevrWorldToLsaRank. The defaults describe the contiguous, stride-1 team
// the comm already says it has; a test that needs a strided or offset team
// installs its own.
static ncclTeam_t DefaultTeamWorld(ncclComm_t comm) {
  if (comm == nullptr) return ncclTeam_t{0, 0, 1};
  return ncclTeam_t{comm->nRanks, comm->rank, 1};
}
std::function<ncclTeam_t(ncclComm_t)> g_devrTeamWorld = DefaultTeamWorld;
extern "C" ncclTeam_t ncclTeamWorld(ncclComm_t comm) { return g_devrTeamWorld(comm); }
extern "C" ncclTeam_t ncclTeamRail(ncclComm_t) { return ncclTeam_t{}; }

// The CFT teams, read by symMemoryObtain and ncclDevrCommCreateInternal. Same
// shape as the real host accessors in nccl_device/core.cc, reading the sizes
// ncclDevrInitOnce already computed, minus their ncclDevrInitOnce call: these
// are reached from inside that very function's callees, so calling it here
// would recurse. Stride 1 for the same SIGFPE reason as the world team above.
static ncclTeam_t DefaultTeamCft(ncclComm_t comm, ncclCftTeamMode_t) {
  if (comm == nullptr) return ncclTeam_t{0, 0, 1};
  return ncclTeam_t{comm->devrState.cftSize, comm->devrState.cftSelf, 1};
}
std::function<ncclTeam_t(ncclComm_t, ncclCftTeamMode_t)> g_devrTeamCft = DefaultTeamCft;
ncclTeam_t ncclTeamCft(ncclComm_t comm, ncclCftTeamMode_t mode) { return g_devrTeamCft(comm, mode); }

static ncclTeam_t DefaultTeamCftMultimem(ncclComm_t comm) {
  if (comm == nullptr) return ncclTeam_t{0, 0, 1};
  return ncclTeam_t{comm->devrState.cftMcSize, comm->devrState.cftMcSelf, 1};
}
std::function<ncclTeam_t(ncclComm_t)> g_devrTeamCftMultimem = DefaultTeamCftMultimem;
ncclTeam_t ncclTeamCftMultimem(ncclComm_t comm) { return g_devrTeamCftMultimem(comm); }

// ---------------------------------------------------------------------------
// CFT logical endpoints (real in cft_dev_runtime.cc).
// ---------------------------------------------------------------------------
// Sizes first: ncclDevrInitOnce stores both into devrState, so the defaults
// mirror the real pair's answer on a GPU without CFT support (gpuCftSupport
// below 13030), which is what a zero-initialised test comm reports.
static int DefaultComputeCftSize(struct ncclComm* comm) {
  return (comm != nullptr && comm->devrState.bigSize != 0) ? comm->devrState.cftSize : 1;
}
std::function<int(struct ncclComm*)> g_devrComputeCftSize = DefaultComputeCftSize;
int computeCftSize(struct ncclComm* comm) { return g_devrComputeCftSize(comm); }

static int DefaultComputeCftMcSize(struct ncclComm* comm) {
  return (comm != nullptr && comm->devrState.bigSize != 0) ? comm->devrState.cftMcSize : 1;
}
std::function<int(struct ncclComm*)> g_devrComputeCftMcSize = DefaultComputeCftMcSize;
int computeCftMcSize(struct ncclComm* comm) { return g_devrComputeCftMcSize(comm); }

// The bind/obtain half. Every call site is guarded by a live logical-endpoint
// id or an explicit CFT request, and a zero-initialised comm has neither, so
// these are link satisfiers on paths no suite in this binary reaches. Success
// rather than ::abort() so a future CFT test fails on its own assertion
// instead of taking the whole binary down.
ncclResult_t symBindTeamLe(struct ncclComm*, struct ncclDevrMemory*, ncclCftLeId) { return ncclSuccess; }
ncclResult_t symUnbindTeamLe(struct ncclComm*, struct ncclDevrMemory*, ncclCftLeId) { return ncclSuccess; }
ncclResult_t symTeamObtainUcLe(struct ncclComm*, struct ncclDevrTeam*, struct ncclDevrState*, bool*) {
  return ncclSuccess;
}
ncclResult_t symTeamObtainMcLe(struct ncclComm*, struct ncclDevrTeam*, struct ncclDevrState*, bool*) {
  return ncclSuccess;
}

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

// ncclRmaProxyEnabled is no longer faked here: the real predicate is linked into
// this binary, so a fake would be a duplicate. Tests drive its terms instead of
// its answer.

// Reached only once GIN is activated, which the GIN gate rejects for every
// comm this binary builds.
ncclResult_t ncclGinDevCommSetup(struct ncclComm*, struct ncclDevCommRequirements const*, struct ncclDevComm*,
                                 uint32_t) {
  return ncclSuccess;
}

// The enqueue-rearch job path: collective_stubs.cc pins
// ncclParamEnqueueRearchEnable to 0, so every call site takes the in-group task
// branch instead and nothing here enqueues a job.
ncclResult_t ncclMgmtTaskEnqueue(struct ncclAsyncJob*, ncclResult_t (*)(struct ncclAsyncJob*), void (*)(void*),
                                 ncclComm_t) {
  return ncclSuccess;
}

// ---------------------------------------------------------------------------
// Params are not cached here (see fakes/dev_runtime_micro_fakes.h), so the default just
// hands back the value the NCCL_PARAM declaration was written with.

void ResetDevRuntimeMicroFakes() {
  // The HIP surface now lives in fakes/hip_fakes.cc, shared with the other
  // micro suites. Reset it to the fail-loud defaults, then install the working
  // host-memory VMM this suite needs -- dev_runtime.cc's symmetric-memory paths
  // cannot run at all against a reserve that returns hipErrorInvalidValue.
  ResetHipFakes();
  InstallHipVmmEmulator();
  ResetNcclFakes();
  // symMemoryExportSegmentHandle hands the fd on to a real close(), so it must
  // be a genuine descriptor; the shared default reports failure instead.
  g_ncclProxyClientGetFdBlocking = [](struct ncclComm*, int, void*, int* fd) {
    if (fd) *fd = open("/dev/null", O_RDONLY);
    return ncclSuccess;
  };
  g_devrBootstrapIntraNodeBarrier               = DefaultIntraNodeBarrier;
  g_devrBootstrapIntraNodeAllGather             = DefaultIntraNodeAllGather;
  g_devrBootstrapAllGather                      = DefaultAllGather;
  g_devrGinRegister                             = DefaultGinRegister;
  g_devrGinDeregister                           = DefaultGinDeregister;
  g_devrRmaProxyConnectOnce                     = DefaultRmaProxyConnectOnce;
  g_devrRmaProxyRegister                        = DefaultRmaProxyRegister;
  g_devrSpaceAlloc                              = DefaultSpaceAlloc;
  g_devrSpaceFree                               = DefaultSpaceFree;
  g_devrPopulateSegmentSizes                = DefaultPopulateSegmentSizes;
  g_devrShadowPoolAlloc                         = DefaultShadowPoolAlloc;
  g_devrShadowPoolFree                          = DefaultShadowPoolFree;
  g_devrShadowPoolToHost                        = DefaultShadowPoolToHost;
  g_devrIntruAddressMapFind                     = DefaultIntruAddressMapFind;
  g_devrBootstrapBarrier                        = DefaultBootstrapBarrier;
  g_devrIntruAddressMapInsert                   = DefaultIntruAddressMapInsert;
  g_devrNcclCommWindowDeregister                = DefaultCommWindowDeregister;
  g_devrTeamWorld                               = DefaultTeamWorld;
  g_devrTeamCft                                 = DefaultTeamCft;
  g_devrTeamCftMultimem                         = DefaultTeamCftMultimem;
  g_devrComputeCftSize                          = DefaultComputeCftSize;
  g_devrComputeCftMcSize                        = DefaultComputeCftMcSize;
  g_devrNcclCommRegister                        = DefaultCommRegister;
  g_devrNcclCommDeregister                      = DefaultCommDeregister;
  g_devrRmaProxyDeregister                      = DefaultRmaProxyDeregister;
  g_devrAllocAndPopulateSegmentWindows      = DefaultDevrAllocAndPopulateSegmentWindows;
  g_devrVerifySegmentLayouts                = DefaultVerifySegmentLayouts;
  g_devrBuildGinSegmentInfos                = DefaultBuildGinSegmentInfos;

  // Not a hook either, but 12 tests assign it directly to steer the
  // POSIX-FD-vs-shareable-handle split in symMemory{Export,ImportAndMap}
  // SegmentHandle, and nothing put it back. Restore the value declared at the
  // top of this file.

  // The liveness set is state, not a hook, but it is just as capable of
  // outliving a test: anything that installs a non-freeing g_devrShadowPoolFree
  // leaves its entries behind, and DefaultShadowPoolToHost keeps honouring
  // them for the rest of the process. Clearing it keeps the header's "a test
  // cannot leak behaviour into the next one" true rather than nearly true.
  //
  // Frees, not just clears: this set is the single owner of every buffer
  // DefaultShadowPoolAlloc handed out and DefaultShadowPoolFree did not take
  // back. Fixtures must not free these themselves -- doing both is a double
  // free, and doing neither is what LeakSanitizer reports at exit.
  for (void* p : ShadowPoolLive()) free(p);
  ShadowPoolLive().clear();
}
