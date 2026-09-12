/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fail-loud stub floor for the core nccl/rccl symbols, satisfying link-time symbol closure for host-only microtests.

// Some targets must omit an individual stub because their unit under test
// already defines that symbol. Each such stub is guarded by its own
// RCCL_STUBS_OMIT_<symbol> macro rather than one target-wide mode switch, so the
// exclusion names exactly what it drops.
//
// Note the limit of this: target_compile_definitions apply to EVERY source in
// the target, so a source added later still sees all of that target's omission
// macros. What the per-symbol scheme buys is legibility and a narrow blast
// radius per symbol -- not source-level isolation.
//
// An omit macro is ONLY for a symbol the unit under test itself defines. If a
// target instead needs a real VALUE where this floor aborts, that symbol wants a
// seam in the fakes file named after its owning production TU, which serves
// every target at once. rcclUseAinic was the counter-example and now lives in
// transport_stubs.cc.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <sched.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "nccl.h"
#include "nccl_fakes.h"  // g_loadParam, for the NCCL_PARAM defaults this floor stands in for
#include "os.h"

#include "nccl_stubs.h"

struct ncclAsyncJob;
struct ncclChannel;
struct ncclComm;
struct ncclCudaContext;
struct ncclDevrWindow;
struct ncclStrongStream;
struct ncclTopoGraph;

ncclResult_t commSetUnrollFactor(struct ncclComm* comm) { ::abort(); }
// This fake does NOT allocate ring->userRanks/rankToIndex like the real initChannel; callers must supply storage.
ncclResult_t g_initChannelResult = ncclSuccess;
int g_initChannelLastId = -1;
ncclResult_t initChannel(struct ncclComm* comm, int channelid) {
  g_initChannelLastId = channelid;
  return g_initChannelResult;
}

// Controllable (was hardcoded success). This is commFree's FIRST NCCLCHECK, so it doubles as the
// "commFree entered" marker for commCleanup's ordering oracle and as the only knob that fails commFree.
std::vector<std::string> g_cleanupCallOrder;
ncclResult_t g_ncclCeFinalizeResult = ncclSuccess;
ncclResult_t ncclCeFinalize(struct ncclComm* comm) {
  g_cleanupCallOrder.push_back("commFree");
  return g_ncclCeFinalizeResult;
}
ncclResult_t ncclCheckMultiRank(struct ncclComm* comm) { ::abort(); }
void ncclCudaContextDrop(struct ncclCudaContext* cxt) { ::abort(); }
// ncclCudaContextTrack and the rest of src/misc/strongstream.cc: strongstream_stubs.cc.
ncclResult_t ncclDdaFabricCommFini(struct ncclComm* comm) { return ncclSuccess; }
ncclResult_t ncclDdaFabricCommInit(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclDdaIpcCommFini(struct ncclComm* comm) { return ncclSuccess; }
ncclResult_t ncclDdaIpcCommInit(struct ncclComm* comm) { ::abort(); }
bool ncclDdaNranksRelaxEnabled() { return false; }
bool ncclDdaUseFabricPath(struct ncclComm* comm) { return false; }
ncclResult_t ncclDevrFinalize(struct ncclComm* comm) { return ncclSuccess; }
ncclResult_t ncclDevrFindWindow(struct ncclComm* comm, void const* userPtr, struct ncclDevrWindow** outWin) { ::abort(); }
bool ncclDevrIsOneLsaTeam(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclGinA2AFinalize(struct ncclComm* comm) { return ncclSuccess; }
ncclResult_t ncclGinAllReduceFinalize(struct ncclComm* comm) { return ncclSuccess; }
ncclResult_t ncclGinFinalize(struct ncclComm* comm) { return ncclSuccess; }
ncclResult_t ncclGinHostFinalize(struct ncclComm* comm) { return ncclSuccess; }
// Omitted when RCCL_STUBS_OMIT_ncclInitKernelsForDevice is defined -- the unit
// under test defines this itself (enqueue.cc:90).
#ifndef RCCL_STUBS_OMIT_ncclInitKernelsForDevice
static ncclResult_t DefaultNcclInitKernelsForDevice(int, int, size_t*) { ::abort(); }
std::function<ncclResult_t(int, int, size_t*)> g_ncclInitKernelsForDevice = DefaultNcclInitKernelsForDevice;
ncclResult_t ncclInitKernelsForDevice(int cudaArch, int maxSharedMem, size_t* maxStackSize) {
  return g_ncclInitKernelsForDevice(cudaArch, maxSharedMem, maxStackSize);
}
#endif
// Controllable (was fail-loud). initTransportsRank:1508 calls this only when the MNNVL scope test at :1507 passes,
// so the CALL COUNTER -- not the result -- is the oracle for that enable/auto/disable logic.
ncclResult_t g_ncclMnnvlCheckResult = ncclSuccess;
int g_ncclMnnvlCheckCalls = 0;
ncclResult_t ncclMnnvlCheck(struct ncclComm* comm) {
  g_ncclMnnvlCheckCalls++;
  return g_ncclMnnvlCheckResult;
}
ncclResult_t ncclNetFinalize(struct ncclComm* comm) { return ncclSuccess; }
// The src/os/*.cc entry points (ncclOsCpuCount, ncclOsGetAffinity, ncclOsSetAffinity,
// ncclOsTopoGetStrFromSys) and their seams: os_fakes.cc.
ncclResult_t ncclProfilerPluginFinalize(struct ncclComm* comm) { return ncclSuccess; }
ncclResult_t ncclProfilerPluginInit(struct ncclComm* comm) { ::abort(); }
// src/plugin/profiler.cc:871. Not fail-loud: ncclPrepareTasks:601 reaches this on
// a happy path, and "no profiler plugin loaded" is the truth for a host-only
// binary that links no plugin, not a steering choice.
bool ncclProfilerPluginLoaded(void) { return false; }
void ncclProfilerProxyTraceDumpIfAny(void* profilerContext) { }
ncclResult_t ncclRasCommFini(const struct ncclComm* comm) { return ncclSuccess; }
ncclResult_t ncclRegCleanup(struct ncclComm* comm) { return ncclSuccess; }
ncclResult_t ncclRmaInit(struct ncclComm* comm) { return ncclSuccess; }
ncclResult_t ncclRmaInitFromParent(struct ncclComm* comm, struct ncclComm* parent) { return ncclSuccess; }
ncclResult_t ncclRmaProxyFinalize(struct ncclComm* comm) { return ncclSuccess; }
// ncclStrongStreamDestruct and the rest of src/misc/strongstream.cc: strongstream_stubs.cc.
static ncclResult_t DefaultNcclSymkFinalize(struct ncclComm*) { return ncclSuccess; }
std::function<ncclResult_t(struct ncclComm*)> g_ncclSymkFinalize = DefaultNcclSymkFinalize;
ncclResult_t ncclSymkFinalize(struct ncclComm* comm) { return g_ncclSymkFinalize(comm); }
ncclResult_t ncclTunerPluginLoad(struct ncclComm* comm) { ::abort(); }
// Recording the comm matters: commCleanup forwards its own argument, so passing anything else would be invisible.
// TRAP: the recording must live here, not in the functor's default -- the default is reachable from the
// std::function's dynamic initializer, which --gc-sections cannot drop, so an extern this file does not
// itself define would become an undefined symbol in every other micro binary that links it.
struct ncclComm* g_ncclTunerPluginUnloadLastComm = nullptr;
static ncclResult_t DefaultNcclTunerPluginUnload(struct ncclComm*) { return ncclSuccess; }
std::function<ncclResult_t(struct ncclComm*)> g_ncclTunerPluginUnload = DefaultNcclTunerPluginUnload;
ncclResult_t ncclTunerPluginUnload(struct ncclComm* comm) {
  g_cleanupCallOrder.push_back("tunerUnload");
  g_ncclTunerPluginUnloadLastComm = comm;
  return g_ncclTunerPluginUnload(comm);
}
// src/rccl_wrap.cc symbols (rcclCommSetP2pShiftSize, rcclCanUseWarpSpeedAuto,
// rcclHierarchicalTempBufferSize, rcclParamWarpSpeedForceEnable,
// rcclParamHierarchicalAllGather, rcclParamHierarchicalReduceScatter):
// rccl_wrap_fakes.cc.
// rcclGetTuningIndexForArch (src/graph/tuning.cc): tuning_fakes.cc.
// rcclUseAinic (src/transport/net.cc): transport_stubs.cc.

ncclResult_t freeChannel(struct ncclChannel*, int, int, int, struct ncclComm*) { return ncclSuccess; }
static ncclResult_t DefaultNcclAsyncLaunch(struct ncclAsyncJob* job, ncclResult_t (*func)(struct ncclAsyncJob*),
                                           void (*undo)(struct ncclAsyncJob*), void (*destructor)(void*),
                                           struct ncclComm*) {
  ncclResult_t ret = func(job);
  if (ret != ncclSuccess && undo) undo(job);
  if (destructor) destructor(job);
  return ret;
}
std::function<ncclResult_t(struct ncclAsyncJob*, ncclResult_t (*)(struct ncclAsyncJob*),
                           void (*)(struct ncclAsyncJob*), void (*)(void*), struct ncclComm*)>
    g_ncclAsyncLaunch = DefaultNcclAsyncLaunch;
ncclResult_t ncclAsyncLaunch(struct ncclAsyncJob* job, ncclResult_t (*func)(struct ncclAsyncJob*),
                             void (*undo)(struct ncclAsyncJob*), void (*destructor)(void*), struct ncclComm* comm) {
  return g_ncclAsyncLaunch(job, func, undo, destructor, comm);
}
// Omitted when RCCL_STUBS_OMIT_ncclParamGraphStreamOrdering is defined -- the
// unit under test emits this via NCCL_PARAM (enqueue.cc:1986). Reads the env
// rather than a hardcoded 0, which forced config.graphStreamOrdering on every
// envConfigOverride call and hid the field.
#ifndef RCCL_STUBS_OMIT_ncclParamGraphStreamOrdering
int64_t ncclParamGraphStreamOrdering() {
  return g_loadParam("GRAPH_STREAM_ORDERING", NCCL_CONFIG_UNDEF_INT);
}
#endif
int64_t rcclParamPxnOptQpUsage() { ::abort(); }  // src/channel.cc:14
static ncclResult_t DefaultCollTraceDestroy(struct ncclComm*) { return ncclSuccess; }
std::function<ncclResult_t(struct ncclComm*)> g_collTraceDestroy = DefaultCollTraceDestroy;
namespace latency_profiler {
ncclResult_t collTraceInit(struct ncclComm*) { ::abort(); }
ncclResult_t collTraceDestroy(struct ncclComm* comm) { return g_collTraceDestroy(comm); }
}  // namespace latency_profiler

static ncclResult_t DefaultNcclCommDestroy(ncclComm_t) { return ncclSuccess; }
std::function<ncclResult_t(ncclComm_t)> g_ncclCommDestroy = DefaultNcclCommDestroy;
ncclResult_t ncclCommDestroy(ncclComm_t comm) { return g_ncclCommDestroy(comm); }
ncclResult_t ncclCommInitRank(ncclComm_t*, int, ncclUniqueId, int) { ::abort(); }
ncclResult_t ncclCommSplit(ncclComm_t, int, int, ncclComm_t*, ncclConfig_t*) { ::abort(); }
char ncclLastError[1024] = {};
thread_local int ncclGroupDepth = 0;
thread_local ncclResult_t ncclGroupError = ncclSuccess;
const char* rcclGitHash = "microtest";

// Read-only process state, deliberately NOT reset per test: nothing in a unit
// under test writes them and no test assigns them. Give one a seam the moment a
// test starts scripting it, because an unrestored global that a test DOES write
// is an order-dependent flake.
int ncclCudaDriverVersionCache = 12000;       // src/misc/cudawrap.cc
bool ncclCudaLaunchBlocking = false;          // src/misc/cudawrap.cc
int ncclProfilerEventMask = 0;                // src/profiler.cc
std::unordered_map<uint64_t, int> ncclDevFuncNameToId;  // generated device table

// Default 1 (!= VerSuccess) means "version unknown", so showVersion()'s runtime-ROCm block is skipped.
int g_getROCmVersionResult = 1;
unsigned int g_rocmVersionMajor = 0;
unsigned int g_rocmVersionMinor = 0;
unsigned int g_rocmVersionPatch = 0;

static ncclResult_t DefaultNcclMemFree(void*) { return ncclSuccess; }
std::function<ncclResult_t(void*)> g_ncclMemFree = DefaultNcclMemFree;

extern "C" {
ncclResult_t ncclMemManagerDestroy(struct ncclComm*) { return ncclSuccess; }
// librocm-core; signature matches rocm-core's, though rocm_version.h is not included in this TU.
int getROCmVersion(unsigned int* major, unsigned int* minor, unsigned int* patch) {
  if (major) *major = g_rocmVersionMajor;
  if (minor) *minor = g_rocmVersionMinor;
  if (patch) *patch = g_rocmVersionPatch;
  return g_getROCmVersionResult;
}
ncclResult_t ncclMemAlloc(void** ptr, size_t size) { ::abort(); }
ncclResult_t ncclMemFree(void* ptr) { return g_ncclMemFree(ptr); }
}

ncclResult_t ncclSymkInitOnce(struct ncclComm* comm) { ::abort(); }

void ResetNcclStubs() {
#ifndef RCCL_STUBS_OMIT_ncclInitKernelsForDevice
  g_ncclInitKernelsForDevice = DefaultNcclInitKernelsForDevice;
#endif
  g_ncclAsyncLaunch = DefaultNcclAsyncLaunch;
  g_ncclMemFree = DefaultNcclMemFree;
  g_ncclSymkFinalize = DefaultNcclSymkFinalize;
  g_ncclCommDestroy = DefaultNcclCommDestroy;
  g_collTraceDestroy = DefaultCollTraceDestroy;
  g_ncclTunerPluginUnload = DefaultNcclTunerPluginUnload;
  g_initChannelResult = ncclSuccess;
  g_initChannelLastId = -1;
  g_cleanupCallOrder.clear();
  g_ncclCeFinalizeResult = ncclSuccess;
  g_ncclMnnvlCheckResult = ncclSuccess;
  g_ncclMnnvlCheckCalls = 0;
  g_ncclTunerPluginUnloadLastComm = nullptr;
  g_getROCmVersionResult = 1;
  g_rocmVersionMajor = 0;
  g_rocmVersionMinor = 0;
  g_rocmVersionPatch = 0;
}
