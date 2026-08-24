/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fail-loud stub floor for the remaining core nccl/rccl symbols (comm
// lifecycle, device/context, subsystem init/finalize, tuner, data + TLS
// symbols, and the public nccl.h API), shared by host-only microtests. These
// satisfy a unit-under-test's link-time symbol closure; the shallower tests
// never call the abort-on-call entries, and benign teardown paths return
// ncclSuccess. A test that needs to drive one replaces that individual entry
// with a real fake. (Controllable, hookable seams live in nccl_fakes.cc.)

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <sched.h>
#include <string>
#include <vector>

#include "nccl.h"
#include "os.h"   // ncclAffinity + ncclOs* declarations

struct ncclAsyncJob;
struct ncclChannel;
struct ncclComm;
struct ncclCudaContext;
struct ncclDevrWindow;
struct ncclStrongStream;
struct ncclTopoGraph;

ncclResult_t commSetUnrollFactor(struct ncclComm* comm) { ::abort(); }
// Controllable (was a fail-loud abort): setupChannel()'s first statement is
// NCCLCHECK(initChannel(...)). Defined in init_fakes.cc alongside the other
// injectable seams; declared here rather than including init_fakes.h, which
// would drag the HIP/nccl fake headers into this stub TU. The fake deliberately
// does NOT allocate ring->userRanks/rankToIndex the way the real one does
// (channel.cc:61-62), so callers supply that storage.
extern ncclResult_t g_initChannelResult;
extern int g_initChannelLastId;
ncclResult_t initChannel(struct ncclComm* comm, int channelid) {
  g_initChannelLastId = channelid;  // so a test can see WHICH channel was asked for
  return g_initChannelResult;
}

ncclResult_t ncclCeFinalize(struct ncclComm* comm) { return ncclSuccess; }
ncclResult_t ncclCheckMultiRank(struct ncclComm* comm) { ::abort(); }
void ncclCudaContextDrop(struct ncclCudaContext* cxt) { ::abort(); }
ncclResult_t ncclCudaContextTrack(struct ncclCudaContext** out) { ::abort(); }
ncclResult_t ncclDdaFabricCommFini(struct ncclComm* comm) { return ncclSuccess; }
ncclResult_t ncclDdaFabricCommInit(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclDdaIpcCommFini(struct ncclComm* comm) { return ncclSuccess; }
ncclResult_t ncclDdaIpcCommInit(struct ncclComm* comm) { ::abort(); }
bool ncclDdaUseFabricPath(struct ncclComm* comm) { return false; }
ncclResult_t ncclDevrFinalize(struct ncclComm* comm) { return ncclSuccess; }
ncclResult_t ncclDevrFindWindow(struct ncclComm* comm, void const* userPtr, struct ncclDevrWindow** outWin) { ::abort(); }
bool ncclDevrIsOneLsaTeam(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclGinFinalize(struct ncclComm* comm) { return ncclSuccess; }
ncclResult_t ncclGinHostFinalize(struct ncclComm* comm) { return ncclSuccess; }
ncclResult_t ncclInitKernelsForDevice(int cudaArch, int maxSharedMem, size_t* maxStackSize) { ::abort(); }
// Controllable (was fail-loud). initTransportsRank:1508 calls this only when the MNNVL scope test at :1507 passes,
// so the CALL COUNTER -- not the result -- is the oracle for that enable/auto/disable logic.
extern ncclResult_t g_ncclMnnvlCheckResult;
extern int g_ncclMnnvlCheckCalls;
ncclResult_t ncclMnnvlCheck(struct ncclComm* comm) {
  g_ncclMnnvlCheckCalls++;
  return g_ncclMnnvlCheckResult;
}
ncclResult_t ncclNetFinalize(struct ncclComm* comm) { return ncclSuccess; }
// Controllable (was fail-loud). initTransportsRank's exit: block (:2403) calls ncclOsCpuCount on EVERY path, so
// nothing in that function is testable until this is seamed; the counter separates exit: from the :1488 bare return.
// Records the mask too: :1608 and exit::2403 both call this, and without the recorder either call site
// could be handed the wrong affinity (affinitySave instead of comm->cpuAffinity) with nothing noticing.
extern int g_ncclOsCpuCountValue;
extern int g_ncclOsCpuCountCalls;
extern std::vector<ncclAffinity> g_ncclOsCpuCountMasks;
int ncclOsCpuCount(const ncclAffinity& affinity) {
  g_ncclOsCpuCountCalls++;
  g_ncclOsCpuCountMasks.push_back(affinity);
  return g_ncclOsCpuCountValue;
}
// Controllable (was fail-loud). A std::function because :1609 writes through the pointer -- though nothing
// ever reads affinitySave back, which is what the AffinitySaveIsNeverRestored test pins.
extern std::function<ncclResult_t(ncclAffinity*)> g_ncclOsGetAffinity;
ncclResult_t ncclOsGetAffinity(ncclAffinity* affinity) { return g_ncclOsGetAffinity(affinity); }
// Controllable (was fail-loud). Records the affinity it was handed: without that, exit::2404 forwarding
// comm->cpuAffinity vs any other mask is unobservable -- a fake that drops an argument untests it.
// Keeps EVERY mask, not just the latest: :1610 and exit::2404 both call this, so a single "last"
// slot lets the exit: write mask what :1610 forwarded -- which left a mutant swapping :1610 to
// affinitySave alive. Tests index the call site they mean.
extern ncclResult_t g_ncclOsSetAffinityResult;
extern std::vector<ncclAffinity> g_ncclOsSetAffinityMasks;
ncclResult_t ncclOsSetAffinity(const ncclAffinity& affinity) {
  g_ncclOsSetAffinityMasks.push_back(affinity);
  return g_ncclOsSetAffinityResult;
}
// Early env/system read reached by ncclInit(). Return a plausible, non-empty,
// multi-token value: ncclInit() strtok_r()s the /proc/version read and then
// strstr()s the resulting token, which would segfault on an empty string
// (strtok_r("") -> NULL). This value is also not "1" and not the Hyper-V BIOS
// string, so the numa_balancing / bios_version branches stay on their benign
// arms.
ncclResult_t ncclOsTopoGetStrFromSys(const char* path, const char* fileName, char* strValue, int maxLen)
{
    if (strValue && maxLen > 0) {
        std::snprintf(strValue, maxLen, "Linux version 6.8.0-microtest");
    }
    return ncclSuccess;
}
ncclResult_t ncclProfilerPluginFinalize(struct ncclComm* comm) { return ncclSuccess; }
ncclResult_t ncclProfilerPluginInit(struct ncclComm* comm) { ::abort(); }
void ncclProfilerProxyTraceDumpIfAny(void* profilerContext) { }
ncclResult_t ncclRasCommFini(const struct ncclComm* comm) { return ncclSuccess; }
ncclResult_t ncclRegCleanup(struct ncclComm* comm) { return ncclSuccess; }
ncclResult_t ncclRmaInit(struct ncclComm* comm) { return ncclSuccess; }  // reached by commAlloc happy path
ncclResult_t ncclRmaInitFromParent(struct ncclComm* comm, struct ncclComm* parent) { return ncclSuccess; }
ncclResult_t ncclRmaProxyFinalize(struct ncclComm* comm) { return ncclSuccess; }
ncclResult_t ncclStrongStreamDestruct(struct ncclStrongStream* ss) { return ncclSuccess; }
ncclResult_t ncclSymkFinalize(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclTunerPluginLoad(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclTunerPluginUnload(struct ncclComm* comm) { ::abort(); }
ncclResult_t rcclCommSetP2pShiftSize(struct ncclComm* comm) { ::abort(); }
// Controllable (was fail-loud). Records gfxarch: :1577 forwards comm->archName into it and stores the
// result in comm->topo->tuning, so without the recorder `IndexForArch(archName)` -> `IndexForArch("")` is invisible.
extern int g_tuningIndexValue;
extern std::string g_tuningIndexLastArch;
int rcclGetTuningIndexForArch(const char* gfxarch) {
  g_tuningIndexLastArch = gfxarch ? gfxarch : "<null>";
  return g_tuningIndexValue;
}
bool rcclUseAinic() { ::abort(); }

// Complex signatures / extern-C APIs / data + TLS.
ncclResult_t freeChannel(struct ncclChannel*, int, int, int, struct ncclComm*) { return ncclSuccess; }
ncclResult_t ncclAsyncLaunch(struct ncclAsyncJob*, ncclResult_t(*)(struct ncclAsyncJob*), void(*)(struct ncclAsyncJob*), void(*)(void*), struct ncclComm*) { ::abort(); }
int64_t ncclParamGraphStreamOrdering() { return 0; }
int64_t rcclParamHierarchicalAllGather() { ::abort(); }
int64_t rcclParamPxnOptQpUsage() { ::abort(); }
namespace latency_profiler { ncclResult_t collTraceInit(struct ncclComm*) { ::abort(); } ncclResult_t collTraceDestroy(struct ncclComm*) { ::abort(); } }
ncclResult_t ncclCommDestroy(ncclComm_t) { ::abort(); }
ncclResult_t ncclCommInitRank(ncclComm_t*, int, ncclUniqueId, int) { ::abort(); }
ncclResult_t ncclCommSplit(ncclComm_t, int, int, ncclComm_t*, ncclConfig_t*) { ::abort(); }
char ncclLastError[1024] = {};
thread_local int ncclGroupDepth = 0;
thread_local ncclResult_t ncclGroupError = ncclSuccess;
const char* rcclGitHash = "microtest";

// showVersion() ROCm-version seam; defined in init_fakes.cc. Declared here for
// the same reason as g_initChannelResult above.
extern int g_getROCmVersionResult;
extern unsigned int g_rocmVersionMajor;
extern unsigned int g_rocmVersionMinor;
extern unsigned int g_rocmVersionPatch;

extern "C" {
ncclResult_t ncclMemManagerDestroy(struct ncclComm*) { return ncclSuccess; }
// librocm-core. Injectable so showVersion()'s runtime-ROCm arm (init.cc:1030) is
// reachable; g_getROCmVersionResult defaults to 1 (!= VerSuccess), preserving the
// original benign "version unknown". Signature matches rocm-core's
// (unsigned int*), which costs nothing here -- rocm_version.h is not included in
// this TU, so there is no declaration for it to conflict with.
int getROCmVersion(unsigned int* major, unsigned int* minor, unsigned int* patch) {
  if (major) *major = g_rocmVersionMajor;
  if (minor) *minor = g_rocmVersionMinor;
  if (patch) *patch = g_rocmVersionPatch;
  return g_getROCmVersionResult;
}
// Public nccl.h API reached only from the deep ncclCommInitRankFunc arm
// (comm->localSizes alloc); C linkage inherited from nccl.h above.
ncclResult_t ncclMemAlloc(void** ptr, size_t size) { ::abort(); }
ncclResult_t ncclMemFree(void* ptr) { ::abort(); }
}

// Deep-path symbols added to src/init.cc after PR #9783 branched (symmetric
// kernels + hierarchical reduce-scatter). Same unreached region as the abort
// floor above -- rcclParamHierarchicalReduceScatter mirrors the existing
// rcclParamHierarchicalAllGather stub, and its guarded block (and the temp-buffer
// size query within it) is never entered by the current tests.
ncclResult_t ncclSymkInitOnce(struct ncclComm* comm) { ::abort(); }
int64_t rcclParamHierarchicalReduceScatter() { ::abort(); }
size_t rcclHierarchicalTempBufferSize(int nNodes, bool allGather, bool reduceScatter) { ::abort(); }
