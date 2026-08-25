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
#include <sched.h>

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
ncclResult_t initChannel(struct ncclComm* comm, int channelid) { ::abort(); }
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
ncclResult_t ncclMnnvlCheck(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclNetFinalize(struct ncclComm* comm) { return ncclSuccess; }
int ncclOsCpuCount(const ncclAffinity& affinity) { ::abort(); }
ncclResult_t ncclOsGetAffinity(ncclAffinity* affinity) { ::abort(); }
ncclResult_t ncclOsSetAffinity(const ncclAffinity& affinity) { ::abort(); }
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
int rcclGetTuningIndexForArch(const char* gfxarch) { ::abort(); }
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

extern "C" {
ncclResult_t ncclMemManagerDestroy(struct ncclComm*) { return ncclSuccess; }
// librocm-core: return non-VerSuccess (benign "version unknown") if ever reached.
int getROCmVersion(int*, int*, int*) { return 1; }
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

// WarpSpeed eligibility (rccl_wrap.cc), referenced by init.cc's willEnableWarpSpeed().
int64_t rcclParamWarpSpeedForceEnable() { ::abort(); }
bool rcclCanUseWarpSpeedAuto(struct ncclComm* comm, int nNodes) { ::abort(); }
