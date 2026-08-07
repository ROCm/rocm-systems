/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * See LICENSE.txt for license information
 ************************************************************************/
// Fail-loud stub floor for src/init.cc deep-path symbols (AICOMRCCL-1685).
// gc-sections retains ncclCommInitRankDev/ncclInit, whose bodies reference the
// whole bootstrap/topology/transport/nvls/gin/proxy surface. The shallower
// tests never CALL these (they return on validation arms), but the refs must LINK, so
// each is an abort-on-call stub. Mostly auto-generated from header decls; see
// gen_stubs.sh. Real fakes replace individual entries as deeper tests need them.
#include <cstdlib>
#include <cstdint>
#include <functional>
#include <sched.h>
#include <cstddef>
#include "nccl.h"
#include "os.h"   // ncclAffinity + ncclOs* declarations
struct ncclBootstrapHandle;
struct ncclComm;
struct ncclCudaContext;
struct ncclDevrWindow;
struct ncclStrongStream;
struct ncclTopoGraph;
struct ncclTopoSystem;
// --- auto-generated fail-loud stubs ---
ncclResult_t bootstrapAllGather(void* commState, void* allData, int size) { ::abort(); }
ncclResult_t bootstrapClose(void* commState) { ::abort(); }
ncclResult_t bootstrapCreateRoot(struct ncclBootstrapHandle* handle, bool idFromEnv) { ::abort(); }
ncclResult_t bootstrapGetUniqueId(struct ncclBootstrapHandle* handle, struct ncclComm* comm) { ::abort(); }
ncclResult_t bootstrapInit(int nHandles, void* handle, struct ncclComm* comm, struct ncclComm* parent) { ::abort(); }
ncclResult_t bootstrapIntraNodeBarrier(void* commState, int* ranks, int rank, int nranks, int tag) { ::abort(); }
ncclResult_t commSetUnrollFactor(struct ncclComm* comm) { ::abort(); }
ncclResult_t initChannel(struct ncclComm* comm, int channelid) { ::abort(); }
ncclResult_t ncclCeFinalize(struct ncclComm* comm) { return ncclSuccess; }
ncclResult_t ncclCheckMultiRank(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclCollNetChainBufferSetup(ncclComm_t comm) { ::abort(); }
ncclResult_t ncclCollNetDirectBufferSetup(ncclComm_t comm) { ::abort(); }
ncclResult_t ncclCollNetSetup(ncclComm_t comm, ncclComm_t parent, struct ncclTopoGraph* graphs[]) { ::abort(); }
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
ncclResult_t ncclGetUserP2pLevel(int* level) { ::abort(); }
ncclResult_t ncclGinFinalize(struct ncclComm* comm) { return ncclSuccess; }
ncclResult_t ncclGinHostFinalize(struct ncclComm* comm) { return ncclSuccess; }
ncclResult_t ncclInitKernelsForDevice(int cudaArch, int maxSharedMem, size_t* maxStackSize) { ::abort(); }
ncclResult_t ncclMnnvlCheck(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclNetFinalize(struct ncclComm* comm) { return ncclSuccess; }
ncclResult_t ncclNvlsBufferSetup(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclNvlsInit(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclNvlsSetup(struct ncclComm* comm, struct ncclComm* parent) { ::abort(); }
ncclResult_t ncclNvlsTreeConnect(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclNvlsTuning(struct ncclComm* comm) { ::abort(); }
int ncclOsCpuCount(const ncclAffinity& affinity) { ::abort(); }
ncclResult_t ncclOsGetAffinity(ncclAffinity* affinity) { ::abort(); }
ncclResult_t ncclOsSetAffinity(const ncclAffinity& affinity) { ::abort(); }
ncclResult_t ncclProfilerPluginFinalize(struct ncclComm* comm) { return ncclSuccess; }
ncclResult_t ncclProfilerPluginInit(struct ncclComm* comm) { ::abort(); }
void ncclProfilerProxyTraceDumpIfAny(void* profilerContext) { }
ncclResult_t ncclProxyCreate(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclProxyDestroy(struct ncclComm* comm) { return ncclSuccess; }
ncclResult_t ncclProxyShmUnlink(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclProxyStop(struct ncclComm* comm) { ::abort(); }
int ncclPxnDisable(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclRasCommFini(const struct ncclComm* comm) { return ncclSuccess; }
ncclResult_t ncclRegCleanup(struct ncclComm* comm) { return ncclSuccess; }
ncclResult_t ncclRmaProxyFinalize(struct ncclComm* comm) { return ncclSuccess; }
ncclResult_t ncclStrongStreamDestruct(struct ncclStrongStream* ss) { return ncclSuccess; }
ncclResult_t ncclSymkFinalize(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclTopoCheckNicFused(struct ncclComm* comm, bool* fused) { ::abort(); }
ncclResult_t ncclTopoCompute(struct ncclTopoSystem* system, struct ncclTopoGraph* graph) { ::abort(); }
ncclResult_t ncclTopoComputeCommCPU(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclTopoComputeP2pChannels(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclTopoComputeP2pChannelsPerPeer(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclTopoComputePaths(struct ncclTopoSystem* system, struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclTopoDumpGraphs(struct ncclTopoSystem* system, int ngraphs, struct ncclTopoGraph** graphs) { ::abort(); }
void ncclTopoFree(struct ncclTopoSystem* system) { ::abort(); }
ncclResult_t ncclTopoGetCpuAffinity(struct ncclTopoSystem* system, int rank, ncclAffinity* affinity) { ::abort(); }
ncclResult_t ncclTopoGetMinNetBw(struct ncclTopoSystem* system, int rank, float* bw) { ::abort(); }
ncclResult_t ncclTopoGetNvbGpus(struct ncclTopoSystem* system, int rank, int* nranks, int** ranks) { ::abort(); }
ncclResult_t ncclTopoGetPxnRanks(struct ncclComm* comm, int** intermediateRanks, int* nranks) { ::abort(); }
ncclResult_t ncclTopoGetSystem(struct ncclComm* comm, struct ncclTopoSystem** system, const char* dumpXmlFile) { ::abort(); }
ncclResult_t ncclTopoInitTunerConstants(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclTopoPathAllDirectNVLink(struct ncclTopoSystem* system, bool* allNvlinkConnected) { ::abort(); }
ncclResult_t ncclTopoPathAllNVLink(struct ncclTopoSystem* system, int* allNvLink) { ::abort(); }
ncclResult_t ncclTopoPrint(struct ncclTopoSystem* system) { ::abort(); }
ncclResult_t ncclTopoPrintGraph(struct ncclTopoSystem* system, struct ncclTopoGraph* graph) { ::abort(); }
ncclResult_t ncclTopoSearchInit(struct ncclTopoSystem* system) { ::abort(); }
ncclResult_t ncclTopoTrimSystem(struct ncclTopoSystem* system, struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclTopoTuneModel(struct ncclComm* comm, int minCompCap, int maxCompCap, struct ncclTopoGraph** graphs) { ::abort(); }
ncclResult_t ncclTransportPatConnect(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclTransportRingConnect(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclTransportTreeConnect(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclTreeBasePostset(struct ncclComm* comm, struct ncclTopoGraph* treeGraph) { ::abort(); }
ncclResult_t ncclTunerPluginLoad(struct ncclComm* comm) { ::abort(); }
ncclResult_t ncclTunerPluginUnload(struct ncclComm* comm) { ::abort(); }
ncclResult_t rcclCommSetP2pShiftSize(struct ncclComm* comm) { ::abort(); }
int rcclGetTuningIndexForArch(const char* gfxarch) { ::abort(); }
bool rcclUseAinic() { ::abort(); }
// --- manual stubs: complex signatures, extern-C APIs, data/TLS ---
struct ncclAsyncJob; struct ncclTopoRanks; struct ncclChannel;
ncclResult_t bootstrapSplit(unsigned long, struct ncclComm*, struct ncclComm*, int, int, int*) { ::abort(); }
ncclResult_t freeChannel(struct ncclChannel*, int, int, int, struct ncclComm*) { return ncclSuccess; }
ncclResult_t ncclAsyncLaunch(struct ncclAsyncJob*, ncclResult_t(*)(struct ncclAsyncJob*), void(*)(struct ncclAsyncJob*), void(*)(void*), struct ncclComm*) { ::abort(); }
ncclResult_t ncclTopoPostset(struct ncclComm*, int*, int*, struct ncclTopoRanks**, int*, struct ncclTopoGraph**, struct ncclComm*, int) { ::abort(); }
ncclResult_t ncclTransportCheckP2pType(struct ncclComm*, bool*, bool*, bool*) { ::abort(); }
ncclResult_t ncclTransportP2pConnect(struct ncclComm*, int, int, int*, int, int*, int) { ::abort(); }
ncclResult_t ncclTransportP2pSetup(struct ncclComm*, struct ncclTopoGraph*, int, bool*) { ::abort(); }
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

// Missed by the generator (init*/extern-C names, ref-to-array, std::function).
struct ncclTopoRanks;
ncclResult_t ncclTopoPreset(struct ncclComm*, struct ncclTopoGraph* (&)[7], struct ncclTopoRanks*) { ::abort(); }
ncclResult_t rcclCheckRomeTopoModelIdxConsensus(int, std::function<int(int)>,
                                                std::function<const char*(int)>,
                                                std::function<unsigned long(int)>) { ::abort(); }
extern "C" {
ncclResult_t ncclMemManagerDestroy(struct ncclComm*) { return ncclSuccess; }
// librocm-core: return non-VerSuccess (benign "version unknown") if ever reached.
int getROCmVersion(int*, int*, int*) { return 1; }
}
