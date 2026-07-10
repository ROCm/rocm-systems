/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
//
// No-op stub definitions for EnqueueTests2.cpp.
//
// EnqueueTests2.cpp #includes the hipified enqueue.cc directly so that the
// static function under test (addP2pToPlan) can be exercised on a CPU host.
// Compiling the whole translation unit drags in references to the rest of
// librccl. Because the test binary deliberately does NOT link librccl, this
// file provides trivial (no-op / error-returning) definitions for every symbol
// enqueue.cc references but does not itself define. The same RCCL headers are
// included so the declarations and types match exactly.

#include "hip/hip_runtime.h"

#include "enqueue.h"
#include "argcheck.h"
#include "channel.h"
#include "transport.h"
#include "register.h"
#include "register_inline.h"
#include "proxy.h"
#include "scheduler.h"
#include "graph.h"
#include "group.h"
#include "profiler.h"
#include "param.h"
#include "os.h"
#include "allocator.h"
#include "strongstream.h"
#include "archinfo.h"
#include "ce_coll.h"
#include "sym_kernels.h"
#include "dev_runtime.h"
#include "rccl_common.h"
#include "utils.h"
#include "roctx.h"
#include "device.h"
#include "rma/rma.h"
#include "rma/rma_proxy.h"
#include "latency_profiler/CollTraceFunc.h"

// ---------------------------------------------------------------------------
// Global variables
// ---------------------------------------------------------------------------
thread_local int ncclDebugNoWarn = 0;
thread_local int ncclGroupDepth = 0;
thread_local ncclResult_t ncclGroupError = ncclSuccess;
thread_local struct ncclComm* ncclGroupCommHead[ncclGroupTaskTypeNum] = {};
thread_local struct ncclComm* ncclGroupCommPreconnectHead = nullptr;
thread_local int ncclGroupBlocking = 0;
thread_local ncclProfilerApiState_t ncclProfilerApiState = {};
int ncclProfilerEventMask = 0;
int ncclCudaDriverVersionCache = 0;
bool ncclCudaLaunchBlocking = false;
struct ncclTransport netTransport = {};
const char* ncclAlgoStr[NCCL_NUM_ALGORITHMS] = {};
const char* ncclProtoStr[NCCL_NUM_PROTOCOLS] = {};
std::unordered_map<uint64_t, int> ncclDevFuncNameToId = {};

// ---------------------------------------------------------------------------
// Argument / comm checks
// ---------------------------------------------------------------------------
ncclResult_t ArgsCheck(struct ncclInfo* info) { return ncclSuccess; }
ncclResult_t CommCheck(struct ncclComm* ptr, const char* opname, const char* ptrname) { return ncclSuccess; }
ncclResult_t ncclCommEnsureReady(ncclComm_t comm) { return ncclSuccess; }
ncclResult_t ncclCommSetAsyncError(ncclComm_t comm, ncclResult_t nextState) { return ncclSuccess; }

// ---------------------------------------------------------------------------
// Scheduler / plan machinery
// ---------------------------------------------------------------------------
ncclResult_t ncclMakeSymmetricTaskList(struct ncclComm* comm, struct ncclTaskColl* task, struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next>* symTaskQueue, struct ncclTaskColl** remainTasksHead) { return ncclSuccess; }
ncclResult_t ncclSymmetricTaskScheduler(struct ncclComm* comm, struct ncclIntruQueue<struct ncclTaskColl, &ncclTaskColl::next>* symTaskQueue, struct ncclKernelPlan* plan) { return ncclSuccess; }
ncclResult_t ncclScheduleBcastTasksToPlan(struct ncclComm* comm, struct ncclKernelPlan* plan, struct ncclKernelPlanBudget* budget) { return ncclSuccess; }
ncclResult_t scheduleRmaTasksToPlan(struct ncclComm* comm, struct ncclKernelPlan* plan) { return ncclSuccess; }
ncclResult_t ncclRmaProxyReclaimPlan(struct ncclComm* comm, struct ncclKernelPlan* plan) { return ncclSuccess; }
ncclResult_t ncclProxySaveOp(struct ncclComm* comm, struct ncclProxyOp* proxyOp, bool* justInquire) { return ncclSuccess; }
ncclResult_t ncclProxyStart(struct ncclComm* comm) { return ncclSuccess; }

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------
ncclResult_t ncclRegisterCollBuffers(struct ncclComm* comm, struct ncclTaskColl* info, void* outRegBufSend[NCCL_MAX_LOCAL_RANKS], void* outRegBufRecv[NCCL_MAX_LOCAL_RANKS], struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue, bool* regNeedConnect) { return ncclSuccess; }
ncclResult_t ncclRegisterCollNvlsBuffers(struct ncclComm* comm, struct ncclTaskColl* info, void* outRegBufSend[NCCL_MAX_LOCAL_RANKS], void* outRegBufRecv[NCCL_MAX_LOCAL_RANKS], struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue, bool* regNeedConnect) { return ncclSuccess; }
ncclResult_t ncclRegisterP2pIpcBuffer(struct ncclComm* comm, void* userbuff, size_t size, int peerRank, int* regFlag, void** regAddr, struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue) { return ncclSuccess; }
ncclResult_t ncclRegisterP2pNetBuffer(struct ncclComm* comm, void* userbuff, size_t size, struct ncclConnector* conn, int* regFlag, void** handle, struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue) { return ncclSuccess; }
ncclResult_t ncclRegLocalIsValid(struct ncclReg* reg, bool* isValid) { if (isValid) *isValid = false; return ncclSuccess; }

// ---------------------------------------------------------------------------
// Profiler
// ---------------------------------------------------------------------------
bool ncclProfilerPluginLoaded(void) { return false; }
bool ncclProfilerProxyDiagEnabled(void) { return false; }
ncclResult_t ncclProfilerAddPidToProxyOp(struct ncclProxyOp* op) { return ncclSuccess; }
ncclResult_t ncclProfilerRecordGroupApiEventState(ncclProfilerEventState_t eState) { return ncclSuccess; }
ncclResult_t ncclProfilerStartGroupApiEvent(struct ncclInfo* info, bool isGraphCaptured) { return ncclSuccess; }
ncclResult_t ncclProfilerStopGroupApiEvent() { return ncclSuccess; }
ncclResult_t ncclProfilerStartCollApiEvent(struct ncclInfo* info, bool isGraphCaptured) { return ncclSuccess; }
ncclResult_t ncclProfilerStopCollApiEvent() { return ncclSuccess; }
ncclResult_t ncclProfilerStartP2pApiEvent(struct ncclInfo* info, bool isGraphCaptured) { return ncclSuccess; }
ncclResult_t ncclProfilerStopP2pApiEvent() { return ncclSuccess; }
ncclResult_t ncclProfilerStartGroupEvent(struct ncclKernelPlan* plan) { return ncclSuccess; }
ncclResult_t ncclProfilerStopGroupEvent(struct ncclKernelPlan* plan) { return ncclSuccess; }
ncclResult_t ncclProfilerStartTaskEvents(struct ncclKernelPlan* plan) { return ncclSuccess; }
ncclResult_t ncclProfilerStopTaskEvents(struct ncclKernelPlan* plan) { return ncclSuccess; }
ncclResult_t ncclProfilerStartKernelLaunchEvent(struct ncclKernelPlan* plan, hipStream_t stream) { return ncclSuccess; }
ncclResult_t ncclProfilerStopKernelLaunchEvent(struct ncclKernelPlan* plan) { return ncclSuccess; }

// ---------------------------------------------------------------------------
// Strong-stream / CUDA graph
// ---------------------------------------------------------------------------
ncclResult_t ncclStrongStreamAcquire(struct ncclCudaGraph graph, struct ncclStrongStream* ss, bool concurrent, hipStream_t* workStream) { return ncclSuccess; }
ncclResult_t ncclStrongStreamAcquiredWorkStream(struct ncclCudaGraph graph, struct ncclStrongStream* ss, bool concurrent, hipStream_t* workStream) { return ncclSuccess; }
ncclResult_t ncclStrongStreamRelease(struct ncclCudaGraph graph, struct ncclStrongStream* ss, bool concurrent) { return ncclSuccess; }
ncclResult_t ncclStreamWaitStream(hipStream_t a, hipStream_t b, hipEvent_t scratchEvent) { return ncclSuccess; }
ncclResult_t ncclStreamAdvanceToEvent(struct ncclCudaGraph g, hipStream_t s, hipEvent_t e) { return ncclSuccess; }
ncclResult_t ncclCudaGetCapturingGraph(struct ncclCudaGraph* graph, hipStream_t stream, int graphUsageMode) { return ncclSuccess; }
ncclResult_t ncclCudaGraphAddDestructor(struct ncclCudaGraph graph, hipHostFn_t fn, void* arg) { return ncclSuccess; }

// ---------------------------------------------------------------------------
// Group state
// ---------------------------------------------------------------------------
ncclResult_t ncclGroupStartInternal() { return ncclSuccess; }
ncclResult_t ncclGroupEndInternal(ncclSimInfo_t* simInfo) { return ncclSuccess; }

// ---------------------------------------------------------------------------
// dev_runtime windows / symmetric registration
// ---------------------------------------------------------------------------
ncclResult_t ncclDevrFindWindow(struct ncclComm* comm, void const* userPtr, struct ncclDevrWindow** outWin) { if (outWin) *outWin = nullptr; return ncclSuccess; }
bool ncclDevrWindowHasSysmemSegment(struct ncclDevrWindow* win) { return false; }
bool ncclDevrWindowIsMultiSegment(struct ncclDevrWindow* win) { return false; }
ncclResult_t ncclGetSymRegType(struct ncclDevrWindow* sendWin, struct ncclDevrWindow* recvWin, ncclSymRegType_t* winRegType) { return ncclSuccess; }
ncclResult_t ncclShadowPoolToHost(struct ncclShadowPool* pool, void* devObj, void** outHostObj) { if (outHostObj) *outHostObj = nullptr; return ncclSuccess; }
bool ncclCeAvailable(struct ncclComm* comm, ncclFunc_t coll, int red, ncclDataType_t ty, ncclSymRegType_t winRegType) { return false; }

// ---------------------------------------------------------------------------
// Misc utilities
// ---------------------------------------------------------------------------
bool IsArchMatch(char const* arch, char const* target) { return false; }
const char* ncclGetEnv(const char* name) { return nullptr; }
int64_t ncclLoadParam(char const* env, int64_t deftVal, int64_t uninitialized, int64_t* cache, int8_t* noCache) { if (cache) *cache = deftVal; return deftVal; }
void* ncclMemoryStack::allocateSpilled(struct ncclMemoryStack* me, size_t size, size_t align) { return nullptr; }
void* ncclOsAlignedAlloc(size_t alignment, size_t size) { return nullptr; }
void ncclOsAlignedFree(void* ptr) {}
ncclResult_t ncclTopoGetAlgoTime(struct ncclComm* comm, int coll, int algorithm, int protocol, size_t nBytes, int numPipeOps, float* time) { if (time) *time = 0.0f; return ncclSuccess; }
int ncclPxnDisable(struct ncclComm* comm) { return 1; }

int64_t ncclParamMaxNchannels() { return -2; }
int64_t ncclParamMinNchannels() { return -2; }

// ---------------------------------------------------------------------------
// RCCL-specific helpers
// ---------------------------------------------------------------------------
bool rcclUseAinic() { return false; }
bool rcclIsArchSupportedForFunc(struct ncclTaskColl* info, char const* archName) { return true; }
int64_t rcclParamDirectReduceScatterThreshold() { return 0; }
int64_t rcclParamPxnOptQpUsage() { return 0; }
ncclResult_t rcclOverrideAlgorithm(const char* ncclAlgoStr[], float table[][NCCL_NUM_PROTOCOLS], struct ncclTaskColl* info) { return ncclSuccess; }
ncclResult_t rcclOverrideProtocol(const char* ncclProtoStr[], float table[][NCCL_NUM_PROTOCOLS], struct ncclTaskColl* info) { return ncclSuccess; }
ncclResult_t rcclOverrideChannels(struct ncclComm* comm, ncclFunc_t coll, size_t nBytes, int& nc) { return ncclSuccess; }
void rcclUpdateCollectiveProtocol(struct ncclComm* comm, size_t const& nBytes, struct ncclTaskColl* info) {}
void rcclUpdateThreadThreshold(struct ncclComm* comm, size_t const& nBytes, struct ncclTaskColl* info, int& threadThreshold) {}
void rcclSetPipelining(struct ncclComm* comm, size_t const& nBytes, struct ncclTaskColl* info) {}
void rcclOptThreadBlockSize(struct ncclComm* comm, struct ncclTaskColl* info, size_t nBytes, int& nThreads) {}

// ---------------------------------------------------------------------------
// roctx
// ---------------------------------------------------------------------------
roctx_scoped_range_in::roctx_scoped_range_in(const char* message) noexcept {}
roctx_scoped_range_in::~roctx_scoped_range_in() noexcept {}

// ---------------------------------------------------------------------------
// Latency profiler
// ---------------------------------------------------------------------------
namespace latency_profiler {
std::unique_ptr<CollTraceEvent> collTraceAquireEventBaseline(ncclKernelPlan* plan, hipStream_t stream) { return nullptr; }
ncclResult_t collTraceRecordStartEvent(ncclComm* comm, hipStream_t launchStream, CollTraceEvent* event) { return ncclSuccess; }
ncclResult_t collTraceRecordEndEvent(ncclComm* comm, ncclKernelPlan* plan, hipStream_t launchStream, std::unique_ptr<CollTraceEvent> event) { return ncclSuccess; }
} // namespace latency_profiler
