// RCCL_COMPAT_STUB: NCCL profiler plugin bindings stubbed out on AMD because
// upstream plugin/profiler.cc references comm fields (graphUsageMode,
// p2pCrossClique, ncclCeCollArgs::*) that do not exist in the RCCL fork.

#include "param.h"
#include "checks.h"
#include "comm.h"
#include "enqueue.h"
#include "utils.h"
#include "proxy.h"
#include "profiler.h"
#include "transport.h"
#include "plugin.h"

#include <cstdint>

thread_local ncclProfilerApiState_t ncclProfilerApiState{};
int ncclProfilerEventMask = 0;

ncclResult_t ncclProfilerPluginInit(struct ncclComm*) { return ncclSuccess; }
ncclResult_t ncclProfilerPluginFinalize(struct ncclComm*) { return ncclSuccess; }

ncclResult_t ncclProfilerStartGroupApiEvent(struct ncclInfo*, bool) { return ncclSuccess; }
ncclResult_t ncclProfilerStopGroupApiEvent() { return ncclSuccess; }
ncclResult_t ncclProfilerRecordGroupApiEventState(ncclProfilerEventState_t) { return ncclSuccess; }

ncclResult_t ncclProfilerStartP2pApiEvent(struct ncclInfo*, bool) { return ncclSuccess; }
ncclResult_t ncclProfilerStopP2pApiEvent() { return ncclSuccess; }

ncclResult_t ncclProfilerStartCollApiEvent(struct ncclInfo*, bool) { return ncclSuccess; }
ncclResult_t ncclProfilerStopCollApiEvent() { return ncclSuccess; }

ncclResult_t ncclProfilerStartKernelLaunchEvent(struct ncclKernelPlan*, cudaStream_t) { return ncclSuccess; }
ncclResult_t ncclProfilerStopKernelLaunchEvent(struct ncclKernelPlan*) { return ncclSuccess; }

ncclResult_t ncclProfilerStartGroupEvent(struct ncclKernelPlan*) { return ncclSuccess; }
ncclResult_t ncclProfilerStopGroupEvent(struct ncclKernelPlan*) { return ncclSuccess; }

ncclResult_t ncclProfilerStartTaskEvents(struct ncclKernelPlan*) { return ncclSuccess; }
ncclResult_t ncclProfilerStopTaskEvents(struct ncclKernelPlan*) { return ncclSuccess; }

ncclResult_t ncclProfilerStartProxyOpEvent(int, struct ncclProxyArgs*) { return ncclSuccess; }
ncclResult_t ncclProfilerStopProxyOpEvent(int, struct ncclProxyArgs*) { return ncclSuccess; }

ncclResult_t ncclProfilerStartSendProxyStepEvent(int, struct ncclProxyArgs*, int) { return ncclSuccess; }
ncclResult_t ncclProfilerStartRecvProxyStepEvent(int, struct ncclProxyArgs*, int) { return ncclSuccess; }
ncclResult_t ncclProfilerStopProxyStepEvent(int, struct ncclProxyArgs*, int) { return ncclSuccess; }

ncclResult_t ncclProfilerStartProxyCtrlEvent(void*, void**) { return ncclSuccess; }
ncclResult_t ncclProfilerStopProxyCtrlEvent(void*) { return ncclSuccess; }

ncclResult_t ncclProfilerStartKernelChEvent(struct ncclProxyArgs*, int, uint64_t) { return ncclSuccess; }
ncclResult_t ncclProfilerStopKernelChEvent(struct ncclProxyArgs*, int, uint64_t) { return ncclSuccess; }

ncclResult_t ncclProfilerRecordProxyOpEventState(int, struct ncclProxyArgs*, ncclProfilerEventState_t) { return ncclSuccess; }
ncclResult_t ncclProfilerRecordProxyStepEventState(int, struct ncclProxyArgs*, int, ncclProfilerEventState_t) { return ncclSuccess; }
ncclResult_t ncclProfilerRecordProxyCtrlEventState(void*, int, ncclProfilerEventState_t) { return ncclSuccess; }

ncclResult_t ncclProfilerAddPidToProxyOp(struct ncclProxyOp*) { return ncclSuccess; }

bool ncclProfilerNeedsProxy(struct ncclComm*, struct ncclProxyOp*) { return false; }
bool ncclProfilerPluginLoaded(void) { return false; }

ncclResult_t ncclProfilerCallback(void**, int, void*, int64_t, void*) { return ncclSuccess; }
