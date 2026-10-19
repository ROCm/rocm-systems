/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See collectives_micro_fakes.h.

#include "collectives_micro_fakes.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>

#include "debug.h"
#include "algorithms/gin/gin_all_reduce.h"
#include "algorithms/gin/gin_alltoall.h"

uint32_t ncclDebugLevelMask = NCCL_DEBUG_LEVEL_MASK_UNINITIALIZED;  // log everything
uint64_t ncclDebugMask = ~uint64_t(0);
thread_local int ncclDebugNoWarn = 0;
std::function<int64_t(const char*, int64_t)> g_loadParam;

namespace collectives_micro {

Fakes fake;

namespace {
ncclResult_t PopEnqueueResult() {
  if (fake.enqueueResults.empty()) return ncclSuccess;
  ncclResult_t res = fake.enqueueResults.front();
  fake.enqueueResults.pop_front();
  return res;
}
}  // namespace

ncclResult_t RecordAddon(const char* fn, const void* s, void* r, size_t n, ncclDataType_t t, ncclRedOp_t o,
                         ncclComm* c, hipStream_t st) {
  fake.trace.push_back(fn);
  fake.addons.push_back({fn, s, r, n, t, o, c, st});
  return fake.addonResult;
}

void RecordLaunch(const void* fn, dim3 grid, dim3 block, void** args, size_t shmem, hipStream_t stream,
                  hipEvent_t stop) {
  fake.trace.push_back("launch");
  fake.launches.push_back({fn, grid, block, shmem, stream, stop, *static_cast<const char**>(args[0]),
                           *static_cast<char**>(args[1]), *static_cast<size_t*>(args[2]),
                           *static_cast<int*>(args[3]), *static_cast<int*>(args[4])});
}

}  // namespace collectives_micro

using collectives_micro::fake;
using collectives_micro::RecordAddon;
using collectives_micro::PopEnqueueResult;

void ResetCollectivesMicroFakes() { fake = collectives_micro::Fakes{}; }

void ncclDebugLog(ncclDebugLogLevel level, unsigned long, const char*, int, const char* fmt, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  fake.log.push_back({level, buf});
}

ncclResult_t ncclEnqueueCheck(ncclInfo* info) {
  fake.trace.push_back(std::string("enqueue ") + info->opName);
  std::vector<size_t> sizes;
  if (info->sizes != nullptr) {
    const size_t n = info->comm->nRanks;
    sizes.assign(info->sizes, info->sizes + 4 * n * n);
  }
  fake.enqueued.push_back({*info, info->comm != nullptr && info->comm->enableDirectReduceScatter, sizes});
  return PopEnqueueResult();
}

ncclResult_t ncclParseCollConfig(const ncclCollConfig_t* config, ncclCollConfig_t* internal) {
  fake.parsedConfig = config;
  if (fake.parseResult != ncclSuccess) return fake.parseResult;
  *internal = *config;
  return ncclSuccess;
}


ncclResult_t rcclSelectAllReduce(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t,
                                 ncclRedOp_t op, hipStream_t stream, bool query, bool hint, rcclCollDecision* d) {
  fake.trace.push_back("select");
  fake.selects.push_back({"AllReduce", sendbuff, recvbuff, count, op, stream, query, hint});
  *d = fake.decision;
  return fake.selectResult;
}
ncclResult_t rcclSelectAllGather(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t,
                                 hipStream_t stream, bool query, bool hint, rcclCollDecision* d) {
  fake.trace.push_back("select");
  fake.selects.push_back({"AllGather", sendbuff, recvbuff, count, ncclNumOps, stream, query, hint});
  *d = fake.decision;
  return fake.selectResult;
}
ncclResult_t rcclSelectReduceScatter(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                     ncclDataType_t, ncclRedOp_t op, bool query, rcclCollDecision* d) {
  fake.trace.push_back("select");
  fake.selects.push_back({"ReduceScatter", sendbuff, recvbuff, count, op, nullptr, query, false});
  *d = fake.decision;
  return fake.selectResult;
}
ncclResult_t rcclSelectAlltoAll(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t,
                                hipStream_t stream, bool query, bool hint, rcclCollDecision* d) {
  fake.trace.push_back("select");
  fake.selects.push_back({"AlltoAll", sendbuff, recvbuff, count, ncclNumOps, stream, query, hint});
  *d = fake.decision;
  return fake.selectResult;
}

ncclResult_t rcclAddonLaunchBegin(ncclComm*, hipStream_t, rcclAddonLaunchState*) {
  fake.trace.push_back("begin");
  return fake.addonBeginResult;
}
ncclResult_t rcclAddonLaunchEnd(ncclComm*, hipStream_t, const rcclAddonLaunchState&, ncclResult_t res) {
  fake.trace.push_back("end");
  fake.addonEndSeen = res;
  return res;
}

#define FAKE_DDA(name)                                                                                   \
  ncclResult_t name(const void* s, void* r, size_t n, ncclDataType_t t, ncclComm* c, hipStream_t st) { \
    return RecordAddon(#name, s, r, n, t, ncclNumOps, c, st);                                           \
  }
#define FAKE_DDA_OP(name)                                                                                \
  ncclResult_t name(const void* s, void* r, size_t n, ncclDataType_t t, ncclRedOp_t o, ncclComm* c,      \
                    hipStream_t st) {                                                                    \
    return RecordAddon(#name, s, r, n, t, o, c, st);                                                     \
  }
FAKE_DDA(ncclAllGatherDdaFabricLL)
FAKE_DDA(ncclAllGatherDdaFabricLL128)
FAKE_DDA(ncclAllGatherDdaFabric)
FAKE_DDA(ncclAllGatherDdaIpc)
FAKE_DDA(ncclAllToAllDdaFabricLL)
FAKE_DDA(ncclAllToAllDdaFabricLL128)
FAKE_DDA(ncclAllToAllDdaFabric)
FAKE_DDA(ncclAllToAllDdaIpc)
FAKE_DDA_OP(ncclAllReduceDdaFabricLL)
FAKE_DDA_OP(ncclAllReduceDdaFabricLL128)
FAKE_DDA_OP(ncclAllReduceDdaFabric)
FAKE_DDA_OP(ncclAllReduceDdaIpc)
FAKE_DDA_OP(ncclReduceScatterDdaFabricLL)
FAKE_DDA_OP(ncclReduceScatterDdaFabricLL128)
FAKE_DDA_OP(ncclReduceScatterDdaFabric)
FAKE_DDA_OP(ncclReduceScatterDdaIpc)
#if defined(ENABLE_ROCSHMEM_GIN)
FAKE_DDA(ncclAllToAllGinSdma)
FAKE_DDA_OP(ncclAllReduceGinSdma)
#endif
#undef FAKE_DDA
#undef FAKE_DDA_OP

bool rcclDdaEnabled(const ncclComm*, size_t totalBytes, size_t threshold, bool query, const char*) {
  fake.ddaEnabledCalls.push_back({totalBytes, threshold, query});
  return fake.ddaEnabled;
}
size_t rcclDdaEntryThreshold(const ncclComm*, ncclFunc_t func) {
  fake.thresholdFuncs.push_back(func);
  return fake.ddaThreshold;
}
int64_t rcclParamDdaEnable() { return fake.ddaEnableParam; }
ncclResult_t rcclGetAlgoName(int, const char** algoName) {
  *algoName = fake.algoName;
  return ncclSuccess;
}
bool rcclUseAllGatherDirect(ncclComm* comm, size_t& msgSize) {
  return fake.useAgDirect && fake.useAgDirect(comm, msgSize);
}
bool rcclUseReduceScatterDirect(ncclComm* comm, size_t& msgSize) {
  return fake.useRsDirect && fake.useRsDirect(comm, msgSize);
}

bool ncclCeAlltoAllvEligible(ncclComm*, ncclDataType_t datatype, ncclSymRegType_t regType, bool sysmem,
                             bool capturing) {
  fake.ceEligCalls.push_back({datatype, regType, nullptr, nullptr, sysmem, capturing});
  return fake.ceElig;
}
bool ncclHierCeAlltoAllvEligible(ncclComm*, ncclDataType_t datatype, ncclSymRegType_t regType, ncclDevrWindow* sendWin,
                                 ncclDevrWindow* recvWin, bool sysmem, bool capturing) {
  fake.hierEligCalls.push_back({datatype, regType, sendWin, recvWin, sysmem, capturing});
  return fake.hierElig;
}

ncclResult_t ncclGroupStart() {
  fake.trace.push_back("groupStart");
  return fake.groupStartResult;
}
ncclResult_t ncclGroupEnd() {
  fake.trace.push_back("groupEnd");
  return fake.groupEndResult;
}
ncclResult_t ncclSend(const void* buff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t, hipStream_t) {
  fake.p2ps.push_back({true, buff, count, datatype, peer});
  return static_cast<int>(fake.p2ps.size()) == fake.p2pFailAt ? ncclInternalError : ncclSuccess;
}
ncclResult_t ncclRecv(void* buff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t, hipStream_t) {
  fake.p2ps.push_back({false, buff, count, datatype, peer});
  auto it = fake.peerMeta.find(peer);
  if (it != fake.peerMeta.end()) std::memcpy(buff, it->second.data(), it->second.size() * sizeof(size_t));
  return static_cast<int>(fake.p2ps.size()) == fake.p2pFailAt ? ncclInternalError : ncclSuccess;
}
ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t, int root, ncclComm_t,
                           hipStream_t) {
  fake.broadcasts.push_back({sendbuff, recvbuff, count, root});
  return ncclSuccess;
}
ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
  *count = comm->nRanks;
  return fake.commCountResult;
}
ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) {
  *rank = comm->rank;
  return fake.userRankResult;
}

ncclResult_t ncclProfilerStartCeCollEvent(ncclComm*, ncclCeCollArgs* args, hipStream_t) {
  fake.trace.push_back("profStart");
  fake.profArgs = *args;
  return fake.profStartResult;
}
ncclResult_t ncclProfilerStopCeCollEvent(ncclComm*, ncclCeCollArgs*, hipStream_t) {
  fake.trace.push_back("profStop");
  return ncclSuccess;
}
ncclResult_t ncclCeAllReduce(ncclComm*, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                             ncclRedOp_t op, hipStream_t, ncclDevrWindow* win, ncclCeCollArgs* args) {
  fake.trace.push_back("ceAllReduce");
  fake.ceArs.push_back({sendbuff, recvbuff, count, datatype, op, win, args});
  return fake.ceArResult;
}

