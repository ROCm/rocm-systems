/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/collectives.cc and the alltoallv_meta.h helpers it includes. Every
// dependency is seamed in this TU; ncclEnqueueCheck records the ncclInfo each entry point builds.

#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "comm.h"
#include "enqueue.h"
#include "rccl_common.h"
#include "ce_coll.h"
#include "dev_runtime.h"
#include "algorithms/gin/gin_all_reduce.h"
#include "algorithms/gin/gin_alltoall.h"

uint32_t ncclDebugLevelMask = NCCL_DEBUG_LEVEL_MASK_UNINITIALIZED;  // log everything
uint64_t ncclDebugMask = ~uint64_t(0);
thread_local int ncclDebugNoWarn = 0;
std::function<int64_t(const char*, int64_t)> g_loadParam;

namespace {

struct LogLine {
  ncclDebugLogLevel level;
  std::string text;
};

struct Enqueued {
  ncclInfo info;
  bool directReduceScatter;
  std::vector<size_t> sizes;  // info.sizes points at the caller's stack, so snapshot it
};

struct Recorded {
  rccl::rcclCall_t type;
  const void* sendbuff;
  void* recvbuff;
  size_t count;
  int root;
  const size_t* sendcounts;
  const size_t* rdispls;
};

struct Select {
  std::string coll;
  const void* sendbuff;
  void* recvbuff;
  size_t count;
  ncclRedOp_t op;
  hipStream_t stream;
  bool query;
  bool hint;
};

struct Addon {
  std::string fn;
  const void* sendbuff;
  void* recvbuff;
  size_t count;
  ncclDataType_t datatype;
  ncclRedOp_t op;
  ncclComm* comm;
  hipStream_t stream;
};

struct DdaEnabledCall {
  size_t totalBytes;
  size_t threshold;
  bool query;
};

struct SymkQuery {
  ncclFunc_t coll;
  int red;
  ncclDataType_t type;
  size_t nElts;
};

struct CeEligibility {
  ncclDataType_t datatype;
  ncclSymRegType_t regType;
  ncclDevrWindow* sendWin;
  ncclDevrWindow* recvWin;
  bool sysmem;
  bool capturing;
};

struct Memcpy {
  void* dst;
  const void* src;
  size_t bytes;
  hipMemcpyKind kind;
  hipStream_t stream;
};

struct P2p {
  bool send;
  const void* buff;
  size_t count;
  ncclDataType_t datatype;
  int peer;
};

struct Bcast {
  const void* sendbuff;
  void* recvbuff;
  size_t count;
  int root;
};

struct CeAr {
  const void* sendbuff;
  void* recvbuff;
  size_t count;
  ncclDataType_t datatype;
  ncclRedOp_t op;
  ncclDevrWindow* win;
  ncclCeCollArgs* args;
};

struct KernelLaunch {
  const void* fn;
  dim3 grid;
  dim3 block;
  size_t shmem;
  hipStream_t stream;
  hipEvent_t stopEvent;
  const char* src;
  char* dst;
  size_t rankOffset;
  int cols;
  int rows;
};

// Every seam's knobs and observations; SetUp resets them with `fake = Fakes{}`.
struct Fakes {
  std::vector<LogLine> log;
  std::vector<std::string> trace;  // dependency calls in issue order
  std::vector<Enqueued> enqueued;
  std::deque<ncclResult_t> enqueueResults;  // consumed per call; empty means success
  const ncclCollConfig_t* parsedConfig = nullptr;
  ncclResult_t parseResult = ncclSuccess;
  std::vector<Recorded> recorded;
  std::vector<bool> recorderSkips;
  ncclResult_t recordResult = ncclSuccess;
  std::vector<Select> selects;
  rcclCollDecision decision{NCCL_ALGO_RING};
  ncclResult_t selectResult = ncclSuccess;
  ncclResult_t addonBeginResult = ncclSuccess;
  ncclResult_t addonEndSeen = ncclInternalError;
  std::vector<Addon> addons;
  ncclResult_t addonResult = ncclSuccess;
  std::vector<DdaEnabledCall> ddaEnabledCalls;
  bool ddaEnabled = false;
  size_t ddaThreshold = 0;
  std::vector<ncclFunc_t> thresholdFuncs;
  int64_t ddaEnableParam = 1;
  const char* algoName = "ADDON";
  std::function<bool(ncclComm*, size_t)> useAgDirect;
  std::function<bool(ncclComm*, size_t)> useRsDirect;
  ncclResult_t symkInitResult = ncclSuccess;
  bool symkAvailable = true;
  std::vector<SymkQuery> symkQueries;
  std::map<const void*, ncclDevrWindow*> windows;
  const void* findWindowFailPtr = nullptr;
  std::set<ncclDevrWindow*> sysmemWindows;
  ncclSymRegType_t regType = ncclSymSendNonregRecvNonreg;
  ncclResult_t regTypeResult = ncclSuccess;
  std::vector<std::pair<ncclDevrWindow*, ncclDevrWindow*>> regTypeArgs;
  std::vector<CeEligibility> ceEligCalls;
  std::vector<CeEligibility> hierEligCalls;
  bool ceElig = false;
  bool hierElig = false;
  bool capturing = false;
  ncclResult_t captureResult = ncclSuccess;
  std::vector<int> captureModes;
  std::vector<size_t> memAllocSizes;
  std::vector<void*> memAllocs;
  int memAllocFailAt = 0;
  int currentDevice = 9;
  hipError_t getDeviceResult = hipSuccess;
  int setDeviceCalls = 0;
  int setDeviceFailAt = 0;
  std::vector<Memcpy> memcpys;
  std::deque<hipError_t> memcpyResults;
  hipError_t syncResult = hipSuccess;
  hipError_t lastError = hipSuccess;
  ncclResult_t groupStartResult = ncclSuccess;
  ncclResult_t groupEndResult = ncclSuccess;
  std::vector<P2p> p2ps;
  int p2pFailAt = 0;
  std::map<int, std::vector<size_t>> peerMeta;  // copied into an ncclRecv buffer from that peer
  std::vector<Bcast> broadcasts;
  ncclResult_t commCountResult = ncclSuccess;
  ncclResult_t userRankResult = ncclSuccess;
  ncclResult_t profStartResult = ncclSuccess;
  ncclCeCollArgs profArgs{};
  std::vector<CeAr> ceArs;
  ncclResult_t ceArResult = ncclSuccess;
  std::vector<uint8_t> peerFlags;
  std::vector<uint8_t> gatheredOwnFlag;
  ncclResult_t bootstrapResult = ncclSuccess;
  std::vector<KernelLaunch> launches;
  dim3 pushedGrid;
  dim3 pushedBlock;
  size_t pushedShmem = 0;
  hipStream_t pushedStream = nullptr;
};
Fakes fake;

ncclResult_t PopEnqueueResult() {
  if (fake.enqueueResults.empty()) return ncclSuccess;
  ncclResult_t res = fake.enqueueResults.front();
  fake.enqueueResults.pop_front();
  return res;
}

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

}  // namespace

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

namespace rccl {
Recorder::Recorder() {}
Recorder::~Recorder() {}
Recorder& Recorder::instance() {
  static Recorder recorder;
  return recorder;
}
void Recorder::skip(bool b) { fake.recorderSkips.push_back(b); }
ncclResult_t Recorder::record(rcclCall_t type, const ncclInfo& info) {
  fake.trace.push_back("record");
  fake.recorded.push_back({type, info.sendbuff, info.recvbuff, info.count, info.root, nullptr, nullptr});
  return fake.recordResult;
}
ncclResult_t Recorder::record(rcclCall_t type, const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t,
                              ncclComm_t, hipStream_t, int root, const size_t sendcounts[], const size_t[],
                              const size_t[], const size_t rdispls[]) {
  fake.trace.push_back("record");
  fake.recorded.push_back({type, sendbuff, recvbuff, count, root, sendcounts, rdispls});
  return fake.recordResult;
}
}  // namespace rccl

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

bool IsArchMatch(char const* arch, char const* target) {
  if (arch == nullptr || target == nullptr) return false;
  return std::strncmp(arch, target, std::strlen(target)) == 0;
}

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

ncclResult_t ncclSymkInitOnce(ncclComm*) { return fake.symkInitResult; }
bool ncclSymkAvailable(ncclComm*, ncclFunc_t coll, int red, ncclDataType_t ty, size_t nElts) {
  fake.symkQueries.push_back({coll, red, ty, nElts});
  return fake.symkAvailable;
}
ncclResult_t ncclDevrFindWindow(ncclComm*, void const* userPtr, ncclDevrWindow** outWin) {
  fake.trace.push_back("findWindow");
  if (userPtr != nullptr && userPtr == fake.findWindowFailPtr) return ncclInternalError;
  auto it = fake.windows.find(userPtr);
  *outWin = it == fake.windows.end() ? nullptr : it->second;
  return ncclSuccess;
}
bool ncclDevrWindowHasSysmemSegment(ncclDevrWindow* win) { return fake.sysmemWindows.count(win) != 0; }
ncclResult_t ncclGetSymRegType(ncclDevrWindow* sendWin, ncclDevrWindow* recvWin, ncclSymRegType_t* winRegType) {
  fake.regTypeArgs.push_back({sendWin, recvWin});
  *winRegType = fake.regType;
  return fake.regTypeResult;
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
ncclResult_t ncclCudaGetCapturingGraph(ncclCudaGraph* graph, hipStream_t, int graphUsageMode) {
  fake.captureModes.push_back(graphUsageMode);
  graph->graphId = fake.capturing ? 7 : ULLONG_MAX;
  return fake.captureResult;
}

ncclResult_t ncclMemAlloc(void** ptr, size_t size) {
  fake.trace.push_back("memAlloc");
  fake.memAllocSizes.push_back(size);
  if (static_cast<int>(fake.memAllocSizes.size()) == fake.memAllocFailAt) return ncclSystemError;
  *ptr = std::calloc(size, 1);
  fake.memAllocs.push_back(*ptr);
  return ncclSuccess;
}

hipError_t hipGetDevice(int* dev) {
  *dev = fake.currentDevice;
  return fake.getDeviceResult;
}
hipError_t hipSetDevice(int dev) {
  fake.trace.push_back("setDevice " + std::to_string(dev));
  if (++fake.setDeviceCalls == fake.setDeviceFailAt) return hipErrorInvalidDevice;
  fake.currentDevice = dev;
  return hipSuccess;
}
hipError_t hipMemcpyAsync(void* dst, const void* src, size_t bytes, hipMemcpyKind kind, hipStream_t stream) {
  fake.trace.push_back("memcpy");
  fake.memcpys.push_back({dst, src, bytes, kind, stream});
  if (dst == nullptr || src == nullptr) {
    ADD_FAILURE() << "hipMemcpyAsync with a null pointer";
    return hipErrorInvalidValue;
  }
  if (!fake.memcpyResults.empty()) {
    hipError_t res = fake.memcpyResults.front();
    fake.memcpyResults.pop_front();
    if (res != hipSuccess) return res;
  }
  std::memcpy(dst, src, bytes);
  return hipSuccess;
}
hipError_t hipStreamSynchronize(hipStream_t) {
  fake.trace.push_back("sync");
  return fake.syncResult;
}
hipError_t hipGetLastError() { return fake.lastError; }
const char* hipGetErrorString(hipError_t) { return "fake hip error"; }

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

ncclResult_t bootstrapAllGather(void*, void* allData, int size) {
  fake.trace.push_back("allGather");
  uint8_t* flags = static_cast<uint8_t*>(allData);
  for (size_t r = 0; r < fake.peerFlags.size(); r++) {
    if (flags[r] == 0 || flags[r] == 1) fake.gatheredOwnFlag.push_back(flags[r]);
  }
  if (fake.bootstrapResult != ncclSuccess) return fake.bootstrapResult;
  for (size_t r = 0; r < fake.peerFlags.size(); r++) {
    if (fake.peerFlags[r] != 0xff) flags[r] = fake.peerFlags[r];
  }
  return ncclSuccess;
}

// The UUT defines hierarchicalShuffle; the target embeds an empty device binary, so its registration
// and launches land here.
extern "C" void** __hipRegisterFatBinary(const void*) {
  static void* handle = nullptr;
  return &handle;
}
extern "C" void __hipRegisterFunction(void**, const void*, char*, const char*, unsigned int, uint3*, uint3*, dim3*,
                                      dim3*, int*) {}
extern "C" void __hipUnregisterFatBinary(void**) {}
// clang >= 23 registers per-kernel profile shadows under -fprofile-instr-generate; these keep the ROCm
// profile runtime out.
extern "C" void __hipRegisterVar(void**, char*, char*, const char*, int, size_t, int, int) {}
extern "C" __attribute__((weak)) void __llvm_profile_offload_register_shadow_variable(void*) {}
extern "C" __attribute__((weak)) void __llvm_profile_offload_register_section_shadow_variable(void*) {}
hipError_t __hipPushCallConfiguration(dim3 grid, dim3 block, size_t shmem, hipStream_t stream) {
  fake.pushedGrid = grid;
  fake.pushedBlock = block;
  fake.pushedShmem = shmem;
  fake.pushedStream = stream;
  return hipSuccess;
}
hipError_t __hipPopCallConfiguration(dim3* grid, dim3* block, size_t* shmem, hipStream_t* stream) {
  *grid = fake.pushedGrid;
  *block = fake.pushedBlock;
  *shmem = fake.pushedShmem;
  *stream = fake.pushedStream;
  return hipSuccess;
}
hipError_t hipLaunchKernel(const void* fn, dim3 grid, dim3 block, void** args, size_t shmem, hipStream_t stream) {
  RecordLaunch(fn, grid, block, args, shmem, stream, nullptr);
  return hipSuccess;
}
hipError_t hipExtLaunchKernel(const void* fn, dim3 grid, dim3 block, void** args, size_t shmem, hipStream_t stream,
                              hipEvent_t, hipEvent_t stop, int) {
  RecordLaunch(fn, grid, block, args, shmem, stream, stop);
  return hipSuccess;
}

#include "fakes/param_redirect.h"
#include "fakes/nvtx_redirect.h"

#include COLLECTIVES_CC_PATH

namespace {

constexpr int kRank = 1;
constexpr int kNRanks = 4;
constexpr int kCudaDev = 3;
constexpr size_t kCount = 32;

bool Logged(ncclDebugLogLevel level, const std::string& needle) {
  for (const LogLine& line : fake.log) {
    if (line.level == level && line.text.find(needle) != std::string::npos) return true;
  }
  return false;
}

class CollectivesMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    fake = Fakes{};
    g_loadParam = [this](const char* env, int64_t deftVal) {
      params_.push_back(env);
      return deftVal;
    };

    comm_ = MakeComm(kRank, kNRanks);
    comm_->nNodes = 1;
    comm_->cudaDev = kCudaDev;
    comm_->config.graphUsageMode = 2;
  }

  void TearDown() override {
    for (void* p : fake.memAllocs) std::free(p);
    fake.memAllocs.clear();
    g_loadParam = nullptr;
  }

  std::unique_ptr<ncclComm> MakeComm(int rank, int nRanks) {
    auto comm = std::make_unique<ncclComm>();
    comm->rank = rank;
    comm->nRanks = nRanks;
    comm->archName = arch_;
    return comm;
  }

  void SetArch(const char* arch) { std::snprintf(arch_, sizeof(arch_), "%s", arch); }

  const ncclInfo& Info(size_t i = 0) { return fake.enqueued.at(i).info; }

  ncclResult_t AllGather(size_t count = kCount) {
    return ncclAllGather_impl(send_.data(), recv_.data(), count, ncclFloat32, comm_.get(), stream_);
  }
  ncclResult_t AlltoAll(size_t count = kCount) {
    return ncclAlltoAll_impl(send_.data(), recv_.data(), count, ncclFloat32, comm_.get(), stream_);
  }
  ncclResult_t AllReduce(size_t count = kCount) {
    return ncclAllReduce_impl(send_.data(), recv_.data(), count, ncclFloat32, ncclProd, comm_.get(), stream_);
  }
  ncclResult_t ReduceScatter(size_t count = kCount) {
    return ncclReduceScatter_impl(send_.data(), recv_.data(), count, ncclFloat32, ncclMin, comm_.get(), stream_);
  }

  void Decide(int algo) { fake.decision.algo = algo; }

  // Each addon arm of one collective: bracketed by the launch prologue/epilogue, no enqueue.
  struct AddonArm {
    int algo;
    const char* fn;
  };
  void ExpectAddonArms(const std::vector<AddonArm>& arms, const std::function<ncclResult_t()>& call, ncclRedOp_t op) {
    for (const AddonArm& arm : arms) {
      SCOPED_TRACE(arm.fn);
      Decide(arm.algo);
      fake.trace.clear();
      fake.addons.clear();
      fake.enqueued.clear();
      fake.addonResult = ncclRemoteError;
      EXPECT_EQ(ncclRemoteError, call());
      EXPECT_EQ((std::vector<std::string>{"record", "select", "begin", arm.fn, "end"}), fake.trace);
      EXPECT_EQ(ncclRemoteError, fake.addonEndSeen);
      ASSERT_EQ(1u, fake.addons.size());
      EXPECT_EQ(send_.data(), fake.addons[0].sendbuff);
      EXPECT_EQ(recv_.data(), fake.addons[0].recvbuff);
      EXPECT_EQ(kCount, fake.addons[0].count);
      EXPECT_EQ(ncclFloat32, fake.addons[0].datatype);
      EXPECT_EQ(op, fake.addons[0].op);
      EXPECT_EQ(comm_.get(), fake.addons[0].comm);
      EXPECT_EQ(stream_, fake.addons[0].stream);
      EXPECT_TRUE(fake.enqueued.empty());
    }
    fake.addonResult = ncclSuccess;
  }

  char arch_[16] = "gfx942";
  std::unique_ptr<ncclComm> comm_;
  hipStream_t stream_ = reinterpret_cast<hipStream_t>(0x5a);
  std::vector<float> send_ = std::vector<float>(kCount * kNRanks);
  std::vector<float> recv_ = std::vector<float>(kCount * kNRanks);
  std::vector<std::string> params_;
};

// String helpers

TEST_F(CollectivesMicrotest, FuncToString_NamesEveryHostFunc) {
  const std::vector<std::pair<ncclFunc_t, const char*>> names = {
      {ncclFuncAllGather, "AllGather"}, {ncclFuncAllReduce, "AllReduce"},   {ncclFuncAlltoAll, "AlltoAll"},
      {ncclFuncAlltoAllv, "AlltoAllv"}, {ncclFuncBroadcast, "Broadcast"},   {ncclFuncGather, "Gather"},
      {ncclFuncRecv, "Recv"},           {ncclFuncReduce, "Reduce"},         {ncclFuncReduceScatter, "ReduceScatter"},
      {ncclFuncScatter, "Scatter"},     {ncclFuncSendRecv, "SendRecv"},     {ncclFuncSend, "Send"},
      {ncclFuncPutSignal, "PutSignal"}, {ncclFuncSignal, "Signal"},         {ncclFuncWaitSignal, "WaitSignal"},
      {ncclFuncAlltoAllPivot, "Invalid"}, {ncclNumFuncs, "Invalid"}};
  for (const auto& [fn, name] : names) EXPECT_STREQ(name, ncclFuncToString(fn)) << fn;
}

TEST_F(CollectivesMicrotest, DevRedOpToString_NamesEveryDeviceOp) {
  EXPECT_STREQ("Sum", ncclDevRedOpToString(ncclDevSum));
  EXPECT_STREQ("Prod", ncclDevRedOpToString(ncclDevProd));
  EXPECT_STREQ("MinMax", ncclDevRedOpToString(ncclDevMinMax));
  EXPECT_STREQ("PreMulSum", ncclDevRedOpToString(ncclDevPreMulSum));
  EXPECT_STREQ("SumPostDiv", ncclDevRedOpToString(ncclDevSumPostDiv));
  EXPECT_STREQ("Unknown", ncclDevRedOpToString(ncclNumDevRedOps));
}

// ncclUint8 has no case, so it logs as "Unknown".
TEST_F(CollectivesMicrotest, DatatypeToString_NamesEveryTypeButUint8) {
  const std::vector<std::pair<ncclDataType_t, const char*>> names = {
      {ncclInt8, "ncclInt8"},       {ncclInt32, "ncclInt32"},           {ncclUint32, "ncclUint32"},
      {ncclInt64, "ncclInt64"},     {ncclUint64, "ncclUint64"},         {ncclFloat16, "ncclFloat16"},
      {ncclFloat32, "ncclFloat32"}, {ncclFloat64, "ncclFloat64"},       {ncclBfloat16, "ncclBfloat16"},
      {ncclFloat8e4m3, "ncclFloat8e4m3"}, {ncclFloat8e5m2, "ncclFloat8e5m2"}, {ncclUint8, "Unknown"}};
  for (const auto& [type, name] : names) EXPECT_STREQ(name, ncclDatatypeToString(type)) << type;
}

TEST_F(CollectivesMicrotest, AlgoAndProtoToString_NameEveryValue) {
  EXPECT_STREQ("TREE", ncclAlgoToString(NCCL_ALGO_TREE));
  EXPECT_STREQ("RING", ncclAlgoToString(NCCL_ALGO_RING));
  EXPECT_STREQ("COLLNET_DIRECT", ncclAlgoToString(NCCL_ALGO_COLLNET_DIRECT));
  EXPECT_STREQ("COLLNET_CHAIN", ncclAlgoToString(NCCL_ALGO_COLLNET_CHAIN));
  EXPECT_STREQ("NVLS", ncclAlgoToString(NCCL_ALGO_NVLS));
  EXPECT_STREQ("NVLS_TREE", ncclAlgoToString(NCCL_ALGO_NVLS_TREE));
  EXPECT_STREQ("PAT", ncclAlgoToString(NCCL_ALGO_PAT));
  EXPECT_STREQ("Unknown", ncclAlgoToString(NCCL_NUM_ALGORITHMS));
  EXPECT_STREQ("LL", ncclProtoToString(NCCL_PROTO_LL));
  EXPECT_STREQ("LL128", ncclProtoToString(NCCL_PROTO_LL128));
  EXPECT_STREQ("SIMPLE", ncclProtoToString(NCCL_PROTO_SIMPLE));
  EXPECT_STREQ("Unknown", ncclProtoToString(NCCL_NUM_PROTOCOLS));
}

TEST_F(CollectivesMicrotest, TypeSize_CoversEveryType) {
  for (ncclDataType_t t : {ncclInt8, ncclUint8, ncclFloat8e4m3, ncclFloat8e5m2}) EXPECT_EQ(1, ncclTypeSize(t)) << t;
  for (ncclDataType_t t : {ncclFloat16, ncclBfloat16}) EXPECT_EQ(2, ncclTypeSize(t)) << t;
  for (ncclDataType_t t : {ncclInt32, ncclUint32, ncclFloat32}) EXPECT_EQ(4, ncclTypeSize(t)) << t;
  for (ncclDataType_t t : {ncclInt64, ncclUint64, ncclFloat64}) EXPECT_EQ(8, ncclTypeSize(t)) << t;
  EXPECT_EQ(-1, ncclTypeSize(ncclNumTypes));
}

TEST_F(CollectivesMicrotest, AlltoAllPivotEnable_ReadsItsEnvWithDefaultOff) {
  EXPECT_EQ(0, rcclParamAlltoAllPivotEnable());
  EXPECT_EQ(std::vector<std::string>{"RCCL_ALL_TO_ALL_PIVOT_ENABLE"}, params_);
}

// Dispatch helpers

TEST_F(CollectivesMicrotest, AllReduceDda_SymmetricEligibleNeverTakesDda) {
  fake.ddaEnabled = true;
  EXPECT_FALSE(rcclAllReduceShouldTakeDdaPath(comm_.get(), kCount, ncclFloat32, true, false, false));
  EXPECT_TRUE(fake.ddaEnabledCalls.empty());
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "AR DDA disqualified: symk eligible"));
}

TEST_F(CollectivesMicrotest, AllReduceDda_CeAllowedYieldsOffGfx1250) {
  fake.ddaEnabled = true;
  EXPECT_FALSE(rcclAllReduceShouldTakeDdaPath(comm_.get(), kCount, ncclFloat32, false, true, false));
  EXPECT_TRUE(fake.ddaEnabledCalls.empty());
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "ceAllReduceAllowed=1 on non-gfx1250"));
}

TEST_F(CollectivesMicrotest, AllReduceDda_Gfx1250FabricRunsEvenWhenCeIsAllowed) {
  SetArch("gfx1250");
  fake.ddaEnabled = true;
  fake.ddaThreshold = 4096;
  EXPECT_TRUE(rcclAllReduceShouldTakeDdaPath(comm_.get(), kCount, ncclFloat16, false, true, false));
  ASSERT_EQ(1u, fake.ddaEnabledCalls.size());
  EXPECT_EQ(kCount * 2, fake.ddaEnabledCalls[0].totalBytes);
  EXPECT_EQ(4096u, fake.ddaEnabledCalls[0].threshold);
  EXPECT_FALSE(fake.ddaEnabledCalls[0].query);
  EXPECT_EQ(std::vector<ncclFunc_t>{ncclFuncAllReduce}, fake.thresholdFuncs);
  EXPECT_TRUE(fake.log.empty());
}

TEST_F(CollectivesMicrotest, AllReduceDda_ExplainsADisabledDda) {
  fake.ddaEnableParam = 0;
  EXPECT_FALSE(rcclAllReduceShouldTakeDdaPath(comm_.get(), kCount, ncclFloat32, false, false, false));
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "AR DDA disqualified: RCCL_DDA_ENABLE=0"));
}

TEST_F(CollectivesMicrotest, AllReduceDda_ExplainsAMessageOverTheThreshold) {
  fake.ddaThreshold = 64;
  EXPECT_FALSE(rcclAllReduceShouldTakeDdaPath(comm_.get(), kCount, ncclFloat64, false, false, false));
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "AR DDA disqualified: msgBytes=256 > entryThreshold=64"));
}

TEST_F(CollectivesMicrotest, AllReduceDda_Gfx1250WithCeAllowedFallsToTheThresholdReason) {
  SetArch("gfx1250");
  EXPECT_FALSE(rcclAllReduceShouldTakeDdaPath(comm_.get(), kCount, ncclFloat32, false, true, false));
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "msgBytes=128 > entryThreshold=0"));
}

TEST_F(CollectivesMicrotest, AllReduceDda_QueryIsSilent) {
  EXPECT_FALSE(rcclAllReduceShouldTakeDdaPath(comm_.get(), kCount, ncclFloat32, true, false, true));
  EXPECT_TRUE(fake.log.empty());
}

TEST_F(CollectivesMicrotest, AllReduceDda_TakesDdaWhenNothingElseServesTheCall) {
  fake.ddaEnabled = true;
  EXPECT_TRUE(rcclAllReduceShouldTakeDdaPath(comm_.get(), kCount, ncclFloat32, false, false, false));
  EXPECT_TRUE(fake.log.empty());
}

TEST_F(CollectivesMicrotest, AlltoAllDda_YieldsToCe) {
  fake.ddaEnabled = true;
  EXPECT_FALSE(rcclAlltoAllShouldTakeDdaPath(comm_.get(), 1024, true));
  EXPECT_TRUE(fake.ddaEnabledCalls.empty());
}

TEST_F(CollectivesMicrotest, AlltoAllDda_AsksDdaWithTheAlltoAllThreshold) {
  fake.ddaThreshold = 2048;
  EXPECT_FALSE(rcclAlltoAllShouldTakeDdaPath(comm_.get(), 1024, false));
  fake.ddaEnabled = true;
  EXPECT_TRUE(rcclAlltoAllShouldTakeDdaPath(comm_.get(), 1024, false));
  ASSERT_EQ(2u, fake.ddaEnabledCalls.size());
  EXPECT_EQ(1024u, fake.ddaEnabledCalls[1].totalBytes);
  EXPECT_EQ(2048u, fake.ddaEnabledCalls[1].threshold);
  EXPECT_EQ(ncclFuncAlltoAll, fake.thresholdFuncs.at(1));
}

TEST_F(CollectivesMicrotest, CommIsSuspended_NeedsAMemoryManager) {
  EXPECT_FALSE(ncclCommIsSuspended(nullptr));
  ncclMemManagerTask task{};
  ncclIntruQueueEnqueue(&comm_->suspendTaskQueue, &task);
  EXPECT_FALSE(ncclCommIsSuspended(comm_.get()));
}

TEST_F(CollectivesMicrotest, CommIsSuspended_QueuedOrReleasedSuspendWithoutResume) {
  ncclMemManager manager{};
  comm_->memManager = &manager;
  EXPECT_FALSE(ncclCommIsSuspended(comm_.get()));
  manager.released = 1;
  EXPECT_TRUE(ncclCommIsSuspended(comm_.get()));
  manager.released = 0;
  ncclMemManagerTask suspend{};
  ncclIntruQueueEnqueue(&comm_->suspendTaskQueue, &suspend);
  EXPECT_TRUE(ncclCommIsSuspended(comm_.get()));
  ncclMemManagerTask resume{};
  ncclIntruQueueEnqueue(&comm_->resumeTaskQueue, &resume);
  EXPECT_FALSE(ncclCommIsSuspended(comm_.get()));
  manager.released = 1;
  EXPECT_FALSE(ncclCommIsSuspended(comm_.get()));
}

TEST_F(CollectivesMicrotest, MustUseEnqueuePath_NullCommCheckModeOrSuspend) {
  EXPECT_TRUE(rcclCollectiveMustUseEnqueuePath(nullptr));
  EXPECT_FALSE(rcclCollectiveMustUseEnqueuePath(comm_.get()));
  comm_->checkMode = ncclCheckModeDebugLocal;
  EXPECT_TRUE(rcclCollectiveMustUseEnqueuePath(comm_.get()));
  comm_->checkMode = ncclCheckModeDefault;
  ncclMemManager manager{};
  manager.released = 1;
  comm_->memManager = &manager;
  EXPECT_TRUE(rcclCollectiveMustUseEnqueuePath(comm_.get()));
}

TEST_F(CollectivesMicrotest, EnqueueSetupNeedsSwitch_OnlyForAlgosThatRewriteTheInfo) {
#ifdef ENABLE_ROCSHMEM
  constexpr bool kGdaRewrites = true;
#else
  constexpr bool kGdaRewrites = false;
#endif
  for (int algo = 0; algo < RCCL_ALGO_COUNT; algo++) {
    const bool rewrites =
        algo == RCCL_DIRECT_ALLGATHER || algo == RCCL_A2A_PIVOT || (kGdaRewrites && algo == RCCL_A2A_GDA);
    EXPECT_EQ(rewrites, rcclEnqueueSetupNeedsSwitch(algo)) << algo;
  }
}

TEST_F(CollectivesMicrotest, HierarchicalShuffleBlocks_StepAt64KiBAnd16MiB) {
  EXPECT_EQ(8, hierarchicalShuffleNumBlocks(64 << 10));
  EXPECT_EQ(16, hierarchicalShuffleNumBlocks((64 << 10) + 1));
  EXPECT_EQ(16, hierarchicalShuffleNumBlocks(16 << 20));
  EXPECT_EQ(32, hierarchicalShuffleNumBlocks((16 << 20) + 1));
}

// Symmetric-kernel request

class CollectivesSymmetricMicrotest : public CollectivesMicrotest {
 protected:
  void SetUp() override {
    CollectivesMicrotest::SetUp();
    comm_->symmetricSupport = 1;
    sendWin_.winFlags = NCCL_WIN_COLL_SYMMETRIC;
    recvWin_.winFlags = NCCL_WIN_COLL_SYMMETRIC;
    fake.windows[send_.data()] = &sendWin_;
    fake.windows[recv_.data()] = &recvWin_;
  }
  bool RequestedWin() {
    return isSymmetricKernelRequestedWin(comm_.get(), ncclFuncAllReduce, ncclDevSum, ncclFloat32, kCount, &sendWin_,
                                         &recvWin_);
  }
  bool Requested(bool agree) {
    return isSymmetricKernelRequested(comm_.get(), ncclFuncAllGather, ncclDevMinMax, ncclBfloat16, kCount,
                                      send_.data(), recv_.data(), agree);
  }
  ncclDevrWindow sendWin_{};
  ncclDevrWindow recvWin_{};
};

TEST_F(CollectivesSymmetricMicrotest, Win_BothWindowsSymmetricAndKernelAvailable) {
  EXPECT_TRUE(RequestedWin());
  ASSERT_EQ(1u, fake.symkQueries.size());
  EXPECT_EQ(ncclFuncAllReduce, fake.symkQueries[0].coll);
  EXPECT_EQ(ncclDevSum, fake.symkQueries[0].red);
  EXPECT_EQ(ncclFloat32, fake.symkQueries[0].type);
  EXPECT_EQ(kCount, fake.symkQueries[0].nElts);
}

TEST_F(CollectivesSymmetricMicrotest, Win_EachPrerequisiteIsRequired) {
  EXPECT_FALSE(isSymmetricKernelRequestedWin(nullptr, ncclFuncAllReduce, ncclDevSum, ncclFloat32, kCount, &sendWin_,
                                             &recvWin_));
  comm_->symmetricSupport = 0;
  EXPECT_FALSE(RequestedWin());
  comm_->symmetricSupport = 1;
  fake.symkInitResult = ncclInternalError;
  EXPECT_FALSE(RequestedWin());
  EXPECT_TRUE(fake.symkQueries.empty());
  fake.symkInitResult = ncclSuccess;
  fake.symkAvailable = false;
  EXPECT_FALSE(RequestedWin());
  fake.symkAvailable = true;
  EXPECT_FALSE(isSymmetricKernelRequestedWin(comm_.get(), ncclFuncAllReduce, ncclDevSum, ncclFloat32, kCount, nullptr,
                                             &recvWin_));
  EXPECT_FALSE(isSymmetricKernelRequestedWin(comm_.get(), ncclFuncAllReduce, ncclDevSum, ncclFloat32, kCount, &sendWin_,
                                             nullptr));
  sendWin_.winFlags = 0;
  EXPECT_FALSE(RequestedWin());
  sendWin_.winFlags = NCCL_WIN_COLL_SYMMETRIC;
  recvWin_.winFlags = 0;
  EXPECT_FALSE(RequestedWin());
}

TEST_F(CollectivesSymmetricMicrotest, Request_LooksUpBothBuffersAndAnswersLocally) {
  comm_->bootstrap = reinterpret_cast<void*>(0xb007);
  EXPECT_TRUE(Requested(false));
  EXPECT_EQ((std::vector<std::string>{"findWindow", "findWindow"}), fake.trace);
  ASSERT_EQ(1u, fake.symkQueries.size());
  EXPECT_EQ(ncclFuncAllGather, fake.symkQueries[0].coll);
  EXPECT_EQ(ncclDevMinMax, fake.symkQueries[0].red);
  EXPECT_EQ(ncclBfloat16, fake.symkQueries[0].type);
  fake.windows.erase(recv_.data());
  EXPECT_FALSE(Requested(false));
  EXPECT_FALSE(isSymmetricKernelRequested(nullptr, ncclFuncAllGather, ncclDevSum, ncclFloat32, kCount, send_.data(),
                                          recv_.data(), true));
}

TEST_F(CollectivesSymmetricMicrotest, Request_AgreementNeedsPeersAndABootstrap) {
  EXPECT_TRUE(Requested(true));  // no bootstrap
  comm_->bootstrap = reinterpret_cast<void*>(0xb007);
  comm_->nRanks = 1;
  EXPECT_TRUE(Requested(true));
  EXPECT_EQ(0, std::count(fake.trace.begin(), fake.trace.end(), "allGather"));
}

TEST_F(CollectivesSymmetricMicrotest, Request_AgreementIsUnanimous) {
  comm_->bootstrap = reinterpret_cast<void*>(0xb007);
  fake.peerFlags = {1, 0xff, 1, 1};
  EXPECT_TRUE(Requested(true));
  EXPECT_EQ((std::vector<uint8_t>{0, 1, 0, 0}), fake.gatheredOwnFlag);  // only this rank's slot is set
  fake.peerFlags = {1, 0xff, 1, 0};
  EXPECT_FALSE(Requested(true));
  fake.peerFlags = {0, 0xff, 1, 1};
  EXPECT_FALSE(Requested(true));
}

// A rank that cannot run the kernel still joins the allgather, reporting 0.
TEST_F(CollectivesSymmetricMicrotest, Request_IneligibleRankStillJoinsTheAgreement) {
  comm_->bootstrap = reinterpret_cast<void*>(0xb007);
  comm_->symmetricSupport = 0;
  fake.peerFlags = {1, 0xff, 1, 1};
  EXPECT_FALSE(Requested(true));
  EXPECT_EQ((std::vector<uint8_t>{0, 0, 0, 0}), fake.gatheredOwnFlag);
}

TEST_F(CollectivesSymmetricMicrotest, Request_FailedAgreementIsNo) {
  comm_->bootstrap = reinterpret_cast<void*>(0xb007);
  fake.peerFlags = {1, 0xff, 1, 1};
  fake.bootstrapResult = ncclRemoteError;
  EXPECT_FALSE(Requested(true));
}

// Config entry points: parse the config, then enqueue with the collective's fixed steps.

TEST_F(CollectivesMicrotest, ConfigEntryPoints_EnqueueTheParsedConfig) {
  ncclCollConfig_t config{};
  config.minCTAs = 7;
  struct Case {
    std::function<ncclResult_t()> call;
    ncclFunc_t coll;
    const char* name;
    ncclRedOp_t op;
    int root;
    int chunkSteps;
    int sliceSteps;
  };
  void* s = send_.data();
  void* r = recv_.data();
  ncclComm* c = comm_.get();
  const std::vector<Case> cases = {
      {[&] { return ncclAllGatherConfig(s, r, kCount, ncclFloat32, c, stream_, &config); }, ncclFuncAllGather,
       "AllGather", ncclSum, 0, ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS},
      {[&] { return ncclAlltoAllConfig(s, r, kCount, ncclFloat32, c, stream_, &config); }, ncclFuncAlltoAll,
       "AlltoAll", ncclSum, 0, ALLTOALL_CHUNKSTEPS, ALLTOALL_SLICESTEPS},
      {[&] { return ncclAllReduceConfig(s, r, kCount, ncclFloat32, ncclMax, c, stream_, &config); },
       ncclFuncAllReduce, "AllReduce", ncclMax, 0, ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS},
      {[&] { return ncclBroadcastConfig(s, r, kCount, ncclFloat32, 2, c, stream_, &config); }, ncclFuncBroadcast,
       "Broadcast", ncclSum, 2, BROADCAST_CHUNKSTEPS, BROADCAST_SLICESTEPS},
      {[&] { return ncclGatherConfig(s, r, kCount, ncclFloat32, 3, c, stream_, &config); }, ncclFuncGather, "Gather",
       ncclSum, 3, GATHER_CHUNKSTEPS, GATHER_SLICESTEPS},
      {[&] { return ncclReduceConfig(s, r, kCount, ncclFloat32, ncclProd, 2, c, stream_, &config); }, ncclFuncReduce,
       "Reduce", ncclProd, 2, REDUCE_CHUNKSTEPS, REDUCE_SLICESTEPS},
      {[&] { return ncclReduceScatterConfig(s, r, kCount, ncclFloat32, ncclMin, c, stream_, &config); },
       ncclFuncReduceScatter, "ReduceScatter", ncclMin, 0, REDUCESCATTER_CHUNKSTEPS, REDUCESCATTER_SLICESTEPS},
      {[&] { return ncclScatterConfig(s, r, kCount, ncclFloat32, 1, c, stream_, &config); }, ncclFuncScatter,
       "Scatter", ncclSum, 1, SCATTER_CHUNKSTEPS, SCATTER_SLICESTEPS},
  };
  for (const Case& tc : cases) {
    SCOPED_TRACE(tc.name);
    fake.enqueued.clear();
    fake.parsedConfig = nullptr;
    fake.enqueueResults = {ncclInvalidUsage};
    EXPECT_EQ(ncclInvalidUsage, tc.call());
    EXPECT_EQ(&config, fake.parsedConfig);
    ASSERT_EQ(1u, fake.enqueued.size());
    const ncclInfo& info = Info();
    EXPECT_EQ(tc.coll, info.coll);
    EXPECT_STREQ(tc.name, info.opName);
    EXPECT_EQ(s, info.sendbuff);
    EXPECT_EQ(r, info.recvbuff);
    EXPECT_EQ(kCount, info.count);
    EXPECT_EQ(ncclFloat32, info.datatype);
    EXPECT_EQ(tc.op, info.op);
    EXPECT_EQ(tc.root, info.root);
    EXPECT_EQ(c, info.comm);
    EXPECT_EQ(stream_, info.stream);
    EXPECT_EQ(tc.chunkSteps, info.chunkSteps);
    EXPECT_EQ(tc.sliceSteps, info.sliceSteps);
    EXPECT_EQ(7, info.collConfig.minCTAs);
    EXPECT_FALSE(info.decisionValid);
    EXPECT_TRUE(fake.recorded.empty());

    fake.enqueued.clear();
    fake.parseResult = ncclInvalidArgument;
    EXPECT_EQ(ncclInvalidArgument, tc.call());
    EXPECT_TRUE(fake.enqueued.empty());
    fake.parseResult = ncclSuccess;
  }
}

// Plain entry points: record the call, then enqueue it unchanged.

TEST_F(CollectivesMicrotest, PlainEntryPoints_RecordThenEnqueue) {
  struct Case {
    std::function<ncclResult_t()> call;
    rccl::rcclCall_t record;
    ncclFunc_t coll;
    const char* name;
    const void* sendbuff;
    void* recvbuff;
    ncclRedOp_t op;
    int root;
    int chunkSteps;
    int sliceSteps;
  };
  void* s = send_.data();
  void* r = recv_.data();
  ncclComm* c = comm_.get();
  const std::vector<Case> cases = {
      {[&] { return ncclBroadcast_impl(s, r, kCount, ncclFloat32, 2, c, stream_); }, rccl::rrBroadcast,
       ncclFuncBroadcast, "Broadcast", s, r, ncclSum, 2, BROADCAST_CHUNKSTEPS, BROADCAST_SLICESTEPS},
      {[&] { return ncclGather_impl(s, r, kCount, ncclFloat32, 3, c, stream_); }, rccl::rrGather, ncclFuncGather,
       "Gather", s, r, ncclSum, 3, GATHER_CHUNKSTEPS, GATHER_SLICESTEPS},
      {[&] { return ncclReduce_impl(s, r, kCount, ncclFloat32, ncclMax, 1, c, stream_); }, rccl::rrReduce,
       ncclFuncReduce, "Reduce", s, r, ncclMax, 1, REDUCE_CHUNKSTEPS, REDUCE_SLICESTEPS},
      {[&] { return ncclScatter_impl(s, r, kCount, ncclFloat32, 2, c, stream_); }, rccl::rrScatter, ncclFuncScatter,
       "Scatter", s, r, ncclSum, 2, SCATTER_CHUNKSTEPS, SCATTER_SLICESTEPS},
      // Send carries its buffer in recvbuff; the peer travels as root.
      {[&] { return ncclSend_impl(s, kCount, ncclFloat32, 3, c, stream_); }, rccl::rrSend, ncclFuncSend, "Send",
       nullptr, s, ncclSum, 3, 1, 1},
      {[&] { return ncclRecv_impl(r, kCount, ncclFloat32, 2, c, stream_); }, rccl::rrRecv, ncclFuncRecv, "Recv",
       nullptr, r, ncclSum, 2, 1, 1},
  };
  for (const Case& tc : cases) {
    SCOPED_TRACE(tc.name);
    fake.enqueued.clear();
    fake.recorded.clear();
    fake.trace.clear();
    fake.enqueueResults = {ncclInvalidUsage};
    EXPECT_EQ(ncclInvalidUsage, tc.call());
    EXPECT_EQ((std::vector<std::string>{"record", std::string("enqueue ") + tc.name}), fake.trace);
    ASSERT_EQ(1u, fake.recorded.size());
    EXPECT_EQ(tc.record, fake.recorded[0].type);
    EXPECT_EQ(kCount, fake.recorded[0].count);
    EXPECT_EQ(tc.root, fake.recorded[0].root);
    const ncclInfo& info = Info();
    EXPECT_EQ(tc.coll, info.coll);
    EXPECT_STREQ(tc.name, info.opName);
    EXPECT_EQ(tc.sendbuff, info.sendbuff);
    EXPECT_EQ(tc.recvbuff, info.recvbuff);
    EXPECT_EQ(kCount, info.count);
    EXPECT_EQ(ncclFloat32, info.datatype);
    EXPECT_EQ(tc.op, info.op);
    EXPECT_EQ(tc.root, info.root);
    EXPECT_EQ(c, info.comm);
    EXPECT_EQ(stream_, info.stream);
    EXPECT_EQ(tc.chunkSteps, info.chunkSteps);
    EXPECT_EQ(tc.sliceSteps, info.sliceSteps);
    EXPECT_EQ(nullptr, info.acc);
    EXPECT_FALSE(info.useDirect);

    fake.enqueued.clear();
    fake.recordResult = ncclSystemError;
    EXPECT_EQ(ncclSystemError, tc.call());
    EXPECT_TRUE(fake.enqueued.empty());
    fake.recordResult = ncclSuccess;
  }
}

TEST_F(CollectivesMicrotest, Bcast_RecordsInPlaceThenBroadcasts) {
  EXPECT_EQ(ncclSuccess, ncclBcast(recv_.data(), kCount, ncclInt8, 3, comm_.get(), stream_));
  ASSERT_EQ(1u, fake.recorded.size());
  EXPECT_EQ(rccl::rrBcast, fake.recorded[0].type);
  EXPECT_EQ(recv_.data(), fake.recorded[0].sendbuff);
  EXPECT_EQ(recv_.data(), fake.recorded[0].recvbuff);
  EXPECT_EQ(3, fake.recorded[0].root);
  ASSERT_EQ(1u, fake.broadcasts.size());
  EXPECT_EQ(recv_.data(), fake.broadcasts[0].sendbuff);
  EXPECT_EQ(recv_.data(), fake.broadcasts[0].recvbuff);
  EXPECT_EQ(kCount, fake.broadcasts[0].count);
  EXPECT_EQ(3, fake.broadcasts[0].root);

  fake.recordResult = ncclSystemError;
  EXPECT_EQ(ncclSystemError, ncclBcast(recv_.data(), kCount, ncclInt8, 3, comm_.get(), stream_));
  EXPECT_EQ(1u, fake.broadcasts.size());
}

TEST_F(CollectivesMicrotest, OneSidedEntryPoints_CarryTheWindowAndSignal) {
  ncclWindow_t win = reinterpret_cast<ncclWindow_t>(0x77);
  EXPECT_EQ(ncclSuccess,
            ncclPutSignal_impl(send_.data(), kCount, ncclInt32, 2, win, 64, 5, 1, 0x3u, comm_.get(), stream_));
  const ncclInfo& put = Info(0);
  EXPECT_EQ(ncclFuncPutSignal, put.coll);
  EXPECT_STREQ("PutSignal", put.opName);
  EXPECT_EQ(send_.data(), put.sendbuff);
  EXPECT_EQ(nullptr, put.recvbuff);
  EXPECT_EQ(kCount, put.count);
  EXPECT_EQ(ncclInt32, put.datatype);
  EXPECT_EQ(2, put.root);
  EXPECT_EQ(64u, put.peerWinOffset);
  EXPECT_EQ(win, put.peerWin);
  EXPECT_EQ(5, put.sigIdx);
  EXPECT_EQ(1, put.ctx);
  EXPECT_EQ(0x3u, put.flags);
  EXPECT_EQ(1, put.chunkSteps);
  EXPECT_EQ(1, put.sliceSteps);

  EXPECT_EQ(ncclSuccess, ncclSignal_impl(3, 6, 2, 0x5u, comm_.get(), stream_));
  const ncclInfo& sig = Info(1);
  EXPECT_EQ(ncclFuncSignal, sig.coll);
  EXPECT_STREQ("Signal", sig.opName);
  EXPECT_EQ(0u, sig.count);
  EXPECT_EQ(ncclInt8, sig.datatype);
  EXPECT_EQ(3, sig.root);
  EXPECT_EQ(nullptr, sig.peerWin);
  EXPECT_EQ(6, sig.sigIdx);
  EXPECT_EQ(2, sig.ctx);
  EXPECT_EQ(0x5u, sig.flags);

  ncclWaitSignalDesc_t descs[2] = {};
  fake.enqueueResults = {ncclInProgress};
  EXPECT_EQ(ncclInProgress, ncclWaitSignal_impl(2, descs, comm_.get(), stream_));
  const ncclInfo& wait = Info(2);
  EXPECT_EQ(ncclFuncWaitSignal, wait.coll);
  EXPECT_STREQ("WaitSignal", wait.opName);
  EXPECT_EQ(ncclInt32, wait.datatype);
  EXPECT_EQ(0, wait.root);
  EXPECT_EQ(2, wait.nDesc);
  EXPECT_EQ(descs, wait.signalDescs);
  EXPECT_EQ(comm_.get(), wait.comm);
  EXPECT_EQ(stream_, wait.stream);
  EXPECT_TRUE(fake.recorded.empty());
}

TEST_F(CollectivesMicrotest, AllReduceWithBias_RejectsANullAccumulator) {
  EXPECT_EQ(ncclInvalidArgument,
            ncclAllReduceWithBias_impl(send_.data(), recv_.data(), kCount, ncclFloat32, ncclSum, comm_.get(), stream_,
                                       nullptr));
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "acc cannot be nullptr"));
  EXPECT_TRUE(fake.recorded.empty());
  EXPECT_TRUE(fake.enqueued.empty());
}

TEST_F(CollectivesMicrotest, AllReduceWithBias_EnqueuesTheAccumulatorWithoutADecision) {
  float acc[kCount] = {};
  EXPECT_EQ(ncclSuccess, ncclAllReduceWithBias_impl(send_.data(), recv_.data(), kCount, ncclFloat32, ncclMax,
                                                    comm_.get(), stream_, acc));
  ASSERT_EQ(1u, fake.recorded.size());
  EXPECT_EQ(rccl::rrAllReduceWithBias, fake.recorded[0].type);
  const ncclInfo& info = Info();
  EXPECT_EQ(ncclFuncAllReduce, info.coll);
  EXPECT_EQ(acc, info.acc);
  EXPECT_EQ(ncclMax, info.op);
  EXPECT_EQ(ALLREDUCE_CHUNKSTEPS, info.chunkSteps);
  EXPECT_EQ(ALLREDUCE_SLICESTEPS, info.sliceSteps);
  EXPECT_FALSE(info.decisionValid);
  EXPECT_TRUE(fake.selects.empty());

  fake.recordResult = ncclSystemError;
  EXPECT_EQ(ncclSystemError, ncclAllReduceWithBias_impl(send_.data(), recv_.data(), kCount, ncclFloat32, ncclMax,
                                                        comm_.get(), stream_, acc));
  EXPECT_EQ(1u, fake.enqueued.size());
}

// gfx950 single-node comms use one chunk and one slice step; other arches only shrink the slices.
TEST_F(CollectivesMicrotest, SliceSteps_FollowArchAndOneSlice) {
  struct Case {
    const char* arch;
    int oneSlice;
    int arChunk, arSlice, agChunk, agSlice, rsChunk, rsSlice;
  };
  const std::vector<Case> cases = {
      {"gfx942", 0, ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS, ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS,
       REDUCESCATTER_CHUNKSTEPS, REDUCESCATTER_SLICESTEPS},
      {"gfx942", 1, ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS_SINGLE_NODE, ALLGATHER_CHUNKSTEPS,
       ALLGATHER_SLICESTEPS_SINGLE_NODE, REDUCESCATTER_CHUNKSTEPS, REDUCESCATTER_SLICESTEPS_SINGLE_NODE},
      {"gfx950", 0, ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS, ALLGATHER_CHUNKSTEPS, ALLGATHER_SLICESTEPS,
       REDUCESCATTER_CHUNKSTEPS, REDUCESCATTER_SLICESTEPS},
      {"gfx950", 1, 1, 1, 1, 1, 1, 1},
  };
  float acc[kCount] = {};
  for (const Case& tc : cases) {
    SCOPED_TRACE(std::string(tc.arch) + " oneSlice=" + std::to_string(tc.oneSlice));
    SetArch(tc.arch);
    comm_->rcclUseOneSlice = tc.oneSlice;
    fake.enqueued.clear();
    ASSERT_EQ(ncclSuccess, ncclAllReduce_impl(send_.data(), recv_.data(), kCount, ncclFloat32, ncclSum, comm_.get(),
                                              stream_));
    ASSERT_EQ(ncclSuccess, ncclAllReduceWithBias_impl(send_.data(), recv_.data(), kCount, ncclFloat32, ncclSum,
                                                      comm_.get(), stream_, acc));
    ASSERT_EQ(ncclSuccess, ncclAllGather_impl(send_.data(), recv_.data(), kCount, ncclFloat32, comm_.get(), stream_));
    ASSERT_EQ(ncclSuccess, ncclReduceScatter_impl(send_.data(), recv_.data(), kCount, ncclFloat32, ncclSum,
                                                  comm_.get(), stream_));
    ASSERT_EQ(4u, fake.enqueued.size());
    EXPECT_EQ(tc.arChunk, Info(0).chunkSteps);
    EXPECT_EQ(tc.arSlice, Info(0).sliceSteps);
    EXPECT_EQ(tc.arChunk, Info(1).chunkSteps);
    EXPECT_EQ(tc.arSlice, Info(1).sliceSteps);
    EXPECT_EQ(tc.agChunk, Info(2).chunkSteps);
    EXPECT_EQ(tc.agSlice, Info(2).sliceSteps);
    EXPECT_EQ(tc.rsChunk, Info(3).chunkSteps);
    EXPECT_EQ(tc.rsSlice, Info(3).sliceSteps);
  }
}

// AllGather dispatch

TEST_F(CollectivesMicrotest, AllGather_SelectsOnceAndEnqueuesTheDecision) {
  fake.decision = {NCCL_ALGO_RING, NCCL_PROTO_LL, 3, true, true};
  fake.enqueueResults = {ncclInProgress};
  EXPECT_EQ(ncclInProgress, AllGather());
  EXPECT_EQ((std::vector<std::string>{"record", "select", "enqueue AllGather"}), fake.trace);
  ASSERT_EQ(1u, fake.recorded.size());
  EXPECT_EQ(rccl::rrAllGather, fake.recorded[0].type);
  ASSERT_EQ(1u, fake.selects.size());
  EXPECT_EQ("AllGather", fake.selects[0].coll);
  EXPECT_EQ(send_.data(), fake.selects[0].sendbuff);
  EXPECT_EQ(recv_.data(), fake.selects[0].recvbuff);
  EXPECT_EQ(kCount, fake.selects[0].count);
  EXPECT_EQ(stream_, fake.selects[0].stream);
  EXPECT_FALSE(fake.selects[0].query);
  EXPECT_FALSE(fake.selects[0].hint);
  const ncclInfo& info = Info();
  EXPECT_EQ(ncclFuncAllGather, info.coll);
  EXPECT_STREQ("AllGather", info.opName);
  EXPECT_EQ(send_.data(), info.sendbuff);
  EXPECT_EQ(recv_.data(), info.recvbuff);
  EXPECT_EQ(kCount, info.count);
  EXPECT_EQ(ncclSum, info.op);
  EXPECT_EQ(comm_.get(), info.comm);
  EXPECT_EQ(stream_, info.stream);
  EXPECT_TRUE(info.decisionValid);
  EXPECT_EQ(NCCL_ALGO_RING, info.decision.algo);
  EXPECT_EQ(NCCL_PROTO_LL, info.decision.protocol);
  EXPECT_EQ(3u, info.decision.nMaxChannels);
  EXPECT_TRUE(info.decision.ceCapturing);
  EXPECT_TRUE(info.decision.ceArGraphAllowed);
  EXPECT_FALSE(info.useDirect);
  EXPECT_TRUE(fake.log.empty());
}

TEST_F(CollectivesMicrotest, AllGather_FailuresBeforeDispatchStopIt) {
  fake.commCountResult = ncclInvalidArgument;
  EXPECT_EQ(ncclInvalidArgument, AllGather());
  fake.commCountResult = ncclSuccess;
  fake.userRankResult = ncclInvalidArgument;
  EXPECT_EQ(ncclInvalidArgument, AllGather());
  EXPECT_TRUE(fake.recorded.empty());
  fake.userRankResult = ncclSuccess;
  fake.recordResult = ncclSystemError;
  EXPECT_EQ(ncclSystemError, AllGather());
  EXPECT_TRUE(fake.selects.empty());
  fake.recordResult = ncclSuccess;
  fake.selectResult = ncclInternalError;
  EXPECT_EQ(ncclInternalError, AllGather());
  EXPECT_TRUE(fake.enqueued.empty());
}

TEST_F(CollectivesMicrotest, AllGather_DdaArmsLaunchAsAddons) {
  ExpectAddonArms({{RCCL_DDA_FABRIC_LL, "ncclAllGatherDdaFabricLL"},
                   {RCCL_DDA_FABRIC_LL128, "ncclAllGatherDdaFabricLL128"},
                   {RCCL_DDA_FABRIC_VMM, "ncclAllGatherDdaFabric"},
                   {RCCL_DDA_IPC, "ncclAllGatherDdaIpc"}},
                  [&] { return AllGather(); }, ncclNumOps);
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "AllGather: taking DDA fabric LL path: nRanks=4 nNodes=1 sendcount=32"));
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "datatype=7 totalBytes=512"));
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "AllGather: taking DDA fabric LL128 path"));
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "AllGather: taking DDA fabric (VMM) path: nRanks=4 nNodes=1 sendcount=32 "
                                    "datatype=7 bytes=128"));
}

TEST_F(CollectivesMicrotest, AllGather_FailedPrologueSkipsTheAddon) {
  Decide(RCCL_DDA_IPC);
  fake.addonBeginResult = ncclSystemError;
  EXPECT_EQ(ncclSystemError, AllGather());
  EXPECT_TRUE(fake.addons.empty());
  EXPECT_EQ(ncclSystemError, fake.addonEndSeen);
}

TEST_F(CollectivesMicrotest, AllGather_CeAndSymmetricEnqueueTheirDecision) {
  for (int algo : {RCCL_CE_REGISTERED, RCCL_CE_SCRATCH, RCCL_SYMMETRIC}) {
    SCOPED_TRACE(algo);
    Decide(algo);
    fake.enqueued.clear();
    EXPECT_EQ(ncclSuccess, AllGather());
    ASSERT_EQ(1u, fake.enqueued.size());
    EXPECT_EQ(algo, Info().decision.algo);
    EXPECT_FALSE(Info().useDirect);
    EXPECT_TRUE(fake.addons.empty());
  }
}

TEST_F(CollectivesMicrotest, AllGather_DirectMarksTheInfo) {
  Decide(RCCL_DIRECT_ALLGATHER);
  EXPECT_EQ(ncclSuccess, AllGather());
  ASSERT_EQ(1u, fake.enqueued.size());
  EXPECT_TRUE(Info().useDirect);
  EXPECT_EQ(RCCL_DIRECT_ALLGATHER, Info().decision.algo);
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "RCCL DIRECT ALLGATHER count = 32, msgSize = 512"));
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "rank = 1"));
}

TEST_F(CollectivesMicrotest, AllGather_EmptyDirectCallIsANoOp) {
  Decide(RCCL_DIRECT_ALLGATHER);
  EXPECT_EQ(ncclSuccess, AllGather(0));
  EXPECT_TRUE(fake.enqueued.empty());
}

TEST_F(CollectivesMicrotest, AllGather_CheckModeDivertsAddonsButKeepsDirect) {
  comm_->checkMode = ncclCheckModeDebugGlobal;
  comm_->rank = 0;
  Decide(RCCL_DDA_IPC);
  EXPECT_EQ(ncclSuccess, AllGather());
  ASSERT_EQ(1u, fake.enqueued.size());
  EXPECT_EQ(RCCL_DDA_IPC, Info().decision.algo);
  EXPECT_TRUE(fake.addons.empty());
  EXPECT_FALSE(Logged(NCCL_LOG_INFO, "impl selected"));

  Decide(RCCL_DIRECT_ALLGATHER);
  EXPECT_EQ(ncclSuccess, AllGather());
  ASSERT_EQ(2u, fake.enqueued.size());
  EXPECT_TRUE(Info(1).useDirect);
}

TEST_F(CollectivesMicrotest, AllGather_NamesTheAddonOnRankZeroOnly) {
  Decide(RCCL_CE_SCRATCH);
  EXPECT_EQ(ncclSuccess, AllGather());
  EXPECT_FALSE(Logged(NCCL_LOG_INFO, "impl selected"));
  comm_->rank = 0;
  Decide(NCCL_ALGO_TREE);
  EXPECT_EQ(ncclSuccess, AllGather());
  EXPECT_FALSE(Logged(NCCL_LOG_INFO, "impl selected"));
  Decide(RCCL_DIRECT_ALLGATHER);
  EXPECT_EQ(ncclSuccess, AllGather());
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "AllGather impl selected: algo ADDON"));
  fake.algoName = nullptr;
  EXPECT_EQ(ncclSuccess, AllGather());
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "AllGather impl selected: algo ?"));
}

// Hierarchical AllGather / ReduceScatter: 2 nodes x 4 local ranks, this rank is node 1, local rank 1.

class CollectivesHierarchicalMicrotest : public CollectivesMicrotest {
 protected:
  static constexpr int kLocal = 4;
  static constexpr int kNodes = 2;

  void SetUp() override {
    CollectivesMicrotest::SetUp();
    comm_->rank = 5;
    comm_->nRanks = kLocal * kNodes;
    comm_->nNodes = kNodes;
    intra_ = MakeComm(1, kLocal);
    inter_ = MakeComm(1, kNodes);
    intra_->tempBuff = &intraTemp_;
    inter_->tempBuff = &interTemp_;
    comm_->hierarchicalIntraComm = intra_.get();
    comm_->hierarchicalInterComm = inter_.get();
    comm_->hierarchicalTempBuffer = temp_.data();
    comm_->addonStopEvent = stopEvent_;
    send_.assign(kCount * kLocal * kNodes, 1.0f);
    recv_.assign(kCount * kLocal * kNodes, 0.0f);
    fake.useAgDirect = [this](ncclComm* comm, size_t msgSize) {
      directAsks_.push_back({comm, msgSize});
      return comm == directComm_ || directComm_ == &any_;
    };
    fake.useRsDirect = fake.useAgDirect;
  }

  struct DirectAsk {
    ncclComm* comm;
    size_t msgSize;
  };
  std::unique_ptr<ncclComm> intra_;
  std::unique_ptr<ncclComm> inter_;
  std::vector<float> temp_ = std::vector<float>(kCount * kLocal * kNodes);
  int intraTemp_ = 0;
  int interTemp_ = 0;
  hipEvent_t stopEvent_ = reinterpret_cast<hipEvent_t>(0xe7);
  ncclComm any_{};
  ncclComm* directComm_ = nullptr;
  std::vector<DirectAsk> directAsks_;
};

TEST_F(CollectivesHierarchicalMicrotest, AllGather_InterThenIntraThenShuffle) {
  Decide(RCCL_HIERARCHICAL_ALLGATHER);
  EXPECT_EQ(ncclSuccess, AllGather());
  EXPECT_EQ((std::vector<std::string>{"record", "select", "begin", "enqueue HierarchicalAllGather-Inter",
                                      "enqueue HierarchicalAllGather-Intra", "launch", "end"}),
            fake.trace);
  ASSERT_EQ(2u, directAsks_.size());
  EXPECT_EQ(inter_.get(), directAsks_[0].comm);
  EXPECT_EQ(kCount * kNodes * sizeof(float), directAsks_[0].msgSize);
  EXPECT_EQ(intra_.get(), directAsks_[1].comm);
  EXPECT_EQ(kCount * kNodes * sizeof(float) * kLocal, directAsks_[1].msgSize);

  const ncclInfo& inter = Info(0);
  EXPECT_EQ(ncclFuncAllGather, inter.coll);
  EXPECT_EQ(send_.data(), inter.sendbuff);
  EXPECT_EQ(recv_.data(), inter.recvbuff);
  EXPECT_EQ(kCount, inter.count);
  EXPECT_EQ(inter_.get(), inter.comm);
  EXPECT_EQ(stream_, inter.stream);
  EXPECT_EQ(ALLGATHER_CHUNKSTEPS, inter.chunkSteps);
  EXPECT_EQ(ALLGATHER_SLICESTEPS, inter.sliceSteps);
  EXPECT_FALSE(inter.useDirect);
  const ncclInfo& intra = Info(1);
  EXPECT_EQ(recv_.data(), intra.sendbuff);
  EXPECT_EQ(temp_.data(), intra.recvbuff);
  EXPECT_EQ(kCount * kNodes, intra.count);
  EXPECT_EQ(intra_.get(), intra.comm);
  EXPECT_EQ(ALLGATHER_SLICESTEPS, intra.sliceSteps);

  ASSERT_EQ(1u, fake.launches.size());
  const KernelLaunch& k = fake.launches[0];
  EXPECT_EQ(reinterpret_cast<const void*>(hierarchicalShuffle), k.fn);
  EXPECT_EQ(8u, k.grid.x);
  EXPECT_EQ(static_cast<unsigned>(HIERARCHICAL_SHUFFLE_THREADS), k.block.x);
  EXPECT_EQ(stream_, k.stream);
  EXPECT_EQ(stopEvent_, k.stopEvent);
  EXPECT_EQ(nullptr, comm_->addonStopEvent);  // taken, so the epilogue does not record it again
  EXPECT_EQ(reinterpret_cast<const char*>(temp_.data()), k.src);
  EXPECT_EQ(reinterpret_cast<char*>(recv_.data()), k.dst);
  EXPECT_EQ(kCount * sizeof(float), k.rankOffset);
  EXPECT_EQ(kNodes, k.cols);
  EXPECT_EQ(kLocal, k.rows);
}

TEST_F(CollectivesHierarchicalMicrotest, AllGather_SingleSliceIntraComm) {
  Decide(RCCL_HIERARCHICAL_ALLGATHER);
  intra_->rcclUseOneSlice = 1;
  EXPECT_EQ(ncclSuccess, AllGather());
  EXPECT_EQ(ALLGATHER_SLICESTEPS_SINGLE_NODE, Info(1).sliceSteps);
  EXPECT_EQ(ALLGATHER_SLICESTEPS, Info(0).sliceSteps);
}

TEST_F(CollectivesHierarchicalMicrotest, AllGather_InPlaceStagesTheOwnSliceFirst) {
  Decide(RCCL_HIERARCHICAL_ALLGATHER);
  const float* own = recv_.data() + comm_->rank * kCount;
  EXPECT_EQ(ncclSuccess, ncclAllGather_impl(own, recv_.data(), kCount, ncclFloat32, comm_.get(), stream_));
  ASSERT_EQ(1u, fake.memcpys.size());
  EXPECT_EQ(temp_.data(), fake.memcpys[0].dst);
  EXPECT_EQ(own, fake.memcpys[0].src);
  EXPECT_EQ(kCount * sizeof(float), fake.memcpys[0].bytes);
  EXPECT_EQ(hipMemcpyDeviceToDevice, fake.memcpys[0].kind);
  EXPECT_EQ(stream_, fake.memcpys[0].stream);
  EXPECT_EQ(temp_.data(), Info(0).sendbuff);
}

TEST_F(CollectivesHierarchicalMicrotest, AllGather_DirectStepsPostPerPeer) {
  Decide(RCCL_HIERARCHICAL_ALLGATHER);
  directComm_ = &any_;
  EXPECT_EQ(ncclSuccess, AllGather());
  ASSERT_EQ(2u, fake.enqueued.size());
  EXPECT_TRUE(Info(0).useDirect);
  EXPECT_STREQ("AllGather", Info(0).opName);
  EXPECT_EQ(inter_.get(), Info(0).comm);
  EXPECT_EQ(send_.data(), Info(0).sendbuff);
  EXPECT_EQ(kCount, Info(0).count);
  EXPECT_TRUE(Info(1).useDirect);
  EXPECT_EQ(intra_.get(), Info(1).comm);
  EXPECT_EQ(recv_.data(), Info(1).sendbuff);
  EXPECT_EQ(temp_.data(), Info(1).recvbuff);
  EXPECT_EQ(kCount * kNodes, Info(1).count);
  EXPECT_EQ(ALLGATHER_CHUNKSTEPS, Info(1).chunkSteps);
  EXPECT_EQ(ALLGATHER_SLICESTEPS, Info(1).sliceSteps);
}

TEST_F(CollectivesHierarchicalMicrotest, AllGather_InterDirectOnlyUpTo16Nodes) {
  Decide(RCCL_HIERARCHICAL_ALLGATHER);
  directComm_ = inter_.get();
  inter_->nRanks = 16;
  EXPECT_EQ(ncclSuccess, AllGather());
  EXPECT_TRUE(Info(0).useDirect);
  inter_->nRanks = 17;
  directAsks_.clear();
  fake.enqueued.clear();
  EXPECT_EQ(ncclSuccess, AllGather());
  ASSERT_EQ(1u, directAsks_.size());
  EXPECT_EQ(intra_.get(), directAsks_[0].comm);
  EXPECT_FALSE(Info(0).useDirect);
}

TEST_F(CollectivesHierarchicalMicrotest, AllGather_EmptyCallLaunchesNothing) {
  Decide(RCCL_HIERARCHICAL_ALLGATHER);
  EXPECT_EQ(ncclSuccess, AllGather(0));
  EXPECT_EQ((std::vector<std::string>{"record", "select", "begin", "end"}), fake.trace);
}

TEST_F(CollectivesHierarchicalMicrotest, AllGather_EachStepFailureStopsTheRest) {
  Decide(RCCL_HIERARCHICAL_ALLGATHER);
  const float* own = recv_.data() + comm_->rank * kCount;
  fake.memcpyResults = {hipErrorOutOfMemory};
  EXPECT_EQ(ncclUnhandledCudaError,
            ncclAllGather_impl(own, recv_.data(), kCount, ncclFloat32, comm_.get(), stream_));
  EXPECT_TRUE(fake.enqueued.empty());

  fake.enqueueResults = {ncclRemoteError};
  EXPECT_EQ(ncclRemoteError, AllGather());
  EXPECT_EQ(1u, fake.enqueued.size());

  fake.enqueued.clear();
  fake.enqueueResults = {ncclSuccess, ncclRemoteError};
  EXPECT_EQ(ncclRemoteError, AllGather());
  EXPECT_EQ(2u, fake.enqueued.size());
  EXPECT_TRUE(fake.launches.empty());

  directComm_ = &any_;
  fake.enqueued.clear();
  fake.enqueueResults = {ncclRemoteError};
  EXPECT_EQ(ncclRemoteError, AllGather());
  EXPECT_EQ(1u, fake.enqueued.size());
  fake.enqueued.clear();
  fake.enqueueResults = {ncclSuccess, ncclRemoteError};
  EXPECT_EQ(ncclRemoteError, AllGather());
  EXPECT_TRUE(fake.launches.empty());

  fake.lastError = hipErrorLaunchFailure;
  EXPECT_EQ(ncclUnhandledCudaError, AllGather());
  EXPECT_EQ(1u, fake.launches.size());
}

TEST_F(CollectivesHierarchicalMicrotest, ShuffleBlocksScaleWithTheGatheredBytes) {
  Decide(RCCL_HIERARCHICAL_ALLGATHER);
  const size_t perRank = (64 << 10) / (kLocal * kNodes) / sizeof(float);
  send_.resize(perRank * kLocal * kNodes + 1);
  recv_.resize(perRank * kLocal * kNodes + kLocal * kNodes);
  temp_.resize(recv_.size());
  comm_->hierarchicalTempBuffer = temp_.data();
  EXPECT_EQ(ncclSuccess, AllGather(perRank));
  EXPECT_EQ(ncclSuccess, AllGather(perRank + 1));
  ASSERT_EQ(2u, fake.launches.size());
  EXPECT_EQ(8u, fake.launches[0].grid.x);
  EXPECT_EQ(16u, fake.launches[1].grid.x);
}

TEST_F(CollectivesHierarchicalMicrotest, ReduceScatter_ShuffleThenIntraThenInter) {
  Decide(RCCL_HIERARCHICAL_REDUCESCATTER);
  EXPECT_EQ(ncclSuccess, ReduceScatter());
  EXPECT_EQ((std::vector<std::string>{"record", "select", "begin", "launch", "enqueue HierarchicalReduceScatter-Intra",
                                      "enqueue HierarchicalReduceScatter-Inter", "end"}),
            fake.trace);
  ASSERT_EQ(1u, fake.launches.size());
  const KernelLaunch& k = fake.launches[0];
  EXPECT_EQ(reinterpret_cast<const void*>(hierarchicalShuffle), k.fn);
  EXPECT_EQ(8u, k.grid.x);
  EXPECT_EQ(static_cast<unsigned>(HIERARCHICAL_SHUFFLE_THREADS), k.block.x);
  EXPECT_EQ(0u, k.shmem);
  EXPECT_EQ(stream_, k.stream);
  EXPECT_EQ(reinterpret_cast<const char*>(send_.data()), k.src);
  EXPECT_EQ(reinterpret_cast<char*>(temp_.data()), k.dst);
  EXPECT_EQ(kCount * sizeof(float), k.rankOffset);
  EXPECT_EQ(kLocal, k.cols);
  EXPECT_EQ(kNodes, k.rows);

  float* intraOut = temp_.data() + intra_->rank * kCount * kNodes;
  const ncclInfo& intra = Info(0);
  EXPECT_EQ(ncclFuncReduceScatter, intra.coll);
  EXPECT_EQ(temp_.data(), intra.sendbuff);
  EXPECT_EQ(intraOut, intra.recvbuff);
  EXPECT_EQ(kCount * kNodes, intra.count);
  EXPECT_EQ(ncclMin, intra.op);
  EXPECT_EQ(intra_.get(), intra.comm);
  EXPECT_EQ(REDUCESCATTER_CHUNKSTEPS, intra.chunkSteps);
  EXPECT_EQ(REDUCESCATTER_SLICESTEPS, intra.sliceSteps);
  const ncclInfo& inter = Info(1);
  EXPECT_EQ(intraOut, inter.sendbuff);
  EXPECT_EQ(recv_.data(), inter.recvbuff);
  EXPECT_EQ(kCount, inter.count);
  EXPECT_EQ(ncclMin, inter.op);
  EXPECT_EQ(inter_.get(), inter.comm);
  EXPECT_EQ(REDUCESCATTER_SLICESTEPS, inter.sliceSteps);
  ASSERT_EQ(1u, directAsks_.size());
  EXPECT_EQ(inter_.get(), directAsks_[0].comm);
  EXPECT_EQ(kCount * sizeof(float) * kNodes, directAsks_[0].msgSize);
}

TEST_F(CollectivesHierarchicalMicrotest, ReduceScatter_SingleSliceIntraComm) {
  Decide(RCCL_HIERARCHICAL_REDUCESCATTER);
  intra_->rcclUseOneSlice = 1;
  EXPECT_EQ(ncclSuccess, ReduceScatter());
  EXPECT_EQ(REDUCESCATTER_SLICESTEPS_SINGLE_NODE, Info(0).sliceSteps);
}

TEST_F(CollectivesHierarchicalMicrotest, ReduceScatter_DirectInterStep) {
  Decide(RCCL_HIERARCHICAL_REDUCESCATTER);
  directComm_ = inter_.get();
  inter_->rcclUseOneSlice = 1;  // the inter step keeps the multi-node steps regardless
  EXPECT_EQ(ncclSuccess, ReduceScatter());
  ASSERT_EQ(3u, fake.enqueued.size());
  float* intraOut = temp_.data() + intra_->rank * kCount * kNodes;
  EXPECT_TRUE(Info(1).useDirect);
  EXPECT_EQ(intraOut, Info(1).sendbuff);
  EXPECT_EQ(&interTemp_, Info(1).recvbuff);
  EXPECT_EQ(inter_.get(), Info(1).comm);
  EXPECT_EQ(REDUCESCATTER_CHUNKSTEPS, Info(1).chunkSteps);
  EXPECT_EQ(REDUCESCATTER_SLICESTEPS, Info(1).sliceSteps);
  EXPECT_FALSE(Info(2).useDirect);
  EXPECT_EQ(recv_.data(), Info(2).recvbuff);
  EXPECT_TRUE(fake.enqueued[1].directReduceScatter);
  EXPECT_TRUE(fake.enqueued[2].directReduceScatter);
  EXPECT_FALSE(inter_->enableDirectReduceScatter);
}

TEST_F(CollectivesHierarchicalMicrotest, ReduceScatter_EachStepFailureStopsTheRest) {
  Decide(RCCL_HIERARCHICAL_REDUCESCATTER);
  EXPECT_EQ(ncclSuccess, ReduceScatter(0));
  EXPECT_TRUE(fake.launches.empty());

  fake.lastError = hipErrorLaunchFailure;
  EXPECT_EQ(ncclUnhandledCudaError, ReduceScatter());
  EXPECT_TRUE(fake.enqueued.empty());
  fake.lastError = hipSuccess;

  fake.enqueueResults = {ncclRemoteError};
  EXPECT_EQ(ncclRemoteError, ReduceScatter());
  EXPECT_EQ(1u, fake.enqueued.size());

  directComm_ = inter_.get();
  fake.enqueued.clear();
  fake.enqueueResults = {ncclSuccess, ncclRemoteError};
  EXPECT_EQ(ncclRemoteError, ReduceScatter());
  EXPECT_EQ(2u, fake.enqueued.size());
}

// AlltoAll dispatch

TEST_F(CollectivesMicrotest, AlltoAll_RecordsSelectsAndEnqueuesTheDecision) {
  for (int algo : std::vector<int>{NCCL_ALGO_RING, RCCL_CE_REGISTERED, RCCL_CE_SCRATCH, RCCL_DIRECT_ALLTOALL}) {
    SCOPED_TRACE(algo);
    Decide(algo);
    fake.trace.clear();
    fake.enqueued.clear();
    fake.recorded.clear();
    fake.selects.clear();
    EXPECT_EQ(ncclSuccess, AlltoAll());
    EXPECT_EQ((std::vector<std::string>{"record", "select", "enqueue AlltoAll"}), fake.trace);
    ASSERT_EQ(1u, fake.recorded.size());
    EXPECT_EQ(rccl::rrAllToAll, fake.recorded[0].type);
    EXPECT_EQ(send_.data(), fake.recorded[0].sendbuff);
    EXPECT_EQ(recv_.data(), fake.recorded[0].recvbuff);
    EXPECT_EQ(kCount, fake.recorded[0].count);
    ASSERT_EQ(1u, fake.selects.size());
    EXPECT_EQ("AlltoAll", fake.selects[0].coll);
    EXPECT_EQ(kCount, fake.selects[0].count);
    EXPECT_EQ(stream_, fake.selects[0].stream);
    EXPECT_FALSE(fake.selects[0].query);
    EXPECT_FALSE(fake.selects[0].hint);
    const ncclInfo& info = Info();
    EXPECT_EQ(ncclFuncAlltoAll, info.coll);
    EXPECT_EQ(send_.data(), info.sendbuff);
    EXPECT_EQ(recv_.data(), info.recvbuff);
    EXPECT_EQ(kCount, info.count);
    EXPECT_EQ(ALLTOALL_CHUNKSTEPS, info.chunkSteps);
    EXPECT_EQ(ALLTOALL_SLICESTEPS, info.sliceSteps);
    EXPECT_TRUE(info.decisionValid);
    EXPECT_EQ(algo, info.decision.algo);
  }
}

TEST_F(CollectivesMicrotest, AlltoAll_RecordOrSelectFailureStops) {
  fake.recordResult = ncclSystemError;
  EXPECT_EQ(ncclSystemError, AlltoAll());
  EXPECT_TRUE(fake.selects.empty());
  fake.recordResult = ncclSuccess;
  fake.selectResult = ncclInternalError;
  EXPECT_EQ(ncclInternalError, AlltoAll());
  EXPECT_TRUE(fake.enqueued.empty());
}

TEST_F(CollectivesMicrotest, AlltoAll_PivotEnqueuesItsOwnInfoEvenInCheckMode) {
  Decide(RCCL_A2A_PIVOT);
  comm_->checkMode = ncclCheckModeDebugLocal;
  fake.enqueueResults = {ncclInProgress};
  EXPECT_EQ(ncclInProgress, AlltoAll());
  ASSERT_EQ(1u, fake.enqueued.size());
  const ncclInfo& info = Info();
  EXPECT_EQ(ncclFuncAlltoAllPivot, info.coll);
  EXPECT_STREQ("AlltoAllPivot", info.opName);
  EXPECT_EQ(send_.data(), info.sendbuff);
  EXPECT_EQ(recv_.data(), info.recvbuff);
  EXPECT_EQ(kCount, info.count);
  EXPECT_EQ(comm_.get(), info.comm);
  EXPECT_EQ(stream_, info.stream);
  EXPECT_EQ(ALLTOALL_PIVOT_CHUNKSTEPS, info.chunkSteps);
  EXPECT_EQ(ALLTOALL_PIVOT_SLICESTEPS, info.sliceSteps);
  EXPECT_FALSE(info.decisionValid);
}

TEST_F(CollectivesMicrotest, AlltoAll_DdaArmsLaunchAsAddons) {
  comm_->rank = 0;
  ExpectAddonArms({{RCCL_DDA_FABRIC_LL, "ncclAllToAllDdaFabricLL"},
                   {RCCL_DDA_FABRIC_LL128, "ncclAllToAllDdaFabricLL128"},
                   {RCCL_DDA_FABRIC_VMM, "ncclAllToAllDdaFabric"},
                   {RCCL_DDA_IPC, "ncclAllToAllDdaIpc"}},
                  [&] { return AlltoAll(); }, ncclNumOps);
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "AlltoAll impl selected: algo ADDON"));
}

TEST_F(CollectivesMicrotest, AlltoAll_CheckModeDivertsDda) {
  comm_->checkMode = ncclCheckModeDebugLocal;
  comm_->rank = 0;
  Decide(RCCL_DDA_FABRIC_LL);
  EXPECT_EQ(ncclSuccess, AlltoAll());
  ASSERT_EQ(1u, fake.enqueued.size());
  EXPECT_EQ(ncclFuncAlltoAll, Info().coll);
  EXPECT_EQ(RCCL_DDA_FABRIC_LL, Info().decision.algo);
  EXPECT_TRUE(fake.addons.empty());
  EXPECT_TRUE(fake.log.empty());
}

TEST_F(CollectivesMicrotest, AlltoAll_NamesTheAddonOnRankZeroOnly) {
  Decide(RCCL_CE_SCRATCH);
  EXPECT_EQ(ncclSuccess, AlltoAll());
  comm_->rank = 0;
  Decide(NCCL_ALGO_RING);
  EXPECT_EQ(ncclSuccess, AlltoAll());
  EXPECT_TRUE(fake.log.empty());
  fake.algoName = nullptr;
  Decide(RCCL_DIRECT_ALLTOALL);
  EXPECT_EQ(ncclSuccess, AlltoAll());
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "AlltoAll impl selected: algo ?"));
}

// AllReduce dispatch

TEST_F(CollectivesMicrotest, AllReduce_RecordsSelectsAndEnqueuesTheDecision) {
  for (int algo : std::vector<int>{NCCL_ALGO_TREE, RCCL_SYMMETRIC, RCCL_CE_REGISTERED}) {
    SCOPED_TRACE(algo);
    Decide(algo);
    fake.trace.clear();
    fake.enqueued.clear();
    fake.recorded.clear();
    fake.selects.clear();
    EXPECT_EQ(ncclSuccess, AllReduce());
    EXPECT_EQ((std::vector<std::string>{"record", "select", "enqueue AllReduce"}), fake.trace);
    ASSERT_EQ(1u, fake.recorded.size());
    EXPECT_EQ(rccl::rrAllReduce, fake.recorded[0].type);
    ASSERT_EQ(1u, fake.selects.size());
    EXPECT_EQ("AllReduce", fake.selects[0].coll);
    EXPECT_EQ(send_.data(), fake.selects[0].sendbuff);
    EXPECT_EQ(recv_.data(), fake.selects[0].recvbuff);
    EXPECT_EQ(kCount, fake.selects[0].count);
    EXPECT_EQ(ncclProd, fake.selects[0].op);
    EXPECT_EQ(stream_, fake.selects[0].stream);
    EXPECT_FALSE(fake.selects[0].query);
    EXPECT_FALSE(fake.selects[0].hint);
    const ncclInfo& info = Info();
    EXPECT_EQ(ncclFuncAllReduce, info.coll);
    EXPECT_EQ(ncclProd, info.op);
    EXPECT_EQ(nullptr, info.acc);
    EXPECT_TRUE(info.decisionValid);
    EXPECT_EQ(algo, info.decision.algo);
  }
}

TEST_F(CollectivesMicrotest, AllReduce_RecordOrSelectFailureStops) {
  fake.recordResult = ncclSystemError;
  EXPECT_EQ(ncclSystemError, AllReduce());
  EXPECT_TRUE(fake.selects.empty());
  fake.recordResult = ncclSuccess;
  fake.selectResult = ncclInternalError;
  EXPECT_EQ(ncclInternalError, AllReduce());
  EXPECT_TRUE(fake.enqueued.empty());
}

TEST_F(CollectivesMicrotest, AllReduce_DdaArmsLaunchAsAddons) {
  ExpectAddonArms({{RCCL_DDA_FABRIC_LL, "ncclAllReduceDdaFabricLL"},
                   {RCCL_DDA_FABRIC_LL128, "ncclAllReduceDdaFabricLL128"},
                   {RCCL_DDA_FABRIC_VMM, "ncclAllReduceDdaFabric"},
                   {RCCL_DDA_IPC, "ncclAllReduceDdaIpc"}},
                  [&] { return AllReduce(); }, ncclProd);
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "AllReduce: taking DDA fabric LL path: nRanks=4 nNodes=1 count=32 datatype=7 "
                                    "bytes=128"));
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "AllReduce: taking DDA fabric LL128 path"));
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "AllReduce: taking DDA fabric (VMM) path"));
}

TEST_F(CollectivesMicrotest, AllReduce_CheckModeDivertsAddons) {
  comm_->checkMode = ncclCheckModeDebugLocal;
  for (int algo : {RCCL_CE_2SHOT, RCCL_DDA_IPC}) {
    Decide(algo);
    fake.enqueued.clear();
    EXPECT_EQ(ncclSuccess, AllReduce());
    ASSERT_EQ(1u, fake.enqueued.size());
    EXPECT_EQ(algo, Info().decision.algo);
  }
  EXPECT_TRUE(fake.addons.empty());
  EXPECT_TRUE(fake.ceArs.empty());
}

TEST_F(CollectivesMicrotest, AllReduce_NamesTheAddonOnRankZeroOnly) {
  Decide(RCCL_SYMMETRIC);
  EXPECT_EQ(ncclSuccess, AllReduce());
  comm_->rank = 0;
  Decide(NCCL_ALGO_PAT);
  EXPECT_EQ(ncclSuccess, AllReduce());
  EXPECT_TRUE(fake.log.empty());
  Decide(RCCL_SYMMETRIC);
  EXPECT_EQ(ncclSuccess, AllReduce());
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "AllReduce impl selected: algo ADDON"));
  fake.algoName = nullptr;
  EXPECT_EQ(ncclSuccess, AllReduce());
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "AllReduce impl selected: algo ?"));
}

TEST_F(CollectivesMicrotest, AllReduce_CeTwoShotRunsInsideAProfilerEvent) {
  Decide(RCCL_CE_2SHOT);
  fake.ceArResult = ncclRemoteError;
  EXPECT_EQ(ncclRemoteError, AllReduce());
  EXPECT_EQ((std::vector<std::string>{"record", "select", "profStart", "begin", "ceAllReduce", "end", "profStop"}),
            fake.trace);
  EXPECT_EQ(ncclFuncAllReduce, fake.profArgs.func);
  EXPECT_EQ(ncclFloat32, fake.profArgs.datatype);
  EXPECT_EQ(ncclProd, fake.profArgs.redOp);
  EXPECT_EQ(kCount, fake.profArgs.nElts);
  EXPECT_EQ(sizeof(float), fake.profArgs.eltSize);
  EXPECT_EQ(reinterpret_cast<uint8_t*>(send_.data()), fake.profArgs.sendBuff);
  EXPECT_EQ(reinterpret_cast<uint8_t*>(recv_.data()), fake.profArgs.recvBuff);
  EXPECT_EQ(nullptr, fake.profArgs.collApiEventHandle);
  ASSERT_EQ(1u, fake.ceArs.size());
  EXPECT_EQ(send_.data(), fake.ceArs[0].sendbuff);
  EXPECT_EQ(recv_.data(), fake.ceArs[0].recvbuff);
  EXPECT_EQ(kCount, fake.ceArs[0].count);
  EXPECT_EQ(ncclFloat32, fake.ceArs[0].datatype);
  EXPECT_EQ(ncclProd, fake.ceArs[0].op);
  EXPECT_EQ(nullptr, fake.ceArs[0].win);
  EXPECT_NE(nullptr, fake.ceArs[0].args);
  EXPECT_TRUE(fake.enqueued.empty());
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "CE 2-shot AllReduce: count=32 datatype=7 op=1 rank=1/4"));
}

TEST_F(CollectivesMicrotest, AllReduce_CeTwoShotSkipsEmptyCallsAndAFailedProfilerStart) {
  Decide(RCCL_CE_2SHOT);
  EXPECT_EQ(ncclSuccess, AllReduce(0));
  EXPECT_EQ((std::vector<std::string>{"record", "select"}), fake.trace);
  fake.profStartResult = ncclInternalError;
  EXPECT_EQ(ncclInternalError, AllReduce());
  EXPECT_TRUE(fake.ceArs.empty());
  EXPECT_EQ(0, std::count(fake.trace.begin(), fake.trace.end(), "profStop"));
}

// ReduceScatter dispatch

TEST_F(CollectivesMicrotest, ReduceScatter_ProbesCaptureIntoTheDecision) {
  comm_->enableDirectReduceScatter = 1;
  fake.capturing = true;
  fake.decision.ceCapturing = false;
  EXPECT_EQ(ncclSuccess, ReduceScatter());
  EXPECT_EQ((std::vector<std::string>{"record", "select", "enqueue ReduceScatter"}), fake.trace);
  ASSERT_EQ(1u, fake.recorded.size());
  EXPECT_EQ(rccl::rrReduceScatter, fake.recorded[0].type);
  ASSERT_EQ(1u, fake.selects.size());
  EXPECT_EQ("ReduceScatter", fake.selects[0].coll);
  EXPECT_EQ(send_.data(), fake.selects[0].sendbuff);
  EXPECT_EQ(recv_.data(), fake.selects[0].recvbuff);
  EXPECT_EQ(kCount, fake.selects[0].count);
  EXPECT_EQ(ncclMin, fake.selects[0].op);
  EXPECT_FALSE(fake.selects[0].query);
  EXPECT_EQ(std::vector<int>{2}, fake.captureModes);
  const ncclInfo& info = Info();
  EXPECT_EQ(ncclFuncReduceScatter, info.coll);
  EXPECT_EQ(ncclMin, info.op);
  EXPECT_TRUE(info.decisionValid);
  EXPECT_TRUE(info.decision.ceCapturing);
  EXPECT_FALSE(fake.enqueued[0].directReduceScatter);  // a stale force flag is cleared first

  fake.enqueued.clear();
  fake.capturing = false;
  fake.decision.ceCapturing = true;
  EXPECT_EQ(ncclSuccess, ReduceScatter());
  EXPECT_FALSE(Info().decision.ceCapturing);
}

TEST_F(CollectivesMicrotest, ReduceScatter_FailuresBeforeDispatchStopIt) {
  fake.commCountResult = ncclInvalidArgument;
  EXPECT_EQ(ncclInvalidArgument, ReduceScatter());
  EXPECT_TRUE(fake.recorded.empty());
  fake.commCountResult = ncclSuccess;
  fake.recordResult = ncclSystemError;
  EXPECT_EQ(ncclSystemError, ReduceScatter());
  EXPECT_TRUE(fake.selects.empty());
  fake.recordResult = ncclSuccess;
  fake.selectResult = ncclInternalError;
  EXPECT_EQ(ncclInternalError, ReduceScatter());
  EXPECT_TRUE(fake.captureModes.empty());
  fake.selectResult = ncclSuccess;
  fake.captureResult = ncclInvalidUsage;
  EXPECT_EQ(ncclInvalidUsage, ReduceScatter());
  EXPECT_TRUE(fake.enqueued.empty());
}

TEST_F(CollectivesMicrotest, ReduceScatter_DdaArmsLaunchAsAddons) {
  comm_->rank = 0;
  ExpectAddonArms({{RCCL_DDA_FABRIC_LL, "ncclReduceScatterDdaFabricLL"},
                   {RCCL_DDA_FABRIC_LL128, "ncclReduceScatterDdaFabricLL128"},
                   {RCCL_DDA_FABRIC_VMM, "ncclReduceScatterDdaFabric"},
                   {RCCL_DDA_IPC, "ncclReduceScatterDdaIpc"}},
                  [&] { return ReduceScatter(); }, ncclMin);
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "ReduceScatter impl selected: algo ADDON"));
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "ReduceScatter: taking DDA fabric LL path: nRanks=4 nNodes=1 recvcount=32 "
                                    "datatype=7 bytes=128"));
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "ReduceScatter: taking DDA fabric LL128 path"));
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "ReduceScatter: taking DDA fabric (VMM) path"));
}

TEST_F(CollectivesMicrotest, ReduceScatter_CheckModeDivertsAddons) {
  comm_->checkMode = ncclCheckModeDebugLocal;
  comm_->rank = 0;
  Decide(RCCL_DIRECT_REDUCESCATTER);
  EXPECT_EQ(ncclSuccess, ReduceScatter());
  ASSERT_EQ(1u, fake.enqueued.size());
  EXPECT_FALSE(Info().useDirect);
  EXPECT_FALSE(fake.enqueued[0].directReduceScatter);
  EXPECT_TRUE(fake.log.empty());
}

TEST_F(CollectivesMicrotest, ReduceScatter_NamesTheAddonOnRankZeroOnly) {
  Decide(RCCL_SYMMETRIC);
  EXPECT_EQ(ncclSuccess, ReduceScatter());
  comm_->rank = 0;
  Decide(NCCL_ALGO_RING);
  EXPECT_EQ(ncclSuccess, ReduceScatter());
  EXPECT_TRUE(fake.log.empty());
  fake.algoName = nullptr;
  Decide(RCCL_SYMMETRIC);
  EXPECT_EQ(ncclSuccess, ReduceScatter());
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "ReduceScatter impl selected: algo ?"));
}

// Direct ReduceScatter stages every peer slice into tempBuff, then reduces it, both under the force flag.
TEST_F(CollectivesMicrotest, ReduceScatter_DirectStagesThenReduces) {
  Decide(RCCL_DIRECT_REDUCESCATTER);
  int tempBuff = 0;
  comm_->tempBuff = &tempBuff;
  comm_->rcclUseOneSlice = 1;
  EXPECT_EQ(ncclSuccess, ReduceScatter());
  EXPECT_EQ((std::vector<std::string>{"record", "select", "enqueue ReduceScatter", "enqueue ReduceScatter"}),
            fake.trace);
  ASSERT_EQ(2u, fake.enqueued.size());
  const ncclInfo& stage = Info(0);
  EXPECT_TRUE(stage.useDirect);
  EXPECT_EQ(send_.data(), stage.sendbuff);
  EXPECT_EQ(&tempBuff, stage.recvbuff);
  EXPECT_EQ(kCount, stage.count);
  EXPECT_EQ(ncclMin, stage.op);
  EXPECT_EQ(REDUCESCATTER_CHUNKSTEPS, stage.chunkSteps);
  EXPECT_EQ(REDUCESCATTER_SLICESTEPS_SINGLE_NODE, stage.sliceSteps);
  const ncclInfo& reduce = Info(1);
  EXPECT_FALSE(reduce.useDirect);
  EXPECT_EQ(send_.data(), reduce.sendbuff);
  EXPECT_EQ(recv_.data(), reduce.recvbuff);
  EXPECT_EQ(REDUCESCATTER_SLICESTEPS_SINGLE_NODE, reduce.sliceSteps);
  EXPECT_TRUE(fake.enqueued[0].directReduceScatter);
  EXPECT_TRUE(fake.enqueued[1].directReduceScatter);
  EXPECT_FALSE(comm_->enableDirectReduceScatter);
  EXPECT_TRUE(Logged(NCCL_LOG_INFO, "RCCL DIRECT REDUCE-SCATTER recvcount=32 msgSize=512 rank=1 nRanks=4 nNodes=1"));
}

TEST_F(CollectivesMicrotest, ReduceScatter_DirectClearsTheForceFlagOnFailure) {
  Decide(RCCL_DIRECT_REDUCESCATTER);
  fake.enqueueResults = {ncclRemoteError};
  EXPECT_EQ(ncclRemoteError, ReduceScatter());
  EXPECT_EQ(1u, fake.enqueued.size());
  EXPECT_FALSE(comm_->enableDirectReduceScatter);
  fake.enqueued.clear();
  fake.enqueueResults = {ncclSuccess, ncclInvalidUsage};
  EXPECT_EQ(ncclInvalidUsage, ReduceScatter());
  EXPECT_EQ(2u, fake.enqueued.size());
  EXPECT_FALSE(comm_->enableDirectReduceScatter);
}

TEST_F(CollectivesMicrotest, ReduceScatter_EmptyDirectCallIsANoOp) {
  Decide(RCCL_DIRECT_REDUCESCATTER);
  comm_->enableDirectReduceScatter = 1;
  EXPECT_EQ(ncclSuccess, ReduceScatter(0));
  EXPECT_TRUE(fake.enqueued.empty());
  EXPECT_FALSE(comm_->enableDirectReduceScatter);
}

// AlltoAllv: per-peer Send/Recv, or a size-metadata exchange that feeds the CE planners.

class CollectivesAlltoAllvMicrotest : public CollectivesMicrotest {
 protected:
  void SetUp() override {
    CollectivesMicrotest::SetUp();
    fake.windows[send_.data()] = &sendWin_;
    fake.windows[recv_.data()] = &recvWin_;
    fake.regType = ncclSymSendRegRecvReg;
  }
  ncclResult_t AlltoAllv() {
    return ncclAlltoAllv_impl(send_.data(), sendcounts_, sdispls_, recv_.data(), recvcounts_, rdispls_, ncclFloat32,
                              comm_.get(), stream_);
  }
  // The [sendSizes, sendDispls, recvSizes, recvDispls] block this rank publishes, in bytes.
  std::vector<size_t> LocalBlock() {
    std::vector<size_t> block(4 * kNRanks);
    ncclAlltoAllvPackLocalSizes(block.data(), kNRanks, sendcounts_, sdispls_, recvcounts_, rdispls_);
    for (size_t& v : block) v *= sizeof(float);
    return block;
  }
  void ScriptPeerBlocks() {
    for (int r = 0; r < kNRanks; r++) {
      fake.peerMeta[r].assign(4 * kNRanks, 0);
      for (int i = 0; i < 4 * kNRanks; i++) fake.peerMeta[r][i] = 1000 * r + i;
    }
  }
  size_t sendcounts_[kNRanks] = {1, 2, 3, 4};
  size_t sdispls_[kNRanks] = {0, 1, 3, 6};
  size_t recvcounts_[kNRanks] = {4, 3, 2, 1};
  size_t rdispls_[kNRanks] = {0, 4, 7, 9};
  ncclDevrWindow sendWin_{};
  ncclDevrWindow recvWin_{};
};

TEST_F(CollectivesAlltoAllvMicrotest, PeerToPeer_PostsEveryPeerAtItsDisplacement) {
  EXPECT_EQ(ncclSuccess, AlltoAllv());
  ASSERT_EQ(1u, fake.recorded.size());
  EXPECT_EQ(rccl::rrAllToAllv, fake.recorded[0].type);
  EXPECT_EQ(send_.data(), fake.recorded[0].sendbuff);
  EXPECT_EQ(recv_.data(), fake.recorded[0].recvbuff);
  EXPECT_EQ(0u, fake.recorded[0].count);
  EXPECT_EQ(sendcounts_, fake.recorded[0].sendcounts);
  EXPECT_EQ(rdispls_, fake.recorded[0].rdispls);
  EXPECT_EQ((std::vector<bool>{true, false}), fake.recorderSkips);
  ASSERT_EQ(2u * kNRanks, fake.p2ps.size());
  for (int r = 0; r < kNRanks; r++) {
    SCOPED_TRACE(r);
    const P2p& send = fake.p2ps[2 * r];
    const P2p& recv = fake.p2ps[2 * r + 1];
    EXPECT_TRUE(send.send);
    EXPECT_EQ(send_.data() + sdispls_[r], send.buff);
    EXPECT_EQ(sendcounts_[r], send.count);
    EXPECT_EQ(ncclFloat32, send.datatype);
    EXPECT_EQ(r, send.peer);
    EXPECT_FALSE(recv.send);
    EXPECT_EQ(recv_.data() + rdispls_[r], recv.buff);
    EXPECT_EQ(recvcounts_[r], recv.count);
    EXPECT_EQ(r, recv.peer);
  }
  EXPECT_EQ((std::vector<std::string>{"record", "findWindow", "findWindow", "groupStart", "groupEnd"}), fake.trace);
  EXPECT_TRUE(fake.enqueued.empty());
  EXPECT_TRUE(fake.memAllocSizes.empty());
}

TEST_F(CollectivesAlltoAllvMicrotest, Eligibility_SeesTheWindowsRegistrationSysmemAndCapture) {
  fake.sysmemWindows.insert(&recvWin_);
  fake.capturing = true;
  EXPECT_EQ(ncclSuccess, AlltoAllv());
  ASSERT_EQ(1u, fake.regTypeArgs.size());
  EXPECT_EQ(&sendWin_, fake.regTypeArgs[0].first);
  EXPECT_EQ(&recvWin_, fake.regTypeArgs[0].second);
  EXPECT_EQ(std::vector<int>{2}, fake.captureModes);
  ASSERT_EQ(1u, fake.ceEligCalls.size());
  EXPECT_EQ(ncclFloat32, fake.ceEligCalls[0].datatype);
  EXPECT_EQ(ncclSymSendRegRecvReg, fake.ceEligCalls[0].regType);
  EXPECT_TRUE(fake.ceEligCalls[0].sysmem);
  EXPECT_TRUE(fake.ceEligCalls[0].capturing);
  ASSERT_EQ(1u, fake.hierEligCalls.size());
  EXPECT_EQ(&sendWin_, fake.hierEligCalls[0].sendWin);
  EXPECT_EQ(&recvWin_, fake.hierEligCalls[0].recvWin);
  EXPECT_TRUE(fake.hierEligCalls[0].sysmem);
  EXPECT_TRUE(fake.hierEligCalls[0].capturing);

  fake.sysmemWindows = {&sendWin_};
  fake.capturing = false;
  EXPECT_EQ(ncclSuccess, AlltoAllv());
  EXPECT_TRUE(fake.ceEligCalls[1].sysmem);
  EXPECT_FALSE(fake.ceEligCalls[1].capturing);
  fake.sysmemWindows.clear();
  EXPECT_EQ(ncclSuccess, AlltoAllv());
  EXPECT_FALSE(fake.ceEligCalls[2].sysmem);
  EXPECT_FALSE(fake.hierEligCalls[2].sysmem);
}

TEST_F(CollectivesAlltoAllvMicrotest, SetupFailuresStopBeforeAnyTraffic) {
  fake.recordResult = ncclSystemError;
  EXPECT_EQ(ncclSystemError, AlltoAllv());
  fake.recordResult = ncclSuccess;
  fake.commCountResult = ncclInvalidArgument;
  EXPECT_EQ(ncclInvalidArgument, AlltoAllv());
  fake.commCountResult = ncclSuccess;
  fake.userRankResult = ncclInvalidArgument;
  EXPECT_EQ(ncclInvalidArgument, AlltoAllv());
  fake.userRankResult = ncclSuccess;
  fake.findWindowFailPtr = send_.data();
  EXPECT_EQ(ncclInternalError, AlltoAllv());
  fake.findWindowFailPtr = recv_.data();
  EXPECT_EQ(ncclInternalError, AlltoAllv());
  fake.findWindowFailPtr = nullptr;
  fake.regTypeResult = ncclInvalidUsage;
  EXPECT_EQ(ncclInvalidUsage, AlltoAllv());
  EXPECT_TRUE(fake.captureModes.empty());
  fake.regTypeResult = ncclSuccess;
  fake.captureResult = ncclRemoteError;
  EXPECT_EQ(ncclRemoteError, AlltoAllv());
  EXPECT_TRUE(fake.ceEligCalls.empty());
  EXPECT_TRUE(fake.p2ps.empty());
}

// A failed post returns without ncclGroupEnd or skip(false), so the caller's group stays open; a fix flips this pin.
TEST_F(CollectivesAlltoAllvMicrotest, PeerToPeer_KnownBug_FailureLeavesTheGroupOpen) {
  fake.groupStartResult = ncclInvalidUsage;
  EXPECT_EQ(ncclInvalidUsage, AlltoAllv());
  EXPECT_TRUE(fake.p2ps.empty());
  fake.groupStartResult = ncclSuccess;
  for (int failAt : {3, 4}) {
    SCOPED_TRACE(failAt);
    fake.p2ps.clear();
    fake.recorderSkips.clear();
    fake.trace.clear();
    fake.p2pFailAt = failAt;
    EXPECT_EQ(ncclInternalError, AlltoAllv());
    EXPECT_EQ(static_cast<size_t>(failAt), fake.p2ps.size());
    EXPECT_EQ(std::vector<bool>{true}, fake.recorderSkips);
    EXPECT_EQ(0, std::count(fake.trace.begin(), fake.trace.end(), "groupEnd"));
  }
  fake.p2pFailAt = 0;
  fake.recorderSkips.clear();
  fake.groupEndResult = ncclRemoteError;
  EXPECT_EQ(ncclRemoteError, AlltoAllv());
  EXPECT_EQ(std::vector<bool>{true}, fake.recorderSkips);
}

TEST_F(CollectivesAlltoAllvMicrotest, Ce_ExchangesTheSizeMatrixThenEnqueuesIt) {
  fake.ceElig = true;
  ScriptPeerBlocks();
  fake.enqueueResults = {ncclInProgress};
  EXPECT_EQ(ncclInProgress, AlltoAllv());
  const size_t nLocal = 4 * kNRanks;
  EXPECT_EQ((std::vector<size_t>{nLocal * sizeof(size_t), nLocal * kNRanks * sizeof(size_t)}), fake.memAllocSizes);
  EXPECT_EQ((std::vector<std::string>{"record", "findWindow", "findWindow", "setDevice 3", "memAlloc", "memAlloc",
                                      "setDevice 9", "memcpy", "groupStart", "groupEnd", "memcpy", "sync",
                                      "enqueue AlltoAllv"}),
            fake.trace);
  EXPECT_EQ(9, fake.currentDevice);

  ASSERT_EQ(2u, fake.memcpys.size());
  EXPECT_EQ(comm_->localSizes, fake.memcpys[0].dst);
  EXPECT_EQ(nLocal * sizeof(size_t), fake.memcpys[0].bytes);
  EXPECT_EQ(hipMemcpyHostToDevice, fake.memcpys[0].kind);
  EXPECT_EQ(stream_, fake.memcpys[0].stream);
  const size_t* local = static_cast<const size_t*>(comm_->localSizes);
  EXPECT_EQ(LocalBlock(), std::vector<size_t>(local, local + nLocal));
  EXPECT_EQ(comm_->gatheredSizes, fake.memcpys[1].src);
  EXPECT_EQ(nLocal * kNRanks * sizeof(size_t), fake.memcpys[1].bytes);
  EXPECT_EQ(hipMemcpyDeviceToHost, fake.memcpys[1].kind);

  ASSERT_EQ(2u * kNRanks, fake.p2ps.size());
  for (int r = 0; r < kNRanks; r++) {
    SCOPED_TRACE(r);
    EXPECT_TRUE(fake.p2ps[2 * r].send);
    EXPECT_EQ(comm_->localSizes, fake.p2ps[2 * r].buff);
    EXPECT_EQ(nLocal, fake.p2ps[2 * r].count);
    EXPECT_EQ(ncclUint64, fake.p2ps[2 * r].datatype);
    EXPECT_EQ(r, fake.p2ps[2 * r].peer);
    EXPECT_FALSE(fake.p2ps[2 * r + 1].send);
    EXPECT_EQ(static_cast<size_t*>(comm_->gatheredSizes) + r * nLocal, fake.p2ps[2 * r + 1].buff);
    EXPECT_EQ(nLocal, fake.p2ps[2 * r + 1].count);
    EXPECT_EQ(ncclUint64, fake.p2ps[2 * r + 1].datatype);
    EXPECT_EQ(r, fake.p2ps[2 * r + 1].peer);
  }

  ASSERT_EQ(1u, fake.enqueued.size());
  const ncclInfo& info = Info();
  EXPECT_EQ(ncclFuncAlltoAllv, info.coll);
  EXPECT_STREQ("AlltoAllv", info.opName);
  EXPECT_EQ(send_.data(), info.sendbuff);
  EXPECT_EQ(recv_.data(), info.recvbuff);
  EXPECT_EQ(0u, info.count);
  EXPECT_EQ(ncclFloat32, info.datatype);
  EXPECT_EQ(ALLTOALL_CHUNKSTEPS, info.chunkSteps);
  EXPECT_EQ(ALLTOALL_SLICESTEPS, info.sliceSteps);
  std::vector<size_t> expected;
  for (int r = 0; r < kNRanks; r++) expected.insert(expected.end(), fake.peerMeta[r].begin(), fake.peerMeta[r].end());
  EXPECT_EQ(expected, fake.enqueued[0].sizes);
  EXPECT_TRUE(fake.recorderSkips.empty());
}

TEST_F(CollectivesAlltoAllvMicrotest, Ce_ReusesTheMetadataBuffers) {
  fake.hierElig = true;
  EXPECT_EQ(ncclSuccess, AlltoAllv());
  void* local = comm_->localSizes;
  void* gathered = comm_->gatheredSizes;
  fake.trace.clear();
  EXPECT_EQ(ncclSuccess, AlltoAllv());
  EXPECT_EQ(2u, fake.memAllocSizes.size());
  EXPECT_EQ(0, std::count(fake.trace.begin(), fake.trace.end(), "setDevice 3"));
  EXPECT_EQ(local, comm_->localSizes);
  EXPECT_EQ(gathered, comm_->gatheredSizes);
  EXPECT_EQ(2u, fake.enqueued.size());
}

TEST_F(CollectivesAlltoAllvMicrotest, Ce_FailedAllocationRestoresTheDevice) {
  fake.ceElig = true;
  fake.memAllocFailAt = 1;
  EXPECT_EQ(ncclSystemError, AlltoAllv());
  EXPECT_EQ(1u, fake.memAllocSizes.size());
  EXPECT_EQ(9, fake.currentDevice);
  EXPECT_EQ(nullptr, comm_->localSizes);

  fake.memAllocSizes.clear();
  fake.memAllocFailAt = 2;
  EXPECT_EQ(ncclSystemError, AlltoAllv());
  EXPECT_EQ(9, fake.currentDevice);
  EXPECT_NE(nullptr, comm_->localSizes);
  EXPECT_EQ(nullptr, comm_->gatheredSizes);
  EXPECT_TRUE(fake.p2ps.empty());

  fake.memAllocSizes.clear();
  fake.memAllocFailAt = 0;
  EXPECT_EQ(ncclSuccess, AlltoAllv());
  EXPECT_EQ(std::vector<size_t>{4 * kNRanks * kNRanks * sizeof(size_t)}, fake.memAllocSizes);
}

TEST_F(CollectivesAlltoAllvMicrotest, Ce_DeviceSwitchFailuresStopTheCall) {
  fake.ceElig = true;
  fake.getDeviceResult = hipErrorInvalidContext;
  EXPECT_EQ(ncclUnhandledCudaError, AlltoAllv());
  EXPECT_EQ(0, fake.setDeviceCalls);
  fake.getDeviceResult = hipSuccess;
  fake.setDeviceFailAt = 1;
  EXPECT_EQ(ncclUnhandledCudaError, AlltoAllv());
  EXPECT_TRUE(fake.memAllocSizes.empty());
  fake.setDeviceCalls = 0;
  fake.setDeviceFailAt = 2;
  EXPECT_EQ(ncclUnhandledCudaError, AlltoAllv());
  EXPECT_EQ(2u, fake.memAllocSizes.size());
  EXPECT_TRUE(fake.p2ps.empty());
}

// A failed size-exchange post returns without ncclGroupEnd, so the caller's group stays open; a fix flips this pin.
TEST_F(CollectivesAlltoAllvMicrotest, Ce_KnownBug_FailedExchangePostLeavesTheGroupOpen) {
  fake.ceElig = true;
  for (int failAt : {1, 2}) {
    SCOPED_TRACE(failAt);
    fake.p2ps.clear();
    fake.trace.clear();
    fake.p2pFailAt = failAt;
    EXPECT_EQ(ncclInternalError, AlltoAllv());
    EXPECT_EQ(static_cast<size_t>(failAt), fake.p2ps.size());
    EXPECT_EQ(0, std::count(fake.trace.begin(), fake.trace.end(), "groupEnd"));
  }
}

TEST_F(CollectivesAlltoAllvMicrotest, Ce_ExchangeFailuresReturnTheFirstError) {
  fake.ceElig = true;
  fake.memcpyResults = {hipErrorInvalidValue};
  EXPECT_EQ(ncclUnhandledCudaError, AlltoAllv());
  EXPECT_TRUE(fake.p2ps.empty());

  fake.groupStartResult = ncclInvalidUsage;
  EXPECT_EQ(ncclInvalidUsage, AlltoAllv());
  fake.groupStartResult = ncclSuccess;
  fake.groupEndResult = ncclRemoteError;
  EXPECT_EQ(ncclRemoteError, AlltoAllv());
  fake.groupEndResult = ncclSuccess;
  fake.memcpys.clear();
  fake.memcpyResults = {hipSuccess, hipErrorInvalidValue};
  EXPECT_EQ(ncclUnhandledCudaError, AlltoAllv());
  EXPECT_EQ(2u, fake.memcpys.size());
  fake.syncResult = hipErrorLaunchFailure;
  EXPECT_EQ(ncclUnhandledCudaError, AlltoAllv());
  EXPECT_TRUE(fake.enqueued.empty());
}

// alltoallv_meta.h

TEST_F(CollectivesMicrotest, AlltoAllvMeta_BlocksAreRankMajorQuadruples) {
  constexpr int n = 3;
  std::vector<size_t> gathered(4 * n * n);
  for (size_t i = 0; i < gathered.size(); i++) gathered[i] = i;
  size_t* g = gathered.data();
  EXPECT_EQ(24u, ncclAlltoAllvMetaBlockOffset(2, n));
  EXPECT_EQ(g + 12, ncclAlltoAllvRankMetaBlock(g, 1, n));
  EXPECT_EQ(g + 12, ncclAlltoAllvSendSizes(g, 1, n));
  EXPECT_EQ(g + 15, ncclAlltoAllvSendDispls(g, 1, n));
  EXPECT_EQ(g + 18, ncclAlltoAllvRecvSizes(g, 1, n));
  EXPECT_EQ(g + 21, ncclAlltoAllvRecvDispls(g, 1, n));
  EXPECT_EQ(12u + 13u + 14u, ncclAlltoAllvTrafficBytes(g, 1, n));
}

TEST_F(CollectivesMicrotest, AlltoAllvMeta_PackLaysOutCountsThenDisplacements) {
  const size_t sc[2] = {1, 2}, sd[2] = {3, 4}, rc[2] = {5, 6}, rd[2] = {7, 8};
  std::vector<size_t> sizes(8);
  ncclAlltoAllvPackLocalSizes(sizes.data(), 2, sc, sd, rc, rd);
  EXPECT_EQ((std::vector<size_t>{1, 2, 3, 4, 5, 6, 7, 8}), sizes);
}

// Each send size from src to dst must equal dst's recv size from src.
TEST_F(CollectivesMicrotest, AlltoAllvMeta_ValidateMatchesSendsAgainstReceives) {
  constexpr int n = 3;
  std::vector<size_t> gathered(4 * n * n, 0);
  size_t* g = gathered.data();
  for (int src = 0; src < n; src++) {
    for (int dst = 0; dst < n; dst++) {
      ncclAlltoAllvSendSizes(g, src, n)[dst] = 10 * src + dst + 1;
      ncclAlltoAllvRecvSizes(g, dst, n)[src] = 10 * src + dst + 1;
    }
  }
  EXPECT_EQ(ncclSuccess, ncclAlltoAllvValidateSizeMatrix(g, n));
  ncclAlltoAllvSendDispls(g, 1, n)[2] = 99;  // displacements are not cross-checked
  EXPECT_EQ(ncclSuccess, ncclAlltoAllvValidateSizeMatrix(g, n));
  ncclAlltoAllvRecvSizes(g, 2, n)[1] += 1;
  EXPECT_EQ(ncclInvalidUsage, ncclAlltoAllvValidateSizeMatrix(g, n));
  ncclAlltoAllvRecvSizes(g, 2, n)[1] -= 1;
  ncclAlltoAllvSendSizes(g, 2, n)[2] = 0;
  EXPECT_EQ(ncclInvalidUsage, ncclAlltoAllvValidateSizeMatrix(g, n));
}

// include/collectives.h: ring address generators the net proxy walks for registered buffers.

struct RingSlot {
  int buf;  // 0 send buffer, 1 recv buffer
  ssize_t offset;
  size_t size;
  void* mhandle;
};

class CollectivesRingMicrotest : public ::testing::Test {
 protected:
  static constexpr int kN = 4;
  static constexpr size_t kGrid = 64;

  void SetUp() override {
    for (int r = 0; r < kN; r++) {
      send_[r].assign(kBuf, 0);
      recv_[r].assign(kBuf, 0);
    }
  }
  RingSlot Classify(int rank, uint8_t* addr, size_t size, void* mhandle) {
    uint8_t* send = send_[rank].data() + kGrid;
    uint8_t* recv = recv_[rank].data() + kGrid;
    if (addr >= recv_[rank].data() && addr < recv_[rank].data() + kBuf) return {1, addr - recv, size, mhandle};
    EXPECT_TRUE(addr >= send_[rank].data() && addr < send_[rank].data() + kBuf) << "address outside both buffers";
    return {0, addr - send, size, mhandle};
  }
  RingSlot Send(RingAlgorithm& algo, int rank, int step) {
    uint8_t* addr = nullptr;
    size_t size = 1234;
    void* mh = nullptr;
    algo.getNextSendAddr(step, &addr, &size, &mh);
    return Classify(rank, addr, size, mh);
  }
  RingSlot Recv(RingAlgorithm& algo, int rank, int step) {
    uint8_t* addr = nullptr;
    size_t size = 1234;
    void* mh = nullptr;
    algo.getNextRecvAddr(step, &addr, &size, &mh);
    uint8_t* again = nullptr;
    algo.getNextRecvAddr(step, &again, nullptr, &mh);  // the proxy also asks without a size
    EXPECT_EQ(addr, again);
    return Classify(rank, addr, size, mh);
  }

  static constexpr size_t kBuf = 16384;
  std::vector<uint8_t> send_[kN];
  std::vector<uint8_t> recv_[kN];
  int sendMh_ = 0, recvMh_ = 0, srecvMh_ = 0;
};

// AllReduce: 4 ranks, 512-byte chunks in 2 slices, 2 full loops plus a 300-byte tail.
class CollectivesRingArMicrotest : public CollectivesRingMicrotest {
 protected:
  static constexpr int kChunkSteps = 4, kSliceSteps = 2;
  static constexpr size_t kChunk = 512, kSlice = 256, kLoop = kN * kChunk, kChannel = 2 * kLoop + 300;
  static constexpr int kStepsPerLoop = 2 * (kN - 1) * kChunkSteps;

  std::unique_ptr<RingARAlgorithm> Make(int rank) {
    return std::make_unique<RingARAlgorithm>(send_[rank].data(), recv_[rank].data(), kN, rank, kChunkSteps,
                                             kSliceSteps, kChunk, kSlice, kGrid, kChannel, sizeof(float), &sendMh_,
                                             &recvMh_, &srecvMh_);
  }
};

TEST_F(CollectivesRingArMicrotest, EverySendLandsWhereTheNextRankReceives) {
  for (int rank = 0; rank < kN; rank++) {
    auto sender = Make(rank);
    auto receiver = Make((rank + 1) % kN);
    for (int step = 0; step < 3 * kStepsPerLoop; step++) {
      SCOPED_TRACE("rank " + std::to_string(rank) + " step " + std::to_string(step));
      RingSlot s = Send(*sender, rank, step);
      RingSlot r = Recv(*receiver, (rank + 1) % kN, step);
      EXPECT_EQ(1, r.buf);
      EXPECT_EQ(&recvMh_, r.mhandle);
      EXPECT_EQ(s.offset, r.offset);
      EXPECT_EQ(s.size, r.size);
    }
  }
}

// Reduce-scatter then allgather: stage k sends chunk (ring index - 1 - k), the first from the user's input.
TEST_F(CollectivesRingArMicrotest, StagesWalkTheChunksBackwardsAroundTheRing) {
  const int rank = 2;
  auto algo = Make(rank);
  for (int loop = 0; loop < 2; loop++) {
    for (int stage = 0; stage < 2 * (kN - 1); stage++) {
      SCOPED_TRACE("loop " + std::to_string(loop) + " stage " + std::to_string(stage));
      const int step = loop * kStepsPerLoop + stage * kChunkSteps;
      const size_t chunk = (rank - 1 - stage + 2 * kN) % kN;
      RingSlot first = Send(*algo, rank, step);
      RingSlot second = Send(*algo, rank, step + kSliceSteps);
      EXPECT_EQ(stage == 0 ? 0 : 1, first.buf);
      EXPECT_EQ(stage == 0 ? static_cast<void*>(&sendMh_) : &srecvMh_, first.mhandle);
      EXPECT_EQ(static_cast<ssize_t>(loop * kLoop + chunk * kChunk), first.offset);
      EXPECT_EQ(first.offset + static_cast<ssize_t>(kSlice), second.offset);
      EXPECT_EQ(kSlice, first.size);
      EXPECT_EQ(kSlice, second.size);
      EXPECT_EQ(Send(*algo, rank, step + 1).offset, first.offset);  // steps within a slice share it
    }
  }
}

// The 300-byte tail splits into 16-byte-aligned chunks of 80 bytes; slices past a chunk's end are empty.
TEST_F(CollectivesRingArMicrotest, TailLoopSplitsTheRemainderAcrossRanks) {
  const int rank = 1;
  auto algo = Make(rank);
  size_t total = 0;
  for (int stage = 0; stage < kN; stage++) {
    SCOPED_TRACE(stage);
    const int step = 2 * kStepsPerLoop + stage * kChunkSteps;
    const size_t chunk = (rank - 1 - stage + 2 * kN) % kN;
    const size_t chunkBytes = std::min<size_t>(80, 300 - chunk * 80);
    RingSlot first = Send(*algo, rank, step);
    RingSlot second = Send(*algo, rank, step + kSliceSteps);
    EXPECT_EQ(static_cast<ssize_t>(2 * kLoop + chunk * 80), first.offset);
    EXPECT_EQ(chunkBytes, first.size + second.size);
    if (chunkBytes <= 64) {
      EXPECT_EQ(0, second.buf);  // nothing left: the base pointer and the send handle
      EXPECT_EQ(0, second.offset);
      EXPECT_EQ(&sendMh_, second.mhandle);
      EXPECT_EQ(0u, second.size);
    }
    total += first.size + second.size;
  }
  EXPECT_EQ(300u, total);
  uint8_t* addr = nullptr;
  size_t size = 1;
  void* mh = nullptr;
  algo->getNextRecvAddr(2 * kStepsPerLoop + kSliceSteps, &addr, &size, &mh);  // chunk 3's second slice
  EXPECT_EQ(recv_[rank].data() + kGrid, addr);  // a receive past its chunk's end also sits at the base
  EXPECT_EQ(0u, size);
}

// AllGather: ring order 0 -> 2 -> 1 -> 3, 256-byte loops in 2 slices, 1 KiB per-rank blocks.
class CollectivesRingAgMicrotest : public CollectivesRingMicrotest {
 protected:
  static constexpr int kChunkSteps = 2, kSliceSteps = 1;
  static constexpr size_t kChunk = 256, kSlice = 128, kChannel = 600, kBlock = 1024;
  static constexpr int kStepsPerLoop = (kN - 1) * kChunkSteps;

  std::unique_ptr<RingAGAlgorithm> Make(int pos) {
    for (int i = 0; i < kN; i++) ringRanks_[pos][i] = kRing[(pos + i) % kN];
    const int rank = kRing[pos];
    return std::make_unique<RingAGAlgorithm>(send_[rank].data(), recv_[rank].data(), kN, ringRanks_[pos],
                                             kChunkSteps, kSliceSteps, kChunk, kSlice, kGrid, kChannel,
                                             sizeof(float), kBlock, &sendMh_, &recvMh_, &srecvMh_);
  }
  // Which rank's block, and where inside it, a send or receive touches.
  std::pair<int, ssize_t> Block(const RingSlot& slot, int self) {
    if (slot.buf == 0) return {self, slot.offset};
    return {static_cast<int>(slot.offset / kBlock), slot.offset % kBlock};
  }

  static constexpr int kRing[kN] = {0, 2, 1, 3};
  int ringRanks_[kN][kN] = {};
};

TEST_F(CollectivesRingAgMicrotest, EverySendLandsWhereTheNextRankReceives) {
  for (int pos = 0; pos < kN; pos++) {
    const int self = kRing[pos], next = kRing[(pos + 1) % kN];
    auto sender = Make(pos);
    auto receiver = Make((pos + 1) % kN);
    for (int step = 0; step < 3 * kStepsPerLoop; step++) {
      SCOPED_TRACE("rank " + std::to_string(self) + " step " + std::to_string(step));
      RingSlot s = Send(*sender, self, step);
      RingSlot r = Recv(*receiver, next, step);
      EXPECT_EQ(1, r.buf);
      EXPECT_EQ(&recvMh_, r.mhandle);
      EXPECT_EQ(Block(s, self), Block(r, next));
      EXPECT_GE(r.size, s.size);  // receives post the full slice
    }
  }
}

TEST_F(CollectivesRingAgMicrotest, StageZeroSendsTheOwnInputThenForwardsPeerBlocks) {
  const int pos = 1, self = kRing[pos];
  auto algo = Make(pos);
  std::set<int> forwarded;
  for (int stage = 0; stage < kN - 1; stage++) {
    RingSlot s = Send(*algo, self, stage * kChunkSteps);
    EXPECT_EQ(stage == 0 ? 0 : 1, s.buf) << stage;
    EXPECT_EQ(stage == 0 ? static_cast<void*>(&sendMh_) : &srecvMh_, s.mhandle) << stage;
    forwarded.insert(Block(s, self).first);
  }
  EXPECT_EQ((std::set<int>{self, kRing[0], kRing[3]}), forwarded);  // everything but the next rank's block
}

TEST_F(CollectivesRingAgMicrotest, TailLoopSendsOnlyTheRemainder) {
  auto algo = Make(0);
  size_t perLoop[3] = {};
  for (int step = 0; step < 3 * kStepsPerLoop; step += kStepsPerLoop) {
    for (int slice = 0; slice < kChunkSteps; slice++) {
      perLoop[step / kStepsPerLoop] += Send(*algo, kRing[0], step + slice).size;
    }
  }
  EXPECT_EQ(kChunk, perLoop[0]);
  EXPECT_EQ(kChunk, perLoop[1]);
  EXPECT_EQ(kChannel - 2 * kChunk, perLoop[2]);
  RingSlot past = Send(*algo, kRing[0], 3 * kStepsPerLoop);
  EXPECT_EQ(0u, past.size);
  // Receives post a full slice, clipped only by the channel end.
  EXPECT_EQ(kSlice, Recv(*algo, kRing[0], 0).size);
  EXPECT_EQ(88u, Recv(*algo, kRing[0], 2 * kStepsPerLoop).size);
  EXPECT_EQ(24u, Recv(*algo, kRing[0], 2 * kStepsPerLoop + 1).size);
}

// Broadcast: 128-byte loops in two 64-byte slices over a 300-byte channel.
class CollectivesRingBcMicrotest : public CollectivesRingMicrotest {
 protected:
  std::unique_ptr<RingBCAlgorithm> Make(int rank, int root, size_t channelSize = 300) {
    return std::make_unique<RingBCAlgorithm>(send_[rank].data(), recv_[rank].data(), rank, root, kN, ringRanks_, 4, 2,
                                             128, 64, kGrid, channelSize, &sendMh_, &recvMh_, &srecvMh_);
  }
  int ringRanks_[kN] = {0, 1, 2, 3};
};

TEST_F(CollectivesRingBcMicrotest, RootSendsItsInputOthersForwardTheirOutput) {
  const std::vector<std::pair<ssize_t, size_t>> expected = {{0, 64},   {0, 64},   {64, 64},  {64, 64},
                                                            {128, 64}, {128, 64}, {192, 64}, {192, 64},
                                                            {256, 44}, {256, 44}, {0, 0},    {0, 0}};
  auto root = Make(2, 2);
  auto other = Make(3, 2);
  for (int step = 0; step < 12; step++) {
    SCOPED_TRACE(step);
    const bool past = step >= 10;
    RingSlot rs = Send(*root, 2, step);
    EXPECT_EQ(0, rs.buf);
    EXPECT_EQ(&sendMh_, rs.mhandle);
    EXPECT_EQ(expected[step].first, rs.offset);
    EXPECT_EQ(expected[step].second, rs.size);
    RingSlot os = Send(*other, 3, step);
    EXPECT_EQ(past ? 0 : 1, os.buf);
    EXPECT_EQ(past ? static_cast<void*>(&sendMh_) : &srecvMh_, os.mhandle);
    EXPECT_EQ(expected[step].first, os.offset);
    EXPECT_EQ(expected[step].second, os.size);
    RingSlot r = Recv(*other, 3, step);
    EXPECT_EQ(1, r.buf);
    EXPECT_EQ(&recvMh_, r.mhandle);
    EXPECT_EQ(expected[step].first, r.offset);
    EXPECT_EQ(expected[step].second, r.size);
  }
}

// A slice starting exactly at the channel end is empty and sits at the buffer base.
TEST_F(CollectivesRingBcMicrotest, SliceAtTheChannelEndIsEmpty) {
  auto other = Make(3, 2, 256);
  RingSlot s = Send(*other, 3, 8);
  EXPECT_EQ(0, s.buf);
  EXPECT_EQ(0, s.offset);
  EXPECT_EQ(0u, s.size);
  RingSlot r = Recv(*other, 3, 8);
  EXPECT_EQ(0, r.offset);
  EXPECT_EQ(0u, r.size);
}

TEST_F(CollectivesRingBcMicrotest, RefCountTracksTheProxyCopies) {
  std::unique_ptr<RingAlgorithm> algo = Make(0, 0);
  EXPECT_EQ(1, algo->incRefCount());
  EXPECT_EQ(2, algo->incRefCount());
  EXPECT_EQ(1, algo->decRefCount());
  EXPECT_EQ(0, algo->decRefCount());
}

// include/collectives.h: PAT step generators. The proxy counts each rank's posts per dimension
// (proxy.cc ncclPatternPatUp/PatDown), so every post must be met by the peer on that dimension.

struct PatPosts {
  std::vector<int> send;  // per dimension
  std::vector<int> recv;
  std::vector<ncclPatStep> ops;
};

template <typename Algo>
PatPosts RunPat(Algo& algo, int nranks) {
  PatPosts posts;
  posts.send.assign(log2Up(nranks) + 1, 0);
  posts.recv.assign(log2Up(nranks) + 1, 0);
  ncclPatStep ps{};
  for (int i = 0; i < 1 << 20; i++) {
    algo.getNextOp(&ps);
    EXPECT_NE(0, ps.flags & PatUsed);
    posts.ops.push_back(ps);
    if (!(ps.flags & PatSkipped)) {
      if (ps.recvDim != -1 && ps.postRecv) posts.recv.at(ps.recvDim)++;
      if (ps.sendDim != -1 && ps.postSend) posts.send.at(ps.sendDim)++;
    }
    if (ps.last == 2) return posts;
  }
  ADD_FAILURE() << "PAT schedule did not finish";
  return posts;
}

struct PatShape {
  int stepSize;
  int maxParallelFactor;
  size_t size;
  int chunkCount;
};

const std::vector<PatShape> kPatShapes = {
    {1024, 2, 64, 1024},     // one chunk, aggregated steps
    {4096, 4, 16, 4096},     // deep aggregation
    {256, 2, 1000, 256},     // several chunks, no aggregation
    {512, 1, 96, 32},        // several small chunks
    {65536, 8, 8, 65536},
};

template <typename T>
void ExpectPatPostsMatch(bool reduceScatter, int sharedConns) {
  for (int nranks : {2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 17}) {
    for (const PatShape& shape : kPatShapes) {
      SCOPED_TRACE("nranks " + std::to_string(nranks) + " size " + std::to_string(shape.size) + " step " +
                   std::to_string(shape.stepSize));
      std::vector<PatPosts> posts;
      for (int rank = 0; rank < nranks; rank++) {
        if (reduceScatter) {
          PatRSAlgorithm<T> algo(shape.stepSize, NCCL_STEPS, shape.maxParallelFactor, 0, shape.size, shape.size,
                                 shape.chunkCount, rank, nranks);
          posts.push_back(RunPat(algo, nranks));
        } else {
          PatAGAlgorithm<T> algo(shape.stepSize, NCCL_STEPS, shape.maxParallelFactor, 0, shape.size, shape.size,
                                 shape.chunkCount, rank, nranks, sharedConns);
          posts.push_back(RunPat(algo, nranks));
        }
      }
      // ReduceScatter and shared-connection AllGather send up the ring of powers of two; plain AllGather down.
      const bool up = reduceScatter || sharedConns;
      for (int rank = 0; rank < nranks; rank++) {
        for (int dim = 0; (1 << dim) < nranks; dim++) {
          const int peer = up ? (rank + (1 << dim)) % nranks : (rank - (1 << dim) + nranks) % nranks;
          EXPECT_EQ(posts[rank].send[dim], posts[peer].recv[dim]) << "rank " << rank << " dim " << dim;
        }
        EXPECT_EQ(0, posts[rank].send[log2Up(nranks)]);
        EXPECT_EQ(0, posts[rank].recv[log2Up(nranks)]);
      }
    }
  }
}

TEST(CollectivesPatMicrotest, ReduceScatterPostsMatchAcrossPeers) { ExpectPatPostsMatch<char>(true, 0); }
TEST(CollectivesPatMicrotest, ReduceScatterPostsMatchForWideTypes) { ExpectPatPostsMatch<double>(true, 0); }
TEST(CollectivesPatMicrotest, AllGatherPostsMatchAcrossPeers) { ExpectPatPostsMatch<char>(false, 0); }
TEST(CollectivesPatMicrotest, SharedConnAllGatherPostsMatchAcrossPeers) { ExpectPatPostsMatch<float>(false, 1); }

TEST(CollectivesPatMicrotest, ReduceScatterTwoRanksSendsThePeerBlockThenReceivesItsOwn) {
  PatRSAlgorithm<char> algo(1024, NCCL_STEPS, 2, 0, 64, 64, 1024, 1, 2);
  EXPECT_EQ(1, algo.getParallelFactor());
  PatPosts posts = RunPat(algo, 2);
  ASSERT_EQ(2u, posts.ops.size());
  const ncclPatStep& send = posts.ops[0];
  EXPECT_EQ(0, send.sendDim);
  EXPECT_EQ(-1, send.recvDim);
  EXPECT_EQ(0u, send.inpIx);  // rank 0's block
  EXPECT_EQ(1, send.postSend);
  EXPECT_EQ(0, send.postRecv);
  EXPECT_EQ(64, send.nelem);
  EXPECT_EQ(0, send.last);
  EXPECT_EQ(PatUsed, send.flags);
  const ncclPatStep& recv = posts.ops[1];
  EXPECT_EQ(-1, recv.sendDim);
  EXPECT_EQ(0, recv.recvDim);
  EXPECT_EQ(64u, recv.inpIx);  // its own block
  EXPECT_EQ(1, recv.postRecv);
  EXPECT_EQ(0, recv.postSend);
  EXPECT_EQ(2, recv.last);
}

TEST(CollectivesPatMicrotest, AllGatherTwoRanksSendsItsBlockThenReceivesThePeers) {
  for (int shared : {0, 1}) {
    PatAGAlgorithm<char> algo(1024, NCCL_STEPS, 2, 0, 64, 64, 1024, 1, 2, shared);
    PatPosts posts = RunPat(algo, 2);
    ASSERT_EQ(2u, posts.ops.size()) << shared;
    EXPECT_EQ(0, posts.ops[0].sendDim);
    EXPECT_EQ(-1, posts.ops[0].recvDim);
    EXPECT_EQ(64u, posts.ops[0].outIx);
    EXPECT_EQ(1, posts.ops[0].postSend);
    EXPECT_EQ(-1, posts.ops[1].sendDim);
    EXPECT_EQ(0, posts.ops[1].recvDim);
    EXPECT_EQ(0u, posts.ops[1].outIx);
    EXPECT_EQ(1, posts.ops[1].postRecv);
    EXPECT_EQ(2, posts.ops[1].last);
  }
}

// Every rank ends each chunk by receiving its own fully reduced block.
TEST(CollectivesPatMicrotest, ReduceScatterReceivesEachOwnChunkOnceLast) {
  for (int nranks : {3, 4, 8, 13}) {
    for (int rank = 0; rank < nranks; rank++) {
      PatRSAlgorithm<char> algo(256, NCCL_STEPS, 2, 0, 1000, 1000, 256, rank, nranks);
      PatPosts posts = RunPat(algo, nranks);
      std::vector<size_t> own;
      for (const ncclPatStep& ps : posts.ops) {
        if (!(ps.flags & PatSkipped) && ps.sendDim == -1 && ps.recvDim == 0 && ps.postRecv) own.push_back(ps.inpIx);
      }
      EXPECT_EQ((std::vector<size_t>{rank * 1000u, rank * 1000u + 256, rank * 1000u + 512, rank * 1000u + 768}), own)
          << nranks << "/" << rank;
      EXPECT_EQ(2, posts.ops.back().last);
    }
  }
}

// Shared connections run the same schedule on mirrored peers, so each block index reflects through the rank.
TEST(CollectivesPatMicrotest, SharedConnAllGatherMirrorsTheBlockOrder) {
  for (int nranks : {3, 4, 5, 8, 12}) {
    for (const PatShape& shape : kPatShapes) {
      for (int rank = 0; rank < nranks; rank++) {
        SCOPED_TRACE(std::to_string(nranks) + "/" + std::to_string(rank) + " size " + std::to_string(shape.size));
        PatAGAlgorithm<char> plainAlgo(shape.stepSize, NCCL_STEPS, shape.maxParallelFactor, 0, shape.size,
                                       shape.size, shape.chunkCount, rank, nranks, 0);
        PatAGAlgorithm<char> sharedAlgo(shape.stepSize, NCCL_STEPS, shape.maxParallelFactor, 0, shape.size,
                                        shape.size, shape.chunkCount, rank, nranks, 1);
        PatPosts plain = RunPat(plainAlgo, nranks);
        PatPosts shared = RunPat(sharedAlgo, nranks);
        ASSERT_EQ(plain.ops.size(), shared.ops.size());
        for (size_t i = 0; i < plain.ops.size(); i++) {
          const ncclPatStep& a = plain.ops[i];
          const ncclPatStep& b = shared.ops[i];
          ASSERT_EQ(a.flags, b.flags) << i;
          EXPECT_EQ(a.sendDim, b.sendDim) << i;
          EXPECT_EQ(a.recvDim, b.recvDim) << i;
          if (a.flags & PatSkipped) continue;
          const size_t blockA = a.outIx / shape.size, blockB = b.outIx / shape.size;
          EXPECT_EQ(a.outIx % shape.size, b.outIx % shape.size) << i;
          EXPECT_EQ((2 * rank - static_cast<int>(blockA) + 2 * nranks) % nranks, static_cast<int>(blockB)) << i;
        }
      }
    }
  }
}

// The final reduce reads the slot where the left neighbour wrote that block's last dim-0 partial.
TEST(CollectivesPatMicrotest, ReduceScatterFinalReadMatchesTheLastPartialWritten) {
  for (int nranks : {3, 4, 5, 8, 16}) {
    for (const PatShape& shape : kPatShapes) {
      SCOPED_TRACE(std::to_string(nranks) + " size " + std::to_string(shape.size));
      std::vector<PatPosts> posts;
      for (int rank = 0; rank < nranks; rank++) {
        PatRSAlgorithm<char> algo(shape.stepSize, NCCL_STEPS, shape.maxParallelFactor, 0, shape.size, shape.size,
                                  shape.chunkCount, rank, nranks);
        posts.push_back(RunPat(algo, nranks));
      }
      for (int rank = 0; rank < nranks; rank++) {
        const PatPosts& left = posts[(rank - 1 + nranks) % nranks];
        for (const ncclPatStep& fin : posts[rank].ops) {
          if ((fin.flags & PatSkipped) || fin.sendDim != -1 || fin.recvDim != 0) continue;
          int lastSlot = -2;
          for (const ncclPatStep& ps : left.ops) {
            if (!(ps.flags & PatSkipped) && ps.sendDim == 0 && ps.inpIx == fin.inpIx) lastSlot = ps.sendOffset;
          }
          EXPECT_EQ(lastSlot, fin.recvOffset) << "rank " << rank << " inpIx " << fin.inpIx;
        }
      }
    }
  }
}

TEST(CollectivesPatMicrotest, ParallelFactorIsCappedByThePostFrequency) {
  PatRSAlgorithm<char> rs(4096, NCCL_STEPS, 8, 0, 16, 16, 4096, 0, 16);
  EXPECT_EQ(8, rs.getParallelFactor());
  PatRSAlgorithm<char> rsSmall(4096, NCCL_STEPS, 8, 0, 4096, 4096, 4096, 0, 16);
  EXPECT_EQ(1, rsSmall.getParallelFactor());
  PatAGAlgorithm<char> ag(4096, NCCL_STEPS, 8, 0, 16, 16, 4096, 0, 16, 0);
  EXPECT_EQ(8, ag.getParallelFactor());
  PatAGAlgorithm<char> agSmall(4096, NCCL_STEPS, 8, 0, 4096, 4096, 4096, 0, 16, 0);
  EXPECT_EQ(1, agSmall.getParallelFactor());
}

// Consecutive ops rotate through the parallel workers' slots in the shared NVLS buffer.
TEST(CollectivesPatMicrotest, NvlsOffsetRotatesThroughTheParallelSlots) {
  PatRSAlgorithm<char> rs(4096, NCCL_STEPS, 4, 0, 16, 16, 4096, 0, 16);
  PatAGAlgorithm<char> ag(4096, NCCL_STEPS, 4, 0, 16, 16, 4096, 0, 16, 0);
  ASSERT_EQ(4, rs.getParallelFactor());
  ASSERT_EQ(4, ag.getParallelFactor());
  ncclPatStep a{}, b{};
  for (int i = 0; i < 9; i++) {
    rs.getNextOp(&a);
    ag.getNextOp(&b);
    EXPECT_EQ((i % 4) * 16, a.nvlsOffset) << i;
    EXPECT_EQ((i % 4) * 16, b.nvlsOffset) << i;
  }
}

}  // namespace
