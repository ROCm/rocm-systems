/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Externals src/collectives.cc reaches, for rccl-UnitTestsMicroCollectives, which compiles it. The ones with
// a shared owner (hip_fakes, recorder_fakes, sym_kernels_fakes, dev_runtime_fakes, strongstream_stubs,
// bootstrap_stubs, nccl_stubs) are driven through that owner's hooks; this file defines the rest. Every call
// is recorded in `fake`, and `fake.trace` keeps the order of calls across all of them.

#ifndef RCCL_TEST_HOST_COLLECTIVES_MICRO_FAKES_H_
#define RCCL_TEST_HOST_COLLECTIVES_MICRO_FAKES_H_

#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "comm.h"
#include "enqueue.h"
#include "rccl_common.h"
#include "ce_coll.h"
#include "dev_runtime.h"

// Backs collectives.cc's RCCL_PARAMs (via param_redirect.h). nccl_fakes.cc owns it elsewhere, but its
// ncclDebugLog only prints, and these tests assert on the level of what the unit logged.
extern std::function<int64_t(const char*, int64_t)> g_loadParam;

namespace collectives_micro {


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

// Every seam's knobs and observations, the shared-fakes hooks' included; ResetCollectivesMicroFakes() resets them.
struct Fakes {
  std::vector<LogLine> log;
  std::vector<std::string> trace;  // dependency calls in issue order
  std::vector<Enqueued> enqueued;
  std::deque<ncclResult_t> enqueueResults;  // consumed per call; empty means success
  const ncclCollConfig_t* parsedConfig = nullptr;
  ncclResult_t parseResult = ncclSuccess;
  std::vector<Recorded> recorded;
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
};
extern Fakes fake;

ncclResult_t RecordAddon(const char* fn, const void* s, void* r, size_t n, ncclDataType_t t, ncclRedOp_t o,
                         ncclComm* c, hipStream_t st);
void RecordLaunch(const void* fn, dim3 grid, dim3 block, void** args, size_t shmem, hipStream_t stream,
                  hipEvent_t stop);

}  // namespace collectives_micro

void ResetCollectivesMicroFakes();

#endif  // RCCL_TEST_HOST_COLLECTIVES_MICRO_FAKES_H_
