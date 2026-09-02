/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#ifndef RCCL_COMMON_H_
#define RCCL_COMMON_H_
#include "nccl_common.h"
#include "nccl.h"
#include "param.h"
#include "core.h"
#include "rccl_decision.h"

typedef enum RcclTunableColls {
  RCCL_UNSUPPORTED_TUNABLE = -1,
  RCCL_RS_TUNABLE = 0,    // reduce_scatter index
  RCCL_AG_TUNABLE = 1,    // all_gather index
  RCCL_AR_TUNABLE = 2,    // all_reduce index
  RCCL_RE_TUNABLE = 3,    // reduce index
  RCCL_BR_TUNABLE = 4,    // broadcast index
  RCCL_TUNABLE_COLLS = 5  // LL/LL64/LL128 tunable collectives count
} rcclTunableIndex_t;

#define CHAN_THRESHOLDS_UNDEFINED 0
#define RCCL_CHANNELS_TUNABLE_ENTRIES 9 // 2,4,8,16,32,40,48,56,64 channels

#define RCCL_LL_LIMITS_UNDEFINED 0
#define RCCL_PROTOCOL_ENTRY_SIZE 4
#define RCCL_PROTOCOL_MIN_IDX 0
#define RCCL_PROTOCOL_MAX_IDX 1
#define RCCL_PROTOCOL_FACTOR_IDX 2
#define RCCL_PROTOCOL_THREAD_THRESHOLD_IDX 3

#define RCCL_SINGLE_NODE_MAX_NTHREADS 256
#define RCCL_GFX950_MAX_NTHREADS 256  // for Simple and LL64/LL128 gfx950
#define RCCL_DEFAULT_MAX_NTHREADS 256 // for Simple and LL64/LL128 other archs
#define RCCL_LL_MAX_NTHREADS 256
#define RCCL_P2P_MAX_NTHREADS 256
#define RCCL_MI3XX_MAX_MULTI_NODE_CHANNELS 64
#define RCCL_MI3XX_MAX_SINGLE_NODE_CHANNELS 56

typedef enum {
  RCCL_VALUE_UNSET = -2,
  RCCL_VALUE_INVALID = -1
} rcclValueState_t;

// RCCL-specific entries in the unified algorithm/implementation identifier
// space. Values extend the native NCCL_ALGO_* range so a single integer (and a
// single rcclGetAlgoName() lookup) can name any backend RCCL might run. These
// are not just "algorithms" in the ring/tree sense — they include full backends
// (Symmetric, CE, DDA). See struct rcclCollDecision.
typedef enum {
  RCCL_DIRECT_ALLGATHER = NCCL_NUM_ALGORITHMS, // Direct AllGather
  RCCL_HIERARCHICAL_ALLGATHER, // Hierarchical AllGather
  RCCL_DIRECT_REDUCESCATTER, // Direct ReduceScatter (per-peer Send/Recv)
  RCCL_HIERARCHICAL_REDUCESCATTER, // Hierarchical ReduceScatter
#ifdef ENABLE_WARP_SPEED
  RCCL_WARP_SPEED,
#endif
  RCCL_SYMMETRIC,       // symmetric-window kernel
  RCCL_CE_2SHOT,        // eager Copy-Engine 2-shot AllReduce (staging buffer)
  RCCL_CE_REGISTERED,   // Copy-Engine via registered windows / CTA_POLICY_ZERO
  RCCL_DDA_FABRIC_LL,   // DDA fabric, LL protocol (small-message fast lane)
  RCCL_DDA_FABRIC_LL128,// DDA fabric, LL128 protocol (mid-message fast lane)
  RCCL_DDA_FABRIC_VMM,  // DDA fabric, VMM/Simple path
  RCCL_DDA_IPC,         // DDA IPC (single-node, fixed nRanks)
  RCCL_ALGO_COUNT
} rcclAddonAlgos_t;

#ifdef RCCL_EXPOSE_STATIC
#define RCCL_STATIC_EXPOSE_CHECK()
#else
#define RCCL_STATIC_EXPOSE_CHECK() \
  do { \
    WARN("Attempting to use internal logic while required static functions are not exposed. Rebuild with " \
         "RCCL_EXPOSE_STATIC enabled"); \
    return ncclInvalidUsage; \
  } while (0)
#endif

inline rcclTunableIndex_t rcclGetTunableIndex(ncclFunc_t const& func) {
  switch (func) {
  case ncclFuncReduceScatter:
    return RCCL_RS_TUNABLE;
  case ncclFuncAllGather:
    return RCCL_AG_TUNABLE;
  case ncclFuncAllReduce:
    return RCCL_AR_TUNABLE;
  case ncclFuncReduce:
    return RCCL_RE_TUNABLE;
  case ncclFuncBroadcast:
    return RCCL_BR_TUNABLE;
  default:
    return RCCL_UNSUPPORTED_TUNABLE; // Invalid or unsupported function
  }
}

inline size_t rcclGetSizePerRank(ncclFunc_t const& func, size_t const& nBytes, int const& nRanks) {
  // Normalize the comparison to sizePerRank as this is essentially what matters in determining protocol choice for the impacted collectives
  // For AG, this is the send size per rank
  // For RS, this is the recv size per rank
  // For AR, this is the send/recv size per rank
  return (func == ncclFuncReduceScatter || func == ncclFuncAllGather || func == ncclFuncBroadcast ||
          func == ncclFuncReduce) ?
           nBytes / nRanks :
           nBytes;
}
ncclResult_t rcclOverrideChannels(struct ncclComm* comm, ncclFunc_t coll, size_t nBytes, int& nc);
void rcclRestrictMaxChannels(struct ncclComm* comm, int& nc);
ncclResult_t rcclGetAlgoProtoIndex(const char* envStr, const char* algoProtoString[], int nEntries, int& result);
ncclResult_t rcclOverrideProtocol(const char* ncclProtoStr[], float table[][NCCL_NUM_PROTOCOLS],
                                  struct ncclTaskColl* info);
ncclResult_t rcclOverrideAlgorithm(const char* ncclAlgoStr[], float table[][NCCL_NUM_PROTOCOLS],
                                   struct ncclTaskColl* info);
void rcclUpdateCollectiveProtocol(struct ncclComm* comm, size_t const& nBytes, struct ncclTaskColl* info);
void rcclUpdateThreadThreshold(struct ncclComm* comm, size_t const& nBytes, struct ncclTaskColl* info,
                               int& threadThreshold);
void rcclSetPipelining(struct ncclComm* comm, size_t const& nBytes, struct ncclTaskColl* info);
void rcclGetMaxNthreads(struct ncclComm* comm, int maxNthreads[]);
void rcclOptThreadBlockSize(struct ncclComm* comm, struct ncclTaskColl* info, size_t nBytes, int& nThreads);
void rcclSetDefaultBuffSizes(struct ncclComm* comm, int defaultBuffSizes[]);
NCCL_API(ncclResult_t, rcclGetAlgoInfo, struct ncclComm* comm, ncclFunc_t coll, uint64_t count, ncclDataType_t dataType,
         int collNetSupport, int nvlsSupport, int numPipeOps, int* algo, int* protocol, int* maxChannels);
// Buffer/op-aware implementation query. Unlike rcclGetAlgoInfo(), this reports
// the full backend RCCL would actually run (CE, DDA, symmetric, or kernel) for
// the given operands, so rccl-tests can attribute numbers to the right label.
// `algo` returns a native NCCL_ALGO_* or rcclAddonAlgos_t value; name it with
// rcclGetAlgoName(). Currently implemented for AllReduce and AllGather; other
// collectives fall back to rcclGetAlgoInfo().
//
// graphCapturing: pass non-zero if the collective will execute under HIP/CUDA
// graph capture. This query is normally issued outside capture (before/after the
// captured run), so RCCL cannot detect graph mode from the stream on its own; the
// caller must declare it. It matters because CE is graph-unsafe and is disabled
// under capture, changing the selected backend (e.g. CE -> DDA/kernel).
NCCL_API(ncclResult_t, rcclGetCollImplInfo, struct ncclComm* comm, ncclFunc_t coll, uint64_t count,
         ncclDataType_t dataType, ncclRedOp_t op, const void* sendbuff, void* recvbuff, int graphCapturing, int* algo,
         int* protocol, int* maxChannels);
// Single source of truth for AllReduce implementation selection. Runs the exact
// priority chain (symmetric -> CE 2-shot -> DDA LL/LL128/VMM/IPC -> CE registered
// -> kernel) and returns the decision.
//   query=false : live dispatch path (ncclAllReduce_impl). ceCapturing is probed
//                 from `stream`; the CE graph latch is ticked; graphCapturingHint
//                 is ignored.
//   query=true  : side-effect-free reporting. The stream is not probed (the query
//                 runs outside capture); graphCapturingHint supplies the capture
//                 state so the reported backend matches a graph-mode run.
ncclResult_t rcclSelectAllReduce(struct ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                 ncclDataType_t datatype, ncclRedOp_t op, cudaStream_t stream, bool query,
                                 bool graphCapturingHint, struct rcclCollDecision* decision);
// Single source of truth for AllGather selection: DDA -> hierarchical -> CE ->
// direct -> ring. query=true fills protocol/nMaxChannels for reporting. CE dispatch
// lives in taskAppend(), so live returns RCCL_CE_REGISTERED but enqueues normally.
// graphCapturingHint (query only) suppresses the graph-unsafe CE branch under capture.
ncclResult_t rcclSelectAllGather(struct ncclComm* comm, const void* sendbuff, void* recvbuff, size_t sendcount,
                                 ncclDataType_t datatype, bool query, bool graphCapturingHint,
                                 struct rcclCollDecision* decision);
// Single source of truth for ReduceScatter selection: symmetric -> DDA fabric
// (LL/LL128/VMM) / DDA IPC -> hierarchical -> Direct -> native ring/pat kernel. RS has no CE.
ncclResult_t rcclSelectReduceScatter(struct ncclComm* comm, const void* sendbuff, void* recvbuff, size_t recvcount,
                                     ncclDataType_t datatype, ncclRedOp_t op, bool query,
                                     struct rcclCollDecision* decision);
// Selection helpers shared between collectives.cc and the wrapped decision logic.
// (rcclDdaEnabled is declared below, next to the DDA param decls.)
bool isSymmetricKernelRequested(struct ncclComm* comm, ncclFunc_t coll, int symkOp, ncclDataType_t datatype,
                                size_t nElts, const void* sendbuff, void* recvbuff, bool agreeAcrossRanks = false);
NCCL_API(ncclResult_t, rcclSymKGetInfo, struct ncclComm* comm, ncclFunc_t coll, uint64_t count, ncclDataType_t dataType,
         ncclRedOp_t op, int* algo, int* protocol, int* maxChannels);
NCCL_API(ncclResult_t, rcclGetAlgoName, int algo, const char** algoName);
NCCL_API(ncclResult_t, rcclGetProtocolName, int protocol, const char** algoName);
bool rcclUseAllGatherDirect(struct ncclComm* comm, size_t& msgSize);
bool rcclUseHierarchicalAllGather(struct ncclComm* comm, size_t msgSize);
bool rcclUseReduceScatterDirect(struct ncclComm* comm, size_t& msgSize);
bool rcclUseHierarchicalReduceScatter(struct ncclComm* comm, size_t msgSize);
size_t rcclHierarchicalTempBufferSize(int nNodes, bool allGather, bool reduceScatter);
// Fills in algo/protocol/channels for a hierarchical AllGather or ReduceScatter.
ncclResult_t rcclHierarchicalAlgoInfo(struct ncclComm* comm, ncclFunc_t coll, uint64_t count, ncclDataType_t dataType,
                                      int* algo, int* protocol, int* maxChannels);
bool rcclUseAlltoAllGda(struct ncclComm* comm);
// Returns true when the CE AllReduce path should be used instead of the standard ring/tree kernels.
// Pass the bias buffer as acc (nullptr when the caller is plain AllReduce).
// Does NOT check ceARTmpBuf initialization; the caller is responsible.
bool rcclUseCeAllReduce(struct ncclComm* comm, size_t count, ncclDataType_t datatype, ncclRedOp_t op, const void* acc);
// Updates the CE AllReduce graph latch from this call's capture state.
// Invoke once per collective (any type) at each CE AR decision point.
void rcclCeAllReduceGraphLatchTick(struct ncclComm* comm, bool ceCapturing);
// Pure query: is CE AllReduce currently allowed on this comm?
bool rcclCeAllReduceAllowed(struct ncclComm* comm);
// Decides whether ncclAllReduce_impl takes the DDA path for this call. Mirrors the guard in
// collectives.cc exactly: DDA runs when the buffers are not symmetric-kernel eligible, CE AllReduce
// will not service the call per the caller-computed `ceAllReduceAllowed` (non-gfx1250 only; gfx1250
// always keeps the DDA fabric path), and DDA is enabled for this arch/size. Host-side and GPU-free so
// the dispatch decision can be unit tested.
bool rcclAllReduceShouldTakeDdaPath(const struct ncclComm* comm, size_t count, ncclDataType_t datatype,
                                    bool symEligible, bool ceAllReduceAllowed);
void rcclSetPxn(struct ncclComm* comm, int& rcclPxnDisable);
void rcclSetP2pNetChunkSize(struct ncclComm* comm, int& rcclP2pNetChunkSize);
ncclResult_t rcclFuncMaxSendRecvCount(ncclFunc_t func, int nRanks, size_t count, size_t& maxCount);
ncclResult_t commSetUnrollFactor(struct ncclComm* comm);
ncclResult_t rcclCommSetP2pShiftSize(struct ncclComm* comm);
bool validHsaScratchEnvSetting(const char* hsaScratchEnv, int hipRuntimeVersion, int firmwareVersion,
                               const char* archName);

// Direct ReduceScatter Limit
RCCL_PARAM_DECLARE(DirectReduceScatterThreshold);
// Hierarchical AllGather enabled
RCCL_PARAM_DECLARE(HierarchicalAllGather);
// Hierarchical ReduceScatter enabled
RCCL_PARAM_DECLARE(HierarchicalReduceScatter);
#define HIERARCHICAL_TEMP_BUFFER_SIZE (128 * 1024 * 1024) // 128MB

// DDA threshold
RCCL_PARAM_DECLARE(DdaThreshold);
RCCL_PARAM_DECLARE(DdaLL);
RCCL_PARAM_DECLARE(DdaLLThreshold);
RCCL_PARAM_DECLARE(DdaLL128);
RCCL_PARAM_DECLARE(DdaLL128Threshold);
RCCL_PARAM_DECLARE(DdaEnable);

// Per-collective DDA AlltoAll thresholds (4 MiB for all supported archs).
constexpr size_t kDdaAlltoAllGfx942ThresholdBytes = 4194304;
constexpr size_t kDdaAlltoAllGfx950ThresholdBytes = 4194304;
constexpr size_t kDdaAlltoAllGfx1250ThresholdBytes = 4194304;

// Returns true when the DDA fast path should be attempted for a collective.
// Per-arch defaults cap the threshold; when 0, gfx950/gfx1250 fall back to
// the user-configurable RCCL_DDA_THRESHOLD env var.
bool rcclDdaEnabled(const ncclComm* comm, size_t totalBytes, size_t gfx942Default, size_t gfx950Default = 0,
                    size_t gfx1250Default = 0);

int getFirmwareVersion();
bool rcclIsArchSupportedForFunc(struct ncclTaskColl* info, char const* archName);

// Decide the host-side value of comm->cheapPostSendFenceOff.
// Returns 1 if the cheap post-send fence must be OFF (kernel uses the full
// __threadfence_system()), or 0 if the cheap post-send fence can be ON.
//   cudaArch             : numeric device arch (comm->cudaArch = 100*major +
//                          10*minor, i.e. gfx942 = 940, gfx950 = 950,
//                          gfx1250 = 1250).
//   param                : RCCL_CHEAP_POST_SEND_FENCE_OFF value
//                          (0 = arch-tuned auto, 1 = force off, 2 = force on).
//   uncachedMemSupported : whether cache-bypassing load/store builtins are
//                          available (HIP_UNCACHED_MEMORY); cheap fence is only
//                          safe when true.
inline int rcclComputeCheapPostSendFenceOff(int cudaArch, int64_t param, bool uncachedMemSupported) {
  // Cheap fence is only safe when cache-bypassing load/store builtins are available.
  if (!uncachedMemSupported) return 1;
  // Force cheap fence on regardless of arch (override auto, e.g. re-enable on gfx950).
  if (param == 2) return 0;
  // Any other non-zero value forces the full __threadfence_system().
  if (param != 0) return 1;
  // Arch-tuned auto: cheap fence on for gfx942 (940) and gfx1250 (1250);
  // off for gfx950 (950) and everything else.
  if (cudaArch == 940 || cudaArch == 1250) return 0;
  return 1;
}
#ifdef ENABLE_WARP_SPEED
RCCL_PARAM_DECLARE(WarpSpeedARThreshold);
RCCL_PARAM_DECLARE(WarpSpeedAutoMode);
void rcclSetWarpSpeedCUs(struct ncclComm* comm, int algo, int threadsPerBlock, int& rcclWarpSpeedChannels);
bool rcclWarpSpeedSupported(struct ncclComm* comm, struct ncclKernelPlan* plan);
ncclResult_t rcclSetWarpSpeedAuto(struct ncclComm* comm, struct ncclTaskColl* info, size_t nBytes);
int rcclGetMaxWarpsPerBlock(struct ncclComm* comm);
bool rcclCanUseWarpSpeedAuto(struct ncclComm* comm, int nNodes);
int rcclWarpSpeedComputeNChannels(struct ncclComm* comm, int nc, int channelMultiplier, int maxChannels,
                                  int adjustedMaxNchannels, bool userUpdatedMaxChannels);
int rcclWarpSpeedAdjustChannels(struct ncclComm* comm, struct ncclTaskColl* info, int nc);
#endif
#endif
