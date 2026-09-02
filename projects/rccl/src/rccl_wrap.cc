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

#include "rccl_common.h"
#include "comm.h"
#include "graph/topo.h"
#include "enqueue.h"
#include <algorithm>
#include "debug.h"
#include "net.h"
#include "amdsmi_wrap.h"
#include "include/graph.h"
#include "register.h"
#include "info.h"
#include "ce_coll.h"
#include "algorithms/dda/all_reduce/dda_all_reduce.h"
#include "algorithms/dda/all_gather/dda_all_gather.h"
#include "algorithms/dda/reduce_scatter/dda_reduce_scatter.h"
#include "group.h"
#include "sym_kernels.h"
#include "dev_runtime.h"
#include "strongstream.h"
#include "tuning.h"

// Use this param to experiment pipelining new data types besides bfloat16
// Make sure you generate the device code with the new data type (i.e. in generate.py)
RCCL_PARAM(PipelineAllDTypes, "PIPELINE_ALL_DATA_TYPES", 0);

// Use this to assess impact of pipelining on performance.
// Otherwise, it is automatically set for certain archs, datatypes and reduction collectives
RCCL_PARAM(disableReduceCopyPipelining, "DISABLE_REDUCE_COPY_PIPELINING", 0);
RCCL_PARAM(DirectAllGatherThreshold, "DIRECT_ALLGATHER_THRESHOLD", 75497472);
RCCL_PARAM(DirectReduceScatterThreshold, "DIRECT_REDUCE_SCATTER_THRESHOLD", 8388608);
RCCL_PARAM(DirectReduceScatterDisable, "DIRECT_REDUCE_SCATTER_DISABLE", 0);
RCCL_PARAM(DirectAllGatherDisable, "DIRECT_ALLGATHER_DISABLE", 0);
RCCL_PARAM(CeAllReduce, "CE_ALLREDUCE", 0);
RCCL_PARAM(ThreadsPerBlock, "THREADS_PER_BLOCK", -1);
RCCL_PARAM(UnrollFactor, "UNROLL_FACTOR", -1);
RCCL_PARAM(ForceCeAllReduce, "FORCE_CE_ALLREDUCE", 0);

// Common DDA protocol-tier knobs, shared by every fabric collective (no
// per-collective variants). For a given collective's size:
//   size <= DdaLLThreshold     -> LL    one-shot (16B lines)
//   size <= DdaLL128Threshold  -> LL128 one-shot (128B lines)
//   otherwise                  -> Simple (flat one-shot / tree two-shot)
// Constraint: DdaLLThreshold <= DdaLL128Threshold <= DdaThreshold. Setting an
// enable flag (or its threshold) to 0 disables that tier and falls through to
// the next, so each protocol can be A/B'd in isolation at runtime.
RCCL_PARAM(DdaEnable, "DDA_ENABLE", 1);
RCCL_PARAM(DdaThreshold, "DDA_THRESHOLD", (size_t)(134217728));           // 128 MiB
RCCL_PARAM(DdaLL, "DDA_LL", 1);
RCCL_PARAM(DdaLLThreshold, "DDA_LL_THRESHOLD", (size_t)(32768));          // 32 KiB
RCCL_PARAM(DdaLLOneShotThreshold, "DDA_LL_ONESHOT_THRESHOLD", (size_t)(1) * 1024 * 1024); // 1 MiB
RCCL_PARAM(DdaLLTwoShotThreshold, "DDA_LL_TWOSHOT_THRESHOLD", (size_t)(16) * 1024 * 1024); // 16 MiB
RCCL_PARAM(DdaLL128, "DDA_LL128", 0);
RCCL_PARAM(DdaLL128Threshold, "DDA_LL128_THRESHOLD", (size_t)(33554432)); // 32 MiB
#ifdef ENABLE_WARP_SPEED
RCCL_PARAM(WarpSpeedCuCount, "WARP_SPEED_CU_COUNT", 0);
RCCL_PARAM(WarpSpeedAutoMode, "WARP_SPEED_AUTO", 1);
RCCL_PARAM(WarpSpeedForceEnable, "WARP_SPEED_FORCE_ENABLE", 0);
RCCL_PARAM(WarpSpeedAGThreshold, "WARP_SPEED_AG_THRESHOLD", 134217728);   // 128 MB for AllGather
RCCL_PARAM(WarpSpeedRSThreshold, "WARP_SPEED_RS_THRESHOLD", 2147483648);  // 2 GB for ReduceScatter
RCCL_PARAM(WarpSpeedARThreshold, "WARP_SPEED_AR_THRESHOLD", 67108864);  // 64 MB for AllReduce
#endif

static inline bool rcclCollSupportsRing(ncclFunc_t func) {
  return (func == ncclFuncAllReduce || func == ncclFuncAllGather || func == ncclFuncReduceScatter ||
          func == ncclFuncBroadcast || func == ncclFuncReduce);
}

static inline bool rcclIsGfx120x(char const* arch) {
  return IsArchMatch(arch, "gfx1200") || IsArchMatch(arch, "gfx1201");
}

int32_t rcclGetProtoForGfx120x(ncclFunc_t collectiveFunc, size_t sizePerRank) {
  int returnVal = NCCL_PROTO_SIMPLE;
  int SingleNodeLLCutoffs[] = {/*ncclFuncBroadcast*/ 1536,
                               /*ncclFuncReduce*/ 8192,
                               /*ncclFuncAllGather*/ 98304,
                               /*ncclFuncReduceScatter*/ 98304,
                               /*ncclFuncAllReduce*/ 16384,
                               /*ncclFuncSendRecv*/ 0,
                               /*ncclFuncSend*/ 0,
                               /*ncclFuncRecv*/ 0};
  if (collectiveFunc < sizeof(SingleNodeLLCutoffs) / sizeof(int)) {
    returnVal = (sizePerRank <= SingleNodeLLCutoffs[collectiveFunc]) ? NCCL_PROTO_LL : NCCL_PROTO_SIMPLE;
  }
  return returnVal;
}

void rcclUpdateCollectiveProtocol(struct ncclComm* comm, size_t const& nBytes, struct ncclTaskColl* info) {
  // Honor user input for protocol choice
  static int userProtocolInput = -2;
  size_t sizePerRank = rcclGetSizePerRank(info->func, nBytes, comm->nRanks);
  if (userProtocolInput == -2) {
    const char* protoStr = getenv("NCCL_PROTO");
    userProtocolInput = !protoStr ? 0 : 1;
  }

  if (!userProtocolInput && IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950") && comm->nNodes == 1 &&
      (info->func == ncclFuncAllGather) && sizePerRank <= 88448) {
    // Change LL protocol threshold
    info->protocol = NCCL_PROTO_LL;
  } else if (!userProtocolInput && IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950") &&
             comm->nNodes == 1 && (info->func == ncclFuncReduceScatter) && sizePerRank <= 1048576) {
#ifdef ENABLE_WARP_SPEED
    if (sizePerRank <= 131072)
#endif
    {
      // Change LL protocol threshold
      info->protocol = NCCL_PROTO_LL;
    }
  } else if (!userProtocolInput && IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx942") &&
             comm->nNodes == 1 && (info->func == ncclFuncReduceScatter) && sizePerRank <= 352128) {
    // Change LL protocol threshold
    info->protocol = NCCL_PROTO_LL;
  } else if (!userProtocolInput && rcclIsGfx120x(comm->topo->nodes[GPU].nodes[0].gpu.gcn)) {
    if (comm->nNodes == 1) {
      info->protocol = rcclGetProtoForGfx120x(info->func, sizePerRank);
    }
    const char* str = ncclGetEnv("NCCL_P2P_DISABLE");
    if (str) {
      int disable = strtol(str, NULL, 0);
      if (disable == 1) {
        info->protocol = NCCL_PROTO_SIMPLE;
      }
    }
  } else if (!userProtocolInput && comm->nNodes >= 2 &&
             (info->func == ncclFuncReduceScatter || info->func == ncclFuncAllGather ||
              info->func == ncclFuncAllReduce || info->func == ncclFuncBroadcast || info->func == ncclFuncReduce)) {
    auto tunableIndex = rcclGetTunableIndex(info->func);
    auto llMin = comm->minMaxLLRange[tunableIndex][NCCL_PROTO_LL][RCCL_PROTOCOL_MIN_IDX];
    auto llMax = comm->minMaxLLRange[tunableIndex][NCCL_PROTO_LL][RCCL_PROTOCOL_MAX_IDX];

    auto ll128Min = comm->minMaxLLRange[tunableIndex][NCCL_PROTO_LL128][RCCL_PROTOCOL_MIN_IDX];
    auto ll128Max = comm->minMaxLLRange[tunableIndex][NCCL_PROTO_LL128][RCCL_PROTOCOL_MAX_IDX];

    // Only override model choices if min/max cutoff points are set in the tuning models
    if ((ll128Max != RCCL_LL_LIMITS_UNDEFINED) || (llMax != RCCL_LL_LIMITS_UNDEFINED)) {
      // Keep it simple unless otherwise required
      info->protocol = NCCL_PROTO_SIMPLE;
      if (sizePerRank <= llMax && sizePerRank > llMin) {
        info->protocol = NCCL_PROTO_LL;
      }
#if defined(ENABLE_LL128)
      // When LL128 is performant, the next condition overrides the previous LL choice
      if (comm->topo->ll128Enabled) {
        if (info->func == ncclFuncAllReduce) {
          if (comm->nNodes > 2) {
            ll128Max *= 3.8; // Scale max message size for n > 2 since Tree has special behavior at 2 nodes
          }
          // ll128Max += (log2i(comm->nNodes) - 1) * comm->minMaxLLRange[tunableIndex][NCCL_PROTO_LL128][RCCL_PROTOCOL_FACTOR_IDX];
        }
        if (sizePerRank <= ll128Max && sizePerRank > ll128Min) {
          info->protocol = NCCL_PROTO_LL128;
        }
      }
#endif
    } else if (IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx942") ||
               IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950")) {
      // Warn that model detection for the above listed architectures did not work as expected
      // Add supported archs to this condition as they come
      // Also make sure the tuning_model and model detection are updated for new archs
      static bool failedWarn = false;
      if (!failedWarn) {
        WARN("LL cutoff points not detected for a supported arch %s", comm->topo->nodes[GPU].nodes[0].gpu.gcn);
        failedWarn = true;
      }
    }
  }
}

ncclResult_t rcclGetAlgoProtoIndex(const char* envStr, const char* algoProtoString[], int nEntries, int& result) {
  if (envStr) {
    for (int i = 0; i < nEntries; ++i) {
      if (strcasecmp(envStr, algoProtoString[i]) == 0) {
        result = i;
        return ncclSuccess;
      }
    }
    static bool failedProtoWarn = false;
    if (!failedProtoWarn) {
      WARN("Invalid algo or protocol string passed %s", envStr);
      failedProtoWarn = true;
      return ncclInvalidUsage;
    }
  }
  return ncclInvalidUsage;
}

extern int64_t ncclParamMinNchannels();
extern int64_t ncclParamMaxNchannels();
extern int64_t rcclParamForceCe();
RCCL_PARAM(ChannelTuningEnable, "CHANNEL_TUNING_ENABLE", 1);

ncclResult_t rcclOverrideChannels(struct ncclComm* comm, ncclFunc_t coll, size_t nBytes, int& nc) {
  if (comm->nNodes < 2 || !rcclParamChannelTuningEnable()) {
    INFO(NCCL_TUNING, "RCCL Channel Tuning not applied");
    return ncclSuccess;
  }

  if ((comm->nRanks == comm->nNodes) && !IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx1151")) {
    INFO(NCCL_TUNING, "RCCL tuning model channel thresholds not applied for single GPU per node case");
    return ncclSuccess;
  }

  auto tunableIndex = rcclGetTunableIndex(coll);
  if (tunableIndex == RCCL_UNSUPPORTED_TUNABLE) {
    INFO(NCCL_TUNING, "tunableIndex:%i not supported", tunableIndex);
    return ncclSuccess;
  }

  int minCTAs = comm->config.minCTAs;
  int maxCTAs = comm->config.maxCTAs;
  int scalingFactor = 1;
#ifdef ENABLE_WARP_SPEED
  if (comm->topo->warpSpeedEnabled) {
    scalingFactor = comm->warpSpeedChannelMultiplier; // each CU can handle 4 warps
  }
#endif
  int minNChannels = ncclParamMinNchannels();
  int maxNChannels = std::max(comm->nChannels / scalingFactor, static_cast<int>(ncclParamMaxNchannels()));
  size_t bytesPerRank = divUp(nBytes, comm->nRanks);

  for (int channelCountIndex = 0; channelCountIndex < RCCL_CHANNELS_TUNABLE_ENTRIES; ++channelCountIndex) {
    size_t minByteThreshold = comm->minMaxChannelThresholds[tunableIndex][channelCountIndex][0];
    size_t maxByteThreshold = comm->minMaxChannelThresholds[tunableIndex][channelCountIndex][1];
    INFO(NCCL_TUNING,
         "nBytes:%lu bytesPerRank:%lu minByteThreshold:%lu maxByteThreshold:%lu  NCCL_MIN_NCHANNELS:%i or "
         "NCCL_MAX_NCHANNELS:%i minCTAs:%i maxCTAs:%i",
         nBytes, bytesPerRank, minByteThreshold, maxByteThreshold, minNChannels, maxNChannels, minCTAs, maxCTAs);
    if (minByteThreshold == CHAN_THRESHOLDS_UNDEFINED || maxByteThreshold == CHAN_THRESHOLDS_UNDEFINED) {
      INFO(NCCL_TUNING, "RCCL tuning model does not define threshold for coll:%i and nbytes:%lu", coll, nBytes);
      break; // Skip undefined thresholds
    }

    if (bytesPerRank > minByteThreshold && bytesPerRank <= maxByteThreshold) {
      int channelCount = comm->minMaxChannelThresholds[tunableIndex][channelCountIndex][2];

      // honor user's min/max channels defined through NCCL_MIN_NCHANNELS and NCCL_MAX_NCHANNELS
      if (channelCount >= minNChannels && channelCount <= maxNChannels && channelCount >= minCTAs &&
          channelCount <= maxCTAs) {
        nc = comm->minMaxChannelThresholds[tunableIndex][channelCountIndex][2];
        INFO(NCCL_TUNING,
             "RCCL tuning model overrides nchannels to %i, channels may be decreased further due to "
             "MinTrafficPerchannel thresholds",
             channelCount);
      } else {
        INFO(NCCL_TUNING,
             "RCCL tuning model cannot override nchannels to %i due to conflicting NCCL_MIN_NCHANNELS:%i or "
             "NCCL_MAX_NCHANNELS:%i minCTAs:%i maxCTAs:%i",
             channelCount, minNChannels, maxNChannels, minCTAs, maxCTAs);
      }

      break;
    }
  }
  return ncclSuccess;
}

ncclResult_t rcclOverrideProtocol(const char* ncclProtoStr[], float table[][NCCL_NUM_PROTOCOLS],
                                  struct ncclTaskColl* info) {
  static const char* protoOverrideEnv = ncclGetEnv("RCCL_OVERRIDE_PROTO");
  static bool validInput = true;
  if (!validInput) return ncclInvalidUsage;

  if (protoOverrideEnv) {
    static int protoVal = NCCL_PROTO_UNDEF;
    if (protoVal == NCCL_PROTO_UNDEF) {
      if (rcclGetAlgoProtoIndex(protoOverrideEnv, ncclProtoStr, NCCL_NUM_PROTOCOLS, protoVal) != ncclSuccess) {
        validInput = false;
        return ncclInvalidUsage;
      }
    }
    if (protoVal > NCCL_PROTO_UNDEF) {
      if (table[info->algorithm][protoVal] == NCCL_ALGO_PROTO_IGNORE) {
        WARN("Failed to force unsupported protocol %s for function %s with datatype %s", protoOverrideEnv,
             ncclFuncToString(info->func), ncclDatatypeToString(info->datatype));
        return ncclInternalError;
      } else {
        info->protocol = protoVal;
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t rcclOverrideAlgorithm(const char* ncclAlgoStr[], float table[][NCCL_NUM_PROTOCOLS],
                                   struct ncclTaskColl* info) {
  static const char* algoOverrideEnv = ncclGetEnv("RCCL_OVERRIDE_ALGO");
  static bool validInput = true;
  if (!validInput) return ncclInvalidUsage;

  if (algoOverrideEnv) {
    static int algoVal = NCCL_ALGO_UNDEF;
    if (algoVal == NCCL_ALGO_UNDEF) {
      if (rcclGetAlgoProtoIndex(algoOverrideEnv, ncclAlgoStr, NCCL_NUM_ALGORITHMS, algoVal) != ncclSuccess) {
        validInput = false;
        return ncclInvalidUsage;
      }
    }
    if (algoVal > NCCL_ALGO_UNDEF) {
      if (table[algoVal][info->protocol] == NCCL_ALGO_PROTO_IGNORE) {
        WARN("Failed to force unsupported algorithm %s for function %s with datatype %s", algoOverrideEnv,
             ncclFuncToString(info->func), ncclDatatypeToString(info->datatype));
        return ncclInternalError;
      } else {
        info->algorithm = algoVal;
      }
    }
  }
  return ncclSuccess;
}

void rcclUpdateThreadThreshold(struct ncclComm* comm, size_t const& nBytes, struct ncclTaskColl* info,
                               int& threadThreshold) {
  // Honor user input for thread thresholds
  static int userChannelControlInput = -2;
  if (userChannelControlInput == -2) {
    const char* inputStr = getenv("NCCL_THREAD_THRESHOLDS");
    if (!inputStr) {
      inputStr = getenv("NCCL_MAX_NCHANNELS");
    }
    if (!inputStr) {
      inputStr = getenv("NCCL_MIN_NCHANNELS");
    }
    userChannelControlInput = !inputStr ? 0 : 1;
  }

  if (!userChannelControlInput && comm->nNodes >= 2 &&
      (info->func == ncclFuncReduceScatter || info->func == ncclFuncAllGather)) {
    auto tunableIndex = rcclGetTunableIndex(info->func);
    auto tunedThreshold = comm->minMaxLLRange[tunableIndex][info->protocol][RCCL_PROTOCOL_THREAD_THRESHOLD_IDX];
    if (tunedThreshold != RCCL_LL_LIMITS_UNDEFINED) {
      threadThreshold = tunedThreshold * comm->nRanks;
    }
  }
}

void rcclSetPipelining(struct ncclComm* comm, size_t const& nBytes, struct ncclTaskColl* info) {
  info->pipeline = 0; // Default to no pipelining
  if (rcclParamdisableReduceCopyPipelining() || IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950")) {
    return;
  }
  const bool dtypeOK = (info->datatype == ncclBfloat16) || rcclParamPipelineAllDTypes();

  if (IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx942") && dtypeOK) {
    switch (info->func) {
    // For multi-node case, we check if the number of bytes (`nBytes`) satisfies
    // the Bf16 Limit Equation for bf16 all_reduce on MI300:
    // 512MB × 2^(log2[nNodes] - 1), nNodes > 1
    // The above equation is derived from the tuning results of the bf16 all_reduce on MI300.
    case ncclFuncAllReduce:
      if (comm->nNodes == 1 ||
          ((comm->nNodes > 1) && nBytes <= (1ULL << 29 /*512MB*/) * (1ULL << (log2i(comm->nNodes) - 1)))) {
        info->pipeline = 1;
      }
      break;

    case ncclFuncReduceScatter:
    case ncclFuncReduce:
      info->pipeline = 1;
      break;

    default:
      break;
    }
  }
}

extern ncclResult_t getAlgoInfo(struct ncclComm* comm, struct ncclTaskColl* task, int collNetSupport, int nvlsSupport,
                                int numPipeOps, ncclSimInfo_t* simInfo = NULL);
extern int rcclKernelPackedChannels(struct ncclComm* comm, ncclFunc_t func, size_t count, ncclDataType_t datatype,
                                    int protocol, int nMaxChannels);

ncclResult_t rcclHierarchicalAlgoInfo(struct ncclComm* comm, ncclFunc_t coll, uint64_t count, ncclDataType_t dataType,
                                      int* algo, int* protocol, int* maxChannels) {
  const bool isAllGather = (coll == ncclFuncAllGather);
  ncclComm* interComm = comm->hierarchicalInterComm;
  ncclComm* intraComm = comm->hierarchicalIntraComm;
  int nNodes = interComm->nRanks;

  *algo =
    isAllGather ? rcclAddonAlgos_t::RCCL_HIERARCHICAL_ALLGATHER : rcclAddonAlgos_t::RCCL_HIERARCHICAL_REDUCESCATTER;

  // Inter-node phase. Direct AllGather is only tuned up to 16 nodes.
  size_t interMsgSize = count * ncclTypeSize(dataType) * nNodes;
  bool interDirect = isAllGather ? (nNodes <= 16 && rcclUseAllGatherDirect(interComm, interMsgSize)) :
                                   rcclUseReduceScatterDirect(interComm, interMsgSize);
  if (interDirect) {
    *protocol = NCCL_PROTO_SIMPLE;
    *maxChannels = interComm->p2pnChannels;
  } else {
    struct ncclTaskColl task = {};
    task.func = coll;
    task.count = count;
    task.datatype = dataType;
    NCCLCHECK(getAlgoInfo(interComm, &task, 0, 0, 1));
    *protocol = task.protocol;
    *maxChannels = task.nMaxChannels;
  }

  // Intra-node phase. The hierarchical ReduceScatter never runs Direct intra-node,
  // so only AllGather gets the fast path here.
  int intraProto, intraChan;
  size_t intraCount = count * nNodes;
  size_t intraMsgSize = intraCount * ncclTypeSize(dataType) * intraComm->nRanks;
  if (isAllGather && rcclUseAllGatherDirect(intraComm, intraMsgSize)) {
    intraProto = NCCL_PROTO_SIMPLE;
    intraChan = intraComm->p2pnChannels;
  } else {
    struct ncclTaskColl task = {};
    task.func = coll;
    task.count = intraCount;
    task.datatype = dataType;
    NCCLCHECK(getAlgoInfo(intraComm, &task, 0, 0, 1));
    intraProto = task.protocol;
    intraChan = task.nMaxChannels;
  }

  // For hierarchical algorithm, only the inter-comm protocol/channels are
  // reported in rccl-tests -A output.
  // The intra-comm values are logged below for debugging purposes
  INFO(NCCL_COLL, "Hierarchical %s inter: proto=%d channels=%d, intra: proto=%d channels=%d", isAllGather ? "AG" : "RS",
       *protocol, *maxChannels, intraProto, intraChan);
  return ncclSuccess;
}

ncclResult_t rcclGetAlgoInfo(struct ncclComm* comm, ncclFunc_t coll, uint64_t count, ncclDataType_t dataType,
                             int collNetSupport, int nvlsSupport, int numPipeOps, int* algo, int* protocol,
                             int* maxChannels) {
  RCCL_STATIC_EXPOSE_CHECK();
  int nRanks;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  size_t msgSize = count * ncclTypeSize(dataType) * nRanks;
  if ((coll == ncclFuncAllGather && rcclUseHierarchicalAllGather(comm, msgSize)) ||
      (coll == ncclFuncReduceScatter && rcclUseHierarchicalReduceScatter(comm, msgSize))) {
    return rcclHierarchicalAlgoInfo(comm, coll, count, dataType, algo, protocol, maxChannels);
  }
  if (coll == ncclFuncAllGather && rcclUseAllGatherDirect(comm, msgSize)) {
    *algo = rcclAddonAlgos_t::RCCL_DIRECT_ALLGATHER;
    *protocol = NCCL_PROTO_SIMPLE; // TODO: consider LL for small messages
    *maxChannels = comm->p2pnChannels;
    return ncclSuccess;
  }
  struct ncclTaskColl task = {};
  task.func = coll;
  task.count = count;
  task.datatype = dataType;
  NCCLCHECK(getAlgoInfo(comm, &task, collNetSupport, nvlsSupport, numPipeOps));
  *protocol = task.protocol;
#ifdef ENABLE_WARP_SPEED
  *maxChannels = task.useWarpSpeed ? task.nMaxChannels / task.nWarps : task.nMaxChannels;
  *algo = task.useWarpSpeed ? rcclAddonAlgos_t::RCCL_WARP_SPEED : task.algorithm;
#else
  *maxChannels = task.nMaxChannels;
  *algo = task.algorithm;
#endif
  return ncclSuccess;
}

ncclResult_t rcclGetCollImplInfo(struct ncclComm* comm, ncclFunc_t coll, uint64_t count, ncclDataType_t dataType,
                                 ncclRedOp_t op, const void* sendbuff, void* recvbuff, int graphCapturing, int* algo,
                                 int* protocol, int* maxChannels) {
  RCCL_STATIC_EXPOSE_CHECK();
  if (algo == nullptr || protocol == nullptr || maxChannels == nullptr) return ncclInvalidArgument;

  // AllReduce and AllGather are wired to the unified decision. They return the
  // actual backend (CE / DDA / symmetric / kernel) for these operands so
  // rccl-tests can label numbers with the implementation that ran. graphCapturing
  // lets the caller declare graph mode, which the (out-of-capture) query cannot
  // detect on its own -- see the header comment.
  if (coll == ncclFuncAllReduce) {
    struct rcclCollDecision decision;
    NCCLCHECK(rcclSelectAllReduce(comm, sendbuff, recvbuff, (size_t)count, dataType, op, /*stream=*/nullptr,
                                  /*query=*/true, /*graphCapturingHint=*/graphCapturing != 0, &decision));
    *algo = decision.algo;
    *protocol = decision.protocol;
    *maxChannels = decision.nMaxChannels;
    return ncclSuccess;
  }

  if (coll == ncclFuncAllGather) {
    struct rcclCollDecision decision;
    NCCLCHECK(rcclSelectAllGather(comm, sendbuff, recvbuff, (size_t)count, dataType, /*query=*/true,
                                  /*graphCapturingHint=*/graphCapturing != 0, &decision));
    *algo = decision.algo;
    *protocol = decision.protocol;
    *maxChannels = decision.nMaxChannels;
    return ncclSuccess;
  }

  if (coll == ncclFuncReduceScatter) {
    struct rcclCollDecision decision;
    NCCLCHECK(rcclSelectReduceScatter(comm, sendbuff, recvbuff, (size_t)count, dataType, op, /*query=*/true,
                                      &decision));
    *algo = decision.algo;
    *protocol = decision.protocol;
    *maxChannels = decision.nMaxChannels;
    return ncclSuccess;
  }

  // Other collectives: fall back to the size/algo query until they are migrated
  // onto rcclSelectXxx().
  return rcclGetAlgoInfo(comm, coll, count, dataType, /*collNetSupport=*/0, /*nvlsSupport=*/0, /*numPipeOps=*/1, algo,
                         protocol, maxChannels);
}

static int symkHostRedOpToDev(ncclRedOp_t op) {
  switch ((int)op) {
  case ncclSum:
    return (int)ncclDevSum;
  case ncclProd:
    return (int)ncclDevProd;
  case ncclMin:
  case ncclMax:
    return (int)ncclDevMinMax;
  case ncclAvg:
    return (int)ncclDevSumPostDiv;
  default:
    return -1;
  }
}

// Fills (algo, protocol, maxChannels) for the symmetric kernel that would run
// for these operands; returns false if symk is unavailable or no kernel is
// picked. Single source shared by rcclSymKGetInfo and the query-mode symmetric
// reporting in rcclSelectAllReduce/rcclSelectAllGather.
static bool rcclSymkQuery(struct ncclComm* comm, ncclFunc_t coll, uint64_t count, ncclDataType_t dataType,
                          ncclRedOp_t op, int* algo, int* protocol, int* maxChannels) {
  // Symmetric kernels need symmetric-window support (cuMem). Without it the
  // windows are never registered symmetric and symk cannot run, so do not report
  // it -- rcclSymKGetInfo's caller then falls back to the actual backend.
  if (comm == nullptr || !comm->symmetricSupport) return false;
  if (coll != ncclFuncAllReduce && coll != ncclFuncAllGather && coll != ncclFuncReduceScatter) return false;
  int devOp = (coll == ncclFuncAllGather) ? (int)ncclDevSum : symkHostRedOpToDev(op);
  if (devOp < 0) return false;
  if (ncclSymkInitOnce(comm) != ncclSuccess) return false;
  if (!ncclSymkAvailable(comm, coll, devOp, dataType, (size_t)count)) return false;
  // NCCL 2.31 replaced ncclSymkPickKernel() with the tuning cost model; restricting
  // the tuning mask to the symmetric kernels reproduces the old query.
  struct ncclTuningInput_t input = {};
  input.comm = comm;
  input.tuningMask = NCCL_TUNING_MASK_SYM_KERNELS;
  input.func = coll;
  input.redOp = op;
  input.devRedOp = (ncclDevRedOp_t)devOp;
  input.datatype = dataType;
  input.nBytes = (size_t)count * ncclTypeSize(dataType);
  input.count = (size_t)count;
  input.countMax = (size_t)count;
  input.nWorks = 1;
  input.winRegType = ncclSymSendRegRecvReg;
  input.minCTAs = comm->config.minCTAs;
  input.maxCTAs = comm->config.maxCTAs;
  input.CTAPolicy = comm->config.CTAPolicy;
  input.nvlsSupport = comm->nvlsSupport && (ncclNvlsSupported(devOp, dataType) || coll == ncclFuncAllGather);
  struct ncclTuningResult_t bestTuning = NCCL_TUNING_RESULT_INIT;
  if (ncclTuningCompute(&input, &bestTuning) != ncclSuccess) return false;
  ncclSymkKernelId kernelId = (ncclSymkKernelId)bestTuning.symKernelId;
  if (kernelId == ncclSymkKernelId_Count) return false;
  *maxChannels = bestTuning.maxChannels;
  *algo = (int)rcclAddonAlgos_t::RCCL_SYMMETRIC;
  *protocol = rcclSymkKernelIdIsLL((int)kernelId) ? NCCL_PROTO_LL : NCCL_PROTO_SIMPLE;
  return true;
}

ncclResult_t rcclSymKGetInfo(struct ncclComm* comm, ncclFunc_t coll, uint64_t count, ncclDataType_t dataType,
                             ncclRedOp_t op, int* algo, int* protocol, int* maxChannels) {
  RCCL_STATIC_EXPOSE_CHECK();
  if (algo == nullptr || protocol == nullptr || maxChannels == nullptr) return ncclInvalidArgument;
  if (rcclSymkQuery(comm, coll, count, dataType, op, algo, protocol, maxChannels)) return ncclSuccess;
  // Symmetric kernel does not apply here (e.g. cuMem disabled -> no symmetric
  // windows, so symk never runs). Report the backend that actually runs instead
  // of failing, so the caller labels its numbers with the real implementation.
  // Buffer pointers are unknown at this ABI, but the selector's gates are
  // buffer-independent for the decision (DDA reads comm state; symmetric/CE
  // window lookups null-safely find nothing), so null operands are safe.
  return rcclGetCollImplInfo(comm, coll, count, dataType, op, /*sendbuff=*/nullptr, /*recvbuff=*/nullptr,
                             /*graphCapturing=*/0, algo, protocol, maxChannels);
}

ncclResult_t rcclGetAlgoName(int algo, const char** algoName) {
  if (algo < 0 || algo >= RCCL_ALGO_COUNT) {
    WARN("Invalid algorithm value: %d", algo);
    return ncclInvalidArgument;
  }
  if (algo >= NCCL_NUM_ALGORITHMS) {
    switch (algo) {
    case rcclAddonAlgos_t::RCCL_DIRECT_ALLGATHER:
      *algoName = "Direct";
      break;
    case rcclAddonAlgos_t::RCCL_HIERARCHICAL_ALLGATHER:
      *algoName = "Hier";
      break;
    case rcclAddonAlgos_t::RCCL_DIRECT_REDUCESCATTER:
      *algoName = "Direct";
      break;
    case rcclAddonAlgos_t::RCCL_HIERARCHICAL_REDUCESCATTER:
      *algoName = "Hier";
      break;
#ifdef ENABLE_WARP_SPEED
    case rcclAddonAlgos_t::RCCL_WARP_SPEED:
      *algoName = "RING*"; // WarpSpeed (*) uses RING algorithm
      break;
#endif
    case rcclAddonAlgos_t::RCCL_SYMMETRIC:
      *algoName = "SYM";
      break;
    case rcclAddonAlgos_t::RCCL_CE_2SHOT:
      *algoName = "CE2";
      break;
    case rcclAddonAlgos_t::RCCL_CE_REGISTERED:
      *algoName = "CE";
      break;
    // Fabric variants all report "DDA"; the protocol column distinguishes
    // LL / LL128 / Simple, so the name needn't repeat it.
    case rcclAddonAlgos_t::RCCL_DDA_FABRIC_LL:
    case rcclAddonAlgos_t::RCCL_DDA_FABRIC_LL128:
    case rcclAddonAlgos_t::RCCL_DDA_FABRIC_VMM:
      *algoName = "DDA";
      break;
    case rcclAddonAlgos_t::RCCL_DDA_IPC:
      *algoName = "DDA-IPC";
      break;
    default:
      WARN("Invalid algorithm value: %d", algo);
      return ncclInvalidArgument;
    }
    return ncclSuccess;
  }
  *algoName = ncclAlgoToString(algo);
  return ncclSuccess;
}

ncclResult_t rcclGetProtocolName(int protocol, const char** protocolName) {
  if (protocol < 0 || protocol >= NCCL_NUM_PROTOCOLS) {
    WARN("Invalid protocol value: %d", protocol);
    return ncclInvalidArgument;
  }
  *protocolName = ncclProtoToString(protocol);
  return ncclSuccess;
}

bool rcclDdaEnabled(const ncclComm* comm, size_t totalBytes, size_t gfx942Default, size_t gfx950Default,
                    size_t gfx1250Default) {
  // The environment parameter can be NCCL_CONFIG_UNDEF_INT when launch order
  // is configured per communicator. Use the resolved communicator value:
  // testing the raw sentinel as a boolean disables DDA by default, while
  // testing only the environment would ignore an explicit config value.
  if (!rcclParamDdaEnable() || comm->config.launchOrderImplicit == 1 || ncclGroupDepth != 0) {
    return false;
  }
  size_t threshold;
  if (IsArchMatch(comm->archName, "gfx1250")) {
    threshold = gfx1250Default ? gfx1250Default : static_cast<size_t>(rcclParamDdaThreshold());
  } else if (IsArchMatch(comm->archName, "gfx942") || IsArchMatch(comm->archName, "gfx950")) {
    if (comm->nRanks < 8) return false;
    if (IsArchMatch(comm->archName, "gfx942")) {
      threshold = gfx942Default;
    } else {
      threshold = gfx950Default ? gfx950Default : static_cast<size_t>(rcclParamDdaThreshold());
    }
  } else {
    return false;
  }
  return threshold > 0 && totalBytes <= threshold;
}

bool rcclUseAlltoAllGda(struct ncclComm* comm) {
#ifdef ENABLE_ROCSHMEM
  if (comm->enableRocshmem && comm->nNodes > 1 && (comm->nRanks / comm->nNodes == 8) &&
      comm->rocshmemThreshold <= 1048576) {
    INFO(NCCL_INIT, "Enabling GDA alltoall for RCCL");
    return true;
  }
#endif
  return false;
}

size_t rcclHierarchicalTempBufferSize(int nNodes, bool allGather, bool reduceScatter) {
  size_t agThreshold = 0;
  if (allGather) {
    if (nNodes >= 32) {
      agThreshold = HIERARCHICAL_TEMP_BUFFER_SIZE; // 128MB
    } else if (nNodes >= 16) {
      agThreshold = HIERARCHICAL_TEMP_BUFFER_SIZE / 2; // 64MB
    } else if (nNodes >= 8) {
      agThreshold = HIERARCHICAL_TEMP_BUFFER_SIZE / 4; // 32MB
    }
  }

  size_t rsThreshold = 0;
  if (reduceScatter) {
    if (nNodes >= 16) {
      rsThreshold = HIERARCHICAL_TEMP_BUFFER_SIZE; // 128MB
    } else if (nNodes >= 8) {
      rsThreshold = HIERARCHICAL_TEMP_BUFFER_SIZE / 2; // 64MB
    }
  }

  return std::max(agThreshold, rsThreshold);
}

RCCL_PARAM(HierarchicalAllGather, "HIERARCHICAL_ALLGATHER", 1);

bool rcclUseHierarchicalAllGather(struct ncclComm* comm, size_t msgSize) {
  if (comm->nNodes < 8) return false;
  if (rcclParamHierarchicalAllGather() != 1) return false;
  if (!comm->hierarchicalCommsInitialized) return false;

  size_t threshold = rcclHierarchicalTempBufferSize(comm->nNodes, /*allGather=*/true, /*reduceScatter=*/false);
  return threshold > 0 && msgSize <= threshold;
}

bool rcclUseAllGatherDirect(struct ncclComm* comm, size_t& msgSize) {
  // Check if user explicitly disabled direct AllGather
  static int userDirectAllGatherInput = rcclParamDirectAllGatherDisable();
  if (userDirectAllGatherInput != 0) {
    INFO(NCCL_INIT, "RCCL DIRECT ALLGATHER has been disabled by environment variable.");
    return false;
  }

  if (rcclUseAinic()) {
    INFO(NCCL_INIT, "RCCL DIRECT ALLGATHER disabled on AINIC. ");
    return false;
  }

  // Check if user explicitly set threshold
  static int userThresholdInput = -2;
  if (userThresholdInput == -2) {
    const char* thresholdStr = getenv("RCCL_DIRECT_ALLGATHER_THRESHOLD");
    userThresholdInput = !thresholdStr ? 0 : 1;
  }

  size_t threshold = rcclParamDirectAllGatherThreshold();

  // Disable Direct AllGather for all architectures when CE-based AllGather is active.
  // CTAPolicy ZERO indicates CE dispatch is enabled; Direct AllGather conflicts with it on
  // single-node topologies regardless of GPU architecture.
  if (!userThresholdInput && comm->nNodes == 1 && comm->symmetricSupport &&
      comm->config.CTAPolicy == NCCL_CTA_POLICY_ZERO) {
    INFO(NCCL_INIT, "RCCL Direct AllGather disabled: CTA policy ZERO, using CE-based AllGather.");
    return false;
  }

  // Only perform auto-selection if user didn't explicitly set the threshold and threshold is not -1
  if (!userThresholdInput && IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950") && threshold != -1) {
    if (comm->nNodes == 1) {
      threshold = 8388608;
    } else if (comm->nNodes < 64) {
      threshold = comm->nNodes * 2097152;
    }
  } else if (!userThresholdInput && IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx942") && threshold != -1) {
    threshold = 4194304;
  }

  comm->enableCustColl = IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950") ||
                         IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx942");

  int rankMultiple = comm->nRanks % 8;

  // return (comm->enableCustColl && (comm->nNodes > 1) && (msgSize <= threshold) && (threshold != -1))
  return (comm->enableCustColl && (msgSize <= threshold) && (threshold != -1) && !rankMultiple);
}

bool rcclUseCeAllReduce(struct ncclComm* comm, size_t count, ncclDataType_t datatype, ncclRedOp_t op, const void* acc) {
  static int enabled = rcclParamCeAllReduce();
  static int force = rcclParamForceCeAllReduce();
  if (!enabled) {
    // Log once per process, not on every eligibility check (called per AllReduce).
    static bool warnedDisabled = false;
    if (!warnedDisabled) {
      warnedDisabled = true;
      INFO(NCCL_INIT, "CE AllReduce not enabled. Set RCCL_CE_ALLREDUCE=1 to enable.");
    }
    return false;
  }

  // The CE kernels never read the bias buffer, so taking this path for
  // ncclAllReduceWithBias would silently drop the bias from the result.
  if (acc != nullptr) return false;

  // Requires single-node symmetric memory support with CTA_POLICY_ZERO (CE mode).
  if (!comm->symmetricSupport) {
    WARN("Skipping CE AllReduce: symmetric support is not enabled");
    return false;
  }
  if (comm->nNodes != 1) {
    WARN("Skipping CE AllReduce: nNodes is not 1");
    return false;
  }

  // count must divide evenly so every rank owns an equal shard.
  if (count == 0 || count % (size_t)comm->nRanks != 0) {
    WARN("Skipping CE AllReduce: count (%zu) is not divisible by nRanks (%d)", count, comm->nRanks);
    return false;
  }

  // Total message must fit within the pre-allocated staging buffer.
  size_t msgBytes = count * ncclTypeSize(datatype);
  if (force) {
    if (msgBytes > NCCL_CE_AR_MAX_MSG_BYTES) {
      WARN("Skipping CE AllReduce despite RCCL_FORCE_CE_ALLREDUCE=1: msgBytes (%zu) > NCCL_CE_AR_MAX_MSG_BYTES (%zu)",
           msgBytes, NCCL_CE_AR_MAX_MSG_BYTES);
      return false;
    }
  }
  if (msgBytes > NCCL_CE_AR_MAX_MSG_BYTES) {
    WARN("Skipping CE AllReduce: msgBytes (%zu) > NCCL_CE_AR_MAX_MSG_BYTES (%zu)", msgBytes, NCCL_CE_AR_MAX_MSG_BYTES);
    return false;
  }

  if (comm->config.CTAPolicy != NCCL_CTA_POLICY_ZERO && !force) {
    WARN("Skipping CE AllReduce: CTA policy is not ZERO");
    return false;
  }

  // Only standard reduction ops with a simple kernel implementation.
  // ncclAvg (maps to SumPostDiv) and user-defined PreMulSum fall back to ring.
  if (op != ncclSum && op != ncclProd && op != ncclMin && op != ncclMax) {
    WARN("Skipping CE AllReduce: unsupported reduction operation");
    return false;
  }

  // Float8 types require specialised handling not yet implemented for CE AR.
  if (datatype == ncclFloat8e4m3 || datatype == ncclFloat8e5m2) {
    WARN("Skipping CE AllReduce: unsupported datatype: Float8");
    return false;
  }

  return true;
}

void rcclCeAllReduceGraphLatchTick(struct ncclComm* comm, bool ceCapturing) {
  if (ceCapturing) {
    if (!comm->ceColl.graphModeSeen) {
      INFO(NCCL_COLL, "Disabling CE AllReduce; graph latch set (rank %d): capture detected", comm->rank);
      comm->ceColl.graphModeSeen = true;
    }
    // Stay latched while capturing, even if an unrelated older plan on this
    // comm was just reclaimed: clearing here would wrongly re-enable CE
    // mid-capture.
  } else if (comm->ceColl.graphModeSeen && comm->localPersistentRefs == 0) {
    // Do not proactively drain comm->callbackQueue to freshen this check.
    // localPersistentRefs is reclaimed via a per-rank async callback with no
    // cross-rank sync, but all ranks must reach the same decision for the
    // same call. The ambient once-every-few-group-ends cadence in group.cc
    // gives every rank's reclaim equal time to complete first; checking more
    // eagerly let ranks diverge and deadlock (confirmed experimentally).
    INFO(NCCL_COLL, "Re-enabling CE AllReduce; graph latch cleared (rank %d): no live captured plans", comm->rank);
    comm->ceColl.graphModeSeen = false;
  }
}

bool rcclCeAllReduceAllowed(struct ncclComm* comm) {
  return !comm->ceColl.graphModeSeen;
}

// Single source of truth for AllReduce implementation selection. See the header
// comment on rcclSelectAllReduce(). The priority chain and every gate below are a
// faithful consolidation of what was previously split between ncclAllReduce_impl()
// (symmetric / CE 2-shot / DDA) and taskAppend() (CE registered / kernel); the
// outcome for any given operands is identical.
ncclResult_t rcclSelectAllReduce(struct ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                 ncclDataType_t datatype, ncclRedOp_t op, cudaStream_t stream, bool query,
                                 bool graphCapturingHint, struct rcclCollDecision* decision) {
  memset(decision, 0, sizeof(*decision));
  decision->algo = NCCL_ALGO_RING;
  decision->protocol = NCCL_PROTO_SIMPLE;
  decision->nMaxChannels = 0;

  const size_t msgBytes = count * ncclTypeSize(datatype);

  // (1) Symmetric-window kernel eligibility takes priority over CE / DDA, exactly
  // as the pre-refactor collectives.cc path did.
  const bool symEligible = (op == ncclSum) && isSymmetricKernelRequested(comm, ncclFuncAllReduce, (int)ncclDevSum,
                                                                         datatype, count, sendbuff, recvbuff);

  // (2) CE AllReduce graph state. CE is graph-unsafe, so capture disables it.
  //  - Live dispatch (query=false): probe the real stream and tick the graph
  //    latch, exactly as the inline code did.
  //  - Reporting (query=true): the query runs outside capture, so the stream
  //    cannot reveal graph mode; the caller declares it via graphCapturingHint.
  //    Mirror what the latch tick would do under capture (ceArGraphAllowed=false)
  //    so CE 2-shot -- gated on the latch, not on ceCapturing directly -- is also
  //    reported as disabled. The tick mutates comm state, so it is never run here.
  bool ceCapturing;
  if (query) {
    ceCapturing = graphCapturingHint;
  } else {
    struct ncclCudaGraph ceGraph;
    NCCLCHECK(ncclCudaGetCapturingGraph(&ceGraph, stream, comm->config.graphUsageMode));
    ceCapturing = ncclCudaGraphValid(ceGraph);
    rcclCeAllReduceGraphLatchTick(comm, ceCapturing);
  }
  bool ceArGraphAllowed = rcclCeAllReduceAllowed(comm);
  if (query && ceCapturing) ceArGraphAllowed = false;
  decision->ceCapturing = ceCapturing;
  decision->ceArGraphAllowed = ceArGraphAllowed;

  // Symmetric-window lookup, hoisted so CE 2-shot / DDA / CE-registered all share
  // it. sendWin/recvWin drive winRegType and the sysmem-segment guard below.
  struct ncclDevrWindow* sendWin = nullptr;
  struct ncclDevrWindow* recvWin = nullptr;
  ncclDevrFindWindow(comm, sendbuff, &sendWin);
  ncclDevrFindWindow(comm, recvbuff, &recvWin);
  const bool hasSysmemSegment = ncclDevrWindowHasSysmemSegment(sendWin) || ncclDevrWindowHasSysmemSegment(recvWin);
  ncclSymRegType_t winRegType;
  NCCLCHECK(ncclGetSymRegType(sendWin, recvWin, &winRegType));

  // develop's single "will CE AllReduce service this call" gate (collectives.cc
  // ncclAllReduce_impl). force = RCCL_FORCE_CE_ALLREDUCE; symReg probes whether the
  // buffers are CE-registrable symmetric windows (uses ncclDevSum, matching develop).
  const bool force = rcclParamForceCeAllReduce() != 0;
  const bool symReg = ncclCeAvailable(comm, ncclFuncAllReduce, (int)ncclDevSum, datatype, winRegType, sendWin, recvWin);
  // This call site never carries a bias buffer (ncclAllReduceWithBias_impl bypasses it entirely
  // and goes straight to taskAppend), so /*acc=*/nullptr here is always correct.
  const bool ceAllReduceAllowed = ncclGroupDepth == 0 && ceArGraphAllowed &&
                                  rcclUseCeAllReduce(comm, count, datatype, op, /*acc=*/nullptr) && (force || symReg);

  // (3) Eager CE 2-shot (staging buffer). Requires !symEligible and an
  // initialized ceARTmpBuf (first call, before init, falls through to enqueue).
  if (!symEligible && ceAllReduceAllowed && comm->ceColl.ceARTmpBuf != NULL) {
    decision->algo = RCCL_CE_2SHOT;
    decision->nMaxChannels = ncclCeLocalReduceBlocks(datatype, count / comm->nRanks);
    return ncclSuccess;
  }

  // (4) DDA fast paths. develop's shared gate: !symEligible, and either gfx1250
  // (fabric, full range) or CE is not going to service this call (!ceAllReduceAllowed),
  // subject to rcclDdaEnabled thresholds -- all folded into the helper.
  const bool ddaFabricArch1250 = IsArchMatch(comm->archName, "gfx1250");
  if (rcclAllReduceShouldTakeDdaPath(comm, count, datatype, symEligible, ceAllReduceAllowed)) {
    if (ddaFabricArch1250) {
      // Small-message fast lane: LL protocol (no GPU barrier).
      if (ncclAllReduceDdaFabricLLEligible(comm, sendbuff, recvbuff, count, datatype, op)) {
        decision->algo = RCCL_DDA_FABRIC_LL;
        decision->protocol = NCCL_PROTO_LL;
        decision->nMaxChannels = ncclAllReduceDdaFabricLLBlocks(comm, count, datatype);
        return ncclSuccess;
      }
      // Mid-size fast lane: LL128 protocol (128B lines, no GPU barrier).
      if (rcclParamDdaLL128() && msgBytes <= (size_t)rcclParamDdaLL128Threshold() &&
          ncclAllReduceDdaFabricLL128Eligible(comm, sendbuff, recvbuff, count, datatype, op)) {
        decision->algo = RCCL_DDA_FABRIC_LL128;
        decision->protocol = NCCL_PROTO_LL128;
        decision->nMaxChannels = ncclAllReduceDdaFabricLL128Blocks(comm, count, datatype);
        return ncclSuccess;
      }
      if (ncclAllReduceDdaFabricEligible(comm, sendbuff, recvbuff, count, datatype, op)) {
        decision->algo = RCCL_DDA_FABRIC_VMM;
        decision->nMaxChannels = ncclAllReduceDdaFabricBlocks(comm, count, datatype);
        return ncclSuccess;
      }
    } else {
      if (ncclAllReduceDdaIpcEligible(comm, sendbuff, recvbuff, count, datatype, op)) {
        decision->algo = RCCL_DDA_IPC;
        decision->nMaxChannels = ncclAllReduceDdaIpcBlocks(comm, count, datatype);
        return ncclSuccess;
      }
    }
  }

  // (5) Enqueue-bound backends: CE registered (Branch B) vs symmetric vs kernel.
  // Reproduce taskAppend()'s AllReduce CE decision exactly so both agree.
  // NOTE: CE registered wins over the symmetric kernel when both are eligible,
  // matching taskAppend (its CE branch is not gated on symEligible; symmetric
  // extraction only sees tasks that fall through to collTaskAppend).
  //
  // develop's taskAppend appends CE for AllReduce iff !hasSysmemSegment && ceAvailable
  // && ((CTAPolicy & ZERO) || force): ceAvailable starts from ncclCeAvailable(op) then
  // is cleared unless graph-allowed, op-supported, count-divisible and RCCL_CE_ALLREDUCE.
  bool ceAvailable = !ceCapturing && ncclCeAvailable(comm, ncclFuncAllReduce, (int)op, datatype, winRegType, sendWin, recvWin);
  const bool ceAllReduceOpSupported = (op == ncclSum || op == ncclProd || op == ncclMin || op == ncclMax);
  if (!ceArGraphAllowed || !ceAllReduceOpSupported || (count % (size_t)comm->nRanks != 0) || !rcclParamCeAllReduce()) {
    ceAvailable = false;
  }
  if (ceAvailable && !hasSysmemSegment && ((comm->config.CTAPolicy & NCCL_CTA_POLICY_ZERO) || force)) {
    decision->algo = RCCL_CE_REGISTERED;
    decision->nMaxChannels = ncclCeLocalReduceBlocks(datatype, count / comm->nRanks);
    return ncclSuccess;
  }

  if (symEligible) {
    decision->algo = RCCL_SYMMETRIC;
    // Reporting only: fill the symk protocol/channels that will actually run.
    // Skipped on the live path (taskAppend recomputes and dispatches symk).
    if (query) {
      int a, p, ch;
      if (rcclSymkQuery(comm, ncclFuncAllReduce, count, datatype, op, &a, &p, &ch)) {
        decision->protocol = p;
        decision->nMaxChannels = ch;
      }
    }
    return ncclSuccess;
  }

  // (6) Standard ring/tree/pat kernel. Fill algo/protocol/channels for reporting
  // (query mode); on the live path taskAppend() recomputes these downstream, so
  // skip the getAlgoInfo() cost there and leave a valid non-CE placeholder.
  decision->algo = NCCL_ALGO_RING;
  decision->protocol = NCCL_PROTO_SIMPLE;
  if (query) {
    struct ncclTaskColl task;
    memset(&task, 0, sizeof(task));
    task.func = ncclFuncAllReduce;
    task.sendbuff = sendbuff;
    task.recvbuff = recvbuff;
    task.count = count;
    task.datatype = datatype;
    NCCLCHECK(getAlgoInfo(comm, &task, /*collNetSupport=*/0, /*nvlsSupport=*/0, /*numPipeOps=*/1, /*simInfo=*/nullptr));
    decision->protocol = task.protocol;
    // Report the traffic-packed channel count the kernel actually runs on, not the
    // tuning cap (task.nMaxChannels), matching the enqueue.cc channel{Lo..Hi} log.
    int packed = rcclKernelPackedChannels(comm, ncclFuncAllReduce, count, datatype, task.protocol, task.nMaxChannels);
#ifdef ENABLE_WARP_SPEED
    // WarpSpeed reports as RING* with channels scaled by nWarps, matching rcclGetAlgoInfo.
    decision->nMaxChannels = task.useWarpSpeed ? task.nMaxChannels / task.nWarps : packed;
    decision->algo = task.useWarpSpeed ? rcclAddonAlgos_t::RCCL_WARP_SPEED : task.algorithm;
#else
    decision->nMaxChannels = packed;
    decision->algo = task.algorithm;
#endif
  }
  return ncclSuccess;
}

// See the header comment on rcclSelectAllGather(). Faithful consolidation of
// ncclAllGather_impl() (DDA) and rcclSelectAllGatherAlgo() (hierarchical / direct
// / ring); the outcome for any given operands is identical.
ncclResult_t rcclSelectAllGather(struct ncclComm* comm, const void* sendbuff, void* recvbuff, size_t sendcount,
                                 ncclDataType_t datatype, bool query, bool graphCapturingHint,
                                 struct rcclCollDecision* decision) {
  memset(decision, 0, sizeof(*decision));
  decision->algo = NCCL_ALGO_RING;
  decision->protocol = NCCL_PROTO_SIMPLE;
  decision->nMaxChannels = 0;

  const size_t typeSize = ncclTypeSize(datatype);
  const size_t totalBytes = (size_t)comm->nRanks * sendcount * typeSize;
  size_t msgSize = totalBytes;

  // (1) DDA fast paths. Symmetric-registered buffers defer to the symmetric
  // kernel (extracted downstream), so DDA is gated on !symEligible, as before.
  const bool symEligible =
    isSymmetricKernelRequested(comm, ncclFuncAllGather, (int)ncclDevSum, datatype, sendcount, sendbuff, recvbuff);
  // symEligible gates DDA below; the symk report itself is deferred until after
  // the CE-registered check so it loses to CE exactly as dispatch does
  // (taskAppend appends the CE task before ncclMakeSymmetricTaskList runs, so
  // symk never reclaims it), mirroring rcclSelectAllReduce.
  if (!symEligible && rcclDdaEnabled(comm, totalBytes, 8388608)) {
    if (IsArchMatch(comm->archName, "gfx1250")) {
      if (rcclParamDdaLL() && msgSize <= (size_t)rcclParamDdaLLThreshold() &&
          ncclAllGatherDdaFabricLLEligible(comm, sendbuff, recvbuff, sendcount, datatype)) {
        decision->algo = RCCL_DDA_FABRIC_LL;
        decision->protocol = NCCL_PROTO_LL;
        decision->nMaxChannels = ncclAllGatherDdaFabricLLBlocks(comm, sendcount, datatype);
        return ncclSuccess;
      }
      if (rcclParamDdaLL128() && msgSize <= (size_t)rcclParamDdaLL128Threshold() &&
          ncclAllGatherDdaFabricLL128Eligible(comm, sendbuff, recvbuff, sendcount, datatype)) {
        decision->algo = RCCL_DDA_FABRIC_LL128;
        decision->protocol = NCCL_PROTO_LL128;
        decision->nMaxChannels = ncclAllGatherDdaFabricLL128Blocks(comm, sendcount, datatype);
        return ncclSuccess;
      }
      if (ncclAllGatherDdaFabricEligible(comm, sendbuff, recvbuff, sendcount, datatype)) {
        decision->algo = RCCL_DDA_FABRIC_VMM;
        decision->nMaxChannels = ncclAllGatherDdaFabricBlocks(comm, sendcount, datatype);
        return ncclSuccess;
      }
    } else if (ncclAllGatherDdaIpcEligible(comm, sendbuff, recvbuff, sendcount, datatype)) {
      decision->algo = RCCL_DDA_IPC;
      decision->nMaxChannels = ncclAllGatherDdaIpcBlocks(comm, sendcount, datatype);
      return ncclSuccess;
    }
  }

  // (2) Hierarchical AllGather. Live dispatch requires being outside a group
  // (rcclSelectAllGatherAlgo); the reporting query always runs outside a group, so
  // the same gate reproduces rcclGetAlgoInfo's group-agnostic reporting.
  if (ncclGroupDepth == 0 && rcclUseHierarchicalAllGather(comm, msgSize)) {
    decision->algo = RCCL_HIERARCHICAL_ALLGATHER;
    if (query) {
      // -A reports the inter-comm proto/channels; intra values are logged only.
      ncclComm* interComm = comm->hierarchicalInterComm;
      ncclComm* intraComm = comm->hierarchicalIntraComm;
      int nNodes = interComm->nRanks;
      size_t interMsgSize = sendcount * typeSize * nNodes;
      if (nNodes <= 16 && rcclUseAllGatherDirect(interComm, interMsgSize)) {
        decision->protocol = NCCL_PROTO_SIMPLE;
        decision->nMaxChannels = interComm->p2pnChannels;
      } else {
        struct ncclTaskColl task;
        task.func = ncclFuncAllGather;
        task.count = sendcount;
        task.datatype = datatype;
        NCCLCHECK(getAlgoInfo(interComm, &task, 0, 0, 1));
        decision->protocol = task.protocol;
        decision->nMaxChannels = task.nMaxChannels;
      }
      int intraProto, intraChan;
      size_t intraCount = sendcount * nNodes;
      size_t intraMsgSize = intraCount * typeSize * intraComm->nRanks;
      if (rcclUseAllGatherDirect(intraComm, intraMsgSize)) {
        intraProto = NCCL_PROTO_SIMPLE;
        intraChan = intraComm->p2pnChannels;
      } else {
        struct ncclTaskColl task;
        task.func = ncclFuncAllGather;
        task.count = intraCount;
        task.datatype = datatype;
        NCCLCHECK(getAlgoInfo(intraComm, &task, 0, 0, 1));
        intraProto = task.protocol;
        intraChan = task.nMaxChannels;
      }
      INFO(NCCL_COLL, "Hierarchical AG inter: proto=%d channels=%u, intra: proto=%d channels=%d", decision->protocol,
           decision->nMaxChannels, intraProto, intraChan);
    }
    return ncclSuccess;
  }

  // (3) CE AllGather. Mirrors taskAppend()'s live gates and outranks Direct, as
  // taskAppend checks CE before the useDirect branch. Reporting only here; the
  // live path re-decides and dispatches CE in taskAppend().
  {
    const bool ceCapturing = query ? graphCapturingHint : false;
    struct ncclDevrWindow* sendWin = nullptr;
    struct ncclDevrWindow* recvWin = nullptr;
    ncclDevrFindWindow(comm, sendbuff, &sendWin);
    ncclDevrFindWindow(comm, recvbuff, &recvWin);
    const bool hasSysmemSegment = ncclDevrWindowHasSysmemSegment(sendWin) || ncclDevrWindowHasSysmemSegment(recvWin);
    ncclSymRegType_t winRegType;
    NCCLCHECK(ncclGetSymRegType(sendWin, recvWin, &winRegType));
    // Branch #2: FORCE_CE via DDA scratch (unregistered windows).
    const bool ceScratch =
      !ceCapturing && ncclCeScratchAvailable(comm, ncclFuncAllGather, (int)ncclSum, datatype, winRegType);
    if (rcclParamForceCe() && ceScratch && winRegType != ncclSymSendRegRecvReg &&
        winRegType != ncclSymSendNonregRecvReg && !hasSysmemSegment && comm->ddaScratch != nullptr &&
        totalBytes <= (size_t)comm->ddaScratchBytes) {
      decision->algo = RCCL_CE_REGISTERED;
      return ncclSuccess;
    }
    // Branch #3: CE via registered symmetric windows.
    const bool ceAvailable =
      !ceCapturing && ncclCeAvailable(comm, ncclFuncAllGather, (int)ncclSum, datatype, winRegType, sendWin, recvWin);
    if (ceAvailable && !hasSysmemSegment) {
      decision->algo = RCCL_CE_REGISTERED;
      return ncclSuccess;
    }
  }

  // (3.5) Symmetric-window kernel. Reported only after CE-registered so it loses
  // to CE exactly as dispatch does; on the live path taskAppend recomputes and
  // the symmetric extraction downstream dispatches symk. Mirrors rcclSelectAllReduce.
  if (query && symEligible) {
    int a, p, ch;
    if (rcclSymkQuery(comm, ncclFuncAllGather, sendcount, datatype, ncclSum, &a, &p, &ch)) {
      decision->algo = a;
      decision->protocol = p;
      decision->nMaxChannels = ch;
      return ncclSuccess;
    }
  }

  // (4) Direct AllGather (per-peer Send/Recv).
  if (rcclUseAllGatherDirect(comm, msgSize)) {
    decision->algo = RCCL_DIRECT_ALLGATHER;
    decision->protocol = NCCL_PROTO_SIMPLE;
    decision->nMaxChannels = comm->p2pnChannels;
    return ncclSuccess;
  }

  // (5) Standard ring kernel. Fill algo/protocol/channels for reporting; the live
  // path recomputes these in taskAppend(), so only the query needs them.
  decision->algo = NCCL_ALGO_RING;
  decision->protocol = NCCL_PROTO_SIMPLE;
  if (query) {
    struct ncclTaskColl task;
    memset(&task, 0, sizeof(task));
    task.func = ncclFuncAllGather;
    task.sendbuff = sendbuff;
    task.recvbuff = recvbuff;
    task.count = sendcount;
    task.datatype = datatype;
    NCCLCHECK(getAlgoInfo(comm, &task, 0, 0, 1));
    decision->protocol = task.protocol;
    // Report the traffic-packed channel count the kernel actually runs on, not the
    // tuning cap (task.nMaxChannels), matching the enqueue.cc channel{Lo..Hi} log.
    int packed =
      rcclKernelPackedChannels(comm, ncclFuncAllGather, sendcount, datatype, task.protocol, task.nMaxChannels);
#ifdef ENABLE_WARP_SPEED
    // WarpSpeed reports as RING* with channels scaled by nWarps, matching rcclGetAlgoInfo.
    decision->nMaxChannels = task.useWarpSpeed ? task.nMaxChannels / task.nWarps : packed;
    decision->algo = task.useWarpSpeed ? rcclAddonAlgos_t::RCCL_WARP_SPEED : task.algorithm;
#else
    decision->nMaxChannels = packed;
    decision->algo = task.algorithm;
#endif
  }
  return ncclSuccess;
}

// See the header comment. Consolidates the backend picking formerly inlined in
// ncclReduceScatter_impl(); the outcome for any given operands is identical.
ncclResult_t rcclSelectReduceScatter(struct ncclComm* comm, const void* sendbuff, void* recvbuff, size_t recvcount,
                                     ncclDataType_t datatype, ncclRedOp_t op, bool query,
                                     struct rcclCollDecision* decision) {
  memset(decision, 0, sizeof(*decision));
  decision->algo = NCCL_ALGO_RING;
  decision->protocol = NCCL_PROTO_SIMPLE;
  decision->nMaxChannels = 0;

  const size_t typeSize = ncclTypeSize(datatype);
  const size_t totalBytes = (size_t)comm->nRanks * recvcount * typeSize;
  const size_t rsShardBytes = recvcount * typeSize;

  // (1) Symmetric eligibility (sum/avg). Reported last but gates DDA IPC / Direct here.
  const bool symEligible =
    (op == ncclSum || op == ncclAvg) &&
    isSymmetricKernelRequested(comm, ncclFuncReduceScatter, (op == ncclAvg) ? (int)ncclDevSumPostDiv : (int)ncclDevSum,
                               datatype, recvcount, sendbuff, recvbuff);

  // (2) DDA fast paths. gfx1250 fabric may win over symmetric (cross-rank identical
  // state); IPC keeps the strict !symEligible guard. No Blocks helpers -> nMaxChannels 0.
  const bool ddaFabricArch = IsArchMatch(comm->archName, "gfx1250");
  if ((!symEligible || ddaFabricArch) && rcclDdaEnabled(comm, totalBytes, 8388608)) {
    if (ddaFabricArch) {
      if (rcclParamDdaLL() && rsShardBytes <= (size_t)rcclParamDdaLLThreshold() &&
          ncclReduceScatterDdaFabricLLEligible(comm, sendbuff, recvbuff, recvcount, datatype, op)) {
        decision->algo = RCCL_DDA_FABRIC_LL;
        decision->protocol = NCCL_PROTO_LL;
        return ncclSuccess;
      }
      if (rcclParamDdaLL128() && rsShardBytes <= (size_t)rcclParamDdaLL128Threshold() &&
          ncclReduceScatterDdaFabricLL128Eligible(comm, sendbuff, recvbuff, recvcount, datatype, op)) {
        decision->algo = RCCL_DDA_FABRIC_LL128;
        decision->protocol = NCCL_PROTO_LL128;
        return ncclSuccess;
      }
      if (ncclReduceScatterDdaFabricEligible(comm, sendbuff, recvbuff, recvcount, datatype, op)) {
        decision->algo = RCCL_DDA_FABRIC_VMM;
        return ncclSuccess;
      }
    } else if (ncclReduceScatterDdaIpcEligible(comm, sendbuff, recvbuff, recvcount, datatype, op)) {
      decision->algo = RCCL_DDA_IPC;
      return ncclSuccess;
    }
  }

  // (3) Hierarchical ReduceScatter (multi-node, sum only). Live dispatch requires
  // being outside a group; the reporting query always runs outside a group, so the
  // same gate reproduces rcclGetAlgoInfo's group-agnostic reporting.
  if (!symEligible && ncclGroupDepth == 0 && op == ncclSum && rcclUseHierarchicalReduceScatter(comm, totalBytes)) {
    decision->algo = RCCL_HIERARCHICAL_REDUCESCATTER;
    if (query) {
      int a, p, ch;
      NCCLCHECK(rcclHierarchicalAlgoInfo(comm, ncclFuncReduceScatter, recvcount, datatype, &a, &p, &ch));
      decision->protocol = p;
      decision->nMaxChannels = ch;
    }
    return ncclSuccess;
  }

  // (4) Direct ReduceScatter (per-peer Send/Recv, native kernel finishes the reduce).
  // That reduce runs with PreOpSrcs=0 / postOp=false, an unscaled sum, so ncclAvg and
  // user-defined PreMulSum (op >= ncclNumOps) fall through to the ring kernel instead.
  size_t directMsgSize = totalBytes;
  if (!symEligible && ncclGroupDepth == 0 && op < ncclAvg && rcclUseReduceScatterDirect(comm, directMsgSize)) {
    decision->algo = RCCL_DIRECT_REDUCESCATTER;
    decision->protocol = NCCL_PROTO_SIMPLE;
    decision->nMaxChannels = comm->p2pnChannels;
    return ncclSuccess;
  }

  // (5) Symmetric kernel. Live path dispatches symk via the downstream extraction.
  if (symEligible) {
    decision->algo = RCCL_SYMMETRIC;
    if (query) {
      int a, p, ch;
      if (rcclSymkQuery(comm, ncclFuncReduceScatter, recvcount, datatype, op, &a, &p, &ch)) {
        decision->protocol = p;
        decision->nMaxChannels = ch;
      }
    }
    return ncclSuccess;
  }

  // (6) Standard ring/pat kernel. Only the query needs algo/proto/channels filled;
  // the live path recomputes these in taskAppend().
  decision->algo = NCCL_ALGO_RING;
  decision->protocol = NCCL_PROTO_SIMPLE;
  if (query) {
    struct ncclTaskColl task;
    memset(&task, 0, sizeof(task));
    task.func = ncclFuncReduceScatter;
    task.sendbuff = sendbuff;
    task.recvbuff = recvbuff;
    task.count = recvcount;
    task.datatype = datatype;
    NCCLCHECK(getAlgoInfo(comm, &task, /*collNetSupport=*/0, /*nvlsSupport=*/0, /*numPipeOps=*/1, /*simInfo=*/nullptr));
    decision->protocol = task.protocol;
    // Report the traffic-packed channel count the kernel actually runs on, not the
    // tuning cap (task.nMaxChannels), matching the enqueue.cc channel{Lo..Hi} log.
    int packed =
      rcclKernelPackedChannels(comm, ncclFuncReduceScatter, recvcount, datatype, task.protocol, task.nMaxChannels);
#ifdef ENABLE_WARP_SPEED
    // WarpSpeed reports as RING* with channels scaled by nWarps, matching rcclGetAlgoInfo.
    decision->nMaxChannels = task.useWarpSpeed ? task.nMaxChannels / task.nWarps : packed;
    decision->algo = task.useWarpSpeed ? rcclAddonAlgos_t::RCCL_WARP_SPEED : task.algorithm;
#else
    decision->nMaxChannels = packed;
    decision->algo = task.algorithm;
#endif
  }
  return ncclSuccess;
}

bool rcclUseReduceScatterDirect(struct ncclComm* comm, size_t& msgSize) {
  // Direct ReduceScatter is supported for MI350 (gfx950):
  // Only if PXN is enabled
  // - 2 nodes: enable for 128KiB .. 2MiB
  // - 4 nodes: enable up to 4MiB
  // - 8 and 16 nodes: enable up to 8MiB
  static int userDirectReduceScatterInput = rcclParamDirectReduceScatterDisable();
  if (userDirectReduceScatterInput != 0) {
    INFO(NCCL_INIT, "RCCL DIRECT REDUCE-SCATTER has been disabled by environment variable.");
    return false;
  }
  const bool archGfx950 = IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950");
  if (!archGfx950) return false;

  // Check if PXN is disabled - Direct Reduce Scatter requires PXN to be enabled
  if (ncclPxnDisable(comm) != 0) {
    INFO(NCCL_INIT, "RCCL DIRECT REDUCE-SCATTER disabled due to PXN being disabled.");
    return false;
  }

  size_t threshold = rcclParamDirectReduceScatterThreshold();
  if (threshold > -1) {
    // Set threshold to 8MiB hard limit
    // NOTE: If the DirectReduceScatterThreshold / hard-limit is increased, ensure TEMP_BUFF_SIZE (init.cc)
    // is increased accordingly -> TEMP_BUFF_SIZE >= 2 * (max enabled msgSize) for headroom.
    threshold = std::min(threshold, (size_t)8388608);
  } else {
    threshold = 8388608;
  }
  INFO(NCCL_INIT, "RCCL DIRECT REDUCE-SCATTER threshold set to: %zu", threshold);

  if (msgSize > threshold) return false;
  // for 2 nodes, enable if msgSize is in 128KiB .. 2MiB range
  if (comm->nNodes == 2) return (msgSize >= (size_t)131072) && (msgSize <= (size_t)2097152);
  // for 4 nodes, enable if msgSize is up to 4MiB
  if (comm->nNodes == 4) return (msgSize <= (size_t)4194304);
  if (comm->nNodes == 8 || comm->nNodes == 16) return true;
  return false;
}

RCCL_PARAM(HierarchicalReduceScatter, "HIERARCHICAL_REDUCE_SCATTER", 0);

bool rcclUseHierarchicalReduceScatter(struct ncclComm* comm, size_t msgSize) {
  if (comm->nNodes < 8 || rcclParamHierarchicalReduceScatter() != 1 || !comm->hierarchicalCommsInitialized) {
    return false;
  }

  size_t threshold = rcclHierarchicalTempBufferSize(comm->nNodes, /*allGather=*/false, /*reduceScatter=*/true);
  return threshold > 0 && msgSize <= threshold;
}

void rcclSetPxn(struct ncclComm* comm, int& rcclPxnDisable) {
  if (comm->pxnDisable != RCCL_VALUE_UNSET) {
    rcclPxnDisable = comm->pxnDisable;
    return;
  }
  const char* inputStr = getenv("NCCL_PXN_DISABLE");
  const bool archGfx942 = IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx942");
  const bool archGfx950 = IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950");
  comm->enableCustColl = (archGfx942 || archGfx950) && (inputStr && !atoi(inputStr));

  if ((!archGfx942 && !archGfx950) || inputStr) {
    rcclPxnDisable = comm->pxnDisable = RCCL_VALUE_INVALID;
    return;
  }
  const int ranksThreshold = (archGfx942) ? 64 : 32;
  int pxnDisable = (comm->nRanks >= ranksThreshold) ? 0 : 1;
  INFO(NCCL_INIT, "RCCL PXN set as %s (nRanks=%d threshold=%d)", !pxnDisable ? "enabled" : "disabled", comm->nRanks,
       ranksThreshold);
  comm->enableCustColl = !pxnDisable;
  rcclPxnDisable = comm->pxnDisable = pxnDisable;
}

void rcclSetP2pNetChunkSize(struct ncclComm* comm, int& rcclP2pNetChunkSize) {
  if (comm->p2pNetChunkSize != RCCL_VALUE_UNSET) {
    rcclP2pNetChunkSize = comm->p2pNetChunkSize;
    return;
  }
  const char* inputStr = getenv("NCCL_P2P_NET_CHUNKSIZE");
  const bool archGfx942 = IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx942");
  const bool archGfx950 = IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950");
  if ((!archGfx942 && !archGfx950) || inputStr) {
    rcclP2pNetChunkSize = comm->p2pNetChunkSize = RCCL_VALUE_INVALID;
    return;
  }

  int p2pNetChunkSize = RCCL_VALUE_UNSET;
  if (archGfx942) p2pNetChunkSize = (comm->nRanks >= 64) ? (1 << 19) : (1 << 17);
  else if (archGfx950)
    p2pNetChunkSize = (comm->nRanks >= 32) ? (1 << 19) : (comm->nRanks >= 16 ? (1 << 18) : (1 << 17));
  else
    WARN("RCCL P2P attempt to set P2P net chunk size for unsupported arch: %s",
         comm->topo->nodes[GPU].nodes[0].gpu.gcn);
  INFO(NCCL_INIT, "RCCL P2P net chunk size default set to: %d (nRanks=%d)", p2pNetChunkSize, comm->nRanks);
  comm->p2pNetChunkSize = p2pNetChunkSize;
  rcclP2pNetChunkSize = p2pNetChunkSize;
}
#ifdef ENABLE_WARP_SPEED
void rcclSetWarpSpeedCUs(struct ncclComm* comm, int algo, int threadsPerBlock, int& rcclWarpSpeedChannels) {
  static int userChannelControlInput = RCCL_VALUE_UNSET;
  int warpsPerBlock = threadsPerBlock / comm->WarpSize;
  // only adjust channels for RING algorithm
  if (algo != NCCL_ALGO_RING) {
    return;
  }
  if (userChannelControlInput == RCCL_VALUE_UNSET) {
    const char* inputStr = getenv("NCCL_THREAD_THRESHOLDS");
    if (!inputStr) {
      inputStr = getenv("NCCL_MAX_NCHANNELS");
    }
    if (!inputStr) {
      inputStr = getenv("NCCL_MIN_NCHANNELS");
    }
    userChannelControlInput = !inputStr ? 0 : 1;
  }
  if (comm->topo->warpSpeedEnabled) {
    if (!userChannelControlInput) {
      if (rcclParamWarpSpeedCuCount() != 0) {
        rcclWarpSpeedChannels = rcclParamWarpSpeedCuCount() * warpsPerBlock;
        INFO(NCCL_INIT, "RCCL Warp CU count set to user defined %ld resulting in %d channels",
             (long)rcclParamWarpSpeedCuCount(), rcclWarpSpeedChannels);
        return;
      }
    }
    // reuse the existing channel tuning logic if possible
    rcclWarpSpeedChannels = std::min(MAXCHANNELS, rcclWarpSpeedChannels * warpsPerBlock);
    INFO(NCCL_INIT, "RCCL Warp Speed Channels set to %d. Warps per block is set to %d", rcclWarpSpeedChannels,
         warpsPerBlock);
  }
}

bool rcclWarpSpeedSupported(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  if (!comm->topo->warpSpeedEnabled || plan->isSymColl) {
    return false;
  }

  // WarpSpeed is not supported currently for the following cases:
  // 1. if any work batch in the plan contains P2P work
  // 2. if the plan contains AllGatherV-fused work; that kernel
  //    does not implement WarpSpeed's warp-level channel distribution
  // 3. or any collective task is not using RING algorithm
  bool hasP2p = !ncclIntruQueueEmpty(&plan->p2pTaskQueue);
  bool hasBcast = !ncclIntruQueueEmpty(&plan->bcastTaskQueue);
  bool hasNonRing = false;
  struct ncclTaskColl* task = ncclIntruQueueHead(&plan->collTaskQueue);
  while (task != nullptr) {
    if (task->algorithm != NCCL_ALGO_RING || !(task->useWarpSpeed)) {
      hasNonRing = true;
      break;
    }
    task = task->next;
  }
  return (!hasP2p && !hasBcast && !hasNonRing);
}

bool rcclIsAboveWarpSpeedThreshold(struct ncclComm* comm, struct ncclTaskColl* info, size_t nBytes) {
  // single node, full subscription thresholds for AllGather and ReduceScatter
  if (info->func == ncclFuncAllReduce && nBytes >= rcclParamWarpSpeedARThreshold()) {
    return true;
  } else if (info->func == ncclFuncAllGather && nBytes >= rcclParamWarpSpeedAGThreshold()) {
    return true;
  } else if (info->func == ncclFuncReduceScatter && nBytes >= rcclParamWarpSpeedRSThreshold()) {
    return true;
  }
  INFO(NCCL_TUNING, "RCCL WarpSpeed not enabled for %s at %zu bytes as it below the warpSpeed threshold",
       ncclFuncToString(info->func), nBytes);
  return false;
}

bool rcclCanUseWarpSpeedAuto(struct ncclComm* comm, int nNodes) {
  return IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950") && (nNodes == 1) &&
         (rcclParamWarpSpeedAutoMode() != 0) && comm->cuCount > 128; // Only use in SPX mode, 256 CU on gfx950
}

bool rcclWarpSpeedChannelCountSupported(struct ncclComm* comm) {
  return comm->nChannels <= (MAXCHANNELS) / 2;
}

ncclResult_t validChannelsForWarpSpeed(struct ncclComm* comm, struct ncclTaskColl* info) {
  if (info->useWarpSpeed && !rcclWarpSpeedChannelCountSupported(comm)) {
    WARN("WarpSpeed does not support more than %d channels. Current number of channels is %d. To avoid hang, run with "
         "RCCL_WARP_SPEED_AUTO=0",
         MAXCHANNELS / 2, comm->nChannels);
    return ncclInvalidArgument;
  }
  return ncclSuccess;
}

ncclResult_t rcclSetWarpSpeedAuto(struct ncclComm* comm, struct ncclTaskColl* info, size_t nBytes) {
  info->useWarpSpeed = false;
  static bool unrollFactorSet = getenv("RCCL_UNROLL_FACTOR") != nullptr;
  if (!comm->topo->warpSpeedEnabled) return ncclSuccess;
  commSetUnrollFactor(comm); // TODO: reset unroll factor per task rather than per comm
  if (!rcclCollSupportsRing(info->func)) return ncclSuccess;
  if (rcclParamWarpSpeedForceEnable() > 0) { // Manual performance mode
    if (info->algorithm != NCCL_ALGO_RING) {
      INFO(NCCL_TUNING,
           "Overriding %s algorithm with RING for nccl%s at %zu bytes as WarpSpeed is requested and only supports RING",
           ncclAlgoToString(info->algorithm), ncclFuncToString(info->func), nBytes);
      info->algorithm = NCCL_ALGO_RING; // Force Ring when WarpSpeed is enabled in manual mode as it only supports Ring
    }
    // TODO: Remove unroll update when all collectives are optimized
    if (!unrollFactorSet) comm->unroll = NCCL_UNROLL_2;
    info->useWarpSpeed = true;
  } else if (rcclCanUseWarpSpeedAuto(comm, comm->nNodes)) { // Auto performance mode
    // No early return based on the algorithm at the start of the function
    // to allow unroll factor to be reverted to default.
    // This can be changed once per-task unroll factor setting is implemented.
    if (info->algorithm != NCCL_ALGO_RING) {
      return ncclSuccess; // If Ring is not selected, assume it is suboptimal and return
    }
    if (info->func == ncclFuncAllReduce || info->func == ncclFuncAllGather || info->func == ncclFuncReduceScatter) {
      // allReduce now benefits from unroll factor of 2 in all modes due to changing its slicing strategy
      // TODO: Remove unroll update when all collectives are optimized
      if (!unrollFactorSet) comm->unroll = NCCL_UNROLL_2;
    }
    if (rcclIsAboveWarpSpeedThreshold(comm, info, nBytes)) {
      // Skip WarpSpeed when the comm exceeds its channel limit (e.g. RCCL_ENABLE_INTRANET=1 drives
      // nChannels to MAXCHANNELS) instead of failing. Force-enable still errors below.
      if (!rcclWarpSpeedChannelCountSupported(comm)) {
        if (comm->rank == 0)
          INFO(NCCL_TUNING, "RCCL WarpSpeed auto-disabled: %d channels exceeds max %d supported", comm->nChannels,
               MAXCHANNELS / 2);
      } else {
        info->nWarps = 4;
        info->useWarpSpeed = true;
      }
    }
  }
  NCCLCHECK(validChannelsForWarpSpeed(comm, info));
  return ncclSuccess;
}

int rcclGetMaxWarpsPerBlock(struct ncclComm* comm) {
  int warpsPerBlock;
  if (comm->nNodes == 1) {
    warpsPerBlock = RCCL_SINGLE_NODE_MAX_NTHREADS /
                    comm->WarpSize; // For single node, we use half the number of threads for perf reasons.
  } else {
    warpsPerBlock = IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950") ?
                      RCCL_GFX950_MAX_NTHREADS / comm->WarpSize :
                      RCCL_DEFAULT_MAX_NTHREADS / comm->WarpSize;
  }
  return warpsPerBlock;
}

// Compute the bandwidth channel count (nc) when WarpSpeed is enabled, scaling the
// base channel count by the per-block warp multiplier.
int rcclWarpSpeedComputeNChannels(struct ncclComm* comm, int nc, int channelMultiplier, int maxChannels,
                                  int adjustedMaxNchannels, bool userUpdatedMaxChannels) {
  const bool singleNode = comm->nNodes == 1;
  const bool isGfx950 = IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950");
  int maxNchannels;
  // If user didn't override, use requested channels; otherwise keep capped max.
  if (!userUpdatedMaxChannels) {
    maxNchannels = nc * comm->nChannels * channelMultiplier;
    nc = singleNode ? maxNchannels : std::min(maxNchannels, maxChannels);
  } else {
    nc = maxNchannels = std::min(adjustedMaxNchannels * channelMultiplier, MAXCHANNELS);
  }

  if (!userUpdatedMaxChannels && isGfx950 && singleNode && comm->nRanks == 8) {
    // For gfx950 single-node, use half the channels since they are doubled on a single node
    // Remove when all collectives have been optimized
    nc /= 2;
  }
  INFO(NCCL_TUNING, "WarpSpeed enabled: warpSpeedChannelMultiplier %d, maxNchannels %d, nc %d", channelMultiplier,
       maxNchannels, nc);
  return nc;
}

// Adjust the per-collective channel count (nc) for WarpSpeed during algo/channel
// tuning. No-op when WarpSpeed is disabled.
int rcclWarpSpeedAdjustChannels(struct ncclComm* comm, struct ncclTaskColl* info, int nc) {
  if (comm->topo->warpSpeedEnabled) {
    nc /= comm->warpSpeedChannelMultiplier;
    // Temporary check as we reduce CU usage for all collectives
    // TODO: Remove this condition after optimizing all collectives
    if (IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950") && comm->nNodes == 1 && comm->nRanks == 8 &&
        info->func != ncclFuncAllReduce && info->func != ncclFuncAllGather && info->func != ncclFuncReduceScatter &&
        ncclParamMaxNchannels() < 0) {
      nc *= 2;
    }
  }
  return nc;
}
#endif

void rcclGetMaxNthreads(struct ncclComm* comm, int maxNthreads[]) {
  if (IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950")) {
    maxNthreads[NCCL_PROTO_SIMPLE] = maxNthreads[NCCL_PROTO_LL128] = RCCL_GFX950_MAX_NTHREADS;
  } else {
    maxNthreads[NCCL_PROTO_SIMPLE] = maxNthreads[NCCL_PROTO_LL128] = RCCL_DEFAULT_MAX_NTHREADS;
  }
  maxNthreads[NCCL_PROTO_LL] = RCCL_LL_MAX_NTHREADS;
}

void rcclOptThreadBlockSize(struct ncclComm* comm, struct ncclTaskColl* info, size_t nBytes, int& nThreads) {
  static int maxNthreads[NCCL_NUM_PROTOCOLS] = {0};
  if (maxNthreads[NCCL_PROTO_SIMPLE] == 0) rcclGetMaxNthreads(comm, maxNthreads);
  if (rcclParamThreadsPerBlock() != -1) {
    nThreads = rcclParamThreadsPerBlock();
    if (nThreads % comm->WarpSize != 0) {
      nThreads = ((nThreads / comm->WarpSize) + 1) * comm->WarpSize;
      INFO(NCCL_INIT, "RCCL Threads per block adjusted to %d to be multiple of warp size %d", nThreads, comm->WarpSize);
    }
    if (nThreads > maxNthreads[NCCL_PROTO_SIMPLE]) {
      nThreads = maxNthreads[NCCL_PROTO_SIMPLE];
      INFO(NCCL_INIT, "RCCL Threads per block reduced to %d to match max threads", nThreads);
    } else if (nThreads < 3 * comm->WarpSize) {
      nThreads = 3 * comm->WarpSize; // min requirement for tree
      INFO(NCCL_INIT, "RCCL Threads per block increased to %d to be at least one warp", nThreads);
    }
    return;
  }
  if (info->algorithm == NCCL_ALGO_TREE) nThreads = maxNthreads[NCCL_PROTO_SIMPLE]; // Tree now uses all threads always.
  if (info->algorithm == NCCL_ALGO_PAT) nThreads = maxNthreads[NCCL_PROTO_SIMPLE];
  if (comm->nNodes == 1)
    nThreads = RCCL_SINGLE_NODE_MAX_NTHREADS; // For single node, we use half the number of threads for perf reasons.
  // The following should be already set correctly by getNthreads
  // but need to override the changes for TREE and PAT in the previous lines
  else if (info->protocol == NCCL_PROTO_LL) nThreads = maxNthreads[NCCL_PROTO_LL];
  // ReduceScatter small count optimization
  if (info->func == ncclFuncReduceScatter && divUp(nBytes, comm->nRanks) <= 524288)
    nThreads = maxNthreads[NCCL_PROTO_LL];
}

void rcclSetDefaultBuffSizes(struct ncclComm* comm, int defaultBuffSizes[]) {
  static int maxNthreads[NCCL_NUM_PROTOCOLS] = {0};
  if (maxNthreads[NCCL_PROTO_SIMPLE] == 0) rcclGetMaxNthreads(comm, maxNthreads);
  defaultBuffSizes[NCCL_PROTO_LL] =
    NCCL_LL_LINES_PER_THREAD * maxNthreads[NCCL_PROTO_LL] * NCCL_STEPS * sizeof(union ncclLLFifoLine);
  defaultBuffSizes[NCCL_PROTO_LL128] =
    rcclLL128ElemsPerThreadFromArch(comm->archName) * maxNthreads[NCCL_PROTO_LL128] * NCCL_STEPS * sizeof(uint64_t);
  defaultBuffSizes[NCCL_PROTO_SIMPLE] = (1 << 22); /* 4MiB */
}

ncclResult_t rcclFuncMaxSendRecvCount(ncclFunc_t func, int nRanks, size_t count, size_t& maxCount) {
  RCCL_STATIC_EXPOSE_CHECK();
  maxCount = ncclFuncMaxSendRecvCount(func, nRanks, count);
  return ncclSuccess;
}

ncclResult_t commSetUnrollFactor(struct ncclComm* comm) {
  if (rcclParamUnrollFactor() != -1) {
    comm->unroll = rcclParamUnrollFactor(); //-1 to map to 0 based indexing
    if (comm->unroll < NCCL_UNROLL_1 || comm->unroll >= NCCL_NUM_UNROLLS) {
      WARN("Invalid RCCL_UNROLL_FACTOR %d specified. Valid values are 0 to %d corresponding to unroll factors of 1, 2, "
           "4, 8, 16, and 32 respectively.",
           comm->unroll, NCCL_NUM_UNROLLS - 1);
      return ncclInvalidArgument;
    }
    if (!ncclDevFuncUnrollGenerated[comm->unroll]) {
      WARN("RCCL_UNROLL_FACTOR %d (unroll %d) was not built for arch %s; its device function table is empty and "
           "dispatching to it would crash. "
           "Rebuild with this unroll factor, or select one that was generated for this build.",
           comm->unroll, (int)(pow(2.0, (double)comm->unroll)), comm->archName);
      return ncclInvalidArgument;
    }
    INFO(NCCL_INIT, "RCCL Unroll Factor (user set): %d", (int)(pow(2.0, (double)comm->unroll)));
    return ncclSuccess;
  }
  if (IsArchMatch(comm->archName, "gfx950")) {
    if (comm->nNodes == 1) comm->unroll = NCCL_UNROLL_1;
    else comm->unroll = NCCL_UNROLL_2;
  } else if (IsArchMatch(comm->archName, "gfx908") || ((IsArchMatch(comm->archName, "gfx942") && comm->cuCount > 80)))
    comm->unroll = NCCL_UNROLL_2;
  else if (IsArchMatch(comm->archName, "gfx1250")) comm->unroll = NCCL_UNROLL_32;
  else comm->unroll = NCCL_UNROLL_4;

  // Guard against a default that wasn't built for this arch (e.g. the generation
  // matrix was narrowed). Fall back to any generated unroll rather than segfault.
  if (!ncclDevFuncUnrollGenerated[comm->unroll]) {
    int fallback = -1;
    for (int u = NCCL_NUM_UNROLLS - 1; u >= NCCL_UNROLL_1; u--) {
      if (ncclDevFuncUnrollGenerated[u]) {
        fallback = u;
        break;
      }
    }
    if (fallback < 0) {
      WARN("No unroll-factor device function tables were generated for arch %s.", comm->archName);
      return ncclInvalidUsage;
    }
    WARN("Default RCCL unroll factor %d was not built for arch %s; falling back to %d. Set RCCL_UNROLL_FACTOR to "
         "override.",
         (int)(pow(2.0, (double)comm->unroll)), comm->archName, (int)(pow(2.0, (double)fallback)));
    comm->unroll = fallback;
  }

  INFO(NCCL_INIT, "RCCL Unroll Factor (pre-set): %d", (int)(pow(2.0, (double)comm->unroll)));
  return ncclSuccess;
}

RCCL_PARAM(P2pChannelShiftSize, "P2P_SHIFT_SIZE", -1);
ncclResult_t rcclCommSetP2pShiftSize(struct ncclComm* comm) {
  int nP2pChannels = comm->p2pnChannels;
  int nChannelsLog2 = countOneBits(nP2pChannels - 1);
  int shiftSize = rcclParamP2pChannelShiftSize();

  // Use bit-reversal for default/invalid shiftSize (device uses shiftSize==-1 for that path).
  if (shiftSize >= nChannelsLog2) {
    comm->p2pChannelShiftSize = -1;
  } else {
    comm->p2pChannelShiftSize = shiftSize;
  }
  return ncclSuccess;
}

int getFirmwareVersion() {
  uint64_t fw_version = 0;
  ncclResult_t res = amd_smi_getFirmwareVersion(0, &fw_version);
  if (res != ncclSuccess) {
    return -1;
  }
  return fw_version;
}

bool validHsaScratchEnvSetting(const char* hsaScratchEnv, int hipRuntimeVersion, int firmwareVersion,
                               char const* archName) {
  bool hsaScratchEnvSet = (hsaScratchEnv && strcmp(hsaScratchEnv, "1") == 0);
  if (hsaScratchEnvSet) {
    return true;
  }
  if (IsArchMatch(archName, "gfx950")) {
    return (hipRuntimeVersion >= 60443484 && firmwareVersion >= 24);
  }
  if (IsArchMatch(archName, "gfx942")) {
    return (hipRuntimeVersion >= 60443484 && firmwareVersion >= 177);
  }
  return true;
}

// Should match get_arch_guard() in generate.py
bool rcclIsArchSupportedForFunc(struct ncclTaskColl* info, const char* archName) {
  bool supported = true;

  if (info->protocol == NCCL_PROTO_LL128) {
#if defined(ENABLE_LL128)
    if (info->acc)
      supported =
        (IsArchMatch(archName, "gfx942") || IsArchMatch(archName, "gfx950") || IsArchMatch(archName, "gfx1250"));
    else
      supported = (IsArchMatch(archName, "gfx942") || IsArchMatch(archName, "gfx950") ||
                   IsArchMatch(archName, "gfx90a") || IsArchMatch(archName, "gfx1250"));
#else
    supported = false;
#endif
  } else if (info->acc) {
    supported =
      (IsArchMatch(archName, "gfx942") || IsArchMatch(archName, "gfx950") || IsArchMatch(archName, "gfx1250"));
  }

  return supported;
}
