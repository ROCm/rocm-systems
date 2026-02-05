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
#include "include/graph.h"

#ifdef USE_AMDSMI
#include "amd_smi/amdsmi.h"
#else
#include "rocm_smi/rocm_smi.h"
#endif

// Use this param to experiment pipelining new data types besides bfloat16
// Make sure you generate the device code with the new data type (i.e. in generate.py)
RCCL_PARAM(PipelineAllDTypes, "PIPELINE_ALL_DATA_TYPES", 0);

// Use this to assess impact of pipelining on performance.
// Otherwise, it is automatically set for certain archs, datatypes and reduction collectives
RCCL_PARAM(disableReduceCopyPipelining, "DISABLE_REDUCE_COPY_PIPELINING", 0);
RCCL_PARAM(DirectAllGatherThreshold, "DIRECT_ALLGATHER_THRESHOLD", 75497472);
RCCL_PARAM(DirectReduceScatterThreshold, "DIRECT_REDUCE_SCATTER_THRESHOLD", 2097152);
RCCL_PARAM(ThreadsPerBlock, "THREADS_PER_BLOCK", -1);
RCCL_PARAM(UnrollFactor, "UNROLL_FACTOR", -1);
#ifdef ENABLE_WARP_SPEED
RCCL_PARAM(WarpSpeedCuCount, "WARP_SPEED_CU_COUNT", 0);
RCCL_PARAM(WarpSpeedAutoMode, "WARP_SPEED_AUTO", 1);
RCCL_PARAM(WarpSpeedForceEnable, "WARP_SPEED_FORCE_ENABLE", 0);
#endif
#define RCCL_WARP_SPEED_MIN_BYTES (1ULL << 26) // 64 MB


void rcclRestrictMaxChannels(struct ncclComm* comm, int& nc ) {

  if (comm->nNodes > 1 && IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950")) {
    const char* maxNChannelsStr = getenv("NCCL_MAX_NCHANNELS");

    if (maxNChannelsStr) {
      char* end = nullptr;
      long userMax = strtol(maxNChannelsStr, &end, 10);

      const bool valid = (end != maxNChannelsStr && *end == '\0' && userMax > 0);
      if (valid) {
        // 64 is the max number of channels for gfx950 multi-node
        userMax = std::min<long>(userMax, 64);
        const int cap = (int)userMax;
        INFO(NCCL_TUNING, "RCCL MaxChannels is capped to: %i", cap);
        // Cap max channels, but don't permanently shrink comm->nChannels
        // based on a small-message tuning decision (which can legitimately pick nc=1).
        nc = std::min(nc, cap);
        comm->nChannels = std::min(comm->nChannels, cap);
      } else {
        // Invalid / non-positive value: treat as "unset" and apply default restriction.
        INFO(NCCL_TUNING, "RCCL MaxChannels: ignoring invalid NCCL_MAX_NCHANNELS='%s', default capping to 48", maxNChannelsStr);
        nc = std::min(nc, 48);
        comm->nChannels = std::min(comm->nChannels, 48);
      }
    } else {
      // Default restriction for gfx950 multi-node when user hasn't set a valid max.
      nc = std::min(nc, 48);
      comm->nChannels = std::min(comm->nChannels, 48);
      INFO(NCCL_TUNING, "RCCL MaxChannels: default capping to 48");
    }
  }
}

static inline bool rcclCollSupportsRing(ncclFunc_t func) {
  return (func == ncclFuncAllReduce ||
          func == ncclFuncAllGather ||
          func == ncclFuncReduceScatter ||
          func == ncclFuncBroadcast ||
          func == ncclFuncReduce);
}

int32_t rcclGetProtoForGfx12(ncclFunc_t collectiveFunc, size_t sizePerRank){
  int returnVal = NCCL_PROTO_SIMPLE;
  int SingleNodeLLCutoffs[] = {
    /*ncclFuncBroadcast*/     1536,
    /*ncclFuncReduce*/        8192,
    /*ncclFuncAllGather*/     98304,
    /*ncclFuncReduceScatter*/ 98304,
    /*ncclFuncAllReduce*/     913532,
    /*ncclFuncSendRecv*/      0,
    /*ncclFuncSend*/          0,
    /*ncclFuncRecv*/          0
  };
  if(collectiveFunc < sizeof(SingleNodeLLCutoffs)/sizeof(int)) {
    returnVal = (sizePerRank <= SingleNodeLLCutoffs[collectiveFunc]) ? NCCL_PROTO_LL : NCCL_PROTO_SIMPLE;
  }
  return returnVal;
}

void rcclUpdateCollectiveProtocol(struct ncclComm* comm, size_t const& nBytes, struct ncclTaskColl* info) {
  // Honor user input for protocol choice
  static int userProtocolInput = -2;
  size_t sizePerRank = rcclGetSizePerRank(info->func, nBytes, comm->nRanks);
  if (userProtocolInput == -2) {
    const char *protoStr = getenv("NCCL_PROTO");
    userProtocolInput = !protoStr ? 0 : 1;
  }

  if (!userProtocolInput && IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950") && comm->nNodes == 1 && (info->func == ncclFuncAllGather) && sizePerRank <= 88448) {
    // Change LL protocol threshold
    info->protocol = NCCL_PROTO_LL;
  } else if (!userProtocolInput && IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950") && comm->nNodes == 1 && (info->func == ncclFuncReduceScatter) && sizePerRank <= 1048576) {
    // Change LL protocol threshold
    info->protocol = NCCL_PROTO_LL;
  } else if (!userProtocolInput && IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx942") && comm->nNodes == 1 && (info->func == ncclFuncReduceScatter) && sizePerRank <= 352128) {
    // Change LL protocol threshold
    info->protocol = NCCL_PROTO_LL;
  } else if (!userProtocolInput && IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx12") && comm->nNodes == 1){
    info->protocol = rcclGetProtoForGfx12( info->func,sizePerRank);
  } else if(!userProtocolInput && comm->nNodes >= 2 && (info->func == ncclFuncReduceScatter || info->func == ncclFuncAllGather || info->func == ncclFuncAllReduce || info->func == ncclFuncBroadcast || info->func == ncclFuncReduce)) {
    auto tunableIndex = rcclGetTunableIndex(info->func);
    auto llMin = comm->minMaxLLRange[tunableIndex][NCCL_PROTO_LL][RCCL_PROTOCOL_MIN_IDX];
    auto llMax = comm->minMaxLLRange[tunableIndex][NCCL_PROTO_LL][RCCL_PROTOCOL_MAX_IDX];

    auto ll128Min = comm->minMaxLLRange[tunableIndex][NCCL_PROTO_LL128][RCCL_PROTOCOL_MIN_IDX];
    auto ll128Max = comm->minMaxLLRange[tunableIndex][NCCL_PROTO_LL128][RCCL_PROTOCOL_MAX_IDX];

    // Only override model choices if min/max cutoff points are set in the tuning models
    if ((ll128Max != RCCL_LL_LIMITS_UNDEFINED) || (llMax != RCCL_LL_LIMITS_UNDEFINED)) {
      // Keep it simple unless otherwise required
      info->protocol = NCCL_PROTO_SIMPLE;
      size_t sizePerRank = rcclGetSizePerRank(info->func, nBytes, comm->nRanks);
      if (sizePerRank <= llMax && sizePerRank > llMin) {
        info->protocol = NCCL_PROTO_LL;
      }
#if defined(ENABLE_LL128)
      // When LL128 is performant, the next condition overrides the previous LL choice
      if (comm->topo->ll128Enabled) {
        if (info->func == ncclFuncAllReduce) {
          if(comm->nNodes > 2) {
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

ncclResult_t rcclGetAlgoProtoIndex(const char *envStr, const char* algoProtoString[], int nEntries, int& result) {
  if(envStr) {
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
RCCL_PARAM(ChannelTuningEnable, "CHANNEL_TUNING_ENABLE", 1);

ncclResult_t rcclOverrideChannels(struct ncclComm* comm, ncclFunc_t coll, size_t nBytes, int& nc) {
  if(comm->nNodes < 2 || !rcclParamChannelTuningEnable()) {
    INFO(NCCL_TUNING, "RCCL Channel Tuning not applied");
    return ncclSuccess;
  }

  auto tunableIndex = rcclGetTunableIndex(coll);
  if(tunableIndex == RCCL_UNSUPPORTED_TUNABLE){
    INFO(NCCL_TUNING, "tunableIndex:%i not supported", tunableIndex);
    return ncclSuccess;
  }

  int minCTAs = comm->config.minCTAs;
  int maxCTAs = comm->config.maxCTAs;
  int scalingFactor = 1;
#ifdef ENABLE_WARP_SPEED
  if(comm->topo->warpSpeedEnabled) {
    scalingFactor = comm->warpSpeedChannelMultiplier; // each CU can handle 4 warps
  }
#endif
  int minNChannels = ncclParamMinNchannels();
  int maxNChannels = std::max(comm->nChannels / scalingFactor, static_cast<int>(ncclParamMaxNchannels()));
  size_t bytesPerRank = divUp(nBytes, comm->nRanks);

  for(int channelCountIndex = 0; channelCountIndex < RCCL_CHANNELS_TUNABLE_ENTRIES; ++channelCountIndex){
    size_t minByteThreshold = comm->minMaxChannelThresholds[tunableIndex][channelCountIndex][0];
    size_t maxByteThreshold = comm->minMaxChannelThresholds[tunableIndex][channelCountIndex][1];
    INFO(NCCL_TUNING, "nBytes:%lu bytesPerRank:%lu minByteThreshold:%lu maxByteThreshold:%lu  NCCL_MIN_NCHANNELS:%i or NCCL_MAX_NCHANNELS:%i minCTAs:%i maxCTAs:%i", nBytes, bytesPerRank, minByteThreshold, maxByteThreshold, minNChannels, maxNChannels, minCTAs, maxCTAs);
    if(minByteThreshold == CHAN_THRESHOLDS_UNDEFINED || maxByteThreshold == CHAN_THRESHOLDS_UNDEFINED) {
      INFO(NCCL_TUNING, "RCCL tuning model does not define threshold for coll:%i and nbytes:%lu", coll, nBytes);
      break; // Skip undefined thresholds
    }

    if(bytesPerRank > minByteThreshold && bytesPerRank <= maxByteThreshold){
      int channelCount = comm->minMaxChannelThresholds[tunableIndex][channelCountIndex][2];

      //honor user's min/max channels defined through NCCL_MIN_NCHANNELS and NCCL_MAX_NCHANNELS
      if(channelCount >= minNChannels && channelCount <= maxNChannels && channelCount >= minCTAs && channelCount <= maxCTAs){
        nc = comm->minMaxChannelThresholds[tunableIndex][channelCountIndex][2];
        INFO(NCCL_TUNING, "RCCL tuning model overrides nchannels to %i, channels may be decreased further due to MinTrafficPerchannel thresholds", channelCount);
      }
      else{
        INFO(NCCL_TUNING, "RCCL tuning model cannot override nchannels to %i due to conflicting NCCL_MIN_NCHANNELS:%i or NCCL_MAX_NCHANNELS:%i minCTAs:%i maxCTAs:%i", channelCount, minNChannels, maxNChannels, minCTAs, maxCTAs);
      }

      break;
    }

  }
  return ncclSuccess;
}

ncclResult_t rcclOverrideProtocol(const char* ncclProtoStr[], float table[][NCCL_NUM_PROTOCOLS], struct ncclTaskColl* info) {
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
        WARN("Failed to force unsupported protocol %s for function %s with datatype %s", protoOverrideEnv, ncclFuncToString(info->func), ncclDatatypeToString(info->datatype));
        return ncclInternalError;
      } else {
        info->protocol = protoVal;
      }
    }
  }
  return ncclSuccess;
}

ncclResult_t rcclOverrideAlgorithm(const char* ncclAlgoStr[], float table[][NCCL_NUM_PROTOCOLS], struct ncclTaskColl* info) {
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
        WARN("Failed to force unsupported algorithm %s for function %s with datatype %s", algoOverrideEnv, ncclFuncToString(info->func), ncclDatatypeToString(info->datatype));
        return ncclInternalError;
      } else {
        info->algorithm = algoVal;
      }
    }
  }
  return ncclSuccess;
}

void rcclUpdateThreadThreshold(struct ncclComm* comm, size_t const& nBytes, struct ncclTaskColl* info, int& threadThreshold) {
  // Honor user input for thread thresholds
  static int userChannelControlInput = -2;
  if (userChannelControlInput == -2) {
    const char *inputStr = getenv("NCCL_THREAD_THRESHOLDS");
    if (!inputStr) {
      inputStr = getenv("NCCL_MAX_NCHANNELS");
    }
    if (!inputStr) {
      inputStr = getenv("NCCL_MIN_NCHANNELS");
    }
    userChannelControlInput = !inputStr ? 0 : 1;
  }

  if(!userChannelControlInput && comm->nNodes >= 2 && (info->func == ncclFuncReduceScatter || info->func == ncclFuncAllGather)) {
    auto tunableIndex = rcclGetTunableIndex(info->func);
    auto tunedThreshold = comm->minMaxLLRange[tunableIndex][info->protocol][RCCL_PROTOCOL_THREAD_THRESHOLD_IDX];
    if(tunedThreshold != RCCL_LL_LIMITS_UNDEFINED) {
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
        if ( comm->nNodes == 1 ||
             ((comm->nNodes > 1) &&
               nBytes <= (1ULL << 29 /*512MB*/) * (1ULL << (log2i(comm->nNodes) - 1))) ) {
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

extern ncclResult_t getAlgoInfo(
    struct ncclComm* comm, struct ncclTaskColl* task,
    int collNetSupport, int nvlsSupport, int numPipeOps, ncclSimInfo_t* simInfo = NULL
);

ncclResult_t rcclGetAlgoInfo(struct ncclComm* comm, ncclFunc_t coll, uint64_t count, ncclDataType_t dataType,
                             int collNetSupport, int nvlsSupport, int numPipeOps,
                             int* algo, int* protocol, int* maxChannels) {
  RCCL_STATIC_EXPOSE_CHECK();
  int nRanks;
  NCCLCHECK(ncclCommCount(comm, &nRanks));
  size_t msgSize = count * ncclTypeSize(dataType) * nRanks;
  if (coll == ncclFuncAllGather && rcclUseAllGatherDirect(comm, msgSize)) {
    *algo = rcclAddonAlgos_t::RCCL_DIRECT_ALLGATHER;
    *protocol = NCCL_PROTO_SIMPLE; // TODO: consider LL for small messages
    *maxChannels = comm->p2pnChannels;
    return ncclSuccess;
  }
  struct ncclTaskColl task;
  task.func = coll;
  task.count = count;
  task.datatype = dataType;
  NCCLCHECK(getAlgoInfo(comm, &task, collNetSupport, nvlsSupport, numPipeOps));
  *protocol = task.protocol;
#ifdef ENABLE_WARP_SPEED
  *maxChannels = task.useWarpSpeed? task.nMaxChannels / task.nWarps : task.nMaxChannels;
  *algo = task.useWarpSpeed? rcclAddonAlgos_t::RCCL_WARP_SPEED : task.algorithm;
#else
  *maxChannels = task.nMaxChannels;
  *algo = task.algorithm;
#endif
  return ncclSuccess;
}

ncclResult_t rcclGetAlgoName(int algo, const char** algoName) {
  if (algo < 0 || algo >= RCCL_ALGO_COUNT) {
    WARN("Invalid algorithm value: %d", algo);
    return ncclInvalidArgument;
  }
  if(algo >= NCCL_NUM_ALGORITHMS) {
    switch(algo) {
      case rcclAddonAlgos_t::RCCL_DIRECT_ALLGATHER:
        *algoName = "Direct";
        break;
      case rcclAddonAlgos_t::RCCL_MSCCL:
        *algoName = "MSCCL";
        break;
      case rcclAddonAlgos_t::RCCL_MSCCLPP:
        *algoName = "MSCCLPP";
        break;
#ifdef ENABLE_WARP_SPEED
      case rcclAddonAlgos_t::RCCL_WARP_SPEED:
        *algoName = "RING*"; // WarpSpeed (*) uses RING algorithm
        break;
#endif
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

bool rcclUseAllToAllGda(struct ncclComm* comm) {

    //TODO: enable on MI350;  currently tested on MI300X
#ifdef ENABLE_ROCSHMEM
  if (comm->enableRocshmem && IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx942") && comm->nNodes > 1 && (comm->nRanks/comm->nNodes == 8) && comm->rocshmemThreshold <= 1048576) {
      INFO(NCCL_INIT, "Enabling GDA alltoall for RCCL");
      return true;
  }
#endif
  return false;
}

bool rcclUseAllGatherDirect(struct ncclComm* comm, size_t& msgSize) {
  // Check if user explicitly disabled direct AllGather
  static int userDirectAllGatherInput = -2;
  if (userDirectAllGatherInput == -2) {
    const char *inputStr = getenv("RCCL_DIRECT_ALLGATHER_DISABLE");
    userDirectAllGatherInput = !inputStr ? 0 : 1;
  }
  if (userDirectAllGatherInput == 1) {
    INFO(NCCL_INIT, "RCCL DIRECT ALLGATHER has been disabled.");
    return false;
  }

  size_t threshold = rcclParamDirectAllGatherThreshold();

  if (IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950") && threshold != -1) {
     if (comm->nNodes == 1) {
        threshold = 8388608;
     } else if (comm->nNodes < 64) {
        threshold = comm->nNodes * 2097152;
     }
  } else if (IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx942") && threshold != -1) {
	threshold = 4194304;
  }

  comm->enableCustColl = IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950") || IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx942");

  int rankMultiple = comm->nRanks % 8;

  //return (comm->enableCustColl && (comm->nNodes > 1) && (msgSize <= threshold) && (threshold != -1))
  return (comm->enableCustColl && (msgSize <= threshold) && (threshold != -1) && !rankMultiple)
    ;
}

bool rcclUseReduceScatterDirect(struct ncclComm* comm, size_t& msgSize) {
  // Direct ReduceScatter is supported for MI350 (gfx950):
  // Only if PXN is enabled
  // - 2 nodes: enable for 128KiB .. 2MiB
  // - 4 and 8 nodes: enable up to 2MiB
  static int userDirectReduceScatterInput = -2;
  if (userDirectReduceScatterInput == -2) {
    const char *inputStr = getenv("RCCL_DIRECT_REDUCE_SCATTER_DISABLE");
    userDirectReduceScatterInput = !inputStr ? 0 : 1;
  }
  if (userDirectReduceScatterInput == 1) {
    INFO(NCCL_INIT, "RCCL DIRECT REDUCE-SCATTER has been disabled.");
    return false;
  }
  const bool archGfx950 = IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950");
  if (!archGfx950) return false;

  // Check if PXN is disabled - Direct Reduce Scatter requires PXN to be enabled
  if(ncclPxnDisable(comm) != 0) {
    INFO(NCCL_INIT, "RCCL DIRECT REDUCE-SCATTER disabled due to PXN being disabled.");
    return false;
  }

  size_t threshold = rcclParamDirectReduceScatterThreshold();
  if (threshold > -1) { 
    // Set threshold to 2MiB hard limit
    // NOTE: If the DirectReduceScatterThreshold / hard-limit is increased, ensure TEMP_BUFF_SIZE (init.cc)
    // is increased accordingly -> TEMP_BUFF_SIZE >= 2 * (max enabled msgSize) for headroom.
    threshold = std::min(threshold, (size_t)2097152);
  } else {
    threshold = 2097152;
  }
  INFO(NCCL_INIT, "RCCL DIRECT REDUCE-SCATTER threshold set to: %zu", threshold);

  if (msgSize > threshold) return false;
  // for 2 nodes, enable if msgSize is in 128KiB .. 2MiB range
  if (comm->nNodes == 2) return msgSize >= (size_t)131072;
  if (comm->nNodes == 8 || comm->nNodes == 4) return true;
  return false;
}


void rcclSetPxn(struct ncclComm* comm,  int& rcclPxnDisable) {
  static int pxnDisable = RCCL_VALUE_UNSET;
  comm->enableCustColl = false;
  if(pxnDisable == RCCL_VALUE_UNSET) {
    const char *inputStr = getenv("NCCL_PXN_DISABLE");
    const bool archGfx942 = IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx942");
    const bool archGfx950 = IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950");
    comm->enableCustColl = (archGfx942 || archGfx950) && (inputStr && !atoi(inputStr));

    if((!archGfx942 && !archGfx950) || inputStr) {
      rcclPxnDisable = pxnDisable = RCCL_VALUE_INVALID;
      return;
    }
    const int ranksThreshold = (archGfx942)? 64 : 32;
    pxnDisable = (comm->nRanks >= ranksThreshold)? 0 : 1;
    INFO(NCCL_INIT, "RCCL PXN set as %s", !pxnDisable? "enabled" : "disabled");
  }
  rcclPxnDisable = pxnDisable;
  comm->enableCustColl = !pxnDisable;
}

void rcclSetP2pNetChunkSize(struct ncclComm* comm,  int& rcclP2pNetChunkSize) {
  static int p2pNetChunkSize = RCCL_VALUE_UNSET;
  if(p2pNetChunkSize == RCCL_VALUE_UNSET) {
    const char *inputStr = getenv("NCCL_P2P_NET_CHUNKSIZE");
    const bool archGfx942 = IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx942");
    const bool archGfx950 = IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950");
    if((!archGfx942 && !archGfx950) || inputStr) {
      rcclP2pNetChunkSize = p2pNetChunkSize = RCCL_VALUE_INVALID;
      return;
    }

    if(archGfx942)
      p2pNetChunkSize = (comm->nRanks >= 64)? (1 << 19) : (1 << 17);
    else  if(archGfx950)
      p2pNetChunkSize = (comm->nRanks >= 32) ? (1 << 19) : (comm->nRanks >= 16 ? (1 << 18) : (1 << 17));
    else
      WARN("RCCL P2P attempt to set P2P net chunk size for unsupported arch: %s", comm->topo->nodes[GPU].nodes[0].gpu.gcn);
    INFO(NCCL_INIT, "RCCL P2P net chunk size default set to: %d", p2pNetChunkSize);
  }
  rcclP2pNetChunkSize = p2pNetChunkSize;
}
#ifdef ENABLE_WARP_SPEED
void rcclSetWarpSpeedCUs(struct ncclComm* comm, int algo, int threadsPerBlock, int& rcclWarpSpeedChannels) {
  static int userChannelControlInput = RCCL_VALUE_UNSET;
  int warpsPerBlock = threadsPerBlock / comm->WarpSize;
  // only adjust channels for RING algorithm
  if(algo != NCCL_ALGO_RING) {
    return;
  }
  if (userChannelControlInput == RCCL_VALUE_UNSET) {
    const char *inputStr = getenv("NCCL_THREAD_THRESHOLDS");
    if (!inputStr) {
      inputStr = getenv("NCCL_MAX_NCHANNELS");
    }
    if (!inputStr) {
      inputStr = getenv("NCCL_MIN_NCHANNELS");
    }
    userChannelControlInput = !inputStr ? 0 : 1;
  }
  if(comm->topo->warpSpeedEnabled) {
    if(!userChannelControlInput) {
      if(rcclParamWarpSpeedCuCount() != 0) {
        rcclWarpSpeedChannels = rcclParamWarpSpeedCuCount() * warpsPerBlock;
        INFO(NCCL_INIT, "RCCL Warp CU count set to user defined %lld resulting in %d channels", rcclParamWarpSpeedCuCount(), rcclWarpSpeedChannels);
        return;
      }
    }
    // reuse the existing channel tuning logic if possible
    rcclWarpSpeedChannels = std::min(MAXCHANNELS, rcclWarpSpeedChannels * warpsPerBlock);
    INFO(NCCL_INIT, "RCCL Warp Speed Channels set to %d. Warps per block is set to %d", rcclWarpSpeedChannels, warpsPerBlock);
  }
}

void rcclSetWarpSpeedSupportAndFinalCuCount(struct ncclComm* comm, struct ncclKernelPlan* plan, int nChannels, int& support, int &cuCount) {
  if(!comm->topo->warpSpeedEnabled) {
    support = 0;
    cuCount = nChannels;
    return;
  }
  // WarpSpeed is not supported currently for the following cases:
  // 1. if any work batch in the plan contains P2P work
  // 2. or any collective task is not using RING algorithm
  bool hasP2p = !ncclIntruQueueEmpty(&plan->p2pTaskQueue);
  bool hasNonRing = false;
  struct ncclTaskColl* task = ncclIntruQueueHead(&plan->collTaskQueue);
  while (task != nullptr) {
    if (task->algorithm != NCCL_ALGO_RING || !(task->useWarpSpeed)) {
      hasNonRing = true;
      break;
    }
    task = task->next;
  }
  int warpsPerBlock = plan->threadPerBlock / comm->WarpSize;
  support = (hasP2p || hasNonRing) ? 0 : 1;
  cuCount = (support == 0)? nChannels : nChannels / warpsPerBlock + ((nChannels % warpsPerBlock) != 0 ? 1 : 0); // each CU can handle warpsPerBlock
}

bool rcclCanUseWarpSpeedAuto(struct ncclComm* comm, int nNodes) {
  return IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950") && (nNodes == 1) && (rcclParamWarpSpeedAutoMode() != 0);
}

void rcclSetWarpSpeedAuto(struct ncclComm* comm, struct ncclTaskColl* info, size_t nBytes) {
  info->useWarpSpeed = false;
  static bool unrollFactorSet = getenv("RCCL_UNROLL_FACTOR") != nullptr;
  if(!comm->topo->warpSpeedEnabled) return;
  commSetUnrollFactor(comm);  // TODO: reset unroll factor per task rather than per comm
  if(!rcclCollSupportsRing(info->func)) return;
  if (rcclParamWarpSpeedForceEnable() > 0) { // Manual performance mode
    if(info->algorithm != NCCL_ALGO_RING) {
      INFO(NCCL_TUNING, "Overriding %s algorithm with RING for nccl%s at %zu bytes as WarpSpeed is requested and only supports RING", ncclAlgoToString(info->algorithm), ncclFuncToString(info->func), nBytes);
      info->algorithm = NCCL_ALGO_RING; // Force Ring when WarpSpeed is enabled in manual mode as it only supports Ring
    }
    // TODO: Remove unroll update when all collectives are optimized
    if(!unrollFactorSet) comm->unroll =  NCCL_UNROLL_2;
    info->useWarpSpeed = true;
  } else if(rcclCanUseWarpSpeedAuto(comm, comm->nNodes)) { // Auto performance mode
    size_t minBytes = 0;
    // No early return based on the algorithm at the start of the function
    // to allow unroll factor to be reverted to default.
    // This can be changed once per-task unroll factor setting is implemented.
    if(info->algorithm != NCCL_ALGO_RING) {
      return; // If Ring is not selected, assume it is suboptimal and return
    }
    if(info->func == ncclFuncAllReduce) {
       // allReduce now benefits from unroll factor of 2 in all modes due to changing its slicing strategy
       // TODO: Remove unroll update when all collectives are optimized
      if(!unrollFactorSet) comm->unroll =  NCCL_UNROLL_2;
      minBytes = RCCL_WARP_SPEED_MIN_BYTES;
    }
    // temporarily disabling WarpSpeed for AllGather and ReduceScatter in auto mode
    // if(info->func == ncclFuncAllReduce || info->func == ncclFuncAllGather) minBytes = RCCL_WARP_SPEED_MIN_BYTES;
    // else if (info->func == ncclFuncReduceScatter) minBytes = RCCL_WARP_SPEED_MIN_BYTES << 2; // ReduceScatter requires higher message size to benefit from WarpSpeed
    if(nBytes >= minBytes && minBytes > 0) {
      info->nWarps = 4;
      info->useWarpSpeed = true;
    }
  }
}

int rcclGetMaxWarpsPerBlock(struct ncclComm* comm) {
  int warpsPerBlock;
  if(comm->nNodes == 1) {
    warpsPerBlock = RCCL_SINGLE_NODE_MAX_NTHREADS / comm->WarpSize; // For single node, we use half the number of threads for perf reasons.
  } else {
    warpsPerBlock = IsArchMatch(comm->topo->nodes[GPU].nodes[0].gpu.gcn, "gfx950")?
                                                          RCCL_GFX950_MAX_NTHREADS / comm->WarpSize:
                                                          RCCL_DEFAULT_MAX_NTHREADS / comm->WarpSize;
  }
  return warpsPerBlock;
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
  if(rcclParamThreadsPerBlock() != -1) {
    nThreads = rcclParamThreadsPerBlock();
    if(nThreads % comm->WarpSize != 0) {
      nThreads = ((nThreads / comm->WarpSize) + 1) * comm->WarpSize;
      INFO(NCCL_INIT, "RCCL Threads per block adjusted to %d to be multiple of warp size %d", nThreads, comm->WarpSize);
    }
    if(nThreads > maxNthreads[NCCL_PROTO_SIMPLE]) {
      nThreads = maxNthreads[NCCL_PROTO_SIMPLE];
      INFO(NCCL_INIT, "RCCL Threads per block reduced to %d to match max threads", nThreads);
    } else if (nThreads < 3 * comm->WarpSize) {
      nThreads = 3 * comm->WarpSize; // min requirement for tree
      INFO(NCCL_INIT, "RCCL Threads per block increased to %d to be at least one warp", nThreads);
    }
    return;
  }
  if (info->algorithm == NCCL_ALGO_TREE) nThreads = maxNthreads[NCCL_PROTO_SIMPLE]; // Tree now uses all threads always.
  if (info->algorithm == NCCL_ALGO_PAT)  nThreads = maxNthreads[NCCL_PROTO_SIMPLE];
  if (comm->nNodes == 1) nThreads = RCCL_SINGLE_NODE_MAX_NTHREADS; // For single node, we use half the number of threads for perf reasons.
  // The following should be already set correctly by getNthreads
  // but need to override the changes for TREE and PAT in the previous lines
  else if (info->protocol == NCCL_PROTO_LL) nThreads =  maxNthreads[NCCL_PROTO_LL];
  // ReduceScatter small count optimization
  if (info->func == ncclFuncReduceScatter && divUp(nBytes, comm->nRanks) <= 524288) nThreads = maxNthreads[NCCL_PROTO_LL];
}

void rcclSetDefaultBuffSizes(struct ncclComm* comm, int defaultBuffSizes[]) {
  static int maxNthreads[NCCL_NUM_PROTOCOLS] = {0};
  if (maxNthreads[NCCL_PROTO_SIMPLE] == 0) rcclGetMaxNthreads(comm, maxNthreads);
  defaultBuffSizes[NCCL_PROTO_LL]     = NCCL_LL_LINES_PER_THREAD*maxNthreads[NCCL_PROTO_LL]*NCCL_STEPS*sizeof(union ncclLLFifoLine);
  defaultBuffSizes[NCCL_PROTO_LL128]  = NCCL_LL128_ELEMS_PER_THREAD*maxNthreads[NCCL_PROTO_LL128]*NCCL_STEPS*sizeof(uint64_t);
  defaultBuffSizes[NCCL_PROTO_SIMPLE] = (1 << 22); /* 4MiB */
}

ncclResult_t rcclFuncMaxSendRecvCount(ncclFunc_t func, int nRanks, size_t count, size_t& maxCount) {
  RCCL_STATIC_EXPOSE_CHECK();
  maxCount = ncclFuncMaxSendRecvCount(func, nRanks, count);
  return ncclSuccess;
}

ncclResult_t commSetUnrollFactor(struct ncclComm* comm) {
  if( rcclParamUnrollFactor() != -1 ) {
    comm->unroll = rcclParamUnrollFactor(); //-1 to map to 0 based indexing
    if(comm->unroll < NCCL_UNROLL_1 || comm->unroll >= NCCL_NUM_UNROLLS) {
      WARN("Invalid RCCL_UNROLL_FACTOR %d specified. Valid values are 0 to 2 corresponding to unroll factors of 1, 2, and 4 respectively.", comm->unroll);
      return ncclInvalidArgument;
    }
    INFO(NCCL_INIT, "RCCL Unroll Factor (user set): %d", (int) (pow(2.0, (double)comm->unroll)));
    return ncclSuccess;
  }
  if(IsArchMatch(comm->archName, "gfx950")) {
    if(comm->nNodes == 1)
      comm->unroll = NCCL_UNROLL_1;
    else
      comm->unroll = NCCL_UNROLL_2;
  }
  else if(IsArchMatch(comm->archName, "gfx908") || ((IsArchMatch(comm->archName, "gfx942") && comm->cuCount > 80)))
    comm->unroll = NCCL_UNROLL_2;
  else
    comm->unroll = NCCL_UNROLL_4;

  INFO(NCCL_INIT, "RCCL Unroll Factor (pre-set): %d", (int) (pow(2.0, (double)comm->unroll)));
  return ncclSuccess;
}

int getFirmwareVersion() {
  uint64_t fw_version = -1;

#ifdef USE_AMDSMI
  amdsmi_status_t ret;
  ret = amdsmi_init(AMDSMI_INIT_AMD_GPUS);
  if (ret != AMDSMI_STATUS_SUCCESS) {
    ERROR("Could not initialize amd-smi");
    return -1;
  }

  uint32_t socket_count = 0;
  amdsmi_get_socket_handles(&socket_count, nullptr);
  std::vector<amdsmi_socket_handle> sockets(socket_count);
  amdsmi_get_socket_handles(&socket_count, sockets.data());

  uint32_t num_gpus_per_socket = 0;
  amdsmi_get_processor_handles(sockets[0], &num_gpus_per_socket, nullptr);
  std::vector<amdsmi_processor_handle> processor_handles(num_gpus_per_socket);
  amdsmi_get_processor_handles(sockets[0], &num_gpus_per_socket, processor_handles.data());

  amdsmi_fw_info_t info;
  ret = amdsmi_get_fw_info(processor_handles[0], &info);
  if (ret != AMDSMI_STATUS_SUCCESS) {
    ERROR("Could not query firmware info using amd-smi");
    return -1;
  }

  fw_version = info.fw_info_list[0].fw_version;

#else
  rsmi_status_t ret;
  ret = rsmi_init(0);
  if (ret != RSMI_STATUS_SUCCESS) {
    ERROR("Could not initialize rocm-smi");
    return -1;
  }

  ret = rsmi_dev_firmware_version_get(0, RSMI_FW_BLOCK_MEC, &fw_version);
  if (ret != RSMI_STATUS_SUCCESS) {
    ERROR("Could not query firmware info using rocm-smi");
    return -1;
  }
#endif

  return fw_version;
}

bool validHsaScratchEnvSetting(const char*hsaScratchEnv, int hipRuntimeVersion, int firmwareVersion, char const* archName) {
  bool hsaScratchEnvSet = (hsaScratchEnv && strcmp(hsaScratchEnv,"1") == 0);
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
      supported = (IsArchMatch(archName, "gfx942") || IsArchMatch(archName, "gfx950"));
    else
      supported = (IsArchMatch(archName, "gfx942") || IsArchMatch(archName, "gfx950") || IsArchMatch(archName, "gfx90a"));
#else
    supported = false;
#endif
  } else if (info->acc) {
    supported = (IsArchMatch(archName, "gfx942") || IsArchMatch(archName, "gfx950"));
  }

  return supported;
}
