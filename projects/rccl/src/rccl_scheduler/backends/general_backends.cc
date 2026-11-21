/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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

// General: the native Ring/Tree/PAT kernels and the host-composed paths built on
// them (Direct per-peer Send/Recv, hierarchical).

#include <algorithm>
#include <cstring>
#include <vector>

#include "archinfo.h"
#include "bootstrap.h"
#include "debug.h"
#include "enqueue.h"
#include "graph.h"
#include "graph/topo.h"
#include "rccl_scheduler/rccl_scheduler.h"
#include "tuning.h"

namespace {
constexpr int AinicMaxDirectAGScale = 8;
}  // namespace

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
  INFO(NCCL_TUNING, "Hierarchical %s inter: proto=%d channels=%d, intra: proto=%d channels=%d", isAllGather ? "AG" : "RS",
       *protocol, *maxChannels, intraProto, intraChan);
  return ncclSuccess;
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

// msgSize is the total gathered size.
static bool rcclHierarchicalAllGatherEligible(struct ncclComm* comm, size_t msgSize) {
  if (comm->nNodes < 8) return false;
  if (rcclParamHierarchicalAllGather() != 1) return false;

  const int64_t minBytesPerRank = rcclParamHierarchicalAllGatherMinBytesPerRank();
  if (minBytesPerRank > 0 && comm->nRanks > 0 && msgSize / (size_t)comm->nRanks < (size_t)minBytesPerRank) return false;

  size_t threshold = rcclHierarchicalTempBufferSize(comm->nNodes, /*allGather=*/true, /*reduceScatter=*/false);
  return threshold > 0 && msgSize <= threshold;
}

bool rcclUseHierarchicalAllGather(struct ncclComm* comm, size_t msgSize) {
  return comm->hierarchicalCommsInitialized && rcclHierarchicalAllGatherEligible(comm, msgSize);
}

// Collective: every rank reaches this on the same eligible AllGathers. The splits
// run only once every rank is ready, and the readiness handshake runs only on the
// 1st, 2nd, 4th, 8th... call, so a rank that stays not ready costs its peers
// O(log n) bootstrap AllGathers.
static ncclResult_t rcclLazyInitHierarchicalComms(struct ncclComm* comm, bool capturing) {
  const uint64_t call = ++comm->hierarchicalLazyCalls;
  if ((call & (call - 1)) != 0) return ncclSuccess;

  // Ready means not capturing and holding the temp buffer. Allocating it before the
  // vote keeps a one-rank allocation failure from landing after the peers split.
  bool localReady = !capturing;
  if (localReady && rcclReserveHierarchicalTempBuffer(comm) != ncclSuccess) {
    (void)hipGetLastError();  // tolerated, so keep it out of the application's next error check
    WARN("Hierarchical collectives: cannot allocate the temp buffer yet, deferring sub-communicator setup");
    localReady = false;
  }
  std::vector<uint8_t> ready(comm->nRanks);
  ready[comm->rank] = localReady;
  NCCLCHECK(bootstrapAllGather(comm->bootstrap, ready.data(), sizeof(uint8_t)));
  if (std::find(ready.begin(), ready.end(), 0) != ready.end()) return ncclSuccess;

  const ncclResult_t res = rcclEnsureHierarchicalComms(comm);
  if (res != ncclSuccess) WARN("Hierarchical collectives: sub-communicator setup failed (%s)", ncclGetErrorString(res));
  return res;
}

bool rcclUseAllGatherDirect(struct ncclComm* comm, size_t& msgSize) {
  // Check if user explicitly disabled direct AllGather
  static int userDirectAllGatherInput = rcclParamDirectAllGatherDisable();
  if (userDirectAllGatherInput < 0) {
    // DIRECT ALLGATHER disabled on AINIC by default on scale >8 nodes, and enabled otherwise.
    if (rcclUseAinic() && (comm->nNodes > AinicMaxDirectAGScale)) {
      INFO(NCCL_INIT, "RCCL DIRECT ALLGATHER disabled on AINIC by default for %d+ nodes. ", AinicMaxDirectAGScale);
      return false;
    }
  } else if (userDirectAllGatherInput != 0) {
    INFO(NCCL_INIT, "RCCL DIRECT ALLGATHER has been disabled by environment variable.");
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

// Sets decision to Ring/Simple and, when query=true, refines proto/channels via
// getAlgoInfo() and rcclKernelPackedChannels(). Used by rcclSelectAllReduce,
// rcclSelectAllGather and rcclSelectReduceScatter for both the NCCL_ALGO env-var
// early-return and the Ring fall-through tail. threadsPerBlock, if set, gets the kernel's block width (query only).
ncclResult_t rcclRingFallback(struct ncclComm* comm, const void* sendbuff,
                              void* recvbuff, ncclFunc_t func, size_t count,
                              ncclDataType_t datatype, bool query,
                              struct rcclCollDecision* decision, uint32_t* threadsPerBlock) {
  decision->algo = NCCL_ALGO_RING;
  decision->protocol = NCCL_PROTO_SIMPLE;
  if (query) {
    struct ncclTaskColl task;
    memset(&task, 0, sizeof(task));
    task.func = func;
    task.sendbuff = sendbuff;
    task.recvbuff = recvbuff;
    task.count = count;
    task.datatype = datatype;
    NCCLCHECK(getAlgoInfo(comm, &task, 0, 0, 1));
    decision->protocol = task.protocol;
    if (threadsPerBlock != nullptr) {
      *threadsPerBlock = (uint32_t)rcclPlanThreadsPerBlock(comm, task.nWarps);
    }
    int packed = rcclKernelPackedChannels(comm, func, count, datatype,
                                  task.protocol, task.nMaxChannels);
#ifdef ENABLE_WARP_SPEED
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
  // rawTaskAppend does not lower useDirect, so under NCCL_ENQUEUE_REARCH_ENABLE nothing would stage tempBuff.
  if (ncclParamEnqueueRearchEnable()) return false;
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
  if (threshold != (size_t)-1) {
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

bool rcclUseHierarchicalReduceScatter(struct ncclComm* comm, size_t msgSize) {
  if (comm->nNodes < 8 || rcclParamHierarchicalReduceScatter() != 1 || !comm->hierarchicalCommsInitialized) {
    return false;
  }

  // msgSize is the total size.
  const int64_t minBytesPerRank = rcclParamHierarchicalReduceScatterMinBytesPerRank();
  if (minBytesPerRank > 0 && comm->nRanks > 0 && msgSize / (size_t)comm->nRanks < (size_t)minBytesPerRank) return false;

  size_t threshold = rcclHierarchicalTempBufferSize(comm->nNodes, /*allGather=*/false, /*reduceScatter=*/true);
  return threshold > 0 && msgSize <= threshold;
}
namespace {

ncclResult_t ringChoose(rcclSchedCtx& c, rcclCollDecision* d, bool* ok) {
  *ok = true;
  return rcclRingFallback(c.comm, c.sendbuff, c.recvbuff, c.func, c.count, c.datatype, c.query, d);
}

bool directEligible(rcclSchedCtx& c) {
  size_t bytes = c.totalBytes;
  switch (c.func) {
  case ncclFuncAllGather: return rcclUseAllGatherDirect(c.comm, bytes);
  // The native reduce after Direct is an unscaled sum: ncclAvg and PreMulSum take Ring.
  case ncclFuncReduceScatter: return c.op < ncclAvg && rcclUseReduceScatterDirect(c.comm, bytes);
  case ncclFuncAlltoAll: return true;
  default: return false;
  }
}

ncclResult_t directChoose(rcclSchedCtx& c, rcclCollDecision* d, bool* ok) {
  switch (c.func) {
  case ncclFuncAllGather: d->algo = RCCL_DIRECT_ALLGATHER; break;
  case ncclFuncReduceScatter: d->algo = RCCL_DIRECT_REDUCESCATTER; break;
  default: d->algo = RCCL_DIRECT_ALLTOALL; break;
  }
  d->protocol = NCCL_PROTO_SIMPLE;
  d->nMaxChannels = c.comm->p2pnChannels;
  *ok = true;
  return ncclSuccess;
}

// Builds the sub-communicators RCCL_HIERARCHICAL_LAZY_INIT deferred. Collective.
ncclResult_t hierPrepare(rcclSchedCtx& c) {
  struct ncclComm* comm = c.comm;
  if (c.func != ncclFuncAllGather || !comm->hierarchicalEligible || comm->hierarchicalCommsInitialized) {
    return ncclSuccess;
  }
  if (!rcclHierarchicalAllGatherEligible(comm, c.totalBytes)) return ncclSuccess;
  return rcclLazyInitHierarchicalComms(comm, c.capturing);
}

bool hierEligible(rcclSchedCtx& c) {
  switch (c.func) {
  case ncclFuncAllGather: return rcclUseHierarchicalAllGather(c.comm, c.totalBytes);
  case ncclFuncReduceScatter: return c.op == ncclSum && rcclUseHierarchicalReduceScatter(c.comm, c.totalBytes);
  default: return false;
  }
}

ncclResult_t hierChoose(rcclSchedCtx& c, rcclCollDecision* d, bool* ok) {
  *ok = true;
  d->algo = c.func == ncclFuncAllGather ? RCCL_HIERARCHICAL_ALLGATHER : RCCL_HIERARCHICAL_REDUCESCATTER;
  if (!c.query) return ncclSuccess;
  int algo, proto, channels;
  NCCLCHECK(rcclHierarchicalAlgoInfo(c.comm, c.func, c.count, c.datatype, &algo, &proto, &channels));
  d->protocol = proto;
  d->nMaxChannels = channels;
  return ncclSuccess;
}

}  // namespace

const rcclBackend kGeneralRing = {"general.ring", 0, nullptr, ringChoose, nullptr, false};
const rcclBackend kGeneralNative = {"general.native", 0, nullptr, nullptr, nullptr, false};
const rcclBackend kGeneralDirect = {"general.direct", 0, directEligible, directChoose, nullptr, false};
const rcclBackend kGeneralHier = {"general.hier", 0, hierEligible, hierChoose, hierPrepare, false};
