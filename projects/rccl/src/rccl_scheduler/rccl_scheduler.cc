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

#include "rccl_scheduler/rccl_scheduler.h"

#include <cstring>

#include "archinfo.h"
#include "debug.h"
#include "dev_runtime.h"
#include "enqueue.h"
#include "group.h"
#include "param.h"
#include "rccl_vars.h"

// Scheduler knobs. Thresholds left unset resolve against the comm's arch table.

// True when the user set `name` to a non-empty value.
bool rcclEnvIsSet(const char* name) {
  const char* v = ncclGetEnv(name);
  return v != nullptr && v[0] != '\0';
}

RCCL_PARAM(DirectAllGatherThreshold, "DIRECT_ALLGATHER_THRESHOLD", 75497472);
RCCL_PARAM(DirectReduceScatterThreshold, "DIRECT_REDUCE_SCATTER_THRESHOLD", 8388608);
RCCL_PARAM(DirectReduceScatterDisable, "DIRECT_REDUCE_SCATTER_DISABLE", 0);
RCCL_PARAM(DirectAllGatherDisable, "DIRECT_ALLGATHER_DISABLE", -1);
// -1 = unset: fall back to the per-arch default (see rcclCeAllReduceEnabled).
RCCL_PARAM(CeAllReduce, "CE_ALLREDUCE", -1);
// -1 = auto (enabled on gfx1250, disabled elsewhere); 0 = disabled; 1 = bypass CTAPolicy check.
RCCL_PARAM(ForceCeAllReduce, "FORCE_CE_ALLREDUCE", -1);
// Opt-in. Unlike CE AllReduce these do not follow the per-arch default.
RCCL_PARAM(CeReduceScatter, "CE_REDUCESCATTER", 0);
// -1 = auto (enabled on gfx1250, disabled elsewhere); 0 = disabled; 1 = bypass CTAPolicy check.
RCCL_PARAM(ForceSymmetric,   "FORCE_SYMMETRIC",    0);
RCCL_PARAM(ForceDda,         "FORCE_DDA",          0);           // non-zero: prefer DDA over all other backends
RCCL_PARAM(CeArMaxMsgBytes,    "CE_AR_2SHOT_MAX_BYTES",   -1);  // -1 = use ceArMax from arch table (2-shot)
RCCL_PARAM(CeArStagingBytes,   "CE_AR_STAGING_BYTES",   -1);  // -1 = use NCCL_CE_AR_STAGING_BYTES default
RCCL_PARAM(CeArRegMaxMsgBytes, "CE_AR_REG_MAX_MSG_BYTES", -1); // -1 = use ceArRegMax (registered)
// -1 = use arch table. When set, overrides ceNonRegMax and ceRegMax for all
// collectives except AllReduce (which has its own RCCL_CE_AR_*_MSG_BYTES knobs).
// Applies regardless of registration mode (non-reg and reg paths both use this cap).
RCCL_PARAM(CeCollMaxBytes, "CE_COLL_MAX_BYTES", -1);
// -1 = use arch table. When set, overrides symMaxR2 and symMaxR2Graph for all
// collectives. Applies regardless of graph-mode or registration mode.
RCCL_PARAM(SymKMaxBytes, "SYM_K_MAX_BYTES", -1);
RCCL_PARAM(CeArMinMsgBytes,  "CE_AR_2SHOT_MIN_BYTES",  -1);
RCCL_PARAM(CeCollMinBytes,   "CE_COLL_MIN_BYTES",    -1);
RCCL_PARAM(SymKMinBytes,     "SYM_K_MIN_BYTES",      -1);

// Common DDA protocol-tier knobs, shared by every fabric collective (no
// per-collective variants). For a given collective's size:
//   size <= DdaLLThreshold     -> LL    one-shot (16B lines)
//   size <= DdaLL128Threshold  -> LL128 one-shot (128B lines)
//   otherwise                  -> Simple (flat one-shot / tree two-shot)
// Constraint: DdaLLThreshold <= DdaLL128Threshold <= DdaThreshold. Setting an
// enable flag (or its threshold) to 0 disables that tier and falls through to
// the next, so each protocol can be A/B'd in isolation at runtime.
//
// The three thresholds default to kDdaThresholdUnset (-1) rather than to their
// sizes: 0 is a meaningful setting (disable the tier), so "unset" needs a value
// of its own for the arch tables to be able to supply a default. Read them
// through rcclDda{LL,LL128,Vmm}Threshold(), which resolve unset against this
// comm's arch table and yield 0 (no DDA) for arches that have none.
RCCL_PARAM(DdaEnable, "DDA_ENABLE", 1);
RCCL_PARAM(DdaThreshold, "DDA_THRESHOLD", kDdaThresholdUnset);
RCCL_PARAM(DdaLL, "DDA_LL", 1);
RCCL_PARAM(DdaLLThreshold, "DDA_LL_THRESHOLD", kDdaThresholdUnset);
RCCL_PARAM(DdaLLOneShotThreshold, "DDA_LL_ONESHOT_THRESHOLD", (size_t)(1) * 1024 * 1024); // 1 MiB
RCCL_PARAM(DdaLLTwoShotThreshold, "DDA_LL_TWOSHOT_THRESHOLD", (size_t)(2) * 1024 * 1024); // 2 MiB
RCCL_PARAM(DdaLL128OneShotThreshold, "DDA_LL128_ONESHOT_THRESHOLD", (size_t)(4) * 1024 * 1024); // 4 MiB
RCCL_PARAM(DdaLL128TwoShotThreshold, "DDA_LL128_TWOSHOT_THRESHOLD", (size_t)(64) * 1024 * 1024); // 64 MiB
// Opt in to the LDS-staged LL128 two-shot AllReduce writer on gfx1250 wave32 from 2 MiB shards.
RCCL_PARAM(DdaLL128AsyncWire, "DDA_LL128_ASYNC_WIRE", 0);
// LL128 one-shot and cooperative two-shot AllReduce spread each slice's peer loop over a warp group
// (all_reduce_dda_ll128_peerpar.h); 0 keeps the one-warp-per-slice kernels.
RCCL_PARAM(DdaLL128ArPeerParallel, "DDA_LL128_AR_PEER_PARALLEL", 1);
RCCL_PARAM(DdaLL128, "DDA_LL128", kDdaThresholdUnset);
RCCL_PARAM(DdaLL128Threshold, "DDA_LL128_THRESHOLD", kDdaThresholdUnset);
// Broadcast DDA: unset follows the arch table (any Broadcast DDA tier), 0 off, >0 on.
RCCL_PARAM(DdaBc, "DDA_BC", kDdaThresholdUnset);
// Development only: lets Broadcast LL run on the single-node IPC DDA path (e.g. gfx950).
RCCL_PARAM(DdaBcAllowIpc, "DDA_BC_ALLOW_IPC", 0);
// When set, bypass the per-arch tuning table entirely and use base-commit
// env-var defaults for all thresholds (DDA, CE, symMaxR2).  Useful for
// isolating arch-table effects without rebuilding.
RCCL_PARAM(IgnoreArchTable, "IGNORE_ARCH_TABLE", 0);
// Logs every rule and fallback the scheduler tries, and why it passed or took it.
RCCL_PARAM(SchedTrace, "SCHED_TRACE", 0);
RCCL_PARAM(HierarchicalAllGather, "HIERARCHICAL_ALLGATHER", 1);

// Per rank, so that fixed-size startup AllGathers (PyTorch DDP gathers one int64)
// stay off the hierarchical path at any job size. 0 removes the floor.
RCCL_PARAM(HierarchicalAllGatherMinBytesPerRank, "HIERARCHICAL_ALLGATHER_MIN_BYTES_PER_RANK", 16);
RCCL_PARAM(HierarchicalLazyInit, "HIERARCHICAL_LAZY_INIT", 0);
RCCL_PARAM(HierarchicalReduceScatter, "HIERARCHICAL_REDUCE_SCATTER", 0);

// Per rank, as for AllGather. 0 removes the floor.
RCCL_PARAM(HierarchicalReduceScatterMinBytesPerRank, "HIERARCHICAL_REDUCE_SCATTER_MIN_BYTES_PER_RANK", 16);

void rcclSchedCtxInit(rcclSchedCtx* ctx, struct ncclComm* comm, ncclFunc_t func, const void* sendbuff,
                      void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, cudaStream_t stream,
                      bool query, bool graphCapturingHint) {
  memset(ctx, 0, sizeof(*ctx));
  ctx->comm = comm;
  ctx->func = func;
  ctx->sendbuff = sendbuff;
  ctx->recvbuff = recvbuff;
  ctx->count = count;
  ctx->datatype = datatype;
  ctx->op = op;
  ctx->stream = stream;
  ctx->query = query;
  ctx->graphCapturingHint = graphCapturingHint;
  ctx->typeSize = ncclTypeSize(datatype);
  ctx->countBytes = count * ctx->typeSize;
  const bool perPeer = func == ncclFuncAllGather || func == ncclFuncReduceScatter || func == ncclFuncAlltoAll;
  ctx->totalBytes = perPeer ? (size_t)comm->nRanks * ctx->countBytes : ctx->countBytes;
  ctx->archTable = extAlgoArchTable(comm);
  ctx->trace = rcclParamSchedTrace() != 0;
  ctx->base.algo = NCCL_ALGO_RING;
  ctx->base.protocol = NCCL_PROTO_SIMPLE;
}

ncclResult_t rcclSchedFindWindows(rcclSchedCtx& ctx) {
  if (ctx.windowsFound) return ncclSuccess;
  ncclDevrFindWindow(ctx.comm, ctx.sendbuff, &ctx.sendWin);
  ncclDevrFindWindow(ctx.comm, ctx.recvbuff, &ctx.recvWin);
  ctx.hasSysmemSegment = ncclDevrWindowHasSysmemSegment(ctx.sendWin) || ncclDevrWindowHasSysmemSegment(ctx.recvWin);
  NCCLCHECK(ncclGetSymRegType(ctx.sendWin, ctx.recvWin, &ctx.winRegType));
  ctx.windowsFound = true;
  return ncclSuccess;
}

namespace {

void trace(const rcclSchedCtx& ctx, const rcclBackend& m, const char* what) {
  if (ctx.trace) INFO(NCCL_TUNING, "%s trace%s: %s %s", ctx.name, ctx.query ? " (query)" : "", m.name, what);
}

ncclResult_t anyHolds(rcclSchedCtx& ctx, rcclFactMask mask, bool* any) {
  *any = false;
  for (int f = 0; mask != 0 && f < RCCL_FACT_COUNT; f++, mask >>= 1) {
    if ((mask & 1) == 0) continue;
    NCCLCHECK(rcclFactHolds(ctx, (rcclFact)f, any));
    if (*any) return ncclSuccess;
  }
  return ncclSuccess;
}

ncclResult_t allHold(rcclSchedCtx& ctx, rcclFactMask mask, bool* all) {
  *all = true;
  for (int f = 0; mask != 0 && f < RCCL_FACT_COUNT; f++, mask >>= 1) {
    if ((mask & 1) == 0) continue;
    NCCLCHECK(rcclFactHolds(ctx, (rcclFact)f, all));
    if (!*all) return ncclSuccess;
  }
  return ncclSuccess;
}

ncclResult_t ruleFit(rcclSchedCtx& ctx, const rcclRule& rule, rcclFit* fit) {
  *fit = RCCL_FIT_OFF;
  bool hit;
  NCCLCHECK(anyHolds(ctx, rule.avoid, &hit));
  if (hit) return ncclSuccess;
  NCCLCHECK(allHold(ctx, rule.need, &hit));
  if (!hit) return ncclSuccess;
  NCCLCHECK(anyHolds(ctx, rule.anySize, &hit));
  if (hit) {
    *fit = RCCL_FIT_IN;
    return ncclSuccess;
  }
  NCCLCHECK(rcclLimitFit(ctx, rule.window, fit));
  if (*fit == RCCL_FIT_IN) {
    NCCLCHECK(anyHolds(ctx, rule.yield, &hit));
    if (hit) *fit = RCCL_FIT_OUT;
  }
  return ncclSuccess;
}

ncclResult_t tryBackend(rcclSchedCtx& ctx, const rcclBackend& m, rcclCollDecision* cand, bool* ok) {
  *ok = false;
  if (m.usesWindows) NCCLCHECK(rcclSchedFindWindows(ctx));
  if (m.prepare != nullptr && !ctx.query) NCCLCHECK(m.prepare(ctx));
  if (m.eligible != nullptr && !m.eligible(ctx)) return ncclSuccess;
  *cand = ctx.base;
  if (m.choose == nullptr) {
    *ok = true;
    return ncclSuccess;
  }
  return m.choose(ctx, cand, ok);
}

// One pass over a row's rules.
struct rcclSearch {
  rcclSchedCtx& ctx;
  const rcclRow& row;
  rcclBackendMask_t preferred;
  bool explicitPreference;
  const rcclBackend* winner;
  rcclCollDecision winnerDecision;
  const rcclBackend* fallback;  // first in-window candidate
  rcclCollDecision fallbackDecision;
};

ncclResult_t consider(rcclSearch& s, const rcclRule& rule) {
  rcclFit fit;
  NCCLCHECK(ruleFit(s.ctx, rule, &fit));
  const rcclBackend& m = *rule.backend;
  if (fit == RCCL_FIT_OFF) {
    trace(s.ctx, m, "skipped: conditions");
    return ncclSuccess;
  }
  if (fit == RCCL_FIT_OUT && (!s.explicitPreference || m.outright)) {
    trace(s.ctx, m, "skipped: outside window");
    return ncclSuccess;
  }
  rcclCollDecision cand;
  bool ok;
  NCCLCHECK(tryBackend(s.ctx, m, &cand, &ok));
  if (!ok) {
    trace(s.ctx, m, "skipped: not eligible");
    return ncclSuccess;
  }
  const bool inWindow = fit == RCCL_FIT_IN;
  const bool isPreferred = s.explicitPreference ? (s.preferred & m.kind) != 0 : inWindow;
  if (m.outright || isPreferred) {
    trace(s.ctx, m, "taken");
    s.winner = &m;
    s.winnerDecision = cand;
  } else if (inWindow && s.fallback == nullptr) {
    trace(s.ctx, m, "kept as fallback (not preferred)");
    s.fallback = &m;
    s.fallbackDecision = cand;
  } else {
    trace(s.ctx, m, "skipped: not preferred");
  }
  return ncclSuccess;
}

ncclResult_t search(rcclSearch& s) {
  for (size_t i = 0; i < s.row.nRules && s.winner == nullptr; i++) {
    const rcclRule& rule = s.row.rules[i];
    if (rule.group == nullptr) {
      NCCLCHECK(consider(s, rule));
      continue;
    }
    for (size_t j = 0; j < rule.group->nRules && s.winner == nullptr; j++)
      NCCLCHECK(consider(s, rcclRuleInGroup(rule, rule.group->rules[j])));
  }
  return ncclSuccess;
}

ncclResult_t decide(const rcclSchedCtx& ctx, const rcclBackend& m, const rcclCollDecision& cand,
                    rcclCollDecision* decision) {
  *decision = cand;
  if (!ctx.query) {
    INFO(NCCL_TUNING, "%s: %s selected algo=%d proto=%d nMaxChannels=%u", ctx.name, m.name, decision->algo, decision->protocol, decision->nMaxChannels);
  }
  return ncclSuccess;
}

ncclResult_t runFallbacks(rcclSchedCtx& ctx, const rcclRow& row, rcclCollDecision* decision) {
  for (size_t i = 0; i + 1 < row.nFallbacks; i++) {
    rcclCollDecision cand;
    bool ok;
    NCCLCHECK(tryBackend(ctx, *row.fallbacks[i], &cand, &ok));
    if (ok) {
      trace(ctx, *row.fallbacks[i], "taken as fallback");
      return decide(ctx, *row.fallbacks[i], cand, decision);
    }
    trace(ctx, *row.fallbacks[i], "skipped fallback: not eligible");
  }
  const rcclBackend& last = *row.fallbacks[row.nFallbacks - 1];
  trace(ctx, last, "taken as last fallback");
  *decision = ctx.base;
  bool ok = true;
  if (last.choose != nullptr) NCCLCHECK(last.choose(ctx, decision, &ok));
  return ncclSuccess;
}

}  // namespace

bool rcclLimitIn(rcclSchedCtx& ctx, rcclLimit limit) {
  rcclFit fit;
  return rcclLimitFit(ctx, limit, &fit) == ncclSuccess && fit == RCCL_FIT_IN;
}

ncclResult_t rcclSchedule(rcclSchedCtx& ctx, const rcclRow& row, rcclCollDecision* decision) {
  ctx.name = row.name;
  for (size_t i = 0; i < row.nPrologue; i++) {
    bool holds;
    NCCLCHECK(rcclFactHolds(ctx, row.prologue[i].fact, &holds));
    if (holds && row.prologue[i].skipRules) {
      if (ctx.trace) INFO(NCCL_TUNING, "%s trace: rules skipped (%s)", ctx.name, rcclFactName(row.prologue[i].fact));
      return runFallbacks(ctx, row, decision);
    }
  }

  const rcclBackendMask_t preferredBackends = row.preferred != nullptr ? row.preferred(ctx.comm) : RCCL_BACKEND_ALL;
  rcclSearch s{ctx, row, preferredBackends, preferredBackends != RCCL_BACKEND_ALL, nullptr, {}, nullptr, {}};
  NCCLCHECK(search(s));
  if (s.winner != nullptr) return decide(ctx, *s.winner, s.winnerDecision, decision);

  if (s.explicitPreference) ATTN("preferred backend (mask=0x%x) unavailable, falling back", (unsigned)preferredBackends);
  if (s.fallback != nullptr) return decide(ctx, *s.fallback, s.fallbackDecision, decision);
  return runFallbacks(ctx, row, decision);
}

const rcclPolicy& rcclPolicyFor(const ncclComm* comm) {
  if (IsArchMatch(comm->archName, kRcclGfx1250Policy.arch)) return kRcclGfx1250Policy;
  return kRcclDefaultPolicy;
}

rcclBackendMask_t rcclAllReducePreferredBackends(const ncclComm* comm) {
  if (rcclParamForceSymmetric()) return RCCL_BACKEND_SYMMETRIC;
  if (rcclParamForceDda()) return RCCL_BACKEND_DDA;
  if (comm->config.CTAPolicy & NCCL_CTA_POLICY_ZERO) return RCCL_BACKEND_CE;
  return RCCL_BACKEND_ALL;
}

// Entry points. Each builds the context and runs its collective's row.
// query=false is the live dispatch path; query=true is side-effect free and
// takes capture from graphCapturingHint (rcclGetCollImplInfo).

ncclResult_t rcclSelectAllReduce(struct ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                 ncclDataType_t datatype, ncclRedOp_t op, cudaStream_t stream, bool query,
                                 bool graphCapturingHint, struct rcclCollDecision* decision) {
  rcclSchedCtx ctx;
  rcclSchedCtxInit(&ctx, comm, ncclFuncAllReduce, sendbuff, recvbuff, count, datatype, op, stream, query,
                   graphCapturingHint);
  return rcclSchedule(ctx, *rcclPolicyFor(comm).allReduce, decision);
}

ncclResult_t rcclSelectAllGather(struct ncclComm* comm, const void* sendbuff, void* recvbuff, size_t sendcount,
                                 ncclDataType_t datatype, cudaStream_t stream, bool query, bool graphCapturingHint,
                                 struct rcclCollDecision* decision) {
  rcclSchedCtx ctx;
  rcclSchedCtxInit(&ctx, comm, ncclFuncAllGather, sendbuff, recvbuff, sendcount, datatype, ncclSum, stream, query,
                   graphCapturingHint);
  return rcclSchedule(ctx, *rcclPolicyFor(comm).allGather, decision);
}

ncclResult_t rcclSelectReduceScatter(struct ncclComm* comm, const void* sendbuff, void* recvbuff, size_t recvcount,
                                     ncclDataType_t datatype, ncclRedOp_t op, bool query,
                                     struct rcclCollDecision* decision, bool graphCapturingHint) {
  rcclSchedCtx ctx;
  rcclSchedCtxInit(&ctx, comm, ncclFuncReduceScatter, sendbuff, recvbuff, recvcount, datatype, op,
                   /*stream=*/nullptr, query, graphCapturingHint);
  return rcclSchedule(ctx, *rcclPolicyFor(comm).reduceScatter, decision);
}

ncclResult_t rcclSelectAlltoAll(struct ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                ncclDataType_t datatype, cudaStream_t stream, bool query, bool graphCapturingHint,
                                struct rcclCollDecision* decision) {
  rcclSchedCtx ctx;
  rcclSchedCtxInit(&ctx, comm, ncclFuncAlltoAll, sendbuff, recvbuff, count, datatype, ncclSum, stream, query,
                   graphCapturingHint);
  return rcclSchedule(ctx, *rcclPolicyFor(comm).alltoAll, decision);
}

ncclResult_t rcclSelectBroadcast(struct ncclComm* comm, size_t count, ncclDataType_t datatype, int root, bool query,
                                 struct rcclCollDecision* decision) {
  if (comm == nullptr) return ncclInvalidArgument;
  rcclSchedCtx ctx;
  rcclSchedCtxInit(&ctx, comm, ncclFuncBroadcast, /*sendbuff=*/nullptr, /*recvbuff=*/nullptr, count, datatype, ncclSum,
                   /*stream=*/nullptr, query, /*graphCapturingHint=*/false);
  ctx.root = root;
  return rcclSchedule(ctx, *rcclPolicyFor(comm).broadcast, decision);
}

// Reporting API for rccl-tests and tools: the backend the scheduler picks for given operands.

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
  // AlltoAll has no tuned ring/tree kernel: taskAppend() splits it into per-peer
  // Send/Recv over the p2p channels, so report that shape. The backend-aware
  // answer (DDA / CE / GIN for these operands) comes from rcclGetCollImplInfo().
  if (coll == ncclFuncAlltoAll) {
    *algo = rcclAddonAlgos_t::RCCL_DIRECT_ALLTOALL;
    *protocol = NCCL_PROTO_SIMPLE;
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

  // AllReduce, AllGather, ReduceScatter, and AlltoAll are wired to the unified
  // decision. They return the actual backend (CE / DDA / symmetric / kernel) for
  // these operands so rccl-tests can label numbers with the implementation that
  // ran. graphCapturing lets the caller declare graph mode, which the
  // (out-of-capture) query cannot detect on its own -- see the header comment.
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
    NCCLCHECK(rcclSelectAllGather(comm, sendbuff, recvbuff, (size_t)count, dataType, /*stream=*/nullptr,
                                  /*query=*/true, /*graphCapturingHint=*/graphCapturing != 0, &decision));
    *algo = decision.algo;
    *protocol = decision.protocol;
    *maxChannels = decision.nMaxChannels;
    return ncclSuccess;
  }

  if (coll == ncclFuncReduceScatter) {
    struct rcclCollDecision decision;
    NCCLCHECK(rcclSelectReduceScatter(comm, sendbuff, recvbuff, (size_t)count, dataType, op, /*query=*/true,
                                      &decision, /*graphCapturingHint=*/graphCapturing != 0));
    *algo = decision.algo;
    *protocol = decision.protocol;
    *maxChannels = decision.nMaxChannels;
    return ncclSuccess;
  }

  if (coll == ncclFuncAlltoAll) {
    struct rcclCollDecision decision;
    NCCLCHECK(rcclSelectAlltoAll(comm, sendbuff, recvbuff, (size_t)count, dataType, /*stream=*/nullptr,
                                 /*query=*/true, /*graphCapturingHint=*/graphCapturing != 0, &decision));
    *algo = decision.algo;
    *protocol = decision.protocol;
    *maxChannels = decision.nMaxChannels;
    return ncclSuccess;
  }

  // Broadcast: report DDA when picked, else the generic query below. Check-mode, suspended and
  // revoked comms always enqueue, so they skip the selector like ncclBroadcast_impl does.
  if (coll == ncclFuncBroadcast && !rcclCollectiveMustUseEnqueuePath(comm)) {
    struct rcclCollDecision decision;
    NCCLCHECK(rcclSelectBroadcast(comm, (size_t)count, dataType, /*root=*/0, /*query=*/true, &decision));
    if (decision.algo >= NCCL_NUM_ALGORITHMS) {
      *algo = decision.algo;
      *protocol = decision.protocol;
      *maxChannels = decision.nMaxChannels;
      return ncclSuccess;
    }
  }

  // Other collectives: fall back to the size/algo query until they are migrated
  // onto rcclSelectXxx().
  return rcclGetAlgoInfo(comm, coll, count, dataType, /*collNetSupport=*/0, /*nvlsSupport=*/0, /*numPipeOps=*/1, algo,
                         protocol, maxChannels);
}

ncclResult_t rcclSymKGetInfo(struct ncclComm* comm, ncclFunc_t coll, uint64_t count, ncclDataType_t dataType,
                             ncclRedOp_t op, int* algo, int* protocol, int* maxChannels) {
  RCCL_STATIC_EXPOSE_CHECK();
  if (algo == nullptr || protocol == nullptr || maxChannels == nullptr) return ncclInvalidArgument;
  if (rcclSymkQuery(comm, coll, count, dataType, op, algo, protocol, maxChannels)) {
    // Across trays AllGather symk runs only as a TDM kernel in the TDM window. No graph flag here: use the eager one.
    if (coll != ncclFuncAllGather || !comm->spansTrays) return ncclSuccess;
    const size_t totalBytes = (size_t)comm->nRanks * count * ncclTypeSize(dataType);
    if (*algo == (int)rcclAddonAlgos_t::RCCL_SYMMETRIC_TDM &&
        rcclSymTdmWindowContains(extAlgoArchTable(comm), ncclFuncAllGather, totalBytes, /*spansTrays=*/true,
                                 /*graphMode=*/false)) {
      return ncclSuccess;
    }
  }
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
    case rcclAddonAlgos_t::RCCL_DIRECT_ALLTOALL:
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
    case rcclAddonAlgos_t::RCCL_SYMMETRIC_TDM:
      *algoName = "SymTDM";
      break;
    case rcclAddonAlgos_t::RCCL_CE_2SHOT:
      *algoName = "CE2";
      break;
    case rcclAddonAlgos_t::RCCL_CE_REGISTERED:
      *algoName = "CE";
      break;
    case rcclAddonAlgos_t::RCCL_CE_SCRATCH:
      *algoName = "CE-Scratch";
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
    case rcclAddonAlgos_t::RCCL_GIN_SDMA:
      *algoName = "GIN-SDMA";
      break;
    case rcclAddonAlgos_t::RCCL_A2A_GDA:
      *algoName = "A2A-GDA";
      break;
    case rcclAddonAlgos_t::RCCL_A2A_GIN_SDMA:
      *algoName = "A2A-GIN";
      break;
    case rcclAddonAlgos_t::RCCL_AG_GIN_SDMA:
      *algoName = "AG-GIN";
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
