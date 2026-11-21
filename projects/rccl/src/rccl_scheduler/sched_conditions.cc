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

// Conditions the rules test: per-call facts (cached booleans) and size windows
// resolved against the comm's arch table.

#include "rccl_scheduler/rccl_scheduler.h"

#include <algorithm>

#include "archinfo.h"
#include "ce_coll.h"
#include "debug.h"
#include "graph.h"
#include "group.h"
#include "sym_kernels.h"

// Returns true when the user has restricted the algorithm set via NCCL_ALGO.
// When true, CE / DDA / Symmetric dispatch is skipped so getAlgoInfo() reaches
// Ring/Tree exactly as the user requested.  Cached to avoid repeated getenv().
bool rcclNcclAlgoEnvIsSet() {
  static int cached = -1;
  if (cached == -1) cached = (ncclGetEnv("NCCL_ALGO") != nullptr) ? 1 : 0;
  return cached == 1;
}


namespace {

ncclResult_t probeCapture(rcclSchedCtx& ctx, bool* capturing) {
  if (ctx.query) {
    *capturing = ctx.graphCapturingHint;
  } else {
    struct ncclCudaGraph graph;
    NCCLCHECK(ncclCudaGetCapturingGraph(&graph, ctx.stream, ctx.comm->config.graphUsageMode));
    *capturing = ncclCudaGraphValid(graph);
    // A capture holding no AllReduce must still keep CE AllReduce off for later eager calls.
    rcclCeAllReduceGraphLatchTick(ctx.comm, *capturing);
  }
  ctx.capturing = *capturing;
  ctx.base.ceCapturing = *capturing;
  return ncclSuccess;
}

// gfx1250: symMaxR2 withdrew symk from a registered AllReduce that DDA serves better.
ncclResult_t symkWithdrawn(rcclSchedCtx& ctx, bool* withdrawn) {
  *withdrawn = false;
  bool fabric, requested;
  NCCLCHECK(rcclFactHolds(ctx, RCCL_FACT_FABRIC, &fabric));
  if (!fabric) return ncclSuccess;
  NCCLCHECK(rcclFactHolds(ctx, RCCL_FACT_SYMK_REQUESTED, &requested));
  if (!requested) return ncclSuccess;
  const size_t bytes = ctx.totalBytes;
  const size_t symMin = rcclSymMinR2CapTab(ctx.archTable, ncclFuncAllReduce);
  const size_t symMax = rcclSymMaxR2CapTab(ctx.archTable, ncclFuncAllReduce, ctx.capturing);
  const bool belowMin = symMin > 0 && bytes < symMin;
  *withdrawn = !belowMin && bytes > symMax && !rcclGinArYieldsToDda(ctx);
  return ncclSuccess;
}

// Under capture the withdrawn call would run Ring; DDA replaces it within Ring's grid.
ncclResult_t rerouteRing(rcclSchedCtx& ctx, bool* holds) {
  *holds = false;
  bool withdrawn, capturing, mustEnqueue;
  NCCLCHECK(symkWithdrawn(ctx, &withdrawn));
  NCCLCHECK(rcclFactHolds(ctx, RCCL_FACT_CAPTURING, &capturing));
  NCCLCHECK(rcclFactHolds(ctx, RCCL_FACT_MUST_ENQUEUE, &mustEnqueue));
  if (!withdrawn || !capturing || mustEnqueue) return ncclSuccess;
  if (ctx.totalBytes > rcclDdaArRegCaptureMaxBytesTab(ctx.archTable, ctx.comm->nRanks)) return ncclSuccess;
  rcclCollDecision ring = {};
  uint32_t threads = 0;
  NCCLCHECK(rcclRingFallback(ctx.comm, ctx.sendbuff, ctx.recvbuff, ncclFuncAllReduce, ctx.count, ctx.datatype,
                             /*query=*/true, &ring, &threads));
  *holds = ring.algo == NCCL_ALGO_RING;  // fp8 plans Tree: no reroute
  if (*holds) {
    ctx.ddaCap.blocks = std::max<uint32_t>(ring.nMaxChannels, 1);
    ctx.ddaCap.threads = std::max<uint32_t>(threads, 1);
  }
  return ncclSuccess;
}

// Eager, the withdrawn call would run CE registered; DDA LL128 replaces it within CE's grid.
ncclResult_t rerouteCeRegistered(rcclSchedCtx& ctx, bool* holds) {
  *holds = false;
  bool withdrawn, capturing, mustEnqueue;
  NCCLCHECK(symkWithdrawn(ctx, &withdrawn));
  NCCLCHECK(rcclFactHolds(ctx, RCCL_FACT_CAPTURING, &capturing));
  if (!withdrawn || capturing) return ncclSuccess;
  NCCLCHECK(rcclSchedFindWindows(ctx));
  if (!rcclCeArRegisteredAvailable(ctx) || !rcclCeArRegisteredInWindow(ctx)) return ncclSuccess;
  const size_t bytes = ctx.totalBytes;
  if (bytes > rcclDdaLL128ThresholdTab(ctx.archTable, ncclFuncAllReduce, ctx.comm->nRanks)) return ncclSuccess;
  if (!rcclDdaArRegEagerShardFitsTab(ctx.archTable, ctx.comm->nRanks, bytes)) return ncclSuccess;
  NCCLCHECK(rcclFactHolds(ctx, RCCL_FACT_MUST_ENQUEUE, &mustEnqueue));
  if (mustEnqueue) return ncclSuccess;
  *holds = true;
  ctx.ddaCap.blocks = (uint32_t)std::max(rcclCeArRegisteredBlocks(ctx), 1);
  ctx.ddaCap.threads = NCCL_CE_REDUCE_THREADS;
  return ncclSuccess;
}

ncclResult_t symkVeto(rcclSchedCtx& ctx, bool* veto) {
  if (ctx.func != ncclFuncAllReduce) return rcclFactHolds(ctx, RCCL_FACT_SYMK_ELIGIBLE, veto);
  // AllReduce: below symMinR2, on a GIN yield or on a reroute, DDA may take a registered call.
  *veto = false;
  bool requested, reroute;
  NCCLCHECK(rcclFactHolds(ctx, RCCL_FACT_SYMK_REQUESTED, &requested));
  if (!requested) return ncclSuccess;
  const size_t symMin = rcclSymMinR2CapTab(ctx.archTable, ncclFuncAllReduce);
  if (symMin > 0 && ctx.totalBytes < symMin) return ncclSuccess;
  if (rcclGinArYieldsToDda(ctx)) return ncclSuccess;
  NCCLCHECK(rcclFactHolds(ctx, RCCL_FACT_REROUTE_RING, &reroute));
  if (reroute) return ncclSuccess;
  NCCLCHECK(rcclFactHolds(ctx, RCCL_FACT_REROUTE_CE_REG, &reroute));
  *veto = !reroute;
  return ncclSuccess;
}

ncclResult_t ceRsOk(rcclSchedCtx& ctx, bool* ok) {
  bool inGroup, graphOk;
  NCCLCHECK(rcclFactHolds(ctx, RCCL_FACT_IN_GROUP, &inGroup));
  NCCLCHECK(rcclFactHolds(ctx, RCCL_FACT_CE_AR_GRAPH_OK, &graphOk));
  *ok = !inGroup && graphOk && rcclUseCeReduceScatter(ctx.comm, ctx.count, ctx.datatype, ctx.op);
  return ncclSuccess;
}

ncclResult_t evaluate(rcclSchedCtx& ctx, rcclFact fact, bool* v) {
  struct ncclComm* comm = ctx.comm;
  switch (fact) {
  case RCCL_FACT_ALGO_ENV: *v = rcclNcclAlgoEnvIsSet(); return ncclSuccess;
  case RCCL_FACT_IN_GROUP: *v = ncclGroupDepth != 0; return ncclSuccess;
  case RCCL_FACT_MUST_ENQUEUE: *v = rcclCollectiveMustUseEnqueuePath(comm); return ncclSuccess;
  case RCCL_FACT_CTA_ZERO: *v = (comm->config.CTAPolicy & NCCL_CTA_POLICY_ZERO) != 0; return ncclSuccess;
  case RCCL_FACT_FABRIC: *v = IsArchMatch(comm->archName, "gfx1250"); return ncclSuccess;
  case RCCL_FACT_IN_PLACE: *v = ctx.sendbuff != nullptr && ctx.sendbuff == ctx.recvbuff; return ncclSuccess;
  case RCCL_FACT_CAPTURING: return probeCapture(ctx, v);
  case RCCL_FACT_CE_AR_GRAPH_OK:
    *v = rcclCeArGraphOk(ctx);
    ctx.base.ceArGraphAllowed = *v;
    return ncclSuccess;
  case RCCL_FACT_SYSMEM:
    NCCLCHECK(rcclSchedFindWindows(ctx));
    *v = ctx.hasSysmemSegment;
    return ncclSuccess;
  case RCCL_FACT_SYMK_REQUESTED:
    NCCLCHECK(rcclSchedFindWindows(ctx));
    *v = rcclSymkRequested(ctx);
    return ncclSuccess;
  case RCCL_FACT_SYMK_ELIGIBLE:
    NCCLCHECK(rcclSchedFindWindows(ctx));
    *v = rcclSymkEligible(ctx);
    return ncclSuccess;
  case RCCL_FACT_SYMK_VETO: return symkVeto(ctx, v);
  case RCCL_FACT_CE_AR_2SHOT_OK:
    *v = rcclUseCeAr2Shot(comm, ctx.count, ctx.datatype, ctx.op, /*acc=*/nullptr);
    return ncclSuccess;
  case RCCL_FACT_CE_RS_OK: return ceRsOk(ctx, v);
  case RCCL_FACT_REROUTE_RING: return rerouteRing(ctx, v);
  case RCCL_FACT_REROUTE_CE_REG: return rerouteCeRegistered(ctx, v);
  case RCCL_FACT_COUNT: break;
  }
  *v = false;
  return ncclSuccess;
}

}  // namespace

ncclResult_t rcclFactHolds(rcclSchedCtx& ctx, rcclFact fact, bool* holds) {
  const rcclFactMask bit = rcclFactBit(fact);
  if ((ctx.factsKnown & bit) == 0) {
    bool v = false;
    NCCLCHECK(evaluate(ctx, fact, &v));
    ctx.factsKnown |= bit;
    if (v) ctx.factsTrue |= bit;
  }
  *holds = (ctx.factsTrue & bit) != 0;
  return ncclSuccess;
}

// Size windows.

namespace {

rcclFit tier(size_t bytes, size_t max) {
  if (max == 0) return RCCL_FIT_OFF;
  return bytes <= max ? RCCL_FIT_IN : RCCL_FIT_OUT;
}

rcclFit hard(bool fits) { return fits ? RCCL_FIT_IN : RCCL_FIT_OFF; }

}  // namespace

ncclResult_t rcclLimitFit(rcclSchedCtx& ctx, rcclLimit limit, rcclFit* fit) {
  const rcclArchThresholds* t = ctx.archTable;
  const ncclFunc_t func = ctx.func;
  const int nRanks = ctx.comm->nRanks;
  const size_t bytes = ctx.totalBytes;
  switch (limit) {
  case RCCL_LIMIT_NONE: *fit = RCCL_FIT_IN; break;
  case RCCL_LIMIT_DDA_ENTRY: *fit = hard(rcclWindowFits(bytes, 0, rcclDdaEntryThresholdTab(t, func, nRanks))); break;
  case RCCL_LIMIT_DDA_LL: *fit = tier(bytes, rcclDdaLLThresholdTab(t, func, nRanks)); break;
  case RCCL_LIMIT_DDA_LL128: *fit = tier(bytes, rcclDdaLL128ThresholdTab(t, func, nRanks)); break;
  case RCCL_LIMIT_DDA_VMM:
    if (func == ncclFuncAlltoAll) {
      *fit = tier(bytes, rcclDdaVmmThresholdTab(t, func, nRanks));
    } else {
      NCCLCHECK(rcclSchedFindWindows(ctx));
      const bool graphMode = func != ncclFuncReduceScatter && ctx.capturing;
      *fit = tier(bytes, rcclDdaVmmThresholdCtxTab(t, func, ctx.winRegType, graphMode, nRanks));
    }
    break;
  case RCCL_LIMIT_DDA_AR_VMM_FIRST:
    NCCLCHECK(rcclSchedFindWindows(ctx));
    *fit = hard(rcclDdaArVmmFirstTab(t, nRanks, bytes, ctx.winRegType));
    break;
  case RCCL_LIMIT_DDA_SCRATCH: *fit = hard(bytes <= (size_t)ctx.comm->ddaScratchBytes); break;
  case RCCL_LIMIT_CE_REG:
    *fit = rcclWindowFits(bytes, rcclCeRegMinTab(t, func), rcclCeRegMaxTab(t, func)) ? RCCL_FIT_IN : RCCL_FIT_OUT;
    break;
  case RCCL_LIMIT_CE_NONREG: *fit = hard(rcclWindowFits(bytes, rcclCeNonRegMinTab(t, func), rcclCeNonRegMaxTab(t, func))); break;
  case RCCL_LIMIT_CE_REG_AG:
    NCCLCHECK(rcclSchedFindWindows(ctx));
    *fit = hard(rcclAllGatherCeRegisteredWindowTab(t, bytes, ctx.winRegType, ctx.capturing));
    break;
  case RCCL_LIMIT_SYMK: {
    const size_t symMin = rcclSymMinR2CapTab(t, func);
    const size_t symMax = rcclSymMaxR2CapTab(t, func, ctx.capturing);
    *fit = (symMin > 0 && bytes < symMin) || bytes > symMax ? RCCL_FIT_OUT : RCCL_FIT_IN;
    break;
  }
  case RCCL_LIMIT_COUNT: *fit = RCCL_FIT_OFF; break;
  }
  return ncclSuccess;
}

// Arch-table resolvers. Env overrides win, then the comm's table, then base defaults.

// This comm's arch table, or NULL when the arch has no DDA tuning at all.
// comm->archThresholds is populated at init; the lookup covers comms assembled
// by hand (unit tests) and resolves them like an SPX comm.
const rcclArchThresholds* extAlgoArchTable(const ncclComm* comm) {
  if (comm == nullptr) return nullptr;
  if (rcclParamIgnoreArchTable() && !IsArchMatch(comm->archName, "gfx942")
                                 && !IsArchMatch(comm->archName, "gfx950")) return nullptr;
  return comm->archThresholds != nullptr ? comm->archThresholds : rcclGetArchThresholds(comm->archName);
}

// kDdaThresholdUnset means the user left the env var alone; anything else is
// their choice and wins, including 0 to disable that tier.
bool ddaThresholdFromEnv(int64_t param, size_t* threshold) {
  if (param == kDdaThresholdUnset) return false;
  *threshold = param > 0 ? (size_t)param : 0;
  return true;
}

// Table entry for `func`. Collectives past AlltoAll have no slot and stay 0.
size_t funcThresholdFromTable(const size_t* caps, ncclFunc_t func) {
  return (unsigned)func < RCCL_DDA_FUNC_COUNT ? caps[func] : 0;
}

// R2 symmetric-kernel size cap. Graph capture uses symMaxR2Graph; eager uses
// symMaxR2. kThreshUnlimited (SIZE_MAX) means no suppression; non-zero literal = byte cap.
// Table entries use kThreshUnlimited explicitly -- 0 is not used in these arrays.
size_t rcclSymMaxR2CapTab(const rcclArchThresholds* table, ncclFunc_t func, bool graphMode) {
  const int64_t param = rcclParamSymKMaxBytes();
  if (param >= 0) return (size_t)param;
  if (table == nullptr) return SIZE_MAX;
  const size_t* caps = graphMode ? table->symMaxR2Graph : table->symMaxR2;
  return funcThresholdFromTable(caps, func);
}

// R2 symmetric-kernel lower-bound per collective.  Below this size DDA is
// faster than symk; symk is suppressed so DDA can win.  0 = no suppression.
size_t rcclSymMinR2CapTab(const rcclArchThresholds* table, ncclFunc_t func) {
  const int64_t param = rcclParamSymKMinBytes();
  if (param >= 0) return (size_t)param;
  if (table == nullptr) return 0;
  return funcThresholdFromTable(table->symMinR2, func);
}

// Where the hierarchical LSA AllGather kernel runs, the table's hierarchical window replaces the
// AllGather symk bounds in both eager and graph mode. RCCL_SYM_K_MIN_BYTES / RCCL_SYM_K_MAX_BYTES still win.
static bool rcclAgHierSymWindow(struct ncclComm* comm, const rcclArchThresholds* table) {
  return table != nullptr && table->agHierSymMinR2 != 0 && ncclSymkAgHierGroups(comm) > 0;
}
size_t rcclAgSymMaxR2Cap(struct ncclComm* comm, const rcclArchThresholds* table, bool graphMode) {
  if (rcclParamSymKMaxBytes() < 0 && rcclAgHierSymWindow(comm, table)) return table->agHierSymMaxR2;
  return rcclSymMaxR2CapTab(table, ncclFuncAllGather, graphMode);
}
size_t rcclAgSymMinR2Cap(struct ncclComm* comm, const rcclArchThresholds* table) {
  if (rcclParamSymKMinBytes() < 0 && rcclAgHierSymWindow(comm, table)) return table->agHierSymMinR2;
  return rcclSymMinR2CapTab(table, ncclFuncAllGather);
}

// True when totalBytes is in the TDM window. RCCL_SYM_K_MIN_BYTES / RCCL_SYM_K_MAX_BYTES narrow it.
bool rcclSymTdmWindowContains(const rcclArchThresholds* table, ncclFunc_t func, size_t totalBytes,
                                     bool spansTrays, bool graphMode) {
  if (table == nullptr) return false;
  size_t windowMin = funcThresholdFromTable(table->symTdmMinR2, func);
  size_t windowMax = funcThresholdFromTable(graphMode ? table->symTdmMaxR2Graph : table->symTdmMaxR2, func);
  if (windowMin == 0 || windowMax == 0) return false;
  const size_t crossTrayMax = funcThresholdFromTable(table->symTdmCrossTrayMaxR2, func);
  if (spansTrays && crossTrayMax != 0) windowMax = std::min(windowMax, crossTrayMax);
  const int64_t userMin = rcclParamSymKMinBytes();
  const int64_t userMax = rcclParamSymKMaxBytes();
  if (userMin >= 0) windowMin = std::max(windowMin, (size_t)userMin);
  if (userMax >= 0) windowMax = std::min(windowMax, (size_t)userMax);
  return totalBytes >= windowMin && totalBytes <= windowMax;
}


size_t rcclCeRegMaxTab(const rcclArchThresholds* table, ncclFunc_t func) {
  const int64_t param = (func == ncclFuncAllReduce) ? rcclParamCeArRegMaxMsgBytes() : rcclParamCeCollMaxBytes();
  if (param >= 0) return (size_t)param;
  if (table == nullptr) return kThreshUnlimited;
  return (size_t)func < RCCL_DDA_FUNC_COUNT ? table->ceRegMax[(size_t)func] : 0;
}

// Registered-path CE lower bound. Shares the same env vars and table field as
// rcclCeNonRegMinTab so that RCCL_CE_COLL_MIN_BYTES applies symmetrically.
size_t rcclCeRegMinTab(const rcclArchThresholds* table, ncclFunc_t func) {
  const int64_t param = (func == ncclFuncAllReduce) ? rcclParamCeArMinMsgBytes() : rcclParamCeCollMinBytes();
  if (param >= 0) return (size_t)param;
  if (table == nullptr) return 0;
  return (size_t)func < RCCL_DDA_FUNC_COUNT ? table->ceRegMin[(size_t)func] : 0;
}

size_t rcclCeNonRegMaxTab(const rcclArchThresholds* table, ncclFunc_t func) {
  const int64_t param = (func == ncclFuncAllReduce) ? rcclParamCeArMaxMsgBytes() : rcclParamCeCollMaxBytes();
  if (param >= 0) return (size_t)param;
  if (table == nullptr)
    return func == ncclFuncAllReduce ? NCCL_CE_AR_TMPBUF_DEFAULT_BYTES : 0;
  return (size_t)func < RCCL_DDA_FUNC_COUNT ? table->ceNonRegMax[(size_t)func] : 0;
}


size_t rcclCeNonRegMinTab(const rcclArchThresholds* table, ncclFunc_t func) {
  const int64_t param = (func == ncclFuncAllReduce) ? rcclParamCeArMinMsgBytes() : rcclParamCeCollMinBytes();
  if (param >= 0) return (size_t)param;
  if (table == nullptr) return 0;
  return (size_t)func < RCCL_DDA_FUNC_COUNT ? table->ceNonRegMin[(size_t)func] : 0;
}
size_t rcclCeNonRegMax(const struct ncclComm* comm, ncclFunc_t func) {
  return rcclCeNonRegMaxTab(extAlgoArchTable(comm), func);
}
size_t rcclCeAr2ShotMin(const struct ncclComm* comm) {
  return rcclCeNonRegMinTab(extAlgoArchTable(comm), ncclFuncAllReduce);
}
size_t rcclCeArNonRegCap(const ncclComm* comm) {
  const int64_t param = rcclParamCeArMaxMsgBytes();
  if (param >= 0) return (size_t)param;
  // Read comm->archThresholds directly (not extAlgoArchTable) so that
  // RCCL_IGNORE_ARCH_TABLE does not substitute the nullptr-fallback 256 MiB
  // default; the ZERO-policy override applies on the same basis as ncclCeInit.
  const size_t tableMax = (comm->archThresholds != nullptr)
      ? comm->archThresholds->ceNonRegMax[ncclFuncAllReduce] : 0;
  if (tableMax != 0) return tableMax;
  const bool zeroPolicyActive = (comm->config.CTAPolicy & NCCL_CTA_POLICY_ZERO) != 0;
  return zeroPolicyActive ? NCCL_CE_AR_TMPBUF_DEFAULT_BYTES : 0;
}

// Returns true when the message is within the CE-registered (2-shot) AllGather
// window. Does not check symk eligibility -- callers gate on !symEligible explicitly.
// CE-registered gate: recv buffer registered and message within ceRegMax[AG].
// symMaxR2 arbitration is handled upstream via symEligible in the selectors;
// graphMode is accepted but unused here so enqueue.cc can pass ceCapturing
// without a separate overload.
bool rcclAllGatherCeRegisteredWindowTab(const rcclArchThresholds* table, size_t totalBytes,
                                               ncclSymRegType_t winRegType, bool /*graphMode*/) {
  const bool recvReg = (winRegType == ncclSymSendRegRecvReg || winRegType == ncclSymSendNonregRecvReg);
  if (!recvReg) return false;
  const size_t regMax = rcclCeRegMaxTab(table, ncclFuncAllGather);
  const size_t regMin = rcclCeRegMinTab(table, ncclFuncAllGather);
  return rcclWindowFits(totalBytes, regMin, regMax);
}
bool rcclAllGatherCeRegisteredWindow(const ncclComm* comm, size_t totalBytes,
                                            ncclSymRegType_t winRegType, bool graphMode) {
  return rcclAllGatherCeRegisteredWindowTab(extAlgoArchTable(comm), totalBytes, winRegType, graphMode);
}



// True when this arch scales its DDA tier caps by rank count and the comm is big enough.
bool rcclDdaRankScaled(const rcclArchThresholds* table, int nRanks) {
  return table != nullptr && table->rankScaledMinRanks > 0 && nRanks >= table->rankScaledMinRanks;
}

// Table cap for one DDA tier: nRanks * the per-rank entry (clipped to
// rankScaledMaxBytes) on rank-scaled comms when that entry is set, else the total.
size_t ddaTierCapFromTable(const rcclArchThresholds* table, const size_t* totals, const size_t* perRank,
                                  ncclFunc_t func, int nRanks) {
  const size_t share = funcThresholdFromTable(perRank, func);
  if (share != 0 && rcclDdaRankScaled(table, nRanks)) {
    return std::min((size_t)nRanks * share, table->rankScaledMaxBytes);
  }
  return funcThresholdFromTable(totals, func);
}

// ddaTierCapFromTable, which ReduceScatter on comms of at least largeCommMinRanks ranks runs to largeCommMaxBytes
// when that is higher. A 0 cap leaves the tier off.
static size_t ddaTierCapWithRsLargeComm(const rcclArchThresholds* table, const size_t* totals, const size_t* perRank,
                                        ncclFunc_t func, int nRanks, int largeCommMinRanks, size_t largeCommMaxBytes) {
  const size_t cap = ddaTierCapFromTable(table, totals, perRank, func, nRanks);
  const bool largeComm = largeCommMinRanks > 0 && nRanks >= largeCommMinRanks;
  if (func != ncclFuncReduceScatter || !largeComm || cap == 0) return cap;
  return std::max(cap, largeCommMaxBytes);
}

// The table's LL tier cap, with the ReduceScatter large-comm window (ddaRsLLLargeComm*).
static size_t ddaLLTierCapFromTable(const rcclArchThresholds* table, ncclFunc_t func, int nRanks) {
  return ddaTierCapWithRsLargeComm(table, table->ddaLLMax, table->ddaLLMaxPerRank, func, nRanks,
                                   table->ddaRsLLLargeCommMinRanks, table->ddaRsLLLargeCommMaxBytes);
}

size_t rcclDdaLLThresholdTab(const rcclArchThresholds* table, ncclFunc_t func, int nRanks) {
  if (!rcclParamDdaLL()) return 0;
  size_t threshold;
  if (ddaThresholdFromEnv(rcclParamDdaLLThreshold(), &threshold)) return threshold;
  if (table == nullptr) return kDdaLLBaseDefault;
  return ddaLLTierCapFromTable(table, func, nRanks);
}
size_t rcclDdaLLThreshold(const ncclComm* comm, ncclFunc_t func) {
  return rcclDdaLLThresholdTab(extAlgoArchTable(comm), func, comm != nullptr ? comm->nRanks : 0);
}

// RCCL_DDA_LL_TWOSHOT_THRESHOLD wins when the user sets it. On rank-scaled comms
// whose table scales the AllReduce LL tier, two-shot runs to that tier's cap;
// otherwise the env var's default applies.
size_t rcclDdaLLTwoShotThresholdTab(const rcclArchThresholds* table, int nRanks) {
  static const bool envSet = rcclEnvIsSet("RCCL_DDA_LL_TWOSHOT_THRESHOLD");
  if (!envSet && rcclDdaRankScaled(table, nRanks) &&
      funcThresholdFromTable(table->ddaLLMaxPerRank, ncclFuncAllReduce) != 0) {
    return rcclDdaLLThresholdTab(table, ncclFuncAllReduce, nRanks);
  }
  return (size_t)rcclParamDdaLLTwoShotThreshold();
}
size_t rcclDdaLLTwoShotThreshold(const ncclComm* comm) {
  return rcclDdaLLTwoShotThresholdTab(extAlgoArchTable(comm), comm != nullptr ? comm->nRanks : 0);
}

// Registered AllReduce in graph capture stays on fabric DDA up to the table's bound; 0 (or no table) = no re-route.
size_t rcclDdaArRegCaptureMaxBytesTab(const rcclArchThresholds* table, int nRanks) {
  if (table == nullptr) {
    return 0;
  }
  return std::max(table->ddaArRegCaptureMinBytes, (size_t)std::max(nRanks, 0) * table->ddaArRegCapturePerRankBytes);
}
size_t rcclDdaArRegCaptureMaxBytes(const ncclComm* comm) {
  return rcclDdaArRegCaptureMaxBytesTab(extAlgoArchTable(comm), comm != nullptr ? comm->nRanks : 0);
}

// Whole per-rank shards in the table's eager band; a 0 bound (or no table) = no re-route.
bool rcclDdaArRegEagerShardFitsTab(const rcclArchThresholds* table, int nRanks, size_t bytes) {
  if (table == nullptr || nRanks < 1 || bytes % (size_t)nRanks != 0) {
    return false;
  }
  const size_t shard = bytes / (size_t)nRanks;
  return rcclWindowFits(shard, table->ddaArRegEagerMinShardBytes, table->ddaArRegEagerMaxShardBytes);
}
bool rcclDdaArRegEagerShardFits(const ncclComm* comm, size_t bytes) {
  return rcclDdaArRegEagerShardFitsTab(extAlgoArchTable(comm), comm != nullptr ? comm->nRanks : 0, bytes);
}


// The table's LL128 tier cap, with the ReduceScatter large-comm window (ddaRsLL128LargeComm*).
static size_t ddaLL128TierCapFromTable(const rcclArchThresholds* table, ncclFunc_t func, int nRanks) {
  return ddaTierCapWithRsLargeComm(table, table->ddaLL128Max, table->ddaLL128MaxPerRank, func, nRanks,
                                   table->ddaRsLL128LargeCommMinRanks, table->ddaRsLL128LargeCommMaxBytes);
}

// Returns the LL128 size cap for this collective, or 0 if the tier is disabled.
// RCCL_DDA_LL128 three-way logic:
//   kDdaThresholdUnset (-1, default): auto — enabled iff arch table has non-zero
//     ddaLL128Max for func; RCCL_IGNORE_ARCH_TABLE=1 (table=nullptr) → 0 (off).
//   0: explicitly disabled for all collectives.
//   >=1: forced on; uses table value when available, else kDdaLL128BaseDefault.
// Callers check "> 0" to decide enablement, so no separate Active() helper needed.
size_t rcclDdaLL128ThresholdTab(const rcclArchThresholds* table, ncclFunc_t func, int nRanks) {
  size_t threshold;
  if (ddaThresholdFromEnv(rcclParamDdaLL128Threshold(), &threshold)) return threshold;
  const int64_t param = rcclParamDdaLL128();
  if (param == 0) return 0;
  if (param == kDdaThresholdUnset) {
    if (table == nullptr) return 0;
    return ddaLL128TierCapFromTable(table, func, nRanks);
  }
  // param > 0: user forced on
  if (table == nullptr) return kDdaLL128BaseDefault;
  return ddaLL128TierCapFromTable(table, func, nRanks);
}
size_t rcclDdaLL128Threshold(const ncclComm* comm, ncclFunc_t func) {
  return rcclDdaLL128ThresholdTab(extAlgoArchTable(comm), func, comm != nullptr ? comm->nRanks : 0);
}

// RCCL_DDA_LL128_TWOSHOT_THRESHOLD wins when the user sets it. When the table
// runs an AllReduce LL128 tier, two-shot runs to that tier's cap (as LL two-shot
// does to the LL tier's); otherwise the env var's default applies.
size_t rcclDdaLL128TwoShotThresholdTab(const rcclArchThresholds* table, int nRanks) {
  static const bool envSet = rcclEnvIsSet("RCCL_DDA_LL128_TWOSHOT_THRESHOLD");
  if (!envSet && table != nullptr && funcThresholdFromTable(table->ddaLL128Max, ncclFuncAllReduce) != 0) {
    return rcclDdaLL128ThresholdTab(table, ncclFuncAllReduce, nRanks);
  }
  return (size_t)rcclParamDdaLL128TwoShotThreshold();
}
size_t rcclDdaLL128TwoShotThreshold(const ncclComm* comm) {
  return rcclDdaLL128TwoShotThresholdTab(extAlgoArchTable(comm), comm != nullptr ? comm->nRanks : 0);
}

// One-shot moves (nRanks - 1) x the message per rank against two-shot's ~2x, so with
// the peer-parallel two-shot its crossover falls as nRanks grows; the arch table
// holds it (ddaArLL128OneShotRankBytes).
size_t rcclDdaLL128OneShotThreshold(const ncclComm* comm) {
  static const bool envSet = rcclEnvIsSet("RCCL_DDA_LL128_ONESHOT_THRESHOLD");
  const size_t cap = (size_t)rcclParamDdaLL128OneShotThreshold();
  if (envSet || rcclParamDdaLL128ArPeerParallel() == 0 || comm == nullptr || comm->nRanks < 2) {
    return cap;
  }
  const rcclArchThresholds* table = extAlgoArchTable(comm);
  if (table == nullptr || table->ddaArLL128OneShotRankBytes == 0) {
    return cap;
  }
  return std::min(cap, table->ddaArLL128OneShotRankBytes / (size_t)comm->nRanks);
}

size_t rcclDdaVmmThresholdTab(const rcclArchThresholds* table, ncclFunc_t func, int nRanks) {
  size_t threshold;
  if (ddaThresholdFromEnv(rcclParamDdaThreshold(), &threshold)) return threshold;
  if (table == nullptr) return kDdaVmmBaseDefault;
  return ddaTierCapFromTable(table, table->ddaVmmMax, table->ddaVmmMaxPerRank, func, nRanks);
}
size_t rcclDdaVmmThreshold(const ncclComm* comm, ncclFunc_t func) {
  return rcclDdaVmmThresholdTab(extAlgoArchTable(comm), func, comm != nullptr ? comm->nRanks : 0);
}

// AllReduce DDA VMM ahead of the LL tier (see rcclArchThresholds::ddaArVmmFirst*).
// Measured on unregistered buffers only; registered ones keep the tier order and
// the R2 cap. Env thresholds turn it off so RCCL_DDA_*_THRESHOLD keep their meaning.
bool rcclDdaArVmmFirstTab(const rcclArchThresholds* table, int nRanks, size_t msgBytes,
                                 ncclSymRegType_t winRegType) {
  if (table == nullptr || table->ddaArVmmFirstMinRanks <= 0 || nRanks < table->ddaArVmmFirstMinRanks) return false;
  if (winRegType != ncclSymSendNonregRecvNonreg) return false;
  size_t unused;
  if (ddaThresholdFromEnv(rcclParamDdaThreshold(), &unused) || ddaThresholdFromEnv(rcclParamDdaLLThreshold(), &unused))
    return false;
  return msgBytes > table->ddaArVmmFirstMinBytes && msgBytes <= (size_t)nRanks * table->ddaArVmmFirstMaxPerRank;
}


// Widest window any enabled DDA tier can serve for this collective. The entry
// gate only decides whether the per-tier checks run, so a tier tuned off (0)
// must not hide the tiers that are still on: gfx1250 ReduceScatter has
// ddaVmmMax = 0 with non-zero LL/LL128 caps, and ddaLL128Max[AllReduce]
// (256 MiB) sits above ddaVmmMax[AllReduce] (16 MiB). RCCL_DDA_LL /
// RCCL_DDA_LL128 = 0 drop their tier so disabling one cannot widen the gate.
// Each tier still applies its own cap at the call site, including the VMM/IPC
// branch, which checks rcclDdaVmmThreshold() there.
// ddaVmmMaxGraph is also folded in: it is 0 for all collectives except AR
// (256 MiB on gfx1250), so it only widens the gate for graph-mode AR calls
// and is a no-op for all other collectives.
size_t rcclDdaEntryThresholdTab(const rcclArchThresholds* table, ncclFunc_t func, int nRanks) {
  size_t cap = rcclDdaVmmThresholdTab(table, func, nRanks);
  cap = std::max(cap, rcclDdaLLThresholdTab(table, func, nRanks));
  cap = std::max(cap, rcclDdaLL128ThresholdTab(table, func, nRanks));
  if (table != nullptr)    cap = std::max(cap, funcThresholdFromTable(table->ddaVmmMaxGraph, func));
  return cap;
}
size_t rcclDdaEntryThreshold(const ncclComm* comm, ncclFunc_t func) {
  return rcclDdaEntryThresholdTab(extAlgoArchTable(comm), func, comm != nullptr ? comm->nRanks : 0);
}

size_t rcclDdaScratchPayloadCap(const ncclComm* comm) {
  size_t cap = 0;
  auto bump = [&](size_t v) {
    if (v > cap) cap = v;
  };
  size_t env = 0;
  if (ddaThresholdFromEnv(rcclParamDdaThreshold(), &env)) bump(env);
  if (ddaThresholdFromEnv(rcclParamDdaLLThreshold(), &env)) bump(env);
  if (ddaThresholdFromEnv(rcclParamDdaLL128Threshold(), &env)) bump(env);

  const rcclArchThresholds* table = extAlgoArchTable(comm);
  if (table == nullptr) {
    // Same pre-table defaults rcclDda{LL,LL128,Vmm}Threshold return when the
    // table is ignored, so fabric scratch covers any DDA path the selector
    // can still pick. Env, when set (including 0 to disable a tier), already
    // won above and is not replaced.
    size_t unused;
    if (!ddaThresholdFromEnv(rcclParamDdaLLThreshold(), &unused)) bump(kDdaLLBaseDefault);
    if (!ddaThresholdFromEnv(rcclParamDdaLL128Threshold(), &unused)) bump(kDdaLL128BaseDefault);
    if (!ddaThresholdFromEnv(rcclParamDdaThreshold(), &unused)) bump(kDdaVmmBaseDefault);
    return cap;
  }

  // All table entries are total message bytes; the selector compares totalBytes
  // directly (the ReduceScatter rule row). Rank-scaled caps are clipped to
// rankScaledMaxBytes, which the table keeps within these totals.
  for (int i = 0; i < RCCL_DDA_FUNC_COUNT; ++i) {
    bump(table->ddaLLMax[i]);
    bump(table->ddaLL128Max[i]);
    bump(table->ddaVmmMax[i]);
    bump(table->ddaVmmMaxR2[i]);
    // Graph VMM is included even for eager-only processes: one buffer must
    // fit a later capture (gfx1250 AR graph cap is 256 MiB).
    bump(table->ddaVmmMaxGraph[i]);
    // AG CE-Scratch (and any other non-AR CE-scratch) copies the receive into
    // ddaScratch, so that window must fit. AR 2-shot uses ceARTmpBuf.
    if (i != (int)ncclFuncAllReduce) bump(table->ceNonRegMax[i]);
  }
  return cap;
}

// Context-aware VMM threshold resolver.  Callers pass the full winRegType so
// the function can apply the correct override without the caller having to
// interpret registration semantics.  Policy: only recv registration shifts the
// DDA VMM cap (send-only registration does not change DDA/CE dispatch).
// When the recv buffer is registered (ncclSymSendNonregRecvReg or
// ncclSymSendRegRecvReg) and the arch table has a non-zero R2 override for
// this collective, that cap wins over the default ddaVmmMax.
// When inside a graph capture (graphMode=true) and the table has a non-zero
// graph-mode override, that cap wins.  Graph-mode is checked first.
// Env var (RCCL_DDA_THRESHOLD) always wins over all context variants.
size_t rcclDdaVmmThresholdCtxTab(const rcclArchThresholds* table, ncclFunc_t func,
                                        ncclSymRegType_t winRegType, bool graphMode, int nRanks) {
  size_t threshold;
  if (ddaThresholdFromEnv(rcclParamDdaThreshold(), &threshold)) return threshold;
  if (table == nullptr) return kDdaVmmBaseDefault;
  if (graphMode) {
    size_t graphCap = funcThresholdFromTable(table->ddaVmmMaxGraph, func);
    if (graphCap != 0) return graphCap;
  }
  const bool recvReg = (winRegType == ncclSymSendNonregRecvReg ||
                         winRegType == ncclSymSendRegRecvReg);
  if (recvReg) {
    size_t r2Cap = funcThresholdFromTable(table->ddaVmmMaxR2, func);
    if (r2Cap != 0) return r2Cap;
  }
  return ddaTierCapFromTable(table, table->ddaVmmMax, table->ddaVmmMaxPerRank, func, nRanks);
}
