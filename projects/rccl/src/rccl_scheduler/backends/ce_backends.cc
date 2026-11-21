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

// CE: copy-engine collectives (registered windows, staged 2-shot, DDA scratch, hierarchical).

#include "ce_coll.h"
#include "archinfo.h"
#include "debug.h"
#include "dev_runtime.h"
#include "rccl_scheduler/rccl_scheduler.h"

// CE AllReduce is only tuned on gfx1250, so it is default-on there and stays off
// everywhere else. Without this gate the arch tables for gfx942/gfx950 (which set
// ceNonRegMax[AR] = 256 MiB) would let unregistered 2-shot, and -- via force --
// registered CE, service AllReduce on arches that were never measured for it.
static bool rcclCeCollArchDefault(const ncclComm* comm) {
  if (comm == nullptr || !IsArchMatch(comm->archName, "gfx1250")) return false;
  return extAlgoArchTable(comm) != nullptr;
}
static bool rcclCeAllReduceEnabledDef(bool archDefault) {
  const int64_t param = rcclParamCeAllReduce();
  if (param >= 0) return param != 0;
  return archDefault;
}
bool rcclCeAllReduceEnabled(const ncclComm* comm) {
  return rcclCeAllReduceEnabledDef(rcclCeCollArchDefault(comm));
}

static bool rcclForceCeAllReduceEnabledDef(bool archDefault) {
  const int64_t param = rcclParamForceCeAllReduce();
  if (param >= 0) return param != 0;
  return archDefault;
}
bool rcclForceCeAllReduceEnabled(const ncclComm* comm) {
  return rcclForceCeAllReduceEnabledDef(rcclCeCollArchDefault(comm));
}


// Whether CE-Scratch is forced for unregistered AllGather / AlltoAll / Gather /
// Scatter. A user-set RCCL_FORCE_CE decides (0 = never, otherwise always). Left
// at its default, the comm-level NCCL_CTA_POLICY_ZERO forces it, and otherwise
// the arch table's forceCeScratch decides.
bool rcclForceCeScratchEnabled(const ncclComm* comm) {
  static const bool envSet = rcclEnvIsSet("RCCL_FORCE_CE");
  const int64_t param = rcclParamForceCe();
  if (param == 0) return false;
  if (envSet) return true;
  if (comm->config.CTAPolicy & NCCL_CTA_POLICY_ZERO) return true;
  const rcclArchThresholds* table = extAlgoArchTable(comm);
  return table == nullptr || table->forceCeScratch;
}

// Symmetric windows have equal offsets on every rank and a min-size bound, so all ranks agree.
void rcclSetCeAllReduceFastPath(struct ncclComm* comm) {
  for (struct ncclTaskColl* task = ncclIntruQueueHead(&comm->planner.collCeTaskQueue); task != nullptr;
       task = task->next) {
    const size_t totalBytes = task->count * ncclTypeSize(task->datatype);
    task->ceAllReduceFastPath = task->func == ncclFuncAllReduce && task->recvWin != nullptr &&
                                (task->recvWin->winFlags & NCCL_WIN_COLL_SYMMETRIC) &&
                                ncclCeRecvRangeContainedInWindow(task->recvWin, task->recvbuff, totalBytes) != 0;
  }
}

// Elements per CE-registered reduce launch: ncclCeAllReduce's chunk for the staging size ncclCeInit resolves.
static size_t rcclCeArRegChunkElems(const ncclComm* comm, size_t count, ncclDataType_t datatype) {
  const size_t initBytes = comm->ceColl.ceArStagingBytes;
  const size_t stagingBytes = initBytes != 0 ? initBytes : ncclCeArStagingBytes(rcclParamCeArStagingBytes());
  const size_t eltSize = ncclTypeSize(datatype);
  const size_t shardBytes = count / (size_t)comm->nRanks * eltSize;
  const size_t slotChunkBytes = ncclCeAllReduceSlotChunkBytes(stagingBytes / (size_t)comm->nRanks);
  return ncclCeAllReduceChunkBytes(shardBytes, slotChunkBytes) / eltSize;
}



bool rcclUseCeAr2Shot(struct ncclComm* comm, size_t count, ncclDataType_t datatype, ncclRedOp_t op,
                        const void* acc) {
  const bool enabled = rcclCeAllReduceEnabled(comm);
  const bool force = rcclForceCeAllReduceEnabled(comm);
  if (!enabled) {
    // Log once per process, not on every eligibility check (called per AllReduce).
    static bool warnedDisabled = false;
    if (!warnedDisabled) {
      warnedDisabled = true;
      INFO(NCCL_INIT, "CE AllReduce not enabled on %s. Set RCCL_CE_ALLREDUCE=1 to enable.", comm->archName);
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

  // Staged CE still addresses every peer by LSA rank, so it needs the same
  // runtime and topology as registered CE; only the window checks differ.
  if (!ncclCeImplemented(ncclFuncAllReduce, ncclDevSum, datatype)) {
    INFO(NCCL_TUNING, "Skipping CE AllReduce: CE is not supported by this driver");
    return false;
  }
  if (!ncclDevrIsOneLsaTeam(comm)) {
    INFO(NCCL_TUNING, "Skipping CE AllReduce: LSA team does not span the comm");
    return false;
  }

  return true;
}

bool rcclUseCeReduceScatter(struct ncclComm* comm, size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op) {
  // Re-read every call. A function-local static latches the first value and
  // ignores the host-test param seam.
  const int enabled = rcclParamCeReduceScatter();
  if (!enabled) {
    static bool warnedDisabled = false;
    if (!warnedDisabled) {
      warnedDisabled = true;
      INFO(NCCL_INIT, "CE ReduceScatter not enabled. Set RCCL_CE_REDUCESCATTER=1 to enable.");
    }
    return false;
  }

  if (!comm->symmetricSupport) {
    WARN("Skipping CE ReduceScatter: symmetric support is not enabled");
    return false;
  }
  if (comm->nNodes != 1) {
    WARN("Skipping CE ReduceScatter: nNodes is not 1");
    return false;
  }
  if (recvcount == 0) {
    WARN("Skipping CE ReduceScatter: recvcount is 0");
    return false;
  }

  // No message-size cap: CE ReduceScatter pipelines through ceARTmpBuf slots, so
  // large messages are chunked rather than rejected (unlike CE AllReduce 2-shot).

  if (comm->config.CTAPolicy != NCCL_CTA_POLICY_ZERO) {
    WARN("Skipping CE ReduceScatter: CTA policy is not ZERO");
    return false;
  }
  if (op != ncclSum && op != ncclProd && op != ncclMin && op != ncclMax) {
    WARN("Skipping CE ReduceScatter: unsupported reduction operation");
    return false;
  }
  if (datatype == ncclFloat8e4m3 || datatype == ncclFloat8e5m2) {
    WARN("Skipping CE ReduceScatter: unsupported datatype: Float8");
    return false;
  }

  // Staged CE addresses every peer by world rank via ncclDevrGetLsaRankPtr, so
  // it needs the same runtime and single-LSA topology as CE AllReduce. Without
  // this, NCCL_LSA_TEAM_SIZE < nRanks maps world ranks into the wrong LSA
  // slots (or returns ncclInvalidArgument) after the reducer has already
  // launched and is spinning on signals nobody writes.
  if (!ncclCeImplemented(ncclFuncReduceScatter, ncclDevSum, datatype)) {
    INFO(NCCL_TUNING, "Skipping CE ReduceScatter: CE is not supported by this driver");
    return false;
  }
  if (!ncclDevrIsOneLsaTeam(comm)) {
    INFO(NCCL_TUNING, "Skipping CE ReduceScatter: LSA team does not span the comm");
    return false;
  }

  return true;
}

void rcclCeAllReduceGraphLatchTick(struct ncclComm* comm, bool ceCapturing) {
  if (ceCapturing) {
    if (!comm->ceColl.graphModeSeen) {
      INFO(NCCL_TUNING, "Disabling CE AllReduce; graph latch set (rank %d): capture detected", comm->rank);
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
    INFO(NCCL_TUNING, "Re-enabling CE AllReduce; graph latch cleared (rank %d): no live captured plans", comm->rank);
    comm->ceColl.graphModeSeen = false;
  }
}
bool rcclCeArGraphSafe(struct ncclComm* comm) {
  return !comm->ceColl.graphModeSeen;
}

// The query cannot see the stream, so a declared capture reads as a set latch.
bool rcclCeArGraphOk(rcclSchedCtx& ctx) {
  return rcclCeArGraphSafe(ctx.comm) && !(ctx.query && ctx.graphCapturingHint);
}

bool rcclCeArRegisteredAvailable(rcclSchedCtx& ctx) {
  struct ncclComm* comm = ctx.comm;
  const ncclRedOp_t op = ctx.op;
  return rcclCeArGraphOk(ctx) && !ctx.capturing &&
         ncclCeAvailable(comm, ncclFuncAllReduce, (int)op, ctx.datatype, ctx.winRegType, ctx.sendWin, ctx.recvWin) &&
         (op == ncclSum || op == ncclProd || op == ncclMin || op == ncclMax) && ctx.count % (size_t)comm->nRanks == 0 &&
         rcclCeAllReduceEnabledDef(rcclCeCollArchDefault(comm));
}

bool rcclCeArRegisteredInWindow(rcclSchedCtx& ctx) {
  return rcclWindowFits(ctx.totalBytes, rcclCeRegMinTab(ctx.archTable, ncclFuncAllReduce),
                        rcclCeRegMaxTab(ctx.archTable, ncclFuncAllReduce));
}

int rcclCeArRegisteredBlocks(rcclSchedCtx& ctx) {
  return ncclCeLocalReduceBlocks(ctx.datatype, rcclCeArRegChunkElems(ctx.comm, ctx.count, ctx.datatype));
}

namespace {

int shardBlocks(rcclSchedCtx& c) {
  const size_t elems = c.func == ncclFuncAllReduce ? c.count / (size_t)c.comm->nRanks : c.count;
  return ncclCeLocalReduceBlocks(c.datatype, elems);
}

ncclResult_t chooseAlgo(rcclCollDecision* d, int algo, bool* ok) {
  d->algo = algo;
  *ok = true;
  return ncclSuccess;
}

bool tmpBufReady(rcclSchedCtx& c) { return c.comm->ceColl.ceARTmpBuf != NULL; }
bool tmpBufMissing(rcclSchedCtx& c) { return c.comm->ceColl.ceARTmpBuf == NULL; }

// AllReduce 2-shot also stays inside the unregistered CE window.
bool twoShotEligible(rcclSchedCtx& c) {
  if (c.func == ncclFuncAllReduce && !rcclLimitIn(c, RCCL_LIMIT_CE_NONREG)) return false;
  return tmpBufReady(c);
}

ncclResult_t twoShotChoose(rcclSchedCtx& c, rcclCollDecision* d, bool* ok) {
  d->nMaxChannels = shardBlocks(c);
  return chooseAlgo(d, RCCL_CE_2SHOT, ok);
}

// First CE ReduceScatter: enqueue runs CE init, which allocates the staging buffer.
ncclResult_t stagingInitChoose(rcclSchedCtx& c, rcclCollDecision* d, bool* ok) {
  d->nMaxChannels = shardBlocks(c);
  return chooseAlgo(d, RCCL_CE_REGISTERED, ok);
}

bool registeredEligible(rcclSchedCtx& c) {
  struct ncclComm* comm = c.comm;
  switch (c.func) {
  case ncclFuncAllReduce: {
    const size_t regMax = rcclCeRegMaxTab(c.archTable, ncclFuncAllReduce);
    const bool envCapOk = rcclParamCeArRegMaxMsgBytes() < 0 || rcclCeArRegisteredInWindow(c);
    return rcclCeArRegisteredAvailable(c) && regMax != 0 && envCapOk;
  }
  case ncclFuncAllGather:
    return ncclCeAvailable(comm, ncclFuncAllGather, (int)ncclSum, c.datatype, c.winRegType, c.sendWin, c.recvWin);
  case ncclFuncAlltoAll:
    return ncclCeAvailable(comm, ncclFuncAlltoAll, ncclDevSum, c.datatype, c.winRegType, c.sendWin, c.recvWin);
  case ncclFuncReduceScatter: return true;  // RCCL_FACT_CE_RS_OK holds the gates
  default: return false;
  }
}

ncclResult_t registeredChoose(rcclSchedCtx& c, rcclCollDecision* d, bool* ok) {
  if (c.func == ncclFuncAllReduce || c.func == ncclFuncReduceScatter) d->nMaxChannels = shardBlocks(c);
  return chooseAlgo(d, RCCL_CE_REGISTERED, ok);
}

// AllReduce on unregistered buffers through the staging buffer before it exists.
bool stagedEligible(rcclSchedCtx& c) {
  const size_t twoShotMin = rcclCeNonRegMinTab(c.archTable, ncclFuncAllReduce);
  const bool zeroWindow = (c.comm->config.CTAPolicy & NCCL_CTA_POLICY_ZERO) && rcclParamForceCeAllReduce() != 0 &&
                          rcclWindowFits(c.totalBytes, twoShotMin, rcclCeArNonRegCap(c.comm));
  const bool forceWindow =
    rcclForceCeAllReduceEnabledDef(rcclCeCollArchDefault(c.comm)) &&
    rcclWindowFits(c.totalBytes, twoShotMin, rcclCeNonRegMaxTab(c.archTable, ncclFuncAllReduce));
  return zeroWindow || forceWindow;
}

bool ceHierEligible(rcclSchedCtx& c) {
  const int red = c.func == ncclFuncAllGather ? (int)ncclSum : (int)ncclDevSum;
  return ncclHierCeAvailable(c.comm, c.func, red, c.datatype, c.winRegType, c.sendWin, c.recvWin);
}

ncclResult_t registeredAlgo(rcclSchedCtx&, rcclCollDecision* d, bool* ok) {
  return chooseAlgo(d, RCCL_CE_REGISTERED, ok);
}

// Unregistered receive staged through comm->ddaScratch.
bool scratchEligible(rcclSchedCtx& c) {
  if (!rcclLimitIn(c, RCCL_LIMIT_DDA_SCRATCH)) return false;
  struct ncclComm* comm = c.comm;
  if (!rcclForceCeScratchEnabled(comm) || comm->ddaScratch == nullptr) return false;
  if (c.func == ncclFuncAllGather) {
    return ncclCeScratchAvailable(comm, ncclFuncAllGather, (int)ncclSum, c.datatype, c.winRegType) &&
           c.winRegType == ncclSymSendNonregRecvNonreg;
  }
  return ncclCeScratchAvailable(comm, c.func, ncclDevSum, c.datatype, c.winRegType);
}

ncclResult_t scratchChoose(rcclSchedCtx&, rcclCollDecision* d, bool* ok) {
  return chooseAlgo(d, RCCL_CE_SCRATCH, ok);
}

}  // namespace

const rcclBackend kCe2Shot = {"ce.2shot", RCCL_BACKEND_CE, twoShotEligible, twoShotChoose, nullptr, false};
const rcclBackend kCeStagingInit = {"ce.staging_init", RCCL_BACKEND_CE, tmpBufMissing, stagingInitChoose, nullptr, false};
const rcclBackend kCeRegistered = {"ce.registered", RCCL_BACKEND_CE, registeredEligible, registeredChoose, nullptr, true};
const rcclBackend kCeStaged = {"ce.staged", RCCL_BACKEND_CE, stagedEligible, registeredChoose, nullptr, false};
const rcclBackend kCeHier = {"ce.hier", RCCL_BACKEND_CE, ceHierEligible, registeredAlgo, nullptr, true};
const rcclBackend kCeScratch = {"ce.scratch", RCCL_BACKEND_CE, scratchEligible, scratchChoose, nullptr, true};
