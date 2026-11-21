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

// DDA: direct peer access over the fabric (gfx1250 LL, LL128, VMM) or IPC.

#include <algorithm>
#include <cstdint>

#include "algorithms/dda/all_gather/dda_all_gather.h"
#include "algorithms/dda/all_reduce/dda_all_reduce.h"
#include "algorithms/dda/alltoall/dda_alltoall.h"
#include "algorithms/dda/broadcast/dda_broadcast.h"
#include "algorithms/dda/reduce_scatter/dda_reduce_scatter.h"
#include "archinfo.h"
#include "debug.h"
#include "group.h"
#include "rccl_scheduler/rccl_scheduler.h"

int rcclDdaFabricVmmBlocks(const ncclComm* comm) {
  if (comm == nullptr) return 0;
  const int cap = comm->ddaFabricMaxBlocks;
  const rcclArchThresholds* table = extAlgoArchTable(comm);
  if (!rcclDdaRankScaled(table, comm->nRanks) || table->ddaVmmBlockBudget <= 0) return cap;
  const int want =
    std::min(table->ddaVmmMaxBlocks, std::max(table->ddaVmmMinBlocks, table->ddaVmmBlockBudget / comm->nRanks));
  return std::min(cap, want);
}

int rcclDdaFabricAgLL128Threads(const ncclComm* comm, size_t totalBytes) {
  const rcclArchThresholds* table = extAlgoArchTable(comm);
  if (table == nullptr ||
      !rcclWindowFits(totalBytes, table->ddaAgLL128ThreadsMinBytes, table->ddaAgLL128ThreadsMaxBytes)) {
    return 0;
  }
  return table->ddaAgLL128Threads;
}

int rcclDdaFabricAgLL128Blocks(const ncclComm* comm, size_t totalBytes) {
  if (comm == nullptr) return 0;
  const int cap = comm->ddaFabricMaxBlocks;
  const rcclArchThresholds* table = extAlgoArchTable(comm);
  if (table == nullptr || table->ddaAgLL128MaxBlocks <= 0 ||
      !rcclWindowFits(totalBytes, table->ddaAgLL128MaxBlocksMinBytes, table->ddaAgLL128MaxBlocksMaxBytes)) {
    return cap;
  }
  return std::min(cap, table->ddaAgLL128MaxBlocks);
}

int rcclDdaFabricA2aLL128Threads(const ncclComm* comm, size_t totalBytes) {
  const rcclArchThresholds* table = extAlgoArchTable(comm);
  if (table == nullptr ||
      !rcclWindowFits(totalBytes, table->ddaA2aLL128ThreadsMinBytes, table->ddaA2aLL128ThreadsMaxBytes)) {
    return 0;
  }
  return table->ddaA2aLL128Threads;
}

int rcclDdaFabricA2aLL128Blocks(const ncclComm* comm, size_t totalBytes) {
  if (comm == nullptr) return 0;
  const int cap = comm->ddaFabricMaxBlocks;
  const rcclArchThresholds* table = extAlgoArchTable(comm);
  if (table == nullptr || table->ddaA2aLL128MaxBlocks <= 0 ||
      !rcclWindowFits(totalBytes, table->ddaA2aLL128MaxBlocksMinBytes, table->ddaA2aLL128MaxBlocksMaxBytes)) {
    return cap;
  }
  return std::min(cap, table->ddaA2aLL128MaxBlocks);
}

int rcclDdaFabricBcLL128Threads(const ncclComm* comm) {
  const rcclArchThresholds* table = extAlgoArchTable(comm);
  if (table == nullptr) return 0;
  return (int)table->ddaLL128BcTwoShotThreads;
}
int rcclDdaFabricRsLL128Threads(const ncclComm* comm) {
  const rcclArchThresholds* table = extAlgoArchTable(comm);
  return table != nullptr ? table->ddaRsLL128Threads : 0;
}

size_t rcclDdaFabricRsLL128PeerMaxBytes(const ncclComm* comm) {
  const rcclArchThresholds* table = extAlgoArchTable(comm);
  if (table == nullptr) {
    return 0;
  }
  const size_t nRanks = (size_t)std::max(comm->nRanks, 0);
  const size_t shard = std::max(table->ddaRsLL128PeerMaxShard, nRanks * table->ddaRsLL128PeerMaxShardPerRank);
  return std::min(nRanks * shard, table->ddaRsLL128PeerMaxBytes);
}


bool rcclDdaEnabled(const ncclComm* comm, size_t totalBytes, size_t threshold,
                    bool query, const char* prefix) {
  // The environment parameter can be NCCL_CONFIG_UNDEF_INT when launch order
  // is configured per communicator. Use the resolved communicator value:
  // testing the raw sentinel as a boolean disables DDA by default, while
  // testing only the environment would ignore an explicit config value.
  if (!rcclParamDdaEnable() || comm->config.launchOrderImplicit == 1 || ncclGroupDepth != 0) {
    if (!query && prefix) INFO(NCCL_TUNING, "%s DDA disqualified: RCCL_DDA_ENABLE=%d launchOrderImplicit=%d ncclGroupDepth=%d", prefix, (int)rcclParamDdaEnable(), comm->config.launchOrderImplicit, ncclGroupDepth);
    return false;
  }
  if (comm->ddaDisabledAfterAbort) {
    if (!query && prefix) INFO(NCCL_TUNING, "%s DDA disqualified: comm was shrunk with NCCL_SHRINK_ABORT", prefix);
    return false;
  }
  if (IsArchMatch(comm->archName, "gfx1250")) {
    // gfx1250 has no nRanks floor.
  } else if (IsArchMatch(comm->archName, "gfx942") || IsArchMatch(comm->archName, "gfx950")) {
    if (comm->nRanks < 8) return false;
  } else {
    return false;
  }
  const bool ok = rcclWindowFits(totalBytes, 0, threshold);
  if (!ok && !query && prefix)
    INFO(NCCL_TUNING, "%s DDA disqualified: totalBytes=%zu > entryThreshold=%zu",
         prefix, totalBytes, threshold);
  return ok;
}

bool rcclDdaDatatypeSupported(const ncclComm* comm, ncclDataType_t datatype) {
  switch (datatype) {
  case ncclFloat32:
  case ncclFloat16:
  case ncclBfloat16:
    return true;
  case ncclFloat8e4m3:
  case ncclFloat8e5m2:
    return comm != nullptr && IsArchMatch(comm->archName, "gfx1250");
  default:
    return false;
  }
}

namespace {

// RCCL_DDA_BC when set, else whether the table has a Broadcast LL or LL128 tier.
bool ddaBcEnabled(const rcclArchThresholds* table, int nRanks) {
  const int64_t param = rcclParamDdaBc();
  if (param != kDdaThresholdUnset) return param > 0;
  return table != nullptr &&
         (ddaTierCapFromTable(table, table->ddaLLMax, table->ddaLLMaxPerRank, ncclFuncBroadcast, nRanks) != 0 ||
          ddaTierCapFromTable(table, table->ddaLL128Max, table->ddaLL128MaxPerRank, ncclFuncBroadcast, nRanks) != 0);
}

// DDA gate. Tier windows come from the rules, so no byte threshold here.
bool ddaOn(rcclSchedCtx& ctx) { return rcclDdaEnabled(ctx.comm, ctx.totalBytes, SIZE_MAX, ctx.query, ctx.name); }

// Calls above the widest enabled tier never take DDA. AlltoAll fabric LL and
// LL128 are bounded by their own windows only.
bool underEntryCap(rcclSchedCtx& ctx, bool fabricLLTier) {
  if (fabricLLTier && ctx.func == ncclFuncAlltoAll) return true;
  return rcclLimitIn(ctx, RCCL_LIMIT_DDA_ENTRY);
}

bool bcOn(rcclSchedCtx& ctx, bool fabricOnly) {
  struct ncclComm* comm = ctx.comm;
  // DDA skips ArgsCheck; let the enqueue path report a bad root.
  if (!ctx.query && (ctx.root < 0 || ctx.root >= comm->nRanks)) return false;
  if (!ddaBcEnabled(ctx.archTable, comm->nRanks) || !ddaOn(ctx)) return false;
  if (comm->ddaFabricMemHandler != nullptr) return true;
  return !fabricOnly && rcclParamDdaBcAllowIpc() && comm->ddaIpcMemHandler != nullptr;
}

bool fabricLLEligible(rcclSchedCtx& c) {
  if (!underEntryCap(c, true)) return false;
  switch (c.func) {
  case ncclFuncBroadcast: return bcOn(c, false) && ncclBroadcastDdaFabricLLEligible(c.comm, c.count, c.datatype);
  case ncclFuncAlltoAll:
    return ddaOn(c) && ncclAllToAllDdaFabricLLEligible(c.comm, c.sendbuff, c.recvbuff, c.count, c.datatype);
  case ncclFuncAllGather:
    return ddaOn(c) && ncclAllGatherDdaFabricLLEligible(c.comm, c.sendbuff, c.recvbuff, c.count, c.datatype);
  case ncclFuncReduceScatter:
    return ddaOn(c) && ncclReduceScatterDdaFabricLLEligible(c.comm, c.sendbuff, c.recvbuff, c.count, c.datatype, c.op);
  case ncclFuncAllReduce:
    return ddaOn(c) && ncclAllReduceDdaFabricLLEligible(c.comm, c.sendbuff, c.recvbuff, c.count, c.datatype, c.op);
  default: return false;
  }
}

ncclResult_t fabricLLChoose(rcclSchedCtx& c, rcclCollDecision* d, bool* ok) {
  d->algo = RCCL_DDA_FABRIC_LL;
  d->protocol = NCCL_PROTO_LL;
  switch (c.func) {
  case ncclFuncBroadcast: d->nMaxChannels = ncclBroadcastDdaFabricLLBlocks(c.comm, c.count, c.datatype); break;
  case ncclFuncAlltoAll: d->nMaxChannels = ncclAllToAllDdaFabricLLBlocks(c.comm, c.count, c.datatype); break;
  case ncclFuncAllGather: d->nMaxChannels = ncclAllGatherDdaFabricLLBlocks(c.comm, c.count, c.datatype); break;
  case ncclFuncReduceScatter: d->nMaxChannels = ncclReduceScatterDdaFabricLLBlocks(c.comm, c.count, c.datatype); break;
  case ncclFuncAllReduce:
    d->ddaCap = c.ddaCap;
    d->nMaxChannels = ncclAllReduceDdaFabricLLBlocks(c.comm, c.count, c.datatype, c.ddaCap);
    break;
  default: break;
  }
  *ok = true;
  return ncclSuccess;
}

bool fabricLL128Eligible(rcclSchedCtx& c) {
  if (!underEntryCap(c, true)) return false;
  switch (c.func) {
  case ncclFuncBroadcast: return bcOn(c, true) && ncclBroadcastDdaFabricLL128Eligible(c.comm, c.count, c.datatype);
  case ncclFuncAlltoAll:
    return ddaOn(c) && ncclAllToAllDdaFabricLL128Eligible(c.comm, c.sendbuff, c.recvbuff, c.count, c.datatype);
  case ncclFuncAllGather:
    return ddaOn(c) && ncclAllGatherDdaFabricLL128Eligible(c.comm, c.sendbuff, c.recvbuff, c.count, c.datatype);
  case ncclFuncReduceScatter:
    return ddaOn(c) &&
           ncclReduceScatterDdaFabricLL128Eligible(c.comm, c.sendbuff, c.recvbuff, c.count, c.datatype, c.op);
  case ncclFuncAllReduce:
    return ddaOn(c) && ncclAllReduceDdaFabricLL128Eligible(c.comm, c.sendbuff, c.recvbuff, c.count, c.datatype, c.op);
  default: return false;
  }
}

ncclResult_t fabricLL128Choose(rcclSchedCtx& c, rcclCollDecision* d, bool* ok) {
  d->algo = RCCL_DDA_FABRIC_LL128;
  d->protocol = NCCL_PROTO_LL128;
  switch (c.func) {
  case ncclFuncBroadcast: d->nMaxChannels = ncclBroadcastDdaFabricLL128Blocks(c.comm, c.count, c.datatype); break;
  case ncclFuncAlltoAll: d->nMaxChannels = ncclAllToAllDdaFabricLL128Blocks(c.comm, c.count, c.datatype); break;
  case ncclFuncAllGather: d->nMaxChannels = ncclAllGatherDdaFabricLL128Blocks(c.comm, c.count, c.datatype); break;
  case ncclFuncReduceScatter:
    d->nMaxChannels = ncclReduceScatterDdaFabricLL128Blocks(c.comm, c.count, c.datatype);
    break;
  case ncclFuncAllReduce:
    d->ddaCap = c.ddaCap;
    d->nMaxChannels = ncclAllReduceDdaFabricLL128Blocks(c.comm, c.count, c.datatype, c.ddaCap);
    break;
  default: break;
  }
  *ok = true;
  return ncclSuccess;
}

bool fabricVmmEligible(rcclSchedCtx& c) {
  if (!underEntryCap(c, false) || !ddaOn(c)) return false;
  switch (c.func) {
  case ncclFuncAlltoAll: return ncclAllToAllDdaFabricEligible(c.comm, c.sendbuff, c.recvbuff, c.count, c.datatype);
  case ncclFuncAllGather: return ncclAllGatherDdaFabricEligible(c.comm, c.sendbuff, c.recvbuff, c.count, c.datatype);
  case ncclFuncReduceScatter:
    return ncclReduceScatterDdaFabricEligible(c.comm, c.sendbuff, c.recvbuff, c.count, c.datatype, c.op);
  case ncclFuncAllReduce:
    return ncclAllReduceDdaFabricEligible(c.comm, c.sendbuff, c.recvbuff, c.count, c.datatype, c.op);
  default: return false;
  }
}

ncclResult_t fabricVmmChoose(rcclSchedCtx& c, rcclCollDecision* d, bool* ok) {
  d->algo = RCCL_DDA_FABRIC_VMM;
  switch (c.func) {
  case ncclFuncAlltoAll: d->nMaxChannels = ncclAllToAllDdaFabricBlocks(c.comm, c.count, c.datatype); break;
  case ncclFuncAllGather: d->nMaxChannels = ncclAllGatherDdaFabricBlocks(c.comm, c.count, c.datatype); break;
  case ncclFuncAllReduce:
    d->ddaCap = c.ddaCap;
    d->nMaxChannels = ncclAllReduceDdaFabricBlocks(c.comm, c.count, c.datatype, c.ddaCap);
    break;
  default: break;  // ReduceScatter has no Blocks helper
  }
  *ok = true;
  return ncclSuccess;
}

bool ipcEligible(rcclSchedCtx& c) {
  if (!underEntryCap(c, false) || !ddaOn(c)) return false;
  switch (c.func) {
  case ncclFuncAlltoAll: return ncclAllToAllDdaIpcEligible(c.comm, c.sendbuff, c.recvbuff, c.count, c.datatype);
  case ncclFuncAllGather: return ncclAllGatherDdaIpcEligible(c.comm, c.sendbuff, c.recvbuff, c.count, c.datatype);
  case ncclFuncReduceScatter:
    return ncclReduceScatterDdaIpcEligible(c.comm, c.sendbuff, c.recvbuff, c.count, c.datatype, c.op);
  case ncclFuncAllReduce: return ncclAllReduceDdaIpcEligible(c.comm, c.sendbuff, c.recvbuff, c.count, c.datatype, c.op);
  default: return false;
  }
}

ncclResult_t ipcChoose(rcclSchedCtx& c, rcclCollDecision* d, bool* ok) {
  d->algo = RCCL_DDA_IPC;
  switch (c.func) {
  case ncclFuncAlltoAll: d->nMaxChannels = ncclAllToAllDdaIpcBlocks(c.comm, c.count, c.datatype); break;
  case ncclFuncAllGather: d->nMaxChannels = ncclAllGatherDdaIpcBlocks(c.comm, c.count, c.datatype); break;
  case ncclFuncAllReduce: d->nMaxChannels = ncclAllReduceDdaIpcBlocks(c.comm, c.count, c.datatype); break;
  default: break;
  }
  *ok = true;
  return ncclSuccess;
}

}  // namespace

const rcclBackend kDdaFabricLL = {"dda.fabric.ll", RCCL_BACKEND_DDA, fabricLLEligible, fabricLLChoose, nullptr, false};
const rcclBackend kDdaFabricLL128 = {"dda.fabric.ll128", RCCL_BACKEND_DDA, fabricLL128Eligible, fabricLL128Choose, nullptr,
                                    false};
const rcclBackend kDdaFabricVmm = {"dda.fabric.vmm", RCCL_BACKEND_DDA, fabricVmmEligible, fabricVmmChoose, nullptr, false};
const rcclBackend kDdaIpc = {"dda.ipc", RCCL_BACKEND_DDA, ipcEligible, ipcChoose, nullptr, false};

namespace {

const rcclRule kFabricTiers[] = {
  Rule(kDdaFabricLL, RCCL_LIMIT_DDA_LL),
  Rule(kDdaFabricLL128, RCCL_LIMIT_DDA_LL128),
  Rule(kDdaFabricVmm, RCCL_LIMIT_DDA_VMM),
};

}  // namespace

const rcclRuleGroup kDdaFabricTiers = {kFabricTiers, sizeof(kFabricTiers) / sizeof(kFabricTiers[0])};
