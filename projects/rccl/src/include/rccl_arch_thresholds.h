/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_ARCH_THRESHOLDS_H_
#define RCCL_ARCH_THRESHOLDS_H_

// Per-architecture DDA/CE/Ring dispatch threshold tables.
// Extracted from graph.h to isolate RCCL-specific dispatch logic from
// NVIDIA-originated graph infrastructure, reducing merge-conflict surface.

#include "nccl_common.h"  // ncclFunc_t, ncclFuncAlltoAll
#include <stddef.h>       // size_t, SIZE_MAX

// DDA arrays are indexed by ncclFunc_t through AlltoAll (ncclFuncAlltoAll == 8).
// All collectives compare total message bytes against the table values.
// 0 disables that tier for that collective. No parallel built-in defaults:
// an unset env var uses this table, and an arch with no table gets 0.
enum { RCCL_DDA_FUNC_COUNT = ncclFuncAlltoAll + 1 };

// Sentinel for symMaxR2/symMaxR2Graph: no upper-bound suppression (keep symk at all sizes).
// Use this instead of 0 in symMaxR2 arrays -- 0 in a DDA array means disabled, which is
// inconsistent with the no-suppression meaning needed for symMaxR2 unused/uncapped slots.
static constexpr size_t kThreshUnlimited = SIZE_MAX;

struct rcclArchThresholds {
  // DDA tier upper bounds, per collective.  gfx1250 uses fabric LL/LL128/VMM;
  // gfx942/gfx950 use ddaVmmMax as the DDA-IPC cap (LL/LL128 unused, stay 0).
  // All arrays are indexed by ncclFunc_t; 0 disables that tier for that coll.
  // All collectives compare total message bytes against the table values.
  size_t ddaLLMax[RCCL_DDA_FUNC_COUNT];     // DDA LL tier:    0 .. ddaLLMax[func]
  size_t ddaLL128Max[RCCL_DDA_FUNC_COUNT];  // DDA LL128 tier: ddaLLMax[func]+1 .. ddaLL128Max[func]
  size_t ddaVmmMax[RCCL_DDA_FUNC_COUNT];    // DDA VMM/IPC cap: ddaLL128Max[func]+1 .. ddaVmmMax[func]

  // R2 variant: DDA VMM cap when recv buffer is registered (winRegType !=
  // ncclSymSendNonregRecvNonreg). Selectors still require !symEligible (AR:
  // !symkRequested) before DDA; registered sum/avg windows never reach this
  // cap. It only shortens DDA when recv is registered and DDA is still
  // eligible (e.g. a non-sum RS). 0 means "use ddaVmmMax".
  size_t ddaVmmMaxR2[RCCL_DDA_FUNC_COUNT];

  // Graph-mode variant: DDA VMM cap to use when the comm is inside a graph
  // capture (graphCapturingHint=true).  CE AllReduce is blocked during graph
  // captures, so DDA can fill the full window that CE would normally absorb.
  // 0 means "use ddaVmmMax" (no graph-specific override).
  size_t ddaVmmMaxGraph[RCCL_DDA_FUNC_COUNT];

  // CE non-registered (scratch) window per collective, total bytes.
  // CE-Scratch fires when ceNonRegMin[func] <= totalBytes <= ceNonRegMax[func].
  // ceNonRegMin[func] = 0 means no lower bound (fires immediately above DDA exit).
  // ceNonRegMax[func] = 0 means CE-Scratch / CE 2-shot disabled for that collective.
  // AllReduce 2-shot: this is a selector cap only. ceARTmpBuf is allocated at
  // NCCL_CE_AR_TMPBUF_DEFAULT_BYTES and grown only when this entry (or
  // RCCL_CE_AR_MAX_MSG_BYTES) is larger.
  size_t ceNonRegMin[RCCL_DDA_FUNC_COUNT];
  size_t ceNonRegMax[RCCL_DDA_FUNC_COUNT];

  // CE registered-window upper bound per collective, total bytes.
  // For AllReduce: registered CE copies through user symmetric windows (tuning cap only).
  // For AllGather: upper bound for CE-registered (R=2). 0 means no upper bound.
  size_t ceRegMax[RCCL_DDA_FUNC_COUNT];

  // Symmetric kernel upper-bound per collective when recv buffer is registered (R2).
  // Above this size CE is faster than symk; setting this withdraws symk as the final
  // choice in rcclSelectAllReduce so CE-registered can win. It does not unblock the
  // CE 2-shot or DDA branches, which stay gated on whether symk was requested at all.
  // kThreshUnlimited means no suppression (symk may win at any size for that collective).
  // Unused collectives default to kThreshUnlimited.
  size_t symMaxR2[RCCL_DDA_FUNC_COUNT];
  // Graph-mode variant of symMaxR2 (graphCapturingHint / stream capture). CE is
  // graph-unsafe, so graph mode typically wants no suppression (keep symk).
  // kThreshUnlimited means no suppression, same convention as symMaxR2 -- not "inherit eager".
  size_t symMaxR2Graph[RCCL_DDA_FUNC_COUNT];
  // Symmetric kernel lower-bound per collective when recv buffer is registered (R2).
  // Below this size DDA is faster than symk; setting this suppresses symk so DDA
  // can win in that sub-range while symk still wins above it.
  // 0 means no suppression (symk may win at any size for that collective).
  size_t symMinR2[RCCL_DDA_FUNC_COUNT];

  // Rank-scaled tier caps, bytes per rank. On a comm of at least
  // rankScaledMinRanks ranks, a non-zero entry replaces the matching
  // ddaLLMax / ddaLL128Max / ddaVmmMax total with nRanks * entry, clipped to
  // rankScaledMaxBytes (which must not exceed the largest total above: fabric
  // scratch is sized from the totals). Smaller comms, and entries of 0, keep the
  // totals. The RCCL_DDA_*_THRESHOLD env vars still win over both.
  size_t ddaLLMaxPerRank[RCCL_DDA_FUNC_COUNT];
  size_t ddaLL128MaxPerRank[RCCL_DDA_FUNC_COUNT];
  size_t ddaVmmMaxPerRank[RCCL_DDA_FUNC_COUNT];
  int rankScaledMinRanks;     // 0 = no rank scaling
  size_t rankScaledMaxBytes;

  // AllReduce only: on comms of at least ddaArVmmFirstMinRanks ranks, DDA VMM
  // takes unregistered messages in (ddaArVmmFirstMinBytes, nRanks *
  // ddaArVmmFirstMaxPerRank] ahead of the LL tier, provided the VMM path is
  // eligible. 0 ranks = off.
  int ddaArVmmFirstMinRanks;
  size_t ddaArVmmFirstMinBytes;
  size_t ddaArVmmFirstMaxPerRank;

  // DDA VMM (AllReduce / ReduceScatter) launches min(init block cap,
  // clamp(ddaVmmBlockBudget / nRanks, ddaVmmMinBlocks, ddaVmmMaxBlocks)) blocks
  // on comms of at least rankScaledMinRanks ranks. Budget 0 = the init block cap.
  int ddaVmmBlockBudget;
  int ddaVmmMinBlocks;
  int ddaVmmMaxBlocks;

  // Ring / Tree crossovers that replace the tuning model's ranges on single-node
  // comms (nNodes == 1). Sizes are per rank as rcclGetSizePerRank() reports them
  // (AllReduce: the whole message, hence nRanks x the entry). 0 = keep the
  // model's range. The two AllReduce LL128 entries apply at any rank count, the
  // rest from rankScaledMinRanks ranks.
  size_t arRingLL128MaxPerRank;   // Ring AllReduce LL128 up to nRanks x this
  bool arTreeLL128Unlimited;      // Tree AllReduce keeps LL128 at every size
  size_t arTreeMaxPerRank;        // AllReduce runs Tree instead of Ring up to nRanks x this,
  size_t arTreeLLMaxPerRank;      // with LL up to nRanks x this and LL128 above
  size_t agRingLL128MaxPerRank;   // AllGather Ring LL128 up to this per rank
  size_t rsRingLL128MinPerRank;   // ReduceScatter Ring LL128 above this per rank,
  size_t rsRingLL128MaxPerRank;   // up to this (2- and 4-byte types)
  size_t rsRingLL128MaxPerRank1B; // or this (1-byte types)
  size_t rsRingSimpleCappedMaxPerRank;  // Ring Simple ReduceScatter above the LL128 band,
  int rsRingSimpleCappedChannels;       // up to this per rank, runs on at most this many channels

  // Force CE-Scratch for unregistered AllGather / AlltoAll / Gather / Scatter
  // whenever the receive fits ddaScratch, on top of the ceNonRegMin/Max window.
  // NCCL_CTA_POLICY_ZERO forces it on every arch (it is the one zero-CU path for
  // unregistered buffers). RCCL_FORCE_CE=0 turns it off everywhere. A null table forces it.
  bool forceCeScratch;
};

// Compute-partition mode a gfx1250 comm is tuned for (RCCL_TUNING_PARTITION_MODE). It picks
// the table only; other archs have one table for every mode.
enum rcclTuningPartition {
  RCCL_TUNING_PARTITION_SPX = 0,
  RCCL_TUNING_PARTITION_DPX = 1,
};

const rcclArchThresholds* rcclGetArchThresholds(const char* gcn,
                                                rcclTuningPartition partition = RCCL_TUNING_PARTITION_SPX);

#endif  // RCCL_ARCH_THRESHOLDS_H_
