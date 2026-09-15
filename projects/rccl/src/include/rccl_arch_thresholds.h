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
// AR/AG/A2A compare total message bytes; RS compares rsShardBytes (per-rank).
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
  // AR/AG/AlltoAll compare total message bytes; RS compares per-rank shard bytes.
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

  // Per-size unroll factor breakpoints for gfx1250.  Each entry is a
  // (maxBytes, unrollIdx) pair: the first entry whose maxBytes >= msgBytes
  // wins.  A terminal entry with maxBytes == SIZE_MAX covers everything larger.
  // unrollIdx values: NCCL_UNROLL_1=0, NCCL_UNROLL_2=1, NCCL_UNROLL_4=2,
  //                   NCCL_UNROLL_8=3, NCCL_UNROLL_16=4, NCCL_UNROLL_32=5.
  // Null pointer means "keep the comm-level default (gfx1250 default = UNROLL_32)".
  struct rcclUnrollEntry { size_t maxBytes; int unrollIdx; };
  const rcclUnrollEntry* unrollMapAR;   // AllReduce per-size unroll breakpoints
  const rcclUnrollEntry* unrollMapAG;   // AllGather per-size unroll breakpoints
  const rcclUnrollEntry* unrollMapRS;   // ReduceScatter per-size unroll breakpoints
  const rcclUnrollEntry* unrollMapA2A;  // AlltoAll per-size unroll breakpoints
};

const rcclArchThresholds* rcclGetArchThresholds(const char* gcn);
int rcclGetTuningIndexForArch(const char* gfxarch);

#endif  // RCCL_ARCH_THRESHOLDS_H_
