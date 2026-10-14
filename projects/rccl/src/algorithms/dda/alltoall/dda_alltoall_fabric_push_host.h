/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host side of the DDA fabric AllToAll one-shot PUSH path
 * (kernel: alltoall_dda_fabric_push.h). Included at the end of
 * dda_alltoall_fabric.cu so it shares that translation unit's device-linker
 * build rule; it has no other coupling to the VMM pull path.
 *
 * Owns a dedicated fabric-exported buffer per rank ([2 MiB flags | data]),
 * separate from comm->ddaScratch: the push path lets peers write into this
 * rank's buffer while it is still copying out of it, so it must not share
 * memory with the LL/LL128/VMM/CE-scratch users of ddaScratch.
 *
 * Out of place only (the selector never routes in place here; in place is deferred).
 * Above the buffer (an op larger than the agreed buffer, e.g. after it shrank): the op runs as
 * several launches ("rounds"), round r moving bytes [r*c, r*c + c) of every per-peer chunk
 * (args.strideBytes = per-peer stride of the user buffers, args.chunkBytes = c). Each round is an
 * ordinary op of the protocol (own epoch, own grid from the same comm-uniform inputs) on disjoint
 * user bytes.
 *
 * The buffer is created by the first AllToAll that can use it (ncclDdaA2aLazyInit,
 * dda_alltoall_fabric_lazy_host.h), never for comms that do not run one. It is persistent memory:
 * ncclCommSuspend does not release it (only ncclCommDestroy / ncclCommAbort do).
 * Env (all RCCL_*):
 *   DDA_A2A_PUSH              1: push path on (0 disables; no buffer)
 *   DDA_A2A_PUSH_BUFFER_SIZE  data bytes per rank (default 1 GiB: every op up to 512 MiB alternates
 *                             halves); ranks agree on the minimum when the buffer is created
 * Selection window: above the arch table's LL128 window, up to rcclArchThresholds::a2aPushMax
 * (rcclSelectAlltoAllFabric); CU-split kernel from a2aPushCuMin. Tuning builds
 * (RCCL_DDA_A2A_TUNING, dda_alltoall.h) also read the geometry values: _CU (1 always, 0 never,
 * -1 table), _CU_MIN, _BPP (0 = rank-count table), _RECV_MULT (4), _THREADS (256), _PIECES
 * (0 = size table), _TILE_BYTES (256 KiB).
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/alltoall/alltoall_dda_fabric_push.h"
#include "alloc.h"
#include "bootstrap.h"
#include "param.h"
#include "rccl_arch_thresholds.h"

#include <algorithm>
#include <vector>

RCCL_PARAM(DdaA2aPush, "DDA_A2A_PUSH", 1);
RCCL_PARAM(DdaA2aPushBufferSize, "DDA_A2A_PUSH_BUFFER_SIZE", 1024ll << 20);  // 1 GiB: ops up to 512 MiB double-buffer
RCCL_A2A_KNOB(DdaA2aPushBpp, "DDA_A2A_PUSH_BPP", 0);  // 0 = rank-count table
RCCL_A2A_KNOB(DdaA2aPushThreads, "DDA_A2A_PUSH_THREADS", 256);
RCCL_A2A_KNOB(DdaA2aPushPieces, "DDA_A2A_PUSH_PIECES", 0);  // 0 = size table
RCCL_A2A_KNOB(DdaA2aPushRecvMult, "DDA_A2A_PUSH_RECV_MULT", 4);
RCCL_A2A_KNOB(DdaA2aPushCu, "DDA_A2A_PUSH_CU", -1);  // CU-split kernel: 1 always, 0 never, -1 from a2aPushCuMin
RCCL_A2A_KNOB(DdaA2aPushCuMin, "DDA_A2A_PUSH_CU_MIN", -1);  // -1: rcclArchThresholds::a2aPushCuMin
RCCL_A2A_KNOB(DdaA2aPushTileBytes, "DDA_A2A_PUSH_TILE_BYTES", 256 << 10);  // CU-split copy tile

namespace {

struct DdaA2aPushState {
  std::vector<void*> peerMaps;  // imported peer buffers (self slot = base), freed at Fini
  void* base = nullptr;
  size_t dataBytes = 0;
  char** peerBaseDev = nullptr;
  // Same-package sibling from topology (rcclDdaA2aTopoSibling; -1 none) and whether every rank of the
  // comm has one (identical on every rank: computed from the gathered peerInfo).
  int sibRank = -1;
  bool allPaired = false;
  // CU-split kernel: dynamic LDS reserved per block, and the kernels whose max-dynamic-LDS attribute
  // was raised to it. Per comm (i.e. per device): the attribute is per device, and one process may
  // drive several GPUs (ncclCommInitAll).
  size_t cuLds = 0;
  std::vector<const void*> cuLdsSet;
};

// CU-split kernel geometry (tuned on DPX, 96 CUs): the row to the same-package sibling (~2x faster
// link) gets 4 blocks, every other row bpp; at least 32 copy blocks (8 ranks: 64 push + 32 copy).
constexpr int kDdaA2aPushCuSibBpp = 4;
constexpr int kDdaA2aPushCuCopyMin = 32;
// Non-CU grid cap, in blocks per comm-wide CU (ddaFabricMaxBlocks, which RCCL_DDA_FABRIC_MAXBLOCKS
// lowers): 960 on 96 CUs. Measured configurations stay as they were (2r 576, 4r 532, 8r 468,
// 16r 948 blocks); above that M drops toward 1, then bpp (32r: M 1 -> 756; 72r: bpp 6, M 1 -> 858).
// A cap bounds the grid; it does not make two push kernels co-resident (see the kernel header).
constexpr unsigned kDdaA2aPushGridPerCu = 10;

// Push blocks of one rank (the sibling row, if any, gets sibBpp blocks; sibBpp > 0 only when every
// rank has a sibling, so the count is the same on every rank).
static int ddaA2aPushNumPush(int nRanks, int bpp, int sibBpp) {
  return sibBpp > 0 ? (nRanks - 2) * bpp + sibBpp : (nRanks - 1) * bpp;
}

// CU-split: one block per CU (LDS-reserved), grid = comm->ddaFabricMaxBlocks (comm-wide min CU cap).
// Largest bpp <= want whose push blocks leave kDdaA2aPushCuCopyMin copy blocks; 0 if none (CU mode
// off). Inputs are comm-uniform, so every rank picks the same kernel and bpp.
static int ddaA2aPushCuBppFit(const ncclComm* comm, int sibBpp, int want) {
  const int budget = comm->ddaFabricMaxBlocks - kDdaA2aPushCuCopyMin;
  for (int b = want; b >= 1; --b) {
    if (ddaA2aPushNumPush(comm->nRanks, b, sibBpp) <= budget) return b;
  }
  return 0;
}

// Blocks per peer row: small comms fill the GPU (P05, measured in place, R0 busBW 64M/256M, CU off):
// 2r 12 -> 64: GPUs 0,2 114/119 -> 335/405, 0,1 255/270 -> 430/540; 4r 12 -> 28: 0-3 250/272 -> 406/453,
// 0,2,4,6 276/307 -> 419/522. n >= 5 keeps the 8r-tuned 12 (unmeasured below 8). CU-split: 10.
static int ddaA2aPushBppWant(const ncclComm* comm, bool cuMode) {
  const int64_t env = rcclParamDdaA2aPushBpp();
  const int n = comm->nRanks;
  int bpp = env > 0 ? (int)env : cuMode ? 10 : n <= 2 ? 64 : n == 3 ? 42 : n == 4 ? 28 : 12;
  if (bpp > dda::common::kDdaA2aPushMaxBpp) bpp = dda::common::kDdaA2aPushMaxBpp;
  return bpp;
}

static int ddaA2aPushCuSibBpp(const DdaA2aPushState* st) { return st->allPaired ? kDdaA2aPushCuSibBpp : 0; }

// CU-split kernel from rcclArchThresholds::a2aPushCuMin (comm-uniform: same arch table on every
// rank); off when its push blocks cannot leave room for the copy blocks (16 ranks keep it with bpp 4).
static size_t ddaA2aPushCuMinBytes(const ncclComm* comm) {
  const int64_t v = rcclParamDdaA2aPushCuMin();
  if (v >= 0) return (size_t)v;
  return comm->archThresholds != nullptr ? comm->archThresholds->a2aPushCuMin : 0;
}
static bool ddaA2aPushCuMode(const ncclComm* comm, const DdaA2aPushState* st, size_t totalBytes) {
  const int64_t v = rcclParamDdaA2aPushCu();
  const size_t cuMin = ddaA2aPushCuMinBytes(comm);
  if (!(v > 0 || (v < 0 && cuMin > 0 && totalBytes >= cuMin))) return false;
  if (v < 0) {
    // Not below 5 ranks: its sibling row and 1-block-per-CU split starve small comms (P05, measured
    // in place 256M: 2r GPUs 0,2 CU 39 vs non-CU 64/peer 416; 0,1 139 vs 537; 4r 0-3 268 vs 453).
    if (comm->nRanks <= 4) return false;
    // Tuned and validated on DPX only, where every rank has a same-package sibling. Without a
    // topology-verified sibling for every rank (SPX trays, a comm that holds one partition of a
    // package) the non-CU kernel runs instead.
    if (!st->allPaired) return false;
  }
  return ddaA2aPushCuBppFit(comm, ddaA2aPushCuSibBpp(st), ddaA2aPushBppWant(comm, true)) >= 1;
}

static int ddaA2aPushRecvMult() {
  const int64_t v = rcclParamDdaA2aPushRecvMult();
  return v < 1 ? 1 : v > 8 ? 8 : (int)v;
}

// Launch geometry of one push round of `totalBytes` (nRanks x per-peer bytes). Every input is
// comm-uniform except sibRank, which is symmetric, so the sender and receiver of every row agree on
// its block count.
struct DdaA2aPushGeom {
  bool cu = false;
  int bpp = 0;
  int m = 1;         // receive blocks per sender block (non-CU)
  int sibRank = -1;  // CU-split only
  int sibBpp = 0;
  int nPush = 0;
  unsigned grid = 0;
};
static DdaA2aPushGeom ddaA2aPushGeom(const ncclComm* comm, size_t totalBytes) {
  const auto* st = static_cast<const DdaA2aPushState*>(comm->ddaA2aPushState);
  const int n = comm->nRanks;
  DdaA2aPushGeom g;
  g.cu = ddaA2aPushCuMode(comm, st, totalBytes);
  if (g.cu) {
    g.sibBpp = ddaA2aPushCuSibBpp(st);
    g.sibRank = g.sibBpp > 0 ? st->sibRank : -1;
    g.bpp = ddaA2aPushCuBppFit(comm, g.sibBpp, ddaA2aPushBppWant(comm, true));  // 8 ranks: 10, 16 ranks: 4 (96 CUs)
    g.nPush = ddaA2aPushNumPush(n, g.bpp, g.sibBpp);
    const unsigned nBlk = (unsigned)comm->ddaFabricMaxBlocks;
    g.grid = nBlk > (unsigned)g.nPush ? nBlk : (unsigned)g.nPush + 1;  // the fit keeps nPush + 32 <= nBlk
    return g;
  }
  // [self: B*M][push: (n-1)*B][recv: (n-1)*B*M], capped at kDdaA2aPushGridPerCu per CU.
  auto grid = [n](int b, int m) { return (unsigned)(b * m + (n - 1) * b + (n - 1) * b * m); };
  const unsigned cap = kDdaA2aPushGridPerCu * (unsigned)comm->ddaFabricMaxBlocks;
  g.bpp = ddaA2aPushBppWant(comm, false);
  g.m = ddaA2aPushRecvMult();
  while (g.m > 1 && grid(g.bpp, g.m) > cap) --g.m;
  while (g.bpp > 1 && grid(g.bpp, g.m) > cap) --g.bpp;
  g.nPush = (n - 1) * g.bpp;
  g.grid = grid(g.bpp, g.m);
  return g;
}

static unsigned ddaA2aPushThreads() {
  const int64_t v = rcclParamDdaA2aPushThreads();
  return (v >= 64 && v <= 1024 && v % 64 == 0) ? (unsigned)v : 256u;
}

} // namespace

ncclResult_t ncclDdaA2aPushCommFini(ncclComm* comm) {
  if (comm == nullptr || comm->ddaA2aPushState == nullptr) return ncclSuccess;
  auto* st = static_cast<DdaA2aPushState*>(comm->ddaA2aPushState);
  CUDACHECKIGNORE(cudaFree(st->peerBaseDev));
  for (size_t i = 0; i < st->peerMaps.size(); ++i) {
    if ((int)i != comm->rank && st->peerMaps[i] != nullptr) (void)ncclCuMemFreeAddr(st->peerMaps[i], comm->memManager);
  }
  (void)ncclCuMemFree(st->base, comm->memManager);
  delete st;
  comm->ddaA2aPushState = nullptr;
  return ncclSuccess;
}

// Capability only; the size window is the selector's (rcclSelectAlltoAllFabric).
// Out of place only (the kernels have no in-place hand-off); any size: above the buffer the op
// runs in rounds (see below).
bool ncclAllToAllDdaFabricPushEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                       ncclDataType_t datatype) {
  if (comm == nullptr || comm->ddaA2aPushState == nullptr || count == 0) return false;
  if (sendbuff == recvbuff) return false;
  const size_t perPeer = count * ncclTypeSize(datatype);
  if (perPeer % 16 != 0) return false;
  if (((uintptr_t)sendbuff | (uintptr_t)recvbuff) % 16 != 0) return false;
  return ncclAllToAllDdaFabricPushRoundBytes(comm, perPeer) >= 16;
}

// Per-peer bytes of one launch. Whole op if it fits the buffer; else R = ceil(P / cMax) equal
// rounds with cMax the largest 4 KiB multiple whose nRanks slots fit the buffer (4 KiB keeps
// every round's piece boundaries and data slots line-aligned). Inputs are comm-uniform
// (count, nRanks, the agreed buffer size), so every rank runs the same rounds and grids.
size_t ncclAllToAllDdaFabricPushRoundBytes(ncclComm* comm, size_t perPeer) {
  auto* st = static_cast<DdaA2aPushState*>(comm->ddaA2aPushState);
  if (st == nullptr) return 0;
  if (perPeer * (size_t)comm->nRanks <= st->dataBytes) return perPeer;
  const size_t cMax = st->dataBytes / (size_t)comm->nRanks / 4096 * 4096;
  if (cMax == 0) return 0;
  const size_t rounds = (perPeer + cMax - 1) / cMax;
  const size_t c = ((perPeer + rounds - 1) / rounds + 4095) / 4096 * 4096;
  return c < cMax ? c : cMax;
}

uint32_t ncclAllToAllDdaFabricPushBlocks(ncclComm* comm, size_t count, ncclDataType_t datatype) {
  const size_t total = ncclAllToAllDdaFabricPushRoundBytes(comm, count * ncclTypeSize(datatype)) * comm->nRanks;
  return ddaA2aPushGeom(comm, total).grid;
}

// One push launch (one protocol epoch) moving per-peer bytes [off, off + chunk) of a call whose
// user buffers have per-peer stride `stride`.
static ncclResult_t ddaA2aPushLaunch(const void* sendbuff, void* recvbuff, size_t stride, size_t off, size_t chunk,
                                     ncclComm* comm, cudaStream_t stream, hipEvent_t stopEvent) {
  auto* st = static_cast<DdaA2aPushState*>(comm->ddaA2aPushState);
  const size_t perPeer = chunk;
  const size_t total = perPeer * comm->nRanks;
  const DdaA2aPushGeom g = ddaA2aPushGeom(comm, total);

  dda::common::DdaA2aPushArgs a;
  a.sendbuff = static_cast<const char*>(sendbuff) + off;
  a.recvbuff = static_cast<char*>(recvbuff) + off;
  a.peerBase = st->peerBaseDev;
  a.selfBase = static_cast<char*>(st->base);
  a.wd = ncclDdaA2aWdCtx(comm);
  a.chunkBytes = perPeer;
  a.strideBytes = stride;
  a.halfBytes = st->dataBytes / 2 / 4096 * 4096;
  a.rank = comm->rank;
  a.nRanks = comm->nRanks;
  // Pieces per block: each costs a store drain, so only large ops split (tuned on DPX, fast fabric state).
  int64_t pieces = rcclParamDdaA2aPushPieces();
  if (pieces <= 0) pieces = total <= (16u << 20) ? 1 : total <= (64u << 20) ? 2 : 4;
  if (pieces > dda::common::kDdaA2aPushMaxPieces) pieces = dda::common::kDdaA2aPushMaxPieces;
  a.pieces = (int)pieces;
  a.recvMult = g.m;
  a.bpp = g.bpp;
  a.sibRank = g.sibRank;
  a.sibBpp = g.sibBpp;
  a.nPush = g.nPush;
  {
    int64_t tb = rcclParamDdaA2aPushTileBytes();
    if (tb < 4096) tb = 4096;
    a.tileVec = (int)(tb / 16);
  }
  const dim3 grid(g.grid);
  const dim3 block(ddaA2aPushThreads());

  INFO(NCCL_COLL, "DDA fabric AllToAll PUSH: nRanks=%d perPeer=%zu off=%zu stride=%zu grid=%u bpp=%d recvMult=%d block=%u pieces=%d cu=%d nPush=%d sibling=%d/%d tileVec=%d",
       comm->nRanks, perPeer, off, stride, grid.x, a.bpp, a.recvMult, block.x, a.pieces, (int)g.cu, a.nPush, a.sibRank,
       a.sibBpp, a.tileVec);

  (void)cudaGetLastError();  // drop sticky errors of handled calls (refused FABRIC export in p2p IPC registration)
  if (g.cu) {
    // CU-split: reserve more than half a CU's LDS so at most one block lands on a CU.
    if (st->cuLds == 0) {
      int dev = 0, v = 0;
      CUDACHECK(cudaGetDevice(&dev));
      if (hipDeviceGetAttribute(&v, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, dev) != hipSuccess || v <= 0) v = 65536;
      st->cuLds = (size_t)v / 2 + 1024;
    }
    const void* fn = (const void*)dda::common::ddaAllToAllFabricPushCu;
    if (std::find(st->cuLdsSet.begin(), st->cuLdsSet.end(), fn) == st->cuLdsSet.end()) {
      CUDACHECK(hipFuncSetAttribute(fn, hipFuncAttributeMaxDynamicSharedMemorySize, (int)st->cuLds));
      st->cuLdsSet.push_back(fn);
    }
    hipExtLaunchKernelGGL(dda::common::ddaAllToAllFabricPushCu, grid, block, st->cuLds, stream,
                          /*startEvent=*/nullptr, stopEvent, /*flags=*/0, a);
  } else {
    hipExtLaunchKernelGGL(dda::common::ddaAllToAllFabricPush, grid, block, 0, stream,
                          /*startEvent=*/nullptr, stopEvent, /*flags=*/0, a);
  }
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}

ncclResult_t ncclAllToAllDdaFabricPush(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                       ncclComm* comm, cudaStream_t stream) {
  const size_t perPeer = count * ncclTypeSize(datatype);
  const size_t c = comm->ddaA2aPushState != nullptr ? ncclAllToAllDdaFabricPushRoundBytes(comm, perPeer) : 0;
  // Multi-round (above the buffer): each round is a full op of the protocol on disjoint user
  // bytes; the addon stop event goes with the last launch.
  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  if (c != 0 && c < perPeer) {
    INFO(NCCL_COLL, "DDA fabric AllToAll PUSH: perPeer=%zu in %zu rounds of %zu", perPeer, (perPeer + c - 1) / c, c);
  }
  // Any failure poisons the comm (rcclDdaA2aPushRun): peers may already run this call.
  return rcclDdaA2aPushRun(
      perPeer, c,
      [&](size_t off, size_t len, bool last) {
        return ddaA2aPushLaunch(sendbuff, recvbuff, perPeer, off, len, comm, stream, last ? stopEvent : nullptr);
      },
      [&] { ncclDdaA2aPoison(comm, kDdaA2aErrLaunch); });
}
