/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host side of the DDA fabric AllToAll ZERO-COPY direct path
 * (kernel: alltoall_dda_fabric_direct.h). Included at the end of
 * dda_alltoall_fabric.cu so it shares that translation unit's device-linker
 * build rule.
 *
 * Owns a small fabric-exported flag buffer per rank, created together with the push buffer by the
 * first AllToAll that can use either (ncclDdaA2aLazyInit). Takes a call only when its receive
 * buffer lies in a symmetric window (ncclCommWindowRegister with NCCL_WIN_COLL_SYMMETRIC, LSA
 * registration): window registration is collective, and the symmetric contract has every rank pass
 * its buffer at the same window offset, so whether a call is windowed is the same on every rank and
 * each peer's recvbuff comes from this rank's LSA mapping. Every pair is written directly.
 * Buffers registered with ncclCommRegister are not used here: that registration is per rank (a
 * rank may register and its peer not), and a per-rank input would let ranks pick different kernels
 * for one call, which hangs (no rank could tell). They keep the push / p2p paths.
 *
 * Env (all RCCL_*):
 *   DDA_A2A_DIRECT              0 disables (default 1; no flag buffer)
 * Selection window: rcclArchThresholds::a2aDirectPerPeerCut / a2aDirectMin / a2aDirectMax
 * (rcclSelectAlltoAllFabric). Tuning builds (RCCL_DDA_A2A_TUNING, dda_alltoall.h) also read the
 * geometry values DDA_A2A_DIRECT_BPP (0 = rank-count table), _SELF_BLOCKS (12), _THREADS (256, max
 * 512), _UNROLL (0 = size table; 8 or 16).
 * See LICENSE.txt for license information.
 ************************************************************************/

#include "algorithms/dda/alltoall/alltoall_dda_fabric_direct.h"
#include "alloc.h"
#include "dev_runtime.h"
#include "param.h"

#include <vector>

RCCL_PARAM(DdaA2aDirect, "DDA_A2A_DIRECT", 1);
RCCL_A2A_KNOB(DdaA2aDirectBpp, "DDA_A2A_DIRECT_BPP", 0);  // 0 = ddaA2aDirectBppAuto(nRanks)
RCCL_A2A_KNOB(DdaA2aDirectSelfBlocks, "DDA_A2A_DIRECT_SELF_BLOCKS", 12);
RCCL_A2A_KNOB(DdaA2aDirectThreads, "DDA_A2A_DIRECT_THREADS", 256);
RCCL_A2A_KNOB(DdaA2aDirectUnroll, "DDA_A2A_DIRECT_UNROLL", 0);  // 0 = size table

namespace {

struct DdaA2aDirectState {
  std::vector<void*> peerMaps;  // imported peer flag buffers (self slot = base), freed at Fini
  void* base = nullptr;
  char** peerFlagsDev = nullptr;
};

static int ddaA2aDirectClamp(int64_t v, int lo, int hi, int dflt) {
  return (v >= lo && v <= hi) ? (int)v : dflt;
}

// Blocks per peer row before the residency cap. Small comms need more blocks per row to fill the
// fabric: P05 (R0 OOP busBW, mt6t, 16M/64M/256M) 2r cross-chip (GPUs 0,2) 12 -> 48: 109/131/129 ->
// 270/455/377 (sibling 0,1: within -6%..+0%); 4r 12 -> 20: 0-3 240/335/340 -> 319/450/517 (0-3 dips to
// 170-280 at 32-64M with 24/28), 0,2,4,6 245/340/358 -> 324/505/550. 8r keeps 12 (A08), 16r is capped
// to 5 by residency anyway. n = 3 is interpolated (unmeasured).
static int ddaA2aDirectBppAuto(int n) { return n <= 2 ? 48 : n <= 4 ? 20 : 12; }

// Grid geometry (residency, see the kernel header): the grid must fit the device at once, so it is
// capped at comm->ddaFabricMaxBlocks (comm-wide min CU count, which RCCL_DDA_FABRIC_MAXBLOCKS
// lowers). All inputs are comm-uniform, so every rank launches the same grid and the same B (the
// DONE[j] / stride partition requires B to match on both sides of a pair).
// 8 ranks x 96 CUs: 12 + 7 x 12 = 96; 16 ranks: 12 + 15 x 5 = 87; 4 ranks 12 + 3 x 20 = 72;
// 2 ranks 12 + 48 = 60.
static bool ddaA2aDirectGeom(const ncclComm* comm, int* bppOut, int* selfOut) {
  const int n = comm->nRanks;
  const int maxBlk = comm->ddaFabricMaxBlocks;
  int sb = ddaA2aDirectClamp(rcclParamDdaA2aDirectSelfBlocks(), 1, 1024, 12);
  int bpp = ddaA2aDirectClamp(rcclParamDdaA2aDirectBpp(), 1, dda::common::kDdaA2aDirectMaxBpp, ddaA2aDirectBppAuto(n));
  if (n < 2 || maxBlk < n) return false;  // not even one block per peer + one self block
  if (sb > maxBlk - (n - 1)) sb = maxBlk - (n - 1);
  const int cap = (maxBlk - sb) / (n - 1);
  if (bpp > cap) bpp = cap;
  if (bppOut) *bppOut = bpp;
  if (selfOut) *selfOut = sb;
  return bpp >= 1 && sb >= 1;
}

// Every peer's recvbuff in this rank's VA, from the symmetric window that holds recvbuff (same
// offset on every rank). False if any peer has no LSA mapping or a misaligned one.
static bool ddaA2aDirectPeerRecv(ncclComm* comm, void* recvbuff, uintptr_t* out) {
  struct ncclDevrWindow* win = nullptr;
  if (ncclDevrFindWindow(comm, recvbuff, &win) != ncclSuccess || win == nullptr) return false;
  const size_t off = (uintptr_t)recvbuff - (uintptr_t)win->userPtr;
  for (int r = 0; r < comm->nRanks; ++r) {
    out[r] = 0;
    if (r == comm->rank) continue;
    int lsa = -1;
    void* ptr = nullptr;
    if (ncclDevrWorldToLsaRank(comm, r, &lsa) != ncclSuccess || lsa < 0 ||
        ncclDevrGetLsaRankPtr(comm, win, off, lsa, &ptr) != ncclSuccess || ptr == nullptr ||
        ((uintptr_t)ptr & 15) != 0)
      return false;
    out[r] = (uintptr_t)ptr;
  }
  return true;
}

} // namespace

ncclResult_t ncclDdaA2aDirectCommFini(ncclComm* comm) {
  if (comm == nullptr || comm->ddaA2aDirectState == nullptr) return ncclSuccess;
  auto* st = static_cast<DdaA2aDirectState*>(comm->ddaA2aDirectState);
  CUDACHECKIGNORE(cudaFree(st->peerFlagsDev));
  for (size_t i = 0; i < st->peerMaps.size(); ++i) {
    if ((int)i != comm->rank && st->peerMaps[i] != nullptr) (void)ncclCuMemFreeAddr(st->peerMaps[i], comm->memManager);
  }
  (void)ncclCuMemFree(st->base, comm->memManager);
  delete st;
  comm->ddaA2aDirectState = nullptr;
  return ncclSuccess;
}

// Shape + window, independent of the path's state (the lazy trigger uses it before the flag buffer
// exists). Every input is the same on every rank under the collective contract: count, datatype,
// nRanks, localRanks; out of place, 16B alignment and "recvbuff in a symmetric window" are the
// caller's to keep the same on every rank (as for the symmetric kernels). No side effects.
bool ncclAllToAllDdaFabricDirectWindowed(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                         ncclDataType_t datatype) {
  if (comm == nullptr || count == 0 || sendbuff == recvbuff) return false;
  if (comm->nRanks > dda::common::kDdaA2aDirectMaxRanks || comm->localRanks != comm->nRanks) return false;
  const size_t perPeer = count * ncclTypeSize(datatype);
  if (perPeer % 16 != 0) return false;
  if (((uintptr_t)sendbuff | (uintptr_t)recvbuff) % 16 != 0) return false;
  struct ncclDevrWindow* win = nullptr;
  if (ncclDevrFindWindow(comm, recvbuff, &win) != ncclSuccess || win == nullptr) return false;
  if (!(win->winFlags & NCCL_WIN_COLL_SYMMETRIC) || !ncclDevrWinRegEnabled(win->winFlags, ncclDevrRegisterLsa))
    return false;
  return (uintptr_t)recvbuff - (uintptr_t)win->userPtr + perPeer * comm->nRanks <= win->size;
}

// Windowed + the agreed flag buffer + geometry + every peer's mapping. No side effects.
bool ncclAllToAllDdaFabricDirectEligible(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t count,
                                         ncclDataType_t datatype) {
  if (comm == nullptr || comm->ddaA2aDirectState == nullptr) return false;
  if (!ncclAllToAllDdaFabricDirectWindowed(comm, sendbuff, recvbuff, count, datatype)) return false;
  if (!ddaA2aDirectGeom(comm, nullptr, nullptr)) return false;
  uintptr_t peerRecv[dda::common::kDdaA2aDirectMaxRanks];
  return ddaA2aDirectPeerRecv(comm, recvbuff, peerRecv);
}

uint32_t ncclAllToAllDdaFabricDirectBlocks(ncclComm* comm) {
  int bpp = 1, sb = 1;
  (void)ddaA2aDirectGeom(comm, &bpp, &sb);
  return (uint32_t)(sb + (comm->nRanks - 1) * bpp);
}

static ncclResult_t ddaA2aDirectLaunch(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                       ncclComm* comm, cudaStream_t stream) {
  auto* st = static_cast<DdaA2aDirectState*>(comm->ddaA2aDirectState);
  const size_t perPeer = count * ncclTypeSize(datatype);
  const size_t total = perPeer * comm->nRanks;
  dda::common::DdaA2aDirectArgs a = {};
  // The selector checked all of these on this call; a failure here is internal.
  if (st == nullptr || !ddaA2aDirectGeom(comm, &a.bpp, &a.selfBlocks) ||
      !ddaA2aDirectPeerRecv(comm, recvbuff, a.peerRecv))
    return ncclInternalError;
  a.sendbuff = static_cast<const char*>(sendbuff);
  a.recvbuff = static_cast<char*>(recvbuff);
  a.peerFlags = st->peerFlagsDev;
  a.selfFlags = static_cast<char*>(st->base);
  a.wd = ncclDdaA2aWdCtx(comm);
  a.chunkBytes = perPeer;
  a.rank = comm->rank;
  a.nRanks = comm->nRanks;
  const dim3 grid(ncclAllToAllDdaFabricDirectBlocks(comm));
  const int64_t thr = rcclParamDdaA2aDirectThreads();
  const dim3 block((thr >= 64 && thr <= 512 && thr % 64 == 0) ? (unsigned)thr : 256u);
  int64_t unroll = rcclParamDdaA2aDirectUnroll();
  if (unroll != 8 && unroll != 16) unroll = total > (256u << 20) ? 16 : 8;

  INFO(NCCL_COLL, "DDA fabric AllToAll DIRECT (zero-copy): nRanks=%d perPeer=%zu grid=%u bpp=%d self=%d block=%u unroll=%lld",
       comm->nRanks, perPeer, grid.x, a.bpp, a.selfBlocks, block.x, (long long)unroll);

  const hipEvent_t stopEvent = rcclTakeAddonStopEvent(comm);
  (void)cudaGetLastError();  // drop sticky errors of handled calls (refused FABRIC export in p2p IPC registration)
  if (unroll == 16) {
    hipExtLaunchKernelGGL((dda::common::ddaAllToAllFabricDirect<16>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, a);
  } else {
    hipExtLaunchKernelGGL((dda::common::ddaAllToAllFabricDirect<8>), grid, block, 0, stream, /*startEvent=*/nullptr,
                          stopEvent, /*flags=*/0, a);
  }
  CUDACHECK(cudaGetLastError());
  return ncclSuccess;
}

ncclResult_t ncclAllToAllDdaFabricDirect(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype,
                                         ncclComm* comm, cudaStream_t stream) {
  // Any failure poisons the comm: peers may already run this call, so this rank's device epoch would
  // be one behind theirs; later AllToAlls are refused instead (as for push, rcclDdaA2aPushRun).
  const ncclResult_t r = ddaA2aDirectLaunch(sendbuff, recvbuff, count, datatype, comm, stream);
  if (r != ncclSuccess) ncclDdaA2aPoison(comm, kDdaA2aErrLaunch);
  return r;
}
