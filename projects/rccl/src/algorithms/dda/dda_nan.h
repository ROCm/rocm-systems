/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

// Host side of the NaN-flag DDA kernels (algorithms/dda/device/CollCommon_nan.h).

#pragma once

#include "algorithms/dda/device/CollCommon_nan.h"
#include "archinfo.h"
#include "comm.h"
#include "p2p.h"
#include "rccl_common.h"
#include "register.h"
#include "register_inline.h"

#include <hip/hip_ext.h>

#include <algorithm>
#include <cstdlib>

// RCCL_DDA_NAN_MAX_BLOCKS caps the grid; it cannot exceed the epoch cells. The
// default is per path: on gfx1250 fabric the large calls scale with the grid up
// to 288 blocks, 3 per WGP (256 leaves 32 of the 96 WGPs a block short); the IPC
// path was tuned at 64.
static inline int rcclDdaNanMaxBlocks(const ncclComm* comm) {
  static const int env = [] {
    const char* s = ncclGetEnv("RCCL_DDA_NAN_MAX_BLOCKS");
    return s ? atoi(s) : 0;
  }();
  const int v = env > 0 ? env : (comm->ddaNanFabricMemHandler != nullptr ? 288 : 64);
  return v > dda::nan::kEpochCells ? dda::nan::kEpochCells : v;
}

// Block size: RCCL_DDA_NAN_THREADS (rounded down to a multiple of 64) when set,
// else the caller's tuned default.
static inline uint32_t rcclDdaNanThreads(uint32_t dflt) {
  static const int env = [] {
    const char* s = ncclGetEnv("RCCL_DDA_NAN_THREADS");
    return s ? atoi(s) / 64 * 64 : 0;
  }();
  const int v = env > 0 ? env : (int)dflt;
  return (uint32_t)(v < 64 ? 64 : (v > (int)dda::nan::kThreads ? (int)dda::nan::kThreads : v));
}

// Small blocks spread a call over more CUs, which shortens its latency: AllReduce
// is fastest at 128 threads, the copy collectives and ReduceScatter at 256.
static inline std::pair<dim3, dim3> rcclDdaNanGeometry(const ncclComm* comm, size_t units,
                                                       uint32_t defaultThreads = 256) {
  return dda::nan::geometry(units, rcclDdaNanMaxBlocks(comm), rcclDdaNanThreads(defaultThreads));
}

// AllGather and AlltoAll through scratch take the per-peer kernels when the
// per-peer block is at most RCCL_DDA_NAN_PEER_MAX bytes (default 128 KiB; 0
// disables them), the all-peer kernels above that.
static inline bool rcclDdaNanPerPeer(size_t slotBytes) {
  static const size_t max = [] {
    const char* s = ncclGetEnv("RCCL_DDA_NAN_PEER_MAX");
    return s ? (size_t)strtoull(s, nullptr, 0) : (size_t)128 << 10;
  }();
  return slotBytes <= max;
}

// The copies' grid; units per peer.
static inline std::pair<dim3, dim3> rcclDdaNanCopyGeometry(const ncclComm* comm, size_t units, bool perPeer) {
  return perPeer ? dda::nan::peerGeometry(units, rcclDdaNanMaxBlocks(comm), rcclDdaNanThreads(256))
                 : rcclDdaNanGeometry(comm, units);
}

// RCCL_DDA_NAN_TDM runs all-peer kernels (not the per-peer or registered ones) through the gfx1250
// tensor data mover when the per-source block is at least RCCL_DDA_NAN_TDM_MIN bytes (default
// 256 KiB): 1 for AllReduce and ReduceScatter, 2 for AllGather and AlltoAll as well. The copies
// gain nothing (0.85-1.0x at 2-32 MiB) while the reductions do (1.06-1.21x at 4-32 MiB). Off by
// default; builds without -DENABLE_TDM_NAN=ON ignore it.
static inline bool rcclDdaNanTdm(const ncclComm* comm, size_t slotBytes, bool copy = false) {
#if ENABLE_TDM_NAN
  static const int mode = [] {
    const char* s = ncclGetEnv("RCCL_DDA_NAN_TDM");
    return s ? atoi(s) : 0;
  }();
  static const size_t min = [] {
    const char* s = ncclGetEnv("RCCL_DDA_NAN_TDM_MIN");
    return s ? (size_t)strtoull(s, nullptr, 0) : (size_t)256 << 10;
  }();
  return mode >= (copy ? 2 : 1) && slotBytes >= min && comm->archName != nullptr &&
         IsArchMatch(comm->archName, "gfx1250");
#else
  (void)comm;
  (void)slotBytes;
  (void)copy;
  return false;
#endif
}

// The TDM kernels' grid; RCCL_DDA_NAN_TDM_BLOCKS caps it (default 384: four blocks on each of the
// 96 WGPs, as many as their LDS holds).
static inline std::pair<dim3, dim3> rcclDdaNanTdmGeometry(size_t units) {
#if ENABLE_TDM_NAN
  static const int env = [] {
    const char* s = ncclGetEnv("RCCL_DDA_NAN_TDM_BLOCKS");
    return s ? atoi(s) : 0;
  }();
  return dda::nan::tdmGeometry(units, std::min(env > 0 ? env : 384, dda::nan::kEpochCells));
#else
  (void)units;
  return {dim3(1), dim3(1)};
#endif
}

static inline dda::nan::Peers rcclDdaNanPeers(const ncclComm* comm) {
  dda::nan::Peers peers;
  for (int i = 0; i < dda::nan::kRanks; i++) peers.p[i] = static_cast<v4u*>(comm->ddaNanPeers[i]);
  return peers;
}

// Units per source that an all-peer call of `units` per source stages per round
// (rcclDdaNanRoundBytes): all of them when they fit one round. Its grid, and
// whether it takes TDM, follow from the round.
static inline size_t rcclDdaNanRound(size_t units) {
  const size_t r = rcclDdaNanRoundBytes() / 16;
  return r > 0 && units > r ? r : units;
}

#if ENABLE_TDM_NAN
template <typename T, dda::nan::TdmKind K>
static inline void rcclDdaNanTdmLaunch(ncclComm* comm, const void* sendbuff, void* recvbuff, size_t units,
                                       hipEvent_t stopEvent, cudaStream_t stream) {
  const size_t round = rcclDdaNanRound(units);
  const auto gridBlock = rcclDdaNanTdmGeometry(round);
  hipExtLaunchKernelGGL((dda::nan::ddaNanTdm<T, K>), gridBlock.first, gridBlock.second, 0, stream,
                        /*startEvent=*/nullptr, stopEvent, /*flags=*/0, rcclDdaNanPeers(comm), comm->ddaNanEpochDev,
                        static_cast<const v4u*>(sendbuff), static_cast<v4u*>(recvbuff), units, round, comm->rank);
}
#endif

// Whether a NaN-flag DDA call whose largest per-source block is `slotBytes` fits:
// in one slot, or in any number of rounds when RCCL_DDA_NAN_ROUND is set.
static inline bool rcclDdaNanFits(const ncclComm* comm, size_t slotBytes) {
  return comm->ddaNanScratch != nullptr && comm->nRanks == dda::nan::kRanks && slotBytes > 0 &&
         slotBytes % 16 == 0 && (slotBytes <= dda::nan::kSlotBytes || rcclDdaNanRoundBytes() > 0);
}

// This rank's recv buffer as every rank maps it, when the user registered it
// (ncclCommRegister) and its IPC export to every peer succeeded. Registration has
// to be symmetric: a rank whose buffer is not registered takes the scratch kernel,
// which its peers' registered kernel does not interoperate with.
static inline bool rcclDdaNanRecvAddrs(ncclComm* comm, void* recvbuff, size_t bytes, size_t slotBytes,
                                       dda::nan::RegAddrs* out) {
  const size_t regMin = rcclDdaNanRegMin();
  if (regMin == 0 || slotBytes < regMin || slotBytes % 16 != 0 || comm->ddaNanScratch == nullptr ||
      comm->nRanks != dda::nan::kRanks)
    return false;
  struct ncclReg* reg = nullptr;
  bool valid = false;
  if (ncclRegFind(comm, recvbuff, bytes, &reg) != ncclSuccess || reg == nullptr) return false;
  if (ncclRegLocalIsValid(reg, &valid) != ncclSuccess || !valid) return false;
  int peerRanks[dda::nan::kRanks];
  int nPeers = 0;
  for (int r = 0; r < comm->nRanks; r++)
    if (r != comm->rank) peerRanks[nPeers++] = r;
  int regFlag = 0;
  uintptr_t offset = 0;
  uintptr_t* devAddrs = nullptr;
  if (ncclIpcLocalRegisterBuffer(comm, recvbuff, bytes, peerRanks, nPeers, NCCL_IPC_COLLECTIVE, &regFlag, &offset,
                                 &devAddrs) != ncclSuccess ||
      !regFlag || reg->regIpcAddrs.hostPeerRmtAddrs == nullptr)
    return false;
  for (int r = 0; r < dda::nan::kRanks; r++) {
    if (r == comm->rank) {
      out->p[r] = static_cast<v4u*>(recvbuff);
      continue;
    }
    const uintptr_t base = reg->regIpcAddrs.hostPeerRmtAddrs[comm->p2pCrossClique ? r : comm->rankToLocalRank[r]];
    if (base == 0) return false;
    out->p[r] = reinterpret_cast<v4u*>(base + offset);
  }
  return true;
}
