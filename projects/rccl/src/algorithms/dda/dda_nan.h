/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information.
 ************************************************************************/

// Host side of the NaN-flag DDA kernels (algorithms/dda/device/CollCommon_nan.h).

#pragma once

#include "algorithms/dda/device/CollCommon_nan.h"
#include "comm.h"
#include "p2p.h"
#include "rccl_common.h"
#include "register.h"
#include "register_inline.h"

#include <cstdlib>

// RCCL_DDA_NAN_MAX_BLOCKS caps the grid; it cannot exceed the epoch cells. The
// default is per path: on gfx1250 fabric the large calls scale with the grid up
// to about 256 blocks; the IPC path was tuned at 64.
static inline int rcclDdaNanMaxBlocks(const ncclComm* comm) {
  static const int env = [] {
    const char* s = ncclGetEnv("RCCL_DDA_NAN_MAX_BLOCKS");
    return s ? atoi(s) : 0;
  }();
  const int v = env > 0 ? env : (comm->ddaNanFabricMemHandler != nullptr ? 256 : 64);
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

static inline dda::nan::Peers rcclDdaNanPeers(const ncclComm* comm) {
  dda::nan::Peers peers;
  for (int i = 0; i < dda::nan::kRanks; i++) peers.p[i] = static_cast<v4u*>(comm->ddaNanPeers[i]);
  return peers;
}

// Whether a NaN-flag DDA call whose largest per-source block is `slotBytes` fits.
static inline bool rcclDdaNanFits(const ncclComm* comm, size_t slotBytes) {
  return comm->ddaNanScratch != nullptr && comm->nRanks == dda::nan::kRanks && slotBytes > 0 &&
         slotBytes % 16 == 0 && slotBytes <= dda::nan::kSlotBytes;
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
