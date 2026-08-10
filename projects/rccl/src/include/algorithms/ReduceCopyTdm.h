/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Optional TDM / block-vector bulk copy in front of device reduceCopy for
 * copy-only (nSrcs == 1, nDsts == 1) transfers.
 *
 * When eligible, uses TDM (gfx1250 + RCCL_REDUCECOPY_TDM_LDS, full CTA) or a
 * worker/block vector bulk copy on aligned buffers. Falls back to per-worker
 * reduceCopy otherwise.
 *
 * SIMPLE-protocol TDM staging is separate from the DDA fabric pool
 * (meta::comms::kTdmLdsBytes): 256 KiB on MI450-class parts (320 KiB LDS/CU).
 ************************************************************************/

#pragma once

#include "algorithms/CollCommonTdm.h"
#include "common_kernel.h"

namespace rccl {

// 256 KiB block staging: 8 issuing warps x 2 windows x 16 KiB/window.
constexpr uint32_t kReduceCopyTdmLdsBytes = 256 * 1024;
constexpr uint32_t kReduceCopyTdmIssueWarps = meta::comms::kTdmIssueWarps;
constexpr uint32_t kReduceCopyTdmWindowBytes = kReduceCopyTdmLdsBytes / (2 * kReduceCopyTdmIssueWarps);

static_assert(kReduceCopyTdmWindowBytes % meta::comms::kTdmRowBytes == 0,
              "reduceCopy TDM window must be a whole number of rows");
static_assert(kReduceCopyTdmLdsBytes <= meta::comms::kTdmMaxLdsPerBlockBytes,
              "reduceCopy TDM staging exceeds MI450 LDS limit");
static_assert(2 * kReduceCopyTdmIssueWarps * kReduceCopyTdmWindowBytes == kReduceCopyTdmLdsBytes,
              "reduceCopy TDM LDS layout mismatch");

#if TDM_SUPPORTED && defined(RCCL_REDUCECOPY_TDM_LDS)
#ifdef RCCL_DEVICE_LINKER
__shared__ __align__(meta::comms::kTdmRowBytes) uint8_t rcclReduceCopyTdmLds[kReduceCopyTdmLdsBytes];
#else
extern __shared__ __align__(meta::comms::kTdmRowBytes) uint8_t rcclReduceCopyTdmLds[kReduceCopyTdmLdsBytes];
#endif

__device__ __forceinline__ uint8_t* reduceCopyTdmLdsPtr() {
  return rcclReduceCopyTdmLds;
}
#else
__device__ __forceinline__ uint8_t* reduceCopyTdmLdsPtr() {
  return nullptr;
}
#endif

constexpr size_t kReduceCopyTdmMinBytes = 4096;

__device__ __forceinline__ uint8_t* reduceCopyTdmWindow(uint8_t* lds, uint32_t which) {
  const uint32_t slot = (threadIdx.x / meta::comms::kTdmLanes) % kReduceCopyTdmIssueWarps;
  return lds + (slot * 2 + which) * kReduceCopyTdmWindowBytes;
}

template <CachePolicy cp = meta::comms::kTdmCachePolicy>
__device__ inline void reduceCopyTdmCopyRangeBytes(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
                                                   size_t nbytes, uint8_t* window0, uint8_t* window1,
                                                   const meta::comms::TdmWarpTile& t) {
#if TDM_SUPPORTED
  if (!t.issuer) {
    return;
  }
  const size_t stride = (size_t)t.nGWarps * kReduceCopyTdmWindowBytes;
  size_t off = (size_t)t.gWarp * kReduceCopyTdmWindowBytes;
  if (off >= nbytes) {
    return;
  }

  uint32_t n = (nbytes - off) < kReduceCopyTdmWindowBytes ? (uint32_t)(nbytes - off) : kReduceCopyTdmWindowBytes;
  tdm::asyncLoadToLDS<SyncPolicy::Async, cp, true>(src + off, window0, n);

  for (uint32_t parity = 0; off < nbytes; ++parity) {
    uint8_t* cur = (parity & 1) ? window1 : window0;
    uint8_t* nxt = (parity & 1) ? window0 : window1;
    const size_t nextOff = off + stride;

    tdm::tdmWait();
    tdm::asyncStoreFromLDS<SyncPolicy::Async, cp, true>(cur, dst + off, n);

    if (nextOff < nbytes) {
      n = (nbytes - nextOff) < kReduceCopyTdmWindowBytes ? (uint32_t)(nbytes - nextOff) : kReduceCopyTdmWindowBytes;
      tdm::asyncLoadToLDS<SyncPolicy::Async, cp, true>(src + nextOff, nxt, n);
    }
    off = nextOff;
  }
  tdm::tdmWait();
#else
  (void)src;
  (void)dst;
  (void)nbytes;
  (void)window0;
  (void)window1;
  (void)t;
#endif
}

__device__ __forceinline__ void vecCopyRangeWorkers(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
                                                    size_t nbytes, int thread, int nWorkers) {
  const size_t nVec = nbytes / sizeof(uint4);
  const uint4* s = reinterpret_cast<const uint4*>(src);
  uint4* d = reinterpret_cast<uint4*>(dst);
  for (size_t i = (size_t)thread; i < nVec; i += (size_t)nWorkers) {
    d[i] = s[i];
  }
}

__device__ __forceinline__ bool reduceCopyBulkCopyEligible(const void* src, const void* dst, size_t nbytes, int nSrcs,
                                                           int nDsts, bool postOp, int multimemSrcs, int multimemDsts,
                                                           int pipeline, bool useAcc) {
  if (nSrcs != 1 || nDsts != 1 || postOp) return false;
  if (multimemSrcs != 0 || multimemDsts != 0 || pipeline != 0 || useAcc) return false;
  if (nbytes < kReduceCopyTdmMinBytes) return false;
#if TDM_SUPPORTED
  return meta::comms::tdmPairAligned(src, dst);
#else
  (void)src;
  (void)dst;
  return (nbytes % sizeof(uint4)) == 0 && (((uintptr_t)src | (uintptr_t)dst) & (sizeof(uint4) - 1)) == 0;
#endif
}

// Block-cooperative bulk copy (full CTA). Every thread in the block must enter
// with the same (src, dst, nbytes). Non-issuing warps in the TDM path return
// early and rejoin at __syncthreads().
__device__ __forceinline__ void reduceCopyBulkCopyBlock(const void* src, void* dst, size_t nbytes,
                                                        uint8_t* tdmLds = nullptr) {
#if TDM_SUPPORTED
  if (tdmLds != nullptr && meta::comms::tdmPairAligned(src, dst)) {
    uint8_t* window0 = reduceCopyTdmWindow(tdmLds, 0);
    uint8_t* window1 = reduceCopyTdmWindow(tdmLds, 1);
    meta::comms::TdmWarpTile tile = meta::comms::tdmWarpTile();
    reduceCopyTdmCopyRangeBytes<meta::comms::kTdmCachePolicy>((const uint8_t*)src, (uint8_t*)dst, nbytes, window0,
                                                              window1, tile);
    __syncthreads();
    return;
  }
#else
  (void)tdmLds;
#endif
  meta::comms::vecCopyRangeBytes((const uint8_t*)src, (uint8_t*)dst, nbytes);
}

// Bulk copy among worker threads only (P2P sendrecv uses a warp subset).
__device__ __forceinline__ void reduceCopyBulkCopyWorkers(const void* src, void* dst, size_t nbytes, int thread,
                                                          int nWorkers) {
  if (thread >= nWorkers) return;
  vecCopyRangeWorkers((const uint8_t*)src, (uint8_t*)dst, nbytes, thread, nWorkers);
}

// Copy-only reduceCopy replacement. When bulk-eligible, uses TDM (full CTA) or
// worker vector bulk; otherwise delegates to reduceCopy unchanged.
template <int Unroll, int useAcc, typename RedFn, typename T, int MultimemSrcs, int MinSrcs, int MaxSrcs,
          int MultimemDsts, int MinDsts, int MaxDsts, int PreOpSrcs, int Pipeline = 0, typename IntBytes>
__device__ __forceinline__ void reduceCopyOrTdm(int thread, int nWorkers, uint64_t redArg, bool postOp, int nSrcs,
                                                void* src, void* dst, IntBytes nElts, uint8_t* tdmLds = nullptr,
                                                bool allowBulk = true) {
  const size_t nbytes = (size_t)nElts * sizeof(T);
  const bool bulkEligible =
    allowBulk &&
    reduceCopyBulkCopyEligible(src, dst, nbytes, nSrcs, 1, postOp, MultimemSrcs, MultimemDsts, Pipeline, useAcc != 0);

  if (bulkEligible) {
    if (nWorkers == blockDim.x) {
      uint8_t* lds = (tdmLds != nullptr) ? tdmLds : reduceCopyTdmLdsPtr();
      reduceCopyBulkCopyBlock(src, dst, nbytes, lds);
      return;
    }
    reduceCopyBulkCopyWorkers(src, dst, nbytes, thread, nWorkers);
    return;
  }

  if (thread >= nWorkers) return;
  void* srcPtrs[1] = {src};
  void* dstPtrs[1] = {dst};
  reduceCopy<Unroll, useAcc, RedFn, T, MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs,
             Pipeline, IntBytes>(thread, nWorkers, redArg, postOp, 1, srcPtrs, 1, dstPtrs, nElts);
}

// reduceCopy call sites that pass pointer arrays: use bulk/TDM when runtime
// nSrcs/nDsts are both 1 and the compile-time path is copy-only.
template <int Unroll, int useAcc, typename RedFn, typename T, int MultimemSrcs, int MinSrcs, int MaxSrcs,
          int MultimemDsts, int MinDsts, int MaxDsts, int PreOpSrcs, int Pipeline = 0, typename IntBytes>
__device__ __forceinline__ void reduceCopyDispatchOrTdm(int thread, int nWorkers, uint64_t redArg, bool postOp,
                                                        int nSrcs, void** srcPtrs, int nDsts, void** dstPtrs,
                                                        IntBytes nElts, void* accPtr = nullptr,
                                                        bool allowBulk = true) {
  if (nSrcs == 1 && nDsts == 1 && MultimemSrcs == 0 && MultimemDsts == 0 && Pipeline == 0 && !(useAcc != 0)) {
    reduceCopyOrTdm<Unroll, useAcc, RedFn, T, MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts,
                    PreOpSrcs, Pipeline>(thread, nWorkers, redArg, postOp, 1, srcPtrs[0], dstPtrs[0], nElts, nullptr,
                                         allowBulk);
    return;
  }

  if (thread >= nWorkers) return;
  reduceCopy<Unroll, useAcc, RedFn, T, MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs,
             Pipeline, IntBytes>(thread, nWorkers, redArg, postOp, nSrcs, srcPtrs, nDsts, dstPtrs, nElts, accPtr);
}

} // namespace rccl
