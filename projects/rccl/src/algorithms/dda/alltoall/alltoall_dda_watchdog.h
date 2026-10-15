/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * Abortable, optionally bounded spin waits for the DDA fabric AllToAll kernels.
 *
 * Every device spin loop of the A2A paths waits on a flag written by another block or another
 * rank. If a peer never arrives (a rank skipped the call, a host failure after peers launched, a
 * residency bug) the kernel would spin until something stops it.
 *
 * Abort (always on): every 8192 spins, and only once a wait has lasted >= 1 ms, a wait reads the
 * communicator's host-pinned abort flag (comm->abortFlagDev, set by ncclCommAbort), as the native
 * RCCL primitives do every NCCL_SPINS_BEFORE_CHECK_ABORT spins. A set flag ends the wait and
 * poisons the communicator's device error word, so every other wait ends at its next check; the
 * kernel then runs to completion on its own (results invalid) and ncclCommAbort returns within
 * ~1 ms of the flag instead of hanging. The 1 ms gate keeps the host-memory read off every short
 * wait (latency-bound LL/LL128 kernels).
 *
 * Timeout (opt-in, RCCL_DDA_A2A_TIMEOUT seconds, default 0 = never): as in NCCL, the device never
 * declares a slow peer dead on its own by default. Rank skew of a minute or more is normal in
 * training (data loading, checkpointing), and a default timeout would fail healthy jobs; frameworks
 * detect hangs on the host (e.g. the PyTorch NCCL watchdog) and call ncclCommAbort, which the abort
 * poll above serves. A positive value makes a wait that exceeds it "trip": the first trip on the
 * communicator stores its wait-site id in the per-comm error word (device memory) and in its host
 * mirror (host-pinned); every other wait of the communicator sees the device word at its next
 * check and gives up too. The host reports a trip as ncclRemoteError from ncclCommGetAsyncError
 * and from every later ncclAlltoAll on that communicator. Test scripts set RCCL_DDA_A2A_TIMEOUT=20.
 *
 * Poison: every A2A kernel checks the device error word at entry and returns without moving any
 * data once it is set (trip or abort), so a graph replay after a failure cannot match stale flags
 * of a desynchronised epoch.
 *
 * Quiet: no device printf anywhere (a printf adds the hostcall buffer to every launch of the
 * kernel and, measured on the LL kernel, 3x the VGPRs and 4x the code); the host names the site.
 *
 * Cost with timeout 0: one add + mask test per spin; the clock and the device error word are read
 * every 1024 spins, the abort flag at most every 8192 spins and only after the first 1 ms of a wait.
 *
 * Kernel-argument cost: a kernel receives ONE pointer (DdaA2aWdCtx) to the per-comm DdaA2aErr, which
 * also holds the constant configuration (host mirror, abort flag, tick counts) read only on the slow
 * path. A by-value aggregate argument stops LLVM's kernarg preloading at that argument, so every
 * later argument and the hidden grid-size arguments come from kernarg memory: one extra dependent
 * memory round trip at kernel entry, freshly written by the host on every eager launch (measured
 * +0.5..1.4 us per call on the small LL / LL128 AllToAll, R03). Keep the argument a scalar pointer.
 * See LICENSE.txt for license information.
 ************************************************************************/

#pragma once

#include <cstdint>

#include "algorithms/dda/alltoall/dda_alltoall.h"  // kDdaA2aErr* (host view of the site ids)

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_PLATFORM_HCC__)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

namespace dda::common {

constexpr uint32_t kDdaA2aWdMask = 1023;       // clock + error word every 1024 spins
constexpr uint32_t kDdaA2aWdAbortMask = 8191;  // abort flag every 8192 spins (after >= abortTicks)

// Wait-site ids (value of the error word; the host names them, see ncclDdaA2aWdSiteName).
enum : uint32_t {
  kDdaWdPushEntry = 1,
  kDdaWdPushReady = 3,
  kDdaWdPushCuEntry = 4,
  kDdaWdPushCuReady = 5,
  kDdaWdDirectEntry = 11,
  kDdaWdDirectDone = 12,
  kDdaWdHybridConsumed = 13,  // Direct with staged pairs: receiver has not copied a round out
  kDdaWdHybridDone = 14,      // Direct with staged pairs: sender has not staged a round
  kDdaWdLL = 21,
  kDdaWdLL128V2 = 23,
  // Two push/direct kernels of one communicator in flight at once (ordering violation by the caller).
  kDdaWdConcurrent = 30,  // == kDdaA2aErrConcurrent (dda_alltoall.h)
  // A wait observed the abort flag (ncclCommAbort, or a shrink-abort / a sibling sharing the flag).
  // Mirrored to the host so later AllToAlls on a comm that outlives the flag are refused, but not
  // reported by ncclCommGetAsyncError (the caller aborted).
  kDdaWdAborted = 0xAB,  // == kDdaA2aErrAborted (dda_alltoall.h)
};

static_assert(kDdaWdConcurrent == kDdaA2aErrConcurrent && kDdaWdAborted == kDdaA2aErrAborted,
              "host and device AllToAll error site ids must match");

// Per-communicator watchdog state (device memory, written once at fabric init). The first two words
// change at run time; the rest is constant configuration, read only on the slow path.
struct DdaA2aErr {
  uint32_t site;      // 0 = healthy; else the first trip's wait-site id (or kDdaWdAborted)
  uint32_t inflight;  // push/direct kernels of this comm between entry and their last block
  uint32_t pad[2];
  uint32_t* errHost;          // host-pinned mirror of site
  const uint32_t* abortFlag;  // comm->abortFlagDev (host-pinned); may be null
  uint64_t ticks;             // opt-in timeout in wall_clock64 ticks; 0 = never trip
  uint64_t abortTicks;        // ~1 ms: a wait reads the abort flag only once it is this old
};

// Passed in every A2A kernel's arguments: a plain pointer (never null on a launch), see the header
// comment on kernel-argument cost.
using DdaA2aWdCtx = DdaA2aErr*;

__device__ __forceinline__ void ddaA2aWdTrip(DdaA2aWdCtx c, uint32_t site) {
  uint32_t expected = 0;
  if (__scoped_atomic_compare_exchange_n(&c->site, &expected, site, /*weak=*/false, __ATOMIC_RELAXED,
                                         __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE)) {
    uint32_t* const host = c->errHost;
    if (host != nullptr) __scoped_atomic_store_n(host, site, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  }
}

// Kernel-entry check (plain load: the word is written by earlier kernels, or by a trip of this one
// that entry need not see). True: the communicator failed (or saw an abort) earlier; move no data.
__device__ __forceinline__ bool ddaA2aPoisoned(DdaA2aWdCtx c) { return c->site != 0; }

// Kernel entry of a push/direct kernel, thread 0 of each block, in two halves so the round trips
// overlap the caller's own entry loads instead of following them:
//   const DdaA2aEntry en = ddaA2aEnterIssue(a.wd);   // FIRST, before the epoch loads / tickets
//   ... the kernel's own entry loads ...
//   sRun = ddaA2aEnterDecide(a.wd, en);              // after them
// Decide returns false = the block must return without touching anything (communicator poisoned,
// or block 0 found another push/direct kernel of this communicator still in flight -- the caller
// broke stream ordering; that trips kDdaWdConcurrent). The kernel's last block calls ddaA2aLeave (as
// does block 0 of a tuning-build launch that skips the last-block path).
// Issue has no branch on a loaded value, so the poison load, block 0's increment and the caller's
// loads are all in flight together (one round trip). The increment is an atomic inc (uinc_wrap with
// limit ~0u, so identical to fetch_add(1) for every value), not fetch_add: LLVM's atomic optimizer
// rewrites a uniform-address fetch_add into one lane's atomic plus a readfirstlane of the result
// right at the branch merge, which waits for it before any later load issues (measured in the
// gfx1250 ISA, R03).
// Detection is a backstop, not a guard: blocks of the second kernel that started before block 0's
// trip landed may already have moved data; the call (and the comm) is reported failed.
struct DdaA2aEntry {
  uint32_t site;
  uint32_t prev;
};
__device__ __forceinline__ DdaA2aEntry ddaA2aEnterIssue(DdaA2aWdCtx c) {
  const bool block0 = blockIdx.x == 0 && blockIdx.y == 0;
  DdaA2aEntry r;
  r.site = c->site;
  r.prev = block0 ? __builtin_amdgcn_atomic_inc32(&c->inflight, ~0u, __ATOMIC_RELAXED, "agent") : 0u;
  return r;
}
__device__ __forceinline__ bool ddaA2aEnterDecide(DdaA2aWdCtx c, const DdaA2aEntry& r) {
  // Opaque here, so LLVM cannot test the increment's result inside block 0's branch in Issue (that
  // test is a wait on the atomic before the caller's loads go out).
  uint32_t prev = r.prev;
  asm volatile("" : "+v"(prev));
  if (r.site != 0) return false;
  if (prev != 0) {
    ddaA2aWdTrip(c, kDdaWdConcurrent);
    return false;
  }
  return true;
}
__device__ __forceinline__ void ddaA2aLeave(DdaA2aWdCtx c) {
  (void)__scoped_atomic_fetch_sub(&c->inflight, 1u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
}

// Slow path, every 1024 spins: start the clock on the first call; give up when the error word is
// set (another wait of this comm tripped or saw an abort), when the abort flag is set (read on every
// 8th call once the wait is >= abortTicks old), or when the opt-in timeout expired.
__device__ __forceinline__ bool ddaA2aWdCheck(uint64_t& t0, uint32_t it, DdaA2aWdCtx c, uint32_t site) {
  const uint64_t now = (uint64_t)wall_clock64();
  if (t0 == 0) {  // first check of this wait: start the clock
    t0 = now ? now : 1;
    return false;
  }
  if (__scoped_atomic_load_n(&c->site, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) != 0) return true;
  const uint64_t elapsed = now - t0;
  if ((it & kDdaA2aWdAbortMask) == 0 && elapsed >= c->abortTicks) {
    const uint32_t* const abortFlag = c->abortFlag;
    if (abortFlag != nullptr && __scoped_atomic_load_n(abortFlag, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM) != 0) {
      ddaA2aWdTrip(c, kDdaWdAborted);
      return true;
    }
  }
  const uint64_t ticks = c->ticks;
  if (ticks != 0 && elapsed > ticks) {
    ddaA2aWdTrip(c, site);
    return true;
  }
  return false;
}

// One instance per wait loop:  DdaA2aWd wd(a.wd); while (!cond) { sleep; if (wd.expired(site)) break; }
struct DdaA2aWd {
  DdaA2aWdCtx c;
  uint64_t t0 = 0;
  uint32_t it = 0;
  __device__ __forceinline__ explicit DdaA2aWd(DdaA2aWdCtx ctx) : c(ctx) {}
  __device__ __forceinline__ bool expired(uint32_t site) {
    if (((++it) & kDdaA2aWdMask) != 0) return false;
    return ddaA2aWdCheck(t0, it, c, site);
  }
  // Warp-uniform variant for loops whose exit must be uniform across the wave (the counter is
  // uniform, so the check runs on the same iteration in every lane).
  __device__ __forceinline__ bool expiredUniform(uint32_t site) {
    if (((++it) & kDdaA2aWdMask) != 0) return false;
    return __any(ddaA2aWdCheck(t0, it, c, site));
  }
};

} // namespace dda::common

// Host: the communicator's watchdog state pointer for a launch (fabric_init.cu).
struct ncclComm;
dda::common::DdaA2aWdCtx ncclDdaA2aWdCtx(const ncclComm* comm);
