/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COMMON_KERNEL_H_
#define NCCL_COMMON_KERNEL_H_

#include "device.h"
#include "op128.h"
#include "reduce_kernel.h"
#include <cstdio>
#include <cstdint>

#include <hip/hip_runtime.h>

#define __syncwarp()

// Define min for ssize_t
inline __device__ int min(int a, ssize_t b) { return (a < b) ? a : b; }

inline __device__ int loadInt(int* ptr) {
  int v;
  v = __atomic_load_n(ptr, __ATOMIC_RELAXED);
  return v;
}

template<typename RedFn, typename T, int Unroll, int BytePerPack,
         int MultimemSrcs, int MinSrcs, int MaxSrcs,
         int MultimemDsts, int MinDsts, int MaxDsts, int PreOpSrcs,
         typename IntBytes, typename SrcPtrFn, typename DstPtrFn>
__device__ __forceinline__ static void reduceCopyPacks(
    int nThreads, int &thread,
    uint64_t redArg, uint64_t *preOpArgs, bool postOp,
    int nSrcs, SrcPtrFn const &srcPtrFn, int nDsts, DstPtrFn const &dstPtrFn,
    IntBytes &nBytesBehind, IntBytes &nBytesAhead
  ) {
  static_assert(std::is_signed<IntBytes>::value, "IntBytes must be a signed integral type.");
  //if (BytePerPack == 0) __trap();

  // A hunk is the amount of contiguous data a warp consumes per loop iteration
  // assuming all threads partake.
  constexpr int BytePerHunk = Unroll*WARP_SIZE*BytePerPack;
  int nWarps = nThreads/WARP_SIZE;
  int warp = thread/WARP_SIZE;
  int lane = thread%WARP_SIZE;

  // This thread's initial position.
  IntBytes threadBytesBehind = nBytesBehind + (warp*BytePerHunk + lane*BytePerPack);
  IntBytes threadBytesAhead = nBytesAhead - (warp*BytePerHunk + lane*BytePerPack);
  // Number of hunks to be consumed over all warps.
  IntBytes nHunksAhead = nBytesAhead/(BytePerHunk + !BytePerHunk);
  // Advance collective position.
  nBytesBehind += nHunksAhead*BytePerHunk;
  nBytesAhead -= nHunksAhead*BytePerHunk;
  if (Unroll==1 && BytePerPack <= nBytesAhead) {
    // Only Unroll=1 can do partial hunks (where not all threads partake).
    nHunksAhead += 1;
    nBytesBehind += nBytesAhead - (nBytesAhead%(BytePerPack + !BytePerPack));
    nBytesAhead = nBytesAhead%(BytePerPack + !BytePerPack);
  }
  nHunksAhead -= warp;

  RedFn redFn(redArg);
  uintptr_t minSrcs[MinSrcs + !MinSrcs];
  uintptr_t minDsts[MinDsts + !MinDsts];
  #pragma unroll
  for (int s=0; s < MinSrcs; s++) {
    minSrcs[s] = cvta_to_global(srcPtrFn(s)) + threadBytesBehind;
  }

  #pragma unroll
  for (int d=0; d < MinDsts; d++) {
    // Yes, for some template arguments this code will be unreachable.  That's fine.
    // coverity[dead_error_line]
    minDsts[d] = cvta_to_global(dstPtrFn(d)) + threadBytesBehind;
  }

  // We dictate loop termination condition according to whether partial hunks
  // can be handled or not.
  while (Unroll==1 ? (BytePerPack <= threadBytesAhead) : (0 < nHunksAhead)) {
    BytePack<BytePerPack> acc[Unroll];

    // minSrcs[0] cannot be nullptr so we always process it
    { RedFn preFn(0 < PreOpSrcs ? preOpArgs[0] : 0);
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        if (0 < MultimemSrcs) {
          // applyLoadMultimem uses relaxed semantics for same reason we use volatile below.
          acc[u] = applyLoadMultimem<RedFn, BytePerPack>(redFn, minSrcs[0]);
        } else {
          // Use volatile loads in case credits are polled for with volatile (instead of acquire).
          acc[u] = ld_volatile_global<BytePerPack>(minSrcs[0]);
          if (0 < PreOpSrcs) acc[u] = applyPreOp(preFn, acc[u]);
        }
        minSrcs[0] += WARP_SIZE*BytePerPack;
      }
    }

    #pragma unroll Unroll
    for (int s=1; s < MinSrcs; s++) {
      // Yes, for some template arguments this code will be unreachable.  That's fine.
      // coverity[dead_error_begin]
      BytePack<BytePerPack> tmp[Unroll];
      // coverity[dead_error_line]
      RedFn preFn(s < PreOpSrcs ? preOpArgs[s] : 0);
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        if (s < MultimemSrcs) {
          // applyLoadMultimem uses relaxed semantics for same reason we use volatile below.
          // coverity[dead_error_line]
          tmp[u] = applyLoadMultimem<RedFn, BytePerPack>(redFn, minSrcs[s]);
        } else {
          // Use volatile loads in case credits are polled for with volatile (instead of acquire).
          tmp[u] = ld_volatile_global<BytePerPack>(minSrcs[s]);
        }
        minSrcs[s] += WARP_SIZE*BytePerPack;
      }
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        // coverity[dead_error_line]
        if (s < PreOpSrcs) tmp[u] = applyPreOp(preFn, tmp[u]);
        acc[u] = applyReduce(redFn, acc[u], tmp[u]);
      }
    }

    for (int s=MinSrcs; (MinSrcs < MaxSrcs) && (s < MaxSrcs) && (s < nSrcs); s++) {
      uintptr_t src = cvta_to_global(srcPtrFn(s)) + threadBytesBehind;
      BytePack<BytePerPack> tmp[Unroll];
      // Yes, for some template arguments this code will be unreachable.  That's fine.
      // coverity[dead_error_line]
      RedFn preFn(s < PreOpSrcs ? preOpArgs[s] : 0);
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        // Use volatile loads in case credits are polled for with volatile (instead of acquire).
        tmp[u] = ld_volatile_global<BytePerPack>(src);
        src += WARP_SIZE*BytePerPack;
      }
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        // Yes, for some template arguments this code will be unreachable.  That's fine.
        // coverity[dead_error_line]
        if (s < PreOpSrcs) tmp[u] = applyPreOp(preFn, tmp[u]);
        acc[u] = applyReduce(redFn, acc[u], tmp[u]);
      }
    }

    if (postOp) {
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++)
        acc[u] = applyPostOp(redFn, acc[u]);
    }

    #pragma unroll Unroll
    for (int d=0; d < MinDsts; d++) {
      #pragma unroll Unroll
      // Yes, for some template arguments this code will be unreachable.  That's fine.
      // coverity[dead_error_begin]
      for (int u=0; u < Unroll; u++) {
        // coverity[dead_error_condition]
        if (d < MultimemDsts) {
          multimem_st_global(minDsts[d], acc[u]);
        } else {
          st_global<BytePerPack>(minDsts[d], acc[u]);
        }
        minDsts[d] += WARP_SIZE*BytePerPack;
      }
    }
    for (int d=MinDsts; (MinDsts < MaxDsts) && (d < MaxDsts) && (d < nDsts); d++) {
      uintptr_t dstPtr = cvta_to_global(dstPtrFn(d));
      uintptr_t dst = dstPtr + threadBytesBehind;
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        st_global<BytePerPack>(dst, acc[u]);
        dst += WARP_SIZE*BytePerPack;
      }
    }

    nWarps = nThreads/WARP_SIZE;
    #pragma unroll
    for (int s=0; s < MinSrcs; s++) {
      minSrcs[s] += (nWarps-1)*BytePerHunk;
    }
    #pragma unroll
    // Yes, for some template arguments this code will be unreachable.  That's fine.
    // coverity[dead_error_line]
    for (int d=0; d < MinDsts; d++) {
      minDsts[d] += (nWarps-1)*BytePerHunk;
    }
    threadBytesBehind += nWarps*BytePerHunk;
    threadBytesAhead -= nWarps*BytePerHunk;
    nHunksAhead -= nWarps;
  }

  nWarps = nThreads/WARP_SIZE;
  warp = thread/WARP_SIZE;
  lane = thread%WARP_SIZE;
  // The last loop iteration could have been partial, i.e. not taken by all
  // threads. The threads that weren't included need an extra subtraction to
  // make the value warp uniform.
  if (Unroll==1 && nHunksAhead > 0) nHunksAhead -= nWarps;
  // Rotate warps so the warp which got the least work here will be warp 0.
  // This effectively assigns: warp = (warp-nHunks+nWarps)%nWarps;
  warp = -nHunksAhead;
  thread = warp*WARP_SIZE + lane;
}

template <typename RedFn, typename SrcPtrFn, typename IntBytes, int MultimemSrcs, int MinSrcs, int MaxSrcs, int PreOpSrcs, int Unroll, int BytePerPack>
__device__ __forceinline__ void loadSources(
  const RedFn& redFn,
  const SrcPtrFn& srcPtrFn,
  IntBytes& globalOffset,
  uintptr_t* minSrcs,
  uint64_t *preOpArgs,
  BytePack<BytePerPack> buff[MaxSrcs + !MaxSrcs][Unroll],
  int nSrcs
) {
  #pragma unroll Unroll
  for (int s = 0; s < MinSrcs; s++) {
    RedFn preFn(s < PreOpSrcs ? preOpArgs[s] : 0);
    #pragma unroll Unroll
    for (int u = 0; u < Unroll; u++) {
      if (s < MultimemSrcs) {
        buff[s][u] = applyLoadMultimem<RedFn, BytePerPack>(redFn, minSrcs[s]);
      } else {
        buff[s][u] = ld_volatile_global<BytePerPack>(minSrcs[s]);
      }
      minSrcs[s] += WARP_SIZE * BytePerPack;
    }
  }
  for (int s = MinSrcs; (MinSrcs < MaxSrcs) && (s < MaxSrcs) && (s < nSrcs); s++) {
    uintptr_t src = cvta_to_global(srcPtrFn(s)) + globalOffset;
    #pragma unroll Unroll
    for (int u = 0; u < Unroll; u++) {
      buff[s][u] = ld_volatile_global<BytePerPack>(src);
      src += WARP_SIZE * BytePerPack;
    }
  }
}

template <typename RedFn, typename DstPtrFn, typename IntBytes, int MultimemDsts, int MinSrcs, int MaxSrcs, int MinDsts, int MaxDsts, int PreOpSrcs, int Unroll, int BytePerPack>
  __device__ __forceinline__ void reduceAndStore(
  RedFn redFn, uint64_t *preOpArgs, BytePack<BytePerPack> buff[MaxSrcs + !MaxSrcs][Unroll],
  uintptr_t *minDsts, bool postOp, int nDsts, DstPtrFn const &dstPtrFn, IntBytes tailThreadBytesBehind, int nSrcs) {
  for (int s = 0; s < MinSrcs; s++) {
    RedFn preFn(s < PreOpSrcs ? preOpArgs[s] : 0);
    #pragma unroll Unroll
    for (int u = 0; u < Unroll; u++) {
      if (s < PreOpSrcs) buff[s][u] = applyPreOp(preFn, buff[s][u]);
      if (s > 0) buff[0][u] = applyReduce(redFn, buff[0][u], buff[s][u]);
    }
  }
  for (int s = MinSrcs; (MinSrcs < MaxSrcs) && (s < MaxSrcs) && (s < nSrcs); s++) {
    RedFn preFn(s < PreOpSrcs ? preOpArgs[s] : 0);
    #pragma unroll Unroll
    for (int u = 0; u < Unroll; u++) {
      if (s < PreOpSrcs) buff[s][u] = applyPreOp(preFn, buff[s][u]);
      buff[0][u] = applyReduce(redFn, buff[0][u], buff[s][u]);
    }
  }
  if (postOp) {
    #pragma unroll Unroll
    for (int u = 0; u < Unroll; u++)
      buff[0][u] = applyPostOp(redFn, buff[0][u]);
  }

  #pragma unroll Unroll
  for (int d = 0; d < MinDsts; d++) {
    #pragma unroll Unroll
    for (int u = 0; u < Unroll; u++) {
      if (d < MultimemDsts) {
        multimem_st_global(minDsts[d], buff[0][u]);
      } else {
        st_global<BytePerPack>(minDsts[d], buff[0][u]);
      }
      minDsts[d] += WARP_SIZE * BytePerPack;
    }
  }
  for (int d = MinDsts; (MinDsts < MaxDsts) && (d < MaxDsts) && (d < nDsts); d++) {
    uintptr_t dstPtr = cvta_to_global(dstPtrFn(d));
    uintptr_t dst = dstPtr + tailThreadBytesBehind;
    #pragma unroll Unroll
    for (int u = 0; u < Unroll; u++) {
      st_global<BytePerPack>(dst, buff[0][u]);
      dst += WARP_SIZE * BytePerPack;
    }
  }
}

template<typename RedFn, typename T, int Unroll, int BytePerPack,
         int MultimemSrcs, int MinSrcs, int MaxSrcs,
         int MultimemDsts, int MinDsts, int MaxDsts, int PreOpSrcs,
         typename IntBytes, typename SrcPtrFn, typename DstPtrFn>
__device__ __forceinline__ static void reduceCopyPacksPipelined(
    int nThreads, int &thread,
    uint64_t redArg, uint64_t *preOpArgs, bool postOp,
    int nSrcs, SrcPtrFn const &srcPtrFn, int nDsts, DstPtrFn const &dstPtrFn,
    IntBytes &nBytesBehind, IntBytes &nBytesAhead
  ) {
  static_assert(std::is_signed<IntBytes>::value, "IntBytes must be a signed integral type.");
  static_assert(MinSrcs <= MaxSrcs, "MinSrcs must be less than or equal to MaxSrcs.");
  //if (BytePerPack == 0) __trap();

  // A hunk is the amount of contiguous data a warp consumes per loop iteration
  // assuming all threads partake.
  constexpr int BytePerHunk = Unroll*WARP_SIZE*BytePerPack;
  int nWarps = nThreads/WARP_SIZE;
  int warp = thread/WARP_SIZE;
  int lane = thread%WARP_SIZE;

  // This thread's initial position.
  IntBytes threadBytesBehind = nBytesBehind + (warp*BytePerHunk + lane*BytePerPack);
  IntBytes threadBytesAhead = nBytesAhead - (warp*BytePerHunk + lane*BytePerPack);
  // Number of hunks to be consumed over all warps.
  IntBytes nHunksAhead = nBytesAhead/(BytePerHunk + !BytePerHunk);
  // Advance collective position.
  nBytesBehind += nHunksAhead*BytePerHunk;
  nBytesAhead -= nHunksAhead*BytePerHunk;
  if (Unroll==1 && BytePerPack <= nBytesAhead) {
    // Only Unroll=1 can do partial hunks (where not all threads partake).
    nHunksAhead += 1;
    nBytesBehind += nBytesAhead - (nBytesAhead%(BytePerPack + !BytePerPack));
    nBytesAhead = nBytesAhead%(BytePerPack + !BytePerPack);
  }
  nHunksAhead -= warp;

  RedFn redFn(redArg);
  uintptr_t minSrcs[MinSrcs + !MinSrcs];
  uintptr_t minDsts[MinDsts + !MinDsts];
  #pragma unroll
  for (int s=0; s < MinSrcs; s++) {
    minSrcs[s] = cvta_to_global(srcPtrFn(s)) + threadBytesBehind;
  }

  #pragma unroll
  for (int d=0; d < MinDsts; d++) {
    // Yes, for some template arguments this code will be unreachable.  That's fine.
    // coverity[dead_error_line]
    minDsts[d] = cvta_to_global(dstPtrFn(d)) + threadBytesBehind;
  }
  BytePack<BytePerPack> acc1[MaxSrcs + !MaxSrcs][Unroll];
  BytePack<BytePerPack> acc2[MaxSrcs + !MaxSrcs][Unroll];
  bool tailProcess = false;
  IntBytes tailThreadBytesBehind;
  // We dictate loop termination condition according to whether partial hunks
  // can be handled or not.
  while (Unroll==1 ? (BytePerPack <= threadBytesAhead) : (0 < nHunksAhead)) {

    // load sources into acc1
    loadSources<RedFn, SrcPtrFn, IntBytes, MultimemSrcs, MinSrcs, MaxSrcs, PreOpSrcs, Unroll, BytePerPack>(
      redFn, srcPtrFn, threadBytesBehind, minSrcs, preOpArgs, acc1, nSrcs
    );

    if(tailProcess) {
      reduceAndStore<RedFn, DstPtrFn, IntBytes, MultimemDsts, MinSrcs, MaxSrcs, MinDsts, MaxDsts, PreOpSrcs, Unroll, BytePerPack>(
        redFn, preOpArgs, acc2, minDsts, postOp, nDsts, dstPtrFn, tailThreadBytesBehind, nSrcs
      );

      #pragma unroll
      for (int d=0; d < MinDsts; d++) {
        minDsts[d] += (nWarps-1)*BytePerHunk;
      }
    }

    #pragma unroll
    for (int s=0; s < MinSrcs; s++) {
      minSrcs[s] += (nWarps-1)*BytePerHunk;
    }
    threadBytesAhead -= nWarps*BytePerHunk;
    nHunksAhead -= nWarps;
    tailProcess = Unroll==1 ? (BytePerPack <= threadBytesAhead) : (0 < nHunksAhead);

    tailThreadBytesBehind = threadBytesBehind;
    threadBytesBehind += nWarps*BytePerHunk;
    if(tailProcess) {
      loadSources<RedFn, SrcPtrFn, IntBytes, MultimemSrcs, MinSrcs, MaxSrcs, PreOpSrcs, Unroll, BytePerPack>(
        redFn, srcPtrFn, threadBytesBehind, minSrcs, preOpArgs, acc2, nSrcs
      );
    }
    reduceAndStore<RedFn, DstPtrFn, IntBytes, MultimemDsts, MinSrcs, MaxSrcs, MinDsts, MaxDsts, PreOpSrcs, Unroll, BytePerPack>(
      redFn, preOpArgs, acc1, minDsts, postOp, nDsts, dstPtrFn, tailThreadBytesBehind, nSrcs
    );

    if(tailProcess) {
      #pragma unroll
      for (int d=0; d < MinDsts; d++) {
        minDsts[d] += (nWarps-1)*BytePerHunk;
      }
      #pragma unroll
      for (int s=0; s < MinSrcs; s++) {
        minSrcs[s] += (nWarps-1)*BytePerHunk;
      }
      tailThreadBytesBehind = threadBytesBehind;
      threadBytesBehind += nWarps*BytePerHunk;
      threadBytesAhead -= nWarps*BytePerHunk;
      nHunksAhead -= nWarps;
    }
  }

  if(tailProcess) {
    reduceAndStore<RedFn, DstPtrFn, IntBytes, MultimemDsts, MinSrcs, MaxSrcs, MinDsts, MaxDsts, PreOpSrcs, Unroll, BytePerPack>(
      redFn, preOpArgs, acc2, minDsts, postOp, nDsts, dstPtrFn, tailThreadBytesBehind, nSrcs
    );
  }
  nWarps = nThreads/WARP_SIZE;
  warp = thread/WARP_SIZE;
  lane = thread%WARP_SIZE;
  // The last loop iteration could have been partial, i.e. not taken by all
  // threads. The threads that weren't included need an extra subtraction to
  // make the value warp uniform.
  if (Unroll==1 && nHunksAhead > 0) nHunksAhead -= nWarps;
  // Rotate warps so the warp which got the least work here will be warp 0.
  // This effectively assigns: warp = (warp-nHunks+nWarps)%nWarps;
  warp = -nHunksAhead;
  thread = warp*WARP_SIZE + lane;
}

template<typename RedFn, typename T, int Unroll, int BytePerPack,
         int MultimemSrcs, int MinSrcs, int MaxSrcs,
         int MultimemDsts, int MinDsts, int MaxDsts, int PreOpSrcs,
         typename IntBytes, typename SrcPtrFn, typename DstPtrFn, typename AccPtrFn>
__device__ __forceinline__ void reduceCopyPacksWithBias(
    int nThreads, int &thread,
    uint64_t redArg, uint64_t *preOpArgs, bool postOp,
    int nSrcs, SrcPtrFn const &srcPtrFn, int nDsts, DstPtrFn const &dstPtrFn,
    IntBytes &nBytesBehind, IntBytes &nBytesAhead, AccPtrFn const &accPtrFn
  ) {
  static_assert(std::is_signed<IntBytes>::value, "IntBytes must be a signed integral type.");
  //if (BytePerPack == 0) __trap();

  // A hunk is the amount of contiguous data a warp consumes per loop iteration
  // assuming all threads partake.
  constexpr int BytePerHunk = Unroll*WARP_SIZE*BytePerPack;
  int nWarps = nThreads/WARP_SIZE;
  int warp = thread/WARP_SIZE;
  int lane = thread%WARP_SIZE;

  // This thread's initial position.
  IntBytes threadBytesBehind = nBytesBehind + (warp*BytePerHunk + lane*BytePerPack);
  IntBytes threadBytesAhead = nBytesAhead - (warp*BytePerHunk + lane*BytePerPack);
  // Number of hunks to be consumed over all warps.
  IntBytes nHunksAhead = nBytesAhead/(BytePerHunk + !BytePerHunk);
  // Advance collective position.
  nBytesBehind += nHunksAhead*BytePerHunk;
  nBytesAhead -= nHunksAhead*BytePerHunk;
  if (Unroll==1 && BytePerPack <= nBytesAhead) {
    // Only Unroll=1 can do partial hunks (where not all threads partake).
    nHunksAhead += 1;
    nBytesBehind += nBytesAhead - (nBytesAhead%(BytePerPack + !BytePerPack));
    nBytesAhead = nBytesAhead%(BytePerPack + !BytePerPack);
  }
  nHunksAhead -= warp;

  RedFn redFn(redArg);
  uintptr_t minSrcs[MinSrcs + !MinSrcs];
  uintptr_t minDsts[MinDsts + !MinDsts];
  uintptr_t accPtr = cvta_to_global(accPtrFn()) + threadBytesBehind;
  BytePack<BytePerPack> bias[Unroll];

  #pragma unroll
  for (int s=0; s < MinSrcs; s++) {
    minSrcs[s] = cvta_to_global(srcPtrFn(s)) + threadBytesBehind;
  }

  #pragma unroll
  for (int d=0; d < MinDsts; d++) {
    // Yes, for some template arguments this code will be unreachable.  That's fine.
    // coverity[dead_error_line]
    minDsts[d] = cvta_to_global(dstPtrFn(d)) + threadBytesBehind;
  }

  // We dictate loop termination condition according to whether partial hunks
  // can be handled or not.
  while (Unroll==1 ? (BytePerPack <= threadBytesAhead) : (0 < nHunksAhead)) {
    BytePack<BytePerPack> acc[Unroll];

    // minSrcs[0] cannot be nullptr so we always process it
    { RedFn preFn(0 < PreOpSrcs ? preOpArgs[0] : 0);
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        if (0 < MultimemSrcs) {
          // applyLoadMultimem uses relaxed semantics for same reason we use volatile below.
          acc[u] = applyLoadMultimem<RedFn, BytePerPack>(redFn, minSrcs[0]);
        } else {
          // Use volatile loads in case credits are polled for with volatile (instead of acquire).
          acc[u] = ld_volatile_global<BytePerPack>(minSrcs[0]);
            // coverity[dead_error_condition]
          bias[u] = ld_volatile_global<BytePerPack>(accPtr);
          accPtr += WARP_SIZE*BytePerPack;
          if (0 < PreOpSrcs) acc[u] = applyPreOp(preFn, acc[u]);
        }
        minSrcs[0] += WARP_SIZE*BytePerPack;
      }
    }

    #pragma unroll Unroll
    for (int s=1; s < MinSrcs; s++) {
      // Yes, for some template arguments this code will be unreachable.  That's fine.
      // coverity[dead_error_begin]
      BytePack<BytePerPack> tmp[Unroll];
      // coverity[dead_error_line]
      RedFn preFn(s < PreOpSrcs ? preOpArgs[s] : 0);
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        if (s < MultimemSrcs) {
          // applyLoadMultimem uses relaxed semantics for same reason we use volatile below.
          // coverity[dead_error_line]
          tmp[u] = applyLoadMultimem<RedFn, BytePerPack>(redFn, minSrcs[s]);
        } else {
          // Use volatile loads in case credits are polled for with volatile (instead of acquire).
          tmp[u] = ld_volatile_global<BytePerPack>(minSrcs[s]);
        }
        minSrcs[s] += WARP_SIZE*BytePerPack;
      }
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        // coverity[dead_error_line]
        if (s < PreOpSrcs) tmp[u] = applyPreOp(preFn, tmp[u]);
        acc[u] = applyReduce(redFn, acc[u], tmp[u]);
      }
    }

    for (int s=MinSrcs; (MinSrcs < MaxSrcs) && (s < MaxSrcs) && (s < nSrcs); s++) {
      uintptr_t src = cvta_to_global(srcPtrFn(s)) + threadBytesBehind;
      BytePack<BytePerPack> tmp[Unroll];
      // Yes, for some template arguments this code will be unreachable.  That's fine.
      // coverity[dead_error_line]
      RedFn preFn(s < PreOpSrcs ? preOpArgs[s] : 0);
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        // Use volatile loads in case credits are polled for with volatile (instead of acquire).
        tmp[u] = ld_volatile_global<BytePerPack>(src);
        src += WARP_SIZE*BytePerPack;
      }
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        // Yes, for some template arguments this code will be unreachable.  That's fine.
        // coverity[dead_error_line]
        if (s < PreOpSrcs) tmp[u] = applyPreOp(preFn, tmp[u]);
        acc[u] = applyReduce(redFn, acc[u], tmp[u]);
      }
    }

    if (postOp) {
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++)
        acc[u] = applyPostOp(redFn, acc[u]);
    }

    #pragma unroll Unroll
    for (int d=0; d < MinDsts; d++) {
      #pragma unroll Unroll
      // Yes, for some template arguments this code will be unreachable.  That's fine.
      // coverity[dead_error_begin]
      for (int u=0; u < Unroll; u++) {
        // coverity[dead_error_condition]
        if (d < MultimemDsts) {
          multimem_st_global(minDsts[d], acc[u]);
        } else {
          if (d == 0)
            st_global<BytePerPack>(minDsts[d], applyReduce(redFn, acc[u], bias[u]));
          else
            st_global<BytePerPack>(minDsts[d], acc[u]);
        }
        minDsts[d] += WARP_SIZE*BytePerPack;
      }
    }
    for (int d=MinDsts; (MinDsts < MaxDsts) && (d < MaxDsts) && (d < nDsts); d++) {
      uintptr_t dstPtr = cvta_to_global(dstPtrFn(d));
      uintptr_t dst = dstPtr + threadBytesBehind;
      #pragma unroll Unroll
      for (int u=0; u < Unroll; u++) {
        st_global<BytePerPack>(dst, acc[u]);
        dst += WARP_SIZE*BytePerPack;
      }
    }

    nWarps = nThreads/WARP_SIZE;
    #pragma unroll
    for (int s=0; s < MinSrcs; s++) {
      minSrcs[s] += (nWarps-1)*BytePerHunk;
    }
    #pragma unroll
    // Yes, for some template arguments this code will be unreachable.  That's fine.
    // coverity[dead_error_line]
    for (int d=0; d < MinDsts; d++) {
      minDsts[d] += (nWarps-1)*BytePerHunk;
    }
    accPtr += (nWarps-1)*BytePerHunk;
    threadBytesBehind += nWarps*BytePerHunk;
    threadBytesAhead -= nWarps*BytePerHunk;
    nHunksAhead -= nWarps;
  }

  nWarps = nThreads/WARP_SIZE;
  warp = thread/WARP_SIZE;
  lane = thread%WARP_SIZE;
  // The last loop iteration could have been partial, i.e. not taken by all
  // threads. The threads that weren't included need an extra subtraction to
  // make the value warp uniform.
  if (Unroll==1 && nHunksAhead > 0) nHunksAhead -= nWarps;
  // Rotate warps so the warp which got the least work here will be warp 0.
  // This effectively assigns: warp = (warp-nHunks+nWarps)%nWarps;
  warp = -nHunksAhead;
  thread = warp*WARP_SIZE + lane;
}

template<int Unroll, int  useAcc, typename RedFn, typename T,
         int MultimemSrcs, int MinSrcs, int MaxSrcs,
         int MultimemDsts, int MinDsts, int MaxDsts, int PreOpSrcs,
         typename IntBytes, int Pipeline, typename SrcPtrFn, typename DstPtrFn, typename AccPtrFn>
__device__ __forceinline__ void reduceCopy(
    int thread, int nThreads,
    uint64_t redArg, uint64_t *preOpArgs, bool postOp,
    int nSrcs, SrcPtrFn const &srcPtrFn, int nDsts, DstPtrFn const &dstPtrFn,
    IntBytes nElts, AccPtrFn const &accPtrFn
  ) {
  static_assert(MultimemSrcs <= MinSrcs && MultimemDsts <= MinDsts, "Multimem pointers cannot exceed respective Min values.");
  //int nWarps = nThreads/WARP_SIZE;
  //int warp = thread/WARP_SIZE;
  int lane = thread%WARP_SIZE;
  // If a multimem src is present then our biggest pack size is limited to what
  // is supported for this redfn/type.
  constexpr int BigPackSize = (MultimemSrcs == 0) ? 16 : LoadMultimem_BigPackSize<RedFn>::BigPackSize;

  if (MaxDsts==0) return;
  if (MinDsts==0 && nDsts==0) return;

  IntBytes nBytesBehind = 0;
  IntBytes nBytesAhead = nElts*sizeof(T);
  //bool useAcc = accPtrFn() != nullptr;

  #if __cpp_if_constexpr
  if constexpr (BigPackSize > sizeof(T)) {
  #else
  if (BigPackSize > sizeof(T)) {
  #endif
    // Check that all pointers are BigPackSize aligned.
    bool aligned = true;
    if (lane < nSrcs) aligned &= 0 == cvta_to_global(srcPtrFn(lane)) % (BigPackSize + !BigPackSize);
    if (lane < nDsts) aligned &= 0 == cvta_to_global(dstPtrFn(lane)) % (BigPackSize + !BigPackSize);
    aligned = !(__any(!aligned));
    if (aligned) {
#if defined(__gfx90a__)
      if constexpr (useAcc)
        reduceCopyPacksWithBias<RedFn, T, ((MinSrcs > 1) ? 2 : Unroll), BigPackSize,
          MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
          (nThreads, thread, redArg, preOpArgs, postOp,
          nSrcs, srcPtrFn, nDsts, dstPtrFn, nBytesBehind, nBytesAhead, accPtrFn);
      else if constexpr (Pipeline)
        reduceCopyPacksPipelined<RedFn, T, ((MinSrcs > 1) ? 2 : Unroll), BigPackSize,
          MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
          (nThreads, thread, redArg, preOpArgs, postOp,
          nSrcs, srcPtrFn, nDsts, dstPtrFn, nBytesBehind, nBytesAhead);
      else
        reduceCopyPacks<RedFn, T, ((MinSrcs > 1) ? 2 : Unroll), BigPackSize,
          MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
          (nThreads, thread, redArg, preOpArgs, postOp,
          nSrcs, srcPtrFn, nDsts, dstPtrFn, nBytesBehind, nBytesAhead);
#else
      if constexpr (useAcc)
        reduceCopyPacksWithBias<RedFn, T, Unroll*((MinSrcs == 1 && MinDsts == 1) ? 2 : 1), BigPackSize,
          MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
          (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
          nSrcs, srcPtrFn, nDsts, dstPtrFn, /*&*/nBytesBehind, /*&*/nBytesAhead, accPtrFn);
      else if constexpr  (Pipeline)
        reduceCopyPacksPipelined<RedFn, T, Unroll*((MinSrcs == 1 && MinDsts == 1) ? 2 : 1), BigPackSize,
        MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
        (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
         nSrcs, srcPtrFn, nDsts, dstPtrFn, /*&*/nBytesBehind, /*&*/nBytesAhead);
      else
        reduceCopyPacks<RedFn, T, Unroll*((MinSrcs == 1 && MinDsts == 1) ? 2 : 1), BigPackSize,
          MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
          (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
          nSrcs, srcPtrFn, nDsts, dstPtrFn, /*&*/nBytesBehind, /*&*/nBytesAhead);
#endif
      if (nBytesAhead == 0) return;

      if constexpr (useAcc)
        reduceCopyPacksWithBias<RedFn, T, /*Unroll=*/1, BigPackSize,
          MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
          (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
          nSrcs, srcPtrFn, nDsts, dstPtrFn, /*&*/nBytesBehind, /*&*/nBytesAhead, accPtrFn);
      else if constexpr (Pipeline)
        reduceCopyPacksPipelined<RedFn, T, /*Unroll=*/1, BigPackSize,
          MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
          (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
          nSrcs, srcPtrFn, nDsts, dstPtrFn, /*&*/nBytesBehind, /*&*/nBytesAhead);
      else
        reduceCopyPacks<RedFn, T, /*Unroll=*/1, BigPackSize,
          MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
          (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
          nSrcs, srcPtrFn, nDsts, dstPtrFn, /*&*/nBytesBehind, /*&*/nBytesAhead);

      if (nBytesAhead == 0) return;
    }
  }

/*
 * For gfx90a,
* Before we had `Unroll/2*(16/sizeof(T))/2`, which does not work with unroll=1
* as unroll=1; `Unroll/2` = 0, which results in the above expression to 0, and is not supported
* This was reformulated to `(Unroll*4 + sizeof(T) - 1)/sizeof(T)`
*
* Before: `Unroll/2*(16/sizeof(T))/2`
*         sizeof(T)
* unroll  1   2   4   8
*   4     16  8   4   2
*   2     8   4   2   1
*   1     0   0   0   0
*
* After: `(Unroll*4 + sizeof(T) - 1)/sizeof(T)`
*         sizeof(T)
* unroll  1   2   4   8
*   4     16  8   4   2
*   2     8   4   2   1
*   1     4   2   1   1
*/
#if defined(__gfx90a__)
  if (MinSrcs > 1) {
    if constexpr (useAcc)
      reduceCopyPacksWithBias<RedFn, T, (Unroll*4 + sizeof(T) - 1)/sizeof(T), sizeof(T),
        MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
        (nThreads, thread, redArg, preOpArgs, postOp,
        nSrcs, srcPtrFn, nDsts, dstPtrFn, nBytesBehind, nBytesAhead, accPtrFn);
    else if constexpr (Pipeline)
      reduceCopyPacksPipelined<RedFn, T, (Unroll*4 + sizeof(T) - 1)/sizeof(T), sizeof(T),
        MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
        (nThreads, thread, redArg, preOpArgs, postOp,
        nSrcs, srcPtrFn, nDsts, dstPtrFn, nBytesBehind, nBytesAhead);
    else
      reduceCopyPacks<RedFn, T, (Unroll*4 + sizeof(T) - 1)/sizeof(T), sizeof(T),
        MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
        (nThreads, thread, redArg, preOpArgs, postOp,
        nSrcs, srcPtrFn, nDsts, dstPtrFn, nBytesBehind, nBytesAhead);
  } else {
    if constexpr (useAcc)
      reduceCopyPacksWithBias<RedFn, T, Unroll*(16/sizeof(T))/2, /*BytePerPack=*/sizeof(T),
        MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
        (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
        nSrcs, srcPtrFn, nDsts, dstPtrFn, /*&*/nBytesBehind, /*&*/nBytesAhead, accPtrFn);
    else if constexpr (Pipeline)
      reduceCopyPacksPipelined<RedFn, T, Unroll*(16/sizeof(T))/2, /*BytePerPack=*/sizeof(T),
        MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
        (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
        nSrcs, srcPtrFn, nDsts, dstPtrFn, /*&*/nBytesBehind, /*&*/nBytesAhead);
    else
      reduceCopyPacks<RedFn, T, Unroll*(16/sizeof(T))/2, /*BytePerPack=*/sizeof(T),
        MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
        (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
        nSrcs, srcPtrFn, nDsts, dstPtrFn, /*&*/nBytesBehind, /*&*/nBytesAhead);
  }
#else
  if constexpr (useAcc)
    reduceCopyPacksWithBias<RedFn, T, Unroll*(16/sizeof(T))/2, /*BytePerPack=*/sizeof(T),
      MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
      (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
      nSrcs, srcPtrFn, nDsts, dstPtrFn, /*&*/nBytesBehind, /*&*/nBytesAhead, accPtrFn);
  else if constexpr (Pipeline)
    reduceCopyPacksPipelined<RedFn, T, Unroll*(16/sizeof(T))/2, /*BytePerPack=*/sizeof(T),
      MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
      (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
      nSrcs, srcPtrFn, nDsts, dstPtrFn, /*&*/nBytesBehind, /*&*/nBytesAhead);
  else
    reduceCopyPacks<RedFn, T, Unroll*(16/sizeof(T))/2, /*BytePerPack=*/sizeof(T),
      MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
      (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
      nSrcs, srcPtrFn, nDsts, dstPtrFn, /*&*/nBytesBehind, /*&*/nBytesAhead);

#endif
  if (nBytesAhead == 0) return;

  if constexpr (useAcc)
    reduceCopyPacksWithBias<RedFn, T, /*Unroll=*/1, /*BytePerPack=*/sizeof(T),
      MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
      (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
      nSrcs, srcPtrFn, nDsts, dstPtrFn, /*&*/nBytesBehind, /*&*/nBytesAhead, accPtrFn);
  else if constexpr (Pipeline)
    reduceCopyPacksPipelined<RedFn, T, /*Unroll=*/1, /*BytePerPack=*/sizeof(T),
      MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
      (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
      nSrcs, srcPtrFn, nDsts, dstPtrFn, /*&*/nBytesBehind, /*&*/nBytesAhead);
  else
    reduceCopyPacks<RedFn, T, /*Unroll=*/1, /*BytePerPack=*/sizeof(T),
      MultimemSrcs, MinSrcs, MaxSrcs, MultimemDsts, MinDsts, MaxDsts, PreOpSrcs>
      (nThreads, /*&*/thread, redArg, preOpArgs, postOp,
      nSrcs, srcPtrFn, nDsts, dstPtrFn, /*&*/nBytesBehind, /*&*/nBytesAhead);

}


template<int Unroll, int useAcc, typename RedFn, typename T,
         int MultimemSrcs, int MinSrcs, int MaxSrcs,
         int MultimemDsts, int MinDsts, int MaxDsts, int PreOpSrcs,
         int Pipeline = 0, typename IntBytes>
__device__ __forceinline__ void reduceCopy(
    int thread, int nThreads,
    uint64_t redArg, uint64_t *preOpArgs, bool postOp,
    int nSrcs, void** srcPtrs, int nDsts, void** dstPtrs,
    IntBytes nElts, void *accPtr = nullptr
  ) {
  reduceCopy<Unroll, useAcc, RedFn, T,
             MultimemSrcs, MinSrcs, MaxSrcs,
             MultimemDsts, MinDsts, MaxDsts, PreOpSrcs, IntBytes, Pipeline>
    (thread, nThreads, redArg, preOpArgs, postOp,
     nSrcs, [=]__device__(int i) { return srcPtrs[i]; },
     nDsts, [=]__device__(int i) { return dstPtrs[i]; }, nElts, [=]__device__() { return accPtr; });
}

#endif // COMMON_KERNEL_H_