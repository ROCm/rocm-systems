/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_SUB_BARRIER_H_
#define RCCL_SUB_BARRIER_H_

// The primitives' sub-block barrier, which synchronizes the workers of one group rather than the
// whole workgroup. Two implementations live here, picked by RCCL_HAVE_NAMED_BARRIERS (device.h):
// gfx1250 uses hardware named barriers, everything else spins on a shared-memory counter.
//
// Include only from common.h, after ncclShmem is declared: the spin path reads it.

#define NCCL_SPINS_BEFORE_CHECK_ABORT 10000

// ARRIVE_ORDER/WAIT_ORDER order the software arrive/spin on BARRIERS_PTR. A workgroup-scope release
// there drains every outstanding store (s_wait_storecnt on gfx1250), so callers that order memory
// with their own fences pass __ATOMIC_RELAXED.
#define barrier_generic_ordered(__THREAD_FENCE, NWORKERS, BARRIER_NEXT, BARRIERS_PTR, ARRIVE_ORDER, WAIT_ORDER) \
  do { \
    if (nthreads == threadsPerBlock) { \
      __THREAD_FENCE; \
      __builtin_amdgcn_s_barrier(); \
    } else { \
      /**const int w = threadIdx.x/WARP_SIZE //unused variable - compiler warning**/; \
      const int wid = threadIdx.x % WARP_SIZE; \
      if (wid == 0) { \
        (BARRIER_NEXT) += (NWORKERS) / WARP_SIZE; \
        __THREAD_FENCE; \
        __scoped_atomic_fetch_add((BARRIERS_PTR), 1, (ARRIVE_ORDER), __MEMORY_SCOPE_WRKGRP); \
        int spins = 0; \
        int rate_limit = 50; \
        while (__scoped_atomic_load_n((BARRIERS_PTR), (WAIT_ORDER), __MEMORY_SCOPE_WRKGRP) < (BARRIER_NEXT)) { \
          spins++; \
          if (spins == NCCL_SPINS_BEFORE_CHECK_ABORT) { \
            if (__atomic_load_n(ncclShmem.comm.abortFlag, __ATOMIC_SEQ_CST)) { \
              ncclShmem.aborted = 1; \
              break; \
            } \
            spins = 0; \
          } \
          if (spins == 0 && rate_limit > 0) { \
            rate_limit--; \
            traceData(__LINE__, threadIdx.x, \
                      __scoped_atomic_load_n((BARRIERS_PTR), (WAIT_ORDER), __MEMORY_SCOPE_WRKGRP), (BARRIER_NEXT)); \
          } \
          __builtin_amdgcn_s_sleep(1); \
        } \
        __asm__ __volatile__("s_wakeup"); \
      } \
    } \
  } while (0)

// One named barrier per group, plus RCCL_PAT_NAMED_BARRIER shared by all PAT
// workers. Keep this the only named-barrier variable so device-linker shards
// agree on barrier IDs.
#if RCCL_HAVE_NAMED_BARRIERS
#define RCCL_PAT_NAMED_BARRIER NCCL_MAX_GROUPS
#define RCCL_NUM_NAMED_BARRIERS (NCCL_MAX_GROUPS + 1)
static_assert(RCCL_NUM_NAMED_BARRIERS <= 16, "gfx1250 has 16 named barriers per workgroup");
#ifdef RCCL_DEVICE_LINKER
__shared__ __amdgpu_named_workgroup_barrier_t rcclNamedBarriers[RCCL_NUM_NAMED_BARRIERS];
#else
extern __shared__ __amdgpu_named_workgroup_barrier_t rcclNamedBarriers[RCCL_NUM_NAMED_BARRIERS];
#endif

// Toolchains disagree on where a named barrier's address keeps its hardware ID:
// bits [9:4], where array indexing steps the ID, or bits [5:0], where only a byte
// step does. Always index rcclNamedBarriers through this.
__device__ __forceinline__ __amdgpu_named_workgroup_barrier_t* rcclNamedBarrier(int i) {
  if (((uintptr_t)&rcclNamedBarriers[0] >> 4) & 0x3f) return &rcclNamedBarriers[i];
  return (__amdgpu_named_workgroup_barrier_t*)((char*)rcclNamedBarriers + i);
}
#endif

// One group's sub-block barrier, and the only place the two mechanisms differ. gfx1250
// synchronizes a subset of a workgroup's waves in hardware, so it holds a slot index and the
// shared-memory counters are compiled out; every other target spins on one. The two bodies cannot
// both compile -- each needs state or builtins the other lacks -- so this switches with the
// preprocessor rather than std::conditional, and nothing outside has to know which it got.
struct rcclSubBarrier {
#if RCCL_HAVE_NAMED_BARRIERS
  int slot;
  __device__ __forceinline__ void bindGroup(int group) { slot = group; }
  __device__ __forceinline__ void bindPat() { slot = RCCL_PAT_NAMED_BARRIER; }
  // The signal also sets the expected wave count, so groups of different widths can reuse a slot.
  // s_barrier_wait 1 waits on the last barrier this wave joined.
  __device__ __forceinline__ void arriveWait(int nworkers) {
    __builtin_amdgcn_s_barrier_join(rcclNamedBarrier(slot));
    __builtin_amdgcn_s_barrier_signal_var(rcclNamedBarrier(slot), nworkers / WARP_SIZE);
    __builtin_amdgcn_s_barrier_wait(1);
  }
  template <typename Fence>
  __device__ __forceinline__ void sync(Fence fence, int nworkers, int nthreads, int threadsPerBlock) {
    fence();
    if (nthreads == threadsPerBlock) __builtin_amdgcn_s_barrier();
    // Stands in for the acquire load the spin loop had; adds no instructions on gfx1250.
    else arriveWait(nworkers), __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
  }
  template <typename Fence>
  __device__ __forceinline__ void stepSync(Fence fence, int nworkers, int nthreads, int threadsPerBlock) {
    fence();
    if (nthreads == threadsPerBlock) __builtin_amdgcn_s_barrier();
    else arriveWait(nworkers);
  }
#else
  uint64_t* counter;
  uint64_t next;
  __device__ __forceinline__ void bindGroup(int group) { counter = &ncclShmem.groups[group].barrier; next = 0; }
  // PAT uses the same barrier for each group.
  __device__ __forceinline__ void bindPat() { counter = &ncclShmem.barrier_pat; next = 0; }
  template <typename Fence>
  __device__ __forceinline__ void sync(Fence fence, int nworkers, int nthreads, int threadsPerBlock) {
    barrier_generic_ordered(fence(), nworkers, next, counter, __ATOMIC_RELEASE, __ATOMIC_ACQUIRE);
  }
  template <typename Fence>
  __device__ __forceinline__ void stepSync(Fence fence, int nworkers, int nthreads, int threadsPerBlock) {
    barrier_generic_ordered(fence(), nworkers, next, counter, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  }
#endif
};

#define barrier_generic(__THREAD_FENCE, NWORKERS, BARRIER) \
  (BARRIER).sync([&] { __THREAD_FENCE; }, NWORKERS, nthreads, threadsPerBlock)

// Hardware barriers need no reset between batches; the spin counters do.
__device__ __forceinline__ void rcclResetSubBarriers(int tid) {
#if !RCCL_HAVE_NAMED_BARRIERS
  if (tid < WARP_SIZE + NCCL_MAX_GROUPS) {
    if (tid == WARP_SIZE) ncclShmem.barrier_pat = 0;
    ncclShmem.groups[tid - WARP_SIZE].barrier = 0;
  }
#endif
}

__device__ __forceinline__ void rcclInitSubBarriers(int tid) {
  rcclResetSubBarriers(tid);
#if RCCL_HAVE_NAMED_BARRIERS
  // LLVM's model requires an init before barriers 1-16 are first used. Any
  // non-zero count works; every arrive passes the real wave count.
  for (int b = 0; b < RCCL_NUM_NAMED_BARRIERS; b++) __builtin_amdgcn_s_barrier_init(rcclNamedBarrier(b), 1);
#endif
}

#endif  // RCCL_SUB_BARRIER_H_
