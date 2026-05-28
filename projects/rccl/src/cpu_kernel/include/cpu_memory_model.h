/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-side memory ordering primitives that model AMD Instinct MI300 (gfx942)
 * device behavior used by RCCL collective kernels.
 ************************************************************************/

#ifndef RCCL_CPU_MEMORY_MODEL_H_
#define RCCL_CPU_MEMORY_MODEL_H_

#include <sched.h>
#include <stdint.h>

// MI300 (gfx942) uses RELAXED stores to peer/global with system-scope flushes at
// protocol boundaries, WORKGROUP-scoped release/acquire on block barriers, and
// SEQ_CST on abort polling — matching src/device/common.h and primitives.h.

enum rcclCpuMemScope {
  rcclCpuScopeWorkgroup = 0,
  rcclCpuScopeSystem = 1
};

static inline void rcclCpuFenceBlock() {
  __atomic_thread_fence(__ATOMIC_RELEASE);
  __atomic_thread_fence(__ATOMIC_ACQUIRE);
}

static inline void rcclCpuFenceSystem() {
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

static inline void rcclCpuStoreRelaxedU32(uint32_t* dst, uint32_t val) {
  __atomic_store_n(dst, val, __ATOMIC_RELAXED);
}

static inline void rcclCpuStoreRelaxedU64(uint64_t* dst, uint64_t val) {
  __atomic_store_n(dst, val, __ATOMIC_RELAXED);
}

static inline uint32_t rcclCpuLoadRelaxedU32(uint32_t const* src) {
  return __atomic_load_n(src, __ATOMIC_RELAXED);
}

static inline uint64_t rcclCpuLoadRelaxedU64(uint64_t const* src) {
  return __atomic_load_n(src, __ATOMIC_RELAXED);
}

static inline uint32_t rcclCpuLoadSeqCstU32(uint32_t const* src) {
  return __atomic_load_n(src, __ATOMIC_SEQ_CST);
}

static inline void rcclCpuStoreReleaseU32(uint32_t* dst, uint32_t val, enum rcclCpuMemScope scope) {
  (void)scope;
  __atomic_store_n(dst, val, __ATOMIC_RELEASE);
}

static inline void rcclCpuBarrierFetchAdd(uint64_t* barrier, int nWarps) {
  (void)nWarps;
  __atomic_fetch_add(barrier, 1, __ATOMIC_RELEASE);
  int spins = 0;
  uint64_t target = __atomic_load_n(barrier, __ATOMIC_RELAXED);
  while (__atomic_load_n(barrier, __ATOMIC_ACQUIRE) < target) {
    if (++spins > 1000000) break;
    sched_yield();
  }
  rcclCpuFenceBlock();
}

#endif  // RCCL_CPU_MEMORY_MODEL_H_
