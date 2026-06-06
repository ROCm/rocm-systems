/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

#ifndef LIBRARY_SRC_GDA_MICROTIMING_HPP_
#define LIBRARY_SRC_GDA_MICROTIMING_HPP_

#include <hip/hip_runtime.h>

namespace rocshmem {

/**
 * Per-put_nbi timestamp slots:
 *   0: app call site (before rocshmem_ctx_putmem_nbi)
 *   1: before isIpcAvailable check
 *   2: before ActiveWFInfo construction
 *   3: before qps[].put_nbi
 *   4: post_wqe_rma entry
 *   5: WQE written to SQ
 *   6: doorbell rung
 *   7: after put_nbi returns
 *   8: after rocshmem_ctx_putmem_nbi returns (app side)
 *
 * The timestamp array is stored in __shared__ memory (LDS) during kernel
 * execution for low-overhead recording (~1 cycle per write vs ~100+ for HBM).
 * A __device__ pointer (g_microtiming_ptr) is set to the shared buffer at
 * kernel start. At kernel end, the data is copied to the __device__ global
 * (g_microtiming) for host readback.
 *
 * Quiet is recorded separately (happens once after the loop):
 *   quiet_start, quiet_end
 */
static constexpr int MICROTIMING_STAMPS_PER_ITER = 9;
static constexpr int MICROTIMING_MAX_ITERS = 256;  // fits in LDS (~18.5 KB)
static constexpr int MICROTIMING_ARRAY_SIZE =
    MICROTIMING_STAMPS_PER_ITER * MICROTIMING_MAX_ITERS;

static constexpr int MICROTIMING_CALIB_SAMPLES = 64;

struct microtiming_t {
  uint64_t ts[MICROTIMING_ARRAY_SIZE];
  uint64_t quiet_start;
  uint64_t quiet_end;
  uint64_t e2e_start;
  uint64_t e2e_end;
  uint64_t calib[MICROTIMING_CALIB_SAMPLES];
  int iter;
  int enabled;
};

using microtiming_shared_ptr = microtiming_t __attribute__((address_space(3))) *;

/**
 * Global storage for host readback. Kernel copies shared data here before exit.
 */
extern __device__ microtiming_t g_microtiming;

/**
 * Pointer used by recording functions. Set to __shared__ buffer in kernel,
 * falls back to &g_microtiming if not redirected.
 */
extern __device__ microtiming_t* g_microtiming_ptr;

__device__ static inline uint64_t microtiming_clock() {
  return __builtin_amdgcn_s_memrealtime();
}

/**
 * Record a timestamp. The pointer version avoids any global memory access —
 * the pointer is kept in a VGPR by the caller.
 */
__device__ static inline void microtiming_record(microtiming_shared_ptr mt, int slot) {
  if (!mt->enabled || threadIdx.x != 0) return;
  int idx = mt->iter * MICROTIMING_STAMPS_PER_ITER + slot;
  if (idx < MICROTIMING_ARRAY_SIZE) {
    mt->ts[idx] = microtiming_clock();
  }
}

/**
 * Fallback: record via flat pointer (for non-LDS paths, e.g. global).
 */
__device__ static inline void microtiming_record_flat(microtiming_t* mt, int slot) {
  if (!mt || !mt->enabled) return;
  int idx = mt->iter * MICROTIMING_STAMPS_PER_ITER + slot;
  if (idx < MICROTIMING_ARRAY_SIZE) {
    mt->ts[idx] = microtiming_clock();
  }
}

/**
 * Fallback: load pointer from global. Used by call sites that don't
 * thread the pointer (e.g., quiet).
 */
__device__ static inline void microtiming_record(int slot) {
  microtiming_record_flat(g_microtiming_ptr, slot);
}

__device__ static inline void microtiming_next_iter(microtiming_shared_ptr mt) {
  if (!mt->enabled || threadIdx.x != 0) return;
  mt->iter++;
}

__device__ static inline void microtiming_next_iter() {
  microtiming_t* mt = g_microtiming_ptr;
  if (!mt || !mt->enabled) return;
  mt->iter++;
}

/**
 * Call from kernel thread 0 to set up shared memory storage.
 * Must be called before any microtiming_record calls.
 */
__device__ static inline void microtiming_init_shared(microtiming_shared_ptr shared_buf) {
  memset((void*)shared_buf, 0, sizeof(microtiming_t));
  g_microtiming_ptr = (microtiming_t*)shared_buf;
}

/**
 * Run a calibration loop to measure s_memrealtime + LDS store overhead.
 * Records back-to-back timestamps into the calib[] array; the host
 * computes the median delta as the per-record cost.
 */
__device__ static inline void microtiming_calibrate(microtiming_shared_ptr mt) {
  #pragma unroll 1
  for (int i = 0; i < MICROTIMING_CALIB_SAMPLES; i++) {
    mt->calib[i] = microtiming_clock();
  }
}

/**
 * Call from kernel thread 0 before kernel exit to copy results to global
 * memory for host readback.
 */
__device__ static inline void microtiming_flush_to_global() {
  microtiming_t* mt = g_microtiming_ptr;
  if (mt && mt != &g_microtiming) {
    memcpy(&g_microtiming, mt, sizeof(microtiming_t));
  }
}

/**
 * Host-side functions to enable/disable and read microtiming data.
 */
__host__ void microtiming_enable();
__host__ void microtiming_disable();
__host__ void microtiming_reset();
__host__ void microtiming_print();

}  // namespace rocshmem

#endif  // LIBRARY_SRC_GDA_MICROTIMING_HPP_
