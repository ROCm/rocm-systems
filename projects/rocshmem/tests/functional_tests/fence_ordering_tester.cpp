/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

#include "fence_ordering_tester.hpp"

#include <hip/hip_runtime.h>

#include "rocshmem/rocshmem.hpp"

using namespace rocshmem;

/******************************************************************************
 * DEVICE HELPERS
 *****************************************************************************/

// Per-iteration fill value — changes each iteration so stale data from
// a previous iteration is distinguishable from fresh data.
__device__ __forceinline__ char iter_fill(int iter, size_t idx) {
  return (char)('a' + ((iter + idx) % 26));
}

static constexpr long CROSS_THRESHOLD_MAGIC = 0x5A5A5A5A5A5A5A5ALL;

// Fill buffer with iteration-specific pattern (all threads in block)
__device__ void fill_buffer_device(char *buf, size_t size, int iter) {
  int tid = get_flat_block_id();
  int block_size = get_flat_block_size();
  for (size_t i = tid; i < size; i += block_size) {
    buf[i] = iter_fill(iter, i);
  }
  __syncthreads();
}

// Validate buffer against iteration-specific expected pattern (parallel within wave).
// Iterates from highest index to lowest — putmem most-likely writes low-to-high, so the tail
// is most likely still in-flight if the fence didn't ensure completion.
// On first error, prints diagnostic including pe, wave_id, chunk_id (pass -1 if not applicable).
__device__ void validate_buffer_device(const char *buf, size_t size,
                                        int iter, int *error_count,
                                        size_t base_idx = 0, int chunk_id = -1,
                                        int pe_hint = -1, int wave_hint = -1) {
  int lane = get_flat_block_id() % wave_SZ();
  __builtin_amdgcn_wave_barrier();
  // Acquire fence: invalidate L1 (GL0/GL1) so that data written by a remote
  // GPU via IPC is visible after wait_until returns.  Agent scope covers all
  // caches on this GPU; workgroup scope would not suffice for cross-GPU writes.
  //TODO: THIS FENCE SHOULD PROBABLY BE IN wait_until rather than here
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  for (size_t i = size - 1 - lane; i < size; i -= wave_SZ()) {
    char expected = iter_fill(iter, base_idx + i);
    if (buf[i] != expected) {
      int old = atomicAdd(error_count, 1);
      if (old == 0) {
        printf("[fence_order] first error: pe=%d wave=%d chunk=%d byte=%zu "
               "expected=0x%02x got=0x%02x iter=%d\n",
               pe_hint, wave_hint, chunk_id, i,
               (unsigned char)expected, (unsigned char)buf[i], iter);
      }
    }
  }
}

// Wait for receiver's ack from the previous iteration before overwriting
// the destination buffer. Called by the SENDER wave.
__device__ void wait_for_validated_ack(uint64_t *signal_validated, int wave_id, int iter) {
  if (is_thread_zero_in_wave()) {
    rocshmem_uint64_wait_until(&signal_validated[wave_id], ROCSHMEM_CMP_GE, (uint64_t)iter);
  }
  __syncthreads();
}

/******************************************************************************
 * TEST 1: NBI Put + Fence + Signal (DeepEP pattern)
 *
 * Sender: putmem_nbi_wave → fence → atomic_add signal
 * Receiver: spin on signal → validate data on device
 *****************************************************************************/
__global__ void FenceOrderPutWaveSignalKernel(
    int loop, int skip, long long int *start_time, long long int *end_time,
    char *s_buf, char *r_buf, size_t size, uint64_t *signal, int *error_count,
    ShmemContextType ctx_type, int wf_size) {
  __shared__ rocshmem_ctx_t ctx;
  int wg_id = get_flat_grid_id();
  int my_pe = rocshmem_my_pe();
  int tid = get_flat_block_id();
  int block_size = get_flat_block_size();
  int waves_per_wg = (block_size - 1) / wf_size + 1;
  int wave_id = wg_id * waves_per_wg + tid / wf_size;
  int total_waves = hipGridDim_x * waves_per_wg;

  rocshmem_wg_ctx_create(ctx_type, &ctx);

  uint64_t *signal_ready = signal;
  uint64_t *signal_validated = &signal[total_waves];

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip) {
      __syncthreads();
      if (tid == 0) start_time[wg_id] = wall_clock64();
    }

    if (my_pe == 0) {
      fill_buffer_device(s_buf + wg_id * size, size, i);
      rocshmem_ctx_putmem_nbi_wave(ctx, r_buf + wave_id * size, s_buf + wg_id * size, size, 1);
      rocshmem_ctx_fence(ctx);
      if (is_thread_zero_in_wave()) {
        rocshmem_ctx_ulong_atomic_add(ctx, &signal_ready[wave_id], 1, 1);
      }
      wait_for_validated_ack(signal_validated, wave_id, i+1);
    } else {
      if (is_thread_zero_in_wave()) {
        rocshmem_uint64_wait_until(&signal_ready[wave_id], ROCSHMEM_CMP_GE, (uint64_t)(i + 1));
      }
      validate_buffer_device(r_buf + wave_id * size, size, i, error_count, 0);
      __syncthreads();
      if (is_thread_zero_in_wave()) {
        rocshmem_ctx_ulong_atomic_add(ctx, &signal_validated[wave_id], 1, 0);
      }
    }
  }

  __syncthreads();
  if (tid == 0) end_time[wg_id] = wall_clock64();

  rocshmem_wg_ctx_destroy(&ctx);
}

/******************************************************************************
 * TEST 2: Cross-Threshold Ordering
 *
 * Large putmem_nbi_wave (SDMA path) → fence → rocshmem_long_p (load/store,
 * acts as both data and signal).  Receiver: wait_until(p_dest) → validate
 * large buffer.  Fence is the ordering primitive between the large put and
 * the scalar p.  If fence doesn't drain SDMA, receiver sees the p value
 * but large buffer has stale data.
 *****************************************************************************/
__global__ void FenceOrderPutLargeSmallKernel(
    int loop, int skip, long long int *start_time, long long int *end_time,
    char *s_buf, char *r_buf,
    size_t size, uint64_t *signal, int *error_count,
    ShmemContextType ctx_type, int wf_size) {
  __shared__ rocshmem_ctx_t ctx;
  int wg_id = get_flat_grid_id();
  int my_pe = rocshmem_my_pe();
  int tid = get_flat_block_id();
  int block_size = get_flat_block_size();
  int waves_per_wg = (block_size - 1) / wf_size + 1;
  int wave_id = wg_id * waves_per_wg + tid / wf_size;
  int total_waves = hipGridDim_x * waves_per_wg;

  rocshmem_wg_ctx_create(ctx_type, &ctx);

  long *flags = reinterpret_cast<long *>(signal);
  uint64_t *signal_validated = &signal[total_waves];

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip) {
      __syncthreads();
      if (tid == 0) start_time[wg_id] = wall_clock64();
    }

    if (my_pe == 0) {
      fill_buffer_device(s_buf + wg_id * size, size, i);
      rocshmem_ctx_putmem_nbi_wave(ctx, r_buf + wave_id * size, s_buf + wg_id * size, size, 1);
      rocshmem_ctx_fence(ctx);
      // Signal via long_p (load/store path, tests cross-threshold ordering)
      if (is_thread_zero_in_wave()) {
        rocshmem_ctx_long_p(ctx, &flags[wave_id], CROSS_THRESHOLD_MAGIC + i, 1);
      }
      wait_for_validated_ack(signal_validated, wave_id, i+1);
    } else {
      // Wait for long_p flag (different from test 1's atomic_add)
      if (is_thread_zero_in_wave()) {
        rocshmem_long_wait_until(&flags[wave_id], ROCSHMEM_CMP_EQ,
                                 CROSS_THRESHOLD_MAGIC + i);
      }
      validate_buffer_device(r_buf + wave_id * size, size, i, error_count);
      __syncthreads();
      if (is_thread_zero_in_wave()) {
        rocshmem_ctx_ulong_atomic_add(ctx, &signal_validated[wave_id], 1, 0);
      }
    }
  }

  __syncthreads();
  if (tid == 0) end_time[wg_id] = wall_clock64();

  rocshmem_wg_ctx_destroy(&ctx);
}

/******************************************************************************
 * TEST 3: Fanout Wave Put + Signal
 *
 * Each wave sends to a different PE (round-robin), fence, signal.
 * Each PE receives from all senders, validates all buffers.
 * Tests per-PE dirty tracking with multiple PEs dirty simultaneously.
 *****************************************************************************/
__global__ void FenceOrderFanoutKernel(
    int loop, int skip, long long int *start_time, long long int *end_time,
    char *s_buf, char *r_buf, size_t size,
    uint64_t *local_signal, uint64_t *remote_signal, int *error_count,
    ShmemContextType ctx_type, int wf_size) {
  __shared__ rocshmem_ctx_t ctx;
  int wg_id = get_flat_grid_id();
  int my_pe = rocshmem_my_pe();
  int num_pes = rocshmem_n_pes();
  int tid = get_flat_block_id();
  int block_size = get_flat_block_size();
  int waves_per_wg = (block_size - 1) / wf_size + 1;
  int wave_id = wg_id * waves_per_wg + tid / wf_size;
  int total_waves = hipGridDim_x * waves_per_wg;

  rocshmem_wg_ctx_create(ctx_type, &ctx);

  long *flags = reinterpret_cast<long *>(local_signal);
  long *remote_flags = reinterpret_cast<long *>(remote_signal);
  uint64_t *signal_validated = &local_signal[total_waves];

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip) {
      __syncthreads();
      if (tid == 0) start_time[wg_id] = wall_clock64();
    }

    // Each wave targets a different PE (including self)
    int target_pe = (my_pe + 1 + wave_id) % num_pes;


    fill_buffer_device(s_buf + wg_id * size, size, i);

    // Send to target PE, fence, signal
    rocshmem_ctx_putmem_nbi_wave(ctx, r_buf + wave_id * size, s_buf + wg_id * size, size,
                                  target_pe);
    rocshmem_ctx_fence(ctx);
    if (is_thread_zero_in_wave()) {
      rocshmem_ctx_long_p(ctx, &remote_flags[wave_id],
                           CROSS_THRESHOLD_MAGIC + i, target_pe);
    }

    // Wait for incoming data, validate, ack to sender
    if (is_thread_zero_in_wave()) {
      rocshmem_long_wait_until(&flags[wave_id], ROCSHMEM_CMP_EQ,
                               CROSS_THRESHOLD_MAGIC + i);
    }
    validate_buffer_device(r_buf + wave_id * size, size, i, error_count,
                           0, -1, my_pe, wave_id);
    __syncthreads();
    if (is_thread_zero_in_wave()) {
      // Inverse of target_pe = (my_pe + 1 + wave_id) % num_pes:
      // find p such that (p + 1 + wave_id) % num_pes == my_pe
      int sender_pe = ((my_pe - 1 - wave_id) % num_pes + num_pes) % num_pes;
      rocshmem_ctx_ulong_atomic_add(ctx, &signal_validated[wave_id], 1, sender_pe);
    }
    wait_for_validated_ack(signal_validated, wave_id, i+1);
  }

  __syncthreads();
  if (tid == 0) end_time[wg_id] = wall_clock64();

  rocshmem_wg_ctx_destroy(&ctx);
}

/******************************************************************************
 * TEST 4: Repeated NBI Put + Fence + Signal (stress)
 *
 * Multiple iterations, per-iteration fence+signal, receiver validates each
 * chunk as it arrives (on device)
 *****************************************************************************/
static constexpr int STRESS_NUM_CHUNKS = 16;

__global__ void FenceOrderPutWaveNbiChunksKernel(
    int loop, int skip, long long int *start_time, long long int *end_time,
    char *s_buf, char *r_buf, size_t chunk_size, uint64_t *signal,
    int *error_count, ShmemContextType ctx_type, int wf_size) {
  __shared__ rocshmem_ctx_t ctx;
  int wg_id = get_flat_grid_id();
  int my_pe = rocshmem_my_pe();
  int tid = get_flat_block_id();
  int block_size = get_flat_block_size();
  int waves_per_wg = (block_size - 1) / wf_size + 1;
  int wave_id = wg_id * waves_per_wg + tid / wf_size;
  int total_waves = hipGridDim_x * waves_per_wg;
  size_t wave_data_size = chunk_size * STRESS_NUM_CHUNKS;

  rocshmem_wg_ctx_create(ctx_type, &ctx);

  long *flags = reinterpret_cast<long *>(signal);
  uint64_t *signal_validated = &signal[total_waves];

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip) {
      __syncthreads();
      if (tid == 0) start_time[wg_id] = wall_clock64();
    }

    if (my_pe == 0) {
      fill_buffer_device(s_buf + wg_id * wave_data_size, wave_data_size, i);
      size_t wave_base = wave_id * wave_data_size;
      size_t s_base = wg_id * wave_data_size;
      for (int c = 0; c < STRESS_NUM_CHUNKS; c++) {
        size_t offset = c * chunk_size;
        rocshmem_ctx_putmem_nbi_wave(ctx, r_buf + wave_base + offset,
                                      s_buf + s_base + offset, chunk_size, 1);
        rocshmem_ctx_fence(ctx);
        if (is_thread_zero_in_wave()) {
          rocshmem_ctx_long_p(ctx, &flags[wave_id],
                               CROSS_THRESHOLD_MAGIC + i * STRESS_NUM_CHUNKS + c, 1);
        }
      }
      wait_for_validated_ack(signal_validated, wave_id, i+1);
    } else {
      size_t wave_base = wave_id * wave_data_size;
      for (int c = 0; c < STRESS_NUM_CHUNKS; c++) {
        if (is_thread_zero_in_wave()) {
          rocshmem_long_wait_until(&flags[wave_id], ROCSHMEM_CMP_GE,
                                   CROSS_THRESHOLD_MAGIC + i * STRESS_NUM_CHUNKS + c);
        }
        size_t offset = c * chunk_size;
        validate_buffer_device(r_buf + wave_base + offset, chunk_size,
                               i, error_count, offset, c, my_pe, wave_id);
      }
      __syncthreads();
      if (is_thread_zero_in_wave()) {
        rocshmem_ctx_ulong_atomic_add(ctx, &signal_validated[wave_id], 1, 0);
      }
    }
  }

  __syncthreads();
  if (tid == 0) end_time[wg_id] = wall_clock64();

  rocshmem_wg_ctx_destroy(&ctx);
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
FenceOrderingTester::FenceOrderingTester(TesterArguments args)
    : Tester(args) {
  // Each wave writes to its own slice of r_buf
  int waves_per_wg = (args.wg_size - 1) / wf_size + 1;
  int total_waves = args.num_wgs * waves_per_wg;
  size_t r_buf_size = (size_t)total_waves * args.max_msg_size;
  size_t s_buf_size = (size_t)args.num_wgs * args.max_msg_size;
  s_buf = (char *)rocshmem_malloc(s_buf_size);
  r_buf = (char *)rocshmem_malloc(r_buf_size);
  // Per-wave data flags + ack flags: 2 * total_waves entries
  signal = (uint64_t *)rocshmem_malloc(2 * total_waves * sizeof(uint64_t));
  CHECK_HIP(hipMallocManaged(&error_count, sizeof(int), hipMemAttachGlobal));
}

FenceOrderingTester::~FenceOrderingTester() {
  rocshmem_free(s_buf);
  rocshmem_free(r_buf);
  rocshmem_free(signal);
  CHECK_HIP(hipFree(error_count));
}

void FenceOrderingTester::resetBuffers(size_t size) {
  // Fill source with recognizable pattern
  for (size_t i = 0; i < size; i++) {
    s_buf[i] = 'a' + (i % 26);
  }
  // Each wave has its own slice; clear all
  int waves_per_wg = (args.wg_size - 1) / wf_size + 1;
  int total_waves = args.num_wgs * waves_per_wg;
  memset(r_buf, 0, (size_t)total_waves * size);
  memset(signal, 0, 2 * total_waves * sizeof(uint64_t));
  *error_count = 0;
}

void FenceOrderingTester::launchKernel(dim3 gridSize, dim3 blockSize,
                                       int loop, size_t size) {
  last_loop = loop;
  size_t shared_bytes = 0;

  switch (_type) {
    case FenceOrderPutWaveSignalTestType:
      hipLaunchKernelGGL(FenceOrderPutWaveSignalKernel,
                         gridSize, blockSize, shared_bytes, stream,
                         loop, args.skip, start_time, end_time,
                         s_buf, r_buf, size, signal, error_count,
                         _shmem_context, wf_size);
      break;
    case FenceOrderPutLargeSmallTestType:
      hipLaunchKernelGGL(FenceOrderPutLargeSmallKernel,
                         gridSize, blockSize, shared_bytes, stream,
                         loop, args.skip, start_time, end_time,
                         s_buf, r_buf,
                         size, signal, error_count, _shmem_context, wf_size);
      break;
    case FenceOrderFanoutTestType:
      // Fanout: each wave sends to a different PE. Signal is symmetric —
      // each PE has its own copy in the symmetric heap.
      hipLaunchKernelGGL(FenceOrderFanoutKernel,
                         gridSize, blockSize, shared_bytes, stream,
                         loop, args.skip, start_time, end_time,
                         s_buf, r_buf, size, signal, signal,
                         error_count, _shmem_context, wf_size);
      break;
    case FenceOrderPutWaveNbiChunksTestType: {
      size_t chunk_size = size / STRESS_NUM_CHUNKS;
      if (chunk_size < 1) chunk_size = 1;
      hipLaunchKernelGGL(FenceOrderPutWaveNbiChunksKernel,
                         gridSize, blockSize, shared_bytes, stream,
                         loop, args.skip, start_time, end_time,
                         s_buf, r_buf, chunk_size, signal, error_count,
                         _shmem_context, wf_size);
      break;
    }
    default:
      break;
  }

  int waves_per_wg = (gridSize.x > 0 && blockSize.x > 0)
                         ? ((int)blockSize.x - 1) / wf_size + 1 : 1;
  int total_waves = gridSize.x * waves_per_wg;
  // Each outer iteration transfers `size` bytes total per wave (for chunks:
  // STRESS_NUM_CHUNKS × chunk_size = size). Count iterations, not chunks,
  // so the framework's BW formula (num_timed_msgs × size / time) is correct.
  num_msgs = (loop + args.skip) * total_waves;
  num_timed_msgs = loop * total_waves;
}

void FenceOrderingTester::verifyResults([[maybe_unused]] size_t size) {
  // Check device-side error count
  int errors = *error_count;
  if (errors > 0) {
    fprintf(stderr, "Fence ordering test %d: %d data validation errors "
            "detected on device\n", _type, errors);
  }

  // Host-side validation as secondary check with detailed error reporting.
  // The kernel boundary flush may mask some races, but this catches
  // persistent corruption and provides diagnostic output.
  int last_iter = last_loop + args.skip - 1;  // matches loop count passed to launchKernel (may be loop_large)

  auto host_check = [&](const char *buf, size_t len, int iter,
                         const char *label) {
    for (size_t i = 0; i < len; i++) {
      char expected = 'a' + ((iter + i) % 26);
      if (buf[i] != expected) {
        fprintf(stderr, "Host %s validation error at idx %zu: "
                "got 0x%02x, expected 0x%02x (iter %d)\n",
                label, i, (unsigned char)buf[i], (unsigned char)expected, iter);
        exit(-1);
      }
    }
  };

  int waves_per_wg = (args.wg_size - 1) / wf_size + 1;
  int total_waves = args.num_wgs * waves_per_wg;

  if (_type == FenceOrderFanoutTestType) {
    // All PEs receive — check every wave slot
    for (int w = 0; w < total_waves; w++) {
      host_check(r_buf + w * size, size, last_iter, "fanout");
    }
  } else if (args.myid == 1) {
    // PE 1 is receiver — check every wave slot
    for (int w = 0; w < total_waves; w++) {
      host_check(r_buf + w * size, size, last_iter, "receiver");
    }
  }
}
