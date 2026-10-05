/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

#include "producer_consumer_tester.hpp"

#include <cstdlib>
#include <rocshmem/rocshmem.hpp>
#include "assembly.hpp"  // uncached_load
#include "verify_results_kernels.hpp"

using namespace rocshmem;

/******************************************************************************
 * DEVICE HELPERS
 *****************************************************************************/
template <bool Relaxed>
__device__ __forceinline__ void pc_produce(char *dst, const char *src,
                                           size_t size, uint64_t *flag,
                                           uint64_t val, int peer) {
  if constexpr (Relaxed) {
    // Cache-bypassing put + waitcnt-only fence + relaxed (system-scope) set.
    rocshmem_putmem(dst, src, size, peer, CommOpt{RelaxedOrdering<true>});
    rocshmem_fence(CommOpt{RelaxedOrdering<true>});
    rocshmem_atomic_set<unsigned long>(flag, val, peer,
                                       CommOpt{RelaxedOrdering<true>});
  } else {
    // Standard ordering: release fence (L2 flush) + seq_cst atomic set.
    rocshmem_putmem(dst, src, size, peer);
    rocshmem_fence();
    rocshmem_ulong_atomic_set(flag, val, peer);
  }
}

template <bool Relaxed>
__device__ __forceinline__ void pc_consume(uint64_t *my_flag, char *my_dst,
                                           uint64_t val, size_t size) {
  if constexpr (Relaxed) {
    // Poll uncached (system-scope) and read the delivered payload uncached:
    // no acquire fence / L2 invalidate needed.
    while (uncached_load(my_flag) < val) {}
    volatile char c = uncached_load(&my_dst[size - 1]);
    (void)c;
  } else {
    // Standard wait_until + acquire (L2 invalidate) before the payload load.
    rocshmem_ulong_wait_until(my_flag, ROCSHMEM_CMP_GE, val);
    __threadfence_system();
    volatile char c = my_dst[size - 1];
    (void)c;
  }
}

/******************************************************************************
 * DEVICE TEST KERNEL (w1z1: one active thread per workgroup)
 *****************************************************************************/
template <bool Relaxed>
__global__ void ProducerConsumerKernel(int loop, int skip,
                                       long long int *start_time,
                                       long long int *end_time, char *src_buf,
                                       char *dst_buf, uint64_t *flag,
                                       uint64_t size) {
  int wg_id = blockIdx.x;
  if (threadIdx.x != 0) return;

  int pe = rocshmem_my_pe();
  int peer = 1 - pe;
  char *my_src = &src_buf[wg_id * size];
  char *slab = &dst_buf[wg_id * size];  // symmetric offset (local read / peer write)
  uint64_t *fl = &flag[wg_id];          // symmetric offset (local wait / peer set)

  // Drain setup loads before the timed loop.
  __builtin_amdgcn_s_waitcnt(0);

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip) start_time[wg_id] = wall_clock64();
    uint64_t val = static_cast<uint64_t>(i + 1);

    if (pe == 0) {
      pc_produce<Relaxed>(slab, my_src, size, fl, val, peer);
      pc_consume<Relaxed>(fl, slab, val, size);
    } else {
      pc_consume<Relaxed>(fl, slab, val, size);
      pc_produce<Relaxed>(slab, my_src, size, fl, val, peer);
    }
  }
  end_time[wg_id] = wall_clock64();
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
ProducerConsumerTester::ProducerConsumerTester(TesterArguments args)
    : Tester(args) {
  const char *e = getenv("ROCSHMEM_PC_RELAXED");
  relaxed_ = (e != nullptr && atoi(e) != 0);
  src_buf = (char *)alloc_test_buffer(max_msg_size * args.num_wgs);
  dst_buf = (char *)alloc_test_buffer(max_msg_size * args.num_wgs);
  flag = (uint64_t *)alloc_test_buffer(sizeof(uint64_t) * args.num_wgs);
  rtt_factor = 2;  // round-trip; report one-way latency
  bw_factor = 2;
}

ProducerConsumerTester::~ProducerConsumerTester() {
  free_test_buffer(src_buf);
  free_test_buffer(dst_buf);
  free_test_buffer(flag);
}

void ProducerConsumerTester::resetBuffers(uint64_t size) {
  CHECK_HIP(hipMemset(src_buf, 'a', max_msg_size * args.num_wgs));
  CHECK_HIP(hipMemset(dst_buf, 0, max_msg_size * args.num_wgs));
  CHECK_HIP(hipMemset(flag, 0, sizeof(uint64_t) * args.num_wgs));
}

void ProducerConsumerTester::launchKernel(dim3 gridSize, dim3 blockSize,
                                         int loop, uint64_t size) {
  if (relaxed_) {
    hipLaunchKernelGGL(ProducerConsumerKernel<true>, gridSize, blockSize, 0,
                       stream, loop, args.skip, start_time, end_time, src_buf,
                       dst_buf, flag, size);
  } else {
    hipLaunchKernelGGL(ProducerConsumerKernel<false>, gridSize, blockSize, 0,
                       stream, loop, args.skip, start_time, end_time, src_buf,
                       dst_buf, flag, size);
  }
  num_msgs = (loop + args.skip) * gridSize.x;
  num_timed_msgs = loop * gridSize.x;
}

void ProducerConsumerTester::verifyResults(uint64_t size) {
  size_t check_bytes = size * args.num_wgs;
  *verification_error = false;
  size_t block = std::min((size_t)1024, check_bytes);
  size_t grid = (check_bytes + block - 1) / block;
  hipLaunchKernelGGL(rocshmem::verify_results_kernel_char, grid, block, 0,
                     stream, dst_buf, check_bytes, check_bytes, 1, 1, 0, 1,
                     verification_error);
  CHECK_HIP(hipStreamSynchronize(stream));
  if (*verification_error) {
    fprintf(stderr,
            "FAIL: producer_consumer dst mismatch (expected 'a') size=%lu "
            "rank=%d\n",
            (unsigned long)size, args.myid);
    rocshmem_global_exit(1);
  }
}
