/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include "host_rma_tester.hpp"

#include <iostream>
#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

HostRmaTester::HostRmaTester(TesterArguments args) : Tester(args) {
  my_pe = rocshmem_my_pe();
  n_pes = rocshmem_n_pes();
  peer  = (my_pe + 1) % n_pes;

  source_buf  = reinterpret_cast<char *>(alloc_test_buffer(max_msg_size));
  dest_buf    = reinterpret_cast<char *>(alloc_test_buffer(max_msg_size));
  amo_buf     = reinterpret_cast<long *>(alloc_test_buffer(sizeof(long)));
  amo_int_buf = reinterpret_cast<int  *>(alloc_test_buffer(sizeof(int)));
  wait_buf    = reinterpret_cast<long *>(alloc_test_buffer(WAIT_NELEMS * sizeof(long)));
}

HostRmaTester::~HostRmaTester() {
  free_test_buffer(source_buf);
  free_test_buffer(dest_buf);
  free_test_buffer(amo_buf);
  free_test_buffer(amo_int_buf);
  free_test_buffer(wait_buf);
}

void HostRmaTester::resetBuffers(size_t size) {
  for (size_t i = 0; i < size; i++)
    source_buf[i] = static_cast<char>('A' + (my_pe + i) % 26);
  memset(dest_buf, 0, size);
  *amo_buf     = 0L;
  *amo_int_buf = 0;
  for (size_t i = 0; i < WAIT_NELEMS; i++)
    wait_buf[i] = 0L;
  p2p_result_idx = SIZE_MAX;
  p2p_result_ncompleted = 0;
}

void HostRmaTester::launchKernel([[maybe_unused]] dim3 gridSize,
                                    [[maybe_unused]] dim3 blockSize,
                                    [[maybe_unused]] int loop,
                                    size_t size) {
  num_msgs       = 1;
  num_timed_msgs = 1;

  // The framework calls rocshmem_barrier_all() before and after launchKernel.
  // The post-launchKernel barrier provides cross-PE visibility.

  switch (_type) {
    // -----------------------------------------------------------------------
    // Default-context RMA — exercises rocshmem_fence / rocshmem_quiet
    // -----------------------------------------------------------------------
    case HostPutmemTestType:
      if (my_pe == 0) {
        rocshmem_putmem(dest_buf, source_buf, size, peer);
        rocshmem_fence();
      }
      break;

    case HostGetmemTestType:
      if (my_pe == 0) {
        rocshmem_getmem(dest_buf, source_buf, size, peer);
        rocshmem_quiet();
      }
      break;

    // -----------------------------------------------------------------------
    // Explicit-context RMA — exercises rocshmem_ctx_fence / rocshmem_ctx_quiet
    // Requires ROCSHMEM_MAX_NUM_HOST_CONTEXTS >= 2 (default context + this one).
    // -----------------------------------------------------------------------
    case HostCtxPutmemTestType: {
      if (my_pe == 0) {
        rocshmem_ctx_t ctx;
        int rc = rocshmem_ctx_create(0, &ctx);
        if (rc != ROCSHMEM_SUCCESS) {
          std::cerr << "[PE " << my_pe
                    << "] IpcHostCtxPutmem: rocshmem_ctx_create failed rc=" << rc
                    << " (set ROCSHMEM_MAX_NUM_HOST_CONTEXTS >= 2)\n";
          rocshmem_global_exit(1);
        }
        rocshmem_ctx_putmem(ctx, dest_buf, source_buf, size, peer);
        rocshmem_ctx_fence(ctx);
        rocshmem_ctx_destroy(ctx);
      }
      break;
    }

    case HostCtxGetmemTestType: {
      if (my_pe == 0) {
        rocshmem_ctx_t ctx;
        int rc = rocshmem_ctx_create(0, &ctx);
        if (rc != ROCSHMEM_SUCCESS) {
          std::cerr << "[PE " << my_pe
                    << "] IpcHostCtxGetmem: rocshmem_ctx_create failed rc=" << rc
                    << " (set ROCSHMEM_MAX_NUM_HOST_CONTEXTS >= 2)\n";
          rocshmem_global_exit(1);
        }
        rocshmem_ctx_getmem(ctx, dest_buf, source_buf, size, peer);
        rocshmem_ctx_quiet(ctx);
        rocshmem_ctx_destroy(ctx);
      }
      break;
    }

    // Long AMOs (64-bit) — exercises rocshmem_long_atomic_fetch_add/cas
    case HostAmoFAddTestType:
      if (my_pe == 0) {
        long old = rocshmem_long_atomic_fetch_add(amo_buf, 1L, peer);
        dest_buf[0] = static_cast<char>(old == 0L ? 1 : 0);
      }
      break;

    case HostAmoFCswapTestType:
      if (my_pe == 0) {
        long old = rocshmem_long_atomic_compare_swap(amo_buf, 0L, 42L, peer);
        dest_buf[0] = static_cast<char>(old == 0L ? 1 : 0);
      }
      break;

    // Int AMOs (32-bit) — exercises rocshmem_int_atomic_fetch_add/cas
    case HostIntAmoFAddTestType:
      if (my_pe == 0) {
        int old = rocshmem_int_atomic_fetch_add(amo_int_buf, 1, peer);
        dest_buf[0] = static_cast<char>(old == 0 ? 1 : 0);
      }
      break;

    case HostIntAmoFCswapTestType:
      if (my_pe == 0) {
        int old = rocshmem_int_atomic_compare_swap(amo_int_buf, 0, 42, peer);
        dest_buf[0] = static_cast<char>(old == 0 ? 1 : 0);
      }
      break;

    // Each PE: fetch_add 1 (fetch path) then atomic_add 1 (non-fetch path).
    // Expected final value on PE 0: 2 * n_pes.
    case HostAmoAllPesTestType:
      rocshmem_int_atomic_fetch_add(amo_int_buf, 1, 0);
      rocshmem_int_atomic_add(amo_int_buf, 1, 0);
      rocshmem_quiet();
      break;

    // Each PE: fetch_add 1 (fetch path) then atomic_add 1 (non-fetch path)
    // to its own amo_int_buf. Expected final value: 2.
    case HostAmoSelfTestType: {
      int old = rocshmem_int_atomic_fetch_add(amo_int_buf, 1, my_pe);
      dest_buf[0] = static_cast<char>(old == 0 ? 1 : 0);
      rocshmem_int_atomic_add(amo_int_buf, 1, my_pe);
      rocshmem_quiet();
      break;
    }

    // Non-fetch atomic_add: long (default ctx) + int (default ctx) +
    // long (explicit ctx). quiet ensures completion before the barrier.
    case HostAmoAddTestType:
      if (my_pe == 0) {
        rocshmem_long_atomic_add(amo_buf, 1L, peer);
        rocshmem_int_atomic_add(amo_int_buf, 1, peer);
        rocshmem_quiet();
        rocshmem_ctx_t ctx;
        int rc = rocshmem_ctx_create(0, &ctx);
        if (rc != ROCSHMEM_SUCCESS) {
          std::cerr << "[PE " << my_pe
                    << "] AmoAdd: rocshmem_ctx_create failed rc=" << rc
                    << " (set ROCSHMEM_MAX_NUM_HOST_CONTEXTS >= 2)\n";
          rocshmem_global_exit(1);
        }
        rocshmem_ctx_long_atomic_add(ctx, amo_buf, 1L, peer);
        rocshmem_ctx_quiet(ctx);
        rocshmem_ctx_destroy(ctx);
      }
      break;

    // P2P Sync: PE 0 writes sentinel via atomic_add + quiet; PE 1 blocks in wait_until.
    // atomic_add works on both MPI and non-MPI IPC paths (rocshmem_long_p aborts on non-MPI).
    case HostWaitUntilTestType:
      if (my_pe == 0) {
        rocshmem_long_atomic_add(amo_buf, 42L, peer);
        rocshmem_quiet();
      } else {
        rocshmem_long_wait_until(amo_buf, ROCSHMEM_CMP_EQ, 42L);
      }
      break;

    // P2P Sync: PE 0 writes; PE 1 polls with rocshmem_test until non-zero.
    case HostTestTestType:
      if (my_pe == 0) {
        rocshmem_long_atomic_add(amo_buf, 42L, peer);
        rocshmem_quiet();
      } else {
        while (!rocshmem_long_test(amo_buf, ROCSHMEM_CMP_EQ, 42L)) {}
      }
      break;

    // P2P Sync: PE 0 writes WAIT_NELEMS values; PE 1 blocks in wait_until_all.
    case HostWaitUntilAllTestType:
      if (my_pe == 0) {
        for (size_t i = 0; i < WAIT_NELEMS; i++)
          rocshmem_long_atomic_add(wait_buf + i, static_cast<long>(i + 1), peer);
        rocshmem_quiet();
      } else {
        rocshmem_long_wait_until_all(wait_buf, WAIT_NELEMS, nullptr,
                                     ROCSHMEM_CMP_NE, 0L);
      }
      break;

    // P2P Sync: PE 0 writes index 2; PE 1 blocks in wait_until_any.
    case HostWaitUntilAnyTestType:
      if (my_pe == 0) {
        rocshmem_long_atomic_add(wait_buf + 2, 42L, peer);
        rocshmem_quiet();
      } else {
        p2p_result_idx = rocshmem_long_wait_until_any(wait_buf, WAIT_NELEMS,
                                                       nullptr, ROCSHMEM_CMP_EQ, 42L);
      }
      break;

    // P2P Sync: PE 0 writes index 1; PE 1 blocks in wait_until_some.
    case HostWaitUntilSomeTestType: {
      size_t indices[WAIT_NELEMS];
      if (my_pe == 0) {
        rocshmem_long_atomic_add(wait_buf + 1, 42L, peer);
        rocshmem_quiet();
      } else {
        p2p_result_ncompleted = rocshmem_long_wait_until_some(wait_buf, WAIT_NELEMS,
                                                               indices, nullptr,
                                                               ROCSHMEM_CMP_EQ, 42L);
        p2p_result_idx = (p2p_result_ncompleted > 0) ? indices[0] : SIZE_MAX;
      }
      break;
    }

    // P2P Sync: PE 0 writes per-element values; PE 1 blocks in wait_until_all_vector.
    case HostWaitUntilAllVectorTestType: {
      long vals_arr[WAIT_NELEMS];
      for (size_t i = 0; i < WAIT_NELEMS; i++) vals_arr[i] = static_cast<long>(i + 1);
      if (my_pe == 0) {
        for (size_t i = 0; i < WAIT_NELEMS; i++)
          rocshmem_long_atomic_add(wait_buf + i, vals_arr[i], peer);
        rocshmem_quiet();
      } else {
        rocshmem_long_wait_until_all_vector(wait_buf, WAIT_NELEMS, nullptr,
                                             ROCSHMEM_CMP_EQ, vals_arr);
      }
      break;
    }

    // P2P Sync: PE 0 writes index 2 (val=3); PE 1 blocks in wait_until_any_vector.
    case HostWaitUntilAnyVectorTestType: {
      long vals_arr[WAIT_NELEMS];
      for (size_t i = 0; i < WAIT_NELEMS; i++) vals_arr[i] = static_cast<long>(i + 1);
      if (my_pe == 0) {
        rocshmem_long_atomic_add(wait_buf + 2, vals_arr[2], peer);
        rocshmem_quiet();
      } else {
        p2p_result_idx = rocshmem_long_wait_until_any_vector(wait_buf, WAIT_NELEMS,
                                                              nullptr, ROCSHMEM_CMP_EQ,
                                                              vals_arr);
      }
      break;
    }

    // P2P Sync: PE 0 writes index 1 (val=2); PE 1 blocks in wait_until_some_vector.
    case HostWaitUntilSomeVectorTestType: {
      long vals_arr[WAIT_NELEMS];
      for (size_t i = 0; i < WAIT_NELEMS; i++) vals_arr[i] = static_cast<long>(i + 1);
      size_t indices[WAIT_NELEMS];
      if (my_pe == 0) {
        rocshmem_long_atomic_add(wait_buf + 1, vals_arr[1], peer);
        rocshmem_quiet();
      } else {
        p2p_result_ncompleted = rocshmem_long_wait_until_some_vector(wait_buf, WAIT_NELEMS,
                                                                      indices, nullptr,
                                                                      ROCSHMEM_CMP_EQ,
                                                                      vals_arr);
        p2p_result_idx = (p2p_result_ncompleted > 0) ? indices[0] : SIZE_MAX;
      }
      break;
    }

    // status={0,1,0,1}: PE 0 writes indices 0,2; PE 1 waits on those two (skips 1,3).
    // Also exercises wait_until_all_vector with per-element vals and same status mask.
    case HostWaitUntilAllStatusTestType: {
      const int status[WAIT_NELEMS] = {0, 1, 0, 1};
      long vals_arr[WAIT_NELEMS];
      for (size_t i = 0; i < WAIT_NELEMS; i++) vals_arr[i] = static_cast<long>(i + 1);
      if (my_pe == 0) {
        rocshmem_long_atomic_add(wait_buf + 0, vals_arr[0], peer);
        rocshmem_long_atomic_add(wait_buf + 2, vals_arr[2], peer);
        rocshmem_quiet();
      } else {
        rocshmem_long_wait_until_all(wait_buf, WAIT_NELEMS, status,
                                     ROCSHMEM_CMP_NE, 0L);
        rocshmem_long_wait_until_all_vector(wait_buf, WAIT_NELEMS, status,
                                            ROCSHMEM_CMP_EQ, vals_arr);
      }
      break;
    }

    // status={1,0,0,0}: PE 0 writes index 1; PE 1 waits for any (skips index 0).
    // Also exercises wait_until_any_vector with same status mask.
    case HostWaitUntilAnyStatusTestType: {
      const int status[WAIT_NELEMS] = {1, 0, 0, 0};
      long vals_arr[WAIT_NELEMS];
      for (size_t i = 0; i < WAIT_NELEMS; i++) vals_arr[i] = static_cast<long>(i + 1);
      if (my_pe == 0) {
        rocshmem_long_atomic_add(wait_buf + 1, vals_arr[1], peer);
        rocshmem_quiet();
      } else {
        p2p_result_idx = rocshmem_long_wait_until_any(wait_buf, WAIT_NELEMS,
                                                       status, ROCSHMEM_CMP_NE, 0L);
        size_t vec_idx = rocshmem_long_wait_until_any_vector(wait_buf, WAIT_NELEMS,
                                                              status, ROCSHMEM_CMP_EQ,
                                                              vals_arr);
        if (p2p_result_idx != vec_idx) {
          std::cerr << "[PE " << my_pe << "] WaitUntilAnyStatus: scalar idx="
                    << p2p_result_idx << " vector idx=" << vec_idx << " mismatch\n";
          rocshmem_global_exit(1);
        }
      }
      break;
    }

    // status={1,0,0,1}: PE 0 writes index 2; PE 1 waits for some (skips 0 and 3).
    // Also exercises wait_until_some_vector with same status mask.
    case HostWaitUntilSomeStatusTestType: {
      const int status[WAIT_NELEMS] = {1, 0, 0, 1};
      long vals_arr[WAIT_NELEMS];
      for (size_t i = 0; i < WAIT_NELEMS; i++) vals_arr[i] = static_cast<long>(i + 1);
      size_t indices[WAIT_NELEMS];
      size_t vec_indices[WAIT_NELEMS];
      if (my_pe == 0) {
        rocshmem_long_atomic_add(wait_buf + 2, vals_arr[2], peer);
        rocshmem_quiet();
      } else {
        p2p_result_ncompleted = rocshmem_long_wait_until_some(wait_buf, WAIT_NELEMS,
                                                               indices, status,
                                                               ROCSHMEM_CMP_NE, 0L);
        p2p_result_idx = (p2p_result_ncompleted > 0) ? indices[0] : SIZE_MAX;
        size_t vec_ncompleted = rocshmem_long_wait_until_some_vector(wait_buf, WAIT_NELEMS,
                                                                      vec_indices, status,
                                                                      ROCSHMEM_CMP_EQ,
                                                                      vals_arr);
        if (vec_ncompleted == 0 || vec_indices[0] != p2p_result_idx) {
          std::cerr << "[PE " << my_pe << "] WaitUntilSomeStatus: scalar idx="
                    << p2p_result_idx << " vector idx="
                    << (vec_ncompleted > 0 ? vec_indices[0] : SIZE_MAX) << " mismatch\n";
          rocshmem_global_exit(1);
        }
      }
      break;
    }

    default:
      break;
  }
}

void HostRmaTester::verifyResults(size_t size) {
  switch (_type) {
    case HostPutmemTestType:
    case HostCtxPutmemTestType:
      // PE 1 checks dest_buf matches PE 0's source pattern.
      if (my_pe == 1) {
        for (size_t i = 0; i < size; i++) {
          char expected = static_cast<char>('A' + (0 + i) % 26);
          if (dest_buf[i] != expected) {
            std::cerr << "[PE " << my_pe << "] Putmem mismatch at [" << i
                      << "]: got " << static_cast<int>(dest_buf[i])
                      << " expected " << static_cast<int>(expected) << "\n";
            rocshmem_global_exit(1);
          }
        }
      }
      break;

    case HostGetmemTestType:
    case HostCtxGetmemTestType:
      // PE 0 checks dest_buf matches PE 1's source pattern.
      if (my_pe == 0) {
        for (size_t i = 0; i < size; i++) {
          char expected = static_cast<char>('A' + (1 + i) % 26);
          if (dest_buf[i] != expected) {
            std::cerr << "[PE " << my_pe << "] Getmem mismatch at [" << i
                      << "]: got " << static_cast<int>(dest_buf[i])
                      << " expected " << static_cast<int>(expected) << "\n";
            rocshmem_global_exit(1);
          }
        }
      }
      break;

    case HostAmoFAddTestType:
      if (my_pe == 0 && dest_buf[0] != 1) {
        std::cerr << "[PE " << my_pe
                  << "] AmoFAdd(long): fetch_add did not return 0\n";
        rocshmem_global_exit(1);
      }
      if (my_pe == 1 && *amo_buf != 1L) {
        std::cerr << "[PE " << my_pe << "] AmoFAdd(long): amo_buf expected 1 got "
                  << *amo_buf << "\n";
        rocshmem_global_exit(1);
      }
      break;

    case HostAmoFCswapTestType:
      if (my_pe == 0 && dest_buf[0] != 1) {
        std::cerr << "[PE " << my_pe
                  << "] AmoFCswap(long): fetch_cas did not return old 0\n";
        rocshmem_global_exit(1);
      }
      if (my_pe == 1 && *amo_buf != 42L) {
        std::cerr << "[PE " << my_pe << "] AmoFCswap(long): amo_buf expected 42 got "
                  << *amo_buf << "\n";
        rocshmem_global_exit(1);
      }
      break;

    case HostIntAmoFAddTestType:
      if (my_pe == 0 && dest_buf[0] != 1) {
        std::cerr << "[PE " << my_pe
                  << "] AmoFAdd(int): fetch_add did not return 0\n";
        rocshmem_global_exit(1);
      }
      if (my_pe == 1 && *amo_int_buf != 1) {
        std::cerr << "[PE " << my_pe << "] AmoFAdd(int): amo_int_buf expected 1 got "
                  << *amo_int_buf << "\n";
        rocshmem_global_exit(1);
      }
      break;

    case HostIntAmoFCswapTestType:
      if (my_pe == 0 && dest_buf[0] != 1) {
        std::cerr << "[PE " << my_pe
                  << "] AmoFCswap(int): fetch_cas did not return old 0\n";
        rocshmem_global_exit(1);
      }
      if (my_pe == 1 && *amo_int_buf != 42) {
        std::cerr << "[PE " << my_pe << "] AmoFCswap(int): amo_int_buf expected 42 got "
                  << *amo_int_buf << "\n";
        rocshmem_global_exit(1);
      }
      break;

    case HostAmoAllPesTestType:
      // PE 0: fetch_add + atomic_add from each PE → 2 * n_pes total.
      if (my_pe == 0 && *amo_int_buf != 2 * n_pes) {
        std::cerr << "[PE 0] AmoAllPes: expected " << 2 * n_pes
                  << " got " << *amo_int_buf << "\n";
        rocshmem_global_exit(1);
      }
      break;

    case HostAmoSelfTestType:
      // Each PE: fetch_add returned 0, final value is 2 (fetch + non-fetch).
      if (dest_buf[0] != 1) {
        std::cerr << "[PE " << my_pe
                  << "] AmoSelf: fetch_add did not return old value 0\n";
        rocshmem_global_exit(1);
      }
      if (*amo_int_buf != 2) {
        std::cerr << "[PE " << my_pe << "] AmoSelf: expected 2 got "
                  << *amo_int_buf << "\n";
        rocshmem_global_exit(1);
      }
      break;

    case HostAmoAddTestType:
      // PE 1: two long adds (default + explicit ctx) → 2; one int add → 1.
      if (my_pe == 1 && *amo_buf != 2L) {
        std::cerr << "[PE " << my_pe << "] AmoAdd(long): expected 2 got "
                  << *amo_buf << "\n";
        rocshmem_global_exit(1);
      }
      if (my_pe == 1 && *amo_int_buf != 1) {
        std::cerr << "[PE " << my_pe << "] AmoAdd(int): expected 1 got "
                  << *amo_int_buf << "\n";
        rocshmem_global_exit(1);
      }
      break;

    case HostWaitUntilTestType:
    case HostTestTestType:
      // PE 1 unblocked from wait_until/test in launchKernel; verify the value arrived.
      if (my_pe == 1 && *amo_buf != 42L) {
        std::cerr << "[PE " << my_pe << "] WaitUntil/Test: expected 42 got "
                  << *amo_buf << "\n";
        rocshmem_global_exit(1);
      }
      break;

    case HostWaitUntilAllTestType:
      // PE 1 unblocked from wait_until_all; verify all elements arrived.
      if (my_pe == 1) {
        for (size_t i = 0; i < WAIT_NELEMS; i++) {
          long expected = static_cast<long>(i + 1);
          if (wait_buf[i] != expected) {
            std::cerr << "[PE " << my_pe << "] WaitUntilAll: wait_buf[" << i
                      << "] expected " << expected << " got " << wait_buf[i] << "\n";
            rocshmem_global_exit(1);
          }
        }
      }
      break;

    case HostWaitUntilAnyTestType:
      if (my_pe == 1) {
        if (p2p_result_idx >= WAIT_NELEMS || wait_buf[p2p_result_idx] != 42L) {
          std::cerr << "[PE " << my_pe << "] WaitUntilAny: bad idx="
                    << p2p_result_idx << "\n";
          rocshmem_global_exit(1);
        }
      }
      break;

    case HostWaitUntilSomeTestType:
      if (my_pe == 1) {
        if (p2p_result_ncompleted == 0 || p2p_result_idx >= WAIT_NELEMS ||
            wait_buf[p2p_result_idx] != 42L) {
          std::cerr << "[PE " << my_pe << "] WaitUntilSome: ncompleted="
                    << p2p_result_ncompleted << " idx=" << p2p_result_idx << "\n";
          rocshmem_global_exit(1);
        }
      }
      break;

    case HostWaitUntilAllVectorTestType:
      if (my_pe == 1) {
        for (size_t i = 0; i < WAIT_NELEMS; i++) {
          long expected = static_cast<long>(i + 1);
          if (wait_buf[i] != expected) {
            std::cerr << "[PE " << my_pe << "] WaitUntilAllVector: wait_buf[" << i
                      << "] expected " << expected << " got " << wait_buf[i] << "\n";
            rocshmem_global_exit(1);
          }
        }
      }
      break;

    case HostWaitUntilAnyVectorTestType:
      if (my_pe == 1) {
        long vals_arr[WAIT_NELEMS];
        for (size_t i = 0; i < WAIT_NELEMS; i++) vals_arr[i] = static_cast<long>(i + 1);
        if (p2p_result_idx >= WAIT_NELEMS ||
            wait_buf[p2p_result_idx] != vals_arr[p2p_result_idx]) {
          std::cerr << "[PE " << my_pe << "] WaitUntilAnyVector: bad idx="
                    << p2p_result_idx << "\n";
          rocshmem_global_exit(1);
        }
      }
      break;

    case HostWaitUntilSomeVectorTestType:
      if (my_pe == 1) {
        long vals_arr[WAIT_NELEMS];
        for (size_t i = 0; i < WAIT_NELEMS; i++) vals_arr[i] = static_cast<long>(i + 1);
        if (p2p_result_ncompleted == 0 || p2p_result_idx >= WAIT_NELEMS ||
            wait_buf[p2p_result_idx] != vals_arr[p2p_result_idx]) {
          std::cerr << "[PE " << my_pe << "] WaitUntilSomeVector: ncompleted="
                    << p2p_result_ncompleted << " idx=" << p2p_result_idx << "\n";
          rocshmem_global_exit(1);
        }
      }
      break;

    // status={0,1,0,1}: only indices 0 and 2 were written and waited on.
    case HostWaitUntilAllStatusTestType:
      if (my_pe == 1) {
        for (size_t i : {0u, 2u}) {
          long expected = static_cast<long>(i + 1);
          if (wait_buf[i] != expected) {
            std::cerr << "[PE " << my_pe << "] WaitUntilAllStatus: wait_buf[" << i
                      << "] expected " << expected << " got " << wait_buf[i] << "\n";
            rocshmem_global_exit(1);
          }
        }
      }
      break;

    // status={1,0,0,0}: PE 0 wrote index 1; returned idx must be 1.
    case HostWaitUntilAnyStatusTestType:
      if (my_pe == 1) {
        if (p2p_result_idx != 1) {
          std::cerr << "[PE " << my_pe << "] WaitUntilAnyStatus: expected idx=1 got "
                    << p2p_result_idx << "\n";
          rocshmem_global_exit(1);
        }
        if (wait_buf[1] != 2L) {
          std::cerr << "[PE " << my_pe << "] WaitUntilAnyStatus: wait_buf[1] expected 2 got "
                    << wait_buf[1] << "\n";
          rocshmem_global_exit(1);
        }
      }
      break;

    // status={1,0,0,1}: PE 0 wrote index 2; returned idx must be 2.
    case HostWaitUntilSomeStatusTestType:
      if (my_pe == 1) {
        if (p2p_result_ncompleted == 0 || p2p_result_idx != 2) {
          std::cerr << "[PE " << my_pe << "] WaitUntilSomeStatus: expected idx=2 got "
                    << p2p_result_idx << " ncompleted=" << p2p_result_ncompleted << "\n";
          rocshmem_global_exit(1);
        }
        if (wait_buf[2] != 3L) {
          std::cerr << "[PE " << my_pe << "] WaitUntilSomeStatus: wait_buf[2] expected 3 got "
                    << wait_buf[2] << "\n";
          rocshmem_global_exit(1);
        }
      }
      break;

    default:
      break;
  }
}
