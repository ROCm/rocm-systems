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
}

HostRmaTester::~HostRmaTester() {
  free_test_buffer(source_buf);
  free_test_buffer(dest_buf);
  free_test_buffer(amo_buf);
  free_test_buffer(amo_int_buf);
}

void HostRmaTester::resetBuffers(size_t size) {
  for (size_t i = 0; i < size; i++)
    source_buf[i] = static_cast<char>('A' + (my_pe + i) % 26);
  memset(dest_buf, 0, size);
  *amo_buf     = 0L;
  *amo_int_buf = 0;
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

    // IpcHostAmoAllPes: all n_pes PEs simultaneously atomic-add 1 to PE 0's
    // amo_int_buf, including PE 0 itself. 
    // Expected final value: n_pes.
    case HostAmoAllPesTestType:
      rocshmem_int_atomic_fetch_add(amo_int_buf, 1, 0);
      break;

    case HostAmoSelfTestType: {
      int old = rocshmem_int_atomic_fetch_add(amo_int_buf, 1, my_pe);
      dest_buf[0] = static_cast<char>(old == 0 ? 1 : 0);
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
      // PE 0: final counter must equal n_pes (all PEs added 1, including self).
      if (my_pe == 0 && *amo_int_buf != n_pes) {
        std::cerr << "[PE 0] AmoAllPes: expected " << n_pes
                  << " got " << *amo_int_buf << "\n";
        rocshmem_global_exit(1);
      }
      break;

    case HostAmoSelfTestType:
      // Each PE: old value must be 0, final value must be 1.
      if (dest_buf[0] != 1) {
        std::cerr << "[PE " << my_pe
                  << "] AmoSelf: fetch_add did not return old value 0\n";
        rocshmem_global_exit(1);
      }
      if (*amo_int_buf != 1) {
        std::cerr << "[PE " << my_pe << "] AmoSelf: expected 1 got "
                  << *amo_int_buf << "\n";
        rocshmem_global_exit(1);
      }
      break;

    default:
      break;
  }
}
