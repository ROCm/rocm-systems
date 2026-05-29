/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

#include "host_amo_tester.hpp"

#include <chrono>
#include <iostream>
#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/

HostAMOTester::HostAMOTester(TesterArguments args) : Tester(args) {
  dest_ = (int*)rocshmem_malloc(sizeof(int));
}

HostAMOTester::~HostAMOTester() {
  rocshmem_free(dest_);
}

void HostAMOTester::resetBuffers([[maybe_unused]] uint64_t size) {
  *dest_ = 0;
}

void HostAMOTester::execute() {
  int myid  = args.myid;
  int n_pes = args.numprocs;

  if (n_pes < 2) {
    if (myid == 0)
      std::cerr << "HostAMOTester requires at least 2 PEs\n";
    return;
  }

  _print_results = (myid == 0);
  num_loops = args.loop;

  resetBuffers(sizeof(int));
  rocshmem_barrier_all();

  auto wall_start = std::chrono::high_resolution_clock::now();

  if (myid == 0) {
    for (int i = 0; i < num_loops + args.skip; i++) {
      switch (_type) {
        case HostAMOFAddTestType:
          rocshmem_int_atomic_fetch_add(dest_, 1, /*pe=*/1);
          break;
        case HostAMOFCswapTestType:
          rocshmem_int_atomic_compare_swap(dest_, i, i + 1, /*pe=*/1);
          break;
        case HostAMOFenceQuietTestType:
          rocshmem_int_atomic_fetch_add(dest_, 1, /*pe=*/1);
          rocshmem_fence();
          rocshmem_quiet();
          break;
        default:
          break;
      }
    }
    rocshmem_quiet();
  }

  rocshmem_barrier_all();

  auto wall_end = std::chrono::high_resolution_clock::now();
  double elapsed_us =
      std::chrono::duration<double, std::micro>(wall_end - wall_start).count();

  if (args.verif)
    verifyResults(sizeof(int));

  rocshmem_barrier_all();

  if (myid == 0) {
    double latency = elapsed_us / num_loops;
    std::cout << "### HostAMO avg latency per op: " << latency << " us ###\n";
  }
}

void HostAMOTester::verifyResults([[maybe_unused]] uint64_t size) {
  int myid = args.myid;

  if (myid != 1) return;

  int expected = num_loops + args.skip;

  if (*dest_ != expected) {
    std::cerr << "PE 1 VERIFY FAILED: dest_=" << *dest_
              << " expected=" << expected << "\n";
    *verification_error = true;
    exit(1);
  }
}
