/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// cluster_group::barrier_arrive() / barrier_wait() synchronise the workgroups
// of a cluster. Built only for cluster-capable archs (see CMakeLists.txt) and
// gated at runtime on clusterLaunch.

#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip/hip_cooperative_groups.h>

#include <vector>

// One cluster of two blocks: rank 0 sums both blocks' contributions after the
// cluster-wide wait, so a broken barrier leaves it reading stale data.
static __global__ void CLUSTER_DIMS(2, 1, 1)
    cluster_split_barrier_kernel(int* data, int* result) {
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();

  const unsigned rank = cluster.thread_rank();
  const unsigned n = cluster.num_threads();

  data[rank] = static_cast<int>(rank) + 1;  // publish before arrive

  auto tok = cluster.barrier_arrive();
  // Thread-local gap work.
  volatile int spin = 0;
  for (int k = 0; k < 4; ++k) spin += k;
  cluster.barrier_wait(std::move(tok));

  if (rank == 0) {
    int sum = 0;
    for (unsigned i = 0; i < n; i++) sum += data[i];
    *result = sum;
  }
}

HIP_TEST_CASE(Unit_cluster_split_barrier) {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  if (prop.clusterLaunch == 0) {
    HIP_SKIP_TEST("cluster launch is not supported on this device.");
  }

  constexpr unsigned blocks = 2, threads = 64;
  constexpr unsigned total = blocks * threads;

  int *d_data, *d_result;
  HIP_CHECK(hipMalloc(&d_data, sizeof(int) * total));
  HIP_CHECK(hipMalloc(&d_result, sizeof(int)));
  HIP_CHECK(hipMemset(d_data, 0, sizeof(int) * total));
  HIP_CHECK(hipMemset(d_result, 0, sizeof(int)));

  // grid dims == cluster dims => one cluster, so thread_rank indexes data[].
  cluster_split_barrier_kernel<<<dim3(blocks, 1, 1), dim3(threads, 1, 1)>>>(
      d_data, d_result);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  int result = 0;
  HIP_CHECK(hipMemcpy(&result, d_result, sizeof(int), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_data));
  HIP_CHECK(hipFree(d_result));

  // sum_{i=1..total} i
  const int expected = static_cast<int>((total * (total + 1)) / 2);
  REQUIRE(result == expected);
}
