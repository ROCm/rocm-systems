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

// One cluster of two blocks: every rank consumes the element its peer in the
// other block published before arriving, so a broken barrier leaves it reading
// stale data.
static __global__ void CLUSTER_DIMS(2, 1, 1)
    cluster_split_barrier_kernel(int* data, int* out) {
  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();

  const unsigned rank = cluster.thread_rank();
  const unsigned n = cluster.num_threads();

  data[rank] = static_cast<int>(rank) + 1;  // publish before arrive

  auto tok = cluster.barrier_arrive();
  // Divergent gap work: rank r runs r iterations, so waves reach the wait at
  // different times and some find the barrier already satisfied while others
  // block. The recurrence is deliberately not affine, or the optimizer would
  // close-form it and every thread would arrive together again.
  unsigned local = 0;
  for (unsigned k = 0; k < rank; ++k) local = (local * 3u + k) & 0xFFFFu;
  cluster.barrier_wait(std::move(tok));

  // Half a cluster away, so the peer is always in the other workgroup.
  out[rank] = data[(rank + n / 2) % n] + static_cast<int>(local);
}

HIP_TEST_CASE(Unit_cluster_split_barrier) {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  if (prop.clusterLaunch == 0) {
    HIP_SKIP_TEST("cluster launch is not supported on this device.");
  }

  constexpr unsigned blocks = 2, threads = 64;
  constexpr unsigned total = blocks * threads;

  int *d_data, *d_out;
  HIP_CHECK(hipMalloc(&d_data, sizeof(int) * total));
  HIP_CHECK(hipMalloc(&d_out, sizeof(int) * total));
  HIP_CHECK(hipMemset(d_data, 0, sizeof(int) * total));
  HIP_CHECK(hipMemset(d_out, 0, sizeof(int) * total));

  // grid dims == cluster dims => one cluster, so thread_rank indexes data[].
  cluster_split_barrier_kernel<<<dim3(blocks, 1, 1), dim3(threads, 1, 1)>>>(
      d_data, d_out);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<int> out(total, 0);
  HIP_CHECK(hipMemcpy(out.data(), d_out, sizeof(int) * total,
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(d_data));
  HIP_CHECK(hipFree(d_out));

  for (unsigned i = 0; i < total; i++) {
    const unsigned peer = (i + total / 2) % total;
    unsigned local = 0;
    for (unsigned k = 0; k < i; ++k) local = (local * 3u + k) & 0xFFFFu;
    INFO("rank " << i << " peer " << peer);
    REQUIRE(out[i] == static_cast<int>(peer) + 1 + static_cast<int>(local));
  }
}
