/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

/**
 * @file gin_qp_factory_test.cpp
 *
 * Standalone MPI test for the GIN QP factory.
 * Exercises: QP creation, MR registration, GPU-initiated RDMA put,
 * quiet (flush), and RDMA atomic (signal model).
 *
 * Run with: mpirun -np 2 ./gin_qp_factory_test
 */

#include <mpi.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

#include "gda/gin_qp_factory.hpp"
#include "gda/queue_pair.hpp"

#define HIP_CHECK(cmd) do {                                      \
  hipError_t e = (cmd);                                          \
  if (e != hipSuccess) {                                         \
    fprintf(stderr, "HIP error %d at %s:%d\n", e, __FILE__, __LINE__); \
    MPI_Abort(MPI_COMM_WORLD, 1);                                \
  }                                                              \
} while(0)

///////////////////////////////////////////////////////////////////////////////
// MPI-based allgather callback for gin_qp_factory
///////////////////////////////////////////////////////////////////////////////

static int mpi_allgather(void *ctx, void *buf, size_t perRankSize) {
  MPI_Comm comm = *(MPI_Comm*)ctx;
  return MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       buf, perRankSize, MPI_BYTE, comm);
}

///////////////////////////////////////////////////////////////////////////////
// GPU kernels
///////////////////////////////////////////////////////////////////////////////

// Simple put kernel: thread 0 puts data from src to dst on peer
__global__ void gin_put_kernel(rocshmem::QueuePair **qps,
                               void *dst, void *src, size_t nbytes,
                               int peer, uint32_t dst_rkey, uint32_t src_lkey) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    rocshmem::ActiveWFInfo wf_info(peer, rocshmem::ThreadScope::thread);
    qps[peer]->put_nbi_with_keys(dst, src, nbytes, peer, wf_info,
                                  dst_rkey, src_lkey);
    qps[peer]->quiet(wf_info);
  }
}

// Atomic add kernel: thread 0 does a remote atomic fetch-and-add
__global__ void gin_atomic_kernel(rocshmem::QueuePair **qps,
                                   void *remote_addr, int64_t value,
                                   int peer) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    rocshmem::ActiveWFInfo wf_info(peer, rocshmem::ThreadScope::thread);
    qps[peer]->atomic_nofetch(remote_addr, value, 0, wf_info);
    qps[peer]->quiet(wf_info);
  }
}

///////////////////////////////////////////////////////////////////////////////
// Test routines
///////////////////////////////////////////////////////////////////////////////

static int test_qp_creation(int rank, int nranks, MPI_Comm comm) {
  printf("[rank %d] Test: QP creation\n", rank);

  rocshmem_gin_qp_set_t qp_set = nullptr;
  void **gpu_qps = nullptr;

  int rc = rocshmem_gin_create_qps(nranks, rank, mpi_allgather, &comm,
                                    &qp_set, &gpu_qps);
  if (rc != 0) {
    fprintf(stderr, "[rank %d] FAIL: rocshmem_gin_create_qps returned %d\n", rank, rc);
    return -1;
  }

  int provider = rocshmem_gin_get_provider(qp_set);
  printf("[rank %d] PASS: QP creation (provider=%d, nranks=%d)\n", rank, provider, nranks);

  // Cleanup
  rocshmem_gin_destroy_qps(qp_set);
  return 0;
}

static int test_put(int rank, int nranks, MPI_Comm comm) {
  printf("[rank %d] Test: RDMA put\n", rank);

  rocshmem_gin_qp_set_t qp_set = nullptr;
  void **gpu_qps = nullptr;

  int rc = rocshmem_gin_create_qps(nranks, rank, mpi_allgather, &comm,
                                    &qp_set, &gpu_qps);
  if (rc != 0) {
    fprintf(stderr, "[rank %d] FAIL: QP creation failed\n", rank);
    return -1;
  }

  // Allocate source and destination buffers
  const size_t nbytes = 1024;
  void *src_buf = nullptr, *dst_buf = nullptr;
  HIP_CHECK(hipMalloc(&src_buf, nbytes));
  HIP_CHECK(hipMalloc(&dst_buf, nbytes));

  // Fill source with rank+1 pattern, zero destination
  uint8_t pattern = (uint8_t)(rank + 1);
  HIP_CHECK(hipMemset(src_buf, pattern, nbytes));
  HIP_CHECK(hipMemset(dst_buf, 0, nbytes));

  // Register both buffers
  void *src_mr = nullptr, *dst_mr = nullptr;
  uint32_t src_lkey, src_rkey, dst_lkey, dst_rkey;

  rc = rocshmem_gin_reg_mr(qp_set, src_buf, nbytes, 0,
                             &src_mr, &src_lkey, &src_rkey);
  if (rc != 0) { fprintf(stderr, "[rank %d] FAIL: src reg_mr\n", rank); return -1; }

  rc = rocshmem_gin_reg_mr(qp_set, dst_buf, nbytes, 0,
                             &dst_mr, &dst_lkey, &dst_rkey);
  if (rc != 0) { fprintf(stderr, "[rank %d] FAIL: dst reg_mr\n", rank); return -1; }

  // Exchange rkeys and buffer addresses
  struct { uint32_t rkey; uintptr_t addr; } local_info, *all_info;
  local_info.rkey = dst_rkey;
  local_info.addr = (uintptr_t)dst_buf;
  all_info = (decltype(all_info))malloc(sizeof(*all_info) * nranks);

  MPI_Allgather(&local_info, sizeof(local_info), MPI_BYTE,
                all_info, sizeof(*all_info), MPI_BYTE, comm);

  // Each rank puts into the next rank's buffer (ring)
  int peer = (rank + 1) % nranks;
  uintptr_t remote_dst = all_info[peer].addr;
  uint32_t remote_rkey = all_info[peer].rkey;

  // Launch put kernel
  gin_put_kernel<<<1, 64>>>(
    (rocshmem::QueuePair**)gpu_qps,
    (void*)remote_dst, src_buf, nbytes,
    peer, remote_rkey, src_lkey);
  HIP_CHECK(hipDeviceSynchronize());

  MPI_Barrier(comm);

  // Verify: our dst_buf should contain pattern from (rank - 1 + nranks) % nranks
  uint8_t *host_buf = (uint8_t*)malloc(nbytes);
  HIP_CHECK(hipMemcpy(host_buf, dst_buf, nbytes, hipMemcpyDeviceToHost));

  int sender = (rank - 1 + nranks) % nranks;
  uint8_t expected = (uint8_t)(sender + 1);
  int errors = 0;
  for (size_t i = 0; i < nbytes; i++) {
    if (host_buf[i] != expected) {
      if (errors < 5)
        fprintf(stderr, "[rank %d] FAIL: dst[%zu] = 0x%02x, expected 0x%02x\n",
                rank, i, host_buf[i], expected);
      errors++;
    }
  }

  if (errors == 0) {
    printf("[rank %d] PASS: RDMA put (%zu bytes from rank %d)\n", rank, nbytes, sender);
  } else {
    fprintf(stderr, "[rank %d] FAIL: %d byte errors in RDMA put\n", rank, errors);
  }

  free(host_buf);
  free(all_info);
  rocshmem_gin_dereg_mr(src_mr);
  rocshmem_gin_dereg_mr(dst_mr);
  HIP_CHECK(hipFree(src_buf));
  HIP_CHECK(hipFree(dst_buf));
  rocshmem_gin_destroy_qps(qp_set);
  return errors ? -1 : 0;
}

static int test_atomic_signal(int rank, int nranks, MPI_Comm comm) {
  printf("[rank %d] Test: RDMA atomic (signal model)\n", rank);

  rocshmem_gin_qp_set_t qp_set = nullptr;
  void **gpu_qps = nullptr;

  int rc = rocshmem_gin_create_qps(nranks, rank, mpi_allgather, &comm,
                                    &qp_set, &gpu_qps);
  if (rc != 0) {
    fprintf(stderr, "[rank %d] FAIL: QP creation failed\n", rank);
    return -1;
  }

  // Allocate a signal counter on each rank
  uint64_t *signal_buf = nullptr;
  HIP_CHECK(hipMalloc(&signal_buf, sizeof(uint64_t)));
  HIP_CHECK(hipMemset(signal_buf, 0, sizeof(uint64_t)));

  // Register with atomic access
  void *signal_mr = nullptr;
  uint32_t sig_lkey, sig_rkey;
  rc = rocshmem_gin_reg_mr(qp_set, signal_buf, sizeof(uint64_t), /*atomic=*/1,
                             &signal_mr, &sig_lkey, &sig_rkey);
  if (rc != 0) { fprintf(stderr, "[rank %d] FAIL: signal reg_mr\n", rank); return -1; }

  // Exchange signal addresses and rkeys
  struct { uint32_t rkey; uintptr_t addr; } local_info, *all_info;
  local_info.rkey = sig_rkey;
  local_info.addr = (uintptr_t)signal_buf;
  all_info = (decltype(all_info))malloc(sizeof(*all_info) * nranks);

  MPI_Allgather(&local_info, sizeof(local_info), MPI_BYTE,
                all_info, sizeof(*all_info), MPI_BYTE, comm);

  MPI_Barrier(comm);

  // Each rank atomically adds 1 to the next rank's signal
  int peer = (rank + 1) % nranks;

  gin_atomic_kernel<<<1, 64>>>(
    (rocshmem::QueuePair**)gpu_qps,
    (void*)all_info[peer].addr, 1,
    peer);
  HIP_CHECK(hipDeviceSynchronize());

  MPI_Barrier(comm);

  // Verify: our signal should be 1 (one sender)
  uint64_t host_signal = 0;
  HIP_CHECK(hipMemcpy(&host_signal, signal_buf, sizeof(uint64_t), hipMemcpyDeviceToHost));

  int errors = 0;
  if (host_signal != 1) {
    fprintf(stderr, "[rank %d] FAIL: signal = %lu, expected 1\n", rank, host_signal);
    errors = 1;
  } else {
    printf("[rank %d] PASS: RDMA atomic signal (value=%lu)\n", rank, host_signal);
  }

  free(all_info);
  rocshmem_gin_dereg_mr(signal_mr);
  HIP_CHECK(hipFree(signal_buf));
  rocshmem_gin_destroy_qps(qp_set);
  return errors ? -1 : 0;
}

///////////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, nranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  if (nranks < 2) {
    if (rank == 0)
      fprintf(stderr, "Error: need at least 2 ranks. Run with: mpirun -np 2 %s\n", argv[0]);
    MPI_Finalize();
    return 1;
  }

  // Select GPU based on local rank (simple round-robin)
  int ndevices = 0;
  HIP_CHECK(hipGetDeviceCount(&ndevices));
  HIP_CHECK(hipSetDevice(rank % ndevices));

  if (rank == 0)
    printf("=== GIN QP Factory Test ===\n");
  printf("[rank %d] Using GPU %d/%d\n", rank, rank % ndevices, ndevices);

  int total_failures = 0;

  // Test 1: QP creation and destruction
  total_failures += (test_qp_creation(rank, nranks, MPI_COMM_WORLD) != 0);
  MPI_Barrier(MPI_COMM_WORLD);

  // Test 2: RDMA put
  total_failures += (test_put(rank, nranks, MPI_COMM_WORLD) != 0);
  MPI_Barrier(MPI_COMM_WORLD);

  // Test 3: RDMA atomic (signal model)
  total_failures += (test_atomic_signal(rank, nranks, MPI_COMM_WORLD) != 0);
  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    if (total_failures == 0)
      printf("\n=== ALL TESTS PASSED ===\n");
    else
      printf("\n=== %d TEST(S) FAILED ===\n", total_failures);
  }

  MPI_Finalize();
  return total_failures ? 1 : 0;
}
