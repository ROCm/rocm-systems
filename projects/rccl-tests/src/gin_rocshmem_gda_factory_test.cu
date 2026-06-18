/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

/**
 * @file gin_rocshmem_gda_factory_test.cu
 *
 * Standalone MPI test for the GIN QP factory.
 * Exercises: QP creation, MR registration, GPU-initiated RDMA put,
 * quiet (flush), and RDMA atomic (signal model).
 *
 * Run with: mpirun -np 2 ./gin_rocshmem_gda_factory_test
 */

#include <mpi.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>

#include <gin/gin_rocshmem_gda_factory.h>
#include <nccl_device/gin/rocshmem_gda/queue_pair_device.h>

#define HIP_CHECK(cmd) do {                                      \
  hipError_t e = (cmd);                                          \
  if (e != hipSuccess) {                                         \
    fprintf(stderr, "HIP error %d at %s:%d\n", e, __FILE__, __LINE__); \
    MPI_Abort(MPI_COMM_WORLD, 1);                                \
  }                                                              \
} while(0)

///////////////////////////////////////////////////////////////////////////////
// MPI-based allgather callback for gin_rocshmem_gda_factory
///////////////////////////////////////////////////////////////////////////////

static int mpi_allgather(void *ctx, void *buf, size_t perRankSize) {
  MPI_Comm comm = *(MPI_Comm*)ctx;
  return MPI_Allgather(MPI_IN_PLACE, perRankSize, MPI_BYTE,
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
    printf("gin_put_kernel: peer=%d dst=%p src=%p bytes=%zu rkey=0x%x lkey=0x%x\n",
           peer, dst, src, nbytes, dst_rkey, src_lkey);
    qps[peer]->put_nbi(dst, dst_rkey, src, src_lkey, nbytes, wf_info);
    printf("gin_put_kernel: put posted, calling quiet\n");
    qps[peer]->quiet(wf_info);
    printf("gin_put_kernel: quiet done\n");
  }
}

// Atomic add kernel using explicit rkey
__global__ void gin_atomic_kernel(rocshmem::QueuePair **qps,
                                             void *remote_addr, uint32_t rkey,
                                             int64_t value, int peer) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    rocshmem::ActiveWFInfo wf_info(peer, rocshmem::ThreadScope::thread);
    printf("gin_atomic_kernel: peer=%d addr=%p rkey=0x%x val=%lld\n",
           peer, remote_addr, rkey, (long long)value);
    qps[peer]->atomic_add(remote_addr, rkey, value, wf_info, /*fence=*/false);
    printf("gin_atomic_kernel: atomic posted, calling quiet\n");
    qps[peer]->quiet(wf_info);
    printf("gin_atomic_kernel: quiet done\n");
  }
}

// Combined put + fenced atomic: matches GIN signal pattern
__global__ void gin_put_signal_kernel(rocshmem::QueuePair **qps,
                                       void *dst, uint32_t dst_rkey,
                                       void *src, uint32_t src_lkey,
                                       size_t nbytes,
                                       void *signal_addr, uint32_t signal_rkey,
                                       int peer) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    rocshmem::ActiveWFInfo wf_info(peer, rocshmem::ThreadScope::thread);
    printf("gin_put_signal: peer=%d put dst=%p rkey=0x%x src=%p lkey=0x%x bytes=%zu\n",
           peer, dst, dst_rkey, src, src_lkey, nbytes);
    qps[peer]->put_nbi(dst, dst_rkey, src, src_lkey, nbytes, wf_info, /*ring_db=*/false);
    printf("gin_put_signal: atomic signal=%p rkey=0x%x fence=1\n", signal_addr, signal_rkey);
    qps[peer]->atomic_add(signal_addr, signal_rkey, 1, wf_info, /*fence=*/true);
    printf("gin_put_signal: calling quiet\n");
    qps[peer]->quiet(wf_info);
    printf("gin_put_signal: done\n");
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
  printf("[rank %d] Test: RDMA atomic_add\n", rank);

  rocshmem_gin_qp_set_t qp_set = nullptr;
  void **gpu_qps = nullptr;

  int rc = rocshmem_gin_create_qps(nranks, rank, mpi_allgather, &comm,
                                    &qp_set, &gpu_qps);
  if (rc != 0) {
    fprintf(stderr, "[rank %d] FAIL: QP creation failed\n", rank);
    return -1;
  }

  uint64_t *signal_buf = nullptr;
  HIP_CHECK(hipMalloc(&signal_buf, sizeof(uint64_t)));
  HIP_CHECK(hipMemset(signal_buf, 0, sizeof(uint64_t)));

  void *signal_mr = nullptr;
  uint32_t sig_lkey, sig_rkey;
  rc = rocshmem_gin_reg_mr(qp_set, signal_buf, sizeof(uint64_t), /*atomic=*/1,
                             &signal_mr, &sig_lkey, &sig_rkey);
  if (rc != 0) { fprintf(stderr, "[rank %d] FAIL: signal reg_mr\n", rank); return -1; }

  struct { uint32_t rkey; uintptr_t addr; } local_info, *all_info;
  local_info.rkey = sig_rkey;
  local_info.addr = (uintptr_t)signal_buf;
  all_info = (decltype(all_info))malloc(sizeof(*all_info) * nranks);

  MPI_Allgather(&local_info, sizeof(local_info), MPI_BYTE,
                all_info, sizeof(*all_info), MPI_BYTE, comm);

  MPI_Barrier(comm);

  int peer = (rank + 1) % nranks;

  gin_atomic_kernel<<<1, 64>>>(
    (rocshmem::QueuePair**)gpu_qps,
    (void*)all_info[peer].addr, all_info[peer].rkey, 1,
    peer);
  HIP_CHECK(hipDeviceSynchronize());

  MPI_Barrier(comm);

  uint64_t host_signal = 0;
  HIP_CHECK(hipMemcpy(&host_signal, signal_buf, sizeof(uint64_t), hipMemcpyDeviceToHost));

  int errors = 0;
  if (host_signal != 1) {
    fprintf(stderr, "[rank %d] FAIL: signal = %lu, expected 1\n", rank, host_signal);
    errors = 1;
  } else {
    printf("[rank %d] PASS: atomic_add (value=%lu)\n", rank, host_signal);
  }

  free(all_info);
  rocshmem_gin_dereg_mr(signal_mr);
  HIP_CHECK(hipFree(signal_buf));
  rocshmem_gin_destroy_qps(qp_set);
  return errors ? -1 : 0;
}

static int test_put_signal(int rank, int nranks, MPI_Comm comm) {
  printf("[rank %d] Test: put_nbi + fenced atomic_add (GIN pattern)\n", rank);

  rocshmem_gin_qp_set_t qp_set = nullptr;
  void **gpu_qps = nullptr;

  int rc = rocshmem_gin_create_qps(nranks, rank, mpi_allgather, &comm,
                                    &qp_set, &gpu_qps);
  if (rc != 0) {
    fprintf(stderr, "[rank %d] FAIL: QP creation failed\n", rank);
    return -1;
  }

  // Allocate data buffers
  const size_t nbytes = 1024;
  void *src_buf = nullptr, *dst_buf = nullptr;
  HIP_CHECK(hipMalloc(&src_buf, nbytes));
  HIP_CHECK(hipMalloc(&dst_buf, nbytes));

  uint8_t pattern = (uint8_t)(rank + 1);
  HIP_CHECK(hipMemset(src_buf, pattern, nbytes));
  HIP_CHECK(hipMemset(dst_buf, 0, nbytes));

  // Register data buffers
  void *src_mr = nullptr, *dst_mr = nullptr;
  uint32_t src_lkey, src_rkey, dst_lkey, dst_rkey;

  rc = rocshmem_gin_reg_mr(qp_set, src_buf, nbytes, 0,
                             &src_mr, &src_lkey, &src_rkey);
  if (rc != 0) { fprintf(stderr, "[rank %d] FAIL: src reg_mr\n", rank); return -1; }

  rc = rocshmem_gin_reg_mr(qp_set, dst_buf, nbytes, 0,
                             &dst_mr, &dst_lkey, &dst_rkey);
  if (rc != 0) { fprintf(stderr, "[rank %d] FAIL: dst reg_mr\n", rank); return -1; }

  // Allocate and register signal buffer
  uint64_t *signal_buf = nullptr;
  HIP_CHECK(hipMalloc(&signal_buf, sizeof(uint64_t)));
  HIP_CHECK(hipMemset(signal_buf, 0, sizeof(uint64_t)));

  void *signal_mr = nullptr;
  uint32_t sig_lkey, sig_rkey;
  rc = rocshmem_gin_reg_mr(qp_set, signal_buf, sizeof(uint64_t), /*atomic=*/1,
                             &signal_mr, &sig_lkey, &sig_rkey);
  if (rc != 0) { fprintf(stderr, "[rank %d] FAIL: signal reg_mr\n", rank); return -1; }

  // Exchange dst rkeys+addrs and signal rkeys+addrs
  struct { uint32_t dst_rkey; uintptr_t dst_addr; uint32_t sig_rkey; uintptr_t sig_addr; } local_info, *all_info;
  local_info.dst_rkey = dst_rkey;
  local_info.dst_addr = (uintptr_t)dst_buf;
  local_info.sig_rkey = sig_rkey;
  local_info.sig_addr = (uintptr_t)signal_buf;
  all_info = (decltype(all_info))malloc(sizeof(*all_info) * nranks);

  MPI_Allgather(&local_info, sizeof(local_info), MPI_BYTE,
                all_info, sizeof(*all_info), MPI_BYTE, comm);

  MPI_Barrier(comm);

  int peer = (rank + 1) % nranks;

  gin_put_signal_kernel<<<1, 64>>>(
    (rocshmem::QueuePair**)gpu_qps,
    (void*)all_info[peer].dst_addr, all_info[peer].dst_rkey,
    src_buf, src_lkey,
    nbytes,
    (void*)all_info[peer].sig_addr, all_info[peer].sig_rkey,
    peer);
  HIP_CHECK(hipDeviceSynchronize());

  MPI_Barrier(comm);

  // Verify data
  int sender = (rank - 1 + nranks) % nranks;
  uint8_t expected = (uint8_t)(sender + 1);
  uint8_t *host_buf = (uint8_t*)malloc(nbytes);
  HIP_CHECK(hipMemcpy(host_buf, dst_buf, nbytes, hipMemcpyDeviceToHost));

  int errors = 0;
  for (size_t i = 0; i < nbytes; i++) {
    if (host_buf[i] != expected) {
      if (errors < 5)
        fprintf(stderr, "[rank %d] FAIL: dst[%zu] = 0x%02x, expected 0x%02x\n",
                rank, i, host_buf[i], expected);
      errors++;
    }
  }

  // Verify signal
  uint64_t host_signal = 0;
  HIP_CHECK(hipMemcpy(&host_signal, signal_buf, sizeof(uint64_t), hipMemcpyDeviceToHost));

  if (host_signal != 1) {
    fprintf(stderr, "[rank %d] FAIL: signal = %lu, expected 1\n", rank, host_signal);
    errors++;
  }

  if (errors == 0) {
    printf("[rank %d] PASS: put+signal GIN pattern (%zu bytes from rank %d, signal=%lu)\n",
           rank, nbytes, sender, host_signal);
  } else {
    fprintf(stderr, "[rank %d] FAIL: %d errors in put+signal\n", rank, errors);
  }

  free(host_buf);
  free(all_info);
  rocshmem_gin_dereg_mr(src_mr);
  rocshmem_gin_dereg_mr(dst_mr);
  rocshmem_gin_dereg_mr(signal_mr);
  HIP_CHECK(hipFree(src_buf));
  HIP_CHECK(hipFree(dst_buf));
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

  // Test 3: RDMA atomic_add (explicit rkey)
  total_failures += (test_atomic_signal(rank, nranks, MPI_COMM_WORLD) != 0);
  MPI_Barrier(MPI_COMM_WORLD);

  // Test 4: put_nbi + fenced atomic_add (GIN pattern)
  total_failures += (test_put_signal(rank, nranks, MPI_COMM_WORLD) != 0);
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
