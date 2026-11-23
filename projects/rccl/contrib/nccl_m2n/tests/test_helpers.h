/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * test_helpers.h — Bootstrap-agnostic CUDA helpers for the basic_api
 * test programs.
 *
 * Mirrors the validation kernels in benchmarks/bench_common.h, but does
 * not depend on MPI so it can be reused from both the MPI and the local
 * (single-process, multi-thread) test binaries.
 *
 * Definitions live in test_helpers.cu (kernels + launchers need nvcc).
 * Callers include only this .h and link the .cu's object file.
 ************************************************************************/

#ifndef TESTS_TEST_HELPERS_H_
#define TESTS_TEST_HELPERS_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <nccl.h>

#include "m2n_checks.h"

/* ======================================================================
 * Env-mutation chokepoints.
 *
 * setenv / unsetenv are documented mt-unsafe per POSIX with no portable
 * mt-safe alternative. Test harness only invokes them pre-MPI_Init /
 * before gtest spawns any worker, so the race-with-concurrent-setenv
 * concern doesn't apply — but the check still fires per call site.
 * Route through these helpers so the suppression lives in one place.
 * ====================================================================*/

static inline void testSetEnv(const char* name, const char* value) {
  // NOLINTNEXTLINE(concurrency-mt-unsafe) — pre-MPI_Init env propagation
  setenv(name, value, 1);
}
static inline void testUnsetEnv(const char* name) {
  // NOLINTNEXTLINE(concurrency-mt-unsafe) — pre-MPI_Init env propagation
  unsetenv(name);
}

static inline bool testUsesNonBlockingComm() {
  const char* value = getenv("NCCL_COMM_BLOCKING");
  return value != nullptr && strcmp(value, "0") == 0;
}

/* ======================================================================
 * Error-checking macros (no MPI dependency).
 * ====================================================================*/

#define TEST_CUDACHECK(cmd)                                                                 \
  do {                                                                                      \
    cudaError_t e = (cmd);                                                                  \
    if (e != cudaSuccess) {                                                                 \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      abort();                                                                              \
    }                                                                                       \
  } while (0)

#define TEST_NCCLCHECK(cmd)                                                                 \
  do {                                                                                      \
    ncclResult_t r = (cmd);                                                                 \
    if (r != ncclSuccess && r != ncclInProgress) {                                          \
      fprintf(stderr, "NCCL error %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
      abort();                                                                              \
    }                                                                                       \
  } while (0)

/* Query the communicator after every NCCL call. A successful return can still
 * leave the operation pending on a non-blocking communicator. */
static inline ncclResult_t testWaitCommReady(ncclComm_t comm) {
  return m2nWaitCommReady(comm);
}

static inline void testWaitComms(const std::vector<ncclComm_t>& comms) {
  for (ncclComm_t comm : comms) {
    TEST_NCCLCHECK(testWaitCommReady(comm));
  }
}

#define TEST_NCCL_COMM_CHECK(comm, cmd)                                                        \
  do {                                                                                         \
    ncclResult_t r = (cmd);                                                                    \
    if (r != ncclSuccess && r != ncclInProgress) {                                              \
      fprintf(stderr, "NCCL error %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(r));   \
      abort();                                                                                  \
    }                                                                                           \
    r = testWaitCommReady((comm));                                                              \
    if (r != ncclSuccess) {                                                                    \
      fprintf(stderr, "NCCL error %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
      abort();                                                                                 \
    }                                                                                          \
  } while (0)

/* Supply comm to wait after the call. Omit it for calls such as destroy,
 * which invalidate the communicator and cannot be polled afterward. */
#define ASSERT_NCCL(cmd, ...) \
  do { \
    ncclResult_t ncclAssertResult = (cmd); \
    auto checkComm = [ncclAssertResult](ncclComm_t comm = nullptr) { \
      if (comm != nullptr) { \
        return testWaitCommReady(comm); \
      } \
      return ncclAssertResult; \
    }; \
    if (ncclAssertResult == ncclSuccess || ncclAssertResult == ncclInProgress) { \
      ncclAssertResult = checkComm(__VA_ARGS__); \
    } \
    ASSERT_EQ(ncclSuccess, ncclAssertResult) << #cmd; \
  } while (0)

static inline ncclResult_t testFinalizeComm(ncclComm_t comm) {
  const ncclResult_t result = ncclCommFinalize(comm);
  if (result != ncclSuccess && result != ncclInProgress) return result;
  return testWaitCommReady(comm);
}

static inline ncclResult_t testDestroyComm(ncclComm_t comm) {
  ncclResult_t result = testFinalizeComm(comm);
  if (result != ncclSuccess) return result;
  return ncclCommDestroy(comm);
}

static inline ncclResult_t testCheckCommStreamResult(ncclComm_t comm, cudaStream_t stream) {
  ncclResult_t result = testWaitCommReady(comm);
  if (result != ncclSuccess) return result;
  const cudaError_t cudaResult = cudaStreamSynchronize(stream);
  if (cudaResult != cudaSuccess) return ncclSystemError;
  return testWaitCommReady(comm);
}

static inline void testCheckCommStream(ncclComm_t comm, cudaStream_t stream, const char* file, int line) {
  const ncclResult_t result = testCheckCommStreamResult(comm, stream);
  if (result != ncclSuccess) {
    fprintf(stderr, "NCCL error %s:%d: %s\n", file, line, ncclGetErrorString(result));
    abort();
  }
}

#define TEST_NCCL_ASYNC_CHECK(comm, stream) testCheckCommStream((comm), (stream), __FILE__, __LINE__)

/* ======================================================================
 * Byte-pattern initializer / validator.
 *
 * Each byte in the global tensor takes value `(globalIdx % 256)` where
 * `globalIdx = g0 + g1 * G1 + g2 * G1 * G2`. The pattern is stable
 * across elementSize and across any same-dim or cross-dim sharding —
 * the validator only consults the byte's *global index*, not where the
 * data was sourced from.
 *
 * `localByteDims` are dimensions in *bytes* (not elements); the caller
 * multiplies the innermost tensor dim by elementSize so the byte
 * pattern is consistent across dtypes.
 * ====================================================================*/

void testInitSourceData(char* buffer, const size_t localByteDims[3], int ndims,
                        int shardDim, /* tensor dim that is sharded; -1 if replicated */
                        int shardIdx, /* this rank's shard index along shardDim */
                        int shardCount, /* number of shards along shardDim */
                        cudaStream_t stream);

bool testValidateDestData(const char* buffer, const size_t localByteDims[3], int ndims, int shardDim, int shardIdx,
                          int shardCount, int worldRank, cudaStream_t stream,
                          unsigned long long* outErrorCount = nullptr);

#endif /* TESTS_TEST_HELPERS_H_ */
