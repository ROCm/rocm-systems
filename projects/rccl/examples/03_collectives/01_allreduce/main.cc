/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "hip/hip_runtime.h"
#include <rccl/rccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/*
 * NCCL AllReduce Example - Collective Communication
 *
 * This example demonstrates the fundamental AllReduce collective operation
 * using NCCL's single-process, multi-GPU approach. AllReduce is one of the most
 * important collective operations in distributed and parallel computing.
 *
 * Learning Objectives:
 * - Understand AllReduce collective communication pattern
 * - Learn NCCL single-process multi-GPU programming model
 * - See how data reduction works across multiple devices
 * - Practice verification and validation of collective results
 *
 */

// Enhanced error checking macro for NCCL operations
// Provides detailed error information including the failed operation
#define NCCLCHECK(cmd)                                                         \
  do {                                                                         \
    ncclResult_t res = cmd;                                                    \
    if (res != ncclSuccess) {                                                  \
      fprintf(stderr, "Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__,   \
              ncclGetErrorString(res));                                        \
      fprintf(stderr, "Failed NCCL operation: %s\n", #cmd);                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUDACHECK(cmd)                                                         \
  do {                                                                         \
    hipError_t err = cmd;                                                     \
    if (err != hipSuccess) {                                                  \
      fprintf(stderr, "Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__,   \
              hipGetErrorString(err));                                        \
      fprintf(stderr, "Failed CUDA operation: %s\n", #cmd);                    \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

int main(int argc, char *argv[]) {
  // ========================================================================
  // STEP 1: Initialize Variables and Detect Available GPUs
  // ========================================================================

  int num_gpus = 0;
  ncclComm_t *comms;
  hipStream_t *streams;
  float **sendbuff;
  float **recvbuff;

  // Get number of CUDA devices
  CUDACHECK(hipGetDeviceCount(&num_gpus));
  if (num_gpus < 1) {
    printf("No CUDA devices found\n");
    return EXIT_FAILURE;
  }

  printf("Using %d devices for collective communication\n", num_gpus);

  // ========================================================================
  // STEP 2: Allocate Memory for Communicators, Streams, and Data Buffers
  // ========================================================================

  // Allocate arrays for per-device resources, and array of pointers for buffers
  comms = (ncclComm_t *)malloc(num_gpus * sizeof(ncclComm_t));
  streams = (hipStream_t *)malloc(num_gpus * sizeof(hipStream_t));
  sendbuff = (float **)malloc(num_gpus * sizeof(float *));
  recvbuff = (float **)malloc(num_gpus * sizeof(float *));

  printf("Memory allocated for %d communicators and streams\n", num_gpus);

  // ========================================================================
  // STEP 3: Initialize NCCL Communicators for All Devices
  // ========================================================================

  // ncclCommInitAll creates communicators for all devices in one call
  // This is the simplest way to set up NCCL for single-process applications
  NCCLCHECK(ncclCommInitAll(comms, num_gpus, NULL));
  printf("NCCL communicators initialized for all devices\n");

  // ========================================================================
  // STEP 4: Create CUDA Streams and Allocate Device Memory
  // ========================================================================

  const size_t size = 32 * 1024 * 1024; // 32M floats for demonstration

  for (int i = 0; i < num_gpus; i++) {
    // Set device context for each GPU
    CUDACHECK(hipSetDevice(i));

    // Create stream for asynchronous operations
    CUDACHECK(hipStreamCreate(&streams[i]));

    // Allocate device memory for send and receive buffers
    CUDACHECK(hipMalloc((void **)&sendbuff[i], size * sizeof(float)));
    CUDACHECK(hipMalloc((void **)&recvbuff[i], size * sizeof(float)));

    // Initialize send buffer: zero the entire buffer, then set first element to
    // rank
    CUDACHECK(hipMemset(sendbuff[i], 0, size * sizeof(float)));
    float rank_value = (float)i;
    CUDACHECK(hipMemcpy(sendbuff[i], &rank_value, sizeof(float),
                         hipMemcpyHostToDevice));

    printf("  Device %d initialized with data value %d\n", i, i);
  }

  // ========================================================================
  // STEP 5: Perform AllReduce Sum Operation
  // ========================================================================

  printf("Starting collective sum operation across all devices\n");

  // NOTE: ncclGroupStart and ncclGroupEnd are essential to avoid
  // deadlock when using ncclCommInitAll and multiple communication calls.
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < num_gpus; i++) {
    // Each device performs combines all contributions and distributes result
    NCCLCHECK(ncclAllReduce(sendbuff[i], recvbuff[i], size, ncclFloat, ncclSum,
                            comms[i], streams[i]));
  }
  NCCLCHECK(ncclGroupEnd());

  // Synchronize all streams to ensure completion
  for (int i = 0; i < num_gpus; i++) {
    CUDACHECK(hipSetDevice(i));
    CUDACHECK(hipStreamSynchronize(streams[i]));
  }

  printf("Collective operation completed\n");

  // ========================================================================
  // STEP 6: Verify Results and Validate Correctness
  // ========================================================================

  // Expected result: sum of all ranks = 0 + 1 + 2 + ... + (num_gpus-1)
  // Note: We only check the first element since that's all we initialized
  float expected = (float)(num_gpus * (num_gpus - 1) / 2);
  printf("Verifying results (expected sum: %.0f)\n", expected);

  bool success = true;
  for (int i = 0; i < num_gpus; i++) {
    float result;
    CUDACHECK(hipSetDevice(i));
    CUDACHECK(hipMemcpy(&result, recvbuff[i], sizeof(float),
                         hipMemcpyDeviceToHost));

    if (result != expected) {
      printf("  Device %d received incorrect result: %.0f (expected %.0f)\n", i,
             result, expected);
      success = false;
    } else {
      printf("  Device %d correctly received sum: %.0f\n", i, result);
    }
  }

  // ========================================================================
  // STEP 7: Cleanup Resources and Report Results
  // ========================================================================

  // Destroy NCCL communicators
  for (int i = 0; i < num_gpus; i++) {
    NCCLCHECK(ncclCommFinalize(comms[i]));
    NCCLCHECK(ncclCommDestroy(comms[i]));
  }

  // Free device memory and destroy streams
  for (int i = 0; i < num_gpus; i++) {
    CUDACHECK(hipSetDevice(i));
    CUDACHECK(hipFree(sendbuff[i]));
    CUDACHECK(hipFree(recvbuff[i]));
    CUDACHECK(hipStreamDestroy(streams[i]));
  }

  // Free host memory
  free(comms);
  free(streams);
  free(sendbuff);
  free(recvbuff);

  if (success) {
    printf("Example completed successfully!\n");
  } else {
    printf("Example failed - incorrect results detected\n");
    return EXIT_FAILURE;
  }

  return 0;
}
