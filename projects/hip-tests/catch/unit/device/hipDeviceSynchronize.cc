/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#include <cstdlib>

/**
 * @addtogroup hipDeviceSynchronize hipDeviceSynchronize
 * @{
 * @ingroup DeviceTest
 * `hipDeviceSynchronize(void)` -
 * Waits on all active streams on current device.
 * When this command is invoked, the host thread gets blocked until all the commands associated
 * with streams associated with the device. HIP does not support multiple blocking modes (yet!).
 */

#define _SIZE sizeof(int) * 1024 * 1024
#define NUM_STREAMS 2
#define NUM_ITERS 1 << 30

namespace {

// Unit_hipDeviceSynchronize_Functional reads host memory before hipDeviceSynchronize().
// That only works if per-stream H2D (and thus kernel/D2H) has not finished yet; a small
// copy size lets very fast GPUs finish the whole stream before the CPU runs the next
// instruction. Use a larger transfer for this test only (not global _SIZE, which would
// bloat Unit_hipDeviceSynchronize_Positive_Nullstream). Override with
// HIP_TEST_DEVICE_SYNCHRONIZE_FUNCTIONAL_COPY_MB (megabytes, 1-8192) if the pre-sync
// assertion still races on your hardware.
size_t functionalDeviceSynchronizeCopyBytes() {
  if (const char* env = std::getenv("HIP_TEST_DEVICE_SYNCHRONIZE_FUNCTIONAL_COPY_MB")) {
    char* end = nullptr;
    unsigned long mb = std::strtoul(env, &end, 10);
    if (end != env && mb > 0 && mb <= 8192) {
      return mb * 1024ULL * 1024ULL;
    }
  }
  // 1 GiB per buffer by default. The large transfer is intentional: it widens the window
  // during which the pre-sync hipStreamQuery still returns hipErrorNotReady, making the
  // best-effort pre-sync assertion (guarded by hipStreamQuery) effective on typical
  // hardware. A smaller value, e.g. 64 MiB, is enough for the copy to finish before the
  // host check on fast ASICs, causing the assertion to be silently skipped every run.
  // Use HIP_TEST_DEVICE_SYNCHRONIZE_FUNCTIONAL_COPY_MB to reduce on memory-constrained
  // runners where the 1 GiB default cannot be satisfied.
  return 1024ULL * 1024 * 1024;
}

}  // namespace

static __global__ void Iter(int* Ad, int num) {
  int tx = threadIdx.x + blockIdx.x * blockDim.x;
  // Kernel loop designed to execute very slowly.
  // so we can test timing-related
  // behavior below
  if (tx == 0) {
    for (int i = 0; i < num; i++) {
      Ad[tx] += 1;
    }
  }
}

/**
 * Test Description
 * ------------------------
 *  - Performs synchronization when no work is enqueued on stream,
 *    utilizing multiple devices.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceSynchronize.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipDeviceSynchronize_Positive_Empty_Streams) {
  const auto device = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(device));
  INFO("Current device: " << device);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *  - Performs synchronization between large kernel execution
 *    and asynchronous copying of the array, on default(null) stream.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceSynchronize.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipDeviceSynchronize_Positive_Nullstream) {
  const auto device = GENERATE(range(0, HipTest::getDeviceCount()));
  HIP_CHECK(hipSetDevice(device));
  INFO("Current device: " << device);

  int *A_h = nullptr, *A_d = nullptr;
  HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&A_h), _SIZE, hipHostMallocDefault));
  A_h[0] = 1;
  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&A_d), _SIZE));

  HIP_CHECK(hipMemcpyAsync(A_d, A_h, _SIZE, hipMemcpyHostToDevice, NULL));
  hipLaunchKernelGGL(HIP_KERNEL_NAME(Iter), dim3(1), dim3(1), 0, NULL, A_d, 1 << 30);
  HIP_CHECK(hipMemcpyAsync(A_h, A_d, _SIZE, hipMemcpyDeviceToHost, NULL));

  HIP_CHECK(hipDeviceSynchronize());
  REQUIRE(1 << 30 == A_h[0] - 1);
  HIP_CHECK(hipHostFree(A_h));
  HIP_CHECK(hipFree(A_d));
}

/**
 * Test Description
 * ------------------------
 *  - Performs synchronization between large kernel execution
 *    and asynchronous copying of the array, on multiple streams.
 *    Uses a larger copy than \c _SIZE so the pre-sync host read usually wins its race;
 *    see functionalDeviceSynchronizeCopyBytes() and
 *    HIP_TEST_DEVICE_SYNCHRONIZE_FUNCTIONAL_COPY_MB.
 * Test source
 * ------------------------
 *  - unit/device/hipDeviceSynchronize.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_hipDeviceSynchronize_Functional) {
  const size_t copyBytes = functionalDeviceSynchronizeCopyBytes();
  int* A[NUM_STREAMS];
  int* Ad[NUM_STREAMS];
  hipStream_t stream[NUM_STREAMS];

  for (int i = 0; i < NUM_STREAMS; i++) {
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&A[i]), copyBytes, hipHostMallocDefault));
    A[i][0] = 1;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&Ad[i]), copyBytes));
    HIP_CHECK(hipStreamCreate(&stream[i]));
  }
  for (int i = 0; i < NUM_STREAMS; i++) {
    HIP_CHECK(hipMemcpyAsync(Ad[i], A[i], copyBytes, hipMemcpyHostToDevice, stream[i]));
  }
  for (int i = 0; i < NUM_STREAMS; i++) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(Iter), dim3(1), dim3(1), 0, stream[i], Ad[i], NUM_ITERS);
  }
  for (int i = 0; i < NUM_STREAMS; i++) {
    HIP_CHECK(hipMemcpyAsync(A[i], Ad[i], copyBytes, hipMemcpyDeviceToHost, stream[i]));
  }


  // Best-effort pre-sync check: assert that the last stream's D2H copy has not yet
  // landed in host memory. We gate this on hipStreamQuery so that very fast ASICs (or
  // HIP_LAUNCH_BLOCKING=1 environments) where the GPU finishes before the host reaches
  // this point simply skip the assertion rather than fail. The large copyBytes buffer
  // (default 1 GiB) widens the race window so this check fires on typical hardware.
  if (hipStreamQuery(stream[NUM_STREAMS - 1]) == hipErrorNotReady) {
    REQUIRE(NUM_ITERS != A[NUM_STREAMS - 1][0] - 1);
  }
  HIP_CHECK(hipDeviceSynchronize());
  REQUIRE(NUM_ITERS == A[NUM_STREAMS - 1][0] - 1);
  for (int i = 0; i < NUM_STREAMS; i++) {
    HIP_CHECK(hipHostFree(A[i]));
    HIP_CHECK(hipFree(Ad[i]));
    HIP_CHECK(hipStreamDestroy(stream[i]));
  }
}


/**
 * End doxygen group DeviceTest.
 * @}
 */
