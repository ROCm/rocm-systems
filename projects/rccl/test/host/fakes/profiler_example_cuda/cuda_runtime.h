/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fake CUDA runtime for profiler-example-ce-test.cc, which compiles the unhipified
// plugins/profiler/example/profiler_plugin_ce.cc on the host. Only that one TU includes
// it, so the calls are internal-linkage and count into g_fakeCuda.

#ifndef RCCL_TEST_HOST_FAKE_CUDA_RUNTIME_H_
#define RCCL_TEST_HOST_FAKE_CUDA_RUNTIME_H_

#include <cstdint>

typedef int cudaError_t;
#define cudaSuccess 0
#define cudaErrorNotReady 600
#define cudaEventDisableTiming 0x02

typedef struct FakeCudaEvent* cudaEvent_t;
typedef struct FakeCudaStream* cudaStream_t;

enum cudaStreamCaptureStatus {
  cudaStreamCaptureStatusNone = 0,
  cudaStreamCaptureStatusActive = 1,
  cudaStreamCaptureStatusInvalidated = 2
};

static struct {
  cudaStreamCaptureStatus captureStatus;
  int eventsCreated;
  int eventsRecorded;
  int eventsDestroyed;
} g_fakeCuda;

static inline cudaError_t cudaStreamIsCapturing(cudaStream_t, cudaStreamCaptureStatus* status) {
  *status = g_fakeCuda.captureStatus;
  return cudaSuccess;
}
static inline cudaError_t cudaGetLastError(void) { return cudaSuccess; }
static inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned) {
  *e = (cudaEvent_t)(uintptr_t)(++g_fakeCuda.eventsCreated);
  return cudaSuccess;
}
static inline cudaError_t cudaEventDestroy(cudaEvent_t) { g_fakeCuda.eventsDestroyed++; return cudaSuccess; }
static inline cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t) { g_fakeCuda.eventsRecorded++; return cudaSuccess; }
// Never complete, so nothing here retires an event behind the test's back.
static inline cudaError_t cudaEventQuery(cudaEvent_t) { return cudaErrorNotReady; }
static inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t, cudaEvent_t) { *ms = 0; return cudaSuccess; }

#endif  // RCCL_TEST_HOST_FAKE_CUDA_RUNTIME_H_
