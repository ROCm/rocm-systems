/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <unistd.h>
#include <cstdint>
#include <iostream>

// Exported by the AMD runtime, not declared in a public header. A registered callback makes the
// runtime report activity for every queue, which it does from its async signal handler.
extern "C" void hipRegisterTracerCallback(int (*function)(uint32_t domain, uint32_t operation_id,
                                                          void* data));

// Enables activity reporting (domain HIP_OPS) and leaves API tracing off. The activity records
// passed in are ignored.
static int Tracer(uint32_t domain, uint32_t, void*) {
  constexpr uint32_t kActivityDomainHipOps = 2;
  return domain == kActivityDomainHipOps ? 0 : 1;
}

static __global__ void Spin(long long ticks) {
  const long long start = wall_clock64();
  while (wall_clock64() - start < ticks) {
  }
}

// Creates and destroys streams with a CU mask while activity reporting is on. The async signal
// handler can then drop the last reference to such a stream; destroying its hardware queue on
// that thread waits for the same thread and hangs. The alarm turns a hang into a failure.
int main() {
  constexpr int kStreams = 200;
  alarm(60);
  hipRegisterTracerCallback(Tracer);

  int clock_khz = 0;
  if (hipDeviceGetAttribute(&clock_khz, hipDeviceAttributeWallClockRate, 0) != hipSuccess) {
    std::cout << "hipDeviceGetAttribute failed" << std::endl;
    return 1;
  }
  const uint32_t cu_mask = 0xffff;
  for (int i = 0; i < kStreams; ++i) {
    hipStream_t stream;
    if (hipExtStreamCreateWithCUMask(&stream, 1, &cu_mask) != hipSuccess) {
      std::cout << "hipExtStreamCreateWithCUMask failed at stream " << i << std::endl;
      return 1;
    }
    // Up to 2 ms of work, so the stream is destroyed both while it runs and after it finished
    Spin<<<1, 1, 0, stream>>>(static_cast<long long>(clock_khz) * (i % 3));
    // The event ends a batch of work, whose completion the async signal handler reports
    hipEvent_t event;
    if (hipEventCreate(&event) != hipSuccess || hipEventRecord(event, stream) != hipSuccess ||
        hipEventDestroy(event) != hipSuccess) {
      std::cout << "event record failed at stream " << i << std::endl;
      return 1;
    }
    if (hipStreamDestroy(stream) != hipSuccess) {
      std::cout << "hipStreamDestroy failed at stream " << i << std::endl;
      return 1;
    }
  }
  if (hipDeviceSynchronize() != hipSuccess) {
    std::cout << "hipDeviceSynchronize failed" << std::endl;
    return 1;
  }
  return 0;
}
