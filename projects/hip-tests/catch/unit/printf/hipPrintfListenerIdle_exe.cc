/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Companion executable for hipPrintfListenerIdle.cc, which starts it with HSA_ENABLE_INTERRUPT=0
// in its environment (SpawnProc::setEnv; set here again before the first HIP call, for runs by
// hand), so the hostcall doorbell is a busy-wait signal, as on WSL, also on a native Linux machine.
// Then it runs one kernel that calls printf (built with -mprintf-kind=hostcall), which starts the
// hostcall listener, and reports how much CPU the process uses while it sleeps.

#include <hip/hip_runtime.h>

#include <time.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

__global__ void listenerIdleGreet() { printf("hostcall serviced\n"); }

static double ProcessCpuSeconds() {
  timespec ts;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
  // Read by the runtime when HIP initializes, so it must be set before the first HIP call.
  setenv("HSA_ENABLE_INTERRUPT", "0", 1);

  listenerIdleGreet<<<1, 1>>>();
  hipError_t err = hipDeviceSynchronize();
  if (err != hipSuccess) {
    std::fprintf(stderr, "hipDeviceSynchronize failed: %s\n", hipGetErrorString(err));
    return 1;
  }
  std::fflush(stdout);

  // One window: a spinning listener uses a whole core for all of it.
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  const double cpu0 = ProcessCpuSeconds();
  const auto t0 = std::chrono::steady_clock::now();
  std::this_thread::sleep_for(std::chrono::milliseconds(1500));
  const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  const double percent = (ProcessCpuSeconds() - cpu0) / wall * 100.0;
  std::printf("idle_cpu_percent %.1f\n", percent);
  return 0;
}
