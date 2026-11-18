/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Companion executable for hostcallStackSize.cc, standalone so the static TLS block below belongs
// to the process that creates the hostcall listener. The device printf() declares
// hidden_hostcall_buffer, so dispatching the kernel creates that listener.

#include <hip/hip_runtime.h>

#include <cstdio>

namespace {
// Comfortably above the 256 KiB stack the hostcall listener used to request.
constexpr size_t kTlsPadBytes = 512 * 1024;
constexpr int kThreads = 4;
}  // namespace

// Initial-exec keeps this in the executable's PT_TLS segment, so it counts towards static TLS.
__thread char tlsPad[kTlsPadBytes] __attribute__((tls_model("initial-exec")));

__global__ void greet() { printf("hostcall serviced\n"); }

int main() {
  // Touch both ends so the whole block is committed and cannot be elided.
  tlsPad[0] = 1;
  tlsPad[kTlsPadBytes - 1] = 2;

  greet<<<1, kThreads>>>();

  hipError_t err = hipDeviceSynchronize();
  if (err != hipSuccess) {
    std::fprintf(stderr, "hipDeviceSynchronize failed: %s\n", hipGetErrorString(err));
    return 1;
  }

  if (tlsPad[0] != 1 || tlsPad[kTlsPadBytes - 1] != 2) {
    std::fprintf(stderr, "TLS block was corrupted\n");
    return 1;
  }

  return 0;
}
