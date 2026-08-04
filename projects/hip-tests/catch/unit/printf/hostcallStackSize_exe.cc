/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Companion executable for hostcallStackSize.cc. It must be a standalone exe because the static
// TLS block below has to belong to the process that creates the hostcall listener, and because
// the driving test needs a process whose stack size flag can be set before runtime init.
//
// glibc allocates a thread's static TLS out of the stack mapping requested through
// pthread_attr_setstacksize(), so a request that cannot also hold the process' static TLS is
// refused with EINVAL. TLS_PAD_KIB bytes of initial-exec TLS put this process in the same
// position as an application that links libraries with large PT_TLS segments.
//
// The device printf() makes the compiler declare hidden_hostcall_buffer, so dispatching the
// kernel creates the listener thread.

#include <hip/hip_runtime.h>

#include <cstdio>

#ifndef TLS_PAD_KIB
#define TLS_PAD_KIB 512
#endif

namespace {
constexpr size_t kTlsPadBytes = static_cast<size_t>(TLS_PAD_KIB) * 1024;
constexpr int kThreads = 4;
}  // namespace

// External linkage plus initial-exec keeps this in the executable's PT_TLS segment so that it is
// accounted for in the process' static TLS size at startup.
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
