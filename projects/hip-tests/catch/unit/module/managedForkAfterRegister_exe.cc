/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>

#include <cerrno>
#include <cstring>
#include <iostream>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

__managed__ int g_managedVal = 7;

__global__ void writeManaged(int value) { g_managedVal = value; }

namespace {

constexpr int kSentinel = 99;

// Returns the child's exit status, or -1 when it did not exit on its own.
int runManagedTouchInChild() {
  // Empty the shared output buffer first, so the child does not re-emit what the parent
  // has already queued.
  std::cout.flush();

  const pid_t pid = fork();
  if (pid < 0) {
    std::cout << "fork failed: " << std::strerror(errno) << std::endl;
    return -1;
  }

  if (pid == 0) {
    writeManaged<<<1, 1>>>(kSentinel);
    const hipError_t launch = hipGetLastError();
    const hipError_t sync = hipDeviceSynchronize();
    const int observed = g_managedVal;

    std::cout << "child: launch=" << hipGetErrorString(launch)
              << " sync=" << hipGetErrorString(sync) << " g_managedVal=" << observed << std::endl;

    const bool ok = (launch == hipSuccess && sync == hipSuccess && observed == kSentinel);
    _exit(ok ? 0 : 1);
  }

  int status = 0;
  while (waitpid(pid, &status, 0) < 0) {
    if (errno != EINTR) {
      std::cout << "waitpid failed: " << std::strerror(errno) << std::endl;
      return -1;
    }
  }
  if (WIFSIGNALED(status)) {
    std::cout << "child terminated by signal " << WTERMSIG(status) << std::endl;
    return -1;
  }
  if (!WIFEXITED(status)) {
    std::cout << "child did not exit normally (status 0x" << std::hex << status << std::dec << ")"
              << std::endl;
    return -1;
  }
  return WEXITSTATUS(status);
}

}  // namespace

int main() {
  const int result = runManagedTouchInChild();
  if (result == 0) {
    std::cout << "OK: forked child initialized and wrote the managed variable" << std::endl;
  }
  return result;
}
