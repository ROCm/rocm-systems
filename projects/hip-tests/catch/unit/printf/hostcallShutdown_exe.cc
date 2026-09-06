/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Companion executable for hostcallShutdown.cc, standalone because the defect it covers only
// shows up while the process is exiting, which cannot be observed from inside a shared test
// binary that has to keep running afterwards.
//
// The runtime spawns the hostcall listener when a kernel declaring hidden_hostcall_buffer is
// dispatched, and stops it when the last such queue goes away. Shutdown has to stay correct even
// when the listener thread has not been scheduled yet, so this process arranges for exactly that
// and then requires itself to exit promptly.

#include <hip/hip_runtime.h>

#include <sched.h>
#include <signal.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

namespace {
// Each cycle starts a listener and stops it again. Only the last one decides whether the runtime
// is left believing a listener is still running, so a handful is plenty.
constexpr int kCycles = 3;
constexpr int kSpinners = 4;

// A clean shutdown takes milliseconds, so anything approaching this is a hang rather than a slow
// machine.
constexpr int kExitTimeoutSeconds = 20;
constexpr int kExitTimedOut = 66;

std::atomic<bool> stopSpinners{false};

// The listener has to still be on its way to its entry point when the runtime tears it down. That
// is a scheduling window, and it opens reliably when the runtime is driven on a busy core, so
// confine everything to one CPU and keep it loaded. Pick a CPU this process is already allowed to
// use rather than assuming CPU 0, since the test may well be running inside a cpuset. Failing to
// pin is not fatal: the test simply becomes less likely to catch a regression, which beats
// failing on a host that forbids it.
void confineToOneCpu() {
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
    std::fprintf(stderr, "warning: could not read CPU affinity, race window will be narrow\n");
    return;
  }

  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (!CPU_ISSET(cpu, &allowed)) {
      continue;
    }
    cpu_set_t one;
    CPU_ZERO(&one);
    CPU_SET(cpu, &one);
    if (sched_setaffinity(0, sizeof(one), &one) != 0) {
      break;
    }
    return;
  }
  std::fprintf(stderr, "warning: could not pin to one CPU, race window will be narrow\n");
}

extern "C" void onExitTimeout(int) {
  static const char message[] = "process failed to finish exiting, hostcall shutdown hung\n";
  // Only async-signal-safe calls are legal here.
  ssize_t ignored = write(STDERR_FILENO, message, sizeof(message) - 1);
  static_cast<void>(ignored);
  _exit(kExitTimedOut);
}

// main() returning is not the end of the story here; the defect hangs in a static destructor
// afterwards, which nothing inside the process is left running to report. An alarm outlives
// main() and still fires during teardown, without leaving a thread of our own alive in the very
// shutdown path under test.
void armExitWatchdog() {
  struct sigaction action {};
  action.sa_handler = onExitTimeout;
  sigemptyset(&action.sa_mask);
  if (sigaction(SIGALRM, &action, nullptr) != 0) {
    std::fprintf(stderr, "warning: could not install exit watchdog\n");
    return;
  }
  alarm(kExitTimeoutSeconds);
}
}  // namespace

// printf() makes the compiler declare hidden_hostcall_buffer, which is what makes the runtime
// spawn the listener. The call is deliberately unreachable: a printf that actually runs would be
// serviced by the listener, which proves the listener is already up and closes the very window
// this test is trying to hit.
__global__ void declaresHostcall(int never) {
  if (never > 0) {
    printf("unreachable\n");
  }
}

int main() {
  confineToOneCpu();

  std::vector<std::thread> spinners;
  spinners.reserve(kSpinners);
  for (int i = 0; i < kSpinners; ++i) {
    spinners.emplace_back([] {
      while (!stopSpinners.load(std::memory_order_relaxed)) {
      }
    });
  }

  int status = 0;
  for (int i = 0; i < kCycles; ++i) {
    hipStream_t stream;
    hipError_t err = hipStreamCreate(&stream);
    if (err != hipSuccess) {
      std::fprintf(stderr, "hipStreamCreate failed: %s\n", hipGetErrorString(err));
      status = 1;
      break;
    }

    declaresHostcall<<<1, 1, 0, stream>>>(0);

    // A launch that never happened would leave no listener to shut down, quietly turning this
    // into a test of nothing.
    err = hipGetLastError();
    if (err != hipSuccess) {
      std::fprintf(stderr, "kernel launch failed: %s\n", hipGetErrorString(err));
      status = 1;
      break;
    }

    // Destroying the stream retires the last queue holding a hostcall buffer, so the listener is
    // torn down here, possibly before it ever ran.
    err = hipStreamDestroy(stream);
    if (err != hipSuccess) {
      std::fprintf(stderr, "hipStreamDestroy failed: %s\n", hipGetErrorString(err));
      status = 1;
      break;
    }
  }

  stopSpinners.store(true, std::memory_order_relaxed);
  for (auto& spinner : spinners) {
    spinner.join();
  }

  armExitWatchdog();
  return status;
}
