/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#include <hip_test_common.hh>

#include <thread>
#include <vector>

namespace {

// Passing 'magic' at runtime stops the compiler from eliminating the printf as
// dead code, so the kernel retains its hidden hostcall-buffer argument. That
// forces the runtime to register a hostcall buffer (and start the listener) at
// dispatch time. Callers always pass 0, so nothing is ever actually printed.
__global__ void kernel_touch_hostcall(int magic) {
  if (magic == 12345) {
    printf("unreachable %d\n", magic);
  }
}

void stress_hostcall_lifecycle(unsigned iterations) {
  for (unsigned i = 0; i < iterations; ++i) {
    hipStream_t stream = nullptr;
    HIP_CHECK_THREAD(hipStreamCreate(&stream));
    kernel_touch_hostcall<<<1, 1, 0, stream>>>(0);
    HIP_CHECK_THREAD(hipGetLastError());
    HIP_CHECK_THREAD(hipStreamSynchronize(stream));
    HIP_CHECK_THREAD(hipStreamDestroy(stream));
  }
}

}  // namespace

// Regression test for the hostcall listener teardown race.
//
// The global hostcall listener is created when the first printf-capable kernel
// is dispatched and torn down when the last stream using it is destroyed. The
// bug: the teardown and re-creation ran outside the listener lock, so one thread
// could destroy the listener while another re-created or used it, causing a
// double-free / use-after-free or a hang.
//
// Many threads repeatedly create a stream, dispatch a kernel, sync and destroy
// the stream so teardowns and creations overlap. It is a probabilistic
// reproducer: it needs sustained repetition (hence the high iteration count) and
// is most powerful under AddressSanitizer.
HIP_TEST_CASE(Stress_printf_HostcallListenerRace) {

  // 4 threads is the sweet spot: enough concurrency to overlap a
  // create with a teardown, while still letting the buffer count reach zero
  // often enough to trigger teardowns. The high iteration count is needed
  // because fewer iterations sharply drop the reproduction rate.
  constexpr unsigned kNumThreads = 4;
  constexpr unsigned kIterations = 2000;

  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (unsigned i = 0; i < kNumThreads; ++i) {
    threads.emplace_back(stress_hostcall_lifecycle, kIterations);
  }
  for (auto& t : threads) {
    t.join();
  }

  HIP_CHECK_THREAD_FINALIZE();
}
