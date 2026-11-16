/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipStreamCompletionSignals hipStreamCompletionSignals
 * @{
 * @ingroup StreamTest
 */

#include <hip_test_common.hh>
#include <utils.hh>

#include <atomic>

/**
 * Test Description
 * ------------------------
 *  - Checks the operations whose completion signals the runtime hints to the command processor
 *    when its firmware decodes signal hints: blocking-sync event waits, whose signals interrupt
 *    only when they reach zero; stream callbacks and host functions, whose signals start at 2 and
 *    interrupt on the first decrement; and event records with and without timing, which turn
 *    timestamps on or off per packet.
 *  - Each kernel increments a counter after spinning, so a wait or callback that runs before the
 *    kernel finished sees a short count.
 * Test source
 * ------------------------
 * - unit/stream/hipStreamCompletionSignals.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.0
 */

namespace {

constexpr int kRounds = 100;
constexpr unsigned long long kSpinUs = 200;

__global__ void SpinThenCount(unsigned long long spin_ticks, unsigned int* counter) {
#if HT_AMD
  const long long start = wall_clock64();
  while (wall_clock64() - start < static_cast<long long>(spin_ticks)) {
  }
#else
  const long long start = clock64();
  while (clock64() - start < static_cast<long long>(spin_ticks)) {
  }
#endif
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    atomicAdd(counter, 1u);
  }
}

struct CallbackOrder {
  volatile unsigned int* counter = nullptr;
  std::atomic<unsigned int> num_calls{0};
  std::atomic<unsigned int> num_out_of_order{0};
};

// One kernel precedes each callback, and the stream holds the next kernel until it returns
void CheckCallbackOrder(void* user_data) {
  auto* order = static_cast<CallbackOrder*>(user_data);
  if (*order->counter != ++order->num_calls) {
    order->num_out_of_order++;
  }
}

void StreamCallback(hipStream_t, hipError_t, void* user_data) { CheckCallbackOrder(user_data); }

}  // namespace

/**
 * Test Description
 * ------------------------
 *  - hipEventSynchronize on a hipEventBlockingSync event returns only after the kernel recorded
 *    before the event finished.
 */
HIP_TEST_CASE(Unit_hipEventSynchronize_BlockingSyncAfterKernel) {
  const unsigned long long spin_ticks = kSpinUs * SpinTicksPerUs();
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  unsigned int* counter = nullptr;
  HIP_CHECK(hipHostMalloc(&counter, sizeof(unsigned int)));
  *counter = 0;
  hipEvent_t blocking_event;
  HIP_CHECK(hipEventCreateWithFlags(&blocking_event, hipEventBlockingSync));

  for (int round = 0; round < kRounds; ++round) {
    SpinThenCount<<<1, 64, 0, stream>>>(spin_ticks, counter);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(blocking_event, stream));
    HIP_CHECK(hipEventSynchronize(blocking_event));
    INFO("round " << round);
    REQUIRE(*reinterpret_cast<volatile unsigned int*>(counter) == round + 1u);
    HIP_CHECK(hipEventQuery(blocking_event));
  }

  HIP_CHECK(hipEventDestroy(blocking_event));
  HIP_CHECK(hipHostFree(counter));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *  - Kernels alternate with hipStreamAddCallback and hipLaunchHostFunc on one stream. Each
 *    callback must run after the kernel before it and before the kernel after it, and the stream
 *    synchronize must wait for all of them.
 */
HIP_TEST_CASE(Unit_hipStreamAddCallback_OrderedWithKernels) {
  const unsigned long long spin_ticks = kSpinUs * SpinTicksPerUs();
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  unsigned int* counter = nullptr;
  HIP_CHECK(hipHostMalloc(&counter, sizeof(unsigned int)));
  *counter = 0;
  CallbackOrder order;
  order.counter = counter;

  for (int round = 0; round < kRounds; ++round) {
    SpinThenCount<<<1, 64, 0, stream>>>(spin_ticks, counter);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipStreamAddCallback(stream, StreamCallback, &order, 0));
    SpinThenCount<<<1, 64, 0, stream>>>(spin_ticks, counter);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipLaunchHostFunc(stream, CheckCallbackOrder, &order));
  }
  HIP_CHECK(hipStreamSynchronize(stream));

  REQUIRE(order.num_calls == 2u * kRounds);
  REQUIRE(order.num_out_of_order == 0u);
  REQUIRE(*reinterpret_cast<volatile unsigned int*>(counter) == 2u * kRounds);

  HIP_CHECK(hipHostFree(counter));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * Test Description
 * ------------------------
 *  - Two timed events around a spinning kernel measure at least the spin, also with an untimed
 *    event recorded between them, and an untimed event pair refuses hipEventElapsedTime.
 */
HIP_TEST_CASE(Unit_hipEventElapsedTime_TimedAndUntimedRecords) {
  constexpr unsigned long long kTimedSpinUs = 2000;
  const unsigned long long spin_ticks = kTimedSpinUs * SpinTicksPerUs();
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  unsigned int* counter = nullptr;
  HIP_CHECK(hipMalloc(&counter, sizeof(unsigned int)));
  hipEvent_t timed_start, timed_stop, untimed_middle, untimed_stop;
  HIP_CHECK(hipEventCreate(&timed_start));
  HIP_CHECK(hipEventCreate(&timed_stop));
  HIP_CHECK(hipEventCreateWithFlags(&untimed_middle, hipEventDisableTiming));
  HIP_CHECK(hipEventCreateWithFlags(&untimed_stop, hipEventDisableTiming));

  for (int round = 0; round < kRounds / 10; ++round) {
    HIP_CHECK(hipEventRecord(timed_start, stream));
    SpinThenCount<<<1, 64, 0, stream>>>(spin_ticks, counter);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(untimed_middle, stream));
    SpinThenCount<<<1, 64, 0, stream>>>(spin_ticks, counter);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(timed_stop, stream));
    HIP_CHECK(hipEventRecord(untimed_stop, stream));
    HIP_CHECK(hipEventSynchronize(untimed_stop));

    float elapsed_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&elapsed_ms, timed_start, timed_stop));
    INFO("round " << round << " elapsed " << elapsed_ms << " ms");
    REQUIRE(elapsed_ms * 1000.0f >= 2 * kTimedSpinUs * 0.9f);
    HIP_CHECK_ERROR(hipEventElapsedTime(&elapsed_ms, untimed_middle, untimed_stop),
                    hipErrorInvalidHandle);
    // The failed call is also the last error, which the next round's launch check would report
    HIP_CHECK_ERROR(hipGetLastError(), hipErrorInvalidHandle);
  }

  HIP_CHECK(hipEventDestroy(untimed_stop));
  HIP_CHECK(hipEventDestroy(untimed_middle));
  HIP_CHECK(hipEventDestroy(timed_stop));
  HIP_CHECK(hipEventDestroy(timed_start));
  HIP_CHECK(hipFree(counter));
  HIP_CHECK(hipStreamDestroy(stream));
}

/**
 * End doxygen group hipStreamCompletionSignals.
 * @}
 */
