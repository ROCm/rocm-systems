/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <atomic>
#include <iostream>
#include <thread>
#include <vector>
constexpr int NN = 1 << 21;
__global__ void kernel_do_nothing(__attribute__((unused)) int a) {
  // empty kernel
}
__global__ void kernel(float* x, float* y, int n) {
  size_t tid{threadIdx.x};
  if (tid < 1) {
    for (int i = 0; i < n; i++) {
      x[i] = sqrt(powf(3.14159, i));
    }
    y[tid] = y[tid] + 1.0f;
  }
}
__global__ void nKernel(float* y) {
  size_t tid{threadIdx.x};
  y[tid] = y[tid] + 1.0f;
}

// wall_clock64() is an AMD device builtin, so this test is AMD-only.
#if HT_AMD

__global__ void verifyStreamPacketOrder(uint32_t* sequence_value, uint32_t expected_value,
                                        uint32_t* ordering_errors, uint64_t delay_cycles) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  if (atomicAdd(sequence_value, 0u) != expected_value) {
    atomicAdd(ordering_errors, 1u);
  }

  const uint64_t start_cycle = wall_clock64();
  while (wall_clock64() - start_cycle < delay_cycles) {
  }
  atomicExch(sequence_value, expected_value + 1);
}

HIP_TEST_CASE(Unit_hipMultiStream_SharedHwQueuePacketOrdering) {
  constexpr uint32_t num_streams = 8;
  constexpr uint32_t rounds_per_phase = 8;
  const uint64_t delay_cycles = isQuickLevel() ? 100000 : 1000000;

  std::vector<hipStream_t> streams(num_streams);
  uint32_t* sequence_values = nullptr;
  uint32_t* ordering_errors = nullptr;
  HIP_CHECK(hipMalloc(&sequence_values, num_streams * sizeof(*sequence_values)));
  HIP_CHECK(hipMalloc(&ordering_errors, sizeof(*ordering_errors)));
  HIP_CHECK(hipMemset(sequence_values, 0, num_streams * sizeof(*sequence_values)));
  HIP_CHECK(hipMemset(ordering_errors, 0, sizeof(*ordering_errors)));

  for (uint32_t stream_idx = 0; stream_idx < num_streams; ++stream_idx) {
    HIP_CHECK(hipStreamCreate(&streams[stream_idx]));
  }

  auto submit_phase = [&](uint32_t first_expected_value) {
    std::atomic<uint32_t> num_ready_threads{0};
    std::atomic<uint32_t> num_round_arrivals{0};
    std::atomic<uint32_t> completed_rounds{0};
    std::atomic<uint32_t> submission_errors{0};
    std::atomic<bool> start_submissions{false};
    std::vector<std::thread> submission_threads;
    submission_threads.reserve(num_streams);

    for (uint32_t stream_idx = 0; stream_idx < num_streams; ++stream_idx) {
      submission_threads.emplace_back([&, stream_idx] {
        num_ready_threads.fetch_add(1, std::memory_order_release);
        while (!start_submissions.load(std::memory_order_acquire)) {
          std::this_thread::yield();
        }

        for (uint32_t round_idx = 0; round_idx < rounds_per_phase; ++round_idx) {
          hipLaunchKernelGGL(verifyStreamPacketOrder, dim3(1), dim3(1), 0, streams[stream_idx],
                             sequence_values + stream_idx, first_expected_value + round_idx,
                             ordering_errors, delay_cycles);
          if (hipGetLastError() != hipSuccess) {
            submission_errors.fetch_add(1, std::memory_order_relaxed);
          }

          if (num_round_arrivals.fetch_add(1, std::memory_order_acq_rel) + 1 == num_streams) {
            num_round_arrivals.store(0, std::memory_order_relaxed);
            completed_rounds.fetch_add(1, std::memory_order_release);
          } else {
            while (completed_rounds.load(std::memory_order_acquire) == round_idx) {
              std::this_thread::yield();
            }
          }
        }
      });
    }

    while (num_ready_threads.load(std::memory_order_acquire) != num_streams) {
      std::this_thread::yield();
    }
    start_submissions.store(true, std::memory_order_release);

    for (auto& submission_thread : submission_threads) {
      submission_thread.join();
    }
    REQUIRE(submission_errors.load(std::memory_order_relaxed) == 0);
  };

  submit_phase(0);
  for (uint32_t stream_idx = 0; stream_idx < num_streams; ++stream_idx) {
    HIP_CHECK(hipStreamSynchronize(streams[stream_idx]));
    HIP_CHECK(hipStreamQuery(streams[stream_idx]));
  }

  // Submit again after synchronization/query gives the runtime an opportunity to release and
  // dynamically reacquire each stream's HW queue.
  submit_phase(rounds_per_phase);
  for (uint32_t stream_idx = 0; stream_idx < num_streams; ++stream_idx) {
    HIP_CHECK(hipStreamSynchronize(streams[stream_idx]));
  }

  uint32_t host_ordering_errors = 0;
  HIP_CHECK(hipMemcpy(&host_ordering_errors, ordering_errors, sizeof(host_ordering_errors),
                      hipMemcpyDeviceToHost));
  REQUIRE(host_ordering_errors == 0);

  for (uint32_t stream_idx = 0; stream_idx < num_streams; ++stream_idx) {
    HIP_CHECK(hipStreamDestroy(streams[stream_idx]));
  }
  HIP_CHECK(hipFree(ordering_errors));
  HIP_CHECK(hipFree(sequence_values));
}

#endif  // HT_AMD

HIP_TEST_CASE(Unit_hipMultiStream_sameDevice) {
  constexpr int num_streams{8};
  hipStream_t streams[num_streams];
  float *data[num_streams], *yd, *xd;
  float y{1.0f}, x{1.0f};
  const int n = isQuickLevel() ? (1 << 12) : NN;
  HIP_CHECK(hipMalloc((void**)&yd, sizeof(float)));
  HIP_CHECK(hipMalloc((void**)&xd, sizeof(float)));
  HIP_CHECK(hipMemcpy(yd, &y, sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(xd, &x, sizeof(float), hipMemcpyHostToDevice));
  for (int i = 0; i < num_streams; i++) {
    HIP_CHECK(hipStreamCreate(&streams[i]));
    HIP_CHECK(hipMalloc(&data[i], n * sizeof(float)));
    hipLaunchKernelGGL(kernel, dim3(1), dim3(1), 0, streams[i], data[i], xd, n);
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(HIP_KERNEL_NAME(nKernel), dim3(1), dim3(1), 0, 0, yd);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipFree(data[i]));
    HIP_CHECK(hipStreamDestroy(streams[i]));
  }
  HIP_CHECK(hipMemcpy(&x, xd, sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&y, yd, sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(xd));
  HIP_CHECK(hipFree(yd));
  REQUIRE(x == Catch::Approx(y));
}

HIP_TEST_CASE(Unit_hipMultiStream_multimeDevice) {
  const int nLoops = isQuickLevel() ? 500 : 50000;
  constexpr int nStreams = 2;
  std::vector<hipStream_t> streams(nStreams);
  int nGpu = 0;
  HIP_CHECK(hipGetDeviceCount(&nGpu));
  if (nGpu < 1) {
    HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice);
  }
  static int device = 0;
  HIP_CHECK(hipSetDevice(device));
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, device));
  INFO("Running on Bus: " << props.pciBusID << " " << props.name);
  for (int i = 0; i < nStreams; i++) {
    HIP_CHECK(hipStreamCreate(&streams[i]));
  }
  for (int k = 0; k <= nLoops; ++k) {
    HIP_CHECK(hipDeviceSynchronize());
    // Launch kernel with default stream
    hipLaunchKernelGGL(kernel_do_nothing, dim3(1), dim3(1), 0, 0, 1);
    HIP_CHECK(hipGetLastError());
    // Launch kernel on all streams
    for (int i = 0; i < nStreams; i++) {
      hipLaunchKernelGGL(kernel_do_nothing, dim3(1), dim3(1), 0, streams[i], 1);
      HIP_CHECK(hipGetLastError());
    }
    // Sync stream 1
    HIP_CHECK(hipStreamSynchronize(streams[0]));
    if (k % 10000 == 0 || k == nLoops) {
      INFO("Iter: " << k);
    }
  }
  HIP_CHECK(hipDeviceSynchronize());
  // Clean up
  for (int i = 0; i < nStreams; i++) {
    HIP_CHECK(hipStreamDestroy(streams[i]));
  }
}
