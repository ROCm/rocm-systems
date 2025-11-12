/*
 Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */

/**
 * @addtogroup hipPerfMultiStreamKernelLaunchAtomic hipPerfMultiStreamKernelLaunchAtomic
 * @{
 * @ingroup perfStreamTest
 * Performance test for multi-stream kernel launch with LDS atomicAdd
 */

#include <hip_test_common.hh>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>

constexpr int kArraySize = 1024;
constexpr int kAtomicIterations = 1000;  // Number of atomicAdd iterations

// Kernel that uses LDS (shared memory) and atomicAdd in a loop
__global__ void atomicAddKernel(int* input) {
  __shared__ int lds_sum;
  
  // Since we launch with only 1 block, blockIdx.x is always 0, so tid = threadIdx.x
  int tid = threadIdx.x;
  
  // Initialize shared memory sum
  if (tid == 0) {
    lds_sum = 0;
  }
  __syncthreads();
  
  // Each thread (except the last one) performs multiple atomicAdd operations
  // Sum only the first 1023 elements
  if (tid < 1023) {
    int value = input[tid];
    for (int i = 0; i < kAtomicIterations; i++) {
      atomicAdd(&lds_sum, value);
    }
  }
  __syncthreads();
  
  // Thread 0 writes the LDS sum back to the last element of the array
  if (tid == 0) {
    input[1023] = lds_sum;
  }
}

// Verify kernel correctness using multiple streams
bool VerifyKernelCorrectness(const int* h_input) {
  std::cout << "\nVerifying kernel correctness with multiple streams..." << std::endl;
  
  constexpr int kVerifyNumStreams = 4;
  
  // Create streams
  std::vector<hipStream_t> streams(kVerifyNumStreams);
  for (auto& s : streams) {
    HIP_CHECK(hipStreamCreate(&s));
  }
  
  // Allocate device memory for each stream
  std::vector<int*> d_verify(kVerifyNumStreams);
  for (auto& d : d_verify) {
    HIP_CHECK(hipMalloc(&d, kArraySize * sizeof(int)));
    HIP_CHECK(hipMemcpy(d, h_input, kArraySize * sizeof(int), hipMemcpyHostToDevice));
  }
  
  // Launch kernel on each stream
  for (size_t s = 0; s < streams.size(); s++) {
    atomicAddKernel<<<1, 1024, 0, streams[s]>>>(d_verify[s]);
  }
  
  // Synchronize all streams
  for (const auto& s : streams) {
    HIP_CHECK(hipStreamSynchronize(s));
  }
  
  // Verify results from all streams
  // Expected result: 1023 threads * kAtomicIterations iterations * value (1)
  int expected_sum = 1023 * kAtomicIterations * 1;
  
  bool success = true;
  
  for (size_t i = 0; i < d_verify.size(); i++) {
    int* h_verify = new int[kArraySize];
    HIP_CHECK(hipMemcpy(h_verify, d_verify[i], kArraySize * sizeof(int), hipMemcpyDeviceToHost));
    
    int actual_sum = h_verify[1023];
    
    if (actual_sum != expected_sum) {
      std::cout << "ERROR: Kernel verification failed on stream " << i << "!" << std::endl;
      std::cout << "Expected: " << expected_sum << ", Got: " << actual_sum << std::endl;
      success = false;
    } else {
      std::cout << "Stream " << i << " verified correctly (result: " << actual_sum << ")" << std::endl;
    }
    
    delete[] h_verify;
  }
  
  if (success) {
    std::cout << "Kernel verification passed on all " << kVerifyNumStreams 
              << " streams! (Expected: " << expected_sum << ")" << std::endl;
  }
  
  // Cleanup
  for (auto& d : d_verify) {
    HIP_CHECK(hipFree(d));
  }
  for (auto& s : streams) {
    HIP_CHECK(hipStreamDestroy(s));
  }
  
  return success;
}

// Perform warmup iterations
void DoWarmup(const std::vector<hipStream_t>& streams, 
              const std::vector<int*>& d_inputs,
              const int* h_input,
              uint64_t warmup_iterations,
              uint64_t dispatches_per_stream) {
  for (uint64_t i = 0; i < warmup_iterations; i++) {
    for (uint32_t j = 0; j < dispatches_per_stream; j++) {
      for (size_t s = 0; s < streams.size(); s++) {
        atomicAddKernel<<<1, 1024, 0, streams[s]>>>(d_inputs[s]);
      }
    }
    for (const auto& s : streams) {
      HIP_CHECK(hipStreamSynchronize(s));
    }
    // Reset data for next warmup iteration
    for (auto& d_input : d_inputs) {
      HIP_CHECK(hipMemcpy(d_input, h_input, kArraySize * sizeof(int), hipMemcpyHostToDevice));
    }
  }
  HIP_CHECK(hipDeviceSynchronize());
}

// Run performance measurement
double RunPerformanceMeasurement(const std::vector<hipStream_t>& streams,
                                  const std::vector<int*>& d_inputs,
                                  uint64_t dispatches_per_stream) {
  using microseconds = std::chrono::duration<double, std::chrono::microseconds::period>;
  
  auto start = std::chrono::steady_clock::now();
  
  for (uint32_t j = 0; j < dispatches_per_stream; j++) {
    for (size_t s = 0; s < streams.size(); s++) {
      atomicAddKernel<<<1, 1024, 0, streams[s]>>>(d_inputs[s]);
    }
  }
  for (const auto& s : streams) {
    HIP_CHECK(hipStreamSynchronize(s));
  }
  
  auto end = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<microseconds>(end - start);
  
  return duration.count();
}

/**
 * Test Description
 * ------------------------
 *  - Launch a kernel with 1024 threads in 1 block that uses LDS and atomicAdd
 *    to sum all array elements. Measures performance with proper warmups across
 *    multiple streams.
 * Test source
 * ------------------------
 *  - perftests/stream/hipPerfMultiStreamKernelLaunchAtomic.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 5.6
 */

TEST_CASE("Perf_hipPerfMultiStreamKernelLaunchAtomic") {
  constexpr uint64_t kWarmupIterations = 10;
  constexpr uint64_t kKernelDispatchesPerStream = 10;
  constexpr uint64_t kStreamsPerExperiment[] = {
    2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
  };
  
  int nGpu = 0;
  HIP_CHECK(hipGetDeviceCount(&nGpu));
  if (nGpu < 1) {
    HipTest::HIP_SKIP_TEST("Skipping because devices < 1");
  }

  int deviceId = 0;
  HIP_CHECK(hipSetDevice(deviceId));
  
  hipDeviceProp_t props;
  HIP_CHECK(hipGetDeviceProperties(&props, deviceId));
  std::cout << "info: running on bus 0x" << std::hex << props.pciBusID << std::dec 
            << " " << props.name << " with " << props.multiProcessorCount 
            << " CUs and device id: " << deviceId << std::endl;

  // Allocate and initialize host array
  int* h_input = new int[kArraySize];
  for (int i = 0; i < kArraySize; i++) {
    h_input[i] = 1;  // Initialize each element to 1
  }

  // Verify kernel correctness first before performance measurements
  if (!VerifyKernelCorrectness(h_input)) {
    delete[] h_input;
    REQUIRE(false);  // Fail the test
  }

  // Log config
  std::cout << "\nUsing atomicAddKernel with LDS" << std::endl;
  std::cout << "Warmup Iterations: " << kWarmupIterations << std::endl;
  std::cout << "Kernel dispatches per stream: " << kKernelDispatchesPerStream << std::endl;
  std::cout << std::setw(20) << "Num Streams" << " | " << std::setw(20) << "Walltime (us)" << std::endl;
  std::cout << std::string(20, '-') << " | " << std::string(20, '-') << std::endl;

  // Loop through different stream counts
  for (const auto& num_streams : kStreamsPerExperiment) {
    // Create streams
    std::vector<hipStream_t> streams(num_streams);
    for (auto& s : streams) {
      HIP_CHECK(hipStreamCreate(&s));
    }

    // Allocate device memory for each stream
    std::vector<int*> d_inputs(num_streams);
    for (auto& d_input : d_inputs) {
      HIP_CHECK(hipMalloc(&d_input, kArraySize * sizeof(int)));
      HIP_CHECK(hipMemcpy(d_input, h_input, kArraySize * sizeof(int), hipMemcpyHostToDevice));
    }

    // Warmup phase
    DoWarmup(streams, d_inputs, h_input, kWarmupIterations, kKernelDispatchesPerStream);

    // Measurement phase
    double walltime_us = RunPerformanceMeasurement(streams, d_inputs, kKernelDispatchesPerStream);

    // Print results for this stream count
    std::cout << std::setw(20) << num_streams << " | " 
              << std::setw(20) << std::fixed << std::setprecision(2) 
              << walltime_us << std::endl;

    // Cleanup for this experiment
    for (auto& d_input : d_inputs) {
      HIP_CHECK(hipFree(d_input));
    }
    for (auto& s : streams) {
      HIP_CHECK(hipStreamDestroy(s));
    }
  }

  // Cleanup
  delete[] h_input;
}

/**
 * End doxygen group perfStreamTest.
 * @}
 */

