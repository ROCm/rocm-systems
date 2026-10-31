// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_missing_wait_test.cpp
/// @brief A kernel that reads a load's result before waiting for it.
///
/// tools/test_hazard_report_crosscheck.py runs it under the core memory-wait checker
/// and the race plugin and compares the hazard records of both.

#include <hip/hip_runtime.h>
#include <vector>

#include <gtest/gtest.h>

// ROCR keeps process-lifetime runtime state allocated after hipDeviceReset().
extern "C" const char *__lsan_default_suppressions() { return "leak:libhsa-runtime64.so\n"; }

// ROCR's own locking is invisible to ThreadSanitizer; see hip_vector_add_test.cpp.
extern "C" const char *__tsan_default_suppressions() {
  return "called_from_lib:libhsa-runtime64.so\n";
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int rc = RUN_ALL_TESTS();
  (void)hipDeviceReset();
  return rc;
}

#define HIP_ASSERT(call)                                                                           \
  do {                                                                                             \
    hipError_t err = (call);                                                                       \
    ASSERT_EQ(err, hipSuccess) << "HIP error: " << hipGetErrorString(err);                         \
  } while (0)

__global__ void missing_wait_kernel(const unsigned *in, unsigned *out) {
  const unsigned id = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned value;
  // The compiler does not track a load inside inline asm, so it inserts no wait.
  asm volatile("global_load_dword %0, %1, off" : "=v"(value) : "v"(in + id) : "memory");
  asm volatile("v_add_u32 %0, %0, 1" : "+v"(value));
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  out[id] = value * 3u;
}

TEST(HipMissingWaitTest, ReadsALoadBeforeItsWait) {
  constexpr unsigned kBlocks = 4;
  constexpr unsigned kThreads = 64;
  std::vector<unsigned> input(kBlocks * kThreads);
  for (unsigned index = 0; index < input.size(); ++index)
    input[index] = index;
  const size_t bytes = input.size() * sizeof(unsigned);
  unsigned *in = nullptr;
  unsigned *out = nullptr;
  HIP_ASSERT(hipMalloc(&in, bytes));
  HIP_ASSERT(hipMalloc(&out, bytes));
  HIP_ASSERT(hipMemcpy(in, input.data(), bytes, hipMemcpyHostToDevice));
  missing_wait_kernel<<<kBlocks, kThreads>>>(in, out);
  HIP_ASSERT(hipDeviceSynchronize());
  std::vector<unsigned> output(input.size());
  HIP_ASSERT(hipMemcpy(output.data(), out, bytes, hipMemcpyDeviceToHost));
  // The simulator completes the load before the early read, so the result is exact.
  for (unsigned index = 0; index < output.size(); ++index)
    EXPECT_EQ(output[index], (index + 1) * 3u) << index;
  HIP_ASSERT(hipFree(in));
  HIP_ASSERT(hipFree(out));
}
