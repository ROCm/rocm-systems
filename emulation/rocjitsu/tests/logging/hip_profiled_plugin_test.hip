// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// End-to-end test for ProfiledExecutionPluginGroup.
// Launches a kernel with RJ_USE_PROFILED_EXECUTION_PLUGIN_GROUP=1 and
// verifies that profile output contains hook timing data.
//
// The profiled group writes HOOK_PROFILE lines to stderr by default.
// This test captures stderr by redirecting it to a temporary file.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

#define HIP_ASSERT(call)                                                                           \
  do {                                                                                             \
    hipError_t err = (call);                                                                       \
    ASSERT_EQ(err, hipSuccess) << "HIP error: " << hipGetErrorString(err);                         \
  } while (0)

__global__ void add_kernel(const float *a, const float *b, float *c) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  c[tid] = a[tid] + b[tid];
}

TEST(ProfiledPluginTest, hook_profile_output) {
  const char *sink_dir = std::getenv("RJ_SINK_DIR");
  ASSERT_NE(sink_dir, nullptr) << "RJ_SINK_DIR not set";
  auto stderr_capture = std::filesystem::path(sink_dir) / "captured_stderr.txt";

  // Redirect stderr to a file so we can inspect profiled output.
  fflush(stderr);
  int saved_stderr = dup(STDERR_FILENO);
  FILE *capture = fopen(stderr_capture.c_str(), "w");
  ASSERT_NE(capture, nullptr);
  dup2(fileno(capture), STDERR_FILENO);

  constexpr int N = 64;
  std::vector<float> h_a(N, 1.0f), h_b(N, 2.0f);

  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  HIP_ASSERT(hipMalloc(&d_a, N * sizeof(float)));
  HIP_ASSERT(hipMalloc(&d_b, N * sizeof(float)));
  HIP_ASSERT(hipMalloc(&d_c, N * sizeof(float)));
  HIP_ASSERT(hipMemcpy(d_a, h_a.data(), N * sizeof(float), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(d_b, h_b.data(), N * sizeof(float), hipMemcpyHostToDevice));

  add_kernel<<<1, N>>>(d_a, d_b, d_c);
  HIP_ASSERT(hipDeviceSynchronize());

  // Restore stderr.
  fflush(stderr);
  dup2(saved_stderr, STDERR_FILENO);
  close(saved_stderr);
  fclose(capture);

  std::ifstream f(stderr_capture);
  std::string log{std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};

  EXPECT_NE(log.find("HOOK_PROFILE"), std::string::npos)
      << "Expected HOOK_PROFILE entries in stderr.\nCaptured:\n"
      << log;
  EXPECT_NE(log.find("beforeExecuteInstruction"), std::string::npos)
      << "Expected beforeExecuteInstruction hook timing.\nCaptured:\n"
      << log;
  EXPECT_NE(log.find("afterExecuteInstruction"), std::string::npos)
      << "Expected afterExecuteInstruction hook timing.\nCaptured:\n"
      << log;

  (void)hipFree(d_a);
  (void)hipFree(d_b);
  (void)hipFree(d_c);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
