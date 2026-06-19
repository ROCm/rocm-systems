// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// End-to-end test for the kernel logging plugin (RJ_LOG=1).
// Launches a simple kernel and verifies the log output contains
// expected dispatch information.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#define HIP_ASSERT(call)                                                                           \
  do {                                                                                             \
    hipError_t err = (call);                                                                       \
    ASSERT_EQ(err, hipSuccess) << "HIP error: " << hipGetErrorString(err);                         \
  } while (0)

__global__ void simple_add(const float *a, const float *b, float *c) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  c[tid] = a[tid] + b[tid];
}

static std::filesystem::path log_path() {
  const char *dir = std::getenv("RJ_SINK_DIR");
  return dir ? std::filesystem::path(dir) / "logging.log" : std::filesystem::path{};
}

TEST(LoggingTest, dispatch_logged) {
  auto path = log_path();
  ASSERT_FALSE(path.empty()) << "RJ_SINK_DIR not set";
  std::filesystem::remove(path);

  constexpr int N = 64;
  std::vector<float> h_a(N), h_b(N);
  for (int i = 0; i < N; ++i) {
    h_a[i] = static_cast<float>(i);
    h_b[i] = static_cast<float>(i * 2);
  }

  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  HIP_ASSERT(hipMalloc(&d_a, N * sizeof(float)));
  HIP_ASSERT(hipMalloc(&d_b, N * sizeof(float)));
  HIP_ASSERT(hipMalloc(&d_c, N * sizeof(float)));
  HIP_ASSERT(hipMemcpy(d_a, h_a.data(), N * sizeof(float), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(d_b, h_b.data(), N * sizeof(float), hipMemcpyHostToDevice));

  simple_add<<<1, N>>>(d_a, d_b, d_c);
  HIP_ASSERT(hipDeviceSynchronize());

  ASSERT_TRUE(std::filesystem::exists(path)) << "Log file not created at " << path;
  std::ifstream f(path);
  std::string log{std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};

  EXPECT_NE(log.find("Kernel #1 dispatch"), std::string::npos)
      << "Expected kernel dispatch log entry.\nLog:\n"
      << log;
  EXPECT_NE(log.find("vgprs="), std::string::npos) << "Expected VGPR count in log.\nLog:\n" << log;

  (void)hipFree(d_a);
  (void)hipFree(d_b);
  (void)hipFree(d_c);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
