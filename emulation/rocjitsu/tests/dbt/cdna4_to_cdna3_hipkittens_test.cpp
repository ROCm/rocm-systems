// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file cdna4_to_cdna3_hipkittens_test.cpp
/// @brief HIP module launches for HipKittens BF16 matmul gfx950 fixtures.

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

constexpr uint32_t kBlock = 256;
constexpr uint32_t kThreads = 512;
constexpr uint32_t kDynamicSharedBytes = 160000;
constexpr uint32_t kFull = 8192;
constexpr uint32_t kFullGridX = (kFull / kBlock) * (kFull / kBlock);

#define HIP_ASSERT(call)                                                                           \
  do {                                                                                             \
    hipError_t err = (call);                                                                       \
    ASSERT_EQ(err, hipSuccess) << "HIP error: " << hipGetErrorString(err);                         \
  } while (0)

struct RuntimeDim {
  size_t v;
};

struct DescriptorDict {};

struct Gl {
  uint16_t *raw_ptr;
  RuntimeDim b;
  RuntimeDim d;
  RuntimeDim r;
  RuntimeDim c;
  DescriptorDict tma_descs;
};

struct MicroGlobals32x16 {
  Gl a;
  Gl b;
  Gl c;
  hipStream_t stream;
};

struct MicroGlobals16x32 {
  Gl a;
  Gl b;
  Gl c;
  hipStream_t stream;
  int M;
  int N;
  int K;
};

static_assert(sizeof(Gl) == 48);
static_assert(sizeof(MicroGlobals32x16) == 152);
static_assert(sizeof(MicroGlobals16x32) == 168);

uint16_t f32_to_bf16_trunc(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<uint16_t>(bits >> 16);
}

float bf16_to_f32(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

Gl make_gl(uint16_t *ptr, size_t rows, size_t cols) {
  return Gl{ptr, RuntimeDim{1}, RuntimeDim{1}, RuntimeDim{rows}, RuntimeDim{cols}, {}};
}

std::string kernel_dir() {
  if (const char *env = std::getenv("ROCJITSU_HIPKITTENS_KERNEL_DIR"))
    return env;
  return (std::filesystem::current_path() / "kernels").string();
}

void load_function(const char *fixture, const char *symbol, hipModule_t *module,
                   hipFunction_t *fn) {
  const std::filesystem::path path =
      std::filesystem::path(kernel_dir()) / (std::string(fixture) + ".hsaco");
  HIP_ASSERT(hipModuleLoad(module, path.c_str()));
  HIP_ASSERT(hipModuleGetFunction(fn, *module, symbol));
  HIP_ASSERT(
      hipFuncSetAttribute(*fn, hipFuncAttributeMaxDynamicSharedMemorySize, kDynamicSharedBytes));
}

void expect_near_bf16_matmul(const std::vector<uint16_t> &actual, const std::vector<uint16_t> &a,
                             const std::vector<uint16_t> &b, uint32_t size, uint32_t ld,
                             float tolerance) {
  uint32_t mismatches = 0;
  for (uint32_t r = 0; r < size; ++r) {
    for (uint32_t c = 0; c < size; ++c) {
      const size_t actual_idx = static_cast<size_t>(r) * ld + c;
      const float got = bf16_to_f32(actual[actual_idx]);
      const float want = static_cast<float>(size) * bf16_to_f32(a[static_cast<size_t>(r) * ld]) *
                         bf16_to_f32(b[static_cast<size_t>(c) * ld]);
      const float diff = std::fabs(got - want);
      if (diff > tolerance) {
        if (mismatches < 12) {
          ADD_FAILURE() << "mismatch at (" << r << "," << c << "): got=" << got
                        << " expected=" << want << " got_bits=0x" << std::hex << actual[actual_idx]
                        << std::dec;
        }
        ++mismatches;
      }
    }
  }
  EXPECT_EQ(mismatches, 0u) << mismatches << " BF16 matmul mismatches";
}

TEST(HipKittensBf16MatmulDbtTest, With16x32DispatchesAndMatchesReference) {
  std::vector<uint16_t> host_a(static_cast<size_t>(kFull) * kFull);
  std::vector<uint16_t> host_b(static_cast<size_t>(kFull) * kFull);

  for (uint32_t r = 0; r < kFull; ++r) {
    const float a_value = static_cast<float>((r % 4) + 1) * 0.0625f;
    std::fill_n(host_a.data() + static_cast<size_t>(r) * kFull, kFull, f32_to_bf16_trunc(a_value));
  }
  for (uint32_t col = 0; col < kFull; ++col) {
    const float b_value = static_cast<float>((col % 5) + 1) * 0.03125f;
    std::fill_n(host_b.data() + static_cast<size_t>(col) * kFull, kFull,
                f32_to_bf16_trunc(b_value));
  }

  uint16_t *dev_a = nullptr;
  uint16_t *dev_b = nullptr;
  uint16_t *dev_c = nullptr;
  HIP_ASSERT(hipMalloc(&dev_a, host_a.size() * sizeof(uint16_t)));
  HIP_ASSERT(hipMalloc(&dev_b, host_b.size() * sizeof(uint16_t)));
  HIP_ASSERT(hipMalloc(&dev_c, host_a.size() * sizeof(uint16_t)));
  HIP_ASSERT(
      hipMemcpy(dev_a, host_a.data(), host_a.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
  HIP_ASSERT(
      hipMemcpy(dev_b, host_b.data(), host_b.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemset(dev_c, 0x5a, host_a.size() * sizeof(uint16_t)));

  hipModule_t module = nullptr;
  hipFunction_t fn = nullptr;
  ASSERT_NO_FATAL_FAILURE(load_function("hipkittens_bf16fp32_256_256_64_32_with16x32",
                                        "_Z8micro_tk13micro_globalsiii", &module, &fn));
  MicroGlobals16x32 g{make_gl(dev_a, kFull, kFull), make_gl(dev_b, kFull, kFull),
                      make_gl(dev_c, kFull, kFull), nullptr,
                      static_cast<int>(kFull),      static_cast<int>(kFull),
                      static_cast<int>(kFull)};
  int arg_m = g.M;
  int arg_n = g.N;
  int arg_k = g.K;
  void *args[] = {&g, &arg_m, &arg_n, &arg_k};
  HIP_ASSERT(hipModuleLaunchKernel(fn, kFullGridX, 1, 1, kThreads, 1, 1, kDynamicSharedBytes,
                                   nullptr, args, nullptr));
  HIP_ASSERT(hipDeviceSynchronize());

  std::vector<uint16_t> actual_full(host_a.size());
  HIP_ASSERT(hipMemcpy(actual_full.data(), dev_c, actual_full.size() * sizeof(uint16_t),
                       hipMemcpyDeviceToHost));
  expect_near_bf16_matmul(actual_full, host_a, host_b, kFull, kFull, 4.0f);

  HIP_ASSERT(hipModuleUnload(module));
  HIP_ASSERT(hipFree(dev_a));
  HIP_ASSERT(hipFree(dev_b));
  HIP_ASSERT(hipFree(dev_c));
}

TEST(HipKittensBf16MatmulDbtTest, With32x16DispatchesAndMatchesReference) {
  std::vector<uint16_t> host_a(static_cast<size_t>(kFull) * kFull);
  std::vector<uint16_t> host_b(static_cast<size_t>(kFull) * kFull);

  for (uint32_t r = 0; r < kFull; ++r) {
    const float a_value = static_cast<float>((r % 4) + 1) * 0.0625f;
    std::fill_n(host_a.data() + static_cast<size_t>(r) * kFull, kFull, f32_to_bf16_trunc(a_value));
  }
  for (uint32_t col = 0; col < kFull; ++col) {
    const float b_value = static_cast<float>((col % 5) + 1) * 0.03125f;
    std::fill_n(host_b.data() + static_cast<size_t>(col) * kFull, kFull,
                f32_to_bf16_trunc(b_value));
  }

  uint16_t *dev_a = nullptr;
  uint16_t *dev_b = nullptr;
  uint16_t *dev_c = nullptr;
  HIP_ASSERT(hipMalloc(&dev_a, host_a.size() * sizeof(uint16_t)));
  HIP_ASSERT(hipMalloc(&dev_b, host_b.size() * sizeof(uint16_t)));
  HIP_ASSERT(hipMalloc(&dev_c, host_a.size() * sizeof(uint16_t)));
  HIP_ASSERT(
      hipMemcpy(dev_a, host_a.data(), host_a.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
  HIP_ASSERT(
      hipMemcpy(dev_b, host_b.data(), host_b.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemset(dev_c, 0x5a, host_a.size() * sizeof(uint16_t)));

  hipModule_t module = nullptr;
  hipFunction_t fn = nullptr;
  ASSERT_NO_FATAL_FAILURE(load_function("hipkittens_bf16fp32_256_256_64_32_with32x16",
                                        "_Z8micro_tk13micro_globals", &module, &fn));
  MicroGlobals32x16 g{make_gl(dev_a, kFull, kFull), make_gl(dev_b, kFull, kFull),
                      make_gl(dev_c, kFull, kFull), nullptr};
  void *args[] = {&g};
  HIP_ASSERT(hipModuleLaunchKernel(fn, kFullGridX, 1, 1, kThreads, 1, 1, kDynamicSharedBytes,
                                   nullptr, args, nullptr));
  HIP_ASSERT(hipDeviceSynchronize());

  std::vector<uint16_t> actual_full(host_a.size());
  HIP_ASSERT(hipMemcpy(actual_full.data(), dev_c, actual_full.size() * sizeof(uint16_t),
                       hipMemcpyDeviceToHost));
  expect_near_bf16_matmul(actual_full, host_a, host_b, kFull, kFull, 4.0f);

  HIP_ASSERT(hipModuleUnload(module));
  HIP_ASSERT(hipFree(dev_a));
  HIP_ASSERT(hipFree(dev_b));
  HIP_ASSERT(hipFree(dev_c));
}

} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  const int rc = RUN_ALL_TESTS();
  (void)hipDeviceReset();
  return rc;
}
