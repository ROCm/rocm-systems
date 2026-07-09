// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_race_gfx1151_test.hip
/// @brief Small gfx1151 HIP race detector smoke tests.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int rc = RUN_ALL_TESTS();
  (void)hipDeviceReset();
  return rc;
}

namespace {

struct RaceRecord {
  std::string type;
  std::string message;
};

std::vector<RaceRecord> parse_race_log() {
  const char *dir = std::getenv("RJ_SINK_DIR");
  if (!dir)
    return {};

  std::ifstream f(std::string(dir) + "/race.log");
  std::vector<RaceRecord> records;
  std::string line;
  while (std::getline(f, line)) {
    if (!line.starts_with("RACE "))
      continue;

    RaceRecord record;
    auto type_pos = line.find("type=");
    if (type_pos != std::string::npos) {
      auto value_begin = type_pos + 5;
      auto value_end = line.find(' ', value_begin);
      record.type = line.substr(value_begin, value_end - value_begin);
    }

    while (std::getline(f, line)) {
      if (line == "END_RACE")
        break;
      if (!record.message.empty())
        record.message += '\n';
      record.message += line;
    }
    records.push_back(std::move(record));
  }
  return records;
}

class Gfx1151RaceTest : public ::testing::Test {
protected:
  template <typename T>
  T *alloc(int n) {
    T *p = nullptr;
    (void)hipMalloc(&p, n * sizeof(T));
    return p;
  }

  template <typename T>
  T *allocWithData(int n) {
    T *p = alloc<T>(n);
    std::vector<T> h(n);
    for (int i = 0; i < n; ++i)
      h[i] = static_cast<T>(i);
    (void)hipMemcpy(p, h.data(), n * sizeof(T), hipMemcpyHostToDevice);
    return p;
  }

  void sync() { (void)hipDeviceSynchronize(); }

  void ExpectNoRace() {
    auto records = parse_race_log();
    EXPECT_TRUE(records.empty()) << "Expected no races, got " << records.size();
  }

  std::vector<RaceRecord> ExpectRace() {
    auto records = parse_race_log();
    EXPECT_FALSE(records.empty()) << "Expected at least one race, got none";
    return records;
  }
};

__global__ void vgpr_no_race_kernel(const float *src, float *dst) {
  int tid = threadIdx.x;
  float val;
  asm volatile("global_load_dword %0, %1, off\n"
               "s_waitcnt vmcnt(0)\n"
               : "=v"(val)
               : "v"(&src[tid]));
  dst[tid] = val;
}

__global__ void vgpr_race_kernel(const float *src, float *dst) {
  int tid = threadIdx.x;
  float val;
  asm volatile("global_load_dword %0, %1, off\n"
               "s_nop 0\n"
               : "=v"(val)
               : "v"(&src[tid]));
  dst[tid] = val;
}

__attribute__((amdgpu_flat_work_group_size(64, 64))) __global__ void
lds_wave32_cross_wave_safe_kernel(int *out) {
  __shared__ int lds[64];
  int tid = threadIdx.x;
  lds[tid] = tid + 100;
  __syncthreads();
  out[tid] = lds[(tid + 32) & 63];
}

__attribute__((amdgpu_flat_work_group_size(64, 64))) __global__ void
lds_wave32_cross_wave_race_kernel(int *out) {
  __shared__ int lds[64];
  int tid = threadIdx.x;
  lds[tid] = tid + 100;
  if (tid >= 32)
    out[tid] = lds[tid - 32];
}

TEST_F(Gfx1151RaceTest, vgpr_waitcnt) {
  auto *src = allocWithData<float>(32);
  auto *dst = alloc<float>(32);
  vgpr_no_race_kernel<<<1, 32>>>(src, dst);
  sync();
  ExpectNoRace();
}

TEST_F(Gfx1151RaceTest, vgpr_waitcnt_race) {
  auto *src = allocWithData<float>(32);
  auto *dst = alloc<float>(32);
  vgpr_race_kernel<<<1, 32>>>(src, dst);
  sync();
  auto records = ExpectRace();
  ASSERT_FALSE(records.empty());
  EXPECT_EQ(records[0].type, "VGPR");
}

TEST_F(Gfx1151RaceTest, lds_wave32_cross_wave) {
  lds_wave32_cross_wave_safe_kernel<<<1, 64>>>(alloc<int>(64));
  sync();
  ExpectNoRace();
}

TEST_F(Gfx1151RaceTest, lds_wave32_cross_wave_race) {
  lds_wave32_cross_wave_race_kernel<<<1, 64>>>(alloc<int>(64));
  sync();
  auto records = ExpectRace();
  ASSERT_FALSE(records.empty());
  EXPECT_EQ(records[0].type, "LDS");
}

} // namespace
