// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "race_log_expectation.hpp"

#include <cstdio>
#include <vector>

namespace rocjitsu::test {

/// Common issue-capacity probes. Target suites instantiate these with their
/// architectural counter capacities and keep target-specific expectations.
template <int YoungerLoads> __global__ void vmcnt_capacity_kernel(const float *src, float *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float producer, younger, result;
  asm volatile("global_load_dword %[producer], %[address], off\n"
               ".rept %c[younger_loads]\n"
               "global_load_dword %[younger], %[address], off\n"
               ".endr\n"
               "v_mov_b32 %[result], %[producer]\n"
               : [producer] "=&v"(producer), [younger] "=&v"(younger), [result] "=&v"(result)
               : [address] "v"(&src[tid]), [younger_loads] "n"(YoungerLoads)
               : "memory");
  dst[tid] = result;
}

template <int YoungerReads> __global__ void lgkmcnt_capacity_kernel(int *dst) {
  __shared__ int lds[64];
  int tid = threadIdx.x;
  lds[tid] = tid + 1;
  __syncthreads();

  int producer, younger, result;
  asm volatile("ds_read_b32 %[producer], %[address]\n"
               ".rept %c[younger_reads]\n"
               "ds_read_b32 %[younger], %[address]\n"
               ".endr\n"
               "v_mov_b32 %[result], %[producer]\n"
               : [producer] "=&v"(producer), [younger] "=&v"(younger), [result] "=&v"(result)
               : [address] "v"(tid * 4), [younger_reads] "n"(YoungerReads)
               : "memory");
  dst[tid] = result;
}

class RaceTestBase : public ::testing::Test {
protected:
  template <typename T> T *alloc(int count) {
    T *pointer = nullptr;
    (void)hipMalloc(&pointer, count * sizeof(T));
    return pointer;
  }

  template <typename T> T *allocWithData(int count) {
    T *pointer = alloc<T>(count);
    std::vector<T> host(count);
    for (int index = 0; index < count; ++index)
      host[index] = static_cast<T>(index);
    (void)hipMemcpy(pointer, host.data(), count * sizeof(T), hipMemcpyHostToDevice);
    return pointer;
  }

  void sync() { (void)hipDeviceSynchronize(); }

  void ExpectNoRace() {
    const RaceLogParseResult parsed = parseRaceLogFromEnvironment();
    ASSERT_TRUE(parsed.ok()) << parsed.error;
    if (!parsed.records.empty()) {
      for (const auto &record : parsed.records) {
        std::fprintf(stderr, "  [%s symbol=%s dispatch=%d %s reg=%d wg=%s] %s\n",
                     record.kernel.c_str(), record.symbol.c_str(), record.dispatch,
                     record.type.c_str(), record.reg, record.workgroup.c_str(),
                     record.message.c_str());
      }
    }
    EXPECT_TRUE(parsed.records.empty()) << "Expected no races, got " << parsed.records.size();
  }

  void ExpectRace(const RaceExpectation &expected) {
    const RaceLogParseResult parsed = parseRaceLogFromEnvironment();
    ASSERT_TRUE(parsed.ok()) << parsed.error;
    const RaceExpectationMatchResult matched = matchRaceExpectation(parsed.records, expected);
    EXPECT_TRUE(matched.ok()) << matched.message();
  }
};

} // namespace rocjitsu::test
