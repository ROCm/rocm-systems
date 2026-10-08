// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "race_test_support.hpp"

#include <vector>

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int rc = RUN_ALL_TESTS();
  (void)hipDeviceReset();
  return rc;
}

namespace {

// Cases ending in _race must report a race; all other cases must remain clean.
using VmemRaceTest = rocjitsu::test::RaceTestBase;

__global__ void compiler_atomic_kernel(unsigned *values, unsigned *output) {
  const unsigned lane = threadIdx.x;
  output[lane] = atomicAdd(values + lane, 1u) * 3u;
}

// Keep the producer, explicit wait, and result use together so a compiler-added
// wait cannot hide the missing-wait controls or the VM-only regression.
template <bool VmWait, bool ExpWait>
__global__ void explicit_atomic_kernel(unsigned *values, unsigned *output) {
  const unsigned lane = threadIdx.x;
  unsigned old;
#if defined(__gfx1151__)
  asm volatile("global_atomic_add_u32 %[old], %[address], %[increment], off glc\n"
               "s_waitcnt vmcnt(%c[vm]) expcnt(%c[exp]) lgkmcnt(63)\n"
#else
  asm volatile("global_atomic_add %[old], %[address], %[increment], off sc0\n"
               "s_waitcnt vmcnt(%c[vm]) expcnt(%c[exp]) lgkmcnt(15)\n"
#endif
               "v_mov_b32 %[old], %[old]\n"
               : [old] "=&v"(old)
               : [address] "v"(values + lane), [increment] "v"(1u), [vm] "n"(VmWait ? 0 : 63),
                 [exp] "n"(ExpWait ? 0 : 7)
               : "memory");
  output[lane] = old * 3u;
}

template <typename Launch> void check_atomic_values(Launch launch) {
  // One full wave32 or half a wave64. Every case has one independently checked
  // dispatch and one expected report for an unsafe result use.
  constexpr unsigned lanes = 32;
  std::vector<unsigned> input(lanes), result(lanes);
  for (unsigned lane = 0; lane < lanes; ++lane)
    input[lane] = 17 + lane;
  unsigned *values = nullptr, *output = nullptr;
  ASSERT_EQ(hipMalloc(&values, input.size() * sizeof(unsigned)), hipSuccess);
  ASSERT_EQ(hipMalloc(&output, result.size() * sizeof(unsigned)), hipSuccess);
  ASSERT_EQ(hipMemcpy(values, input.data(), input.size() * sizeof(unsigned), hipMemcpyHostToDevice),
            hipSuccess);
  launch(values, output, lanes);
  ASSERT_EQ(hipGetLastError(), hipSuccess);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
  ASSERT_EQ(hipMemcpy(input.data(), values, input.size() * sizeof(unsigned), hipMemcpyDeviceToHost),
            hipSuccess);
  ASSERT_EQ(
      hipMemcpy(result.data(), output, result.size() * sizeof(unsigned), hipMemcpyDeviceToHost),
      hipSuccess);
  for (unsigned lane = 0; lane < lanes; ++lane) {
    EXPECT_EQ(input[lane], 18 + lane);
    EXPECT_EQ(result[lane], (17 + lane) * 3u);
  }
  EXPECT_EQ(hipFree(output), hipSuccess);
  EXPECT_EQ(hipFree(values), hipSuccess);
}

TEST_F(VmemRaceTest, compiler_generated) {
  ASSERT_NO_FATAL_FAILURE(
      check_atomic_values([](unsigned *values, unsigned *output, unsigned lanes) {
        compiler_atomic_kernel<<<1, lanes>>>(values, output);
      }));
  ExpectNoRace();
}

TEST_F(VmemRaceTest, vmcnt_only) {
  ASSERT_NO_FATAL_FAILURE(
      check_atomic_values([](unsigned *values, unsigned *output, unsigned lanes) {
        explicit_atomic_kernel<true, false><<<1, lanes>>>(values, output);
      }));
  ExpectNoRace();
}

static constexpr rocjitsu::test::RaceExpectation kMissingVmWait{
    .kernel = "explicit_atomic_kernel",
    .type = "VGPR",
    .access = "read",
    .wave = 0,
    .lane = 0,
    .producer = "global_atomic_add",
    .consumer = "v_mov_b32",
};

TEST_F(VmemRaceTest, no_wait_race) {
  ASSERT_NO_FATAL_FAILURE(
      check_atomic_values([](unsigned *values, unsigned *output, unsigned lanes) {
        explicit_atomic_kernel<false, false><<<1, lanes>>>(values, output);
      }));
  ExpectRace(kMissingVmWait);
}

TEST_F(VmemRaceTest, expcnt_only_race) {
  ASSERT_NO_FATAL_FAILURE(
      check_atomic_values([](unsigned *values, unsigned *output, unsigned lanes) {
        explicit_atomic_kernel<false, true><<<1, lanes>>>(values, output);
      }));
  ExpectRace(kMissingVmWait);
}

// Exchange data across waves through generic FLAT accesses. Drain the memory
// counters without waiting on EXP; the synchronized case must retire its LDS
// writes at the barrier, while the missing-barrier case must still report.
template <bool Synchronized> __global__ void flat_lds_exchange_kernel(unsigned *output) {
  __shared__ unsigned tile[128];
  const unsigned tid = threadIdx.x;
  const uint64_t write_address = reinterpret_cast<uint64_t>(&tile[tid]);
#if defined(__gfx1151__)
  asm volatile("flat_store_b32 %[address], %[value]\n"
               "s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(7)\n"
               "s_waitcnt_vscnt null, 0\n"
#else
  asm volatile("flat_store_dword %[address], %[value]\n"
               "s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(7)\n"
#endif
               :
               : [address] "v"(write_address), [value] "v"(tid + 11)
               : "memory");
  if constexpr (Synchronized)
    __syncthreads();
  const uint64_t read_address = reinterpret_cast<uint64_t>(&tile[127 - tid]);
  unsigned value;
#if defined(__gfx1151__)
  asm volatile("flat_load_b32 %[value], %[address]\n"
#else
  asm volatile("flat_load_dword %[value], %[address]\n"
#endif
               "s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(7)\n"
               : [value] "=v"(value)
               : [address] "v"(read_address)
               : "memory");
  output[tid] = value;
}

template <bool Synchronized> void check_flat_lds_exchange() {
  unsigned *output = nullptr;
  std::vector<unsigned> result(128);
  ASSERT_EQ(hipMalloc(&output, result.size() * sizeof(unsigned)), hipSuccess);
  flat_lds_exchange_kernel<Synchronized><<<1, 128>>>(output);
  ASSERT_EQ(hipGetLastError(), hipSuccess);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
  ASSERT_EQ(
      hipMemcpy(result.data(), output, result.size() * sizeof(unsigned), hipMemcpyDeviceToHost),
      hipSuccess);
  if constexpr (Synchronized) {
    for (unsigned tid = 0; tid < result.size(); ++tid)
      EXPECT_EQ(result[tid], 127 - tid + 11);
  }
  EXPECT_EQ(hipFree(output), hipSuccess);
}

TEST_F(VmemRaceTest, flat_lds_exchange) {
  ASSERT_NO_FATAL_FAILURE(check_flat_lds_exchange<true>());
  ExpectNoRace();
}

TEST_F(VmemRaceTest, flat_lds_exchange_race) {
  ASSERT_NO_FATAL_FAILURE(check_flat_lds_exchange<false>());
  ExpectRace({.kernel = "flat_lds_exchange_kernel",
              .type = "LDS",
              .access = "read",
              .consumer = "flat_load"});
}

} // namespace
