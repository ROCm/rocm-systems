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

using VmemAtomicRaceTest = rocjitsu::test::RaceTestBase;

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

TEST_F(VmemAtomicRaceTest, compiler_generated) {
  ASSERT_NO_FATAL_FAILURE(
      check_atomic_values([](unsigned *values, unsigned *output, unsigned lanes) {
        compiler_atomic_kernel<<<1, lanes>>>(values, output);
      }));
  ExpectNoRace();
}

TEST_F(VmemAtomicRaceTest, vmcnt_only) {
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

TEST_F(VmemAtomicRaceTest, no_wait) {
  ASSERT_NO_FATAL_FAILURE(
      check_atomic_values([](unsigned *values, unsigned *output, unsigned lanes) {
        explicit_atomic_kernel<false, false><<<1, lanes>>>(values, output);
      }));
  ExpectRace(kMissingVmWait);
}

TEST_F(VmemAtomicRaceTest, expcnt_only) {
  ASSERT_NO_FATAL_FAILURE(
      check_atomic_values([](unsigned *values, unsigned *output, unsigned lanes) {
        explicit_atomic_kernel<false, true><<<1, lanes>>>(values, output);
      }));
  ExpectRace(kMissingVmWait);
}

} // namespace
