// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <gtest/gtest.h>

namespace {

__global__ void read_shader_rtc(uint64_t *result) { result[0] = wall_clock64(); }

TEST(HipShaderRtcTest, AqlKernelReadsTheCalibratedGpuClock) {
  ASSERT_EQ(hipSetDevice(0), hipSuccess);
  hsa_agent_t gpu{};
  const auto select_gpu = [](hsa_agent_t agent, void *result) {
    hsa_device_type_t type{};
    auto status = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if (status == HSA_STATUS_SUCCESS && type == HSA_DEVICE_TYPE_GPU) {
      *static_cast<hsa_agent_t *>(result) = agent;
      return HSA_STATUS_INFO_BREAK;
    }
    return status;
  };
  ASSERT_EQ(hsa_iterate_agents(select_gpu, &gpu), HSA_STATUS_INFO_BREAK);
  uint64_t *output = nullptr;
  ASSERT_EQ(hipMalloc(&output, sizeof(*output)), hipSuccess);
  hsa_amd_clock_counters_t before{}, after{};
  const auto info = static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_CLOCK_COUNTERS);
  ASSERT_EQ(hsa_agent_get_info(gpu, info, &before), HSA_STATUS_SUCCESS);
  read_shader_rtc<<<1, 1>>>(output);
  ASSERT_EQ(hipGetLastError(), hipSuccess);
  uint64_t shader = 0;
  ASSERT_EQ(hipMemcpy(&shader, output, sizeof(shader), hipMemcpyDeviceToHost), hipSuccess);
  ASSERT_EQ(hsa_agent_get_info(gpu, info, &after), HSA_STATUS_SUCCESS);
  EXPECT_LE(before.gpu_clock_counter, shader);
  EXPECT_LE(shader, after.gpu_clock_counter);
  EXPECT_EQ(hipFree(output), hipSuccess);
}

} // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
