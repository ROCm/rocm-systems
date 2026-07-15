/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "amd_smi/impl/amd_smi_test_internal.h"
#include "amd_smi/impl/amd_smi_wsl.h"

namespace {

constexpr int kHipSuccess = 0;
constexpr int kHipError = 1;

int get_device_status = kHipSuccess;
int current_device = 0;
int set_device_status = kHipSuccess;
int mem_get_info_status = kHipSuccess;
size_t mock_free_bytes = 0;
size_t mock_total_bytes = 0;
uint32_t mem_get_info_calls = 0;
std::vector<int> set_device_calls;

int TestGetDevice(int* device) {
  if (get_device_status == kHipSuccess) {
    *device = current_device;
  }
  return get_device_status;
}

int TestSetDevice(int device) {
  set_device_calls.push_back(device);
  return set_device_status;
}

int TestMemGetInfo(size_t* free_bytes, size_t* total_bytes) {
  ++mem_get_info_calls;
  if (mem_get_info_status == kHipSuccess) {
    *free_bytes = mock_free_bytes;
    *total_bytes = mock_total_bytes;
  }
  return mem_get_info_status;
}

class WslScopedDeviceTest : public testing::Test {
 protected:
  void SetUp() override {
    get_device_status = kHipSuccess;
    current_device = 0;
    set_device_status = kHipSuccess;
    mem_get_info_status = kHipSuccess;
    mock_free_bytes = 0;
    mock_total_bytes = 0;
    mem_get_info_calls = 0;
    set_device_calls.clear();
  }
};

TEST_F(WslScopedDeviceTest, RestoresSavedDevice) {
  current_device = 3;

  EXPECT_TRUE(amd::smi::amdsmi_test_wsl_scoped_device(TestGetDevice, TestSetDevice, 7));
  EXPECT_EQ(set_device_calls, (std::vector<int>{7, 3}));
}

TEST_F(WslScopedDeviceTest, FallsBackToDeviceZeroWhenGetDeviceFails) {
  get_device_status = kHipError;

  EXPECT_TRUE(amd::smi::amdsmi_test_wsl_scoped_device(TestGetDevice, TestSetDevice, 7));
  EXPECT_EQ(set_device_calls, (std::vector<int>{7, 0}));
}

TEST_F(WslScopedDeviceTest, DoesNotRestoreWhenSetDeviceFails) {
  current_device = 3;
  set_device_status = kHipError;

  EXPECT_FALSE(amd::smi::amdsmi_test_wsl_scoped_device(TestGetDevice, TestSetDevice, 7));
  EXPECT_EQ(set_device_calls, (std::vector<int>{7}));
}

TEST_F(WslScopedDeviceTest, ReturnsExactMemoryBytesAndRestoresDevice) {
  current_device = 3;
  mock_total_bytes = 109679810150ULL;
  mock_free_bytes = 109600118374ULL;
  amd::smi::WslVramUsageBytes usage;

  EXPECT_EQ(amd::smi::amdsmi_test_wsl_get_vram_usage_bytes(TestGetDevice, TestSetDevice,
                                                           TestMemGetInfo, 7, &usage),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(usage.total, 109679810150ULL);
  EXPECT_EQ(usage.used, 79691776ULL);
  EXPECT_EQ(mem_get_info_calls, 1U);
  EXPECT_EQ(set_device_calls, (std::vector<int>{7, 3}));
}

TEST_F(WslScopedDeviceTest, RestoresDeviceWhenMemoryQueryFails) {
  current_device = 3;
  mem_get_info_status = kHipError;
  amd::smi::WslVramUsageBytes usage;

  EXPECT_EQ(amd::smi::amdsmi_test_wsl_get_vram_usage_bytes(TestGetDevice, TestSetDevice,
                                                           TestMemGetInfo, 7, &usage),
            AMDSMI_STATUS_API_FAILED);
  EXPECT_EQ(mem_get_info_calls, 1U);
  EXPECT_EQ(set_device_calls, (std::vector<int>{7, 3}));
}

TEST_F(WslScopedDeviceTest, ClampsUsedBytesWhenFreeExceedsTotal) {
  mock_total_bytes = 100;
  mock_free_bytes = 200;
  amd::smi::WslVramUsageBytes usage;

  EXPECT_EQ(amd::smi::amdsmi_test_wsl_get_vram_usage_bytes(TestGetDevice, TestSetDevice,
                                                           TestMemGetInfo, 7, &usage),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(usage.total, 100U);
  EXPECT_EQ(usage.used, 0U);
}

TEST_F(WslScopedDeviceTest, DoesNotQueryMemoryWhenSetDeviceFails) {
  current_device = 3;
  set_device_status = kHipError;
  amd::smi::WslVramUsageBytes usage;

  EXPECT_EQ(amd::smi::amdsmi_test_wsl_get_vram_usage_bytes(TestGetDevice, TestSetDevice,
                                                           TestMemGetInfo, 7, &usage),
            AMDSMI_STATUS_API_FAILED);
  EXPECT_EQ(mem_get_info_calls, 0U);
  EXPECT_EQ(set_device_calls, (std::vector<int>{7}));
}

TEST(amdsmitstReadOnly, WslFillAsicInfo) {
  amd::smi::WslGpuInfo gpu;
  gpu.market_name = "AMD Radeon Test GPU";
  gpu.device_id = 0x1586;
  gpu.num_compute_units = 40;
  amdsmi_asic_info_t info{};

  ASSERT_EQ(amd::smi::wsl_fill_asic_info(gpu, &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_STREQ(info.market_name, "AMD Radeon Test GPU");
  EXPECT_EQ(info.vendor_id, 0x1002U);
  EXPECT_EQ(info.device_id, 0x1586U);
  EXPECT_EQ(info.num_of_compute_units, 40U);
  EXPECT_EQ(info.rev_id, UINT32_MAX);
}

TEST(amdsmitstReadOnly, WslFillBoardInfo) {
  amd::smi::WslGpuInfo gpu;
  gpu.market_name = "AMD Radeon Test GPU";
  amdsmi_board_info_t info{};

  ASSERT_EQ(amd::smi::wsl_fill_board_info(gpu, &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_STREQ(info.product_name, "AMD Radeon Test GPU");
  EXPECT_STREQ(info.manufacturer_name, "Advanced Micro Devices, Inc. [AMD/ATI]");
}

TEST(amdsmitstReadOnly, WslFillInfoRejectsNullOutput) {
  const amd::smi::WslGpuInfo gpu;

  EXPECT_EQ(amd::smi::wsl_fill_asic_info(gpu, nullptr), AMDSMI_STATUS_INVAL);
  EXPECT_EQ(amd::smi::wsl_fill_board_info(gpu, nullptr), AMDSMI_STATUS_INVAL);
  EXPECT_EQ(amd::smi::wsl_fill_vram_info(gpu, nullptr), AMDSMI_STATUS_INVAL);
  EXPECT_EQ(amd::smi::wsl_fill_vram_usage(gpu, nullptr), AMDSMI_STATUS_INVAL);
  EXPECT_EQ(amd::smi::wsl_get_vram_usage_bytes(gpu, nullptr), AMDSMI_STATUS_INVAL);
  EXPECT_EQ(amd::smi::wsl_generate_device_uuid(gpu, nullptr), AMDSMI_STATUS_INVAL);
}

TEST(amdsmitstReadOnly, WslUuidIsStableAndDistinctByBdf) {
  amd::smi::WslGpuInfo first;
  first.device_id = 0x1586;
  first.bdf.bdf.domain_number = 0;
  first.bdf.bdf.bus_number = 0xc3;
  first.bdf.bdf.device_number = 0;
  first.bdf.bdf.function_number = 0;
  amd::smi::WslGpuInfo second = first;
  second.bdf.bdf.bus_number = 0xc4;
  char first_uuid[AMDSMI_GPU_UUID_SIZE] = {};
  char first_uuid_again[AMDSMI_GPU_UUID_SIZE] = {};
  char second_uuid[AMDSMI_GPU_UUID_SIZE] = {};

  ASSERT_EQ(amd::smi::wsl_generate_device_uuid(first, first_uuid), AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(amd::smi::wsl_generate_device_uuid(first, first_uuid_again), AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(amd::smi::wsl_generate_device_uuid(second, second_uuid), AMDSMI_STATUS_SUCCESS);
  EXPECT_STREQ(first_uuid, first_uuid_again);
  EXPECT_NE(std::strcmp(first_uuid, second_uuid), 0);
}

}  // namespace
