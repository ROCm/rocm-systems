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

// Tests for the rocdxg_smi_* public C ABI (new telemetry functions).
//
// The whole file is a no-op unless the WSL backend was compiled in
// (ENABLE_WSL_BACKEND=ON), so it links cleanly in native builds. When compiled
// in, tests fall into two groups:
//
//   WslSmiNull*  — null-output-pointer → INVAL; no /dev/dxg required.
//   WslSmiLive*  — end-to-end queries that require a live WSL GPU; skipped
//                  automatically when rocdxg_smi_get_device_count returns 0.

#include "config/amd_smi_config.h"

#if defined(ENABLE_WSL_BACKEND)

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

#include "hsakmt/hsakmt.h"
#include "hsakmt/rocdxg_smi.h"

namespace {

// ---------------------------------------------------------------------------
// Fixture: open KFD and confirm at least one GPU is visible.
// Tests in WslSmiLive are skipped when no GPU is found.
// ---------------------------------------------------------------------------
class WslSmiLive : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    HSAKMT_STATUS r = hsaKmtOpenKFD();
    if (r != HSAKMT_STATUS_SUCCESS && r != HSAKMT_STATUS_KERNEL_ALREADY_OPENED) return;
    kfd_opened_ = true;

    // Enumerate GPU nodes using the same path as WSLGPUBackend::TryPopulate.
    // rocdxg_smi_get_device_count requires the full topology snapshot which
    // hsaKmtAcquireSystemProperties sets up.
    HsaSystemProperties sys_props = {};
    r = hsaKmtAcquireSystemProperties(&sys_props);
    if (r != HSAKMT_STATUS_SUCCESS) return;

    for (uint32_t i = 0; i < sys_props.NumNodes; ++i) {
      HsaNodeProperties node_props = {};
      if (hsaKmtGetNodeProperties(i, &node_props) != HSAKMT_STATUS_SUCCESS) continue;
      if (node_props.NumFComputeCores > 0 && !gpu_ok_) {
        node_id_ = i;
        gpu_ok_ = true;
      }
    }
    // Do NOT call hsaKmtReleaseSystemProperties here — it clears the internal
    // wdevices_ snapshot that rocdxg_smi_* functions need for the process lifetime.
    // It is called in TearDownTestSuite instead.
  }

  static void TearDownTestSuite() {
    if (kfd_opened_) {
      hsaKmtReleaseSystemProperties();
      hsaKmtCloseKFD();
    }
  }

  void SetUp() override {
    if (!gpu_ok_)
      GTEST_SKIP() << "No WSL GPU available (no node with NumFComputeCores > 0)";
  }

  static bool kfd_opened_;
  static bool gpu_ok_;
  static uint32_t node_id_;
};

bool WslSmiLive::kfd_opened_ = false;
bool WslSmiLive::gpu_ok_     = false;
uint32_t WslSmiLive::node_id_ = 0;

}  // namespace

// ============================================================================
// WslSmiNull — null-pointer rejection; no /dev/dxg required
// ============================================================================

TEST(WslSmiNull, PowerInfoNullReturnsInval) {
  EXPECT_EQ(rocdxg_smi_get_power_info(0, nullptr), HSAKMT_STATUS_INVALID_PARAMETER);
}

TEST(WslSmiNull, TemperatureNullReturnsInval) {
  EXPECT_EQ(rocdxg_smi_get_temperature(0, 0, 0, nullptr), HSAKMT_STATUS_INVALID_PARAMETER);
}

TEST(WslSmiNull, ClockInfoNullReturnsInval) {
  EXPECT_EQ(rocdxg_smi_get_clock_info(0, 0, nullptr), HSAKMT_STATUS_INVALID_PARAMETER);
}

TEST(WslSmiNull, PcieInfoNullReturnsInval) {
  EXPECT_EQ(rocdxg_smi_get_pcie_info(0, nullptr), HSAKMT_STATUS_INVALID_PARAMETER);
}

TEST(WslSmiNull, DeviceInfoNullReturnsInval) {
  EXPECT_EQ(rocdxg_smi_get_device_info(0, nullptr), HSAKMT_STATUS_INVALID_PARAMETER);
}

TEST(WslSmiNull, GpuMetricsInfoNullReturnsInval) {
  EXPECT_EQ(rocdxg_smi_get_gpu_metrics_info(0, nullptr), HSAKMT_STATUS_INVALID_PARAMETER);
}

TEST(WslSmiNull, DeviceCountNullReturnsInval) {
  EXPECT_EQ(rocdxg_smi_get_device_count(nullptr), HSAKMT_STATUS_INVALID_PARAMETER);
}

// Unsupported sensor types must not crash and must return NOT_SUPPORTED or INVAL.
TEST(WslSmiNull, TemperatureUnsupportedMetric) {
  int64_t temp = 0;
  // metric != 0 is currently unsupported for all sensor types
  HSAKMT_STATUS r = rocdxg_smi_get_temperature(0, 0, /*metric=*/99, &temp);
  EXPECT_TRUE(r == HSAKMT_STATUS_NOT_SUPPORTED || r == HSAKMT_STATUS_INVALID_NODE_UNIT ||
              r == HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED)
      << "unexpected status: " << r;
}

TEST(WslSmiNull, ClockInfoUnsupportedType) {
  rocdxg_smi_clock_info_t info{};
  // clk_type 99 is not GFX/SOC/MEM — must not crash
  HSAKMT_STATUS r = rocdxg_smi_get_clock_info(0, 99, &info);
  EXPECT_TRUE(r == HSAKMT_STATUS_NOT_SUPPORTED || r == HSAKMT_STATUS_INVALID_NODE_UNIT ||
              r == HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED)
      << "unexpected status: " << r;
}

// ============================================================================
// WslSmiLive — requires a live WSL GPU (skipped otherwise)
// ============================================================================

TEST_F(WslSmiLive, DeviceCountNonZero) {
  uint32_t count = 0;
  ASSERT_EQ(rocdxg_smi_get_device_count(&count), HSAKMT_STATUS_SUCCESS);
  EXPECT_GT(count, 0u);
}

TEST_F(WslSmiLive, DeviceInfoPopulated) {
  rocdxg_smi_device_info_t info{};
  ASSERT_EQ(rocdxg_smi_get_device_info(node_id_, &info), HSAKMT_STATUS_SUCCESS);

  // BDF: at least one field non-zero
  EXPECT_TRUE(info.bdf.bus_number != 0 || info.bdf.device_number != 0 ||
              info.bdf.domain_number != 0);

  // ASIC: vendor=AMD, non-zero CUs and device_id
  EXPECT_EQ(info.asic.vendor_id, 0x1002u);
  EXPECT_GT(info.asic.num_of_compute_units, 0u);
  EXPECT_NE(info.asic.device_id, 0u);

  // Board: product_name and manufacturer_name non-empty
  EXPECT_NE(info.board.product_name[0], '\0');
  EXPECT_NE(info.board.manufacturer_name[0], '\0');

  // VRAM: size non-zero
  EXPECT_GT(info.vram.vram_size_mb, 0u);
}

TEST_F(WslSmiLive, VramUsageWithinTotal) {
  rocdxg_smi_vram_usage_t usage{};
  ASSERT_EQ(rocdxg_smi_get_vram_usage(node_id_, &usage), HSAKMT_STATUS_SUCCESS);
  EXPECT_GT(usage.vram_total_mb, 0u);
  EXPECT_LE(usage.vram_used_mb, usage.vram_total_mb);
}

TEST_F(WslSmiLive, PowerInfoSuccessOrNotSupported) {
  rocdxg_smi_power_info_t info{};
  HSAKMT_STATUS r = rocdxg_smi_get_power_info(node_id_, &info);
  EXPECT_TRUE(r == HSAKMT_STATUS_SUCCESS || r == HSAKMT_STATUS_NOT_SUPPORTED)
      << "unexpected status: " << r;
  if (r == HSAKMT_STATUS_SUCCESS) {
    // current_socket_power is in Watts — sanity bounds 0-1000 W
    EXPECT_LT(info.current_socket_power, 1000u);
  }
}

TEST_F(WslSmiLive, TemperatureEdgeSuccessOrNotSupported) {
  int64_t temp = -1;
  HSAKMT_STATUS r = rocdxg_smi_get_temperature(node_id_,
                                                /*sensor_type=*/0,  // EDGE
                                                /*metric=*/0,        // CURRENT
                                                &temp);
  EXPECT_TRUE(r == HSAKMT_STATUS_SUCCESS || r == HSAKMT_STATUS_NOT_SUPPORTED)
      << "unexpected status: " << r;
  if (r == HSAKMT_STATUS_SUCCESS) {
    EXPECT_GT(temp, 0);    // non-negative °C
    EXPECT_LT(temp, 150);  // sanity: below 150 °C
  }
}

TEST_F(WslSmiLive, ClockInfoGfxSuccessOrNotSupported) {
  rocdxg_smi_clock_info_t info{};
  HSAKMT_STATUS r = rocdxg_smi_get_clock_info(node_id_, /*GFX=*/0, &info);
  EXPECT_TRUE(r == HSAKMT_STATUS_SUCCESS || r == HSAKMT_STATUS_NOT_SUPPORTED)
      << "unexpected status: " << r;
  if (r == HSAKMT_STATUS_SUCCESS) {
    // max_clk always comes from static WKMI data — must be non-zero
    EXPECT_GT(info.max_clk, 0u);
  }
}

TEST_F(WslSmiLive, ClockInfoMemSuccessOrNotSupported) {
  rocdxg_smi_clock_info_t info{};
  HSAKMT_STATUS r = rocdxg_smi_get_clock_info(node_id_, /*MEM=*/4, &info);
  EXPECT_TRUE(r == HSAKMT_STATUS_SUCCESS || r == HSAKMT_STATUS_NOT_SUPPORTED)
      << "unexpected status: " << r;
  if (r == HSAKMT_STATUS_SUCCESS) {
    EXPECT_GT(info.max_clk, 0u);
  }
}

TEST_F(WslSmiLive, PcieInfoSuccessOrNotSupported) {
  rocdxg_smi_pcie_info_t info{};
  HSAKMT_STATUS r = rocdxg_smi_get_pcie_info(node_id_, &info);
  EXPECT_TRUE(r == HSAKMT_STATUS_SUCCESS || r == HSAKMT_STATUS_NOT_SUPPORTED ||
              r == HSAKMT_STATUS_BUFFER_TOO_SMALL)
      << "unexpected status: " << r;
  if (r == HSAKMT_STATUS_SUCCESS) {
    EXPECT_GT(info.max_pcie_width, 0u);
    EXPECT_GE(info.pcie_interface_version, 1u);
    EXPECT_LE(info.pcie_interface_version, 5u);
  }
}

TEST_F(WslSmiLive, DeviceInfoDriverAndVbiosFields) {
  rocdxg_smi_device_info_t info{};
  HSAKMT_STATUS r = rocdxg_smi_get_device_info(node_id_, &info);
  EXPECT_TRUE(r == HSAKMT_STATUS_SUCCESS || r == HSAKMT_STATUS_NOT_SUPPORTED)
      << "unexpected status: " << r;
  if (r == HSAKMT_STATUS_SUCCESS) {
    // At least one driver field must be non-empty
    bool driver_filled = info.driver.driver_version[0] != '\0' ||
                         info.driver.driver_date[0]    != '\0' ||
                         info.driver.driver_name[0]    != '\0';
    EXPECT_TRUE(driver_filled);
  }
}

TEST_F(WslSmiLive, GpuMetricsInfoSuccessOrNotSupported) {
  rocdxg_smi_gpu_metrics_info_t info{};
  HSAKMT_STATUS r = rocdxg_smi_get_gpu_metrics_info(node_id_, &info);
  EXPECT_TRUE(r == HSAKMT_STATUS_SUCCESS || r == HSAKMT_STATUS_NOT_SUPPORTED)
      << "unexpected status: " << r;
  // No further value assertions: all fields are UINT32_MAX when the
  // corresponding PMLog sensor is unavailable, which is valid.
}

TEST_F(WslSmiLive, ProcessEnumSuccessOrNotSupported) {
  uint32_t count = 0;
  HSAKMT_STATUS r = rocdxg_smi_enum_processes(node_id_, &count, nullptr);
  EXPECT_TRUE(r == HSAKMT_STATUS_SUCCESS || r == HSAKMT_STATUS_NOT_SUPPORTED)
      << "unexpected status: " << r;
}

#endif  // ENABLE_WSL_BACKEND
