// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Tests for the rocdxg_smi_* public C ABI.
//
// No-op unless the WSL backend was compiled in (ENABLE_WSL_BACKEND=ON).
// Tests split into two groups:
//   Null*  — null-output-pointer → INVAL; no /dev/dxg required.
//   Live*  — end-to-end queries; skipped when no WSL GPU is present.

#include "config/amd_smi_config.h"

#if defined(ENABLE_WSL_BACKEND)

#include <gtest/gtest.h>
#include <sys/stat.h>

#include <cstdint>
#include <cstring>

#include "hsakmt/hsakmt.h"
#include "hsakmt/rocdxg_smi.h"

namespace {

// ---------------------------------------------------------------------------
// Suite fixture: opens KFD and enumerates GPU nodes at suite startup.
// Live tests call RequireGpu() to skip when no WSL GPU is present.
// ---------------------------------------------------------------------------
class WslFunctionalReadOnly : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    struct stat st{};
    if (stat("/dev/dxg", &st) != 0) return;  // not WSL2
    wsl_present_ = true;

    HSAKMT_STATUS r = hsaKmtOpenKFD();
    if (r != HSAKMT_STATUS_SUCCESS && r != HSAKMT_STATUS_KERNEL_ALREADY_OPENED) return;
    kfd_opened_ = true;

    // Enumerate GPU nodes using the same path as WSLGPUBackend::try_populate.
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

  // Call at the start of live tests that require an actual WSL GPU.
  // Uses GTEST_SKIP_ directly so the macro's `return` exits the test body.
#define RequireGpu()                                                                          \
  do {                                                                                        \
    if (!wsl_present_) GTEST_SKIP() << "No /dev/dxg — not running under WSL2";                \
    if (!gpu_ok_) GTEST_SKIP() << "No WSL GPU available (no node with NumFComputeCores > 0)"; \
  } while (0)

  static bool wsl_present_;
  static bool kfd_opened_;
  static bool gpu_ok_;
  static uint32_t node_id_;
};

bool WslFunctionalReadOnly::wsl_present_ = false;
bool WslFunctionalReadOnly::kfd_opened_ = false;
bool WslFunctionalReadOnly::gpu_ok_ = false;
uint32_t WslFunctionalReadOnly::node_id_ = 0;

}  // namespace

// ============================================================================
// Null-pointer rejection — no /dev/dxg required
// ============================================================================

TEST_F(WslFunctionalReadOnly, NullPowerInfoReturnsInval) {
  EXPECT_EQ(rocdxg_smi_get_power_info(0, nullptr), HSAKMT_STATUS_INVALID_PARAMETER);
}

TEST_F(WslFunctionalReadOnly, NullTemperatureReturnsInval) {
  EXPECT_EQ(rocdxg_smi_get_temperature(0, 0, 0, nullptr), HSAKMT_STATUS_INVALID_PARAMETER);
}

TEST_F(WslFunctionalReadOnly, NullClockInfoReturnsInval) {
  EXPECT_EQ(rocdxg_smi_get_clock_info(0, 0, nullptr), HSAKMT_STATUS_INVALID_PARAMETER);
}

TEST_F(WslFunctionalReadOnly, NullPcieInfoReturnsInval) {
  EXPECT_EQ(rocdxg_smi_get_pcie_info(0, nullptr), HSAKMT_STATUS_INVALID_PARAMETER);
}

TEST_F(WslFunctionalReadOnly, NullDeviceInfoReturnsInval) {
  EXPECT_EQ(rocdxg_smi_get_device_info(0, nullptr), HSAKMT_STATUS_INVALID_PARAMETER);
}

TEST_F(WslFunctionalReadOnly, NullGpuMetricsInfoReturnsInval) {
  EXPECT_EQ(rocdxg_smi_get_gpu_metrics_info(0, nullptr), HSAKMT_STATUS_INVALID_PARAMETER);
}

TEST_F(WslFunctionalReadOnly, NullDeviceCountReturnsInval) {
  EXPECT_EQ(rocdxg_smi_get_device_count(nullptr), HSAKMT_STATUS_INVALID_PARAMETER);
}

// Unsupported sensor types must not crash and must return NOT_SUPPORTED or INVAL.
TEST_F(WslFunctionalReadOnly, TemperatureUnsupportedMetric) {
  int64_t temp = 0;
  HSAKMT_STATUS r = rocdxg_smi_get_temperature(0, 0, /*metric=*/99, &temp);
  EXPECT_TRUE(r == HSAKMT_STATUS_NOT_SUPPORTED || r == HSAKMT_STATUS_INVALID_NODE_UNIT ||
              r == HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED)
      << "unexpected status: " << r;
}

TEST_F(WslFunctionalReadOnly, ClockInfoUnsupportedType) {
  rocdxg_smi_clock_info_t info{};
  HSAKMT_STATUS r = rocdxg_smi_get_clock_info(0, 99, &info);
  EXPECT_TRUE(r == HSAKMT_STATUS_NOT_SUPPORTED || r == HSAKMT_STATUS_INVALID_NODE_UNIT ||
              r == HSAKMT_STATUS_KERNEL_IO_CHANNEL_NOT_OPENED)
      << "unexpected status: " << r;
}

// ============================================================================
// Live queries — require a WSL GPU (skipped otherwise)
// ============================================================================

TEST_F(WslFunctionalReadOnly, LiveDeviceCountNonZero) {
  RequireGpu();
  uint32_t count = 0;
  ASSERT_EQ(rocdxg_smi_get_device_count(&count), HSAKMT_STATUS_SUCCESS);
  EXPECT_GT(count, 0u);
}

TEST_F(WslFunctionalReadOnly, LiveDeviceInfoPopulated) {
  RequireGpu();
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

TEST_F(WslFunctionalReadOnly, LiveVramUsageWithinTotal) {
  RequireGpu();
  rocdxg_smi_vram_usage_t usage{};
  ASSERT_EQ(rocdxg_smi_get_vram_usage(node_id_, &usage), HSAKMT_STATUS_SUCCESS);
  EXPECT_GT(usage.vram_total_mb, 0u);
  EXPECT_LE(usage.vram_used_mb, usage.vram_total_mb);
}

TEST_F(WslFunctionalReadOnly, LivePowerInfoSuccessOrNotSupported) {
  RequireGpu();
  rocdxg_smi_power_info_t info{};
  HSAKMT_STATUS r = rocdxg_smi_get_power_info(node_id_, &info);
  EXPECT_TRUE(r == HSAKMT_STATUS_SUCCESS || r == HSAKMT_STATUS_NOT_SUPPORTED)
      << "unexpected status: " << r;
  if (r == HSAKMT_STATUS_SUCCESS) {
    // current_socket_power is in Watts — sanity bounds 0-1000 W
    EXPECT_LT(info.current_socket_power, 1000u);
  }
}

TEST_F(WslFunctionalReadOnly, LiveTemperatureEdgeSuccessOrNotSupported) {
  RequireGpu();
  int64_t temp = -1;
  HSAKMT_STATUS r = rocdxg_smi_get_temperature(node_id_,
                                               /*sensor_type=*/0,  // EDGE
                                               /*metric=*/0,       // CURRENT
                                               &temp);
  EXPECT_TRUE(r == HSAKMT_STATUS_SUCCESS || r == HSAKMT_STATUS_NOT_SUPPORTED)
      << "unexpected status: " << r;
  if (r == HSAKMT_STATUS_SUCCESS) {
    EXPECT_GT(temp, 0);    // non-negative °C
    EXPECT_LT(temp, 150);  // sanity: below 150 °C
  }
}

TEST_F(WslFunctionalReadOnly, LiveClockInfoGfxSuccessOrNotSupported) {
  RequireGpu();
  rocdxg_smi_clock_info_t info{};
  HSAKMT_STATUS r = rocdxg_smi_get_clock_info(node_id_, /*GFX=*/0, &info);
  EXPECT_TRUE(r == HSAKMT_STATUS_SUCCESS || r == HSAKMT_STATUS_NOT_SUPPORTED)
      << "unexpected status: " << r;
  if (r == HSAKMT_STATUS_SUCCESS) {
    // max_clk always comes from static WKMI data — must be non-zero
    EXPECT_GT(info.max_clk, 0u);
  }
}

TEST_F(WslFunctionalReadOnly, LiveClockInfoMemSuccessOrNotSupported) {
  RequireGpu();
  rocdxg_smi_clock_info_t info{};
  HSAKMT_STATUS r = rocdxg_smi_get_clock_info(node_id_, /*MEM=*/4, &info);
  EXPECT_TRUE(r == HSAKMT_STATUS_SUCCESS || r == HSAKMT_STATUS_NOT_SUPPORTED)
      << "unexpected status: " << r;
  if (r == HSAKMT_STATUS_SUCCESS) {
    EXPECT_GT(info.max_clk, 0u);
  }
}

TEST_F(WslFunctionalReadOnly, LivePcieInfoSuccessOrNotSupported) {
  RequireGpu();
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

TEST_F(WslFunctionalReadOnly, LiveDeviceInfoDriverAndVbiosFields) {
  RequireGpu();
  rocdxg_smi_device_info_t info{};
  HSAKMT_STATUS r = rocdxg_smi_get_device_info(node_id_, &info);
  EXPECT_TRUE(r == HSAKMT_STATUS_SUCCESS || r == HSAKMT_STATUS_NOT_SUPPORTED)
      << "unexpected status: " << r;
  if (r == HSAKMT_STATUS_SUCCESS) {
    // At least one driver field must be non-empty
    bool driver_filled = info.driver.driver_version[0] != '\0' ||
                         info.driver.driver_date[0] != '\0' || info.driver.driver_name[0] != '\0';
    EXPECT_TRUE(driver_filled);
  }
}

TEST_F(WslFunctionalReadOnly, LiveGpuMetricsInfoSuccessOrNotSupported) {
  RequireGpu();
  rocdxg_smi_gpu_metrics_info_t info{};
  HSAKMT_STATUS r = rocdxg_smi_get_gpu_metrics_info(node_id_, &info);
  EXPECT_TRUE(r == HSAKMT_STATUS_SUCCESS || r == HSAKMT_STATUS_NOT_SUPPORTED)
      << "unexpected status: " << r;
  // No further value assertions: all fields are UINT32_MAX when the
  // corresponding PMLog sensor is unavailable, which is valid.
}

TEST_F(WslFunctionalReadOnly, LiveProcessEnumSuccessOrNotSupported) {
  RequireGpu();
  uint32_t count = 0;
  HSAKMT_STATUS r = rocdxg_smi_enum_processes(node_id_, &count, nullptr);
  EXPECT_TRUE(r == HSAKMT_STATUS_SUCCESS || r == HSAKMT_STATUS_NOT_SUPPORTED)
      << "unexpected status: " << r;
}

#endif  // ENABLE_WSL_BACKEND
