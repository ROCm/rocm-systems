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

// Tests for WSLGPUBackend's public IGPUBackend interface.
//
// No-op unless the WSL backend was compiled in (ENABLE_WSL_BACKEND=ON).
// Tests split into two groups:
//   Null*  — null-output-pointer → INVAL; no /dev/dxg required.
//   Live*  — end-to-end queries; skipped when no WSL GPU is present.

#include "config/amd_smi_config.h"

#if defined(ENABLE_WSL_BACKEND)

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <set>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_gpu_backend.h"
#include "amd_smi/impl/amd_smi_gpu_device.h"
#include "amd_smi/impl/amd_smi_processor.h"
#include "amd_smi/impl/amd_smi_socket.h"
#include "amd_smi/impl/amd_smi_wsl_device.h"

using amd::smi::AMDSmiGPUDevice;
using amd::smi::AMDSmiProcessor;
using amd::smi::AMDSmiSocket;
using amd::smi::IGPUBackend;
using amd::smi::WSLGPUBackend;

namespace {

// ---------------------------------------------------------------------------
// Suite fixture: populates the WSL backend at suite startup via the same
// public bootstrap path amdsmi itself uses (WSLGPUBackend::try_populate).
// Live tests call RequireGpu() to skip when no WSL GPU is present.
// ---------------------------------------------------------------------------
class WslFunctionalReadOnly : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    amdsmi_status_t r = WSLGPUBackend::try_populate(sockets_, processors_);
    if (r != AMDSMI_STATUS_SUCCESS) return;  // not WSL, or no adapter opened
    if (sockets_.empty()) return;

    auto* device = static_cast<AMDSmiGPUDevice*>(*processors_.begin());
    backend_ = device->backend();
    gpu_ok_ = backend_ != nullptr;
  }

  static void TearDownTestSuite() {
    WSLGPUBackend::shutdown();
    for (auto* socket : sockets_) delete socket;
    sockets_.clear();
    processors_.clear();
    backend_ = nullptr;
    gpu_ok_ = false;
  }

  // Call at the start of live tests that require an actual WSL GPU.
  // Uses GTEST_SKIP_ directly so the macro's `return` exits the test body.
#define RequireGpu()                                                \
  do {                                                              \
    if (!gpu_ok_)                                                   \
      GTEST_SKIP() << "No WSL GPU available (try_populate did not " \
                      "find/open an adapter)";                      \
  } while (0)

  static std::vector<AMDSmiSocket*> sockets_;
  static std::set<AMDSmiProcessor*> processors_;
  static IGPUBackend* backend_;
  static bool gpu_ok_;
};

std::vector<AMDSmiSocket*> WslFunctionalReadOnly::sockets_;
std::set<AMDSmiProcessor*> WslFunctionalReadOnly::processors_;
IGPUBackend* WslFunctionalReadOnly::backend_ = nullptr;
bool WslFunctionalReadOnly::gpu_ok_ = false;

}  // namespace

// ============================================================================
// Null-pointer rejection — no /dev/dxg required. A fresh WSLGPUBackend isn't
// reachable without try_populate() succeeding, so these run against backend_
// when available and are skipped otherwise (nullptr-checks are identical
// across all WSLGPUBackend instances since they run before any adapter I/O).
// ============================================================================

TEST_F(WslFunctionalReadOnly, NullAsicInfoReturnsInval) {
  if (!gpu_ok_) GTEST_SKIP() << "No WSL GPU available";
  EXPECT_EQ(backend_->get_asic_info(nullptr), AMDSMI_STATUS_INVAL);
}

TEST_F(WslFunctionalReadOnly, NullBoardInfoReturnsInval) {
  if (!gpu_ok_) GTEST_SKIP() << "No WSL GPU available";
  EXPECT_EQ(backend_->get_board_info(nullptr), AMDSMI_STATUS_INVAL);
}

TEST_F(WslFunctionalReadOnly, NullPowerInfoReturnsInval) {
  if (!gpu_ok_) GTEST_SKIP() << "No WSL GPU available";
  EXPECT_EQ(backend_->get_power_info(nullptr), AMDSMI_STATUS_INVAL);
}

TEST_F(WslFunctionalReadOnly, NullTemperatureReturnsInval) {
  if (!gpu_ok_) GTEST_SKIP() << "No WSL GPU available";
  EXPECT_EQ(backend_->get_temp_metric(AMDSMI_TEMPERATURE_TYPE_EDGE, AMDSMI_TEMP_CURRENT, nullptr),
            AMDSMI_STATUS_INVAL);
}

TEST_F(WslFunctionalReadOnly, NullClockInfoReturnsInval) {
  if (!gpu_ok_) GTEST_SKIP() << "No WSL GPU available";
  EXPECT_EQ(backend_->get_clock_info(AMDSMI_CLK_TYPE_SYS, nullptr), AMDSMI_STATUS_INVAL);
}

TEST_F(WslFunctionalReadOnly, NullPcieInfoReturnsInval) {
  if (!gpu_ok_) GTEST_SKIP() << "No WSL GPU available";
  EXPECT_EQ(backend_->get_pcie_info(nullptr), AMDSMI_STATUS_INVAL);
}

TEST_F(WslFunctionalReadOnly, NullGpuMetricsInfoReturnsInval) {
  if (!gpu_ok_) GTEST_SKIP() << "No WSL GPU available";
  EXPECT_EQ(backend_->get_gpu_metrics_info(nullptr), AMDSMI_STATUS_INVAL);
}

TEST_F(WslFunctionalReadOnly, NullMemoryUsageReturnsInval) {
  if (!gpu_ok_) GTEST_SKIP() << "No WSL GPU available";
  EXPECT_EQ(backend_->get_memory_usage(AMDSMI_MEM_TYPE_VRAM, nullptr), AMDSMI_STATUS_INVAL);
}

TEST_F(WslFunctionalReadOnly, NullUuidReturnsInval) {
  if (!gpu_ok_) GTEST_SKIP() << "No WSL GPU available";
  unsigned int len = AMDSMI_GPU_UUID_SIZE;
  EXPECT_EQ(backend_->get_uuid(&len, nullptr), AMDSMI_STATUS_INVAL);
  EXPECT_EQ(backend_->get_uuid(nullptr, nullptr), AMDSMI_STATUS_INVAL);
}

// Unsupported sensor/clock types must not crash and must return NOT_SUPPORTED.
TEST_F(WslFunctionalReadOnly, TemperatureUnsupportedMetric) {
  if (!gpu_ok_) GTEST_SKIP() << "No WSL GPU available";
  int64_t temp = 0;
  amdsmi_status_t r = backend_->get_temp_metric(
      AMDSMI_TEMPERATURE_TYPE_EDGE, static_cast<amdsmi_temperature_metric_t>(99), &temp);
  EXPECT_EQ(r, AMDSMI_STATUS_NOT_SUPPORTED);
}

TEST_F(WslFunctionalReadOnly, ClockInfoUnsupportedType) {
  if (!gpu_ok_) GTEST_SKIP() << "No WSL GPU available";
  amdsmi_clk_info_t info{};
  amdsmi_status_t r = backend_->get_clock_info(static_cast<amdsmi_clk_type_t>(99), &info);
  EXPECT_EQ(r, AMDSMI_STATUS_NOT_SUPPORTED);
}

TEST_F(WslFunctionalReadOnly, MemoryTypeUnsupportedReturnsNotSupported) {
  if (!gpu_ok_) GTEST_SKIP() << "No WSL GPU available";
  uint64_t used = 0;
  amdsmi_status_t r = backend_->get_memory_usage(AMDSMI_MEM_TYPE_GTT, &used);
  EXPECT_EQ(r, AMDSMI_STATUS_NOT_SUPPORTED);
}

// ============================================================================
// Live queries — require a WSL GPU (skipped otherwise)
// ============================================================================

TEST_F(WslFunctionalReadOnly, LiveDeviceCountNonZero) {
  RequireGpu();
  EXPECT_GT(sockets_.size(), 0u);
  EXPECT_GT(processors_.size(), 0u);
}

TEST_F(WslFunctionalReadOnly, LiveAsicInfoPopulated) {
  RequireGpu();
  amdsmi_asic_info_t info{};
  ASSERT_EQ(backend_->get_asic_info(&info), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.vendor_id, 0x1002u);
  EXPECT_GT(info.num_of_compute_units, 0u);
  EXPECT_NE(info.device_id, 0u);
}

TEST_F(WslFunctionalReadOnly, LiveBoardInfoPopulated) {
  RequireGpu();
  amdsmi_board_info_t info{};
  ASSERT_EQ(backend_->get_board_info(&info), AMDSMI_STATUS_SUCCESS);
  EXPECT_NE(info.product_name[0], '\0');
  EXPECT_NE(info.manufacturer_name[0], '\0');
}

TEST_F(WslFunctionalReadOnly, LiveVramInfoSizeNonZero) {
  RequireGpu();
  amdsmi_vram_info_t info{};
  ASSERT_EQ(backend_->get_vram_info(&info), AMDSMI_STATUS_SUCCESS);
  EXPECT_GT(info.vram_size, 0u);
}

TEST_F(WslFunctionalReadOnly, LiveVramUsageWithinTotal) {
  RequireGpu();
  uint64_t total = 0;
  uint64_t used = 0;
  ASSERT_EQ(backend_->get_memory_total(AMDSMI_MEM_TYPE_VRAM, &total), AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(backend_->get_memory_usage(AMDSMI_MEM_TYPE_VRAM, &used), AMDSMI_STATUS_SUCCESS);
  EXPECT_GT(total, 0u);
  EXPECT_LE(used, total);
}

TEST_F(WslFunctionalReadOnly, LivePowerInfoSuccessOrNotSupported) {
  RequireGpu();
  amdsmi_power_info_t info{};
  amdsmi_status_t r = backend_->get_power_info(&info);
  EXPECT_TRUE(r == AMDSMI_STATUS_SUCCESS || r == AMDSMI_STATUS_API_FAILED)
      << "unexpected status: " << r;
  if (r == AMDSMI_STATUS_SUCCESS && info.current_socket_power != UINT32_MAX) {
    // current_socket_power is in Watts — sanity bounds 0-1000 W
    EXPECT_LT(info.current_socket_power, 1000u);
  }
}

TEST_F(WslFunctionalReadOnly, LiveTemperatureEdgeSuccessOrNotSupported) {
  RequireGpu();
  int64_t temp = -1;
  amdsmi_status_t r =
      backend_->get_temp_metric(AMDSMI_TEMPERATURE_TYPE_EDGE, AMDSMI_TEMP_CURRENT, &temp);
  EXPECT_TRUE(r == AMDSMI_STATUS_SUCCESS || r == AMDSMI_STATUS_NOT_SUPPORTED ||
              r == AMDSMI_STATUS_API_FAILED)
      << "unexpected status: " << r;
  if (r == AMDSMI_STATUS_SUCCESS) {
    EXPECT_GT(temp, 0);    // non-negative °C
    EXPECT_LT(temp, 150);  // sanity: below 150 °C
  }
}

TEST_F(WslFunctionalReadOnly, LiveClockInfoGfxSuccessOrNotSupported) {
  RequireGpu();
  amdsmi_clk_info_t info{};
  amdsmi_status_t r = backend_->get_clock_info(AMDSMI_CLK_TYPE_SYS, &info);
  EXPECT_TRUE(r == AMDSMI_STATUS_SUCCESS || r == AMDSMI_STATUS_NOT_SUPPORTED)
      << "unexpected status: " << r;
  if (r == AMDSMI_STATUS_SUCCESS) {
    // max_clk always comes from static device info — must be non-zero
    EXPECT_GT(info.max_clk, 0u);
  }
}

TEST_F(WslFunctionalReadOnly, LiveClockInfoMemSuccessOrNotSupported) {
  RequireGpu();
  amdsmi_clk_info_t info{};
  amdsmi_status_t r = backend_->get_clock_info(AMDSMI_CLK_TYPE_MEM, &info);
  EXPECT_TRUE(r == AMDSMI_STATUS_SUCCESS || r == AMDSMI_STATUS_NOT_SUPPORTED)
      << "unexpected status: " << r;
  if (r == AMDSMI_STATUS_SUCCESS) {
    EXPECT_GT(info.max_clk, 0u);
  }
}

TEST_F(WslFunctionalReadOnly, LivePcieInfoSuccessOrNotSupported) {
  RequireGpu();
  amdsmi_pcie_info_t info{};
  amdsmi_status_t r = backend_->get_pcie_info(&info);
  EXPECT_TRUE(r == AMDSMI_STATUS_SUCCESS || r == AMDSMI_STATUS_API_FAILED)
      << "unexpected status: " << r;
  if (r == AMDSMI_STATUS_SUCCESS) {
    EXPECT_GT(info.pcie_static.max_pcie_width, 0u);
    EXPECT_GE(info.pcie_static.pcie_interface_version, 1u);
    EXPECT_LE(info.pcie_static.pcie_interface_version, 5u);
  }
}

TEST_F(WslFunctionalReadOnly, LiveDriverInfoFieldsPopulated) {
  RequireGpu();
  amdsmi_driver_info_t info{};
  ASSERT_EQ(backend_->get_driver_info(&info), AMDSMI_STATUS_SUCCESS);
  // At least one driver field must be non-empty
  bool driver_filled =
      info.driver_version[0] != '\0' || info.driver_date[0] != '\0' || info.driver_name[0] != '\0';
  EXPECT_TRUE(driver_filled);
}

TEST_F(WslFunctionalReadOnly, LiveGpuMetricsInfoSuccessOrNotSupported) {
  RequireGpu();
  amdsmi_gpu_metrics_t info{};
  amdsmi_status_t r = backend_->get_gpu_metrics_info(&info);
  EXPECT_TRUE(r == AMDSMI_STATUS_SUCCESS || r == AMDSMI_STATUS_API_FAILED)
      << "unexpected status: " << r;
  // No further value assertions: unavailable PMLog sensors leave their fields
  // at the 0xFF sentinel, which is valid.
}

TEST_F(WslFunctionalReadOnly, LiveUuidSuccessOrNotSupported) {
  RequireGpu();
  char uuid[AMDSMI_GPU_UUID_SIZE] = {};
  unsigned int len = AMDSMI_GPU_UUID_SIZE;
  amdsmi_status_t r = backend_->get_uuid(&len, uuid);
  EXPECT_TRUE(r == AMDSMI_STATUS_SUCCESS || r == AMDSMI_STATUS_NOT_SUPPORTED)
      << "unexpected status: " << r;
  if (r == AMDSMI_STATUS_SUCCESS) {
    EXPECT_EQ(len, static_cast<unsigned int>(AMDSMI_GPU_UUID_SIZE));
  }
}

#endif  // ENABLE_WSL_BACKEND
