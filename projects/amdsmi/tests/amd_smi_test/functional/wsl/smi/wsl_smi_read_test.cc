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

// Read-only functional coverage for the librocdxg C ABI behind the WSL backend.
//
// librocdxg is resolved with dlopen, the same way amd_smi_wsl_device.cc does it,
// so this binary carries no link-time dependency on it. Every test skips when
// the library or a WSL GPU is absent, which is the normal case on native Linux.

#include "config/amd_smi_config.h"

#if defined(ENABLE_WSL_BACKEND)

#include <dlfcn.h>
#include <gtest/gtest.h>
#include <sys/stat.h>

#include <cstdint>
#include <cstring>

#include "amd_smi/impl/wsl/rocdxg_abi.h"

namespace {

// Subset of librocdxg resolved for these tests.
struct RocdxgSyms {
  HSAKMT_STATUS (*hsaKmtOpenKFD)() = nullptr;
  HSAKMT_STATUS (*hsaKmtCloseKFD)() = nullptr;
  HSAKMT_STATUS (*hsaKmtAcquireSystemProperties)(HsaSystemProperties*) = nullptr;
  HSAKMT_STATUS (*hsaKmtReleaseSystemProperties)() = nullptr;
  HSAKMT_STATUS (*rocdxg_smi_get_device_count)(uint32_t*) = nullptr;
  HSAKMT_STATUS (*rocdxg_smi_get_device_info)(uint32_t, rocdxg_smi_device_info_t*) = nullptr;
  HSAKMT_STATUS (*rocdxg_smi_get_vram_usage)(uint32_t, rocdxg_smi_vram_usage_t*) = nullptr;
  HSAKMT_STATUS (*rocdxg_smi_get_power_info)(uint32_t, rocdxg_smi_power_info_t*) = nullptr;
  HSAKMT_STATUS (*rocdxg_smi_get_temperature)(uint32_t, uint32_t, uint32_t, int64_t*) = nullptr;
  HSAKMT_STATUS (*rocdxg_smi_get_clock_info)(uint32_t, uint32_t,
                                             rocdxg_smi_clock_info_t*) = nullptr;
  HSAKMT_STATUS (*rocdxg_smi_get_pcie_info)(uint32_t, rocdxg_smi_pcie_info_t*) = nullptr;
  HSAKMT_STATUS (*rocdxg_smi_get_gpu_metrics_info)(uint32_t,
                                                   rocdxg_smi_gpu_metrics_info_t*) = nullptr;
  HSAKMT_STATUS (*rocdxg_smi_enum_processes)(uint32_t, uint32_t*,
                                             rocdxg_smi_process_info_t*) = nullptr;
};

template <typename T>
bool bind(void* handle, const char* name, T& fn) {
  fn = reinterpret_cast<T>(dlsym(handle, name));
  return fn != nullptr;
}

// Opens librocdxg and the DXG topology once for the suite.
class WslFunctionalReadOnly : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    struct stat st{};
    if (stat("/dev/dxg", &st) != 0) return;
    wsl_present_ = true;

    handle_ = dlopen("librocdxg.so.1", RTLD_NOW | RTLD_LOCAL);
    if (handle_ == nullptr) handle_ = dlopen("librocdxg.so", RTLD_NOW | RTLD_LOCAL);
    if (handle_ == nullptr) return;

    bool ok = true;
    ok &= bind(handle_, "hsaKmtOpenKFD", syms_.hsaKmtOpenKFD);
    ok &= bind(handle_, "hsaKmtCloseKFD", syms_.hsaKmtCloseKFD);
    ok &= bind(handle_, "hsaKmtAcquireSystemProperties", syms_.hsaKmtAcquireSystemProperties);
    ok &= bind(handle_, "hsaKmtReleaseSystemProperties", syms_.hsaKmtReleaseSystemProperties);
    ok &= bind(handle_, "rocdxg_smi_get_device_count", syms_.rocdxg_smi_get_device_count);
    ok &= bind(handle_, "rocdxg_smi_get_device_info", syms_.rocdxg_smi_get_device_info);
    ok &= bind(handle_, "rocdxg_smi_get_vram_usage", syms_.rocdxg_smi_get_vram_usage);
    ok &= bind(handle_, "rocdxg_smi_get_power_info", syms_.rocdxg_smi_get_power_info);
    ok &= bind(handle_, "rocdxg_smi_get_temperature", syms_.rocdxg_smi_get_temperature);
    ok &= bind(handle_, "rocdxg_smi_get_clock_info", syms_.rocdxg_smi_get_clock_info);
    ok &= bind(handle_, "rocdxg_smi_get_pcie_info", syms_.rocdxg_smi_get_pcie_info);
    ok &= bind(handle_, "rocdxg_smi_get_gpu_metrics_info", syms_.rocdxg_smi_get_gpu_metrics_info);
    ok &= bind(handle_, "rocdxg_smi_enum_processes", syms_.rocdxg_smi_enum_processes);
    if (!ok) return;
    lib_ok_ = true;

    HSAKMT_STATUS r = syms_.hsaKmtOpenKFD();
    if (r != HSAKMT_STATUS_SUCCESS && r != HSAKMT_STATUS_KERNEL_ALREADY_OPENED) return;
    kfd_opened_ = true;

    // Builds the WDDM device list every rocdxg_smi_* call reads. Released in
    // TearDownTestSuite, not here: releasing drops that snapshot.
    HsaSystemProperties sys_props = {};
    if (syms_.hsaKmtAcquireSystemProperties(&sys_props) != HSAKMT_STATUS_SUCCESS) return;

    uint32_t count = 0;
    if (syms_.rocdxg_smi_get_device_count(&count) != HSAKMT_STATUS_SUCCESS) return;
    if (count > 0) gpu_ok_ = true;
  }

  static void TearDownTestSuite() {
    if (kfd_opened_) {
      syms_.hsaKmtReleaseSystemProperties();
      syms_.hsaKmtCloseKFD();
    }
    if (handle_ != nullptr) dlclose(handle_);
  }

  // Requires only that librocdxg loaded, so ABI contract tests can run.
  static void RequireLib() {
    if (!wsl_present_) GTEST_SKIP() << "No /dev/dxg, not running under WSL2";
    if (!lib_ok_) GTEST_SKIP() << "librocdxg not loadable";
  }

  // Requires an actual WSL GPU behind the loaded library.
  static void RequireGpu() {
    RequireLib();
    if (!gpu_ok_) GTEST_SKIP() << "No WSL GPU reported by rocdxg_smi_get_device_count";
  }

  static bool wsl_present_;
  static bool lib_ok_;
  static bool kfd_opened_;
  static bool gpu_ok_;
  static void* handle_;
  static RocdxgSyms syms_;
};

bool WslFunctionalReadOnly::wsl_present_ = false;
bool WslFunctionalReadOnly::lib_ok_ = false;
bool WslFunctionalReadOnly::kfd_opened_ = false;
bool WslFunctionalReadOnly::gpu_ok_ = false;
void* WslFunctionalReadOnly::handle_ = nullptr;
RocdxgSyms WslFunctionalReadOnly::syms_;

}  // namespace

// ============================================================================
// ABI contract: null output pointers are rejected
// ============================================================================

TEST_F(WslFunctionalReadOnly, NullDeviceCountReturnsInval) {
  RequireLib();
  EXPECT_EQ(syms_.rocdxg_smi_get_device_count(nullptr), HSAKMT_STATUS_INVALID_PARAMETER);
}

TEST_F(WslFunctionalReadOnly, NullDeviceInfoReturnsInval) {
  RequireLib();
  EXPECT_EQ(syms_.rocdxg_smi_get_device_info(0, nullptr), HSAKMT_STATUS_INVALID_PARAMETER);
}

// A caller built against a different layout must be rejected, not written past.
TEST_F(WslFunctionalReadOnly, DeviceInfoRejectsWrongStructSize) {
  RequireLib();
  rocdxg_smi_device_info_t info{};
  info.struct_size = sizeof(info) - 1;
  EXPECT_EQ(syms_.rocdxg_smi_get_device_info(0, &info), HSAKMT_STATUS_BUFFER_TOO_SMALL);

  info.struct_size = 0;
  EXPECT_EQ(syms_.rocdxg_smi_get_device_info(0, &info), HSAKMT_STATUS_BUFFER_TOO_SMALL);
}

TEST_F(WslFunctionalReadOnly, NullVramUsageReturnsInval) {
  RequireLib();
  EXPECT_EQ(syms_.rocdxg_smi_get_vram_usage(0, nullptr), HSAKMT_STATUS_INVALID_PARAMETER);
}

TEST_F(WslFunctionalReadOnly, NullPowerInfoReturnsInval) {
  RequireLib();
  EXPECT_EQ(syms_.rocdxg_smi_get_power_info(0, nullptr), HSAKMT_STATUS_INVALID_PARAMETER);
}

TEST_F(WslFunctionalReadOnly, NullTemperatureReturnsInval) {
  RequireLib();
  EXPECT_EQ(syms_.rocdxg_smi_get_temperature(0, 0, 0, nullptr), HSAKMT_STATUS_INVALID_PARAMETER);
}

TEST_F(WslFunctionalReadOnly, NullClockInfoReturnsInval) {
  RequireLib();
  EXPECT_EQ(syms_.rocdxg_smi_get_clock_info(0, 0, nullptr), HSAKMT_STATUS_INVALID_PARAMETER);
}

TEST_F(WslFunctionalReadOnly, NullPcieInfoReturnsInval) {
  RequireLib();
  EXPECT_EQ(syms_.rocdxg_smi_get_pcie_info(0, nullptr), HSAKMT_STATUS_INVALID_PARAMETER);
}

TEST_F(WslFunctionalReadOnly, NullGpuMetricsInfoReturnsInval) {
  RequireLib();
  EXPECT_EQ(syms_.rocdxg_smi_get_gpu_metrics_info(0, nullptr), HSAKMT_STATUS_INVALID_PARAMETER);
}

TEST_F(WslFunctionalReadOnly, NullProcessCountReturnsInval) {
  RequireLib();
  EXPECT_EQ(syms_.rocdxg_smi_enum_processes(0, nullptr, nullptr), HSAKMT_STATUS_INVALID_PARAMETER);
}

// ============================================================================
// Live queries against a real WSL GPU
// ============================================================================

TEST_F(WslFunctionalReadOnly, LiveDeviceCountNonZero) {
  RequireGpu();
  uint32_t count = 0;
  ASSERT_EQ(syms_.rocdxg_smi_get_device_count(&count), HSAKMT_STATUS_SUCCESS);
  EXPECT_GT(count, 0u);
}

TEST_F(WslFunctionalReadOnly, LiveDeviceInfoPopulated) {
  RequireGpu();
  rocdxg_smi_device_info_t info{};
  info.struct_size = sizeof(info);
  ASSERT_EQ(syms_.rocdxg_smi_get_device_info(0, &info), HSAKMT_STATUS_SUCCESS);
  EXPECT_EQ(info.struct_size, sizeof(info)) << "library should report what it filled";
  EXPECT_EQ(info.asic.vendor_id, 0x1002u) << "expected an AMD vendor id";
  EXPECT_GT(info.vram.vram_size_mb, 0u);
  EXPECT_NE(info.board.product_name[0], '\0') << "product name should not be empty";
}

TEST_F(WslFunctionalReadOnly, LiveDeviceInfoIsStableAcrossCalls) {
  RequireGpu();
  rocdxg_smi_device_info_t a{};
  rocdxg_smi_device_info_t b{};
  a.struct_size = sizeof(a);
  b.struct_size = sizeof(b);
  ASSERT_EQ(syms_.rocdxg_smi_get_device_info(0, &a), HSAKMT_STATUS_SUCCESS);
  ASSERT_EQ(syms_.rocdxg_smi_get_device_info(0, &b), HSAKMT_STATUS_SUCCESS);
  EXPECT_EQ(a.asic.device_id, b.asic.device_id);
  EXPECT_EQ(a.bdf.bus_number, b.bdf.bus_number);
}

TEST_F(WslFunctionalReadOnly, LiveDeviceInfoRejectsOutOfRangeNode) {
  RequireGpu();
  uint32_t count = 0;
  ASSERT_EQ(syms_.rocdxg_smi_get_device_count(&count), HSAKMT_STATUS_SUCCESS);
  rocdxg_smi_device_info_t info{};
  info.struct_size = sizeof(info);
  EXPECT_EQ(syms_.rocdxg_smi_get_device_info(count, &info), HSAKMT_STATUS_INVALID_NODE_UNIT);
}

TEST_F(WslFunctionalReadOnly, LiveVramUsageWithinTotal) {
  RequireGpu();
  rocdxg_smi_vram_usage_t usage{};
  HSAKMT_STATUS r = syms_.rocdxg_smi_get_vram_usage(0, &usage);
  if (r == HSAKMT_STATUS_NOT_SUPPORTED) GTEST_SKIP() << "vram usage unsupported";
  ASSERT_EQ(r, HSAKMT_STATUS_SUCCESS);
  EXPECT_LE(usage.vram_used_mb, usage.vram_total_mb);
}

TEST_F(WslFunctionalReadOnly, LivePowerInfoSuccessOrNotSupported) {
  RequireGpu();
  rocdxg_smi_power_info_t power{};
  HSAKMT_STATUS r = syms_.rocdxg_smi_get_power_info(0, &power);
  EXPECT_TRUE(r == HSAKMT_STATUS_SUCCESS || r == HSAKMT_STATUS_NOT_SUPPORTED)
      << "unexpected status: " << r;
}

TEST_F(WslFunctionalReadOnly, LiveTemperatureSuccessOrNotSupported) {
  RequireGpu();
  int64_t temp = 0;
  HSAKMT_STATUS r = syms_.rocdxg_smi_get_temperature(0, 0, 0, &temp);
  EXPECT_TRUE(r == HSAKMT_STATUS_SUCCESS || r == HSAKMT_STATUS_NOT_SUPPORTED)
      << "unexpected status: " << r;
}

TEST_F(WslFunctionalReadOnly, LiveClockInfoSuccessOrNotSupported) {
  RequireGpu();
  rocdxg_smi_clock_info_t clk{};
  HSAKMT_STATUS r = syms_.rocdxg_smi_get_clock_info(0, 0, &clk);
  if (r == HSAKMT_STATUS_NOT_SUPPORTED) GTEST_SKIP() << "clock info unsupported";
  ASSERT_EQ(r, HSAKMT_STATUS_SUCCESS);
  if (clk.clk != UINT32_MAX && clk.max_clk != UINT32_MAX) {
    EXPECT_LE(clk.clk, clk.max_clk);
  }
}

TEST_F(WslFunctionalReadOnly, LivePcieInfoSuccessOrNotSupported) {
  RequireGpu();
  rocdxg_smi_pcie_info_t pcie{};
  HSAKMT_STATUS r = syms_.rocdxg_smi_get_pcie_info(0, &pcie);
  EXPECT_TRUE(r == HSAKMT_STATUS_SUCCESS || r == HSAKMT_STATUS_NOT_SUPPORTED)
      << "unexpected status: " << r;
}

TEST_F(WslFunctionalReadOnly, LiveGpuMetricsInfoSuccessOrNotSupported) {
  RequireGpu();
  rocdxg_smi_gpu_metrics_info_t metrics{};
  HSAKMT_STATUS r = syms_.rocdxg_smi_get_gpu_metrics_info(0, &metrics);
  if (r == HSAKMT_STATUS_NOT_SUPPORTED) GTEST_SKIP() << "gpu metrics unsupported";
  ASSERT_EQ(r, HSAKMT_STATUS_SUCCESS);

  // Individual fields report UINT32_MAX when a sensor is unavailable, so no
  // per-field bound can be required. SUCCESS with every field unset would mean
  // nothing was read and must not pass silently.
  const uint32_t unset = UINT32_MAX;
  EXPECT_TRUE(metrics.average_gfx_activity != unset || metrics.average_umc_activity != unset ||
              metrics.current_socket_power != unset || metrics.current_fan_speed_percent != unset)
      << "gpu metrics returned SUCCESS with every sensor unset";

  if (metrics.average_gfx_activity != unset) {
    EXPECT_LE(metrics.average_gfx_activity, 100u);
  }
  if (metrics.average_umc_activity != unset) {
    EXPECT_LE(metrics.average_umc_activity, 100u);
  }
  if (metrics.current_fan_speed_percent != unset) {
    EXPECT_LE(metrics.current_fan_speed_percent, 100u);
  }
}

TEST_F(WslFunctionalReadOnly, LiveProcessEnumSizingProbe) {
  RequireGpu();
  uint32_t count = 0;
  HSAKMT_STATUS r = syms_.rocdxg_smi_enum_processes(0, &count, nullptr);
  EXPECT_TRUE(r == HSAKMT_STATUS_SUCCESS || r == HSAKMT_STATUS_NOT_SUPPORTED)
      << "unexpected status: " << r;
}

#endif  // ENABLE_WSL_BACKEND
