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

// Unit tests for the WSL backend seam. The whole file is a no-op unless the
// backend was compiled in (ENABLE_WSL_BACKEND=ON), so it links cleanly in native
// builds and exercises the mock backend + activation logic when enabled.

#include "config/amd_smi_config.h"

#if defined(AMDSMI_ENABLE_WSL_BACKEND)

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_wsl_backend.h"

using amd::smi::AMDSmiWslBackend;

namespace {

// RAII guard so a test's AMDSMI_WSL_MODE change never leaks into other tests.
// Safe with compute_active() (uncached); do not use to influence active(), whose
// result is cached in a process-lifetime static on first call.
class ScopedWslModeEnv {
 public:
  explicit ScopedWslModeEnv(const char* value) {
    const char* prev = std::getenv("AMDSMI_WSL_MODE");
    had_prev_ = prev != nullptr;
    if (had_prev_) {
      prev_ = prev;
    }
    if (value == nullptr) {
      ::unsetenv("AMDSMI_WSL_MODE");
    } else {
      ::setenv("AMDSMI_WSL_MODE", value, 1);
    }
  }
  ~ScopedWslModeEnv() {
    if (had_prev_) {
      ::setenv("AMDSMI_WSL_MODE", prev_.c_str(), 1);
    } else {
      ::unsetenv("AMDSMI_WSL_MODE");
    }
  }

 private:
  bool had_prev_ = false;
  std::string prev_;
};

}  // namespace

// compute_active() is the uncached decision; it reflects the env var on each
// call, which is why it (not active()) is used for these assertions.
TEST(WslBackendActivation, OffWhenUnset) {
  ScopedWslModeEnv env(nullptr);
  EXPECT_FALSE(AMDSmiWslBackend::compute_active());
}

TEST(WslBackendActivation, OnWhenExplicitlyEnabled) {
  ScopedWslModeEnv env("1");
  EXPECT_TRUE(AMDSmiWslBackend::compute_active());
}

TEST(WslBackendActivation, OffForNonOneValues) {
  for (const char* v : {"0", "true", "yes", ""}) {
    ScopedWslModeEnv env(v);
    EXPECT_FALSE(AMDSmiWslBackend::compute_active()) << "value=" << v;
  }
}

TEST(WslBackendMock, AsicInfoReturnsSyntheticData) {
  amdsmi_asic_info_t info{};
  EXPECT_EQ(AMDSmiWslBackend::instance().get_gpu_asic_info(nullptr, &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.vendor_id, 0x1002u);
  EXPECT_NE(std::strstr(info.market_name, "mock"), nullptr);
}

TEST(WslBackendMock, PowerInfoReturnsSyntheticData) {
  amdsmi_power_info_t info{};
  EXPECT_EQ(AMDSmiWslBackend::instance().get_power_info(nullptr, &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.current_socket_power, 42u);
}

TEST(WslBackendMock, BoardInfoReturnsSyntheticData) {
  amdsmi_board_info_t info{};
  EXPECT_EQ(AMDSmiWslBackend::instance().get_gpu_board_info(nullptr, &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_NE(std::strstr(info.product_name, "mock"), nullptr);
}

TEST(WslBackendMock, TempCurrentSupportedOthersNot) {
  int64_t temp = -1;
  EXPECT_EQ(AMDSmiWslBackend::instance().get_temp_metric(nullptr, AMDSMI_TEMPERATURE_TYPE_EDGE,
                                                         AMDSMI_TEMP_CURRENT, &temp),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(temp, 47);
  EXPECT_EQ(AMDSmiWslBackend::instance().get_temp_metric(nullptr, AMDSMI_TEMPERATURE_TYPE_EDGE,
                                                         AMDSMI_TEMP_MAX, &temp),
            AMDSMI_STATUS_NOT_SUPPORTED);
}

TEST(WslBackendMock, NullOutputPointerReturnsInval) {
  EXPECT_EQ(AMDSmiWslBackend::instance().get_power_info(nullptr, nullptr), AMDSMI_STATUS_INVAL);
  EXPECT_EQ(AMDSmiWslBackend::instance().get_gpu_asic_info(nullptr, nullptr), AMDSMI_STATUS_INVAL);
  EXPECT_EQ(AMDSmiWslBackend::instance().get_gpu_board_info(nullptr, nullptr), AMDSMI_STATUS_INVAL);
  EXPECT_EQ(AMDSmiWslBackend::instance().get_temp_metric(nullptr, AMDSMI_TEMPERATURE_TYPE_EDGE,
                                                         AMDSMI_TEMP_CURRENT, nullptr),
            AMDSMI_STATUS_INVAL);
}

#endif  // AMDSMI_ENABLE_WSL_BACKEND
