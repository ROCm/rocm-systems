// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Unit tests for WSLGPUBackend activation and detection logic.
// No-op unless the WSL backend was compiled in (ENABLE_WSL_BACKEND=ON).

#include "config/amd_smi_config.h"

#if defined(ENABLE_WSL_BACKEND)

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <set>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_processor.h"
#include "amd_smi/impl/amd_smi_socket.h"
#include "amd_smi/impl/amd_smi_wsl_device.h"
#include "amd_smi/impl/amd_smi_wsl_syms.h"
#include "hsakmt/hsakmt.h"

namespace amd::smi::testing {

// Builds a backend for a synthetic node; the constructor is private because
// production instances come only from TryPopulate(), which needs /dev/dxg.
class WSLGPUBackendTestPeer {
 public:
  static std::unique_ptr<WSLGPUBackend> Make() {
    HsaNodeProperties props = {};
    return std::unique_ptr<WSLGPUBackend>(new WSLGPUBackend(0, 0, props));
  }
};

}  // namespace amd::smi::testing

using amd::smi::AMDSmiProcessor;
using amd::smi::AMDSmiSocket;
using amd::smi::WSLGPUBackend;
using amd::smi::testing::WSLGPUBackendTestPeer;

namespace {

// Power limit librocdxg reports, in W, for the next fake_get_power_info() call.
uint32_t g_fake_power_limit_w = 0;

HSAKMT_STATUS fake_get_power_info(uint32_t, rocdxg_smi_power_info_t* info) {
  *info = {};
  info->current_socket_power = 25;
  info->power_limit = g_fake_power_limit_w;
  return HSAKMT_STATUS_SUCCESS;
}

// Swaps the librocdxg power entry point for fake_get_power_info() and restores
// the original on scope exit.
class ScopedFakePowerInfo {
 public:
  explicit ScopedFakePowerInfo(uint32_t power_limit_w)
      : saved_(amd::smi::g_wsl_syms.rocdxg_smi_get_power_info) {
    g_fake_power_limit_w = power_limit_w;
    amd::smi::g_wsl_syms.rocdxg_smi_get_power_info = fake_get_power_info;
  }
  ~ScopedFakePowerInfo() { amd::smi::g_wsl_syms.rocdxg_smi_get_power_info = saved_; }

  ScopedFakePowerInfo(const ScopedFakePowerInfo&) = delete;
  ScopedFakePowerInfo& operator=(const ScopedFakePowerInfo&) = delete;

 private:
  decltype(amd::smi::WslSyms::rocdxg_smi_get_power_info) saved_;
};

}  // namespace

// IsActive() is false before any TryPopulate() call.
TEST(GpuUnit, InactiveByDefault) { EXPECT_FALSE(WSLGPUBackend::IsActive()); }

// TryPopulate() on a machine without /dev/dxg returns NOT_SUPPORTED.
// Skipped on real WSL machines where /dev/dxg is present.
TEST(GpuUnit, TryPopulateWithoutDxg) {
  if (access("/dev/dxg", F_OK) == 0) {
    GTEST_SKIP() << "/dev/dxg present — skipped on WSL machines";
  }
  std::vector<AMDSmiSocket*> sockets;
  std::set<AMDSmiProcessor*> processors;
  amdsmi_status_t r = WSLGPUBackend::TryPopulate(sockets, processors);
  EXPECT_EQ(r, AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_TRUE(sockets.empty());
  EXPECT_TRUE(processors.empty());
  EXPECT_FALSE(WSLGPUBackend::IsActive());
}

// librocdxg reports the power limit in W; the amdsmi power_limit and power_cap
// fields are in uW, so 330 W must come back as 330000000 uW.
TEST(GpuUnit, PowerLimitAndCapConvertedToMicroWatts) {
  ScopedFakePowerInfo fake(330);
  auto backend = WSLGPUBackendTestPeer::Make();

  amdsmi_power_info_t power = {};
  ASSERT_EQ(backend->GetPowerInfo(&power), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(power.power_limit, 330000000u) << "power_limit not converted from W to uW";
  EXPECT_EQ(power.socket_power, 25u) << "socket_power is in W and must not be scaled";

  amdsmi_power_cap_info_t cap = {};
  ASSERT_EQ(backend->GetPowerCapInfo(&cap), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(cap.power_cap, 330000000u) << "power_cap not converted from W to uW";
}

// An unknown limit (UINT32_MAX) or one too large to express in uW stays N/A.
TEST(GpuUnit, PowerLimitSentinelPreserved) {
  for (uint32_t limit_w : {std::numeric_limits<uint32_t>::max(), 4295u}) {
    ScopedFakePowerInfo fake(limit_w);
    auto backend = WSLGPUBackendTestPeer::Make();

    amdsmi_power_info_t power = {};
    ASSERT_EQ(backend->GetPowerInfo(&power), AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(power.power_limit, std::numeric_limits<uint32_t>::max())
        << "power_limit " << limit_w << " W should report the N/A sentinel";

    amdsmi_power_cap_info_t cap = {};
    ASSERT_EQ(backend->GetPowerCapInfo(&cap), AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(cap.power_cap, std::numeric_limits<uint64_t>::max())
        << "power_cap for " << limit_w << " W should report the N/A sentinel";
  }
}

#endif  // ENABLE_WSL_BACKEND
