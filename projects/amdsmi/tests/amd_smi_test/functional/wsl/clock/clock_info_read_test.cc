// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// amdsmi_get_clock_info() through the WSL backend: fields rocdxg cannot report
// must surface as N/A sentinels, not zero. Skipped unless the WSL backend is active.

#include "config/amd_smi_config.h"

#if defined(ENABLE_WSL_BACKEND)

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_wsl_device.h"
#include "functional/wsl/wsl_fixture.h"

namespace {

std::vector<amdsmi_processor_handle> GpuHandles() {
  std::vector<amdsmi_processor_handle> handles;
  uint32_t socket_count = 0;
  if (amdsmi_get_socket_handles(&socket_count, nullptr) != AMDSMI_STATUS_SUCCESS) return handles;
  std::vector<amdsmi_socket_handle> sockets(socket_count);
  if (amdsmi_get_socket_handles(&socket_count, sockets.data()) != AMDSMI_STATUS_SUCCESS) {
    return handles;
  }
  for (auto socket : sockets) {
    uint32_t count = 0;
    if (amdsmi_get_processor_handles(socket, &count, nullptr) != AMDSMI_STATUS_SUCCESS) continue;
    std::vector<amdsmi_processor_handle> procs(count);
    if (amdsmi_get_processor_handles(socket, &count, procs.data()) != AMDSMI_STATUS_SUCCESS) {
      continue;
    }
    handles.insert(handles.end(), procs.begin(), procs.begin() + count);
  }
  return handles;
}

}  // namespace

TEST_F(WslFunctionalReadOnly, ClockInfoUnreportedFieldsAreNotAvailable) {
  if (amdsmi_init(AMDSMI_INIT_AMD_GPUS) != AMDSMI_STATUS_SUCCESS) {
    GTEST_SKIP() << "amdsmi_init failed";
  }
  if (!amd::smi::WSLGPUBackend::IsActive()) {
    amdsmi_shut_down();
    GTEST_SKIP() << "WSL backend not active";
  }

  auto handles = GpuHandles();
  EXPECT_FALSE(handles.empty());

  for (auto handle : handles) {
    for (auto clk_type : {AMDSMI_CLK_TYPE_GFX, AMDSMI_CLK_TYPE_MEM}) {
      amdsmi_clk_info_t info{};
      amdsmi_status_t r = amdsmi_get_clock_info(handle, clk_type, &info);
      EXPECT_TRUE(r == AMDSMI_STATUS_SUCCESS || r == AMDSMI_STATUS_NOT_SUPPORTED)
          << "clk_type " << clk_type << " unexpected status: " << r;
      if (r != AMDSMI_STATUS_SUCCESS) continue;
      EXPECT_GT(info.max_clk, 0u);
      EXPECT_EQ(info.min_clk, UINT32_MAX);
      EXPECT_EQ(info.clk_locked, UINT8_MAX);
      EXPECT_EQ(info.clk_deep_sleep, UINT8_MAX);
    }
  }
  amdsmi_shut_down();
}

#endif  // ENABLE_WSL_BACKEND
