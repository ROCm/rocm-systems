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

// Unit tests for WSL backend activation, detection, and the IGPUBackend
// default contract. No-op unless ENABLE_WSL_BACKEND=ON.
//
// These run on native Linux: the paths under test are the ones taken when
// /dev/dxg is absent.

#include "config/amd_smi_config.h"

#if defined(ENABLE_WSL_BACKEND)

#include <gtest/gtest.h>
#include <unistd.h>

#include <set>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_gpu_backend.h"
#include "amd_smi/impl/amd_smi_processor.h"
#include "amd_smi/impl/amd_smi_socket.h"
#include "amd_smi/impl/amd_smi_wsl_device.h"

using amd::smi::AMDSmiProcessor;
using amd::smi::AMDSmiSocket;
using amd::smi::IGPUBackend;
using amd::smi::WSLGPUBackend;

namespace {

bool on_wsl() { return access("/dev/dxg", F_OK) == 0; }

// Exercises IGPUBackend's defaults without overriding anything.
class UnimplementedBackend : public IGPUBackend {};

}  // namespace

// Every IGPUBackend method defaults to NOT_SUPPORTED so a partially
// implemented backend degrades instead of returning uninitialized data.
TEST(WslUnit, UnimplementedBackendReturnsNotSupported) {
  UnimplementedBackend backend;

  amdsmi_asic_info_t asic{};
  EXPECT_EQ(backend.get_asic_info(&asic), AMDSMI_STATUS_NOT_SUPPORTED);

  amdsmi_board_info_t board{};
  EXPECT_EQ(backend.get_board_info(&board), AMDSMI_STATUS_NOT_SUPPORTED);

  amdsmi_vram_info_t vram{};
  EXPECT_EQ(backend.get_vram_info(&vram), AMDSMI_STATUS_NOT_SUPPORTED);

  uint64_t total = 0;
  EXPECT_EQ(backend.get_memory_total(AMDSMI_MEM_TYPE_VRAM, &total), AMDSMI_STATUS_NOT_SUPPORTED);

  amdsmi_power_info_t power{};
  EXPECT_EQ(backend.get_power_info(&power), AMDSMI_STATUS_NOT_SUPPORTED);

  int64_t speed = 0;
  EXPECT_EQ(backend.get_fan_speed(0, &speed), AMDSMI_STATUS_NOT_SUPPORTED);
}

// is_active() is false before any try_populate() call.
TEST(WslUnit, InactiveByDefault) { EXPECT_FALSE(WSLGPUBackend::is_active()); }

// try_populate() without /dev/dxg reports NOT_SUPPORTED and touches nothing,
// which is what lets amdsmi_init() fall through to the native Linux path.
TEST(WslUnit, TryPopulateWithoutDxg) {
  if (on_wsl()) GTEST_SKIP() << "/dev/dxg present, skipped on WSL machines";

  std::vector<AMDSmiSocket*> sockets;
  std::set<AMDSmiProcessor*> processors;
  EXPECT_EQ(WSLGPUBackend::try_populate(sockets, processors), AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_TRUE(sockets.empty());
  EXPECT_TRUE(processors.empty());
  EXPECT_FALSE(WSLGPUBackend::is_active());
}

// The detection path has no first-call side effect, so a retry after a failed
// init behaves identically.
TEST(WslUnit, TryPopulateWithoutDxgIsRepeatable) {
  if (on_wsl()) GTEST_SKIP() << "/dev/dxg present, skipped on WSL machines";

  std::vector<AMDSmiSocket*> sockets;
  std::set<AMDSmiProcessor*> processors;
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(WSLGPUBackend::try_populate(sockets, processors), AMDSMI_STATUS_NOT_SUPPORTED)
        << "attempt " << i;
  }
  EXPECT_TRUE(sockets.empty());
  EXPECT_FALSE(WSLGPUBackend::is_active());
}

// shutdown() is called unconditionally from AMDSmiSystem::cleanup(), including
// when try_populate() never ran.
TEST(WslUnit, ShutdownWithoutPopulateIsSafe) {
  if (on_wsl()) GTEST_SKIP() << "/dev/dxg present, skipped on WSL machines";

  EXPECT_EQ(WSLGPUBackend::shutdown(), AMDSMI_STATUS_SUCCESS);
  EXPECT_FALSE(WSLGPUBackend::is_active());
  EXPECT_EQ(WSLGPUBackend::shutdown(), AMDSMI_STATUS_SUCCESS);
}

// On WSL, is_active() must agree with what try_populate() reported.
TEST(WslUnit, TryPopulateOnWslAgreesWithActiveState) {
  if (!on_wsl()) GTEST_SKIP() << "No /dev/dxg, not running under WSL2";

  std::vector<AMDSmiSocket*> sockets;
  std::set<AMDSmiProcessor*> processors;
  amdsmi_status_t r = WSLGPUBackend::try_populate(sockets, processors);

  if (r == AMDSMI_STATUS_SUCCESS) {
    EXPECT_TRUE(WSLGPUBackend::is_active());
    EXPECT_FALSE(sockets.empty());
    EXPECT_FALSE(processors.empty());
    EXPECT_EQ(WSLGPUBackend::shutdown(), AMDSMI_STATUS_SUCCESS);
    EXPECT_FALSE(WSLGPUBackend::is_active());
  } else {
    // DRIVER_NOT_LOADED (no librocdxg) or NOT_FOUND (no GPU) leave it inactive.
    EXPECT_TRUE(r == AMDSMI_STATUS_DRIVER_NOT_LOADED || r == AMDSMI_STATUS_NOT_FOUND)
        << "unexpected status: " << r;
    EXPECT_FALSE(WSLGPUBackend::is_active());
  }
}

#endif  // ENABLE_WSL_BACKEND
