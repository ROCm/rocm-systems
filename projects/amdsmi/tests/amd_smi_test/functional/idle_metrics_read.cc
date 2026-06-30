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

#include "idle_metrics_read.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include "../test_common.h"
#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_utils.h"

namespace {
constexpr const char* kIdleEnv = "AMDSMI_SKIP_GPU_METRICS_ON_IDLE";

void WriteFile(const std::string& path, const std::string& contents) {
  std::ofstream f(path, std::ios::trunc);
  f << contents;
}
}  // namespace

TestIdleMetricsRead::TestIdleMetricsRead() : TestBase() {
  set_title("AMDSMI Idle Metrics Gating Test");
  set_description(
      "Verifies AMDSMI_SKIP_GPU_METRICS_ON_IDLE: gpu_metrics reads are skipped "
      "(returning AMDSMI_STATUS_BUSY) on a runtime-suspended GPU, and are left "
      "unchanged on non-suspended GPUs such as Instinct (no runtime PM).");
}

TestIdleMetricsRead::~TestIdleMetricsRead(void) {}

void TestIdleMetricsRead::SetUp(void) {
  TestBase::SetUp();
  return;
}

void TestIdleMetricsRead::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestIdleMetricsRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestIdleMetricsRead::Close() { TestBase::Close(); }

void TestIdleMetricsRead::Run(void) {
  TestBase::Run();
  PRINT_VERBOSITY();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  // Save/restore the env var so the test never leaks process-wide state.
  const char* orig = std::getenv(kIdleEnv);
  const bool had_orig = (orig != nullptr);
  const std::string orig_val = had_orig ? std::string(orig) : std::string();

  // ---- Part A: hardware-free helper coverage (runs on every ASIC) ----

  // skip_gpu_metrics_on_idle_enabled() env parsing.
  unsetenv(kIdleEnv);
  EXPECT_FALSE(skip_gpu_metrics_on_idle_enabled()) << "unset must be disabled";
  for (const char* on : {"1", "true", "TRUE", "On", "yes", "YES"}) {
    setenv(kIdleEnv, on, 1);
    EXPECT_TRUE(skip_gpu_metrics_on_idle_enabled()) << "value=" << on;
  }
  for (const char* off : {"0", "false", "no", "", "garbage"}) {
    setenv(kIdleEnv, off, 1);
    EXPECT_FALSE(skip_gpu_metrics_on_idle_enabled()) << "value=" << off;
  }

  // is_gpu_runtime_suspended() parsing of runtime_status contents. Only an
  // exact "suspended" gates; "active"/"unsupported" (Instinct / no runtime PM),
  // the "suspending" transient, empty, and a missing file all return false.
  const std::string dir = ::testing::TempDir();
  const std::string suspended = dir + "amdsmi_idle_suspended";
  const std::string suspended_nonl = dir + "amdsmi_idle_suspended_nonl";
  const std::string active = dir + "amdsmi_idle_active";
  const std::string suspending = dir + "amdsmi_idle_suspending";
  const std::string unsupported = dir + "amdsmi_idle_unsupported";
  const std::string empty = dir + "amdsmi_idle_empty";

  WriteFile(suspended, "suspended\n");
  WriteFile(suspended_nonl, "suspended");
  WriteFile(active, "active\n");
  WriteFile(suspending, "suspending\n");
  WriteFile(unsupported, "unsupported\n");
  WriteFile(empty, "");

  EXPECT_TRUE(is_gpu_runtime_suspended(suspended));
  EXPECT_TRUE(is_gpu_runtime_suspended(suspended_nonl));
  EXPECT_FALSE(is_gpu_runtime_suspended(active));
  EXPECT_FALSE(is_gpu_runtime_suspended(suspending));
  EXPECT_FALSE(is_gpu_runtime_suspended(unsupported));
  EXPECT_FALSE(is_gpu_runtime_suspended(empty));
  EXPECT_FALSE(is_gpu_runtime_suspended(dir + "amdsmi_idle_missing"));

  for (const std::string& f : {suspended, suspended_nonl, active, suspending, unsupported, empty}) {
    std::remove(f.c_str());
  }

  // ---- Part B: per-device behavior (gating + Instinct/awake no-op guard) ----
  for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
    PrintDeviceHeader(processor_handles_[i]);

    // Baseline with the feature OFF.
    unsetenv(kIdleEnv);
    amdsmi_gpu_metrics_t base_metrics = {};
    amdsmi_status_t base = amdsmi_get_gpu_metrics_info(processor_handles_[i], &base_metrics);
    // EXPECT (not ASSERT) so the env-var restore at the end of Run() still runs.
    EXPECT_TRUE(base == AMDSMI_STATUS_SUCCESS || base == AMDSMI_STATUS_NOT_SUPPORTED ||
                base == AMDSMI_STATUS_BUSY)
        << "unexpected baseline status: " << base;

    // Determine suspend state via the non-waking PM node.
    bool dev_suspended = false;
    amd::smi::AMDSmiGPUDevice* gpu_device = nullptr;
    if (get_gpu_device_from_handle(processor_handles_[i], &gpu_device) == AMDSMI_STATUS_SUCCESS) {
      const std::string rt_path =
          "/sys/class/drm/" + gpu_device->get_gpu_path() + "/device/power/runtime_status";
      dev_suspended = is_gpu_runtime_suspended(rt_path);
    }

    // Enable the feature and re-query.
    setenv(kIdleEnv, "1", 1);
    amdsmi_gpu_metrics_t gated_metrics = {};
    amdsmi_status_t gated = amdsmi_get_gpu_metrics_info(processor_handles_[i], &gated_metrics);

    if (dev_suspended) {
      // Gate should short-circuit to BUSY. The device can leave "suspended"
      // between our check and the gate's own check (inherent TOCTOU), so accept
      // either BUSY or the same status the ungated read produced.
      EXPECT_TRUE(gated == AMDSMI_STATUS_BUSY || gated == base)
          << "expected BUSY or the ungated result on a runtime-suspended GPU";
    } else {
      // Instinct (no runtime PM) and awake GPUs must be unaffected: enabling the
      // feature must not change the outcome versus the baseline.
      EXPECT_EQ(gated, base) << "idle gating changed the result on a non-suspended GPU";
    }

    IF_VERB(STANDARD) {
      std::cout << "\truntime_suspended: " << (dev_suspended ? "true" : "false")
                << " | baseline status: " << base << " | gated status: " << gated << std::endl;
    }
  }

  // Restore original env state.
  if (had_orig) {
    setenv(kIdleEnv, orig_val.c_str(), 1);
  } else {
    unsetenv(kIdleEnv);
  }
}
