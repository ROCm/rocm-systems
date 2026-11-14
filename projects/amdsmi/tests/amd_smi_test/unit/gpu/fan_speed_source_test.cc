// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Regression tests for which sysfs node backs the fan-speed getters.
// ROCm/rocm-systems#8684: on an OverDrive-enabled RDNA3/RDNA4 GPU
// (amdgpu.ppfeaturemask=0xffffffff) the getters read
// gpu_od/fan_ctrl/fan_minimum_pwm -- the user-set fan-curve floor and its
// OD_RANGE -- so amd-smi reported "SPEED: 0 / USAGE: 0.0 %" while the fan was
// spinning. The live duty cycle is hwmon pwm1 / pwm1_max.
//
// The fixture below is a fake sysfs tree, so this runs on any host: no
// OverDrive hardware, no gpu_od node and no GPU at all are required. A
// hardware test cannot cover this, since the broken branch is only reachable
// on a GPU that exposes gpu_od.

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <string>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_drm.h"
#include "amd_smi/impl/amd_smi_gpu_device.h"
#include "rocm_smi/rocm_smi_device.h"
#include "rocm_smi/rocm_smi_main.h"
#include "rocm_smi/rocm_smi_monitor.h"
#include "test_fixture_utils.h"

namespace {

// Values read from a live Navi21 board: pwm1=51 of pwm1_max=255 (20%) with the
// fan turning at ~1157 RPM.
constexpr int64_t kLivePwm = 51;
constexpr uint64_t kLivePwmMax = 255;

// A stock OverDrive fan curve: floor 0, OD_RANGE 15-100. Reading this node as
// the live duty cycle is what produced 0 / 0.0 %.
constexpr char kFanMinimumPwm[] =
    "FAN_MINIMUM_PWM:\n"
    "0\n"
    "OD_RANGE:\n"
    "MINIMUM_PWM: 15 100\n";

// A fake OverDrive-enabled GPU: the gpu_od fan node plus the hwmon pwm files, registered with
// RocmSMI and AMDSmiSystem for the life of the object so the public API can reach it.
class FakeOverDriveDevice {
 public:
  FakeOverDriveDevice()
      : device_(tree_.root() + "/card0"),
        gpu_device_(device_.index(), "card0", amdsmi_bdf_t{}, drm_),
        registration_(&gpu_device_) {
    tree_.WriteFile("card0/device/gpu_od/fan_ctrl/fan_minimum_pwm", kFanMinimumPwm);
    tree_.WriteFile("card0/device/hwmon/hwmon0/pwm1", std::to_string(kLivePwm) + "\n");
    tree_.WriteFile("card0/device/hwmon/hwmon0/pwm1_max", std::to_string(kLivePwmMax) + "\n");

    const std::string hwmon = tree_.root() + "/card0/device/hwmon/hwmon0";
    amd::smi::RocmSMI::getInstance().devices()[device_.index()]->set_monitor(
        std::make_shared<amd::smi::Monitor>(hwmon, nullptr));
  }

  amdsmi_processor_handle handle() const { return registration_.handle(); }

 private:
  ScopedAmdsmiInit init_;
  FakeSysfsTree tree_;
  ScopedRocmSmiDevice device_;
  amd::smi::AMDSmiDrm drm_;
  amd::smi::AMDSmiGPUDevice gpu_device_;
  amd::smi::testing::ScopedProcessorRegistration registration_;
};

}  // namespace

// The reported speed is the running pwm1, not the fan-curve floor (0).
TEST(GpuUnit, FanSpeedComesFromHwmonNotOverDriveFloor) {
  FakeOverDriveDevice fake;

  int64_t speed = -1;
  ASSERT_EQ(amdsmi_get_gpu_fan_speed(fake.handle(), 0, &speed), AMDSMI_STATUS_SUCCESS);
  EXPECT_NE(speed, 0) << "fan_minimum_pwm floor reported as live speed (issue #8684)";
  EXPECT_EQ(speed, kLivePwm);
}

// The reported maximum is pwm1_max, not the OD_RANGE maximum (100). Pairing a
// pwm1 speed with an OD_RANGE max would misscale the usage percentage
// amd-smi prints.
TEST(GpuUnit, FanSpeedMaxComesFromHwmonNotOverDriveRange) {
  FakeOverDriveDevice fake;

  uint64_t max_speed = 0;
  ASSERT_EQ(amdsmi_get_gpu_fan_speed_max(fake.handle(), 0, &max_speed), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(max_speed, kLivePwmMax);
}
