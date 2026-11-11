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
#include <unistd.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_device.h"
#include "rocm_smi/rocm_smi_main.h"
#include "rocm_smi/rocm_smi_monitor.h"

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

void WriteFile(const std::filesystem::path& path, const std::string& contents) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path);
  ASSERT_TRUE(out.is_open()) << "cannot write " << path;
  out << contents;
}

// Builds a fake OverDrive-enabled GPU under a temp dir, registers it with the
// RocmSMI device list and removes it again on destruction.
class FakeOverDriveDevice {
 public:
  FakeOverDriveDevice() {
    root_ = std::filesystem::temp_directory_path() / ("amdsmi_fan_od_" + std::to_string(getpid()));
    std::filesystem::remove_all(root_);

    const std::filesystem::path card = root_ / "card0";
    const std::filesystem::path hwmon = card / "device" / "hwmon" / "hwmon0";

    WriteFile(card / "device" / "gpu_od" / "fan_ctrl" / "fan_minimum_pwm", kFanMinimumPwm);
    WriteFile(hwmon / "pwm1", std::to_string(kLivePwm) + "\n");
    WriteFile(hwmon / "pwm1_max", std::to_string(kLivePwmMax) + "\n");

    auto dev = std::make_shared<amd::smi::Device>(card.string(), nullptr);
    dev->set_monitor(std::make_shared<amd::smi::Monitor>(hwmon.string(), nullptr));

    auto& devices = amd::smi::RocmSMI::getInstance().devices();
    index_ = static_cast<uint32_t>(devices.size());
    devices.push_back(dev);
  }

  ~FakeOverDriveDevice() {
    auto& devices = amd::smi::RocmSMI::getInstance().devices();
    if (index_ < devices.size()) {
      devices.erase(devices.begin() + index_);
    }
    std::filesystem::remove_all(root_);
  }

  FakeOverDriveDevice(const FakeOverDriveDevice&) = delete;
  FakeOverDriveDevice& operator=(const FakeOverDriveDevice&) = delete;

  uint32_t index() const { return index_; }

 private:
  std::filesystem::path root_;
  uint32_t index_ = 0;
};

}  // namespace

// The reported speed is the running pwm1, not the fan-curve floor (0).
TEST(GpuUnit, FanSpeedComesFromHwmonNotOverDriveFloor) {
  FakeOverDriveDevice fake;

  int64_t speed = -1;
  ASSERT_EQ(rsmi_dev_fan_speed_get(fake.index(), 0, &speed), RSMI_STATUS_SUCCESS);
  EXPECT_NE(speed, 0) << "fan_minimum_pwm floor reported as live speed (issue #8684)";
  EXPECT_EQ(speed, kLivePwm);
}

// The reported maximum is pwm1_max, not the OD_RANGE maximum (100). Pairing a
// pwm1 speed with an OD_RANGE max would misscale the usage percentage
// amd-smi prints.
TEST(GpuUnit, FanSpeedMaxComesFromHwmonNotOverDriveRange) {
  FakeOverDriveDevice fake;

  uint64_t max_speed = 0;
  ASSERT_EQ(rsmi_dev_fan_speed_max_get(fake.index(), 0, &max_speed), RSMI_STATUS_SUCCESS);
  EXPECT_EQ(max_speed, kLivePwmMax);
}
