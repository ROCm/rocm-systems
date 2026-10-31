// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "power_stabilization_read_write.h"

#include <fcntl.h>
#include <gtest/gtest.h>
#include <linux/fs.h>
#include <stdint.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <iostream>
#include <string>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_gpu_device.h"
#include "amd_smi/impl/amd_smi_utils.h"
#include "test_common.h"

namespace {

// Sets FS_IMMUTABLE_FL on a raw sysfs path so even a root-owned write is
// rejected with EPERM, letting us exercise amdsmi_set_power_stabilization_param's
// NO_PERM branch without relying on plain rwx bits (root bypasses those).
// RAII so the flag is always cleared, even if an ASSERT_* aborts the test early.
class ScopedImmutable {
 public:
  explicit ScopedImmutable(const std::string& path) : path_(path) {
    fd_ = open(path_.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) return;
    if (ioctl(fd_, FS_IOC_GETFLAGS, &orig_attr_) != 0) return;
    int attr = orig_attr_ | FS_IMMUTABLE_FL;
    applied_ = (ioctl(fd_, FS_IOC_SETFLAGS, &attr) == 0);
  }

  ScopedImmutable(const ScopedImmutable&) = delete;
  ScopedImmutable& operator=(const ScopedImmutable&) = delete;

  ~ScopedImmutable() {
    if (fd_ >= 0) {
      if (applied_) {
        ioctl(fd_, FS_IOC_SETFLAGS, &orig_attr_);
      }
      close(fd_);
    }
  }

  bool applied() const { return applied_; }

 private:
  std::string path_;
  int fd_ = -1;
  int orig_attr_ = 0;
  bool applied_ = false;
};

// Restores one Power Stabilization parameter to a caller-supplied value when
// the guard goes out of scope. gtest's ASSERT_*/CHK_ERR_ASRT macros exit
// Run() with a bare `return`, not a C++ exception, so this ordinary RAII
// destructor runs during that unwind the same as it would on any other
// early return -- no try/catch needed. Without this, a fatal assertion
// partway through a parameter's write -> read-back -> assert round trip
// left that parameter (and, worst case, Power Stabilization's own ENABLE
// flag) mutated on the device for the rest of the test run.
class ScopedPowerStabilizationRestore {
 public:
  ScopedPowerStabilizationRestore(amdsmi_processor_handle processor_handle,
                                  amdsmi_power_stabilization_param_id_t param_id,
                                  amdsmi_value_t restore_value)
      : processor_handle_(processor_handle), param_id_(param_id), restore_value_(restore_value) {}

  ~ScopedPowerStabilizationRestore() {
    amdsmi_set_power_stabilization_param(processor_handle_, param_id_, &restore_value_);
  }

  ScopedPowerStabilizationRestore(const ScopedPowerStabilizationRestore&) = delete;
  ScopedPowerStabilizationRestore& operator=(const ScopedPowerStabilizationRestore&) = delete;

 private:
  amdsmi_processor_handle processor_handle_;
  amdsmi_power_stabilization_param_id_t param_id_;
  amdsmi_value_t restore_value_;
};

}  // namespace

TestPowerStabilizationReadWrite::TestPowerStabilizationReadWrite() : TestBase() {
  set_title("AMDSMI Power Stabilization Read/Write Test");
  set_description(
      "The Power Stabilization tests verify that HAWK dummy-compute "
      "insertion and ramp-management settings can be read and written "
      "properly.");
}

TestPowerStabilizationReadWrite::~TestPowerStabilizationReadWrite(void) {}

void TestPowerStabilizationReadWrite::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestPowerStabilizationReadWrite::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestPowerStabilizationReadWrite::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestPowerStabilizationReadWrite::Close() {
  // This will close handles opened within amdsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

void TestPowerStabilizationReadWrite::Run(void) {
  amdsmi_status_t ret;
  amdsmi_power_stabilization_info_t info;

  TestBase::Run();
  PRINT_VERBOSITY();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    PrintDeviceHeader(processor_handles_[dv_ind]);

    DISPLAY_AMDSMI_API("amdsmi_get_power_stabilization_info", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    ret = amdsmi_get_power_stabilization_info(processor_handles_[dv_ind], &info);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);

    if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
      std::cout << "\tPower Stabilization not supported on this device. Skipping." << std::endl;
      continue;
    }
    CHK_ERR_ASRT(ret)

    // Null info pointer must be reported as INVAL, not crash.
    DISPLAY_AMDSMI_API("amdsmi_get_power_stabilization_info", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    ret = amdsmi_get_power_stabilization_info(processor_handles_[dv_ind], nullptr);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_INVAL);
    ASSERT_EQ(ret, AMDSMI_STATUS_INVAL);

    // Log the as-found state. These are whatever this GPU's mock/live sysfs
    // node currently holds -- nothing has written them, so they are not
    // asserted against fixed literals here. Every settable field below is
    // instead exercised by writing a known value via
    // amdsmi_set_power_stabilization_param(), reading it back, asserting on
    // *that*, and restoring the original value afterward.
    IF_VERB(STANDARD) {
      std::cout << "\thost_control_enabled: " << info.host_control_enabled << std::endl;
      std::cout << "\tenabled: " << info.enabled << std::endl;
      std::cout << "\tpower_floor: " << info.power_floor << std::endl;
      std::cout << "\tdelay: " << info.delay << std::endl;
      std::cout << "\tinsertion_priority: " << info.insertion_priority << std::endl;
      std::cout << "\tramp_enabled: " << info.ramp_enabled << std::endl;
      std::cout << "\tramp_max_ramp_up_rate: " << info.ramp_max_ramp_up_rate << std::endl;
      std::cout << "\tramp_max_ramp_down_rate: " << info.ramp_max_ramp_down_rate << std::endl;
      std::cout << "\tramp_min_time_at_power_floor: " << info.ramp_min_time_at_power_floor
                << std::endl;
      std::cout << "\tramp_stop_delay: " << info.ramp_stop_delay << std::endl;
    }
    // host_control_enabled is read-only: no param_id maps to it (see
    // power_stabilization_param_relpath()), so there is no way for this test
    // to set up a known value to assert against. The successful read above
    // (CHK_ERR_ASRT) is the only thing this test can verify for this field.

    // value.type mismatch must be rejected with INVAL, independent of param_id.
    amdsmi_value_t bad_value{};
    bad_value.type = AMDSMI_VALUE_TYPE_DOUBLE;
    bad_value.data.double_value = 1.0;
    DISPLAY_AMDSMI_API("amdsmi_set_power_stabilization_param", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    ret = amdsmi_set_power_stabilization_param(
        processor_handles_[dv_ind], AMDSMI_POWER_STABILIZATION_PARAM_POWER_FLOOR, &bad_value);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_INVAL);
    ASSERT_EQ(ret, AMDSMI_STATUS_INVAL);

    // Reverse direction: an INT64 value where a DOUBLE param is expected
    // must also be rejected with INVAL.
    amdsmi_value_t bad_value_reverse{};
    bad_value_reverse.type = AMDSMI_VALUE_TYPE_INT64;
    bad_value_reverse.data.int64_value = 1;
    ret = amdsmi_set_power_stabilization_param(
        processor_handles_[dv_ind], AMDSMI_POWER_STABILIZATION_PARAM_RAMP_MAX_RAMP_UP_RATE,
        &bad_value_reverse);
    ASSERT_EQ(ret, AMDSMI_STATUS_INVAL);

    // Nullptr value must be rejected with INVAL.
    ret = amdsmi_set_power_stabilization_param(
        processor_handles_[dv_ind], AMDSMI_POWER_STABILIZATION_PARAM_POWER_FLOOR, nullptr);
    ASSERT_EQ(ret, AMDSMI_STATUS_INVAL);

    // An invalid param_id must be rejected with INVAL.
    amdsmi_value_t unused_value{};
    unused_value.type = AMDSMI_VALUE_TYPE_INT64;
    unused_value.data.int64_value = 1;
    int invalid_param_id = 999;
    ret = amdsmi_set_power_stabilization_param(
        processor_handles_[dv_ind],
        static_cast<amdsmi_power_stabilization_param_id_t>(invalid_param_id), &unused_value);
    ASSERT_EQ(ret, AMDSMI_STATUS_INVAL);

    // Set + read-back ENABLE, restoring the original value afterward. The
    // restore guard is constructed before the mutating write so it is live
    // for the whole round trip: if CHK_ERR_ASRT/ASSERT_EQ below returns out
    // of Run() early, the guard's destructor still restores ENABLE when this
    // block's scope unwinds (see ScopedPowerStabilizationRestore above).
    {
      bool orig_enabled = info.enabled;
      amdsmi_value_t restore_enabled{};
      restore_enabled.type = AMDSMI_VALUE_TYPE_INT64;
      restore_enabled.data.int64_value = orig_enabled ? 1 : 0;
      ScopedPowerStabilizationRestore guard(
          processor_handles_[dv_ind], AMDSMI_POWER_STABILIZATION_PARAM_ENABLE, restore_enabled);

      amdsmi_value_t new_enabled{};
      new_enabled.type = AMDSMI_VALUE_TYPE_INT64;
      new_enabled.data.int64_value = orig_enabled ? 0 : 1;

      DISPLAY_AMDSMI_API("amdsmi_set_power_stabilization_param", "gpu=" + std::to_string(dv_ind),
                         VERB(STANDARD));
      ret = amdsmi_set_power_stabilization_param(
          processor_handles_[dv_ind], AMDSMI_POWER_STABILIZATION_PARAM_ENABLE, &new_enabled);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
      CHK_ERR_ASRT(ret)

      ret = amdsmi_get_power_stabilization_info(processor_handles_[dv_ind], &info);
      CHK_ERR_ASRT(ret)
      ASSERT_EQ(info.enabled, !orig_enabled);
    }

    // Set + read-back one top-level param (POWER_FLOOR), restoring the
    // original value afterward (RAII guard, see ENABLE above).
    {
      uint32_t orig_power_floor = info.power_floor;
      amdsmi_value_t restore_power_floor{};
      restore_power_floor.type = AMDSMI_VALUE_TYPE_INT64;
      restore_power_floor.data.int64_value = static_cast<int64_t>(orig_power_floor);
      ScopedPowerStabilizationRestore guard(processor_handles_[dv_ind],
                                            AMDSMI_POWER_STABILIZATION_PARAM_POWER_FLOOR,
                                            restore_power_floor);

      // Toggle between two known-valid points rather than unconditionally
      // adding 5, which would write an out-of-range value (power_floor's
      // documented max is 100) if orig_power_floor is already near the top.
      amdsmi_value_t new_power_floor{};
      new_power_floor.type = AMDSMI_VALUE_TYPE_INT64;
      new_power_floor.data.int64_value =
          static_cast<int64_t>(orig_power_floor) + (orig_power_floor >= 50 ? -5 : 5);

      DISPLAY_AMDSMI_API("amdsmi_set_power_stabilization_param", "gpu=" + std::to_string(dv_ind),
                         VERB(STANDARD));
      ret = amdsmi_set_power_stabilization_param(processor_handles_[dv_ind],
                                                 AMDSMI_POWER_STABILIZATION_PARAM_POWER_FLOOR,
                                                 &new_power_floor);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
      CHK_ERR_ASRT(ret)

      DISPLAY_AMDSMI_API("amdsmi_get_power_stabilization_info", "gpu=" + std::to_string(dv_ind),
                         VERB(STANDARD));
      ret = amdsmi_get_power_stabilization_info(processor_handles_[dv_ind], &info);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
      CHK_ERR_ASRT(ret)
      ASSERT_EQ(info.power_floor, static_cast<uint32_t>(new_power_floor.data.int64_value));
    }

    // Set + read-back DELAY, restoring the original value afterward (RAII
    // guard, see ENABLE above).
    {
      uint32_t orig_delay = info.delay;
      amdsmi_value_t restore_delay{};
      restore_delay.type = AMDSMI_VALUE_TYPE_INT64;
      restore_delay.data.int64_value = static_cast<int64_t>(orig_delay);
      ScopedPowerStabilizationRestore guard(processor_handles_[dv_ind],
                                            AMDSMI_POWER_STABILIZATION_PARAM_DELAY, restore_delay);

      amdsmi_value_t new_delay_param{};
      new_delay_param.type = AMDSMI_VALUE_TYPE_INT64;
      new_delay_param.data.int64_value = static_cast<int64_t>(orig_delay) + 1;

      DISPLAY_AMDSMI_API("amdsmi_set_power_stabilization_param", "gpu=" + std::to_string(dv_ind),
                         VERB(STANDARD));
      ret = amdsmi_set_power_stabilization_param(
          processor_handles_[dv_ind], AMDSMI_POWER_STABILIZATION_PARAM_DELAY, &new_delay_param);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
      CHK_ERR_ASRT(ret)

      ret = amdsmi_get_power_stabilization_info(processor_handles_[dv_ind], &info);
      CHK_ERR_ASRT(ret)
      ASSERT_EQ(info.delay, orig_delay + 1);
    }

    // Set + read-back INSERTION_PRIORITY, restoring the original value
    // afterward (RAII guard, see ENABLE above).
    {
      amdsmi_hawk_insertion_priority_t orig_priority = info.insertion_priority;
      amdsmi_value_t restore_priority{};
      restore_priority.type = AMDSMI_VALUE_TYPE_INT64;
      restore_priority.data.int64_value = static_cast<int64_t>(orig_priority);
      ScopedPowerStabilizationRestore guard(processor_handles_[dv_ind],
                                            AMDSMI_POWER_STABILIZATION_PARAM_INSERTION_PRIORITY,
                                            restore_priority);

      amdsmi_hawk_insertion_priority_t new_priority =
          static_cast<amdsmi_hawk_insertion_priority_t>((static_cast<int>(orig_priority) + 1) % 3);
      amdsmi_value_t new_priority_value{};
      new_priority_value.type = AMDSMI_VALUE_TYPE_INT64;
      new_priority_value.data.int64_value = static_cast<int64_t>(new_priority);

      DISPLAY_AMDSMI_API("amdsmi_set_power_stabilization_param", "gpu=" + std::to_string(dv_ind),
                         VERB(STANDARD));
      ret = amdsmi_set_power_stabilization_param(
          processor_handles_[dv_ind], AMDSMI_POWER_STABILIZATION_PARAM_INSERTION_PRIORITY,
          &new_priority_value);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
      CHK_ERR_ASRT(ret)

      ret = amdsmi_get_power_stabilization_info(processor_handles_[dv_ind], &info);
      CHK_ERR_ASRT(ret)
      ASSERT_EQ(info.insertion_priority, new_priority);
    }

    // Set + read-back RAMP_ENABLE, restoring the original value afterward
    // (RAII guard, see ENABLE above).
    {
      bool orig_ramp_enabled = info.ramp_enabled;
      amdsmi_value_t restore_ramp_enabled{};
      restore_ramp_enabled.type = AMDSMI_VALUE_TYPE_INT64;
      restore_ramp_enabled.data.int64_value = orig_ramp_enabled ? 1 : 0;
      ScopedPowerStabilizationRestore guard(processor_handles_[dv_ind],
                                            AMDSMI_POWER_STABILIZATION_PARAM_RAMP_ENABLE,
                                            restore_ramp_enabled);

      amdsmi_value_t new_ramp_enabled{};
      new_ramp_enabled.type = AMDSMI_VALUE_TYPE_INT64;
      new_ramp_enabled.data.int64_value = orig_ramp_enabled ? 0 : 1;

      DISPLAY_AMDSMI_API("amdsmi_set_power_stabilization_param", "gpu=" + std::to_string(dv_ind),
                         VERB(STANDARD));
      ret = amdsmi_set_power_stabilization_param(processor_handles_[dv_ind],
                                                 AMDSMI_POWER_STABILIZATION_PARAM_RAMP_ENABLE,
                                                 &new_ramp_enabled);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
      CHK_ERR_ASRT(ret)

      ret = amdsmi_get_power_stabilization_info(processor_handles_[dv_ind], &info);
      CHK_ERR_ASRT(ret)
      ASSERT_EQ(info.ramp_enabled, !orig_ramp_enabled);
    }

    // Set + read-back a ramp/ param (RAMP_MAX_RAMP_UP_RATE), restoring
    // afterward (RAII guard, see ENABLE above), to confirm the ramp/
    // subdirectory path is exercised too.
    {
      double orig_ramp_up = info.ramp_max_ramp_up_rate;
      amdsmi_value_t restore_ramp_up{};
      restore_ramp_up.type = AMDSMI_VALUE_TYPE_DOUBLE;
      restore_ramp_up.data.double_value = orig_ramp_up;
      ScopedPowerStabilizationRestore guard(processor_handles_[dv_ind],
                                            AMDSMI_POWER_STABILIZATION_PARAM_RAMP_MAX_RAMP_UP_RATE,
                                            restore_ramp_up);

      // Toggle between two known-valid points rather than unconditionally
      // adding 5, which would write an out-of-range value (ramp rates'
      // documented max is 100) if orig_ramp_up is already near the top.
      amdsmi_value_t new_ramp_up{};
      new_ramp_up.type = AMDSMI_VALUE_TYPE_DOUBLE;
      new_ramp_up.data.double_value = orig_ramp_up + (orig_ramp_up >= 50 ? -5 : 5);

      DISPLAY_AMDSMI_API("amdsmi_set_power_stabilization_param", "gpu=" + std::to_string(dv_ind),
                         VERB(STANDARD));
      ret = amdsmi_set_power_stabilization_param(
          processor_handles_[dv_ind], AMDSMI_POWER_STABILIZATION_PARAM_RAMP_MAX_RAMP_UP_RATE,
          &new_ramp_up);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
      CHK_ERR_ASRT(ret)

      ret = amdsmi_get_power_stabilization_info(processor_handles_[dv_ind], &info);
      CHK_ERR_ASRT(ret)
      ASSERT_DOUBLE_EQ(info.ramp_max_ramp_up_rate, new_ramp_up.data.double_value);
    }

    // Set + read-back RAMP_MAX_RAMP_DOWN_RATE, restoring afterward (RAII
    // guard, see ENABLE above).
    {
      double orig_ramp_down = info.ramp_max_ramp_down_rate;
      amdsmi_value_t restore_ramp_down{};
      restore_ramp_down.type = AMDSMI_VALUE_TYPE_DOUBLE;
      restore_ramp_down.data.double_value = orig_ramp_down;
      ScopedPowerStabilizationRestore guard(
          processor_handles_[dv_ind], AMDSMI_POWER_STABILIZATION_PARAM_RAMP_MAX_RAMP_DOWN_RATE,
          restore_ramp_down);

      // Toggle between two known-valid points rather than unconditionally
      // adding 5, which would write an out-of-range value (ramp rates'
      // documented max is 100) if orig_ramp_down is already near the top.
      amdsmi_value_t new_ramp_down{};
      new_ramp_down.type = AMDSMI_VALUE_TYPE_DOUBLE;
      new_ramp_down.data.double_value = orig_ramp_down + (orig_ramp_down >= 50 ? -5 : 5);

      DISPLAY_AMDSMI_API("amdsmi_set_power_stabilization_param", "gpu=" + std::to_string(dv_ind),
                         VERB(STANDARD));
      ret = amdsmi_set_power_stabilization_param(
          processor_handles_[dv_ind], AMDSMI_POWER_STABILIZATION_PARAM_RAMP_MAX_RAMP_DOWN_RATE,
          &new_ramp_down);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
      CHK_ERR_ASRT(ret)

      ret = amdsmi_get_power_stabilization_info(processor_handles_[dv_ind], &info);
      CHK_ERR_ASRT(ret)
      ASSERT_DOUBLE_EQ(info.ramp_max_ramp_down_rate, new_ramp_down.data.double_value);
    }

    // Set + read-back RAMP_MIN_TIME_AT_POWER_FLOOR, restoring afterward
    // (RAII guard, see ENABLE above).
    {
      uint32_t orig_ramp_min_time = info.ramp_min_time_at_power_floor;
      amdsmi_value_t restore_ramp_min_time{};
      restore_ramp_min_time.type = AMDSMI_VALUE_TYPE_INT64;
      restore_ramp_min_time.data.int64_value = static_cast<int64_t>(orig_ramp_min_time);
      ScopedPowerStabilizationRestore guard(
          processor_handles_[dv_ind], AMDSMI_POWER_STABILIZATION_PARAM_RAMP_MIN_TIME_AT_POWER_FLOOR,
          restore_ramp_min_time);

      amdsmi_value_t new_ramp_min_time{};
      new_ramp_min_time.type = AMDSMI_VALUE_TYPE_INT64;
      new_ramp_min_time.data.int64_value = static_cast<int64_t>(orig_ramp_min_time) + 1;

      DISPLAY_AMDSMI_API("amdsmi_set_power_stabilization_param", "gpu=" + std::to_string(dv_ind),
                         VERB(STANDARD));
      ret = amdsmi_set_power_stabilization_param(
          processor_handles_[dv_ind], AMDSMI_POWER_STABILIZATION_PARAM_RAMP_MIN_TIME_AT_POWER_FLOOR,
          &new_ramp_min_time);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
      CHK_ERR_ASRT(ret)

      ret = amdsmi_get_power_stabilization_info(processor_handles_[dv_ind], &info);
      CHK_ERR_ASRT(ret)
      ASSERT_EQ(info.ramp_min_time_at_power_floor, orig_ramp_min_time + 1);
    }

    // Set + read-back RAMP_STOP_DELAY, restoring afterward (RAII guard, see
    // ENABLE above).
    {
      uint32_t orig_ramp_stop_delay = info.ramp_stop_delay;
      amdsmi_value_t restore_ramp_stop_delay{};
      restore_ramp_stop_delay.type = AMDSMI_VALUE_TYPE_INT64;
      restore_ramp_stop_delay.data.int64_value = static_cast<int64_t>(orig_ramp_stop_delay);
      ScopedPowerStabilizationRestore guard(processor_handles_[dv_ind],
                                            AMDSMI_POWER_STABILIZATION_PARAM_RAMP_STOP_DELAY,
                                            restore_ramp_stop_delay);

      amdsmi_value_t new_ramp_stop_delay{};
      new_ramp_stop_delay.type = AMDSMI_VALUE_TYPE_INT64;
      new_ramp_stop_delay.data.int64_value = static_cast<int64_t>(orig_ramp_stop_delay) + 1;

      DISPLAY_AMDSMI_API("amdsmi_set_power_stabilization_param", "gpu=" + std::to_string(dv_ind),
                         VERB(STANDARD));
      ret = amdsmi_set_power_stabilization_param(processor_handles_[dv_ind],
                                                 AMDSMI_POWER_STABILIZATION_PARAM_RAMP_STOP_DELAY,
                                                 &new_ramp_stop_delay);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
      CHK_ERR_ASRT(ret)

      ret = amdsmi_get_power_stabilization_info(processor_handles_[dv_ind], &info);
      CHK_ERR_ASRT(ret)
      ASSERT_EQ(info.ramp_stop_delay, orig_ramp_stop_delay + 1);
    }

    // NO_PERM: mark the target sysfs leaf immutable (blocks even root) and
    // confirm the write is rejected, then the guard restores the flag. The
    // card path is derived from the device under test in this loop
    // iteration via its AMDSmiGPUDevice (the same get_gpu_path() the
    // production code itself uses to build the sysfs base path, see
    // Device::get_power_stabilization_root_path() in rocm_smi_device.cc),
    // not hardcoded, so this check is meaningful on multi-GPU systems where
    // the Power Stabilization node isn't on card0.
    {
      amd::smi::AMDSmiGPUDevice* gpu_device = nullptr;
      ret = get_gpu_device_from_handle(processor_handles_[dv_ind], &gpu_device);
      ASSERT_EQ(ret, AMDSMI_STATUS_SUCCESS);
      ASSERT_NE(gpu_device, nullptr);
      const std::string delay_path =
          "/sys/class/drm/" + gpu_device->get_gpu_path() + "/device/power_stabilization/delay";
      ScopedImmutable guard(delay_path);
      if (!guard.applied()) {
        std::cout << "\t** Could not set FS_IMMUTABLE_FL on " << delay_path
                  << " (requires CAP_LINUX_IMMUTABLE / root). Skipping NO_PERM check." << std::endl;
      } else {
        amdsmi_value_t new_delay{};
        new_delay.type = AMDSMI_VALUE_TYPE_INT64;
        new_delay.data.int64_value = static_cast<int64_t>(info.delay) + 1;

        DISPLAY_AMDSMI_API("amdsmi_set_power_stabilization_param", "gpu=" + std::to_string(dv_ind),
                           VERB(STANDARD));
        ret = amdsmi_set_power_stabilization_param(
            processor_handles_[dv_ind], AMDSMI_POWER_STABILIZATION_PARAM_DELAY, &new_delay);
        DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_NO_PERM);
        ASSERT_EQ(ret, AMDSMI_STATUS_NO_PERM);
      }
    }
  }
}
