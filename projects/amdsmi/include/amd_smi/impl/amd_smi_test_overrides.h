// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

// Test hooks: product code reads these instead of hardcoded paths, so a unit test can point it at a
// fake. Product code only gets the getters here. The setters are declared in the tests' own
// test_fixture_utils.h, so product code cannot call them, and every hook is hidden so no shared
// library exports one. Only the paths tests have needed so far are hooked; add one here when a new
// test needs another. Compiled into both libamd_smi and rocm_smi.
#define AMDSMI_TEST_HOOK __attribute__((visibility("hidden")))

namespace amd::smi::test_hooks {

// Root of the DRM sysfs tree smi_amdgpu_get_enabled_blocks() reads. Defaults to "/sys/class/drm/",
// so production reads "/sys/class/drm/card0/device/ras/features". A test sets it to a FakeSysfsTree
// root such as "/tmp/amdsmi_fake_sysfs_XXXXXX/" to read
// "/tmp/amdsmi_fake_sysfs_XXXXXX/card0/device/ras/features" instead.
AMDSMI_TEST_HOOK const std::string& SysfsDrmRoot();

// KFD device node rsmi_event_notification_init() opens. Defaults to "/dev/kfd". A test points it at
// an empty placeholder file such as "/tmp/amdsmi_fake_sysfs_XXXXXX/kfd" (ScopedFakeKfdEventStream
// does this); ScopedFakeIoctl answers the ioctls on it, so no real device is needed.
AMDSMI_TEST_HOOK const std::string& KfdDevicePath();

}  // namespace amd::smi::test_hooks
