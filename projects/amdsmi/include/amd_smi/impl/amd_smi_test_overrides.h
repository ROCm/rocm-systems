// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

// Library-local test seam for the sysfs root smi_amdgpu_get_enabled_blocks()
// (src/amd_smi/amd_smi_utils.cc) reads .../ras/features from. Defaults to the
// real "/sys/class/drm/" prefix -- e.g. production reads
// "/sys/class/drm/card0/device/ras/features". A test that overrides this to
// "/tmp/amdsmi_fake_sysfs_XXXXXX/" (see FakeSysfsTree in
// tests/amd_smi_test/test_fixture_utils.h) makes that same code read
// "/tmp/amdsmi_fake_sysfs_XXXXXX/card0/device/ras/features" instead. Not
// amdsmi_-prefixed, so the linker version script keeps it out of
// libamd_smi.so; tests reach it through the static archive. Only
// amd::smi::testing::ScopedOverride<std::string> (test_fixture_utils.h) is
// meant to call the setter -- never call it bare from a test body.
// Safe to call from any number of ordinary functions invoked at runtime (the
// intended path for migrating other sysfs reads onto this seam); unsafe only
// if some global/static object's own constructor called it during static init.
const std::string& smi_amdgpu_sysfs_drm_root();
void smi_amdgpu_set_sysfs_drm_root_for_testing(const std::string& root);
