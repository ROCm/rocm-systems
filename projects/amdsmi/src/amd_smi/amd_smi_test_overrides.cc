// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "amd_smi/impl/amd_smi_test_overrides.h"

namespace amd::smi::test_hooks {

namespace {
// Not thread-safe for tests: only single-threaded tests mutate these (via
// amd::smi::testing::ScopedOverride), never while another thread may be using the library.
// Production only reads them, possibly from several threads.
std::string g_sysfs_drm_root = "/sys/class/drm/";
std::string g_kfd_device_path = "/dev/kfd";
}  // namespace

// The setters are declared only in the tests (test_fixture_utils.h), not in the header above.
const std::string& SysfsDrmRoot() { return g_sysfs_drm_root; }

AMDSMI_TEST_HOOK void SetSysfsDrmRoot(const std::string& root) { g_sysfs_drm_root = root; }

const std::string& KfdDevicePath() { return g_kfd_device_path; }

AMDSMI_TEST_HOOK void SetKfdDevicePath(const std::string& path) { g_kfd_device_path = path; }

}  // namespace amd::smi::test_hooks
