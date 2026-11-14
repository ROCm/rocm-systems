// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "amd_smi/impl/amd_smi_test_overrides.h"

namespace amd::smi::test_hooks {

namespace {
// Not thread-safe for tests: only single-threaded tests mutate these (via
// amd::smi::testing::ScopedOverride), never while another thread may be using the library.
// Production only reads them, possibly from several threads.
std::string g_sysfs_drm_root = "/sys/class/drm";
std::string g_app_tmp_dir = "/tmp";
std::string g_kfd_nodes_path_root = "/sys/class/kfd/kfd/topology/nodes";
}  // namespace

// The setters are declared only in the tests (test_fixture_utils.h), not in the header above.
// This is so production code does not try to use the setter calls, only test code.

const std::string& SysfsDrmRoot() { return g_sysfs_drm_root; }

AMDSMI_TEST_HOOK void SetSysfsDrmRoot(const std::string& root) { g_sysfs_drm_root = root; }

const std::string& KfdNodesPathRoot() { return g_kfd_nodes_path_root; }

AMDSMI_TEST_HOOK void SetKfdNodesPathRoot(const std::string& root) { g_kfd_nodes_path_root = root; }

const std::string& AppTmpDir() { return g_app_tmp_dir; }

AMDSMI_TEST_HOOK void SetAppTmpDir(const std::string& dir) { g_app_tmp_dir = dir; }

}  // namespace amd::smi::test_hooks
