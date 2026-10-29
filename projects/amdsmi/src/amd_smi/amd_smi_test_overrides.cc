// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "amd_smi/impl/amd_smi_test_overrides.h"

namespace {
// Not thread-safe: only the single-threaded tests mutate it (via
// amd::smi::testing::ScopedOverride); production never writes it.
std::string g_sysfs_drm_root = "/sys/class/drm/";
}  // namespace

// See amd_smi_test_overrides.h for the contract.
const std::string& smi_amdgpu_sysfs_drm_root() { return g_sysfs_drm_root; }

void smi_amdgpu_set_sysfs_drm_root_for_testing(const std::string& root) { g_sysfs_drm_root = root; }
