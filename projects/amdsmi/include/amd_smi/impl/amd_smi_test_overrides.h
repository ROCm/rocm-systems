// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include "amd_smi/impl/fdinfo.h"

// Test hooks: product code reads these instead of hardcoded paths, so a unit test can point it at a
// fake. Product code only gets the getters here. The setters are declared in the tests' own
// test_fixture_utils.h, so product code cannot call them, and every hook is hidden so no shared
// library exports one. Only the paths tests have needed so far are hooked; add one here when a new
// test needs another.
#define AMDSMI_TEST_HOOK __attribute__((visibility("hidden")))

namespace amd::smi::test_hooks {

// Root of the DRM sysfs tree that product code reads device files from, with no trailing "/".
// Defaults to "/sys/class/drm", so production reads "/sys/class/drm/card0/device/ras/features". A
// test sets it to a FakeSysfsTree root such as "/tmp/amdsmi_fake_sysfs_XXXXXX" to read
// "/tmp/amdsmi_fake_sysfs_XXXXXX/card0/device/ras/features" instead.
AMDSMI_TEST_HOOK const std::string& SysfsDrmRoot();

// Root of the KFD topology tree that KFDNode and IOLink discovery read. Defaults to
// "/sys/class/kfd/kfd/topology/nodes"; a test sets it to a FakeSysfsTree root.
AMDSMI_TEST_HOOK const std::string& KfdNodesPathRoot();

// Directory where ROCm SMI keeps its persistent boot-state files (rocmsmi_boot_*). Defaults to
// "/tmp", with no trailing "/". Real discovery writes them, so a test redirects them to keep from
// leaving files that a later real run on the same host would read.
AMDSMI_TEST_HOOK const std::string& AppTmpDir();

// Reader amdsmi_get_gpu_process_info_self() uses for the caller's fdinfo files. Defaults to null,
// which reads /proc/self/fdinfo; a test supplies fake DRM client text for descriptors it opened.
AMDSMI_TEST_HOOK const SelfFdinfoReader& SelfFdinfoReaderHook();

}  // namespace amd::smi::test_hooks
