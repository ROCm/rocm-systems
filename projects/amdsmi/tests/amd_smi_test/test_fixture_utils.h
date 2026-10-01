// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <sys/stat.h>
#include <sys/types.h>

#include <cstdint>
#include <functional>
#include <string>

#include "amd_smi/amdsmi.h"

namespace amd::smi {
class AMDSmiProcessor;
}  // namespace amd::smi

// Generic (not RAS-specific) fake-sysfs fixture tree, reusable by any unit
// test that needs a real, temporary, hardware-inert file tree -- e.g. for
// pointing a ScopedOverride<std::string> sysfs-root seam at real files instead
// of in-memory strings. mkdtemp()'s a unique /tmp directory (e.g.
// "/tmp/amdsmi_fake_sysfs_Ab3xYz") on construction (collision-free even
// across parallel CI shards). Deleted automatically -- along with every file
// written into it -- as soon as this object goes out of scope, i.e. at the
// end of the TEST() body that declared it; nothing needs to be cleaned up by
// hand, and nothing from one test can leak into the next.
class FakeSysfsTree {
 public:
  FakeSysfsTree();
  ~FakeSysfsTree();

  FakeSysfsTree(const FakeSysfsTree&) = delete;
  FakeSysfsTree& operator=(const FakeSysfsTree&) = delete;

  // The temp directory path, e.g. "/tmp/amdsmi_fake_sysfs_Ab3xYz" -- for
  // pointing a sysfs-root override guard at it.
  const std::string& root() const { return root_; }

  // Writes relative_path (creating parent directories as needed) with the
  // given content. This is how a test defines fixture content -- e.g.
  // WriteFile("card0/device/ras/features", "feature mask: 0x1abf877\n")
  // creates the real file at root() + "/card0/device/ras/features"
  // (e.g. "/tmp/amdsmi_fake_sysfs_Ab3xYz/card0/device/ras/features").
  void WriteFile(const std::string& relative_path, const std::string& content);

  // Deletes a previously-written file, for simulating one that existed and
  // then disappeared mid-scenario. Distinct from "never existed," which is
  // just never calling WriteFile() for that path.
  void RemoveFile(const std::string& relative_path);

  // chmod()'s a previously-written file, e.g. 0000, to simulate a
  // permission-denied read (EACCES). Has no effect when the test binary runs
  // as root, since root bypasses file-mode read checks entirely -- callers
  // must guard a permission-denied test case with a geteuid() == 0 skip.
  void SetPermissions(const std::string& relative_path, mode_t mode);

 private:
  std::string root_;
};

namespace amd::smi::testing {

// Generic, header-only RAII guard: saves the current value (via getter),
// installs a new one (via setter), and restores the saved value on
// destruction -- even if a gtest ASSERT_* returns early, since destructors of
// stack objects still run. This is the only intended way to change any
// test-only override in this binary; there is no bare "set and forget"
// setter. Being fully inline/templated, this guard itself never crosses the
// shared-library ABI boundary, so it carries no export/visibility concerns
// and is reusable for future fake interfaces (ioctl, libdrm loader, an ESMI
// seam for CPU, ...) without duplicating guard boilerplate.
template <typename T>
class ScopedOverride {
 public:
  ScopedOverride(std::function<T()> getter, std::function<void(const T&)> setter, T new_value)
      : setter_(std::move(setter)), old_value_(getter()) {
    setter_(new_value);
  }
  ~ScopedOverride() { setter_(old_value_); }

  ScopedOverride(const ScopedOverride&) = delete;
  ScopedOverride& operator=(const ScopedOverride&) = delete;

 private:
  std::function<void(const T&)> setter_;
  T old_value_;
};

// RAII: registers/unregisters a synthetic processor with AMDSmiSystem so
// amd-smi's real public API dispatch (handle_to_processor()) can find it
// without real hardware discovery. Reusable for any processor type --
// AMDSmiSystem picks the right internal set from the processor's own
// get_processor_type().
class ScopedProcessorRegistration {
 public:
  explicit ScopedProcessorRegistration(amd::smi::AMDSmiProcessor* processor);
  ~ScopedProcessorRegistration();

  ScopedProcessorRegistration(const ScopedProcessorRegistration&) = delete;
  ScopedProcessorRegistration& operator=(const ScopedProcessorRegistration&) = delete;

  amdsmi_processor_handle handle() const;

 private:
  amd::smi::AMDSmiProcessor* processor_;
};

}  // namespace amd::smi::testing

// RAII: pushes a real, hardware-inert rocm_smi::Device onto RocmSMI's device
// vector (already public/mutable; no test seam needed for this part -- see
// docs/design/faking-external-interfaces.md). Restores the vector to its
// prior size on destruction so index assignment stays predictable across
// tests run in the same process. Reusable by any GPU unit test, not just RAS.
class ScopedRocmSmiDevice {
 public:
  explicit ScopedRocmSmiDevice(const std::string& path);
  ~ScopedRocmSmiDevice();

  ScopedRocmSmiDevice(const ScopedRocmSmiDevice&) = delete;
  ScopedRocmSmiDevice& operator=(const ScopedRocmSmiDevice&) = delete;

  uint32_t index() const { return index_; }

 private:
  uint32_t index_;
};

// RAII: calls the real amdsmi_init()/amdsmi_shut_down() around a test case.
// kNoHardwareDiscoveryInitFlags is not a public amdsmi_init_flags_t value --
// amdsmi_init_flags_t has no defined "none" flag. Zero happens to make every
// `if (flags & AMDSMI_INIT_AMD_*)` gate in AMDSmiSystem::init()
// (amd_smi_system.cc) false, skipping all hardware discovery and returning
// AMDSMI_STATUS_SUCCESS -- an internal implementation detail, not a
// documented contract, so this is pinned as a named constant with this
// comment rather than a bare 0. See docs/design/faking-external-interfaces.md's
// "How to reuse this" section. Reusable by any unit test needing amdsmi_init(),
// not just RAS.
class ScopedAmdsmiInit {
 public:
  static constexpr uint64_t kNoHardwareDiscoveryInitFlags = 0;

  ScopedAmdsmiInit();
  ~ScopedAmdsmiInit();

  ScopedAmdsmiInit(const ScopedAmdsmiInit&) = delete;
  ScopedAmdsmiInit& operator=(const ScopedAmdsmiInit&) = delete;
};
