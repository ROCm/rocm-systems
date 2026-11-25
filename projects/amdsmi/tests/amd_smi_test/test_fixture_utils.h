// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <sys/stat.h>
#include <sys/types.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>

#include "amd_smi/amdsmi.h"
#include "rocm_smi/rocm_smi_kfd.h"

namespace amd::smi {
class AMDSmiProcessor;
}  // namespace amd::smi

// Setters for the hooks in amd_smi_test_overrides.h, declared here and not there so product code
// cannot call them. Change a hook only through ScopedOverride<T>, never bare from a test body.
namespace amd::smi::test_hooks {
void SetSysfsDrmRoot(const std::string& root);
void SetKfdNodesPathRoot(const std::string& root);
void SetAppTmpDir(const std::string& dir);
}  // namespace amd::smi::test_hooks

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

  // First line of a previously written file, or "" if it does not exist: reads back what
  // production code wrote.
  std::string ReadFile(const std::string& relative_path) const;

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

// RAII: sets an environment variable for the life of one test body and restores the previous
// state on exit -- the old value, or unset if there was none -- so the change cannot reach other
// tests. Not thread-safe; use it only in single-threaded tests.
class ScopedEnvVar {
 public:
  ScopedEnvVar(const char* name, const char* value);
  ~ScopedEnvVar();

  ScopedEnvVar(const ScopedEnvVar&) = delete;
  ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

 private:
  std::string name_;
  std::optional<std::string> old_value_;
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

// RAII wrapper around an inotify(7) watch on a single, already-existing file
// path. Tallies file operations (open, write-close) performed by ANY
// process on that path -- including a forked-and-privilege-dropped child --
// with zero changes to the production code under test. `path` must already
// exist when constructed. Construct BEFORE locking a file down (e.g. via
// FakeSysfsTree::SetPermissions()), not after, to keep IsAvailable() true
// regardless of which privilege level the test binary runs under. See
// docs/design/faking-external-interfaces.md's "Monitoring file operations
// in tests" section for the full design, constraints, and alternatives
// considered.
class ScopedFileWatch {
 public:
  // event_mask is a bitwise-OR of IN_* flags (<sys/inotify.h>), e.g.
  // IN_OPEN | IN_CLOSE_WRITE, naming which operations this watch observes.
  ScopedFileWatch(const std::string& path, uint32_t event_mask);
  ~ScopedFileWatch();

  ScopedFileWatch(const ScopedFileWatch&) = delete;
  ScopedFileWatch& operator=(const ScopedFileWatch&) = delete;

  // False if the watch couldn't be established -- inotify unavailable,
  // `path` didn't exist at construction time, or the constructing process
  // lacked read permission on `path` at that time (inotify_add_watch(2)).
  // Tests should GTEST_SKIP() (or skip just the affected assertion), not assert/fail, when this is
  // false: it means the mechanism itself is unusable here, not that zero events happened.
  bool IsAvailable() const { return watch_fd_ >= 0; }

  // Cumulative count of events matching any bit in `mask` seen since
  // construction. Drains pending inotify events on every call.
  uint32_t Count(uint32_t mask);

  // Convenience wrappers for the two cases this codebase needs today.
  uint32_t OpenCount();
  uint32_t WriteCount();

 private:
  // uint32_t is guaranteed exactly 32 bits by <cstdint>; this names that fact
  // instead of repeating the literal 32 at each use.
  static constexpr std::size_t kBitsPerMask = std::numeric_limits<uint32_t>::digits;

  void Drain();

  int inotify_fd_ = -1;
  int watch_fd_ = -1;
  std::array<uint32_t, kBitsPerMask> event_tally_{};  // indexed by bit position of each IN_* flag
};

// Runs `fn` as an unprivileged user and returns its status, so a test sees real file-permission
// failures whoever runs the suite. Not root: calls `fn` directly. Root bypasses file modes, so
// `fn` runs in a forked child that drops to "nobody"; the child only reports the status back.
amdsmi_status_t RunAsUnprivilegedUser(const std::function<amdsmi_status_t()>& fn);

}  // namespace amd::smi::testing

// RAII: pushes a real, hardware-inert rocm_smi::Device onto RocmSMI's device
// vector (already public/mutable; no test seam needed for this part -- see
// docs/design/faking-external-interfaces.md). Restores the vector to its
// prior size on destruction so index assignment stays predictable across
// tests run in the same process. Reusable by any GPU unit test, not just RAS.
//
// Also gives the device a render node, as real GPUs have: renderD<128 + index> is a symlink next
// to `path` pointing at it, so files written under the card directory resolve through either name.
class ScopedRocmSmiDevice {
 public:
  explicit ScopedRocmSmiDevice(const std::string& path);
  ~ScopedRocmSmiDevice();

  ScopedRocmSmiDevice(const ScopedRocmSmiDevice&) = delete;
  ScopedRocmSmiDevice& operator=(const ScopedRocmSmiDevice&) = delete;

  uint32_t index() const { return index_; }
  uint32_t render_minor() const;

 private:
  std::string RenderLink(const std::string& dir) const {
    return dir + "/renderD" + std::to_string(render_minor());
  }

  uint32_t index_;
  std::string link_dir_;  // Directory holding the renderD link; empty if none was created.
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

// KFD topology nodes for real GPU discovery, written into a FakeSysfsTree. A GPU's KFD node index
// and DRM card index are unrelated numbers; its DRM entries are card<C> and renderD<128 + C>.
//
// WriteGpuNode() writes what discovery needs: the properties Initialize() reads, a distinct
// nonzero location_id per node (a bdfid of 0 counts as unset), its render node, and an io_links
// directory, which discovery requires even when the node has no links. As in sysfs, card<C> links
// to the render node's directory, so a file written under either name is one file, and amd-smi
// reads it through renderD<128 + C>. Call it before writing files under card<C>.
void WriteGpuNode(FakeSysfsTree& tree, uint32_t node_idx, uint32_t card_idx, uint64_t gpu_id);

// A CPU node: discovery reads its links like any other node's, so it needs an io_links directory.
void WriteCpuNode(FakeSysfsTree& tree, uint32_t node_idx);

// /sys/class/kfd/kfd/topology/nodes/<node_idx>/io_links/<link_idx>/properties
void WriteIoLink(FakeSysfsTree& tree, uint32_t node_idx, uint32_t link_idx, uint32_t node_to,
                 amd::smi::IO_LINK_TYPE type, uint64_t weight);

// Runs the real amdsmi_init() GPU discovery over `tree`, which holds both the KFD topology and the
// DRM entries, and hands out the handles it found by KFD node index. ROCm SMI's boot-state files
// go into the tree too, so nothing is left in /tmp. amdsmi_shut_down() cleans up. Write every
// node and the files under it before constructing this. Throws if init fails.
class FakeKfdTopology {
 public:
  explicit FakeKfdTopology(FakeSysfsTree& tree);
  ~FakeKfdTopology();

  FakeKfdTopology(const FakeKfdTopology&) = delete;
  FakeKfdTopology& operator=(const FakeKfdTopology&) = delete;

  amdsmi_processor_handle handle(uint32_t node_index) const { return handles_.at(node_index); }

 private:
  amd::smi::testing::ScopedOverride<std::string> kfd_root_guard_;
  amd::smi::testing::ScopedOverride<std::string> drm_root_guard_;
  amd::smi::testing::ScopedOverride<std::string> tmp_dir_guard_;
  std::map<uint32_t, amdsmi_processor_handle> handles_;
};
