// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <sys/stat.h>
#include <sys/types.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_test_overrides.h"

namespace amd::smi {
class AMDSmiProcessor;
}  // namespace amd::smi

// Setters for the hooks in amd_smi_test_overrides.h, declared here and not there so product code
// cannot call them. Change a hook only through ScopedOverride<T>, never bare from a test body.
namespace amd::smi::test_hooks {
void SetSysfsDrmRoot(const std::string& root);
void SetKfdDevicePath(const std::string& path);
}  // namespace amd::smi::test_hooks

// Linker-wrapped ioctl(); see ScopedFakeIoctl. Not for tests to call.
extern "C" int __wrap_ioctl(int fd, unsigned long request, ...);

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

// RAII: fakes the ioctl() calls product code issues, per request code. The test binary is linked
// with -Wl,--wrap=ioctl (tests/amd_smi_test/CMakeLists.txt), so every ioctl() in the library -- C++
// and the vendored C sources alike -- reaches the active fake with no change to product code. While
// a fake is active, a request no handler claims fails with ENOTTY and never reaches a real device;
// with none active, ioctl() behaves as usual. Fakes nest: an inner fake answers the requests it has
// handlers for and passes the rest to the outer one, so independent fixtures compose. Only code
// linked into the test binary is reached; ioctls issued inside a shared library (libdrm) are not.
class ScopedFakeIoctl {
 public:
  // Returns what ioctl() returns, setting errno when it returns -1. `arg` is the request's
  // argument.
  using Handler = std::function<int(void* arg)>;

  ScopedFakeIoctl();
  ~ScopedFakeIoctl();

  ScopedFakeIoctl(const ScopedFakeIoctl&) = delete;
  ScopedFakeIoctl& operator=(const ScopedFakeIoctl&) = delete;

  // Answers ioctl `request` with `handler`; a later call for the same request replaces it.
  void RegisterHandler(unsigned long request, Handler handler);

  // Same, with the argument as its real type:
  // RegisterHandler<kfd_ioctl_get_version_args>(REQ, [](auto& args) { ...; return 0; });
  template <typename Arg, typename Fn>
  void RegisterHandler(unsigned long request, Fn fn) {
    RegisterHandler(
        request, Handler([fn = std::move(fn)](void* arg) { return fn(*static_cast<Arg*>(arg)); }));
  }

  // Answers `request` by failing: ioctl() returns -1 with errno set to `error`.
  void RegisterFailure(unsigned long request, int error);

  // How many times this fake's handler for `request` has run.
  int CallCount(unsigned long request) const;

 private:
  friend int ::__wrap_ioctl(int fd, unsigned long request, ...);
  static int Dispatch(unsigned long request, void* arg);
  static ScopedFakeIoctl* active_;

  std::map<unsigned long, Handler> handlers_;
  std::map<unsigned long, int> calls_;
  ScopedFakeIoctl* previous_active_;
};

// Answers AMDKFD_IOC_GET_VERSION on `ioctl` with the given KFD ioctl version.
void RegisterFakeKfdVersion(ScopedFakeIoctl& ioctl, uint32_t major, uint32_t minor);

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

// RAII: stands in for /dev/kfd and the KFD SMI event stream it hands out, so the real
// amdsmi_init_gpu_event_notification(), amdsmi_set_gpu_event_notification_mask(),
// amdsmi_get_gpu_event_notification() and amdsmi_stop_gpu_event_notification() run without
// hardware. Built on ScopedFakeIoctl. Like the kernel, the stream starts with an empty event mask
// and only queues events whose bit the library has written with the mask call. Declare it after
// the ScopedRocmSmiDevice it serves; the test must call init before and stop after the other calls.
class ScopedFakeKfdEventStream {
 public:
  explicit ScopedFakeKfdEventStream(uint32_t rocm_smi_device_index);
  ~ScopedFakeKfdEventStream();

  ScopedFakeKfdEventStream(const ScopedFakeKfdEventStream&) = delete;
  ScopedFakeKfdEventStream& operator=(const ScopedFakeKfdEventStream&) = delete;

  // Queues one line as the kernel writes it ("<event number in hex> <payload>\n"), unless the mask
  // written so far excludes `event`. Event numbers outside 1..64 have no mask bit and always queue.
  void Write(uint32_t event, const std::string& payload);

  // How many times the library has asked this fake for the KFD version.
  int VersionQueryCount() const;

 private:
  uint32_t device_index_;
  FakeSysfsTree kfd_tree_;  // holds a plain file for the library to open as /dev/kfd
  amd::smi::testing::ScopedFakeIoctl ioctl_;
  int stream_fd_ = -1;  // this end of the socket pair
  int device_fd_ = -1;  // the library's end, until the library asks KFD for the stream
  uint64_t mask_ = 0;
  std::optional<amd::smi::testing::ScopedOverride<std::string>> path_override_;
};
