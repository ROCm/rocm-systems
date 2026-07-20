// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/kmd/linux/rpc.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "linux/uapi/kfd_ioctl.h"
RJ_DIAGNOSTIC_POP

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

constexpr const char *kKfdPath = "/dev/kfd";
constexpr const char *kTopologyNodesPaths[] = {
    "/sys/devices/virtual/kfd/kfd/topology/nodes",
    "/sys/class/kfd/kfd/topology/nodes",
};
constexpr uint32_t kGuestGpuId = 38144;
constexpr int kChildCount = 4;
constexpr int kStressThreadCount = 8;
constexpr int kStressIterations = 25;
constexpr uint64_t kAllocVa = 0x1000000000ULL;
constexpr uint64_t kAllocSize = 4096;

// Resolve the directory holding the config-path handoff file, mirroring the exact
// precedence in load_dbt_guest_config_from_runtime_config() so a config the test
// installs is the one the in-process DBT guest hook actually reads:
//   1. $ROCJITSU_INVOCATION_DIR (the launcher's per-invocation dir), then
//   2. <$ROCJITSU_RUNTIME_DIR>/<pid>/ (the per-PID tier the reader probes next;
//      the test and the hook share this process, so getpid() matches), then
//   3. <$ROCJITSU_RUNTIME_DIR>/ (the well-known location for the no-launcher case).
// The reader opens the FIRST tier whose config_path handoff actually exists, so this
// must probe them in the same order rather than short-circuiting on tier 1 merely
// being set: an $ROCJITSU_INVOCATION_DIR pointing at a dir without config_path must
// still fall through to the per-PID/base tiers a valid handoff may live in. When no
// tier has a handoff yet, return the highest-priority writable tier so
// install_inline_dbt_config() writes where the reader looks first.
// rpc_default_runtime_dir() already treats an unset OR empty $ROCJITSU_RUNTIME_DIR as
// "use $XDG_RUNTIME_DIR/rocjitsu or /tmp/rocjitsu-<uid>", so this never returns
// nullopt on that account — otherwise the test would write nowhere while the runtime
// hook still reads the default location.
std::optional<std::filesystem::path> config_handoff_dir() {
  std::vector<std::filesystem::path> tiers;
  if (const char *inv = std::getenv("ROCJITSU_INVOCATION_DIR"); inv && *inv)
    tiers.emplace_back(inv);
  const std::filesystem::path base(rocjitsu::rpc_default_runtime_dir());
  tiers.push_back(base / std::to_string(getpid()));
  tiers.push_back(base);

  std::error_code ec;
  for (const auto &tier : tiers) {
    if (std::filesystem::exists(tier / "config_path", ec))
      return tier;
  }
  // No handoff exists yet: install into the highest-priority tier the reader probes.
  return tiers.front();
}

std::optional<std::string> read_active_config_json() {
  const auto dir = config_handoff_dir();
  if (!dir)
    return std::nullopt;

  std::ifstream active_config(*dir / "config_path");
  std::string configured_path;
  if (!std::getline(active_config, configured_path) || configured_path.empty())
    return std::nullopt;

  std::ifstream config(std::filesystem::absolute(configured_path).lexically_normal());
  if (!config)
    return std::nullopt;
  return std::string(std::istreambuf_iterator<char>(config), std::istreambuf_iterator<char>());
}

bool install_inline_dbt_config(std::string simulator_json, const char *host_isa,
                               uint32_t lds_size_kb, std::string_view external_host_config = {}) {
  const auto dir = config_handoff_dir();
  if (!dir)
    return false;

  try {
    const std::filesystem::path runtime(*dir);
    std::filesystem::create_directories(runtime);
    const std::filesystem::path config_path = runtime / "inline_dbt_failure_config.json";

    const size_t insert_pos = simulator_json.find('{');
    if (insert_pos == std::string::npos)
      return false;
    std::ostringstream dbt_guest;
    dbt_guest << R"(
  "dbt_guest": {
    "enabled": true,
    "guest_isa": "gfx950",
    "host_isa": ")"
              << host_isa << R"(",
    "execution_backend": "simulator",)";
    if (!external_host_config.empty())
      dbt_guest << R"(
    "simulator_config": ")"
                << external_host_config << R"(",)";
    dbt_guest << R"(
    "guest_device": {
      "gpu_id": 38144,
      "gfx_target_version": 90500,
      "simd_count": 64,
      "max_waves_per_simd": 8,
      "num_shader_engines": 4,
      "num_cu_per_sh": 4,
      "simd_per_cu": 4,
      "wave_front_size": 64,
      "max_slots_scratch_cu": 32,
      "lds_size_kb": )"
              << lds_size_kb << R"(
    }
  },)";
    simulator_json.insert(insert_pos + 1, dbt_guest.str());

    std::ofstream config(config_path);
    if (!config)
      return false;
    config << simulator_json;
    config.close();
    if (!config)
      return false;

    std::ofstream active_config(runtime / "config_path");
    if (!active_config)
      return false;
    active_config << config_path.string() << '\n';
    return active_config.good();
  } catch (const std::filesystem::filesystem_error &) {
    return false;
  }
}

bool read_gpu_id(const std::string &path, uint32_t *gpu_id) {
  FILE *file = fopen(path.c_str(), "r");
  if (!file)
    return false;

  unsigned parsed = 0;
  const int scanned = fscanf(file, "%u", &parsed);
  fclose(file);
  if (scanned != 1)
    return false;

  *gpu_id = static_cast<uint32_t>(parsed);
  return true;
}

std::string topology_nodes_path() {
  for (const char *path : kTopologyNodesPaths) {
    if (access(path, R_OK | X_OK) == 0)
      return path;
  }
  return {};
}

bool guest_gpu_is_visible() {
  std::string nodes_path = topology_nodes_path();
  if (nodes_path.empty())
    return false;

  DIR *dir = opendir(nodes_path.c_str());
  if (!dir)
    return false;

  bool found = false;
  while (auto *entry = readdir(dir)) {
    if (std::strcmp(entry->d_name, ".") == 0 || std::strcmp(entry->d_name, "..") == 0)
      continue;

    uint32_t gpu_id = 0;
    std::string gpu_id_path = nodes_path + "/" + entry->d_name + "/gpu_id";
    if (read_gpu_id(gpu_id_path, &gpu_id) && gpu_id == kGuestGpuId) {
      found = true;
      break;
    }
  }

  closedir(dir);
  return found;
}

bool read_process_aperture_count(int fd, uint32_t *count) {
  kfd_ioctl_get_process_apertures_new_args args{};
  if (ioctl(fd, AMDKFD_IOC_GET_PROCESS_APERTURES_NEW, &args) != 0)
    return false;

  *count = args.num_of_nodes;
  return true;
}

int child_process() {
  int fd = open(kKfdPath, O_RDWR | O_CLOEXEC);
  if (fd < 0)
    return 2;

  const bool visible = guest_gpu_is_visible();
  close(fd);
  return visible ? 0 : 3;
}

int child_reset_process(int inherited_fd, uint32_t expected_reopened_count) {
  uint32_t inherited_count = 0;
  if (read_process_aperture_count(inherited_fd, &inherited_count))
    return 7;

  int fd = open(kKfdPath, O_RDWR | O_CLOEXEC);
  if (fd < 0)
    return 2;

  uint32_t reopened_count = 0;
  const bool count_ok = read_process_aperture_count(fd, &reopened_count);
  const bool visible_after_reopen = guest_gpu_is_visible();
  close(fd);

  if (!count_ok)
    return 5;
  if (reopened_count < expected_reopened_count)
    return 6;
  return visible_after_reopen ? 0 : 3;
}

TEST(GuestKfdMultiprocessTest, ForkedChildrenDoNotRemoveParentOverlay) {
  if (access(kKfdPath, R_OK | W_OK) != 0)
    GTEST_SKIP() << kKfdPath << " is not available: " << std::strerror(errno);

  int parent_fd = open(kKfdPath, O_RDWR | O_CLOEXEC);
  ASSERT_GE(parent_fd, 0);
  if (!guest_gpu_is_visible()) {
    close(parent_fd);
    FAIL() << "guest GPU overlay is not visible";
  }

  std::vector<pid_t> children;
  children.reserve(kChildCount);
  for (int i = 0; i < kChildCount; ++i) {
    pid_t pid = fork();
    ASSERT_GE(pid, 0) << "fork failed: " << std::strerror(errno);

    if (pid == 0)
      _exit(child_process());

    children.push_back(pid);
  }

  for (pid_t pid : children) {
    int status = 0;
    ASSERT_EQ(waitpid(pid, &status, 0), pid);
    ASSERT_TRUE(WIFEXITED(status)) << "child " << pid << " did not exit normally";
    EXPECT_EQ(WEXITSTATUS(status), 0) << "child " << pid << " failed";
  }

  EXPECT_TRUE(guest_gpu_is_visible()) << "parent guest topology disappeared after forked children";
  close(parent_fd);
}

TEST(GuestKfdMultiprocessTest, ForkedChildDropsInheritedGuestDriverBeforeReopen) {
  if (access(kKfdPath, R_OK | W_OK) != 0)
    GTEST_SKIP() << kKfdPath << " is not available: " << std::strerror(errno);

  int parent_fd = open(kKfdPath, O_RDWR | O_CLOEXEC);
  ASSERT_GE(parent_fd, 0);
  if (!guest_gpu_is_visible()) {
    close(parent_fd);
    FAIL() << "guest GPU overlay is not visible";
  }

  uint32_t parent_count_before = 0;
  ASSERT_TRUE(read_process_aperture_count(parent_fd, &parent_count_before));

  pid_t pid = fork();
  ASSERT_GE(pid, 0) << "fork failed: " << std::strerror(errno);
  if (pid == 0)
    _exit(child_reset_process(parent_fd, parent_count_before));

  int status = 0;
  ASSERT_EQ(waitpid(pid, &status, 0), pid);
  ASSERT_TRUE(WIFEXITED(status)) << "child " << pid << " did not exit normally";
  EXPECT_EQ(WEXITSTATUS(status), 0) << "child " << pid << " did not reset and reopen cleanly";

  uint32_t parent_count_after = 0;
  EXPECT_TRUE(read_process_aperture_count(parent_fd, &parent_count_after));
  EXPECT_EQ(parent_count_after, parent_count_before);
  EXPECT_TRUE(guest_gpu_is_visible()) << "parent guest topology disappeared after child reset";
  close(parent_fd);
}

TEST(GuestKfdConcurrencyTest, ConcurrentOpenIoctlAndSysfsRedirect) {
  if (access(kKfdPath, R_OK | W_OK) != 0)
    GTEST_SKIP() << kKfdPath << " is not available: " << std::strerror(errno);

  int probe_fd = open(kKfdPath, O_RDWR | O_CLOEXEC);
  ASSERT_GE(probe_fd, 0);
  if (!guest_gpu_is_visible()) {
    close(probe_fd);
    FAIL() << "guest GPU overlay is not visible";
  }
  close(probe_fd);

  std::atomic<int> open_failures{0};
  std::atomic<int> ioctl_failures{0};
  std::atomic<int> sysfs_failures{0};
  std::vector<std::thread> threads;
  threads.reserve(kStressThreadCount);
  for (int thread_idx = 0; thread_idx < kStressThreadCount; ++thread_idx) {
    threads.emplace_back([&]() {
      for (int iter = 0; iter < kStressIterations; ++iter) {
        int fd = open(kKfdPath, O_RDWR | O_CLOEXEC);
        if (fd < 0) {
          open_failures.fetch_add(1, std::memory_order_relaxed);
          continue;
        }

        kfd_ioctl_get_version_args version{};
        if (ioctl(fd, AMDKFD_IOC_GET_VERSION, &version) != 0)
          ioctl_failures.fetch_add(1, std::memory_order_relaxed);
        if (!guest_gpu_is_visible())
          sysfs_failures.fetch_add(1, std::memory_order_relaxed);
        close(fd);
      }
    });
  }

  for (auto &thread : threads)
    thread.join();
  EXPECT_EQ(open_failures.load(std::memory_order_relaxed), 0);
  EXPECT_EQ(ioctl_failures.load(std::memory_order_relaxed), 0);
  EXPECT_EQ(sysfs_failures.load(std::memory_order_relaxed), 0);
}

TEST(GuestKfdMemoryTest, GuestAllocationMmapOffsetIsRejected) {
  if (access(kKfdPath, R_OK | W_OK) != 0)
    GTEST_SKIP() << kKfdPath << " is not available: " << std::strerror(errno);

  int fd = open(kKfdPath, O_RDWR | O_CLOEXEC);
  ASSERT_GE(fd, 0);
  if (!guest_gpu_is_visible()) {
    close(fd);
    FAIL() << "guest GPU overlay is not visible";
  }

  kfd_ioctl_alloc_memory_of_gpu_args alloc{};
  alloc.va_addr = kAllocVa;
  alloc.size = kAllocSize;
  alloc.gpu_id = kGuestGpuId;
  alloc.flags = KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE;
  ASSERT_EQ(ioctl(fd, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &alloc), 0) << std::strerror(errno);
  EXPECT_NE(alloc.handle, 0u);
  EXPECT_NE(alloc.mmap_offset, 0u);

  errno = 0;
  void *mapped = mmap(nullptr, kAllocSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                      static_cast<off_t>(alloc.mmap_offset));
  if (mapped != MAP_FAILED) {
    munmap(mapped, kAllocSize);
    ADD_FAILURE() << "guest synthetic allocation mmap unexpectedly succeeded";
  } else {
    EXPECT_EQ(errno, ENODEV);
  }

  kfd_ioctl_free_memory_of_gpu_args free_args{};
  free_args.handle = alloc.handle;
  EXPECT_EQ(ioctl(fd, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &free_args), 0) << std::strerror(errno);
  close(fd);
}

TEST(GuestKfdFailureTest, NonexistentSimulatorConfigFailsCleanly) {
  const auto dir = config_handoff_dir();
  ASSERT_TRUE(dir.has_value());
  const std::string nonexistent = (*dir / "nonexistent_simulator_config.json").string();
  ASSERT_TRUE(install_inline_dbt_config(R"({"max_ticks": 1})", "gfx942", 64, nonexistent));

  errno = 0;
  const int fd = open(kKfdPath, O_RDWR | O_CLOEXEC);
  EXPECT_EQ(fd, -1);
  EXPECT_EQ(errno, ENODEV);
  if (fd >= 0)
    close(fd);
}

TEST(GuestKfdFailureTest, SimulatorHostIsaMismatchFailsCleanly) {
  const auto simulator_json = read_active_config_json();
  ASSERT_TRUE(simulator_json.has_value());
  ASSERT_TRUE(install_inline_dbt_config(*simulator_json, "gfx940", 64));

  errno = 0;
  const int fd = open(kKfdPath, O_RDWR | O_CLOEXEC);
  EXPECT_EQ(fd, -1);
  EXPECT_EQ(errno, ENODEV);
  if (fd >= 0)
    close(fd);
}

TEST(GuestKfdFailureTest, SimulatorOversizedLdsFailsCleanly) {
  const auto simulator_json = read_active_config_json();
  ASSERT_TRUE(simulator_json.has_value());
  ASSERT_TRUE(install_inline_dbt_config(*simulator_json, "gfx942", 65));

  errno = 0;
  const int fd = open(kKfdPath, O_RDWR | O_CLOEXEC);
  EXPECT_EQ(fd, -1);
  EXPECT_EQ(errno, ENODEV);
  if (fd >= 0)
    close(fd);
}

// Overwriting the interposer's hidden real /dev/kfd fd number via dup2 must not
// leave a stale KFD classification behind. In guest/DBT mode open("/dev/kfd")
// returns an app-facing dup while the real fd stays internal; if some dup2/dup3
// target reuses that hidden number, the interposer must stop routing KFD ioctls
// on it (GuestKfd::invalidate_primary_fd clears real_kfd_fd_ + ready_), and app
// dups plus fresh opens must keep working via a lazily reopened real fd.
//
// The hidden fd number is not directly observable, so this brute-forces the
// candidate space: it dup2's a pipe end over every low fd number (skipping the
// ones the test itself owns). Whichever number was the hidden real fd gets
// overwritten; the invariant under test is that KFD routing stays correct
// regardless of which number that was.
TEST(GuestKfdFdReuseTest, OverwritingHiddenRealFdKeepsRoutingCorrect) {
  if (access(kKfdPath, R_OK | W_OK) != 0)
    GTEST_SKIP() << kKfdPath << " is not available: " << std::strerror(errno);

  auto version_ok = [](int f) {
    kfd_ioctl_get_version_args version{};
    return ioctl(f, AMDKFD_IOC_GET_VERSION, &version) == 0;
  };

  int app_fd = open(kKfdPath, O_RDWR | O_CLOEXEC);
  ASSERT_GE(app_fd, 0);
  if (!guest_gpu_is_visible()) {
    close(app_fd);
    FAIL() << "guest GPU overlay is not visible";
  }
  ASSERT_TRUE(version_ok(app_fd));

  // A second app-facing dup holds the guest process open across the overwrite.
  int keeper = dup(app_fd);
  ASSERT_GE(keeper, 0);
  ASSERT_TRUE(version_ok(keeper));

  int pipefd[2];
  ASSERT_EQ(pipe(pipefd), 0);

  // Overwrite every low fd number except the ones this test owns. Whichever
  // number is the interposer's hidden real /dev/kfd fd gets clobbered here,
  // driving invalidate_primary_fd() on that backend.
  constexpr int kMaxProbeFd = 64;
  for (int target = 3; target < kMaxProbeFd; ++target) {
    if (target == app_fd || target == keeper || target == pipefd[0] || target == pipefd[1])
      continue;
    // dup2 onto an unopened number just creates a new dup of the pipe read end;
    // onto an open number it atomically closes the old one first. Either way the
    // target now names the pipe, not KFD.
    if (dup2(pipefd[0], target) != target)
      continue;
    // The overwritten number must never route KFD ioctls.
    EXPECT_FALSE(version_ok(target)) << "fd " << target << " still routed KFD after dup2 overwrite";
    close(target);
  }

  // The surviving app dup must still route KFD, and a fresh open must succeed
  // (the interposer lazily reopens the hidden real fd if it was the clobbered
  // number), still seeing the guest overlay.
  EXPECT_TRUE(version_ok(keeper)) << "keeper dup stopped routing after hidden-fd overwrite";

  int reopened = open(kKfdPath, O_RDWR | O_CLOEXEC);
  ASSERT_GE(reopened, 0) << "reopen after hidden-fd overwrite failed: " << std::strerror(errno);
  EXPECT_TRUE(version_ok(reopened));
  EXPECT_TRUE(guest_gpu_is_visible());

  close(reopened);
  close(keeper);
  close(app_fd);
  close(pipefd[0]);
  close(pipefd[1]);
}

} // namespace
