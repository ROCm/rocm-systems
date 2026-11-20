// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Validates the amdgpu version split, the DKMS and amdgpu-kmod package lookups, and how a package
// is matched to the loaded module for amdsmi_get_gpu_driver_info_v2(). No GPU required.

#include <fcntl.h>
#include <gtest/gtest.h>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "amd_smi/impl/amd_smi_utils.h"
#include "test_fixture_utils.h"

namespace {

namespace fs = std::filesystem;
using std::chrono_literals::operator""ms;
using std::chrono_literals::operator""s;

constexpr auto kPackageName = "amdgpu";
constexpr auto kOtherPackageName = "not-amdgpu";
// Above the cap on a dkms.conf, so reading it must be refused.
constexpr size_t kOversizedDkmsConf = (2U * 1024U * 1024U);
// Parks the saved stdin and stdout above any descriptor the lookup can open.
constexpr int kSavedFdMin = 100;
constexpr auto kEndWaitLimit = 20s;
constexpr auto kEndWaitStep = 10ms;
constexpr auto kVersionA = "6.19.13-2370380.24.04";
constexpr auto kVersionB = "6.19.20-2450390.24.04";
constexpr auto kKernelRelease = "6.8.0-124-generic";
constexpr auto kKernelMachine = "x86_64";
constexpr auto kKmodRelease = "5.14.0-611.el9.x86_64";
constexpr auto kNoExecSkip = "the temp directory is mounted noexec";

using Versions = std::vector<std::string>;

// The directory of a fixture. Throws, which fails the test, when the fixture has none: its paths
// would then be relative to the working directory, or to /.
auto RootOf(const FakeSysfsTree& tree) -> const std::string& {
  if (tree.root().empty()) {
    throw std::runtime_error{"the fixture has no temp directory"};
  }
  return tree.root();
}

// A DKMS tree like /var/lib/dkms/amdgpu, with the package sources beside it, in a temp directory.
class DkmsTree {
 public:
  DkmsTree() : dkms_root_{fs::path{RootOf(tree_)} / "dkms"} { fs::create_directories(dkms_root_); }

  auto dkms_root() const -> std::string { return dkms_root_.string(); }

  auto source_tree(const std::string& version) const -> fs::path {
    return fs::path{tree_.root()} / "usr_src" / ("amdgpu-" + version);
  }

  // <dkms_root>/<version>/source -> the source tree, whose dkms.conf names `conf_name` and
  // `conf_version`. An empty `conf_name` leaves dkms.conf out.
  auto AddPackage(const std::string& version, const std::string& conf_name = kPackageName,
                  const std::string& conf_version = {}) -> void {
    const auto version_dir = dkms_root_ / version;
    const auto source = source_tree(version);
    fs::create_directories(version_dir);
    fs::create_directories(source);
    if (!conf_name.empty()) {
      tree_.WriteFile(fs::relative(source / "amd/dkms/dkms.conf", tree_.root()).string(),
                      "PACKAGE_NAME=\"" + conf_name + "\"\nPACKAGE_VERSION=\"" +
                          (conf_version.empty() ? version : conf_version) + "\"\n");
      fs::create_symlink("amd/dkms/dkms.conf", source / "dkms.conf");
    }
    fs::create_directory_symlink(source, version_dir / "source");
  }

  // <dkms_root>/kernel-<release>-<machine> -> `target`, which DKMS writes relative to the root.
  auto AddKernelSymlink(const fs::path& target) -> void {
    fs::create_directories(dkms_root_ / target);
    fs::create_directory_symlink(
        target, dkms_root_ / (std::string{"kernel-"} + kKernelRelease + "-" + kKernelMachine));
  }

  auto kernel_dir(const std::string& version) const -> fs::path {
    return fs::path{version} / kKernelRelease / kKernelMachine;
  }

 private:
  FakeSysfsTree tree_;
  fs::path dkms_root_;
};

auto ActiveDkmsVersion(const DkmsTree& tree, std::string* version) -> amdsmi_status_t {
  return smi_amdgpu_get_active_dkms_version(tree.dkms_root(), kKernelRelease, kKernelMachine,
                                            version);
}

struct ChildLookup {
  bool has_returned;
  bool is_rejected;
};

// Runs the lookup in a child that is ended when it does not return in time, so a reader that
// blocks on a FIFO fails the test instead of hanging the suite.
auto LookUpInChild(const DkmsTree& tree) -> ChildLookup {
  const auto child = fork();
  if (child == 0) {
    auto version = std::string{};
    const auto status = ActiveDkmsVersion(tree, &version);
    _exit(((status == AMDSMI_STATUS_NOT_SUPPORTED) && version.empty()) ? 0 : 1);
  }
  auto child_status = 0;
  auto has_returned = false;
  for (auto waited = 0ms; (!has_returned) && (waited < kEndWaitLimit); waited += kEndWaitStep) {
    has_returned = (waitpid(child, &child_status, WNOHANG) == child);
    if (!has_returned) {
      std::this_thread::sleep_for(kEndWaitStep);
    }
  }
  if (!has_returned) {
    kill(child, SIGKILL);
    waitpid(child, &child_status, 0);
  }
  return {has_returned, (WIFEXITED(child_status)) && (WEXITSTATUS(child_status) == 0)};
}

// What uname reports on the host the DkmsTree is built for.
auto RunningKernel() -> utsname {
  auto uts = utsname{};
  std::strcpy(uts.release, kKernelRelease);
  std::strcpy(uts.machine, kKernelMachine);
  return uts;
}

// Stands in for the amdgpu-kmod query, and records how it was asked.
struct KmodLookup {
  explicit KmodLookup(Versions kmod_versions = {}) : versions{std::move(kmod_versions)} {}

  auto AsFunction() {
    return [this](std::string_view asked_release) {
      ++calls;
      release = asked_release;
      return versions;
    };
  }

  Versions versions;
  int calls = 0;
  std::string release;
};

auto OpenFdCount() -> std::ptrdiff_t {
  return std::distance(fs::directory_iterator{"/proc/self/fd"}, fs::directory_iterator{});
}

// What the stand-in rpm names as its database directory, and what is there.
enum class RpmDatabase {
  kSqlite,
  kBerkeleyDb,
  kNdb,
  kOtherFileOnly,
  kEmptyDirectory,
  kNotADirectory,
  kMissing,
  kRelativePath,  // a database, named relative to the working directory
};

// A script that stands in for rpm. It names <root>/db as its database, and runs `body` only for
// the query the library is expected to make, so a wrong argument list fails the test. `eval_body`
// runs before it names its database.
class FakeRpm {
 public:
  explicit FakeRpm(const std::string& body, RpmDatabase database = RpmDatabase::kSqlite,
                   const std::string& eval_body = {}) {
    // Owner-only before anything is written: the tests run this script, in CI as root.
    fs::permissions(RootOf(tree_), fs::perms::owner_all);
    const auto db = fs::path{tree_.root()} / "db";
    switch (database) {
      case RpmDatabase::kSqlite:
      case RpmDatabase::kRelativePath:
        tree_.WriteFile("db/rpmdb.sqlite", "");
        break;
      case RpmDatabase::kBerkeleyDb:
        tree_.WriteFile("db/Packages", "");
        break;
      case RpmDatabase::kNdb:
        tree_.WriteFile("db/Packages.db", "");
        break;
      case RpmDatabase::kOtherFileOnly:
        tree_.WriteFile("db/.rpm.lock", "");
        break;
      case RpmDatabase::kEmptyDirectory:
        fs::create_directories(db);
        break;
      case RpmDatabase::kNotADirectory:
        tree_.WriteFile("db", "not a database\n");
        break;
      case RpmDatabase::kMissing:
        break;
    }
    const auto named = (database == RpmDatabase::kRelativePath)
                           ? fs::relative(db, fs::current_path()).string()
                           : db.string();
    auto script = std::string{"#!/bin/sh\n"};
    script += "if [ \"$#\" = 2 ] && [ \"$1\" = --eval ] && [ \"$2\" = '%{_dbpath}' ]; then\n";
    script += eval_body + "  echo '" + named + "'\n  exit 0\nfi\n";
    script += "[ \"$#\" = 4 ] && [ \"$1\" = -qa ] && [ \"$2\" = --qf ] && ";
    script += "[ \"$3\" = '%{VERSION}-%{RELEASE}\\n' ] && ";
    script += std::string{"[ \"$4\" = amdgpu-kmod-"} + kKmodRelease + " ] || exit 9\n";
    script += ": > \"$0.queried\"\n" + body;
    tree_.WriteFile("rpm", script);
    tree_.SetPermissions("rpm", 0700);
  }

  auto path() const -> std::string { return tree_.root() + "/rpm"; }

  auto runnable() const -> bool { return access(path().c_str(), X_OK) == 0; }

  // Whether the script left the file <path>.<note>. It leaves "queried" when asked for the package.
  auto left(const std::string& note) const -> bool { return fs::exists(path() + "." + note); }

  // The process ID a body wrote with `echo $$ > "$0.pid"`, or 0.
  auto pid() const -> pid_t {
    auto pid = pid_t{0};
    std::ifstream{path() + ".pid"} >> pid;
    return pid;
  }

 private:
  FakeSysfsTree tree_;
};

constexpr auto kPerlPath = "/usr/bin/perl";

// A stand-in rpm that is no shell script, since a shell sets signal masks for itself. Writes
// <path>.sig with SigBlk and SigIgn as read from /proc/self/status.
class SignalProbeRpm {
 public:
  SignalProbeRpm() {
    fs::permissions(RootOf(tree_), fs::perms::owner_all);
    tree_.WriteFile("db/rpmdb.sqlite", "");
    const auto db = (fs::path{tree_.root()} / "db").string();
    const auto sig_file = path() + ".sig";
    auto script = std::string{"#!"} + kPerlPath + "\n";
    script += "if ($ARGV[0] eq \"--eval\") { print \"" + db + "\\n\"; exit 0 }\n";
    script += "open(my $in, \"<\", \"/proc/self/status\") or exit 1;\n";
    script += "open(my $out, \">\", \"" + sig_file + "\") or exit 1;\n";
    script += "while (<$in>) { print $out \"$1: $2\\n\" if /^(Sig(?:Blk|Ign)):\\s*(\\S+)/ }\n";
    script += "print \"6.16.13.000000-2412345.1\\n\";\n";
    tree_.WriteFile("rpm", script);
    tree_.SetPermissions("rpm", 0700);
  }

  auto path() const -> std::string { return tree_.root() + "/rpm"; }

  auto is_runnable() const -> bool {
    return (access(path().c_str(), X_OK) == 0) && (access(kPerlPath, X_OK) == 0);
  }

  // The mask the stand-in found for `name`, "SigBlk" or "SigIgn", or empty when it has not run.
  auto mask(const std::string& name) const -> std::string {
    auto file = std::ifstream{path() + ".sig"};
    for (auto key = std::string{}, value = std::string{}; file >> key >> value;) {
      if (key == name + ":") {
        return value;
      }
    }
    return {};
  }

 private:
  FakeSysfsTree tree_;
};

auto Prints(std::string_view text, int exit_code = 0) -> std::string {
  return "printf '%s' '" + std::string{text} + "'\nexit " + std::to_string(exit_code) + "\n";
}

auto KmodVersions(const FakeRpm& rpm, Versions* versions) -> amdsmi_status_t {
  return smi_amdgpu_get_kmod_versions(rpm.path(), kKmodRelease, versions);
}

// Ends a stand-in rpm that outlived its test. Only a child that is still running is signalled:
// once it has been reaped, its process ID can belong to another process.
auto EndProcess(pid_t pid) -> void {
  if ((pid > 0) && (waitpid(pid, nullptr, WNOHANG) == 0) && (kill(pid, SIGKILL) == 0)) {
    waitpid(pid, nullptr, 0);
  }
  // The library holds a descriptor for a late rpm until its next lookup, which lets go of it.
  auto none = std::vector<std::string>{};
  smi_amdgpu_get_kmod_versions("/nonexistent/amdsmi/rpm", kKmodRelease, &none);
}

// Looks up the kmod versions through a stand-in rpm that runs `setup` and then stays for twice
// the deadline before it ends by itself. A stand-in that was still starting at the deadline is
// tried again with a longer one, so a slow machine does not change what is tested. Null when the
// stand-in cannot be run at all.
auto LookUpWithRpmThatDoesNotExit(const std::string& setup) -> std::unique_ptr<FakeRpm> {
  for (auto timeout = 100ms;; timeout *= 2) {
    // One sleep: on a busy machine a loop of short ones takes far longer than their sum.
    const auto nap_ms = (2 * timeout + 200ms).count();
    const auto nap =
        std::to_string(nap_ms / 1000) + "." + std::to_string(1000 + (nap_ms % 1000)).substr(1);
    auto rpm = std::make_unique<FakeRpm>("echo $$ > \"$0.pid\"\n" + setup +
                                         ": > \"$0.ready\"\nsleep " + nap + "\n: > \"$0.ended\"\n");
    if (!rpm->runnable()) {
      return nullptr;
    }

    auto versions = Versions{"stale"};
    const auto start = std::chrono::steady_clock::now();
    EXPECT_EQ(smi_amdgpu_get_kmod_versions(rpm->path(), kKmodRelease, &versions, timeout),
              AMDSMI_STATUS_NOT_SUPPORTED);
    EXPECT_LT(std::chrono::steady_clock::now() - start, timeout + 2s);
    EXPECT_TRUE(versions.empty());
    if (rpm->left("ready") || (timeout >= 3s)) {
      EXPECT_TRUE(rpm->left("ready"))
          << "the stand-in rpm had not started after " << timeout.count() << " ms";
      return rpm;
    }
    EndProcess(rpm->pid());
  }
}

// Whether a stand-in rpm that the lookup left running ended by itself. Reaps it either way.
auto EndedByItself(const FakeRpm& rpm) -> bool {
  for (auto waited = 0ms; (!rpm.left("ended")) && (waited < 20s); waited += 10ms) {
    std::this_thread::sleep_for(10ms);
  }
  const auto ended = rpm.left("ended");
  EndProcess(rpm.pid());
  return ended;
}

constexpr auto kFoundByRpm = "6.16.13.000000-2412345.1\n";

// How many times the stand-in made by CountingRpm was asked for the package.
auto QueryCount(const FakeRpm& rpm) -> std::ptrdiff_t {
  auto lines = std::ifstream{rpm.path() + ".count"};
  return std::count(std::istreambuf_iterator<char>{lines}, std::istreambuf_iterator<char>{}, '\n');
}

// Prints the package once its file <path>.ok exists, and fails until then.
auto CountingRpm() -> std::unique_ptr<FakeRpm> {
  return std::make_unique<FakeRpm>("echo x >> \"$0.count\"\n[ -e \"$0.ok\" ] || exit 1\n" +
                                   Prints(kFoundByRpm));
}

auto MakeAnswerable(const FakeRpm& rpm) -> void { std::ofstream{rpm.path() + ".ok"}; }

// Whether a stand-in rpm that the lookup left running ended by itself. Leaves it to be reaped.
auto HasEndedWithoutReaping(const FakeRpm& rpm) -> bool {
  for (auto waited = 0ms; (!rpm.left("ended")) && (waited < kEndWaitLimit);
       waited += kEndWaitStep) {
    std::this_thread::sleep_for(kEndWaitStep);
  }
  return rpm.left("ended");
}

// Waits until the process of a stand-in rpm has ended, whether or not it was reaped.
auto WaitUntilEnded(pid_t pid) -> void {
  const auto stat_path = "/proc/" + std::to_string(pid) + "/stat";
  for (auto waited = 0ms; waited < kEndWaitLimit; waited += kEndWaitStep) {
    auto stat = std::string{};
    std::getline(std::ifstream{stat_path}, stat);
    if ((stat.empty()) || (stat.find(") Z") != std::string::npos)) {
      return;
    }
    std::this_thread::sleep_for(kEndWaitStep);
  }
}

#if defined(SYS_pidfd_open) && (defined(__x86_64__) || defined(__aarch64__))
// Makes pidfd_open fail with ENOSYS for this process and its children, as on a kernel before 5.3
// or under a seccomp profile that does not list it. It cannot be undone, so only a child calls it.
auto was_pidfd_open_blocked() -> bool {
#if defined(__x86_64__)
  constexpr auto kArch = AUDIT_ARCH_X86_64;
#else
  constexpr auto kArch = AUDIT_ARCH_AARCH64;
#endif
  sock_filter program[] = {
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, arch)),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, kArch, 1, 0),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(seccomp_data, nr)),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_pidfd_open, 0, 1),
      BPF_STMT(BPF_RET | BPF_K, (SECCOMP_RET_ERRNO | ENOSYS)),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
  };
  auto filter = sock_fprog{static_cast<unsigned short>(std::size(program)), program};
  return (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) == 0) &&
         (prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &filter) == 0);
}
#else
auto was_pidfd_open_blocked() -> bool { return false; }
#endif

constexpr int kNoSeccompExit = 10;
constexpr int kLateRpmNotLateExit = 11;
constexpr int kLateRpmNeverNoticedExit = 12;
constexpr int kDescriptorLeakExit = 13;

// Runs in the child of the test that blocks pidfd_open. Returns 0 when a late rpm without a pidfd
// is noticed once it ended and leaves no pipe open.
auto LookUpLateRpmWithoutPidfd(const FakeRpm& slow, const FakeRpm& good) -> int {
  if (!was_pidfd_open_blocked()) {
    return kNoSeccompExit;
  }
  const auto open_fds = OpenFdCount();
  auto versions = Versions{};
  if (smi_amdgpu_get_kmod_versions(slow.path(), kKmodRelease, &versions, 400ms) !=
      AMDSMI_STATUS_NOT_SUPPORTED) {
    return kLateRpmNotLateExit;
  }
  for (auto waited = 0ms; (!slow.left("ended")) && (waited < 3s); waited += kEndWaitStep) {
    std::this_thread::sleep_for(kEndWaitStep);
  }
  for (auto waited = 0ms; waited < kEndWaitLimit; waited += kEndWaitStep) {
    if (KmodVersions(good, &versions) == AMDSMI_STATUS_SUCCESS) {
      return (OpenFdCount() == open_fds) ? 0 : kDescriptorLeakExit;
    }
    std::this_thread::sleep_for(kEndWaitStep);
  }
  return kLateRpmNeverNoticedExit;
}

}  // namespace

TEST(SystemUnit, DriverVersionsSplitModuleVersionAndAttachDkmsBuild) {
  auto info = amdsmi_driver_info_v2_t{};

  EXPECT_EQ(
      smi_amdgpu_parse_driver_versions("6.19.14.31400000", "6.19.14-2370381.24.04", "", &info),
      AMDSMI_STATUS_SUCCESS);
  EXPECT_STREQ(info.driver_kernel_version, "6.19.14");
  EXPECT_STREQ(info.amdgpu_driver_version, "31400000");
  EXPECT_STREQ(info.driver_build_version, "2370381");
  EXPECT_STREQ(info.driver_full_version, "6.19.14.31400000-2370381");
  EXPECT_EQ(info.driver_package, AMDSMI_DRIVER_PACKAGE_DKMS);
}

TEST(SystemUnit, DriverVersionsReadDkmsBuildWithAnyDistroSuffix) {
  for (const auto* dkms_version : {"6.19.14-2370381", "6.19.14-2370381.el9"}) {
    auto info = amdsmi_driver_info_v2_t{};

    EXPECT_EQ(smi_amdgpu_parse_driver_versions("6.19.14.31400000", dkms_version, "", &info),
              AMDSMI_STATUS_SUCCESS);
    EXPECT_STREQ(info.driver_build_version, "2370381") << dkms_version;
  }
}

TEST(SystemUnit, DriverVersionsWithoutPackageHaveNoBuild) {
  auto info = amdsmi_driver_info_v2_t{};

  EXPECT_EQ(smi_amdgpu_parse_driver_versions("6.19.14.31400000", "", "", &info),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_STREQ(info.driver_kernel_version, "6.19.14");
  EXPECT_STREQ(info.amdgpu_driver_version, "31400000");
  EXPECT_STREQ(info.driver_build_version, "");
  EXPECT_STREQ(info.driver_full_version, "6.19.14.31400000");
  EXPECT_EQ(info.driver_package, AMDSMI_DRIVER_PACKAGE_NONE);
}

TEST(SystemUnit, DriverVersionsTakeKmodBuildWithoutDkmsPackage) {
  auto info = amdsmi_driver_info_v2_t{};

  // amdgpu-kmod modules report A.B.C; the package version carries the amdgpu release.
  EXPECT_EQ(smi_amdgpu_parse_driver_versions("6.16.13", "", "6.16.13.000000-2412345", &info),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_STREQ(info.driver_kernel_version, "6.16.13");
  EXPECT_STREQ(info.amdgpu_driver_version, "");
  EXPECT_STREQ(info.driver_build_version, "2412345");
  EXPECT_STREQ(info.driver_full_version, "6.16.13-2412345");
  EXPECT_EQ(info.driver_package, AMDSMI_DRIVER_PACKAGE_KMOD);
}

TEST(SystemUnit, DriverVersionsPreferDkmsWhenBothPackagesMatch) {
  auto info = amdsmi_driver_info_v2_t{};

  EXPECT_EQ(smi_amdgpu_parse_driver_versions("6.19.14.31400000", "6.19.14-2370381.24.04",
                                             "6.19.14.31400000-2412345", &info),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_STREQ(info.driver_build_version, "2370381");
  EXPECT_EQ(info.driver_package, AMDSMI_DRIVER_PACKAGE_DKMS);
}

TEST(SystemUnit, DriverVersionsTakeKmodWhenDkmsIsForAnotherModule) {
  auto info = amdsmi_driver_info_v2_t{};

  EXPECT_EQ(smi_amdgpu_parse_driver_versions("6.16.13", "7.1.0-2500001.el9",
                                             "6.16.13.000000-2412345", &info),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_STREQ(info.driver_build_version, "2412345");
  EXPECT_STREQ(info.driver_full_version, "6.16.13-2412345");
  EXPECT_EQ(info.driver_package, AMDSMI_DRIVER_PACKAGE_KMOD);
}

TEST(SystemUnit, DriverVersionsIgnorePackagesOfAnotherModule) {
  auto info = amdsmi_driver_info_v2_t{};

  // A module built outside the installed packages is not theirs, and has no build.
  EXPECT_EQ(smi_amdgpu_parse_driver_versions("6.19.4", "6.19.14-2370381.24.04",
                                             "6.16.13.000000-2412345", &info),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_STREQ(info.driver_kernel_version, "6.19.4");
  EXPECT_STREQ(info.driver_build_version, "");
  EXPECT_STREQ(info.driver_full_version, "6.19.4");
  EXPECT_EQ(info.driver_package, AMDSMI_DRIVER_PACKAGE_NONE);
}

TEST(SystemUnit, DriverVersionsIgnorePackagesWhenModuleHasNoVersion) {
  for (const auto* module_version : {"N/A", ""}) {
    auto info = amdsmi_driver_info_v2_t{};

    EXPECT_EQ(smi_amdgpu_parse_driver_versions(module_version, "6.19.14-2370381.24.04",
                                               "6.16.13.000000-2412345", &info),
              AMDSMI_STATUS_SUCCESS);
    EXPECT_STREQ(info.driver_kernel_version, "") << module_version;
    EXPECT_STREQ(info.amdgpu_driver_version, "") << module_version;
    EXPECT_STREQ(info.driver_build_version, "") << module_version;
    EXPECT_STREQ(info.driver_full_version, "") << module_version;
    EXPECT_EQ(info.driver_package, AMDSMI_DRIVER_PACKAGE_NONE) << module_version;
  }
}

TEST(SystemUnit, DriverVersionsPassThroughUnexpectedModuleFormat) {
  auto info = amdsmi_driver_info_v2_t{};

  EXPECT_EQ(
      smi_amdgpu_parse_driver_versions("6.8.0-60-generic", "6.19.14-2370381.24.04", "", &info),
      AMDSMI_STATUS_SUCCESS);
  EXPECT_STREQ(info.driver_kernel_version, "");
  EXPECT_STREQ(info.amdgpu_driver_version, "");
  EXPECT_STREQ(info.driver_build_version, "");
  EXPECT_STREQ(info.driver_full_version, "6.8.0-60-generic");
  EXPECT_EQ(info.driver_package, AMDSMI_DRIVER_PACKAGE_NONE);
}

TEST(SystemUnit, DriverVersionsIgnoreMalformedPackageVersion) {
  auto info = amdsmi_driver_info_v2_t{};

  EXPECT_EQ(smi_amdgpu_parse_driver_versions("6.19.14.31400000", "6.19.14", "6.19.14-rc1", &info),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_STREQ(info.driver_build_version, "");
  EXPECT_STREQ(info.driver_full_version, "6.19.14.31400000");
  EXPECT_EQ(info.driver_package, AMDSMI_DRIVER_PACKAGE_NONE);
}

TEST(SystemUnit, DriverVersionsOverwriteStaleFieldsAndKeepBaseFields) {
  auto info = amdsmi_driver_info_v2_t{};
  std::strcpy(info.driver_version, "6.19.14.31400000");
  std::strcpy(info.driver_date, "2015/01/01 00:00");
  std::strcpy(info.driver_name, "amdgpu");
  std::strcpy(info.driver_kernel_version, "stale");
  std::strcpy(info.amdgpu_driver_version, "stale");
  std::strcpy(info.driver_build_version, "stale");
  std::strcpy(info.driver_full_version, "stale");
  info.driver_package = AMDSMI_DRIVER_PACKAGE_KMOD;

  EXPECT_EQ(smi_amdgpu_parse_driver_versions("N/A", "", "", &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_STREQ(info.driver_version, "6.19.14.31400000");
  EXPECT_STREQ(info.driver_date, "2015/01/01 00:00");
  EXPECT_STREQ(info.driver_name, "amdgpu");
  EXPECT_STREQ(info.driver_kernel_version, "");
  EXPECT_STREQ(info.amdgpu_driver_version, "");
  EXPECT_STREQ(info.driver_build_version, "");
  EXPECT_STREQ(info.driver_full_version, "");
  EXPECT_EQ(info.driver_package, AMDSMI_DRIVER_PACKAGE_NONE);
}

TEST(SystemUnit, DriverVersionsRejectNullOutput) {
  EXPECT_EQ(
      smi_amdgpu_parse_driver_versions("6.19.14.31400000", "6.19.14-2370381.24.04", "", nullptr),
      AMDSMI_STATUS_INVAL);
}

TEST(SystemUnit, DriverVersionsIgnoreVersionsTooLongToParse) {
  // std::regex recurses once per character, so text that cannot fit a field never reaches it.
  const auto limit = size_t{AMDSMI_MAX_STRING_LENGTH};
  const auto prefix = std::string{"6.19.14-2370381."};
  for (const auto length : {limit - 1, limit, size_t{4096}}) {
    auto info = amdsmi_driver_info_v2_t{};
    // The length is in the distribution suffix, so the build fits whenever the version parses.
    const auto dkms_version = prefix + std::string(length - prefix.size(), '4');

    EXPECT_EQ(smi_amdgpu_parse_driver_versions("6.19.14.31400000", dkms_version, "", &info),
              AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(info.driver_package,
              (length < limit) ? AMDSMI_DRIVER_PACKAGE_DKMS : AMDSMI_DRIVER_PACKAGE_NONE)
        << length;
  }

  auto info = amdsmi_driver_info_v2_t{};
  const auto module_version = "6.19.14." + std::string(limit - 8, '3');
  EXPECT_EQ(smi_amdgpu_parse_driver_versions(module_version, "6.19.14-2370381", "", &info),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_STREQ(info.driver_kernel_version, "");
  EXPECT_EQ(info.driver_package, AMDSMI_DRIVER_PACKAGE_NONE);
}

TEST(SystemUnit, DriverVersionsReportNoPackageWhenTheFullVersionDoesNotFit) {
  // driver_full_version is the module version, a dash and the build.
  const auto build = std::string{"2370381"};
  const auto longest_fit = size_t{AMDSMI_MAX_STRING_LENGTH} - 1 - build.size() - 1;
  for (const auto fits : {true, false}) {
    auto info = amdsmi_driver_info_v2_t{};
    const auto module_version =
        "6.19.14." + std::string((fits ? longest_fit : longest_fit + 1) - 8, '3');

    EXPECT_EQ(smi_amdgpu_parse_driver_versions(module_version, "6.19.14-" + build, "", &info),
              AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(info.driver_package, fits ? AMDSMI_DRIVER_PACKAGE_DKMS : AMDSMI_DRIVER_PACKAGE_NONE);
    EXPECT_EQ(std::string{info.driver_build_version}, fits ? build : "");
    EXPECT_EQ(std::string{info.driver_full_version},
              fits ? module_version + "-" + build : module_version);
  }
}

TEST(SystemUnit, DriverLookupReportsDkmsPackageWithoutAskingKmod) {
  auto tree = DkmsTree{};
  tree.AddPackage(kVersionA);
  tree.AddKernelSymlink(tree.kernel_dir(kVersionA));
  auto kmod = KmodLookup{{"6.19.13.000000-2412345"}};
  auto info = amdsmi_driver_info_v2_t{};

  EXPECT_EQ(smi_amdgpu_get_driver_versions("6.19.13.31400000", tree.dkms_root(), RunningKernel(),
                                           kmod.AsFunction(), &info),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(kmod.calls, 0);
  EXPECT_EQ(info.driver_package, AMDSMI_DRIVER_PACKAGE_DKMS);
  EXPECT_STREQ(info.driver_full_version, "6.19.13.31400000-2370380");
}

TEST(SystemUnit, DriverLookupAsksKmodForTheRunningKernelWithoutDkmsPackage) {
  auto tree = DkmsTree{};
  auto kmod = KmodLookup{{"6.16.13.000000-2412345"}};
  auto info = amdsmi_driver_info_v2_t{};

  EXPECT_EQ(smi_amdgpu_get_driver_versions("6.16.13", tree.dkms_root(), RunningKernel(),
                                           kmod.AsFunction(), &info),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(kmod.calls, 1);
  EXPECT_EQ(kmod.release, kKernelRelease);
  EXPECT_EQ(info.driver_package, AMDSMI_DRIVER_PACKAGE_KMOD);
  EXPECT_STREQ(info.driver_full_version, "6.16.13-2412345");
}

TEST(SystemUnit, DriverLookupAsksKmodWhenDkmsPackageIsForAnotherModule) {
  auto tree = DkmsTree{};
  tree.AddPackage(kVersionA);
  tree.AddKernelSymlink(tree.kernel_dir(kVersionA));
  auto kmod = KmodLookup{{"6.16.13.000000-2412345"}};
  auto info = amdsmi_driver_info_v2_t{};

  EXPECT_EQ(smi_amdgpu_get_driver_versions("6.16.13", tree.dkms_root(), RunningKernel(),
                                           kmod.AsFunction(), &info),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(kmod.calls, 1);
  EXPECT_EQ(info.driver_package, AMDSMI_DRIVER_PACKAGE_KMOD);
  EXPECT_STREQ(info.driver_build_version, "2412345");
}

TEST(SystemUnit, DriverLookupTakesTheOnlyKmodVersionOfTheModule) {
  // Beside a version for another module, in either order, and when it is listed twice.
  for (const auto& versions : {Versions{"6.16.6.000000-2400001", "6.16.13.000000-2412345"},
                               Versions{"6.16.13.000000-2412345", "6.16.6.000000-2400001"},
                               Versions{"6.16.13.000000-2412345", "6.16.13.000000-2412345"}}) {
    auto tree = DkmsTree{};
    auto kmod = KmodLookup{versions};
    auto info = amdsmi_driver_info_v2_t{};

    EXPECT_EQ(smi_amdgpu_get_driver_versions("6.16.13", tree.dkms_root(), RunningKernel(),
                                             kmod.AsFunction(), &info),
              AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(info.driver_package, AMDSMI_DRIVER_PACKAGE_KMOD) << versions.front();
    EXPECT_STREQ(info.driver_full_version, "6.16.13-2412345") << versions.front();
  }
}

TEST(SystemUnit, DriverLookupReportsNoKmodPackageWhenTwoBuildsMatchTheModule) {
  // Which of the two the loaded module comes from is not known.
  auto tree = DkmsTree{};
  auto kmod = KmodLookup{{"6.16.13.000000-2412345", "6.16.13.000000-2400001"}};
  auto info = amdsmi_driver_info_v2_t{};

  EXPECT_EQ(smi_amdgpu_get_driver_versions("6.16.13", tree.dkms_root(), RunningKernel(),
                                           kmod.AsFunction(), &info),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.driver_package, AMDSMI_DRIVER_PACKAGE_NONE);
  EXPECT_STREQ(info.driver_build_version, "");
  EXPECT_STREQ(info.driver_full_version, "6.16.13");
}

TEST(SystemUnit, DriverLookupSkipsPackagesWhenModuleHasNoVersion) {
  auto tree = DkmsTree{};
  tree.AddPackage(kVersionA);
  tree.AddKernelSymlink(tree.kernel_dir(kVersionA));
  auto kmod = KmodLookup{{"6.16.13.000000-2412345"}};
  auto info = amdsmi_driver_info_v2_t{};

  EXPECT_EQ(smi_amdgpu_get_driver_versions("N/A", tree.dkms_root(), RunningKernel(),
                                           kmod.AsFunction(), &info),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(kmod.calls, 0);
  EXPECT_EQ(info.driver_package, AMDSMI_DRIVER_PACKAGE_NONE);
  EXPECT_STREQ(info.driver_full_version, "");
}

TEST(SystemUnit, DriverLookupRejectsNullOutput) {
  auto tree = DkmsTree{};
  auto kmod = KmodLookup{};

  EXPECT_EQ(smi_amdgpu_get_driver_versions("6.16.13", tree.dkms_root(), RunningKernel(),
                                           kmod.AsFunction(), nullptr),
            AMDSMI_STATUS_INVAL);
}

TEST(SystemUnit, ActiveDkmsVersionSelectsKernelSymlinkPackage) {
  auto tree = DkmsTree{};
  tree.AddPackage(kVersionA);
  tree.AddPackage(kVersionB);
  tree.AddKernelSymlink(tree.kernel_dir(kVersionA));

  auto active_version = std::string{};
  EXPECT_EQ(ActiveDkmsVersion(tree, &active_version), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(active_version, kVersionA);
}

TEST(SystemUnit, ActiveDkmsVersionSelectsAbsoluteKernelSymlink) {
  auto tree = DkmsTree{};
  tree.AddPackage(kVersionA);
  tree.AddKernelSymlink(fs::path{tree.dkms_root()} / tree.kernel_dir(kVersionA));

  auto active_version = std::string{};
  EXPECT_EQ(ActiveDkmsVersion(tree, &active_version), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(active_version, kVersionA);
}

TEST(SystemUnit, ActiveDkmsVersionRejectsMissingKernelSymlink) {
  auto tree = DkmsTree{};
  tree.AddPackage(kVersionA);

  auto active_version = std::string{"stale"};
  EXPECT_EQ(ActiveDkmsVersion(tree, &active_version), AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(active_version, "");
}

TEST(SystemUnit, ActiveDkmsVersionRejectsKernelSymlinkToUnregisteredPackage) {
  auto tree = DkmsTree{};
  tree.AddPackage(kVersionA);
  tree.AddKernelSymlink(tree.kernel_dir(kVersionB));

  auto active_version = std::string{"stale"};
  EXPECT_EQ(ActiveDkmsVersion(tree, &active_version), AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(active_version, "");
}

TEST(SystemUnit, ActiveDkmsVersionRejectsKernelSymlinkThatIsNotVersionShaped) {
  auto tree = DkmsTree{};
  // Registered like any package, so only its name keeps it from being selected.
  tree.AddPackage("not-a-version");
  tree.AddKernelSymlink(tree.kernel_dir("not-a-version"));

  auto active_version = std::string{"stale"};
  EXPECT_EQ(ActiveDkmsVersion(tree, &active_version), AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(active_version, "");
}

TEST(SystemUnit, ActiveDkmsVersionRejectsKernelSymlinkOutsideDkmsRoot) {
  auto tree = DkmsTree{};
  tree.AddPackage(kVersionA);
  tree.AddKernelSymlink(tree.source_tree(kVersionA));

  auto active_version = std::string{"stale"};
  EXPECT_EQ(ActiveDkmsVersion(tree, &active_version), AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(active_version, "");
}

TEST(SystemUnit, ActiveDkmsVersionRejectsMissingDkmsConf) {
  auto tree = DkmsTree{};
  tree.AddPackage(kVersionA, "");
  tree.AddKernelSymlink(tree.kernel_dir(kVersionA));

  auto active_version = std::string{"stale"};
  EXPECT_EQ(ActiveDkmsVersion(tree, &active_version), AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(active_version, "");
}

TEST(SystemUnit, ActiveDkmsVersionDoesNotBlockOnAFifoNobodyWritesTo) {
  auto tree = DkmsTree{};
  tree.AddPackage(kVersionA, "");
  tree.AddKernelSymlink(tree.kernel_dir(kVersionA));
  ASSERT_EQ(mkfifo((tree.source_tree(kVersionA) / "dkms.conf").c_str(), 0600), 0);

  const auto lookup = LookUpInChild(tree);

  EXPECT_TRUE(lookup.has_returned) << "the lookup blocked on a FIFO";
  EXPECT_TRUE(lookup.is_rejected);
}

TEST(SystemUnit, ActiveDkmsVersionRejectsAFifoThatHoldsADkmsConf) {
  auto tree = DkmsTree{};
  tree.AddPackage(kVersionA, "");
  tree.AddKernelSymlink(tree.kernel_dir(kVersionA));
  const auto fifo = tree.source_tree(kVersionA) / "dkms.conf";
  ASSERT_EQ(mkfifo(fifo.c_str(), 0600), 0);

  // Filled before the lookup, so a reader that skips the type check reads a valid dkms.conf.
  const auto reader = open(fifo.c_str(), O_RDONLY | O_NONBLOCK);
  ASSERT_GE(reader, 0);
  const auto writer = open(fifo.c_str(), O_WRONLY | O_NONBLOCK);
  ASSERT_GE(writer, 0);
  const auto conf =
      std::string{"PACKAGE_NAME=\""} + kPackageName + "\"\nPACKAGE_VERSION=\"" + kVersionA + "\"\n";
  ASSERT_EQ(write(writer, conf.data(), conf.size()), static_cast<ssize_t>(conf.size()));
  close(writer);

  const auto lookup = LookUpInChild(tree);
  close(reader);

  EXPECT_TRUE(lookup.has_returned) << "the lookup blocked on a FIFO";
  EXPECT_TRUE(lookup.is_rejected);
}

TEST(SystemUnit, ActiveDkmsVersionRejectsOversizedDkmsConf) {
  auto tree = DkmsTree{};
  tree.AddPackage(kVersionA);
  tree.AddKernelSymlink(tree.kernel_dir(kVersionA));
  {
    auto conf = std::ofstream{tree.source_tree(kVersionA) / "amd/dkms/dkms.conf", std::ios::app};
    conf << std::string(kOversizedDkmsConf, '#') << "\n";
  }

  auto active_version = std::string{"stale"};
  EXPECT_EQ(ActiveDkmsVersion(tree, &active_version), AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(active_version, "");
}

TEST(SystemUnit, ActiveDkmsVersionRejectsPackageNameMismatch) {
  auto tree = DkmsTree{};
  tree.AddPackage(kVersionA, kOtherPackageName);
  tree.AddKernelSymlink(tree.kernel_dir(kVersionA));

  auto active_version = std::string{"stale"};
  EXPECT_EQ(ActiveDkmsVersion(tree, &active_version), AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(active_version, "");
}

TEST(SystemUnit, ActiveDkmsVersionRejectsPackageVersionMismatch) {
  auto tree = DkmsTree{};
  tree.AddPackage(kVersionA, kPackageName, kVersionB);
  tree.AddKernelSymlink(tree.kernel_dir(kVersionA));

  auto active_version = std::string{"stale"};
  EXPECT_EQ(ActiveDkmsVersion(tree, &active_version), AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(active_version, "");
}

TEST(SystemUnit, ActiveDkmsVersionRejectsEmptyReleaseOrMachine) {
  auto tree = DkmsTree{};
  tree.AddPackage(kVersionA);
  tree.AddKernelSymlink(tree.kernel_dir(kVersionA));
  // What an empty release or machine would look up, so only the check rejects it.
  const auto dkms_root = fs::path{tree.dkms_root()};
  fs::create_directory_symlink(tree.kernel_dir(kVersionA),
                               dkms_root / (std::string{"kernel--"} + kKernelMachine));
  fs::create_directory_symlink(tree.kernel_dir(kVersionA),
                               dkms_root / (std::string{"kernel-"} + kKernelRelease + "-"));

  auto active_version = std::string{"stale"};
  EXPECT_EQ(
      smi_amdgpu_get_active_dkms_version(tree.dkms_root(), "", kKernelMachine, &active_version),
      AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(active_version, "");

  active_version = "stale";
  EXPECT_EQ(
      smi_amdgpu_get_active_dkms_version(tree.dkms_root(), kKernelRelease, "", &active_version),
      AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(active_version, "");
}

TEST(SystemUnit, ActiveDkmsVersionRejectsNullOutput) {
  auto tree = DkmsTree{};

  EXPECT_EQ(ActiveDkmsVersion(tree, nullptr), AMDSMI_STATUS_INVAL);
}

TEST(SystemUnit, KmodVersionReadsBuildFromRpmRelease) {
  const auto rpm = FakeRpm{Prints("6.16.13.000000-2412345.1\n")};
  if (!rpm.runnable()) GTEST_SKIP() << kNoExecSkip;

  auto versions = Versions{"stale"};
  EXPECT_EQ(KmodVersions(rpm, &versions), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(versions, Versions{"6.16.13.000000-2412345"});
  EXPECT_TRUE(rpm.left("queried"));
}

TEST(SystemUnit, KmodVersionLeavesNoDescriptorOpenAfterAnAnswer) {
  const auto rpm = FakeRpm{Prints("6.16.13.000000-2412345.1\n")};
  if (!rpm.runnable()) {
    GTEST_SKIP() << kNoExecSkip;
  }

  // The first lookup is a warm-up: state the library or libc opens lazily is not a leak.
  auto versions = Versions{};
  ASSERT_EQ(KmodVersions(rpm, &versions), AMDSMI_STATUS_SUCCESS);
  const auto open_fds = OpenFdCount();
  EXPECT_EQ(KmodVersions(rpm, &versions), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(OpenFdCount(), open_fds);
}

TEST(SystemUnit, KmodVersionListsEveryInstalledVersion) {
  const auto rpm = FakeRpm{Prints("6.16.13.000000-2412345.el9\n6.16.6.000000-2400001.el9\n")};
  if (!rpm.runnable()) GTEST_SKIP() << kNoExecSkip;

  auto versions = Versions{};
  EXPECT_EQ(KmodVersions(rpm, &versions), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(versions, (Versions{"6.16.13.000000-2412345", "6.16.6.000000-2400001"}));
}

TEST(SystemUnit, KmodVersionListsNothingWhenPackageIsNotInstalled) {
  // rpm -qa exits 0 with no output for a package that is not installed.
  const auto rpm = FakeRpm{Prints("")};
  if (!rpm.runnable()) {
    GTEST_SKIP() << kNoExecSkip;
  }

  auto versions = Versions{"stale"};
  EXPECT_EQ(KmodVersions(rpm, &versions), AMDSMI_STATUS_SUCCESS);
  EXPECT_TRUE(versions.empty());
  EXPECT_TRUE(rpm.left("queried"));
}

TEST(SystemUnit, KmodVersionNotSupportedWhenRpmFailsWithoutOutput) {
  const auto rpm = FakeRpm{Prints("", 1)};
  if (!rpm.runnable()) GTEST_SKIP() << kNoExecSkip;

  auto versions = Versions{"stale"};
  EXPECT_EQ(KmodVersions(rpm, &versions), AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_TRUE(versions.empty());
}

TEST(SystemUnit, KmodVersionNotSupportedWhenRpmFailsAfterPrintingVersion) {
  const auto rpm = FakeRpm{Prints("6.16.13.000000-2412345.1\n", 1)};
  if (!rpm.runnable()) GTEST_SKIP() << kNoExecSkip;

  auto versions = Versions{"stale"};
  EXPECT_EQ(KmodVersions(rpm, &versions), AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_TRUE(versions.empty());
}

TEST(SystemUnit, KmodVersionNotSupportedWhenRpmIsEndedBySignalAfterPrintingVersion) {
  // SIGKILL, so that the signal setup of the test runner cannot keep the stand-in alive.
  const auto rpm = FakeRpm{"printf '%s\\n' '6.16.13.000000-2412345.1'\nkill -KILL $$\n"};
  if (!rpm.runnable()) GTEST_SKIP() << kNoExecSkip;

  auto versions = Versions{"stale"};
  EXPECT_EQ(KmodVersions(rpm, &versions), AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_TRUE(versions.empty());
}

TEST(SystemUnit, KmodVersionNotSupportedForReleaseWithoutBuild) {
  const auto rpm = FakeRpm{Prints("6.16.13.000000-custom.1\n")};
  if (!rpm.runnable()) GTEST_SKIP() << kNoExecSkip;

  auto versions = Versions{"stale"};
  EXPECT_EQ(KmodVersions(rpm, &versions), AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_TRUE(versions.empty());
}

TEST(SystemUnit, KmodVersionNotSupportedWhenAListedVersionIsNotUnderstood) {
  // It could be the package of the loaded module.
  const auto rpm = FakeRpm{Prints("6.16.13.000000-2412345.1\n6.16.13.000000-custom.1\n")};
  if (!rpm.runnable()) GTEST_SKIP() << kNoExecSkip;

  auto versions = Versions{"stale"};
  EXPECT_EQ(KmodVersions(rpm, &versions), AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_TRUE(versions.empty());
}

TEST(SystemUnit, KmodVersionAsksForThePackageWhereRpmHasAPackagesFile) {
  for (const auto database : {RpmDatabase::kSqlite, RpmDatabase::kBerkeleyDb, RpmDatabase::kNdb}) {
    const auto rpm = FakeRpm{Prints("6.16.13.000000-2412345.1\n"), database};
    if (!rpm.runnable()) GTEST_SKIP() << kNoExecSkip;

    auto versions = Versions{};
    EXPECT_EQ(KmodVersions(rpm, &versions), AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(versions, Versions{"6.16.13.000000-2412345"});
    EXPECT_TRUE(rpm.left("queried"));
  }
}

TEST(SystemUnit, KmodVersionDoesNotAskForThePackageWithoutAPackagesFile) {
  // The query would make rpm create a database there.
  for (const auto database :
       {RpmDatabase::kOtherFileOnly, RpmDatabase::kEmptyDirectory, RpmDatabase::kNotADirectory,
        RpmDatabase::kMissing, RpmDatabase::kRelativePath}) {
    const auto rpm = FakeRpm{Prints("6.16.13.000000-2412345.1\n"), database};
    if (!rpm.runnable()) GTEST_SKIP() << kNoExecSkip;

    auto versions = Versions{"stale"};
    EXPECT_EQ(KmodVersions(rpm, &versions), AMDSMI_STATUS_NOT_SUPPORTED);
    EXPECT_TRUE(versions.empty());
    EXPECT_FALSE(rpm.left("queried"));
  }
}

TEST(SystemUnit, KmodVersionGivesRpmAFixedEnvironment) {
  // rpm takes macro files from HOME, and from the working directory when HOME is not set.
  const auto leak = amd::smi::testing::ScopedEnvVar{"AMDSMI_TEST_RPM_ENV_LEAK", "1"};
  const auto rpm = FakeRpm{
      "[ -z \"${AMDSMI_TEST_RPM_ENV_LEAK+set}\" ] && [ \"$LC_ALL\" = C ] && "
      "[ \"$HOME\" = / ] || exit 8\n" +
      Prints("6.16.13.000000-2412345.1\n")};
  if (!rpm.runnable()) GTEST_SKIP() << kNoExecSkip;

  auto versions = Versions{};
  EXPECT_EQ(KmodVersions(rpm, &versions), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(versions, Versions{"6.16.13.000000-2412345"});
}

TEST(SystemUnit, KmodVersionKeepsTheCallersDescriptorsOutOfRpm) {
  // Not close-on-exec, and above the descriptors a shell opens for itself.
  const auto low = open("/dev/null", O_RDONLY);
  const auto inheritable = fcntl(low, F_DUPFD, 100);
  close(low);
  ASSERT_GE(inheritable, 100);
  const auto rpm = FakeRpm{"[ -e \"/proc/$$/fd/" + std::to_string(inheritable) +
                           "\" ] && : > \"$0.inherited\"\n" + Prints("6.16.13.000000-2412345.1\n")};
  const auto runnable = rpm.runnable();
  auto versions = Versions{};
  const auto status = runnable ? KmodVersions(rpm, &versions) : AMDSMI_STATUS_SUCCESS;
  close(inheritable);
  if (!runnable) GTEST_SKIP() << kNoExecSkip;

  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_FALSE(rpm.left("inherited"));
}

TEST(SystemUnit, KmodVersionStartsRpmWithDefaultSignals) {
  const auto rpm = SignalProbeRpm{};
  if (!rpm.is_runnable()) {
    GTEST_SKIP() << kNoExecSkip << ", or " << kPerlPath << " is missing";
  }

  // The caller blocks every signal and ignores two. rpm must start with neither.
  auto blocked = sigset_t{};
  auto previous_mask = sigset_t{};
  sigfillset(&blocked);
  struct sigaction ignore = {};
  struct sigaction previous_pipe = {};
  struct sigaction previous_term = {};
  ignore.sa_handler = SIG_IGN;
  ASSERT_EQ(sigaction(SIGPIPE, &ignore, &previous_pipe), 0);
  ASSERT_EQ(sigaction(SIGTERM, &ignore, &previous_term), 0);
  ASSERT_EQ(pthread_sigmask(SIG_SETMASK, &blocked, &previous_mask), 0);
  auto versions = Versions{};
  const auto status = smi_amdgpu_get_kmod_versions(rpm.path(), kKmodRelease, &versions);
  ASSERT_EQ(pthread_sigmask(SIG_SETMASK, &previous_mask, nullptr), 0);
  ASSERT_EQ(sigaction(SIGTERM, &previous_term, nullptr), 0);
  ASSERT_EQ(sigaction(SIGPIPE, &previous_pipe, nullptr), 0);

  // perl ignores other signals by itself, so only the two the caller ignored are checked.
  constexpr auto kCallerIgnoredBits =
      ((uint64_t{1} << (SIGTERM - 1)) | (uint64_t{1} << (SIGPIPE - 1)));
  ASSERT_FALSE(rpm.mask("SigBlk").empty());
  ASSERT_FALSE(rpm.mask("SigIgn").empty());
  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(std::stoull(rpm.mask("SigBlk"), nullptr, 16), uint64_t{0});
  EXPECT_EQ((std::stoull(rpm.mask("SigIgn"), nullptr, 16) & kCallerIgnoredBits), uint64_t{0});
}

TEST(SystemUnit, KmodVersionReadsRpmWhenStdioIsClosed) {
  const auto rpm = FakeRpm{Prints("6.16.13.000000-2412345.1\n")};
  if (!rpm.runnable()) {
    GTEST_SKIP() << kNoExecSkip;
  }

  // The pipe then lands on descriptors 0 and 1, where the output end must still reach rpm. This
  // fails only on glibc older than 2.29, where dup2(fd, fd) keeps close-on-exec.
  const auto saved_in = fcntl(STDIN_FILENO, F_DUPFD_CLOEXEC, kSavedFdMin);
  const auto saved_out = fcntl(STDOUT_FILENO, F_DUPFD_CLOEXEC, kSavedFdMin);
  ASSERT_GE(saved_in, kSavedFdMin);
  ASSERT_GE(saved_out, kSavedFdMin);
  close(STDIN_FILENO);
  close(STDOUT_FILENO);
  auto versions = Versions{};
  const auto status = KmodVersions(rpm, &versions);
  dup2(saved_in, STDIN_FILENO);
  dup2(saved_out, STDOUT_FILENO);
  close(saved_in);
  close(saved_out);

  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(versions, Versions{"6.16.13.000000-2412345"});
}

#if defined(__GLIBC__) && ((__GLIBC__ > 2) || ((__GLIBC__ == 2) && (__GLIBC_MINOR__ >= 34)))
// Before glibc 2.34 the descriptors are listed in the parent, which this race cannot be closed on.
TEST(SystemUnit, KmodVersionKeepsDescriptorsOpenedDuringTheCallOutOfRpm) {
  // Counts rpm's descriptors, which are the same on each run when none leaks. Without the close
  // action a run leaks only now and then, so this fails by chance within the runs.
  const auto rpm = FakeRpm{"set -- /proc/$$/fd/*\necho $# >> \"$0.fds\"\n" +
                           Prints("6.16.13.000000-2412345.1\n")};
  if (!rpm.runnable()) {
    GTEST_SKIP() << kNoExecSkip;
  }

  auto is_done = std::atomic<bool>{false};
  auto churn = std::thread{[&is_done] {
    while (!is_done) {
      close(open("/dev/null", O_RDONLY));  // not close-on-exec, as a library thread may open
    }
  }};
  constexpr auto kRuns = 300;
  auto versions = Versions{};
  auto status = AMDSMI_STATUS_SUCCESS;
  for (auto run = 0; (run < kRuns) && (status == AMDSMI_STATUS_SUCCESS); ++run) {
    status = KmodVersions(rpm, &versions);
  }
  is_done = true;
  churn.join();

  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  auto counts = std::vector<int>{};
  auto file = std::ifstream{rpm.path() + ".fds"};
  for (auto count = 0; file >> count;) {
    counts.push_back(count);
  }
  ASSERT_EQ(counts.size(), static_cast<size_t>(kRuns));
  const auto lowest = *std::min_element(counts.begin(), counts.end());
  EXPECT_EQ(std::count(counts.begin(), counts.end(), lowest), kRuns)
      << "rpm inherited a descriptor opened during the call";
}
#endif

TEST(SystemUnit, KmodVersionLetsRpmFinishOutputItDoesNotKeep) {
  // More than a pipe holds: rpm would block, or be ended by SIGPIPE, if the rest were left
  // unread. The stand-in notes that it wrote all of it.
  const auto rpm = FakeRpm{
      "printf '%s\\n' '6.16.13.000000-2412345.1'\n"
      "head -c 1000000 /dev/zero && : > \"$0.wrote\"\n"};
  if (!rpm.runnable()) GTEST_SKIP() << kNoExecSkip;

  // An answer that was not kept whole is not used.
  auto versions = Versions{"stale"};
  EXPECT_EQ(smi_amdgpu_get_kmod_versions(rpm.path(), kKmodRelease, &versions, 30s),
            AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_TRUE(versions.empty());
  EXPECT_TRUE(rpm.left("wrote"));
}

TEST(SystemUnit, KmodVersionUsesOnlyAnAnswerThatFitsWhatIsKept) {
  // 25 bytes a row: 100 rows fit the 4096 bytes that are kept, 300 rows do not.
  for (const auto rows : {100, 300}) {
    const auto rpm = FakeRpm{"i=0\nwhile [ \"$i\" -lt " + std::to_string(rows) +
                             " ]; do echo '6.16.13.000000-2412345.1'; i=$((i + 1)); done\n"};
    if (!rpm.runnable()) GTEST_SKIP() << kNoExecSkip;

    auto versions = Versions{"stale"};
    EXPECT_EQ(smi_amdgpu_get_kmod_versions(rpm.path(), kKmodRelease, &versions, 30s),
              (rows == 100) ? AMDSMI_STATUS_SUCCESS : AMDSMI_STATUS_NOT_SUPPORTED)
        << rows;
    EXPECT_EQ(versions.size(), (rows == 100) ? size_t{100} : size_t{0}) << rows;
  }
}

TEST(SystemUnit, KmodVersionGivesBothRpmCommandsOneDeadline) {
  // Each command takes 0.4 s and fits the 0.6 s by itself; both together do not.
  const auto rpm =
      FakeRpm{"echo $$ > \"$0.pid\"\nsleep 0.4\n" + Prints("6.16.13.000000-2412345.1\n"),
              RpmDatabase::kSqlite, "  sleep 0.4\n"};
  if (!rpm.runnable()) GTEST_SKIP() << kNoExecSkip;

  auto versions = Versions{"stale"};
  const auto status = smi_amdgpu_get_kmod_versions(rpm.path(), kKmodRelease, &versions, 600ms);
  EndProcess(rpm.pid());
  if (!rpm.left("queried")) GTEST_SKIP() << "the first command alone took longer than the limit";

  EXPECT_EQ(status, AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_TRUE(versions.empty());
}

TEST(SystemUnit, KmodVersionLeavesRpmThatDoesNotAnswerToEndByItself) {
  // The stand-in notes every signal it can catch. A killed rpm can leave a Berkeley DB rpm
  // database unusable, and at the deadline its process id can belong to another process.
  const auto rpm = LookUpWithRpmThatDoesNotExit("trap ': > \"$0.signalled\"' HUP INT QUIT TERM\n");
  if (rpm == nullptr) GTEST_SKIP() << kNoExecSkip;

  EXPECT_TRUE(EndedByItself(*rpm));
  EXPECT_FALSE(rpm->left("signalled"));
}

TEST(SystemUnit, KmodVersionGivesUpOnRpmThatEndsItsOutputButDoesNotExit) {
  // The version is complete, but the deadline also covers how long rpm takes to exit.
  const auto rpm =
      LookUpWithRpmThatDoesNotExit("printf '%s\\n' '6.16.13.000000-2412345.1'\nexec 1>&-\n");
  if (rpm == nullptr) GTEST_SKIP() << kNoExecSkip;

  EXPECT_TRUE(EndedByItself(*rpm));
}

TEST(SystemUnit, KmodVersionReadsRpmWhenTheHostIgnoresSigchld) {
  const auto rpm = FakeRpm{Prints("6.16.13.000000-2412345.1\n")};
  if (!rpm.runnable()) GTEST_SKIP() << kNoExecSkip;

  // The kernel then reaps rpm itself, and waitpid() finds no child to report on.
  struct sigaction ignore = {};
  struct sigaction previous = {};
  ignore.sa_handler = SIG_IGN;
  ASSERT_EQ(sigaction(SIGCHLD, &ignore, &previous), 0);
  auto versions = Versions{};
  const auto status = KmodVersions(rpm, &versions);
  ASSERT_EQ(sigaction(SIGCHLD, &previous, nullptr), 0);

  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(versions, Versions{"6.16.13.000000-2412345"});
}

TEST(SystemUnit, KmodVersionNotSupportedWhenRpmFailsAndTheHostIgnoresSigchld) {
  // No exit status is left, so empty output must not be taken for "package not installed".
  const auto rpm = FakeRpm{Prints("", 1)};
  if (!rpm.runnable()) {
    GTEST_SKIP() << kNoExecSkip;
  }

  struct sigaction ignore = {};
  struct sigaction previous = {};
  ignore.sa_handler = SIG_IGN;
  ASSERT_EQ(sigaction(SIGCHLD, &ignore, &previous), 0);
  auto versions = Versions{"stale"};
  const auto status = KmodVersions(rpm, &versions);
  ASSERT_EQ(sigaction(SIGCHLD, &previous, nullptr), 0);

  EXPECT_EQ(status, AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_TRUE(versions.empty());
}

TEST(SystemUnit, KmodVersionNotSupportedWhenRpmCannotBeStarted) {
  // A directory cannot be run.
  const auto not_a_program = FakeSysfsTree{};
  const auto open_fds = OpenFdCount();

  auto versions = Versions{"stale"};
  EXPECT_EQ(smi_amdgpu_get_kmod_versions(not_a_program.root(), kKmodRelease, &versions),
            AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_TRUE(versions.empty());
  EXPECT_EQ(OpenFdCount(), open_fds);
}

TEST(SystemUnit, KmodVersionNotSupportedWithoutRpm) {
  auto versions = Versions{"stale"};

  EXPECT_EQ(smi_amdgpu_get_kmod_versions("/nonexistent/amdsmi/rpm", kKmodRelease, &versions),
            AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_TRUE(versions.empty());
}

TEST(SystemUnit, KmodVersionRejectsEmptyRelease) {
  // Without the check the lookup would still fail, but only after it has started rpm.
  const auto rpm = FakeRpm{Prints(kFoundByRpm), RpmDatabase::kSqlite, "  : > \"$0.evaluated\"\n"};
  if (!rpm.runnable()) {
    GTEST_SKIP() << kNoExecSkip;
  }

  auto versions = Versions{"stale"};
  EXPECT_EQ(smi_amdgpu_get_kmod_versions(rpm.path(), "", &versions), AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_TRUE(versions.empty());
  EXPECT_FALSE(rpm.left("evaluated"));
}

TEST(SystemUnit, KmodVersionRejectsNullOutput) {
  EXPECT_EQ(smi_amdgpu_get_kmod_versions(kRpmPath, kKmodRelease, nullptr), AMDSMI_STATUS_INVAL);
}

TEST(SystemUnit, KmodCacheKeepsAFoundAnswer) {
  const auto rpm = CountingRpm();
  if (!rpm->runnable()) {
    GTEST_SKIP() << kNoExecSkip;
  }
  MakeAnswerable(*rpm);
  auto cache = KmodVersionCache{rpm->path(), 0ms};

  EXPECT_EQ(cache.versions(kKmodRelease), Versions{"6.16.13.000000-2412345"});
  EXPECT_EQ(cache.versions(kKmodRelease), Versions{"6.16.13.000000-2412345"});
  EXPECT_EQ(QueryCount(*rpm), 1);
}

TEST(SystemUnit, KmodCacheKeepsTheAnswerThatThePackageIsNotInstalled) {
  const auto rpm = FakeRpm{"echo x >> \"$0.count\"\n" + Prints("")};
  if (!rpm.runnable()) {
    GTEST_SKIP() << kNoExecSkip;
  }
  auto cache = KmodVersionCache{rpm.path(), 0ms};

  EXPECT_TRUE(cache.versions(kKmodRelease).empty());
  EXPECT_TRUE(cache.versions(kKmodRelease).empty());
  EXPECT_EQ(QueryCount(rpm), 1);
}

TEST(SystemUnit, KmodCacheAsksAgainForAFailureOnceTheIntervalHasPassed) {
  const auto rpm = CountingRpm();
  if (!rpm->runnable()) {
    GTEST_SKIP() << kNoExecSkip;
  }
  auto cache = KmodVersionCache{rpm->path(), 0ms};

  EXPECT_TRUE(cache.versions(kKmodRelease).empty());
  MakeAnswerable(*rpm);
  EXPECT_EQ(cache.versions(kKmodRelease), Versions{"6.16.13.000000-2412345"});
  EXPECT_EQ(QueryCount(*rpm), 2);
}

TEST(SystemUnit, KmodCacheKeepsAFailureUntilTheIntervalHasPassed) {
  const auto rpm = CountingRpm();
  if (!rpm->runnable()) {
    GTEST_SKIP() << kNoExecSkip;
  }
  auto cache = KmodVersionCache{rpm->path(), std::chrono::hours{1}};

  EXPECT_TRUE(cache.versions(kKmodRelease).empty());
  MakeAnswerable(*rpm);
  EXPECT_TRUE(cache.versions(kKmodRelease).empty());
  EXPECT_EQ(QueryCount(*rpm), 1);
}

TEST(SystemUnit, KmodCacheAsksAgainForAnotherRelease) {
  const auto rpm = CountingRpm();
  if (!rpm->runnable()) {
    GTEST_SKIP() << kNoExecSkip;
  }
  MakeAnswerable(*rpm);
  auto cache = KmodVersionCache{rpm->path(), std::chrono::hours{1}};

  EXPECT_FALSE(cache.versions(kKmodRelease).empty());
  EXPECT_TRUE(cache.versions("6.8.0-124-generic").empty());
  EXPECT_FALSE(cache.versions(kKmodRelease).empty());
}

TEST(SystemUnit, KmodCacheAsksOnceForCallersThatArriveTogether) {
  const auto rpm = CountingRpm();
  if (!rpm->runnable()) {
    GTEST_SKIP() << kNoExecSkip;
  }
  MakeAnswerable(*rpm);
  auto cache = KmodVersionCache{rpm->path(), std::chrono::hours{1}};

  constexpr size_t kCallers = 8;
  auto answers = std::vector<Versions>(kCallers);
  auto callers = std::vector<std::thread>{};
  for (size_t caller = 0; caller < kCallers; ++caller) {
    callers.emplace_back(
        [&cache, &answers, caller] { answers[caller] = cache.versions(kKmodRelease); });
  }
  for (auto& caller : callers) {
    caller.join();
  }

  for (const auto& answer : answers) {
    EXPECT_EQ(answer, Versions{"6.16.13.000000-2412345"});
  }
  EXPECT_EQ(QueryCount(*rpm), 1);
}

TEST(SystemUnit, KmodVersionStartsNoRpmWhileALateOneRunsAndReapsItAfterwards) {
  const auto late = LookUpWithRpmThatDoesNotExit("");
  if (!late) {
    GTEST_SKIP() << kNoExecSkip;
  }
  const auto late_pid = late->pid();
  const auto rpm = FakeRpm{Prints(kFoundByRpm)};
  if (!rpm.runnable()) {
    EndProcess(late_pid);
    GTEST_SKIP() << kNoExecSkip;
  }

  auto versions = Versions{"stale"};
  EXPECT_EQ(KmodVersions(rpm, &versions), AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_TRUE(versions.empty());
  EXPECT_FALSE(rpm.left("queried"));

  EXPECT_TRUE(HasEndedWithoutReaping(*late));
  WaitUntilEnded(late_pid);
  EXPECT_EQ(KmodVersions(rpm, &versions), AMDSMI_STATUS_SUCCESS);
  EXPECT_FALSE(fs::exists("/proc/" + std::to_string(late_pid))) << "the late rpm was not reaped";
}

TEST(SystemUnit, KmodCacheGivesEveryFirstCallerTheListWhileRpmIsSlow) {
  // The callers arrive while the one lookup runs. None may get an empty answer for it.
  const auto rpm = FakeRpm{"echo x >> \"$0.count\"\nsleep 0.5\n" + Prints(kFoundByRpm)};
  if (!rpm.runnable()) {
    GTEST_SKIP() << kNoExecSkip;
  }
  auto cache = KmodVersionCache{rpm.path(), std::chrono::hours{1}};

  constexpr size_t kCallers = 8;
  auto answers = std::vector<Versions>(kCallers);
  auto callers = std::vector<std::thread>{};
  for (size_t caller = 0; caller < kCallers; ++caller) {
    callers.emplace_back(
        [&cache, &answers, caller] { answers[caller] = cache.versions(kKmodRelease); });
  }
  for (auto& caller : callers) {
    caller.join();
  }

  for (const auto& answer : answers) {
    EXPECT_EQ(answer, Versions{"6.16.13.000000-2412345"});
  }
  EXPECT_EQ(QueryCount(rpm), 1);
}

TEST(SystemUnit, KmodCacheDoesNotHoldACallerThatHasAnAnswerForTheWholeRpmRun) {
  // Fails until <path>.ok exists. Once <path>.slow exists it also takes 2 s.
  const auto rpm = FakeRpm{
      "if [ -e \"$0.slow\" ]; then : > \"$0.started\"; sleep 2; fi\n"
      "[ -e \"$0.ok\" ] || exit 1\n" +
      Prints(kFoundByRpm)};
  if (!rpm.runnable()) {
    GTEST_SKIP() << kNoExecSkip;
  }
  auto cache = KmodVersionCache{rpm.path(), 50ms};
  EXPECT_TRUE(cache.versions(kKmodRelease).empty());
  MakeAnswerable(rpm);
  std::ofstream{rpm.path() + ".slow"};
  std::this_thread::sleep_for(100ms);

  auto refreshed = Versions{};
  auto refresher = std::thread{[&cache, &refreshed] { refreshed = cache.versions(kKmodRelease); }};
  for (auto waited = 0ms; (!rpm.left("started")) && (waited < kEndWaitLimit);
       waited += kEndWaitStep) {
    std::this_thread::sleep_for(kEndWaitStep);
  }
  const auto start = std::chrono::steady_clock::now();
  const auto stale = cache.versions(kKmodRelease);
  const auto waited = std::chrono::steady_clock::now() - start;
  refresher.join();

  EXPECT_TRUE(stale.empty());
  EXPECT_LT(waited, 1s);
  EXPECT_EQ(refreshed, Versions{"6.16.13.000000-2412345"});
}

TEST(SystemUnit, KmodVersionLeavesALateRpmToWriteItsOutput) {
  // The stand-in writes after the deadline. If the library had closed its end of the pipe, that
  // write would end it with SIGPIPE before it notes that it ended.
  const auto rpm = FakeRpm{"echo $$ > \"$0.pid\"\nsleep 1.5\nprintf '%s' '" +
                           std::string{kFoundByRpm} + "'\n: > \"$0.ended\"\n"};
  if (!rpm.runnable()) {
    GTEST_SKIP() << kNoExecSkip;
  }

  auto versions = Versions{};
  EXPECT_EQ(smi_amdgpu_get_kmod_versions(rpm.path(), kKmodRelease, &versions, 1s),
            AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_TRUE(EndedByItself(rpm));
}

TEST(SystemUnit, KmodVersionStartsNoRpmWhenTheDeadlineHasPassed) {
  const auto rpm = FakeRpm{Prints(kFoundByRpm), RpmDatabase::kSqlite, "  : > \"$0.evaluated\"\n"};
  if (!rpm.runnable()) {
    GTEST_SKIP() << kNoExecSkip;
  }

  // A late rpm from an earlier test would end the lookup before the deadline check is reached.
  EndProcess(0);
  auto versions = Versions{"stale"};
  EXPECT_EQ(smi_amdgpu_get_kmod_versions(rpm.path(), kKmodRelease, &versions, 0ms),
            AMDSMI_STATUS_NOT_SUPPORTED);
  // Waits as long as the marker could take on a slow host, and stops at once if it shows up.
  for (auto waited = 0ms; (!rpm.left("evaluated")) && (waited < 1s); waited += kEndWaitStep) {
    std::this_thread::sleep_for(kEndWaitStep);
  }
  EXPECT_FALSE(rpm.left("evaluated"));
  EndProcess(0);
}

TEST(SystemUnit, KmodVersionNoticesALateRpmEndedWhenNoPidfdCanBeOpened) {
  const auto slow = FakeRpm{"sleep 1\n: > \"$0.ended\"\n"};
  const auto good = FakeRpm{Prints(kFoundByRpm)};
  if ((!slow.runnable()) || (!good.runnable())) {
    GTEST_SKIP() << kNoExecSkip;
  }

  const auto child = fork();
  if (child == 0) {
    _exit(LookUpLateRpmWithoutPidfd(slow, good));
  }
  ASSERT_GT(child, 0);
  auto child_status = 0;
  auto has_returned = false;
  for (auto waited = 0ms; (!has_returned) && (waited < (2 * kEndWaitLimit));
       waited += kEndWaitStep) {
    has_returned = (waitpid(child, &child_status, WNOHANG) == child);
    if (!has_returned) {
      std::this_thread::sleep_for(kEndWaitStep);
    }
  }
  if (!has_returned) {
    kill(child, SIGKILL);
    waitpid(child, &child_status, 0);
  }

  ASSERT_TRUE(has_returned);
  ASSERT_TRUE(WIFEXITED(child_status));
  if (WEXITSTATUS(child_status) == kNoSeccompExit) {
    GTEST_SKIP() << "seccomp cannot be used here to make pidfd_open fail";
  }
  EXPECT_EQ(WEXITSTATUS(child_status), 0);
}
