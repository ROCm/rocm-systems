// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Validates the amdgpu version split, the DKMS and amdgpu-kmod package lookups, and how a package
// is matched to the loaded module for amdsmi_get_gpu_driver_info_v2(). No GPU required.

#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <csignal>
#include <cstddef>
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
    script += "[ \"$#\" = 4 ] && [ \"$1\" = -q ] && [ \"$2\" = --qf ] && ";
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
  EXPECT_STREQ(info.driver_package_version, "6.19.14-2370381");
}

TEST(SystemUnit, DriverVersionsReadDkmsBuildWithAnyDistroSuffix) {
  for (const auto* dkms_version : {"6.19.14-2370381", "6.19.14-2370381.el9"}) {
    auto info = amdsmi_driver_info_v2_t{};

    EXPECT_EQ(smi_amdgpu_parse_driver_versions("6.19.14.31400000", dkms_version, "", &info),
              AMDSMI_STATUS_SUCCESS);
    EXPECT_STREQ(info.driver_build_version, "2370381") << dkms_version;
    EXPECT_STREQ(info.driver_package_version, "6.19.14-2370381") << dkms_version;
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
  EXPECT_STREQ(info.driver_package_version, "");
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
  EXPECT_STREQ(info.driver_package_version, "6.16.13.000000-2412345");
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
  EXPECT_STREQ(info.driver_package_version, "6.16.13.000000-2412345");
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
  EXPECT_STREQ(info.driver_package_version, "");
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
    EXPECT_STREQ(info.driver_package_version, "") << module_version;
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
  EXPECT_STREQ(info.driver_package_version, "");
}

TEST(SystemUnit, DriverVersionsIgnoreMalformedPackageVersion) {
  auto info = amdsmi_driver_info_v2_t{};

  EXPECT_EQ(smi_amdgpu_parse_driver_versions("6.19.14.31400000", "6.19.14", "6.19.14-rc1", &info),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_STREQ(info.driver_build_version, "");
  EXPECT_STREQ(info.driver_full_version, "6.19.14.31400000");
  EXPECT_EQ(info.driver_package, AMDSMI_DRIVER_PACKAGE_NONE);
  EXPECT_STREQ(info.driver_package_version, "");
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
  std::strcpy(info.driver_package_version, "stale");
  info.driver_package = AMDSMI_DRIVER_PACKAGE_KMOD;

  EXPECT_EQ(smi_amdgpu_parse_driver_versions("N/A", "", "", &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_STREQ(info.driver_version, "6.19.14.31400000");
  EXPECT_STREQ(info.driver_date, "2015/01/01 00:00");
  EXPECT_STREQ(info.driver_name, "amdgpu");
  EXPECT_STREQ(info.driver_kernel_version, "");
  EXPECT_STREQ(info.amdgpu_driver_version, "");
  EXPECT_STREQ(info.driver_build_version, "");
  EXPECT_STREQ(info.driver_full_version, "");
  EXPECT_STREQ(info.driver_package_version, "");
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
    EXPECT_EQ(std::string{info.driver_package_version}, fits ? "6.19.14-" + build : "");
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
  EXPECT_STREQ(info.driver_package_version, "6.19.13-2370380");
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
  EXPECT_STREQ(info.driver_package_version, "6.16.13.000000-2412345");
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
    EXPECT_STREQ(info.driver_package_version, "6.16.13.000000-2412345") << versions.front();
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

TEST(SystemUnit, KmodVersionListsEveryInstalledVersion) {
  const auto rpm = FakeRpm{Prints("6.16.13.000000-2412345.el9\n6.16.6.000000-2400001.el9\n")};
  if (!rpm.runnable()) GTEST_SKIP() << kNoExecSkip;

  auto versions = Versions{};
  EXPECT_EQ(KmodVersions(rpm, &versions), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(versions, (Versions{"6.16.13.000000-2412345", "6.16.6.000000-2400001"}));
}

TEST(SystemUnit, KmodVersionNotSupportedWhenPackageIsNotInstalled) {
  const auto rpm = FakeRpm{Prints("package amdgpu-kmod is not installed\n", 1)};
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
  auto versions = Versions{"stale"};

  EXPECT_EQ(smi_amdgpu_get_kmod_versions(kRpmPath, "", &versions), AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_TRUE(versions.empty());
}

TEST(SystemUnit, KmodVersionRejectsNullOutput) {
  EXPECT_EQ(smi_amdgpu_get_kmod_versions(kRpmPath, kKmodRelease, nullptr), AMDSMI_STATUS_INVAL);
}
