// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Validates DKMS package discovery against a temporary tree. No GPU required.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include "amd_smi/impl/amd_smi_utils.h"

namespace {

namespace fs = std::filesystem;

constexpr auto kPackageName = "amdgpu";
constexpr auto kOtherPackageName = "not-amdgpu";
constexpr auto kVersionA = "6.19.13-2370380.24.04";
constexpr auto kVersionB = "6.19.20-2450390.24.04";
constexpr auto kMissingDkmsRoot = "/tmp/amdsmi_missing_dkms_root";

class DkmsTreeFixture {
 public:
  DkmsTreeFixture() {
    base_ = fs::temp_directory_path() / "amdsmi_dkms_versions_test";
    fs::remove_all(base_);
    dkms_root_ = base_ / "dkms";
    source_tree_prefix_ = (base_ / "usr_src/amdgpu-").string();
    fs::create_directories(dkms_root_);
    fs::create_directories(base_ / "usr_src");
  }

  ~DkmsTreeFixture() { fs::remove_all(base_); }

  auto dkms_root() const -> const fs::path& { return dkms_root_; }

  auto source_tree_prefix() const -> std::string_view { return source_tree_prefix_; }

  auto add_package(const std::string& version, bool use_relative_source_symlink = false) -> void {
    const auto version_dir = dkms_root_ / version;
    fs::create_directories(version_dir);
    const auto source_tree = fs::path{source_tree_prefix_ + version};
    const auto dkms_conf_dir = source_tree / "amd/dkms";
    fs::create_directories(dkms_conf_dir);

    auto conf_stream = std::ofstream{dkms_conf_dir / "dkms.conf"};
    conf_stream << "PACKAGE_NAME=\"" << kPackageName << "\"\n";
    conf_stream << "PACKAGE_VERSION=\"" << version << "\"\n";

    fs::create_symlink("amd/dkms/dkms.conf", source_tree / "dkms.conf");

    if (use_relative_source_symlink) {
      const auto relative_source = fs::relative(source_tree, version_dir);
      fs::create_directory_symlink(relative_source, version_dir / "source");
      return;
    }

    fs::create_directory_symlink(source_tree, version_dir / "source");
  }

  auto add_kernel_symlink() -> void {
    const auto target = dkms_root_ / kVersionA / "kernel";
    fs::create_directories(target);
    fs::create_directory_symlink(target, dkms_root_ / "kernel-5.19.0-38-generic-x86_64");
  }

 private:
  fs::path base_{};
  fs::path dkms_root_{};
  std::string source_tree_prefix_{};
};

}  // namespace

TEST(SystemUnit, DkmsVersionsCollectsValidPackages) {
  auto fixture = DkmsTreeFixture{};
  fixture.add_package(kVersionA);
  fixture.add_package(kVersionB);

  auto packages = smi_amdgpu_dkms_packages_t{};
  const auto status = smi_amdgpu_get_dkms_versions_from(fixture.dkms_root().string(),
                                                        fixture.source_tree_prefix(), &packages);

  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(packages.size(), 2u);
  EXPECT_EQ(packages.at(kVersionA), kPackageName);
  EXPECT_EQ(packages.at(kVersionB), kPackageName);
}

TEST(SystemUnit, ActiveDkmsVersionSelectsExactLoadedVersion) {
  const auto packages =
      smi_amdgpu_dkms_packages_t{{kVersionA, kPackageName}, {kVersionB, kPackageName}};
  auto active_version = std::string{};

  EXPECT_EQ(smi_amdgpu_get_active_dkms_version(packages, kVersionB, &active_version),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(active_version, kVersionB);
}

TEST(SystemUnit, ActiveDkmsVersionRejectsMissingExactMatch) {
  const auto packages = smi_amdgpu_dkms_packages_t{{kVersionA, kPackageName}};
  auto active_version = std::string{"unchanged"};

  EXPECT_EQ(smi_amdgpu_get_active_dkms_version(packages, kVersionB, &active_version),
            AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_TRUE(active_version.empty());
}

TEST(SystemUnit, ActiveDkmsVersionRejectsUnavailableLoadedVersion) {
  const auto packages = smi_amdgpu_dkms_packages_t{{"N/A", kPackageName}};
  auto active_version = std::string{"unchanged"};

  EXPECT_EQ(smi_amdgpu_get_active_dkms_version(packages, "N/A", &active_version),
            AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_TRUE(active_version.empty());
}

TEST(SystemUnit, ActiveDkmsVersionRejectsNullOutput) {
  const auto packages = smi_amdgpu_dkms_packages_t{{kVersionA, kPackageName}};

  EXPECT_EQ(smi_amdgpu_get_active_dkms_version(packages, kVersionA, nullptr), AMDSMI_STATUS_INVAL);
}

TEST(SystemUnit, GetAmdgpuDkmsVersionRequiresInitialization) {
  char version[AMDSMI_MAX_STRING_LENGTH] = {};

  EXPECT_EQ(amdsmi_get_amdgpu_dkms_version(version, sizeof(version)), AMDSMI_STATUS_NOT_INIT);
}

TEST(SystemUnit, GetAmdgpuDkmsVersionNullptrRequiresInitialization) {
  EXPECT_EQ(amdsmi_get_amdgpu_dkms_version(nullptr, AMDSMI_MAX_STRING_LENGTH),
            AMDSMI_STATUS_NOT_INIT);
}

TEST(SystemUnit, GetAmdgpuDkmsVersionShortLengthRequiresInitialization) {
  char version[1] = {};

  EXPECT_EQ(amdsmi_get_amdgpu_dkms_version(version, sizeof(version)), AMDSMI_STATUS_NOT_INIT);
}

TEST(SystemUnit, DkmsVersionsSkipsKernelSymlinks) {
  auto fixture = DkmsTreeFixture{};
  fixture.add_package(kVersionA);
  fixture.add_kernel_symlink();

  auto packages = smi_amdgpu_dkms_packages_t{};
  const auto status = smi_amdgpu_get_dkms_versions_from(fixture.dkms_root().string(),
                                                        fixture.source_tree_prefix(), &packages);

  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(packages.size(), 1u);
  EXPECT_EQ(packages.begin()->first, kVersionA);
}

TEST(SystemUnit, DkmsVersionsReturnsPackagesInLexicographicOrder) {
  auto fixture = DkmsTreeFixture{};
  fixture.add_package(kVersionB);
  fixture.add_package(kVersionA);

  auto packages = smi_amdgpu_dkms_packages_t{};
  const auto status = smi_amdgpu_get_dkms_versions_from(fixture.dkms_root().string(),
                                                        fixture.source_tree_prefix(), &packages);

  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(packages.size(), 2u);
  auto package = packages.begin();
  EXPECT_EQ(package->first, kVersionA);
  ++package;
  EXPECT_EQ(package->first, kVersionB);
}

TEST(SystemUnit, DkmsVersionsReturnsSuccessForEmptyRoot) {
  auto fixture = DkmsTreeFixture{};

  auto packages = smi_amdgpu_dkms_packages_t{};
  packages.emplace(kVersionA, kPackageName);
  const auto status = smi_amdgpu_get_dkms_versions_from(fixture.dkms_root().string(),
                                                        fixture.source_tree_prefix(), &packages);

  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_TRUE(packages.empty());
}

TEST(SystemUnit, DkmsVersionsAcceptsRelativeSourceSymlink) {
  auto fixture = DkmsTreeFixture{};
  fixture.add_package(kVersionA, true);

  auto packages = smi_amdgpu_dkms_packages_t{};
  const auto status = smi_amdgpu_get_dkms_versions_from(fixture.dkms_root().string(),
                                                        fixture.source_tree_prefix(), &packages);

  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(packages.size(), 1u);
  EXPECT_EQ(packages.begin()->first, kVersionA);
}

TEST(SystemUnit, DkmsVersionsRejectsMissingDkmsConf) {
  auto fixture = DkmsTreeFixture{};
  const auto version_dir = fixture.dkms_root() / kVersionA;
  const auto source_tree = fs::path{std::string{fixture.source_tree_prefix()} + kVersionA};
  fs::create_directories(version_dir);
  fs::create_directories(source_tree);
  fs::create_directory_symlink(source_tree, version_dir / "source");

  auto packages = smi_amdgpu_dkms_packages_t{};
  const auto status = smi_amdgpu_get_dkms_versions_from(fixture.dkms_root().string(),
                                                        fixture.source_tree_prefix(), &packages);

  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_TRUE(packages.empty());
}

TEST(SystemUnit, DkmsVersionsRejectsWrongSourceSymlinkTarget) {
  auto fixture = DkmsTreeFixture{};
  fixture.add_package(kVersionA);
  const auto version_dir = fixture.dkms_root() / kVersionA;
  const auto decoy_tree = fixture.dkms_root() / "decoy";
  fs::create_directories(decoy_tree);
  fs::remove(version_dir / "source");
  fs::create_directory_symlink(decoy_tree, version_dir / "source");

  auto packages = smi_amdgpu_dkms_packages_t{};
  const auto status = smi_amdgpu_get_dkms_versions_from(fixture.dkms_root().string(),
                                                        fixture.source_tree_prefix(), &packages);

  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_TRUE(packages.empty());
}

TEST(SystemUnit, DkmsVersionsRejectsPackageNameMismatch) {
  auto fixture = DkmsTreeFixture{};
  const auto version_dir = fixture.dkms_root() / kVersionA;
  const auto source_tree = fs::path{std::string{fixture.source_tree_prefix()} + kVersionA};
  const auto dkms_conf_dir = source_tree / "amd/dkms";
  fs::create_directories(version_dir);
  fs::create_directories(dkms_conf_dir);

  auto conf_stream = std::ofstream{dkms_conf_dir / "dkms.conf"};
  conf_stream << "PACKAGE_NAME=\"" << kOtherPackageName << "\"\n";
  conf_stream << "PACKAGE_VERSION=\"" << kVersionA << "\"\n";
  fs::create_symlink("amd/dkms/dkms.conf", source_tree / "dkms.conf");
  fs::create_directory_symlink(source_tree, version_dir / "source");

  auto packages = smi_amdgpu_dkms_packages_t{};
  const auto status = smi_amdgpu_get_dkms_versions_from(fixture.dkms_root().string(),
                                                        fixture.source_tree_prefix(), &packages);

  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_TRUE(packages.empty());
}

TEST(SystemUnit, DkmsVersionsRejectsPackageVersionMismatch) {
  auto fixture = DkmsTreeFixture{};
  const auto version_dir = fixture.dkms_root() / kVersionA;
  const auto source_tree = fs::path{std::string{fixture.source_tree_prefix()} + kVersionA};
  const auto dkms_conf_dir = source_tree / "amd/dkms";
  fs::create_directories(version_dir);
  fs::create_directories(dkms_conf_dir);

  auto conf_stream = std::ofstream{dkms_conf_dir / "dkms.conf"};
  conf_stream << "PACKAGE_NAME=\"" << kPackageName << "\"\n";
  conf_stream << "PACKAGE_VERSION=\"" << kVersionB << "\"\n";
  fs::create_symlink("amd/dkms/dkms.conf", source_tree / "dkms.conf");
  fs::create_directory_symlink(source_tree, version_dir / "source");

  auto packages = smi_amdgpu_dkms_packages_t{};
  const auto status = smi_amdgpu_get_dkms_versions_from(fixture.dkms_root().string(),
                                                        fixture.source_tree_prefix(), &packages);

  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_TRUE(packages.empty());
}

TEST(SystemUnit, DkmsVersionsRejectsNullOutput) {
  const auto status = smi_amdgpu_get_dkms_versions_from(kMissingDkmsRoot, nullptr);
  EXPECT_EQ(status, AMDSMI_STATUS_INVAL);
}

TEST(SystemUnit, DkmsVersionsReturnsNotSupportedForMissingRoot) {
  auto packages = smi_amdgpu_dkms_packages_t{};
  const auto status = smi_amdgpu_get_dkms_versions_from(kMissingDkmsRoot, &packages);

  EXPECT_EQ(status, AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_TRUE(packages.empty());
}
