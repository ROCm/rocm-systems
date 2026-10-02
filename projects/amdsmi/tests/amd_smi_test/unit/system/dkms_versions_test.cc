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
constexpr auto kKernelRelease = "6.8.0-124-generic";
constexpr auto kKernelMachine = "x86_64";
constexpr auto kKernelSymlinkPrefix = "kernel-";

class DkmsTreeFixture {
 public:
  DkmsTreeFixture() {
    base_ = fs::temp_directory_path() / "amdsmi_dkms_versions_test";
    auto err_code = std::error_code{};
    fs::remove_all(base_, err_code);
    dkms_root_ = base_ / "dkms";
    source_tree_prefix_ = (base_ / "usr_src/amdgpu-").string();
    fs::create_directories(dkms_root_);
    fs::create_directories(base_ / "usr_src");
  }

  DkmsTreeFixture(const DkmsTreeFixture&) = delete;
  auto operator=(const DkmsTreeFixture&) -> DkmsTreeFixture& = delete;
  DkmsTreeFixture(DkmsTreeFixture&&) = delete;
  auto operator=(DkmsTreeFixture&&) -> DkmsTreeFixture& = delete;

  ~DkmsTreeFixture() {
    auto err_code = std::error_code{};
    fs::remove_all(base_, err_code);
  }

  auto dkms_root() const -> const fs::path& { return dkms_root_; }

  auto source_tree_prefix() const -> std::string_view { return source_tree_prefix_; }

  auto add_package(const std::string& version) -> void {
    const auto paths = write_package_files(version);
    fs::create_directory_symlink(paths.source_tree, paths.version_dir / "source");
  }

  auto add_package_with_relative_source_symlink(const std::string& version) -> void {
    const auto paths = write_package_files(version);
    const auto relative_source = fs::relative(paths.source_tree, paths.version_dir);
    fs::create_directory_symlink(relative_source, paths.version_dir / "source");
  }

  auto add_kernel_symlink(const std::string& version, std::string_view release = kKernelRelease,
                          std::string_view machine = kKernelMachine) -> void {
    const auto target = ((dkms_root_ / version) / std::string{release}) / std::string{machine};
    fs::create_directories(target);
    fs::create_directory_symlink(fs::relative(target, dkms_root_),
                                 dkms_root_ / make_kernel_link_name(release, machine));
  }

  auto add_absolute_kernel_symlink(const std::string& version,
                                   std::string_view release = kKernelRelease,
                                   std::string_view machine = kKernelMachine) -> void {
    const auto target = ((dkms_root_ / version) / std::string{release}) / std::string{machine};
    fs::create_directories(target);
    fs::create_directory_symlink(target, dkms_root_ / make_kernel_link_name(release, machine));
  }

  auto add_malformed_kernel_symlink(std::string_view target,
                                    std::string_view release = kKernelRelease,
                                    std::string_view machine = kKernelMachine) -> void {
    fs::create_directory_symlink(fs::path{std::string{target}},
                                 dkms_root_ / make_kernel_link_name(release, machine));
  }

 private:
  struct PackagePaths {
    fs::path version_dir;
    fs::path source_tree;
  };

  auto write_package_files(const std::string& version) -> PackagePaths {
    auto paths = PackagePaths{};
    paths.version_dir = dkms_root_ / version;
    paths.source_tree = fs::path{source_tree_prefix_ + version};
    const auto dkms_conf_dir = paths.source_tree / "amd/dkms";
    fs::create_directories(paths.version_dir);
    fs::create_directories(dkms_conf_dir);

    auto conf_stream = std::ofstream{dkms_conf_dir / "dkms.conf"};
    conf_stream << "PACKAGE_NAME=\"" << kPackageName << "\"\n";
    conf_stream << "PACKAGE_VERSION=\"" << version << "\"\n";
    fs::create_symlink("amd/dkms/dkms.conf", paths.source_tree / "dkms.conf");
    return paths;
  }

  auto make_kernel_link_name(std::string_view release, std::string_view machine) const
      -> std::string {
    auto link_name = std::string{kKernelSymlinkPrefix};
    link_name.append(release);
    link_name.push_back('-');
    link_name.append(machine);
    return link_name;
  }

  fs::path base_{};
  fs::path dkms_root_{};
  std::string source_tree_prefix_{};
};

}  // namespace

TEST(SystemUnit, DriverVersionsSplitModuleAndPackageVersions) {
  auto info = amdsmi_driver_info_t{};

  EXPECT_EQ(smi_amdgpu_parse_driver_versions("6.19.14.31400000", "6.19.14-2370381.24.04", &info),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_STREQ(info.driver_kernel_version, "6.19.14");
  EXPECT_STREQ(info.amdgpu_driver_version, "31400000");
  EXPECT_STREQ(info.driver_version, "6.19.14.31400000");
  EXPECT_STREQ(info.driver_build_version, "2370381");
  EXPECT_STREQ(info.driver_full_version, "6.19.14.31400000-2370381");
}

TEST(SystemUnit, DriverVersionsKeepBuildEmptyWithoutActiveDkmsPackage) {
  auto info = amdsmi_driver_info_t{};

  EXPECT_EQ(smi_amdgpu_parse_driver_versions("6.19.14.31400000", "", &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_STREQ(info.driver_kernel_version, "6.19.14");
  EXPECT_STREQ(info.amdgpu_driver_version, "31400000");
  EXPECT_STREQ(info.driver_version, "6.19.14.31400000");
  EXPECT_STREQ(info.driver_build_version, "");
  EXPECT_STREQ(info.driver_full_version, "6.19.14.31400000");
}

TEST(SystemUnit, DriverVersionsExtractBuildWithoutDistroSuffix) {
  auto info = amdsmi_driver_info_t{};

  EXPECT_EQ(smi_amdgpu_parse_driver_versions("6.19.14.31400000", "6.19.14-2370381", &info),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_STREQ(info.driver_kernel_version, "6.19.14");
  EXPECT_STREQ(info.amdgpu_driver_version, "31400000");
  EXPECT_STREQ(info.driver_version, "6.19.14.31400000");
  EXPECT_STREQ(info.driver_build_version, "2370381");
  EXPECT_STREQ(info.driver_full_version, "6.19.14.31400000-2370381");
}

TEST(SystemUnit, DriverVersionsExtractBuildWithNonUbuntuDistroSuffix) {
  auto info = amdsmi_driver_info_t{};

  EXPECT_EQ(smi_amdgpu_parse_driver_versions("6.19.14.31400000", "6.19.14-2370381.el8", &info),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_STREQ(info.driver_kernel_version, "6.19.14");
  EXPECT_STREQ(info.amdgpu_driver_version, "31400000");
  EXPECT_STREQ(info.driver_version, "6.19.14.31400000");
  EXPECT_STREQ(info.driver_build_version, "2370381");
  EXPECT_STREQ(info.driver_full_version, "6.19.14.31400000-2370381");
}

TEST(SystemUnit, DriverVersionsPassThroughUnexpectedFormats) {
  auto info = amdsmi_driver_info_t{};

  EXPECT_EQ(smi_amdgpu_parse_driver_versions("6.8.0-60-generic", "6.19.14-2370381.24.04", &info),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_STREQ(info.driver_kernel_version, "");
  EXPECT_STREQ(info.amdgpu_driver_version, "");
  EXPECT_STREQ(info.driver_version, "6.8.0-60-generic");
  EXPECT_STREQ(info.driver_build_version, "");
  EXPECT_STREQ(info.driver_full_version, "6.8.0-60-generic");
}

TEST(SystemUnit, DriverVersionsThreePartModuleVersionWithoutBuild) {
  auto info = amdsmi_driver_info_t{};

  EXPECT_EQ(smi_amdgpu_parse_driver_versions("6.19.4", "", &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_STREQ(info.driver_kernel_version, "6.19.4");
  EXPECT_STREQ(info.amdgpu_driver_version, "");
  EXPECT_STREQ(info.driver_version, "6.19.4");
  EXPECT_STREQ(info.driver_build_version, "");
  EXPECT_STREQ(info.driver_full_version, "6.19.4");
}

TEST(SystemUnit, DriverVersionsThreePartModuleVersionWithBuild) {
  auto info = amdsmi_driver_info_t{};

  EXPECT_EQ(smi_amdgpu_parse_driver_versions("6.19.4", "6.19.4-2193512.24.04", &info),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_STREQ(info.driver_kernel_version, "6.19.4");
  EXPECT_STREQ(info.amdgpu_driver_version, "");
  EXPECT_STREQ(info.driver_version, "6.19.4");
  EXPECT_STREQ(info.driver_build_version, "2193512");
  EXPECT_STREQ(info.driver_full_version, "6.19.4-2193512");
}

TEST(SystemUnit, DriverVersionsIgnoreBuildFromOtherDkmsPackage) {
  auto info = amdsmi_driver_info_t{};

  EXPECT_EQ(smi_amdgpu_parse_driver_versions("6.19.4", "6.19.14-2370381.24.04", &info),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_STREQ(info.driver_kernel_version, "6.19.4");
  EXPECT_STREQ(info.amdgpu_driver_version, "");
  EXPECT_STREQ(info.driver_version, "6.19.4");
  EXPECT_STREQ(info.driver_build_version, "");
  EXPECT_STREQ(info.driver_full_version, "6.19.4");
}

TEST(SystemUnit, DriverVersionsModuleVersionUnavailable) {
  auto info = amdsmi_driver_info_t{};

  EXPECT_EQ(smi_amdgpu_parse_driver_versions("N/A", "6.19.14-2370381.24.04", &info),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_STREQ(info.driver_kernel_version, "");
  EXPECT_STREQ(info.amdgpu_driver_version, "");
  EXPECT_STREQ(info.driver_version, "N/A");
  EXPECT_STREQ(info.driver_build_version, "");
  EXPECT_STREQ(info.driver_full_version, "");
}

TEST(SystemUnit, DriverVersionsRejectNullOutput) {
  EXPECT_EQ(smi_amdgpu_parse_driver_versions("6.19.14.31400000", "6.19.14-2370381.24.04", nullptr),
            AMDSMI_STATUS_INVAL);
}

TEST(SystemUnit, ActiveDkmsVersionSelectsKernelSymlinkPackage) {
  auto fixture = DkmsTreeFixture{};
  fixture.add_package(kVersionA);
  fixture.add_package(kVersionB);
  fixture.add_kernel_symlink(kVersionA);

  auto active_version = std::string{};
  EXPECT_EQ(
      smi_amdgpu_get_active_dkms_version(fixture.dkms_root().string(), fixture.source_tree_prefix(),
                                         kKernelRelease, kKernelMachine, &active_version),
      AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(active_version, kVersionA);
}

TEST(SystemUnit, ActiveDkmsVersionRejectsMissingKernelSymlink) {
  auto fixture = DkmsTreeFixture{};
  fixture.add_package(kVersionA);

  auto active_version = std::string{"unchanged"};
  EXPECT_EQ(
      smi_amdgpu_get_active_dkms_version(fixture.dkms_root().string(), fixture.source_tree_prefix(),
                                         kKernelRelease, kKernelMachine, &active_version),
      AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_TRUE(active_version.empty());
}

TEST(SystemUnit, ActiveDkmsVersionRejectsUnvalidatedKernelSymlinkTarget) {
  auto fixture = DkmsTreeFixture{};
  fixture.add_package(kVersionA);
  fixture.add_kernel_symlink(kVersionB);

  auto active_version = std::string{"unchanged"};
  EXPECT_EQ(
      smi_amdgpu_get_active_dkms_version(fixture.dkms_root().string(), fixture.source_tree_prefix(),
                                         kKernelRelease, kKernelMachine, &active_version),
      AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_TRUE(active_version.empty());
}

TEST(SystemUnit, ActiveDkmsVersionRejectsNullOutput) {
  auto fixture = DkmsTreeFixture{};

  EXPECT_EQ(
      smi_amdgpu_get_active_dkms_version(fixture.dkms_root().string(), fixture.source_tree_prefix(),
                                         kKernelRelease, kKernelMachine, nullptr),
      AMDSMI_STATUS_INVAL);
}

TEST(SystemUnit, ActiveDkmsVersionRejectsEmptyRelease) {
  auto fixture = DkmsTreeFixture{};
  fixture.add_package(kVersionA);
  fixture.add_kernel_symlink(kVersionA);

  auto active_version = std::string{"unchanged"};
  EXPECT_EQ(
      smi_amdgpu_get_active_dkms_version(fixture.dkms_root().string(), fixture.source_tree_prefix(),
                                         "", kKernelMachine, &active_version),
      AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_TRUE(active_version.empty());
}

TEST(SystemUnit, ActiveDkmsVersionRejectsEmptyMachine) {
  auto fixture = DkmsTreeFixture{};
  fixture.add_package(kVersionA);
  fixture.add_kernel_symlink(kVersionA);

  auto active_version = std::string{"unchanged"};
  EXPECT_EQ(
      smi_amdgpu_get_active_dkms_version(fixture.dkms_root().string(), fixture.source_tree_prefix(),
                                         kKernelRelease, "", &active_version),
      AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_TRUE(active_version.empty());
}

TEST(SystemUnit, ActiveDkmsVersionSelectsAbsoluteKernelSymlink) {
  auto fixture = DkmsTreeFixture{};
  fixture.add_package(kVersionA);
  fixture.add_absolute_kernel_symlink(kVersionA);

  auto active_version = std::string{};
  EXPECT_EQ(
      smi_amdgpu_get_active_dkms_version(fixture.dkms_root().string(), fixture.source_tree_prefix(),
                                         kKernelRelease, kKernelMachine, &active_version),
      AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(active_version, kVersionA);
}

TEST(SystemUnit, ActiveDkmsVersionRejectsMalformedKernelSymlinkTarget) {
  auto fixture = DkmsTreeFixture{};
  fixture.add_package(kVersionA);
  fixture.add_malformed_kernel_symlink(
      (std::string{"not-a-version/"} + kKernelRelease + "/" + kKernelMachine));

  auto active_version = std::string{"unchanged"};
  EXPECT_EQ(
      smi_amdgpu_get_active_dkms_version(fixture.dkms_root().string(), fixture.source_tree_prefix(),
                                         kKernelRelease, kKernelMachine, &active_version),
      AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_TRUE(active_version.empty());
}

TEST(SystemUnit, ActiveDkmsVersionAcceptsRelativeSourceSymlink) {
  auto fixture = DkmsTreeFixture{};
  fixture.add_package_with_relative_source_symlink(kVersionA);
  fixture.add_kernel_symlink(kVersionA);

  auto active_version = std::string{};
  EXPECT_EQ(
      smi_amdgpu_get_active_dkms_version(fixture.dkms_root().string(), fixture.source_tree_prefix(),
                                         kKernelRelease, kKernelMachine, &active_version),
      AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(active_version, kVersionA);
}

TEST(SystemUnit, ActiveDkmsVersionRejectsMissingDkmsConf) {
  auto fixture = DkmsTreeFixture{};
  const auto version_dir = fixture.dkms_root() / kVersionA;
  const auto source_tree = fs::path{std::string{fixture.source_tree_prefix()} + kVersionA};
  fs::create_directories(version_dir);
  fs::create_directories(source_tree);
  fs::create_directory_symlink(source_tree, version_dir / "source");
  fixture.add_kernel_symlink(kVersionA);

  auto active_version = std::string{"unchanged"};
  EXPECT_EQ(
      smi_amdgpu_get_active_dkms_version(fixture.dkms_root().string(), fixture.source_tree_prefix(),
                                         kKernelRelease, kKernelMachine, &active_version),
      AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_TRUE(active_version.empty());
}

TEST(SystemUnit, ActiveDkmsVersionRejectsWrongSourceSymlinkTarget) {
  auto fixture = DkmsTreeFixture{};
  fixture.add_package(kVersionA);
  const auto version_dir = fixture.dkms_root() / kVersionA;
  const auto decoy_tree = fixture.dkms_root() / "decoy";
  fs::create_directories(decoy_tree);
  fs::remove(version_dir / "source");
  fs::create_directory_symlink(decoy_tree, version_dir / "source");
  fixture.add_kernel_symlink(kVersionA);

  auto active_version = std::string{"unchanged"};
  EXPECT_EQ(
      smi_amdgpu_get_active_dkms_version(fixture.dkms_root().string(), fixture.source_tree_prefix(),
                                         kKernelRelease, kKernelMachine, &active_version),
      AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_TRUE(active_version.empty());
}

TEST(SystemUnit, ActiveDkmsVersionRejectsPackageNameMismatch) {
  auto fixture = DkmsTreeFixture{};
  const auto version_dir = fixture.dkms_root() / kVersionA;
  const auto source_tree = fs::path{std::string{fixture.source_tree_prefix()} + kVersionA};
  const auto dkms_conf_dir = source_tree / "amd/dkms";
  fs::create_directories(version_dir);
  fs::create_directories(dkms_conf_dir);

  auto conf_stream = std::ofstream{dkms_conf_dir / "dkms.conf"};
  conf_stream << "PACKAGE_NAME=\"" << kOtherPackageName << "\"\n";
  conf_stream << "PACKAGE_VERSION=\"" << kVersionA << "\"\n";
  conf_stream.close();
  fs::create_symlink("amd/dkms/dkms.conf", source_tree / "dkms.conf");
  fs::create_directory_symlink(source_tree, version_dir / "source");
  fixture.add_kernel_symlink(kVersionA);

  auto active_version = std::string{"unchanged"};
  EXPECT_EQ(
      smi_amdgpu_get_active_dkms_version(fixture.dkms_root().string(), fixture.source_tree_prefix(),
                                         kKernelRelease, kKernelMachine, &active_version),
      AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_TRUE(active_version.empty());
}

TEST(SystemUnit, ActiveDkmsVersionRejectsPackageVersionMismatch) {
  auto fixture = DkmsTreeFixture{};
  const auto version_dir = fixture.dkms_root() / kVersionA;
  const auto source_tree = fs::path{std::string{fixture.source_tree_prefix()} + kVersionA};
  const auto dkms_conf_dir = source_tree / "amd/dkms";
  fs::create_directories(version_dir);
  fs::create_directories(dkms_conf_dir);

  auto conf_stream = std::ofstream{dkms_conf_dir / "dkms.conf"};
  conf_stream << "PACKAGE_NAME=\"" << kPackageName << "\"\n";
  conf_stream << "PACKAGE_VERSION=\"" << kVersionB << "\"\n";
  conf_stream.close();
  fs::create_symlink("amd/dkms/dkms.conf", source_tree / "dkms.conf");
  fs::create_directory_symlink(source_tree, version_dir / "source");
  fixture.add_kernel_symlink(kVersionA);

  auto active_version = std::string{"unchanged"};
  EXPECT_EQ(
      smi_amdgpu_get_active_dkms_version(fixture.dkms_root().string(), fixture.source_tree_prefix(),
                                         kKernelRelease, kKernelMachine, &active_version),
      AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_TRUE(active_version.empty());
}
