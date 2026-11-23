// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "driver_info_read.h"

#include <gtest/gtest.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <climits>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <regex>
#include <set>
#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "test_common.h"

namespace {

// The amdgpu-dkms version DKMS recorded for the running kernel, or "" when there is no such
// record. Read here and not through the library: the test must not share the lookup it checks.
std::string DkmsVersionOfRunningKernel() {
  utsname uts = {};
  if (uname(&uts) != 0) return "";
  const std::string root = "/var/lib/dkms/amdgpu/";
  const std::string link = root + "kernel-" + uts.release + "-" + uts.machine;
  char target[PATH_MAX] = {};
  if (readlink(link.c_str(), target, sizeof(target) - 1) <= 0) return "";

  // DKMS writes <version>/<release>/<machine>, relative to the root.
  const std::string relative = target;
  const std::string version = relative.substr(0, relative.find('/'));
  if (version.empty()) return "";
  std::ifstream conf(root + version + "/source/dkms.conf");
  bool is_amdgpu = false;
  bool is_this_version = false;
  for (std::string line; std::getline(conf, line);) {
    is_amdgpu = is_amdgpu || (line == "PACKAGE_NAME=\"amdgpu\"");
    is_this_version = is_this_version || (line == "PACKAGE_VERSION=\"" + version + "\"");
  }
  return (is_amdgpu && is_this_version) ? version : "";
}

// The VERSION-RELEASE of each amdgpu-kmod package rpm has for the running kernel, or nothing when
// rpm or its database is not there. Like the library, rpm is asked only where its database has a
// packages file: a query creates a database where there is none.
std::vector<std::string> KmodVersionsOfRunningKernel() {
  utsname uts = {};
  if (uname(&uts) != 0) return {};
  const std::string release = uts.release;
  // The release becomes part of a shell command.
  if (!std::regex_match(release, std::regex{R"([A-Za-z0-9._+~-]+)"})) return {};

  const auto rpm = [](const std::string& arguments, std::vector<std::string>* lines) {
    const std::string command =
        "timeout 30 env -i LC_ALL=C HOME=/ /usr/bin/rpm " + arguments + " 2>/dev/null";
    FILE* pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) return false;
    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
      std::string line = buffer;
      if (!line.empty() && (line.back() == '\n')) line.pop_back();
      lines->push_back(line);
    }
    return pclose(pipe) == 0;
  };

  std::vector<std::string> dbpath;
  if (!rpm("--eval '%{_dbpath}'", &dbpath) || (dbpath.size() != 1) || dbpath[0].empty() ||
      (dbpath[0][0] != '/')) {
    return {};
  }
  bool has_database = false;
  for (const char* file : {"/Packages", "/Packages.db", "/rpmdb.sqlite"}) {
    has_database = has_database || (access((dbpath[0] + file).c_str(), F_OK) == 0);
  }
  std::vector<std::string> versions;
  if (!has_database ||
      !rpm("-q --qf '%{VERSION}-%{RELEASE}\\n' amdgpu-kmod-" + release, &versions)) {
    return {};
  }
  return versions;
}

}  // namespace

TestDriverInfoRead::TestDriverInfoRead() : TestBase() {
  set_title("AMDSMI Driver Info Read Test");
  set_description(
      "This test verifies that amdsmi_get_gpu_driver_info_v2() agrees with "
      "amdsmi_get_gpu_driver_info(), that it splits the amdgpu version, and that it reports the "
      "DKMS package of the running kernel, or else its amdgpu-kmod RPM, with its build, when "
      "that is the loaded driver.");
}

TestDriverInfoRead::~TestDriverInfoRead(void) {}

void TestDriverInfoRead::SetUp(void) {
  TestBase::SetUp();
  return;
}

void TestDriverInfoRead::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestDriverInfoRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestDriverInfoRead::Close() { TestBase::Close(); }

void TestDriverInfoRead::Run(void) {
  TestBase::Run();
  PRINT_VERBOSITY();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  // A.B.C[.D] for the module, A.B.C[.D]-<build>[.<distro>] for the package.
  static const std::regex module_format{R"(^(\d+\.\d+\.\d+)(?:\.(\d+))?$)"};
  static const std::regex package_format{R"(^(\d+\.\d+\.\d+)(?:\.\d+)*-(\d+)(?:\..*)?$)"};
  const std::string dkms_version = DkmsVersionOfRunningKernel();
  std::smatch dkms;
  const bool has_dkms = std::regex_match(dkms_version, dkms, package_format);
  // Asked before the library's time-limited rpm query, so rpm's own caches are warm and the
  // library's query is not the cold one.
  const std::vector<std::string> kmod_versions = KmodVersionsOfRunningKernel();
  uint32_t answered = 0;

  for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
    IF_VERB(STANDARD) {
      std::cout << "\t*************************" << std::endl;
      std::cout << "\t**Device index: " << i << std::endl;
    }

    amdsmi_driver_info_t v1 = {};
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_driver_info", "gpu=" + std::to_string(i), VERB(STANDARD));
    const amdsmi_status_t v1_err = amdsmi_get_gpu_driver_info(processor_handles_[i], &v1);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, v1_err, AMDSMI_STATUS_SUCCESS);

    amdsmi_driver_info_v2_t info = {};
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_driver_info_v2", "gpu=" + std::to_string(i), VERB(STANDARD));
    const amdsmi_status_t err = amdsmi_get_gpu_driver_info_v2(processor_handles_[i], &info);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, v1_err);
    ASSERT_EQ(err, v1_err);
    if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
      continue;
    }
    CHK_ERR_ASRT(err)
    ++answered;

    IF_VERB(STANDARD) {
      std::cout << "\t**Driver name: " << info.driver_name << std::endl;
      std::cout << "\t**Driver date: " << info.driver_date << std::endl;
      std::cout << "\t**Driver version: " << info.driver_version << std::endl;
      std::cout << "\t**Driver kernel version: " << info.driver_kernel_version << std::endl;
      std::cout << "\t**amdgpu driver version: " << info.amdgpu_driver_version << std::endl;
      std::cout << "\t**Driver build version: " << info.driver_build_version << std::endl;
      std::cout << "\t**Driver full version: " << info.driver_full_version << std::endl;
      std::cout << "\t**Driver package: " << info.driver_package << std::endl;
    }

    EXPECT_STREQ(info.driver_version, v1.driver_version);
    EXPECT_STREQ(info.driver_date, v1.driver_date);
    EXPECT_STREQ(info.driver_name, v1.driver_name);

    const std::string version = v1.driver_version;
    std::smatch module;
    const bool has_module = std::regex_match(version, module, module_format);
    EXPECT_EQ(std::string{info.driver_kernel_version}, has_module ? module[1].str() : "");
    EXPECT_EQ(std::string{info.amdgpu_driver_version}, has_module ? module[2].str() : "");

    // The builds rpm lists for amdgpu-kmod with the kernel version of the loaded module.
    std::set<std::string> kmod_builds;
    for (const std::string& kmod_version : kmod_versions) {
      std::smatch kmod;
      if (has_module && std::regex_match(kmod_version, kmod, package_format) &&
          (kmod[1] == module[1])) {
        kmod_builds.insert(kmod[2].str());
      }
    }

    const std::string build = info.driver_build_version;
    if (has_module && has_dkms && (dkms[1] == module[1])) {
      // DKMS recorded a package with the kernel version of the loaded module for this kernel.
      EXPECT_EQ(info.driver_package, AMDSMI_DRIVER_PACKAGE_DKMS);
      EXPECT_EQ(build, dkms[2].str());
      EXPECT_EQ(std::string{info.driver_full_version}, version + "-" + dkms[2].str());
    } else if (kmod_builds.size() == 1) {
      // No DKMS package of the module, and rpm has one amdgpu-kmod build of it. Several would
      // give none: which of them is loaded is not known.
      const std::string& kmod_build = *kmod_builds.begin();
      EXPECT_EQ(info.driver_package, AMDSMI_DRIVER_PACKAGE_KMOD);
      EXPECT_EQ(build, kmod_build);
      EXPECT_EQ(std::string{info.driver_full_version}, version + "-" + kmod_build);
    } else if (kmod_builds.size() > 1) {
      EXPECT_EQ(info.driver_package, AMDSMI_DRIVER_PACKAGE_NONE);
      EXPECT_EQ(build, "");
    } else if (info.driver_package == AMDSMI_DRIVER_PACKAGE_NONE) {
      EXPECT_EQ(build, "");
      EXPECT_EQ(std::string{info.driver_full_version}, (version == "N/A") ? "" : version);
    } else {
      // A package this test could not look up itself comes with its build.
      ASSERT_TRUE(has_module);
      ASSERT_FALSE(build.empty());
      EXPECT_EQ(std::string{info.driver_full_version}, version + "-" + build);
    }

    DISPLAY_AMDSMI_API("amdsmi_get_gpu_driver_info_v2", "gpu=" + std::to_string(i) + ", info=null",
                       VERB(STANDARD));
    const amdsmi_status_t null_err = amdsmi_get_gpu_driver_info_v2(processor_handles_[i], nullptr);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, null_err, AMDSMI_STATUS_INVAL);
    ASSERT_EQ(null_err, AMDSMI_STATUS_INVAL);
  }

  if (answered == 0) {
    GTEST_SKIP() << "no device reported driver info";
  }
}
