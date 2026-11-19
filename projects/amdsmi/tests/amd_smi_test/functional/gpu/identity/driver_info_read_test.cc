// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "driver_info_read.h"

#include <gtest/gtest.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <climits>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>

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

}  // namespace

TestDriverInfoRead::TestDriverInfoRead() : TestBase() {
  set_title("AMDSMI Driver Info Read Test");
  set_description(
      "This test verifies that amdsmi_get_gpu_driver_info_v2() agrees with "
      "amdsmi_get_gpu_driver_info(), that it splits the amdgpu version, and that it reports the "
      "DKMS package of the running kernel, with its build, when that is the loaded driver.");
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

    const std::string build = info.driver_build_version;
    if (has_module && has_dkms && (dkms[1] == module[1])) {
      // DKMS recorded a package with the kernel version of the loaded module for this kernel.
      EXPECT_EQ(info.driver_package, AMDSMI_DRIVER_PACKAGE_DKMS);
      EXPECT_EQ(build, dkms[2].str());
      EXPECT_EQ(std::string{info.driver_full_version}, version + "-" + dkms[2].str());
    } else if (info.driver_package == AMDSMI_DRIVER_PACKAGE_NONE) {
      EXPECT_EQ(build, "");
      EXPECT_EQ(std::string{info.driver_full_version}, (version == "N/A") ? "" : version);
    } else {
      // A package this test cannot look up itself, such as amdgpu-kmod, comes with its build.
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
