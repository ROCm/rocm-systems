// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "test_fixture_utils.h"

#include <gtest/gtest.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <system_error>

#include "amd_smi/impl/amd_smi_processor.h"
#include "amd_smi/impl/amd_smi_system.h"
#include "rocm_smi/rocm_smi_device.h"
#include "rocm_smi/rocm_smi_main.h"

namespace fs = std::filesystem;

FakeSysfsTree::FakeSysfsTree() {
  std::string tmpl = "/tmp/amdsmi_fake_sysfs_XXXXXX";
  const char* dir = mkdtemp(tmpl.data());
  if (dir == nullptr) {
    ADD_FAILURE() << "FakeSysfsTree: mkdtemp() failed";
    return;
  }
  root_ = dir;
}

FakeSysfsTree::~FakeSysfsTree() {
  if (root_.empty()) return;
  std::error_code ec;
  fs::remove_all(root_, ec);
}

void FakeSysfsTree::WriteFile(const std::string& relative_path, const std::string& content) {
  if (root_.empty()) return;  // mkdtemp() already failed; ADD_FAILURE()'d, avoid touching cwd.
  fs::path full_path = fs::path(root_) / relative_path;
  std::error_code ec;
  fs::create_directories(full_path.parent_path(), ec);
  std::ofstream out(full_path);
  out << content;
}

void FakeSysfsTree::RemoveFile(const std::string& relative_path) {
  if (root_.empty()) return;
  std::error_code ec;
  fs::remove(fs::path(root_) / relative_path, ec);
}

void FakeSysfsTree::SetPermissions(const std::string& relative_path, mode_t mode) {
  if (root_.empty()) return;
  if (chmod((fs::path(root_) / relative_path).c_str(), mode) != 0) {
    ADD_FAILURE() << "FakeSysfsTree::SetPermissions(" << relative_path << ", " << mode
                  << "): chmod() failed: " << strerror(errno);
  }
}

namespace amd::smi::testing {

ScopedProcessorRegistration::ScopedProcessorRegistration(amd::smi::AMDSmiProcessor* processor)
    : processor_(processor) {
  amd::smi::AMDSmiSystem::getInstance().register_processor_for_testing(processor_);
}

ScopedProcessorRegistration::~ScopedProcessorRegistration() {
  amd::smi::AMDSmiSystem::getInstance().unregister_processor_for_testing(processor_);
}

amdsmi_processor_handle ScopedProcessorRegistration::handle() const {
  return reinterpret_cast<amdsmi_processor_handle>(processor_);
}

}  // namespace amd::smi::testing

ScopedRocmSmiDevice::ScopedRocmSmiDevice(const std::string& path) {
  auto& devices = amd::smi::RocmSMI::getInstance().devices();
  index_ = static_cast<uint32_t>(devices.size());
  devices.push_back(std::make_shared<amd::smi::Device>(path, nullptr));
}

ScopedRocmSmiDevice::~ScopedRocmSmiDevice() {
  auto& devices = amd::smi::RocmSMI::getInstance().devices();
  if (index_ < devices.size()) devices.resize(index_);
}

ScopedAmdsmiInit::ScopedAmdsmiInit() {
  EXPECT_EQ(amdsmi_init(kNoHardwareDiscoveryInitFlags), AMDSMI_STATUS_SUCCESS);
}

ScopedAmdsmiInit::~ScopedAmdsmiInit() { amdsmi_shut_down(); }
