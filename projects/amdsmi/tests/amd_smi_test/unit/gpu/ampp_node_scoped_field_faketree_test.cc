// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Hardware-free regression guard for the primary/non-primary gating in
// rocm_smi_ampp.cc: unlike ampp_node_scoped_field_gating_test.cc (which
// exercises real/mocked hardware and GTEST_SKIP_s when no such hardware is
// present), these tests fabricate their own amd::smi::Device backed by a
// temp directory -- so the is_primary_gpu() guard at rsmi_dev_ampp_fields_get()
// and rsmi_dev_ampp_profile_configure() (rocm_smi_ampp.cc) is exercised
// unconditionally in CI and fails if either guard is removed, regardless of
// what hardware the runner has.
//
// This is possible for AMPP (unlike NPM's board_path) because
// Device::get_ampp_root_path() derives entirely from the Device's own path_,
// and amd::smi::RocmSMI::devices() is a public, mutable vector -- so a test
// can append a fake Device and address it by its index (dv_ind) through the
// rsmi_dev_ampp_* entry points directly, without any new test-injection hook
// in production headers.
#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_ampp.h"
#include "rocm_smi/rocm_smi_device.h"
#include "rocm_smi/rocm_smi_main.h"

namespace fs = std::filesystem;

namespace {

void WriteFile(const fs::path& path, const std::string& contents) {
  fs::create_directories(path.parent_path());
  std::ofstream ofs(path);
  ofs << contents;
}

std::string ReadFile(const fs::path& path) {
  std::ifstream ifs(path);
  std::string line;
  std::getline(ifs, line);
  return line;
}

// Builds a fake app_modes/ tree under a temp dir, appends a Device pointed at
// it to the live amd::smi::RocmSMI::devices() vector, and pops it back off on
// destruction -- so the global device list is only perturbed for the
// lifetime of a single TEST() body (gtest runs this binary's TEST()s
// sequentially within one process, see tests/README.md).
class ScopedFakeAmppDevice {
 public:
  explicit ScopedFakeAmppDevice(uint16_t xgmi_physical_id) {
    root_ = fs::temp_directory_path() /
            fs::path("amdsmi_ampp_faketree_" + std::to_string(reinterpret_cast<uintptr_t>(this)));
    fs::create_directories(root_ / "device" / "app_modes" / "profile_5");
    fs::create_directories(root_ / "device" / "app_modes" / "config");

    // xgmi_physical_id is read via strtoul(..., 16): plain decimal digits
    // below 16 (0, 3) are valid hex too, so no "0x" prefix is needed.
    WriteFile(root_ / "device" / "xgmi_physical_id", std::to_string(xgmi_physical_id));

    // profile_5 is the one writable custom slot (bit 5 of the mask below);
    // its field files live directly under it for the read path.
    WriteFile(root_ / "device" / "app_modes" / "profile_5" / "node_power_limit", "100 W");
    WriteFile(root_ / "device" / "app_modes" / "profile_5" / "PPT0_LIMIT", "200 W");

    // config/ is the shared staging area the write path validates/commits
    // against (rsmi_dev_ampp_profile_configure() in rocm_smi_ampp.cc).
    WriteFile(root_ / "device" / "app_modes" / "config" / "writable_slot_mask", "0x20");
    WriteFile(root_ / "device" / "app_modes" / "config" / "profile", "");
    WriteFile(root_ / "device" / "app_modes" / "config" / "node_power_limit", "");
    WriteFile(root_ / "device" / "app_modes" / "config" / "commit", "");

    auto& devices = amd::smi::RocmSMI::getInstance().devices();
    devices.push_back(std::make_shared<amd::smi::Device>(root_.string(), nullptr));
    dv_ind_ = static_cast<uint32_t>(devices.size() - 1);
  }

  ~ScopedFakeAmppDevice() {
    auto& devices = amd::smi::RocmSMI::getInstance().devices();
    devices.pop_back();
    std::error_code ec;
    fs::remove_all(root_, ec);
  }

  uint32_t dv_ind() const { return dv_ind_; }

  std::string ReadCommit() const {
    return ReadFile(root_ / "device" / "app_modes" / "config" / "commit");
  }

 private:
  fs::path root_;
  uint32_t dv_ind_;
};

}  // namespace

// ---------------------------------------------------------------------
// Read path: rsmi_dev_ampp_fields_get()
// ---------------------------------------------------------------------

TEST(GpuUnit, AmppFakeTreeNodeScopedFieldHiddenOnNonPrimaryGpu) {
  ScopedFakeAmppDevice fake(/*xgmi_physical_id=*/3);

  uint32_t num_fields = 0;
  ASSERT_EQ(rsmi_dev_ampp_fields_get(fake.dv_ind(), "profile_5", &num_fields, nullptr),
            RSMI_STATUS_SUCCESS);
  ASSERT_EQ(num_fields, 1u);

  std::vector<rsmi_ampp_field_t> fields(num_fields);
  ASSERT_EQ(rsmi_dev_ampp_fields_get(fake.dv_ind(), "profile_5", &num_fields, fields.data()),
            RSMI_STATUS_SUCCESS);
  EXPECT_STREQ(fields[0].name, "PPT0_LIMIT");
}

TEST(GpuUnit, AmppFakeTreeNodeScopedFieldVisibleOnPrimaryGpu) {
  ScopedFakeAmppDevice fake(/*xgmi_physical_id=*/0);

  uint32_t num_fields = 0;
  ASSERT_EQ(rsmi_dev_ampp_fields_get(fake.dv_ind(), "profile_5", &num_fields, nullptr),
            RSMI_STATUS_SUCCESS);
  ASSERT_EQ(num_fields, 2u);
}

// ---------------------------------------------------------------------
// Write path: rsmi_dev_ampp_profile_configure()
// ---------------------------------------------------------------------

TEST(GpuUnit, AmppFakeTreeNodeScopedFieldCommitNeverWrittenOnNonPrimaryGpu) {
  ScopedFakeAmppDevice fake(/*xgmi_physical_id=*/3);
  ASSERT_EQ(fake.ReadCommit(), "");

  rsmi_ampp_field_t field{};
  std::snprintf(field.name, sizeof(field.name), "%s", "node_power_limit");
  field.value = 42;

  EXPECT_EQ(rsmi_dev_ampp_profile_configure(fake.dv_ind(), "profile_5", &field, 1),
            RSMI_STATUS_INVALID_ARGS);
  // The guard this PR adds must reject before any sysfs write -- if it were
  // removed, config/commit would read back "1".
  EXPECT_EQ(fake.ReadCommit(), "");
}

TEST(GpuUnit, AmppFakeTreeNodeScopedFieldCommitWrittenOnPrimaryGpu) {
  ScopedFakeAmppDevice fake(/*xgmi_physical_id=*/0);
  ASSERT_EQ(fake.ReadCommit(), "");

  rsmi_ampp_field_t field{};
  std::snprintf(field.name, sizeof(field.name), "%s", "node_power_limit");
  field.value = 42;

  EXPECT_EQ(rsmi_dev_ampp_profile_configure(fake.dv_ind(), "profile_5", &field, 1),
            RSMI_STATUS_SUCCESS);
  EXPECT_EQ(fake.ReadCommit(), "1");
}
