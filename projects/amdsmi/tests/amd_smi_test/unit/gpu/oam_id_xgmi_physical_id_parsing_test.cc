// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Regression test: amdgpu prints xgmi_physical_id (the sysfs source of
// amdsmi_asic_info_t.oam_id) in decimal, but get_id() in rocm_smi.cc used to
// parse every id it reads -- including this one -- as hex. A board reporting
// "18" (decimal) would misparse as 0x18 == 24. Exercises the real
// rsmi_dev_xgmi_physical_id_get() -> get_id() path with a fabricated
// rocm_smi::Device, mirroring fan_speed_source_test.cc's pattern for a
// similarly low-level sysfs-parsing fix. No WSL-backend short-circuit exists
// at this layer (that gate lives in amd_smi.cc's amdsmi_* wrappers, not in
// rocm_smi.cc), so SKIP_IF_WSL_BACKEND_ACTIVE() does not apply here.

#include <gtest/gtest.h>

#include <cstdint>

#include "rocm_smi/rocm_smi.h"
#include "test_fixture_utils.h"

namespace {

constexpr uint16_t kDecimalPhysicalId = 18;    // sysfs content: "18"
constexpr uint16_t kHexMisparsedValue = 0x18;  // what base-16 parsing would yield instead

}  // namespace

TEST(GpuUnit, XgmiPhysicalIdParsesSysfsValueAsDecimalNotHex) {
  FakeSysfsTree tree;
  tree.WriteFile("card0/device/xgmi_physical_id", "18\n");
  ScopedRocmSmiDevice device(tree.root() + "/card0");

  uint16_t id = 0;
  ASSERT_EQ(rsmi_dev_xgmi_physical_id_get(device.index(), &id), RSMI_STATUS_SUCCESS);
  EXPECT_EQ(id, kDecimalPhysicalId);
  EXPECT_NE(id, kHexMisparsedValue);
}

// Every other id get_id() reads (e.g. "revision") must stay on the base-16
// branch: amdgpu prints those in hex, unlike xgmi_physical_id.
TEST(GpuUnit, RevisionIdStillParsesSysfsValueAsHex) {
  FakeSysfsTree tree;
  tree.WriteFile("card0/device/revision", "18\n");
  ScopedRocmSmiDevice device(tree.root() + "/card0");

  uint16_t revision = 0;
  ASSERT_EQ(rsmi_dev_revision_get(device.index(), &revision), RSMI_STATUS_SUCCESS);
  EXPECT_EQ(revision, kHexMisparsedValue);
  EXPECT_NE(revision, kDecimalPhysicalId);
}
