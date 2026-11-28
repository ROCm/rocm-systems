// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// amdsmi_get_gpu_fabric_info() against a synthetic device/ualink sysfs tree in the format amdgpu
// writes (drivers/gpu/drm/amd/amdgpu/amdgpu_ualink.c), through the public API with a registered
// synthetic GPU, so no fabric hardware is needed.

#include <gtest/gtest.h>

#include <cstdint>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_drm.h"
#include "amd_smi/impl/amd_smi_gpu_device.h"
#include "amd_smi/impl/amd_smi_test_overrides.h"
#include "test_common.h"
#include "test_fixture_utils.h"

namespace {

using amd::smi::AMDSmiDrm;
using amd::smi::AMDSmiGPUDevice;
using amd::smi::testing::ScopedOverride;
using amd::smi::testing::ScopedProcessorRegistration;
namespace test_hooks = amd::smi::test_hooks;

constexpr char kCard[] = "card_test_fabric";
constexpr uint32_t kUnset = UINT32_MAX;

// "0 1 2 ... count-1", as the kernel prints the set bits of the vPoD's accelerator bitmap.
std::string AcceleratorIdList(uint32_t count) {
  std::string ids;
  for (uint32_t id = 0; id < count; ++id) {
    ids += (id ? " " : "") + std::to_string(id);
  }
  return ids;
}

void WriteUalinkFile(FakeSysfsTree& tree, const std::string& name, const std::string& content) {
  tree.WriteFile(std::string(kCard) + "/device/ualink/" + name, content);
}

// One file per attribute, each ending in a newline as sysfs_emit() writes it.
void WriteUalinkTree(FakeSysfsTree& tree) {
  WriteUalinkFile(tree, "link_type", "UALoE\n");
  WriteUalinkFile(tree, "accel_id", "5\n");
  WriteUalinkFile(tree, "bandwidth", "12500\n");
  WriteUalinkFile(tree, "latency", "0\n");
  WriteUalinkFile(tree, "ppod_id", "00112233-4455-6677-8899-aabbccddeeff\n");
  WriteUalinkFile(tree, "ppod_size", "40\n");
  WriteUalinkFile(tree, "vpod_id", "3\n");
  WriteUalinkFile(tree, "vpod_size", "40\n");
  WriteUalinkFile(tree, "vpod_active_accels", AcceleratorIdList(40) + "\n");
  WriteUalinkFile(tree, "local_accels", "4 5 6 7\n");
  WriteUalinkFile(tree, "addr_mode", "source-aliasing\n");
  WriteUalinkFile(tree, "accel_state", "active\n");
}

std::vector<uint32_t> ActiveBitmap(const amdsmi_fabric_info_t& info) {
  const auto& words = info.fabric_info.v1.vpod_active_accelerators;
  return std::vector<uint32_t>(std::begin(words), std::end(words));
}

std::vector<uint32_t> EmptyBitmap() {
  return std::vector<uint32_t>(AMDSMI_FABRIC_ACTIVE_ACCELERATORS_BITMAP_SIZE, 0);
}

// Builds the synthetic device stack under tree.root() and calls the public API.
amdsmi_status_t ReadFabricInfo(const FakeSysfsTree& tree, amdsmi_fabric_info_t* info) {
  ScopedOverride<std::string> root_guard(test_hooks::SysfsDrmRoot, test_hooks::SetSysfsDrmRoot,
                                         tree.root());
  ScopedRocmSmiDevice rocm_device(tree.root() + "/" + kCard);
  AMDSmiDrm drm;
  AMDSmiGPUDevice gpu_device(rocm_device.index(), kCard, amdsmi_bdf_t{}, drm);
  ScopedProcessorRegistration registration(&gpu_device);

  const bool verbose = GetTestVerbosity() >= TestBase::VERBOSE_STANDARD;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_fabric_info", std::string("card=") + kCard, verbose);
  const amdsmi_status_t status = amdsmi_get_gpu_fabric_info(registration.handle(), info);
  DISPLAY_AMDSMI_STATUS(verbose, __FILE__, __LINE__, status, AMDSMI_STATUS_SUCCESS,
                        AMDSMI_STATUS_NO_DATA);
  return status;
}

TEST(GpuUnit, FabricInfoReadsKernelUalinkFormat) {
  ScopedAmdsmiInit smi_init;
  FakeSysfsTree tree;
  WriteUalinkTree(tree);

  amdsmi_fabric_info_t info{};
  ASSERT_EQ(ReadFabricInfo(tree, &info), AMDSMI_STATUS_SUCCESS);

  const auto& v1 = info.fabric_info.v1;
  EXPECT_EQ(info.fabric_version, 1u);
  EXPECT_EQ(v1.fabric_type, AMDSMI_FABRIC_TYPE_UALOE);
  EXPECT_EQ(v1.accelerator_id, 5u);
  EXPECT_EQ(v1.bandwidth, 12500u);
  EXPECT_EQ(v1.ppod_size, 40u);
  EXPECT_EQ(v1.vpod_id, 3u);
  EXPECT_EQ(v1.vpod_size, 40u);
  EXPECT_EQ(v1.addr_mode, AMDSMI_FABRIC_NPA_ADDRESS_MODE_SOURCE_ALIASING);
  EXPECT_EQ(v1.accel_state, AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_ACTIVE);

  // 40 active accelerators: IDs 0-31 fill the first word, 32-39 the low byte of the second.
  auto expected_bitmap = EmptyBitmap();
  expected_bitmap[0] = 0xFFFFFFFFu;
  expected_bitmap[1] = 0xFFu;
  EXPECT_EQ(ActiveBitmap(info), expected_bitmap);

  const std::vector<uint32_t> local(v1.local_accelerators, v1.local_accelerators + 4);
  EXPECT_EQ(local, (std::vector<uint32_t>{4, 5, 6, 7}));
  EXPECT_EQ(v1.local_accelerators[4], kUnset);
}

TEST(GpuUnit, FabricInfoAddrModeKernelStrings) {
  const std::vector<std::pair<std::string, amdsmi_fabric_npa_address_mode_t>> cases = {
      {"source-aliasing", AMDSMI_FABRIC_NPA_ADDRESS_MODE_SOURCE_ALIASING},
      {"source-identification", AMDSMI_FABRIC_NPA_ADDRESS_MODE_SOURCE_IDENTIFICATION},
      {"invalid", AMDSMI_FABRIC_NPA_ADDRESS_MODE_UNKNOWN},
  };

  ScopedAmdsmiInit smi_init;
  for (const auto& [sysfs_value, expected] : cases) {
    FakeSysfsTree tree;
    WriteUalinkTree(tree);
    WriteUalinkFile(tree, "addr_mode", sysfs_value + "\n");

    amdsmi_fabric_info_t info{};
    ASSERT_EQ(ReadFabricInfo(tree, &info), AMDSMI_STATUS_SUCCESS) << sysfs_value;
    EXPECT_EQ(info.fabric_info.v1.addr_mode, expected) << sysfs_value;
  }
}

// Every accel_state the kernel prints. "configured" is what kernels before the pPoD/vPoD split
// printed for the stage both new strings replace; "invalid" is printed for an out-of-range state.
TEST(GpuUnit, FabricInfoAccelStateKernelStrings) {
  const std::vector<std::pair<std::string, amdsmi_fabric_accelerator_vpod_state_t>> cases = {
      {"unconfigured", AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_UNCONFIGURED},
      {"ppod_configured", AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_CONFIGURED},
      {"vpod_configured", AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_CONFIGURED},
      {"configured", AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_CONFIGURED},
      {"ready", AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_READY},
      {"active", AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_ACTIVE},
      {"error", AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_ERROR},
      {"invalid", AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_UNKNOWN},
  };

  ScopedAmdsmiInit smi_init;
  for (const auto& [sysfs_value, expected] : cases) {
    FakeSysfsTree tree;
    WriteUalinkTree(tree);
    WriteUalinkFile(tree, "accel_state", sysfs_value + "\n");

    amdsmi_fabric_info_t info{};
    ASSERT_EQ(ReadFabricInfo(tree, &info), AMDSMI_STATUS_SUCCESS) << sysfs_value;
    EXPECT_EQ(info.fabric_info.v1.accel_state, expected) << sysfs_value;
  }
}

// Bit N % 32 of word N / 32 is accelerator N, across word boundaries and up to the last bit.
TEST(GpuUnit, FabricInfoVpodActiveAcceleratorsSparseBitmap) {
  ScopedAmdsmiInit smi_init;
  FakeSysfsTree tree;
  WriteUalinkTree(tree);
  WriteUalinkFile(tree, "vpod_active_accels", "0 31 32 255 1023\n");

  amdsmi_fabric_info_t info{};
  ASSERT_EQ(ReadFabricInfo(tree, &info), AMDSMI_STATUS_SUCCESS);

  auto expected_bitmap = EmptyBitmap();
  expected_bitmap[0] = 0x80000001u;
  expected_bitmap[1] = 0x1u;
  expected_bitmap[7] = 0x80000000u;
  expected_bitmap[31] = 0x80000000u;
  EXPECT_EQ(ActiveBitmap(info), expected_bitmap);
}

// IDs the bitmap cannot hold and tokens that are not IDs are skipped, not written out of bounds.
TEST(GpuUnit, FabricInfoVpodActiveAcceleratorsSkipsUnrepresentableIds) {
  ScopedAmdsmiInit smi_init;
  FakeSysfsTree tree;
  WriteUalinkTree(tree);
  WriteUalinkFile(tree, "vpod_active_accels", "-1 1024 4294967295 bad 3\n");

  amdsmi_fabric_info_t info{};
  ASSERT_EQ(ReadFabricInfo(tree, &info), AMDSMI_STATUS_SUCCESS);

  auto expected_bitmap = EmptyBitmap();
  expected_bitmap[0] = 0x8u;
  EXPECT_EQ(ActiveBitmap(info), expected_bitmap);
}

// An unconfigured vPoD has empty accelerator lists: the kernel prints a bare newline. No
// accelerator is active, and no local accelerator ID is reported, not ID 0.
TEST(GpuUnit, FabricInfoEmptyAcceleratorLists) {
  ScopedAmdsmiInit smi_init;
  FakeSysfsTree tree;
  WriteUalinkTree(tree);
  WriteUalinkFile(tree, "vpod_active_accels", "\n");
  WriteUalinkFile(tree, "local_accels", "\n");

  amdsmi_fabric_info_t info{};
  ASSERT_EQ(ReadFabricInfo(tree, &info), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(ActiveBitmap(info), EmptyBitmap());
  const auto& local = info.fabric_info.v1.local_accelerators;
  EXPECT_EQ(std::vector<uint32_t>(std::begin(local), std::end(local)),
            std::vector<uint32_t>(AMDSMI_FABRIC_MAX_LOCAL_GPUS, kUnset));
}

// The version names the payload only once a v1 field was read: a single readable attribute is
// enough, and without one (no ualink directory, or nothing usable in it) it keeps its sentinel.
TEST(GpuUnit, FabricInfoVersionFollowsReadData) {
  ScopedAmdsmiInit smi_init;
  FakeSysfsTree tree;

  amdsmi_fabric_info_t info{};
  ASSERT_EQ(ReadFabricInfo(tree, &info), AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(info.fabric_version, kUnset);

  WriteUalinkFile(tree, "accel_state", "");
  ASSERT_EQ(ReadFabricInfo(tree, &info), AMDSMI_STATUS_NO_DATA);
  EXPECT_EQ(info.fabric_version, kUnset);

  WriteUalinkFile(tree, "accel_id", "53\n");
  ASSERT_EQ(ReadFabricInfo(tree, &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(info.fabric_version, 1u);
  EXPECT_EQ(info.fabric_info.v1.accelerator_id, 53u);
}

}  // namespace
