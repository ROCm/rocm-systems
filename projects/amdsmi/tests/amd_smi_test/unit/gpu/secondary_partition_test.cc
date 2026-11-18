// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// A secondary compute partition reads the partition mode of its GPU from the GPU's PCI device.
// amdgpu creates the partition attributes only on the PCI device. The render node of a secondary
// partition belongs to an amdgpu_xcp platform device, so a container given only that partition
// saw no mode. Runs the real discovery over a fake KFD, DRM and PCI tree, with no GPU.

#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_test_overrides.h"
#include "test_fixture_utils.h"

namespace {

namespace fs = std::filesystem;
using amd::smi::testing::ScopedOverride;

constexpr uint32_t kNode = 1;
constexpr uint32_t kBus = 0x85;
constexpr uint32_t kPartition = 1;
constexpr const char* kParent = "pci/0000:85:00.0/";

// KFD reports the partition at the address of its GPU, with the partition index in the function
// bits. Its render node belongs to the platform device amdgpu_xcp_3, not to the PCI device.
void WriteSecondaryPartition(FakeSysfsTree& tree) {
  tree.WriteFile(std::to_string(kNode) + "/gpu_id", "4242\n");
  tree.WriteFile(std::to_string(kNode) + "/properties",
                 "hive_id 0\nsimd_arrays_per_engine 1\narray_count 1\ncu_per_simd_array 1\n"
                 "location_id " +
                     std::to_string((kBus << 8) | kPartition) +
                     "\ndomain 0\ndrm_render_minor 129\n");
  tree.WriteFile(std::to_string(kNode) + "/io_links/.keep", "");
  tree.WriteFile("amdgpu_xcp_3/drm/card1/.keep", "");
  tree.WriteFile("amdgpu_xcp_3/drm/renderD129/.keep", "");
  const fs::path root(tree.root());
  fs::create_directory(root / "renderD129");
  fs::create_directory_symlink("../amdgpu_xcp_3", root / "renderD129" / "device");
  fs::create_directory_symlink("renderD129", root / "card1");
}

void WriteParent(FakeSysfsTree& tree, const std::string& mode, const std::string& isolation) {
  const std::string parent = kParent;
  tree.WriteFile(parent + "current_compute_partition", mode + "\n");
  tree.WriteFile(parent + "available_compute_partition", "SPX, DPX, QPX, CPX\n");
  tree.WriteFile(parent + "compute_partition_config/supported_xcp_configs", "SPX, DPX, QPX, CPX\n");
  tree.WriteFile(parent + "current_memory_partition", "NPS1\n");
  tree.WriteFile(parent + "available_memory_partition", "NPS1, NPS4\n");
  tree.WriteFile(parent + "enforce_isolation", isolation + "\n");
}

class PciRoot {
 public:
  explicit PciRoot(const FakeSysfsTree& tree, const std::string& dir = "pci")
      : guard_(amd::smi::test_hooks::SysfsPciDevicesRoot,
               amd::smi::test_hooks::SetSysfsPciDevicesRoot, tree.root() + "/" + dir) {}

 private:
  ScopedOverride<std::string> guard_;
};

}  // namespace

TEST(GpuUnit, SecondaryPartitionReportsTheModeOfItsGpu) {
  FakeSysfsTree tree;
  WriteSecondaryPartition(tree);
  WriteParent(tree, "DPX", "0 0");
  PciRoot pci(tree);
  FakeKfdTopology topology(tree);
  const amdsmi_processor_handle handle = topology.handle(kNode);

  amdsmi_accelerator_partition_profile_t profile{};
  uint32_t partition_id[8] = {};
  ASSERT_EQ(amdsmi_get_gpu_accelerator_partition_profile(handle, &profile, partition_id),
            AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(profile.profile_type, AMDSMI_ACCELERATOR_PARTITION_DPX);
  EXPECT_EQ(profile.num_partitions, 2u);
  EXPECT_EQ(profile.profile_index, 1u);
  EXPECT_EQ(profile.memory_caps.nps_flags.nps4_cap, 1u);
  EXPECT_EQ(partition_id[0], kPartition);

  char mode[16] = {};
  ASSERT_EQ(amdsmi_get_gpu_compute_partition(handle, mode, sizeof(mode)), AMDSMI_STATUS_SUCCESS);
  EXPECT_STREQ(mode, "DPX");
  char nps[16] = {};
  ASSERT_EQ(amdsmi_get_gpu_memory_partition(handle, nps, sizeof(nps)), AMDSMI_STATUS_SUCCESS);
  EXPECT_STREQ(nps, "NPS1");
}

// CPX has no fixed partition count. Without gpu_metrics, the count is the number of per-partition
// values in the GPU's enforce_isolation.
TEST(GpuUnit, SecondaryCpxPartitionCountsThePartitionsOfItsGpu) {
  FakeSysfsTree tree;
  WriteSecondaryPartition(tree);
  WriteParent(tree, "CPX", "0 0 0 0 0 0 0 0");
  PciRoot pci(tree);
  FakeKfdTopology topology(tree);

  amdsmi_accelerator_partition_profile_t profile{};
  uint32_t partition_id[8] = {};
  ASSERT_EQ(
      amdsmi_get_gpu_accelerator_partition_profile(topology.handle(kNode), &profile, partition_id),
      AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(profile.profile_type, AMDSMI_ACCELERATOR_PARTITION_CPX);
  EXPECT_EQ(profile.num_partitions, 8u);
}

// A set on a secondary partition returns what it returned before the partition read the attributes
// of its GPU, also for the GPU's current mode, and does not write the GPU's attributes.
TEST(GpuUnit, SecondaryPartitionSetsFailAsBefore) {
  FakeSysfsTree tree;
  WriteSecondaryPartition(tree);
  WriteParent(tree, "DPX", "0 0");
  FakeKfdTopology topology(tree);
  const amdsmi_processor_handle handle = topology.handle(kNode);
  const auto set_all = [handle] {
    return std::vector<amdsmi_status_t>{
        amdsmi_set_gpu_compute_partition(handle, AMDSMI_COMPUTE_PARTITION_DPX),
        amdsmi_set_gpu_compute_partition(handle, AMDSMI_COMPUTE_PARTITION_SPX),
        amdsmi_set_gpu_accelerator_partition_profile(handle, 1),
        amdsmi_set_gpu_memory_partition_mode(handle, AMDSMI_MEMORY_PARTITION_NPS1),
        amdsmi_set_gpu_process_isolation(handle, 1),
    };
  };

  std::vector<amdsmi_status_t> without_gpu_files;
  {
    PciRoot no_pci(tree, "no_pci");
    without_gpu_files = set_all();
  }
  PciRoot pci(tree);
  const std::vector<amdsmi_status_t> with_gpu_files = set_all();

  EXPECT_EQ(with_gpu_files, without_gpu_files);
  for (const amdsmi_status_t status : with_gpu_files) {
    EXPECT_NE(status, AMDSMI_STATUS_SUCCESS);
  }
  EXPECT_EQ(tree.ReadFile(std::string(kParent) + "current_compute_partition"), "DPX");
  EXPECT_EQ(tree.ReadFile(std::string(kParent) + "current_memory_partition"), "NPS1");
  EXPECT_EQ(tree.ReadFile(std::string(kParent) + "enforce_isolation"), "0 0");
  EXPECT_FALSE(fs::exists(tree.root() + "/amdgpu_xcp_3/current_compute_partition"));
}

// A device whose render node belongs to a PCI device, for example an SR-IOV VF, reports only its
// own attributes, even when a PCI device at function 0 of its address has a mode.
TEST(GpuUnit, PciDeviceWithoutModeDoesNotReadAnotherFunction) {
  constexpr uint32_t kGpuNode = 1;
  constexpr uint32_t kGpuCard = 0;
  FakeSysfsTree tree;
  WriteGpuNode(tree, kGpuNode, kGpuCard, 4242);
  tree.WriteFile("pci/0000:02:00.0/current_compute_partition", "DPX\n");
  PciRoot pci(tree);
  FakeKfdTopology topology(tree);

  char mode[16] = {};
  EXPECT_EQ(amdsmi_get_gpu_compute_partition(topology.handle(kGpuNode), mode, sizeof(mode)),
            AMDSMI_STATUS_NOT_SUPPORTED);
}
