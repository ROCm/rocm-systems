// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// A spatial partition has no serial or config space of its own and its
// parent's BDF names the whole GPU, so it is adopted only from the values the
// kernel publishes under its own node. These tests fabricate that node.

#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "src/cuid_gpu.h"
#include "src/cuid_util.h"
#include "test_common.h"

namespace {

class FakeSysfs {
 public:
  FakeSysfs() : dir_("cuid_partition_test_"), root_(dir_.path()) {}

  const std::string& device_path() const { return root_; }

  void PublishPartition() {
    mkdir((root_ + "/xcp").c_str(), 0755);
    Write(root_ + "/xcp/cuid_unit_id", "2");
  }

 private:
  static void Write(const std::string& path, const char* value) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << value << "\n";  // the kernel emits a trailing newline
  }
  ScopedTempDir dir_;
  const std::string root_;
};

}  // namespace

TEST(cuidtst, PartitionIsSkippedWhenTheDriverPublishesNothing) {
  FakeSysfs fake;
  ASSERT_FALSE(fake.device_path().empty());

  // No xcp directory (older driver, amdgpu.cuid=0, or unpartitionable): skip
  // rather than fabricate an identifier every partition would share.
  EXPECT_TRUE(CuidGpu::partition_attr_dir_for_device(fake.device_path()).empty());

  amdcuid_gpu_info info = {};
  EXPECT_EQ(CuidGpu::discover_partition(&info, fake.device_path()), AMDCUID_STATUS_UNSUPPORTED);
}

TEST(cuidtst, PartitionIsNamedByItsOwnNode) {
  FakeSysfs fake;
  ASSERT_FALSE(fake.device_path().empty());
  fake.PublishPartition();

  const std::string attr_dir = CuidGpu::partition_attr_dir_for_device(fake.device_path());
  ASSERT_EQ(attr_dir, fake.device_path() + "/xcp");

  // Idempotent: handed the partition node itself, it stays there rather than
  // looking for an xcp directory inside it.
  EXPECT_EQ(CuidGpu::partition_attr_dir_for_device(attr_dir), attr_dir);

  amdcuid_gpu_info info = {};
  ASSERT_EQ(CuidGpu::discover_partition(&info, fake.device_path()), AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(info.partition_attr_dir, attr_dir);
  // The device node is the partition's, not the card's: two partitions of one
  // GPU must not resolve to the same thing.
  EXPECT_EQ(info.render_node, attr_dir);
  EXPECT_EQ(info.header.device_type, AMDCUID_DEVICE_TYPE_GPU);
  EXPECT_TRUE(info.bdf.empty()) << "a partition must not claim its parent's BDF";
}

TEST(cuidtst, UnnamedVfCannotAdoptPartitionAttributes) {
  FakeSysfs fake;
  ASSERT_FALSE(fake.device_path().empty());
  fake.PublishPartition();

  amdcuid_gpu_info info = {};
  info.header.device_type = AMDCUID_DEVICE_TYPE_GPU;
  info.render_node = fake.device_path() + "/xcp";
  info.unnamed_vf = true;
  CuidGpu vf(info);

  EXPECT_FALSE(vf.get_info().partition_metadata_valid);
  uint64_t fingerprint = 0xDEADBEEF;
  if (geteuid() == 0) {
    EXPECT_EQ(vf.get_hardware_fingerprint(fingerprint), AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND);
    EXPECT_EQ(fingerprint, 0u);
  } else {
    EXPECT_EQ(vf.get_hardware_fingerprint(fingerprint), AMDCUID_STATUS_PERMISSION_DENIED);
  }
  amdcuid_primary_id primary = {};
  EXPECT_EQ(vf.get_primary_cuid(primary), AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND);
  amdcuid_derived_id derived = {};
  EXPECT_EQ(vf.get_derived_cuid(derived), AMDCUID_STATUS_UNSUPPORTED);
}

namespace {

void WriteFile(const std::string& path, const std::string& value) {
  std::ofstream(path, std::ios::binary | std::ios::trunc) << value << "\n";
}

void MakeDirs(const std::string& path) {
  for (size_t at = path.find('/', 1); at != std::string::npos; at = path.find('/', at + 1))
    mkdir(path.substr(0, at).c_str(), 0755);
  mkdir(path.c_str(), 0755);
}

// fffe:63:00.0 with two partitions and no cuid_derived anywhere: the first
// partition's node is the PCI device, the second an amdgpu_xcp platform device
// that only the KFD topology ties back to its parent.
class UnpublishedPartitions {
 public:
  UnpublishedPartitions() : dir_("cuid_partition_temp_"), root_(dir_.path()) {
    const std::string pci = root_ + "/pci/fffe:63:00.0";
    MakeDirs(pci + "/drm/renderD128");
    MakeDirs(pci + "/xcp");
    WriteFile(pci + "/vendor", "0x1002");
    WriteFile(pci + "/device", "0x73a3");
    WriteFile(pci + "/revision", "0x00");
    WriteFile(pci + "/xcp/cuid_unit_id", "2");
    const std::string xcp = root_ + "/platform/amdgpu_xcp_1";
    MakeDirs(xcp + "/drm/renderD129");
    MakeDirs(xcp + "/xcp");
    WriteFile(xcp + "/xcp/cuid_unit_id", "3");
    MakeDirs(root_ + "/kfd/1");
    MakeDirs(root_ + "/kfd/2");
    WriteFile(root_ + "/kfd/1/properties",
              "drm_render_minor 128\ndomain 65534\nlocation_id 25344\nunique_id 7");
    WriteFile(root_ + "/kfd/2/properties",
              "drm_render_minor 129\ndomain 65534\nlocation_id 25345\nunique_id 7");
  }

  CuidGpu Partition(const std::string& device) const {
    amdcuid_gpu_info info = {};
    info.header.device_type = AMDCUID_DEVICE_TYPE_GPU;
    info.render_node = device + "/xcp";
    info.partition_attr_dir = info.render_node;
    info.pci_root = root_ + "/pci";
    info.kfd_nodes = root_ + "/kfd";
    return CuidGpu(info);
  }
  CuidGpu First() const { return Partition(root_ + "/pci/fffe:63:00.0"); }
  CuidGpu Second() const { return Partition(root_ + "/platform/amdgpu_xcp_1"); }
  const std::string& root() const { return root_; }

 private:
  ScopedTempDir dir_;
  const std::string root_;
};

std::string Uuid(const amdcuid_derived_id& id) {
  return CuidUtilities::get_cuid_as_string(&id.UUIDv8_representation);
}

// The temporary CUID of fffe:63:00.0 1002:73a3 rev 00 with `unit_id`.
std::string Expected(uint16_t unit_id) {
  CuidUtilities::AuxiliaryInput aux;
  aux.format = CuidUtilities::kAuxFormatPcie;
  CuidUtilities::set_pcie_routing(aux, "fffe:63:00.0");
  aux.device_id = 0x73A3;
  aux.vendor_id = 0x1002;
  aux.component_type = AMDCUID_DEVICE_TYPE_GPU;
  aux.unit_id = unit_id;
  amdcuid_derived_id derived = {};
  EXPECT_EQ(CuidUtilities::temporary_cuid(aux, &derived), AMDCUID_STATUS_SUCCESS);
  return Uuid(derived);
}

}  // namespace

TEST(cuidtst, PartitionParentIsFoundThroughKfd) {
  UnpublishedPartitions fake;
  CuidPartitionParent parent;
  ASSERT_EQ(CuidGpu::partition_parent(fake.root() + "/platform/amdgpu_xcp_1/xcp",
                                      fake.root() + "/pci", fake.root() + "/kfd", parent),
            AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(parent.bdf, "fffe:63:00.0");
  EXPECT_EQ(parent.vendor_id, 0x1002);
  EXPECT_EQ(parent.device_id, 0x73A3);
  ASSERT_EQ(CuidGpu::partition_parent(fake.root() + "/pci/fffe:63:00.0/xcp", fake.root() + "/pci",
                                      fake.root() + "/kfd", parent),
            AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(parent.bdf, "fffe:63:00.0");
}

// Two partitions of one GPU, neither with a usable cuid_derived, get distinct
// temporary CUIDs, and the whole GPU keeps the one it had.
TEST(cuidtst, UnpublishedPartitionsGetDistinctTemporaryCuids) {
  uint8_t machine_id[16];
  if (CuidUtilities::host_machine_id(machine_id) != AMDCUID_STATUS_SUCCESS)
    GTEST_SKIP() << "no machine-id, so no temporary CUID";
  UnpublishedPartitions fake;
  amdcuid_derived_id first = {}, second = {}, whole = {};
  const CuidGpu first_gpu = fake.First();
  const CuidGpu second_gpu = fake.Second();
  ASSERT_EQ(first_gpu.get_derived_cuid(first, nullptr), AMDCUID_STATUS_SUCCESS);
  ASSERT_EQ(second_gpu.get_derived_cuid(second, nullptr), AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(first_gpu.derived_source(), AMDCUID_SOURCE_LIBRARY);
  bool temporary = false;
  ASSERT_EQ(second_gpu.is_temporary_cuid(&temporary, nullptr), AMDCUID_STATUS_SUCCESS);
  EXPECT_TRUE(temporary);

  amdcuid_gpu_info info = {};
  info.bdf = "fffe:63:00.0";
  info.render_node = "/nonexistent/cuid-partition-parent";
  info.header.fields.gpu.device_id = 0x73A3;
  info.header.fields.gpu.vendor_id = 0x1002;
  ASSERT_EQ(CuidGpu(info).get_derived_cuid(whole, nullptr), AMDCUID_STATUS_SUCCESS);

  EXPECT_EQ(Uuid(first), Expected(2));
  EXPECT_EQ(Uuid(second), Expected(3));
  EXPECT_EQ(Uuid(whole), Expected(0));
  EXPECT_NE(Uuid(first), Uuid(second));
  EXPECT_NE(Uuid(first), Uuid(whole));
  EXPECT_NE(Uuid(second), Uuid(whole));
}

// A stock amdgpu publishes no cuid_unit_id. KFD's location_id carries the
// partition index k in the parent's function bits, and n is the number of KFD
// GPU nodes with that parent, so partition k of n is UnitID n*(n-1)/2 + 1 + k,
// the value amdgpu would publish. Here the GPU is in DPX: 2 and 3.
TEST(cuidtst, StockPartitionsTakeTheirUnitIdFromKfd) {
  uint8_t machine_id[16];
  if (CuidUtilities::host_machine_id(machine_id) != AMDCUID_STATUS_SUCCESS)
    GTEST_SKIP() << "no machine-id, so no temporary CUID";
  UnpublishedPartitions fake;
  const std::string& root = fake.root();
  unlink((root + "/pci/fffe:63:00.0/xcp/cuid_unit_id").c_str());
  unlink((root + "/platform/amdgpu_xcp_1/xcp/cuid_unit_id").c_str());
  WriteFile(root + "/kfd/1/properties", "drm_render_minor 128\ndomain 65534\nlocation_id 25344");
  WriteFile(root + "/kfd/2/properties", "drm_render_minor 129\ndomain 65534\nlocation_id 25345");

  for (const auto& c : {std::make_pair(root + "/pci/fffe:63:00.0/xcp", 2),
                        std::make_pair(root + "/platform/amdgpu_xcp_1/xcp", 3)}) {
    SCOPED_TRACE(c.first);
    CuidPartitionParent parent;
    ASSERT_EQ(CuidGpu::partition_parent(c.first, root + "/pci", root + "/kfd", parent),
              AMDCUID_STATUS_SUCCESS);
    EXPECT_EQ(parent.bdf, "fffe:63:00.0");
    EXPECT_EQ(parent.kfd_unit_id, c.second);

    amdcuid_gpu_info info = {};
    info.partition_attr_dir = c.first;
    info.render_node = c.first;
    info.pci_root = root + "/pci";
    info.kfd_nodes = root + "/kfd";
    const CuidGpu partition(info);
    amdcuid_derived_id derived = {};
    ASSERT_EQ(partition.get_derived_cuid(derived, nullptr), AMDCUID_STATUS_SUCCESS);
    EXPECT_EQ(Uuid(derived), Expected(static_cast<uint16_t>(c.second)));
  }

  // One KFD node for the GPU is SPX: no partition, so no UnitID of its own.
  unlink((root + "/kfd/2/properties").c_str());
  CuidPartitionParent parent;
  ASSERT_EQ(CuidGpu::partition_parent(root + "/pci/fffe:63:00.0/xcp", root + "/pci", root + "/kfd",
                                      parent),
            AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(parent.kfd_unit_id, 0);
}

// A partition index outside the number of nodes its parent has is not a
// layout the rule knows: that partition has no UnitID and so no CUID, never 0.
TEST(cuidtst, InconsistentKfdNodesGiveNoUnitId) {
  UnpublishedPartitions fake;
  const std::string& root = fake.root();
  unlink((root + "/pci/fffe:63:00.0/xcp/cuid_unit_id").c_str());
  unlink((root + "/platform/amdgpu_xcp_1/xcp/cuid_unit_id").c_str());
  WriteFile(root + "/kfd/2/properties", "drm_render_minor 129\ndomain 65534\nlocation_id 25346");
  CuidPartitionParent parent;
  EXPECT_NE(CuidGpu::partition_parent(root + "/platform/amdgpu_xcp_1/xcp", root + "/pci",
                                      root + "/kfd", parent),
            AMDCUID_STATUS_SUCCESS);

  amdcuid_gpu_info info = {};
  info.partition_attr_dir = root + "/platform/amdgpu_xcp_1/xcp";
  info.render_node = info.partition_attr_dir;
  info.pci_root = root + "/pci";
  info.kfd_nodes = root + "/kfd";
  amdcuid_derived_id derived = {};
  EXPECT_NE(CuidGpu(info).get_derived_cuid(derived, nullptr), AMDCUID_STATUS_SUCCESS);
}

// The KFD vectors of the CUID specification, on an MI350X at 0000:f5:00.0.
TEST(cuidtst, KfdUnitIdVectors) {
  const ScopedTempDir dir("cuid_kfd_vectors_");
  ASSERT_FALSE(dir.path().empty());
  const std::string root = dir.path();
  const std::string pci = root + "/pci/0000:f5:00.0";
  MakeDirs(pci + "/drm/renderD128");
  MakeDirs(pci + "/xcp");
  WriteFile(pci + "/vendor", "0x1002");
  WriteFile(pci + "/device", "0x75a0");
  WriteFile(pci + "/revision", "0x00");
  for (int i = 1; i < 8; ++i) {
    const std::string xcp = root + "/platform/amdgpu_xcp_" + std::to_string(i);
    MakeDirs(xcp + "/drm/renderD" + std::to_string(128 + i));
    MakeDirs(xcp + "/xcp");
  }
  struct Case {
    const char* name;
    std::vector<int> locations;  // node i renders minor 128 + i
    int target;                  // index into locations
    bool ok;
    uint16_t unit_id;
  };
  const Case cases[] = {
      {"K-SPX", {0xf500}, 0, true, 0},
      {"K-QPX", {0xf500, 0xf501, 0xf502, 0xf503}, 2, true, 9},
      {"K-CPX", {0xf500, 0xf501, 0xf502, 0xf503, 0xf504, 0xf505, 0xf506, 0xf507}, 7, true, 36},
      {"K-GAP", {0xf500, 0xf501, 0xf503}, 1, false, 0},
      {"K-DUP", {0xf500, 0xf500}, 1, false, 0},
  };
  for (const auto& c : cases) {
    SCOPED_TRACE(c.name);
    const std::string kfd = root + "/kfd-" + c.name;
    for (size_t i = 0; i < c.locations.size(); ++i) {
      MakeDirs(kfd + "/" + std::to_string(i + 1));
      WriteFile(kfd + "/" + std::to_string(i + 1) + "/properties",
                "drm_render_minor " + std::to_string(128 + i) + "\ndomain 0\nlocation_id " +
                    std::to_string(c.locations[i]));
    }
    const std::string attr_dir =
        c.target == 0 ? pci + "/xcp"
                      : root + "/platform/amdgpu_xcp_" + std::to_string(c.target) + "/xcp";
    CuidPartitionParent parent;
    const amdcuid_status_t status = CuidGpu::partition_parent(attr_dir, root + "/pci", kfd, parent);
    if (!c.ok) {
      EXPECT_NE(status, AMDCUID_STATUS_SUCCESS);
      continue;
    }
    ASSERT_EQ(status, AMDCUID_STATUS_SUCCESS);
    EXPECT_EQ(parent.bdf, "0000:f5:00.0");
    EXPECT_EQ(parent.kfd_unit_id, c.unit_id);
  }
}

// KFD prints unique_id, a 64-bit serial that exceeds LLONG_MAX on about half of
// all dies, among the values the scan reads. Lines 20-41 of an MI350X CPX
// node's properties, regrafted onto fffe:63:00.0 in DPX.
TEST(cuidtst, KfdUnitIdSurvivesAnUnsignedUniqueId) {
  UnpublishedPartitions fake;
  const std::string& root = fake.root();
  for (int i : {0, 1}) {
    WriteFile(root + "/kfd/" + std::to_string(i + 1) + "/properties",
              "vendor_id 4098\ndevice_id 30112\nlocation_id " + std::to_string(25344 + i) +
                  "\ndomain 65534\ndrm_render_minor " + std::to_string(128 + i) +
                  "\nhive_id 5923454485891121163\nnum_sdma_engines 2\n"
                  "num_sdma_xgmi_engines 0\nnum_sdma_queues_per_engine 8\nnum_cp_queues 24\n"
                  "cwsr_size 22687744\nctl_stack_size 12288\nmax_engine_clk_fcompute 2200\n"
                  "local_mem_size 0\nfw_version 44\ncapability 2889327296\ncapability2 1\n"
                  "debug_prop 1494\nsdma_fw_version 14\nunique_id 10535140504318509076\n"
                  "num_xcc 4\nmax_engine_clk_ccompute 3300\n");
  }
  for (const auto& c : {std::make_pair(root + "/pci/fffe:63:00.0/xcp", 2),
                        std::make_pair(root + "/platform/amdgpu_xcp_1/xcp", 3)}) {
    SCOPED_TRACE(c.first);
    CuidPartitionParent parent;
    ASSERT_EQ(CuidGpu::partition_parent(c.first, root + "/pci", root + "/kfd", parent),
              AMDCUID_STATUS_SUCCESS);
    EXPECT_EQ(parent.bdf, "fffe:63:00.0");
    EXPECT_EQ(parent.kfd_unit_id, c.second);
  }
}
