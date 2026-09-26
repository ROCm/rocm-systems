// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// A spatial partition has no serial or config space of its own and its
// parent's BDF names the whole GPU, so it is adopted only from the values the
// kernel publishes under its own node. These tests fabricate that node.

#include <gtest/gtest.h>
#include <sys/stat.h>

#include <cstring>
#include <fstream>
#include <string>

#include "src/cuid_gpu.h"
#include "src/cuid_util.h"
#include "test_common.h"

namespace {

// P-1 and D-1 from the cross-layer conformance vectors, so the values here are
// the ones both layers agree on rather than arbitrary strings.
constexpr char kPrimary[] = "d4abaad3-9b34-8c50-9800-028dcc084200";
constexpr char kDerived[] = "10133e37-3995-80bc-a402-7074c5c1c44c";

class FakeSysfs {
 public:
  FakeSysfs() : dir_("cuid_partition_test_"), root_(dir_.path()) {}

  const std::string& device_path() const { return root_; }

  void PublishPartition(const char* primary, const char* derived) {
    mkdir((root_ + "/xcp").c_str(), 0755);
    if (primary) Write(root_ + "/xcp/cuid_primary", primary);
    if (derived) Write(root_ + "/xcp/cuid_derived", derived);
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
  fake.PublishPartition(kPrimary, kDerived);

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

TEST(cuidtst, PartitionAnswersWithTheKernelsValues) {
  FakeSysfs fake;
  ASSERT_FALSE(fake.device_path().empty());
  fake.PublishPartition(kPrimary, kDerived);

  amdcuid_gpu_info info = {};
  ASSERT_EQ(CuidGpu::discover_partition(&info, fake.device_path()), AMDCUID_STATUS_SUCCESS);
  CuidGpu partition(info);

  std::string path;
  ASSERT_EQ(partition.driver_attribute_path(CuidUtilities::kDriverPrimaryAttribute, path),
            AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(path, info.partition_attr_dir + "/cuid_primary");

  amdcuid_id_t want_primary = {};
  amdcuid_id_t want_derived = {};
  ASSERT_EQ(CuidUtilities::uuid_string_to_uint8(kPrimary, want_primary.bytes),
            AMDCUID_STATUS_SUCCESS);
  ASSERT_EQ(CuidUtilities::uuid_string_to_uint8(kDerived, want_derived.bytes),
            AMDCUID_STATUS_SUCCESS);

  amdcuid_primary_id primary = {};
  ASSERT_EQ(partition.get_primary_cuid(primary), AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(std::memcmp(primary.UUIDv8_representation.bytes, want_primary.bytes, 16), 0);

  uint64_t fingerprint = 0;
  if (geteuid() == 0) {
    ASSERT_EQ(partition.get_hardware_fingerprint(fingerprint), AMDCUID_STATUS_SUCCESS);
    EXPECT_EQ(fingerprint, 0x06c5349bd3aaabd4u);  // S-DSN / P-1 serial, little-endian.
  } else {
    EXPECT_EQ(partition.get_hardware_fingerprint(fingerprint), AMDCUID_STATUS_PERMISSION_DENIED);
  }

  amdcuid_derived_id derived = {};
  ASSERT_EQ(partition.get_derived_cuid(derived, nullptr), AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(std::memcmp(derived.UUIDv8_representation.bytes, want_derived.bytes, 16), 0);
}

TEST(cuidtst, UnnamedVfCannotAdoptPartitionAttributes) {
  FakeSysfs fake;
  ASSERT_FALSE(fake.device_path().empty());
  fake.PublishPartition(kPrimary, kDerived);

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
  EXPECT_EQ(vf.driver_primary_cuid(primary), AMDCUID_STATUS_UNSUPPORTED);
  EXPECT_EQ(vf.get_primary_cuid(primary), AMDCUID_STATUS_UNSUPPORTED);
  amdcuid_derived_id derived = {};
  EXPECT_EQ(vf.get_derived_cuid(derived), AMDCUID_STATUS_UNSUPPORTED);
}

TEST(cuidtst, PartitionWithoutAPublishedPrimaryDoesNotInventOne) {
  FakeSysfs fake;
  ASSERT_FALSE(fake.device_path().empty());
  // cuid_primary is 0400. An unprivileged caller must not fall through to the
  // auxiliary path, which would give every partition the parent's routing id.
  fake.PublishPartition(nullptr, kDerived);

  amdcuid_gpu_info info = {};
  ASSERT_EQ(CuidGpu::discover_partition(&info, fake.device_path()), AMDCUID_STATUS_SUCCESS);

  // discover_partition leaves the BDF empty; set it so the auxiliary path could
  // succeed and the test depends on the explicit refusal.
  info.bdf = "0000:63:00.0";
  CuidGpu partition(info);

  amdcuid_primary_id primary = {};
  EXPECT_EQ(partition.get_primary_cuid(primary), AMDCUID_STATUS_UNSUPPORTED)
      << "a partition with no driver-published primary invented one";
}
