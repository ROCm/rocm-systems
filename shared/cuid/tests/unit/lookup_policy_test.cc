// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Lookups that must be answered from what the caller supplied and what is
// published, without the key: argument validation, temporariness, provenance,
// and the temporary identity of a key-gated component without a key.

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "src/cuid_device.h"
#include "src/cuid_device_manager.h"
#include "src/cuid_util.h"
#include "src/hmac.h"
#include "test_common.h"

namespace {

// A component with no BDF, so the driver cannot answer and only its primary can.
class PrimaryOnlyDevice : public CuidDevice {
 public:
  PrimaryOnlyDevice(bool auxiliary, amdcuid_status_t primary_status = AMDCUID_STATUS_SUCCESS)
      : auxiliary_(auxiliary), primary_status_(primary_status) {}
  amdcuid_device_type_t type() const override { return AMDCUID_DEVICE_TYPE_OTHER; }
  amdcuid_status_t get_primary_cuid(amdcuid_primary_id& id) const override {
    if (primary_status_ != AMDCUID_STATUS_SUCCESS) return primary_status_;
    return CuidUtilities::generate_primary_cuid(0x1234567890ull, 0, 1, 0x74a1, 0x1002,
                                                AMDCUID_DEVICE_TYPE_OTHER, &id, auxiliary_);
  }
  amdcuid_status_t get_hardware_fingerprint(uint64_t& fingerprint) const override {
    fingerprint = 0x1234567890ull;
    return AMDCUID_STATUS_SUCCESS;
  }
  amdcuid_status_t primary_status_ = AMDCUID_STATUS_SUCCESS;

 private:
  bool auxiliary_;
};

// A PCI component whose driver attributes live in a directory the test fills.
class PublishedDevice : public PrimaryOnlyDevice {
 public:
  explicit PublishedDevice(std::string dir) : PrimaryOnlyDevice(false), dir_(std::move(dir)) {}
  amdcuid_status_t driver_attribute_path(const std::string& attribute,
                                         std::string& path) const override {
    path = dir_ + "/" + attribute;
    return AMDCUID_STATUS_SUCCESS;
  }

 private:
  std::string dir_;
};

// A CPU-like component whose derived CUID needs the node key.
class KeyGatedDevice : public PrimaryOnlyDevice {
 public:
  KeyGatedDevice() : PrimaryOnlyDevice(false) {}
  bool key_gated_identity() const override { return true; }
  amdcuid_status_t get_auxiliary_primary_cuid(amdcuid_primary_id& id) const override {
    return CuidUtilities::generate_primary_cuid(0xabcdefull, 0, 1, 0x74a1, 0x1002,
                                                AMDCUID_DEVICE_TYPE_OTHER, &id, true);
  }
};

bool SameId(const amdcuid_id_t& a, const amdcuid_id_t& b) {
  return std::memcmp(a.bytes, b.bytes, sizeof(a.bytes)) == 0;
}

// read(2) calls made by this thread so far, or -1 where procfs cannot say.
long long ThreadReadCalls() {
  std::ifstream io("/proc/thread-self/io");
  std::string field;
  long long value = 0;
  while (io >> field >> value) {
    if (field == "syscr:") return value;
  }
  return -1;
}

}  // namespace

TEST(cuidtstUnprivileged, HandleByBdfRejectsAnythingButABdf) {
  amdcuid_id_t handle{};
  for (const char* bad : {"0000:03:00.0/../../../../etc", "03:00.0", "", "0000:03:00.0 ",
                          "../../../../../dev/null", "0000:03:00.00"}) {
    EXPECT_EQ(amdcuid_get_handle_by_bdf(bad, AMDCUID_DEVICE_TYPE_GPU, &handle),
              AMDCUID_STATUS_INVALID_ARGUMENT)
        << "'" << bad << "'";
  }
  uint8_t out[16];
  const std::string high_bit = std::string(35, 'a') + "\xe9";
  EXPECT_EQ(CuidUtilities::uuid_string_to_uint8(high_bit, out), AMDCUID_STATUS_INVALID_ARGUMENT);
}

// Temporariness is a property of the primary, not of the key: it must be
// answerable where no key is at hand.
TEST(cuidtstUnprivileged, TemporaryCuidNeedsNoKey) {
  bool temporary = true;
  EXPECT_EQ(PrimaryOnlyDevice(false).is_temporary_cuid(&temporary), AMDCUID_STATUS_SUCCESS);
  EXPECT_FALSE(temporary);
  EXPECT_EQ(PrimaryOnlyDevice(true).is_temporary_cuid(&temporary), AMDCUID_STATUS_SUCCESS);
  EXPECT_TRUE(temporary);
  EXPECT_EQ(
      PrimaryOnlyDevice(false, AMDCUID_STATUS_PERMISSION_DENIED).is_temporary_cuid(&temporary),
      AMDCUID_STATUS_PERMISSION_DENIED);
}

// derived_source() names the stage of the last successful derivation, and
// nothing after a failed one.
TEST(cuidtstUnprivileged, FailedDerivationLeavesNoSource) {
  uint8_t key[key_length];
  std::memset(key, 0x5a, sizeof(key));
  cuid_hmac hmac(key);

  PrimaryOnlyDevice device(false);
  EXPECT_EQ(device.derived_source(), AMDCUID_SOURCE_UNKNOWN);
  amdcuid_derived_id derived{};
  ASSERT_EQ(device.get_derived_cuid(derived, &hmac), AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(device.derived_source(), AMDCUID_SOURCE_LIBRARY);

  device.primary_status_ = AMDCUID_STATUS_FILE_ERROR;
  EXPECT_EQ(device.get_derived_cuid(derived, &hmac), AMDCUID_STATUS_FILE_ERROR);
  EXPECT_EQ(device.derived_source(), AMDCUID_SOURCE_UNKNOWN);

  PrimaryOnlyDevice keyless(false);
  EXPECT_NE(keyless.get_derived_cuid(derived, nullptr), AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(keyless.derived_source(), AMDCUID_SOURCE_UNKNOWN);
}

// A cuid_primary without cuid_derived is not a driver publication: the device
// falls through to the later stages instead of failing on a value this library
// cannot complete, or cannot read without privilege.
TEST(cuidtstUnprivileged, PrimaryWithoutDerivedIsUnpublished) {
  const ScopedTempDir dir("cuid_published_");
  ASSERT_FALSE(dir.path().empty());
  const std::string& root = dir.path();
  const std::string primary = root + "/" + CuidUtilities::kDriverPrimaryAttribute;
  const std::string derived = root + "/" + CuidUtilities::kDriverDerivedAttribute;
  std::ofstream(primary) << "d4abaad3-9b34-8c50-9800-028dcc084200\n";

  PublishedDevice device(root);
  amdcuid_primary_id id{};
  EXPECT_EQ(device.driver_primary_cuid(id), AMDCUID_STATUS_UNSUPPORTED);

  std::ofstream(derived) << "3513616f-020c-8dbf-a802-59aa9896b410\n";
  EXPECT_EQ(device.driver_primary_cuid(id), AMDCUID_STATUS_SUCCESS);
}

// Without a key a key-gated component is temporary rather than underived, and
// is_temporary_cuid() agrees with the derivation.
TEST(cuidtstUnprivileged, KeyGatedComponentIsTemporaryWithoutAKey) {
  KeyGatedDevice device;
  uint8_t key[key_length];
  std::memset(key, 0x5a, sizeof(key));
  cuid_hmac hmac(key);
  amdcuid_derived_id with{};
  bool temporary = true;
  ASSERT_EQ(device.get_derived_cuid(with, &hmac), AMDCUID_STATUS_SUCCESS);
  ASSERT_EQ(device.is_temporary_cuid(&temporary, &hmac), AMDCUID_STATUS_SUCCESS);
  EXPECT_FALSE(temporary);

  amdcuid_derived_id without{};
  const amdcuid_status_t status = device.get_derived_cuid(without, nullptr);
  if (status == AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND)
    GTEST_SKIP() << "no machine-id, so no temporary CUID";
  ASSERT_EQ(status, AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(device.derived_source(), AMDCUID_SOURCE_LIBRARY);
  ASSERT_EQ(device.is_temporary_cuid(&temporary, nullptr), AMDCUID_STATUS_SUCCESS);
  EXPECT_TRUE(temporary);
  EXPECT_FALSE(SameId(with.UUIDv8_representation, without.UUIDv8_representation));
}

// A property query answers from the handle's device alone: it neither re-reads
// the node key nor re-derives every other device. Compared against a cold
// enumeration, and bounded outright, since a warm one can read almost nothing.
TEST(cuidtstUnprivileged, PropertyQueryDoesNotReenumerate) {
  if (ThreadReadCalls() < 0) GTEST_SKIP() << "no /proc/thread-self/io";
  const long long before = ThreadReadCalls();
  if (amdcuid_refresh() != AMDCUID_STATUS_SUCCESS)
    GTEST_SKIP() << "this host cannot identify all of its components";
  uint32_t count = 0;
  if (amdcuid_get_all_handles(nullptr, &count) != AMDCUID_STATUS_INSUFFICIENT_SIZE || count < 2)
    GTEST_SKIP() << "needs at least two components";
  std::vector<amdcuid_id_t> handles(count);
  ASSERT_EQ(amdcuid_get_all_handles(handles.data(), &count), AMDCUID_STATUS_SUCCESS);
  const long long enumeration = ThreadReadCalls() - before;

  // One device's own attributes take a handful of reads; enumeration, hundreds.
  constexpr long long kQueryReadCeiling = 16;
  for (const auto& handle : handles) {
    amdcuid_device_type_t type = AMDCUID_DEVICE_TYPE_NONE;
    uint32_t length = sizeof(type);
    const long long start = ThreadReadCalls();
    ASSERT_EQ(amdcuid_query_device_property(handle, AMDCUID_QUERY_DEVICE_TYPE, &type, &length),
              AMDCUID_STATUS_SUCCESS);
    const long long reads = ThreadReadCalls() - start;
    EXPECT_LE(reads, kQueryReadCeiling) << amdcuid_id_to_string(handle);
    EXPECT_LT(reads, enumeration) << amdcuid_id_to_string(handle);
  }
}

// An ordinary user has no node key, so every CPU, NIC, NPU and Platform it can
// list is on its temporary CUID.
TEST(cuidtstUnprivileged, NonGpuComponentsAreTemporaryWithoutRoot) {
  if (geteuid() == 0) GTEST_SKIP() << "root may hold the node key";
  uint64_t probe = 0;
  if (CuidUtilities::make_fallback_fingerprint(CuidUtilities::AuxiliaryInput{}, probe) !=
      AMDCUID_STATUS_SUCCESS)
    GTEST_SKIP() << "no machine-id, so no temporary CUID";
  uint32_t count = 0;
  if (amdcuid_get_all_handles(nullptr, &count) != AMDCUID_STATUS_INSUFFICIENT_SIZE)
    GTEST_SKIP() << "no components";
  std::vector<amdcuid_id_t> handles(count);
  ASSERT_EQ(amdcuid_get_all_handles(handles.data(), &count), AMDCUID_STATUS_SUCCESS);
  size_t checked = 0;
  for (const auto& handle : handles) {
    amdcuid_device_type_t type = AMDCUID_DEVICE_TYPE_NONE;
    uint32_t length = sizeof(type);
    ASSERT_EQ(amdcuid_query_device_property(handle, AMDCUID_QUERY_DEVICE_TYPE, &type, &length),
              AMDCUID_STATUS_SUCCESS);
    if (type == AMDCUID_DEVICE_TYPE_GPU) continue;
    bool temporary = false;
    length = sizeof(temporary);
    ASSERT_EQ(
        amdcuid_query_device_property(handle, AMDCUID_QUERY_TEMPORARY_CUID, &temporary, &length),
        AMDCUID_STATUS_SUCCESS);
    EXPECT_TRUE(temporary) << "type " << type << ": " << amdcuid_id_to_string(handle);
    ++checked;
  }
  if (checked == 0) GTEST_SKIP() << "no component other than a GPU";
}
