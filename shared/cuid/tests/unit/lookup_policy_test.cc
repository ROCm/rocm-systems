// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Lookups that must be answered from what the caller supplied and what is
// published, without the key: argument validation, temporariness, provenance,
// and the temporary identity of a key-gated component without a key.

#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "src/cuid_device.h"
#include "src/cuid_device_manager.h"
#include "src/cuid_nic.h"
#include "src/cuid_util.h"
#include "src/hmac.h"
#include "test_common.h"

namespace {

// A component with no BDF, so the driver cannot answer and only its primary can.
class PrimaryOnlyDevice : public CuidDevice {
 public:
  PrimaryOnlyDevice(bool auxiliary, amdcuid_status_t primary_status = AMDCUID_STATUS_SUCCESS)
      : primary_status_(primary_status), auxiliary_(auxiliary) {}
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

// A cuid_primary without cuid_unit_id is not a driver publication: the device
// falls through to the later stages instead of failing on a value this library
// cannot complete, or cannot read without privilege.
TEST(cuidtstUnprivileged, PrimaryWithoutUnitIdIsUnpublished) {
  const ScopedTempDir dir("cuid_published_");
  ASSERT_FALSE(dir.path().empty());
  const std::string& root = dir.path();
  const std::string primary = root + "/" + CuidUtilities::kDriverPrimaryAttribute;
  const std::string unit_id = root + "/" + CuidUtilities::kDriverUnitIdAttribute;
  std::ofstream(primary) << "d4abaad3-9b34-8c50-9800-028dcc084200\n";

  PublishedDevice device(root);
  amdcuid_primary_id id{};
  EXPECT_EQ(device.driver_primary_cuid(id), AMDCUID_STATUS_UNSUPPORTED);

  std::ofstream(unit_id) << "0\n";
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

// A NIC's UnitID is its PCI function number, so two functions of one card that
// report one serial number do not share an identity. Without a BDF it is 0.
TEST(cuidtstUnprivileged, NicUnitIdIsItsPciFunction) {
  for (const auto& c : {std::make_pair("0000:31:00.0", 0), std::make_pair("0000:31:00.1", 1),
                        std::make_pair("0000:c1:00.7", 7), std::make_pair("", 0)}) {
    amdcuid_nic_info info{};
    info.bdf = c.first;
    uint16_t unit_id = 0xFFFF;
    EXPECT_EQ(CuidNic(info).get_unit_id(unit_id), AMDCUID_STATUS_SUCCESS) << c.first;
    EXPECT_EQ(unit_id, c.second) << c.first;
  }
}

// Interfaces that share a PCI function, such as switchdev representors, are
// one component: one per BDF, or the index sees one CUID twice and refuses the
// whole inventory. The interface kept is the first by name.
TEST(cuidtstUnprivileged, NicDiscoveryKeepsOneInterfacePerFunction) {
  const ScopedTempDir dir("cuid_net_");
  ASSERT_FALSE(dir.path().empty());
  const std::string& root = dir.path();
  ASSERT_EQ(mkdir((root + "/net").c_str(), 0700), 0);
  ASSERT_EQ(mkdir((root + "/pci").c_str(), 0700), 0);
  for (const char* bdf : {"fffe:fe:1f.0", "fffe:fe:1f.1"}) {
    const std::string fn = root + "/pci/" + bdf;
    ASSERT_EQ(mkdir(fn.c_str(), 0700), 0);
    std::ofstream(fn + "/vendor") << "0x1022\n";
  }
  for (const auto& iface :
       {std::make_pair("rep0", "fffe:fe:1f.0"), std::make_pair("eth0", "fffe:fe:1f.0"),
        std::make_pair("eth1", "fffe:fe:1f.1")}) {
    const std::string net = root + "/net/" + iface.first;
    ASSERT_EQ(mkdir(net.c_str(), 0700), 0);
    ASSERT_EQ(symlink((root + "/pci/" + iface.second).c_str(), (net + "/device").c_str()), 0);
  }

  std::vector<DevicePtr> nics;
  ASSERT_EQ(CuidNic::discover(root + "/net", nics), AMDCUID_STATUS_SUCCESS);
  ASSERT_EQ(nics.size(), 2u);
  std::string bdf, path;
  ASSERT_EQ(nics[0]->get_bdf(bdf), AMDCUID_STATUS_SUCCESS);
  ASSERT_EQ(nics[0]->get_device_path(path), AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(bdf, "fffe:fe:1f.0");
  EXPECT_EQ(path, root + "/net/eth0");
  ASSERT_EQ(nics[1]->get_bdf(bdf), AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(bdf, "fffe:fe:1f.1");
}

// An interface is excluded for having no PCI function, not for its name: lo is
// skipped, a physical lom1 is not.
TEST(cuidtstUnprivileged, NicDiscoveryKeepsInterfacesNamedLikeLoopback) {
  const ScopedTempDir dir("cuid_net_");
  ASSERT_FALSE(dir.path().empty());
  const std::string& root = dir.path();
  const std::string fn = root + "/pci/fffe:fe:1f.0";
  ASSERT_EQ(mkdir((root + "/net").c_str(), 0700), 0);
  ASSERT_EQ(mkdir((root + "/pci").c_str(), 0700), 0);
  ASSERT_EQ(mkdir(fn.c_str(), 0700), 0);
  std::ofstream(fn + "/vendor") << "0x1022\n";
  ASSERT_EQ(mkdir((root + "/net/lo").c_str(), 0700), 0);
  ASSERT_EQ(mkdir((root + "/net/lom1").c_str(), 0700), 0);
  ASSERT_EQ(symlink(fn.c_str(), (root + "/net/lom1/device").c_str()), 0);

  std::vector<DevicePtr> nics;
  ASSERT_EQ(CuidNic::discover(root + "/net", nics), AMDCUID_STATUS_SUCCESS);
  ASSERT_EQ(nics.size(), 1u);
  std::string path;
  ASSERT_EQ(nics[0]->get_device_path(path), AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(path, root + "/net/lom1");
}

// A lookup by an interface discovery did not keep answers with the function's
// handle, rather than adding a second component with the same CUID.
TEST(cuidtstUnprivileged, NicLookupBySiblingInterfaceFindsTheFunction) {
  const ScopedTempDir dir("cuid_net_");
  ASSERT_FALSE(dir.path().empty());
  const std::string& root = dir.path();
  const std::string fn = root + "/pci/fffe:fe:1f.0";
  ASSERT_EQ(mkdir((root + "/net").c_str(), 0700), 0);
  ASSERT_EQ(mkdir((root + "/pci").c_str(), 0700), 0);
  ASSERT_EQ(mkdir(fn.c_str(), 0700), 0);
  std::ofstream(fn + "/vendor") << "0x1022\n";
  for (const char* iface : {"cuidtst0", "cuidtst1"}) {
    const std::string net = root + "/net/" + iface;
    ASSERT_EQ(mkdir(net.c_str(), 0700), 0);
    ASSERT_EQ(symlink(fn.c_str(), (net + "/device").c_str()), 0);
  }
  std::vector<DevicePtr> nics;
  ASSERT_EQ(CuidNic::discover(root + "/net", nics), AMDCUID_STATUS_SUCCESS);
  ASSERT_EQ(nics.size(), 1u);
  amdcuid_derived_id derived{};
  const amdcuid_status_t status = nics[0]->get_derived_cuid(derived, nullptr);
  if (status == AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND)
    GTEST_SKIP() << "no machine-id, so no temporary CUID";
  ASSERT_EQ(status, AMDCUID_STATUS_SUCCESS);

  CuidDeviceManager& mgr = CuidDeviceManager::instance();
  mgr.shutdown();
  ASSERT_EQ(mgr.index_handle(nics[0], derived.UUIDv8_representation), AMDCUID_STATUS_SUCCESS);
  amdcuid_id_t kept{}, sibling{};
  const amdcuid_status_t kept_status = amdcuid_get_handle_by_dev_path(
      (root + "/net/cuidtst0").c_str(), AMDCUID_DEVICE_TYPE_NIC, &kept);
  const amdcuid_status_t sibling_status = amdcuid_get_handle_by_dev_path(
      (root + "/net/cuidtst1").c_str(), AMDCUID_DEVICE_TYPE_NIC, &sibling);
  const size_t known = mgr.devices().size();
  mgr.shutdown();

  ASSERT_EQ(kept_status, AMDCUID_STATUS_SUCCESS);
  EXPECT_TRUE(SameId(kept, derived.UUIDv8_representation));
  ASSERT_EQ(sibling_status, AMDCUID_STATUS_SUCCESS);
  EXPECT_TRUE(SameId(sibling, kept));
  EXPECT_EQ(known, 1u);
}

// A property query answers from the handle's device alone and does not
// re-derive every other device. Compared against a cold enumeration, and
// bounded outright, since a warm one can read almost nothing. Root is left out:
// its query also reads cuid_seed, once per amdgpu device while no key is set.
TEST(cuidtstUnprivileged, PropertyQueryDoesNotReenumerate) {
  if (geteuid() == 0) GTEST_SKIP() << "root reads cuid_seed on every query";
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

namespace {

// A PCI function whose serial the test changes, as a re-read can.
class SerialDevice : public CuidDevice {
 public:
  SerialDevice(std::string bdf, uint64_t serial) : serial(serial), bdf_(std::move(bdf)) {}
  amdcuid_device_type_t type() const override { return AMDCUID_DEVICE_TYPE_OTHER; }
  amdcuid_status_t get_primary_cuid(amdcuid_primary_id& id) const override {
    return CuidUtilities::generate_primary_cuid(serial, 0, 1, 0x74a1, 0x1002,
                                                AMDCUID_DEVICE_TYPE_OTHER, &id, true);
  }
  amdcuid_status_t get_hardware_fingerprint(uint64_t& fingerprint) const override {
    fingerprint = serial;
    return AMDCUID_STATUS_SUCCESS;
  }
  amdcuid_status_t get_bdf(std::string& bdf) const override {
    bdf = bdf_;
    return AMDCUID_STATUS_SUCCESS;
  }
  amdcuid_status_t get_device_path(std::string& path) const override {
    path = "/sys/bus/pci/devices/" + bdf_;
    return AMDCUID_STATUS_SUCCESS;
  }
  uint64_t serial;

 private:
  std::string bdf_;
};

}  // namespace

// Two components whose CUIDs coincide are identified by neither. Both are left
// out and every other component is still listed; a lookup of either is refused.
TEST(cuidtstUnprivileged, CollidingComponentsAreLeftOutOfTheInventory) {
  const auto a = std::make_shared<SerialDevice>("fffe:fd:00.0", 0x1111);
  const auto b = std::make_shared<SerialDevice>("fffe:fd:01.0", 0x2222);
  const auto c = std::make_shared<SerialDevice>("fffe:fd:02.0", 0x3333);
  amdcuid_derived_id da{}, db{}, dc{};
  const amdcuid_status_t status = a->get_derived_cuid(da, nullptr);
  if (status == AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND)
    GTEST_SKIP() << "no machine-id, so no temporary CUID";
  ASSERT_EQ(status, AMDCUID_STATUS_SUCCESS);
  ASSERT_EQ(b->get_derived_cuid(db, nullptr), AMDCUID_STATUS_SUCCESS);
  ASSERT_EQ(c->get_derived_cuid(dc, nullptr), AMDCUID_STATUS_SUCCESS);

  CuidDeviceManager& mgr = CuidDeviceManager::instance();
  mgr.shutdown();
  ASSERT_EQ(mgr.index_handle(a, da.UUIDv8_representation), AMDCUID_STATUS_SUCCESS);
  ASSERT_EQ(mgr.index_handle(b, db.UUIDv8_representation), AMDCUID_STATUS_SUCCESS);
  ASSERT_EQ(mgr.index_handle(c, dc.UUIDv8_representation), AMDCUID_STATUS_SUCCESS);
  b->serial = a->serial;
  const amdcuid_status_t rebuilt = mgr.build_cuid_index();
  const auto handles = mgr.get_all_handles();
  const bool resolves = mgr.lookup_by_handle(da.UUIDv8_representation) != nullptr;
  const amdcuid_status_t lookup = mgr.index_handle(a, da.UUIDv8_representation);
  mgr.shutdown();

  EXPECT_EQ(rebuilt, AMDCUID_STATUS_SUCCESS);
  ASSERT_EQ(handles.size(), 1u);
  EXPECT_TRUE(SameId(handles[0], dc.UUIDv8_representation));
  EXPECT_FALSE(resolves);
  EXPECT_EQ(lookup, AMDCUID_STATUS_INVALID_FORMAT);
}

TEST(cuidtstUnprivileged, AnInventoryOfOnlyCollidingComponentsIsEmpty) {
  const auto a = std::make_shared<SerialDevice>("fffe:fd:00.0", 0x1111);
  const auto b = std::make_shared<SerialDevice>("fffe:fd:01.0", 0x2222);
  amdcuid_derived_id da{}, db{};
  const amdcuid_status_t status = a->get_derived_cuid(da, nullptr);
  if (status == AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND)
    GTEST_SKIP() << "no machine-id, so no temporary CUID";
  ASSERT_EQ(status, AMDCUID_STATUS_SUCCESS);
  ASSERT_EQ(b->get_derived_cuid(db, nullptr), AMDCUID_STATUS_SUCCESS);

  CuidDeviceManager& mgr = CuidDeviceManager::instance();
  mgr.shutdown();
  ASSERT_EQ(mgr.index_handle(a, da.UUIDv8_representation), AMDCUID_STATUS_SUCCESS);
  ASSERT_EQ(mgr.index_handle(b, db.UUIDv8_representation), AMDCUID_STATUS_SUCCESS);
  b->serial = a->serial;
  const amdcuid_status_t rebuilt = mgr.build_cuid_index();
  const auto handles = mgr.get_all_handles();
  const amdcuid_status_t lookup = mgr.index_handle(a, da.UUIDv8_representation);
  mgr.shutdown();

  EXPECT_EQ(rebuilt, AMDCUID_STATUS_SUCCESS);
  EXPECT_TRUE(handles.empty());
  EXPECT_EQ(lookup, AMDCUID_STATUS_INVALID_FORMAT);
}
