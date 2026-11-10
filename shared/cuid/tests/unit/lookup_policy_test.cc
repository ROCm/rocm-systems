// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Lookups that must be answered from what the caller supplied and what is
// published, without the key: argument validation, temporariness, provenance,
// and the temporary identity of a key-gated component without a key.

#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cctype>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "src/cuid_device.h"
#include "src/cuid_device_manager.h"
#include "src/cuid_nic.h"
#include "src/cuid_npu.h"
#include "src/cuid_util.h"
#include "src/hmac.h"
#include "test_common.h"

namespace {

// A component with no BDF, so the driver cannot answer and only its primary,
// or for a temporary one its temporary input, can.
class PrimaryOnlyDevice : public CuidDevice {
 public:
  PrimaryOnlyDevice(bool temporary, amdcuid_status_t primary_status = AMDCUID_STATUS_SUCCESS)
      : primary_status_(primary_status), temporary_(temporary) {}
  amdcuid_device_type_t type() const override { return AMDCUID_DEVICE_TYPE_OTHER; }
  amdcuid_status_t get_primary_cuid(amdcuid_primary_id& id) const override {
    if (primary_status_ != AMDCUID_STATUS_SUCCESS) return primary_status_;
    if (temporary_) return AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND;
    return CuidUtilities::generate_primary_cuid(0x1234567890ull, 0, 1, 0x74a1, 0x1002,
                                                AMDCUID_DEVICE_TYPE_OTHER, &id);
  }
  amdcuid_status_t get_temporary_input(CuidUtilities::AuxiliaryInput& input) const override {
    input = {};
    input.format = CuidUtilities::kAuxFormatPcie;
    input.device_id = 0x74a1;
    input.vendor_id = 0x1002;
    input.component_type = AMDCUID_DEVICE_TYPE_OTHER;
    return AMDCUID_STATUS_SUCCESS;
  }
  amdcuid_status_t get_hardware_fingerprint(uint64_t& fingerprint) const override {
    fingerprint = 0x1234567890ull;
    return AMDCUID_STATUS_SUCCESS;
  }
  amdcuid_status_t primary_status_ = AMDCUID_STATUS_SUCCESS;

 private:
  bool temporary_;
};

// A CPU-like component whose derived CUID needs the node key.
class KeyGatedDevice : public PrimaryOnlyDevice {
 public:
  KeyGatedDevice() : PrimaryOnlyDevice(false) {}
  bool key_gated_identity() const override { return true; }
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
  for (const char* bad : {"0000:03:00.0/../../../../etc", "03:00.0", "", "0000:03:00.0 "})
    EXPECT_EQ(CuidUtilities::canonical_bdf(bad), "") << "'" << bad << "'";
  uint8_t out[16];
  const std::string high_bit = std::string(35, 'a') + "\xe9";
  EXPECT_EQ(CuidUtilities::uuid_string_to_uint8(high_bit, out), AMDCUID_STATUS_INVALID_ARGUMENT);
}

TEST(cuidtstUnprivileged, CanonicalBdfIsTheSysfsSpelling) {
  EXPECT_EQ(CuidUtilities::canonical_bdf("0000:a6:00.0"), "0000:a6:00.0");
  EXPECT_EQ(CuidUtilities::canonical_bdf("0000:A6:00.0"), "0000:a6:00.0");
  EXPECT_EQ(CuidUtilities::canonical_bdf("00000000:a6:00.0"), "0000:a6:00.0");
  EXPECT_EQ(CuidUtilities::canonical_bdf("0000ABCD:Ff:1F.7"), "abcd:ff:1f.7");
  EXPECT_EQ(CuidUtilities::canonical_bdf("00010000:04:00.0"), "10000:04:00.0");
  EXPECT_EQ(CuidUtilities::canonical_bdf("FFFFFFFF:04:00.0"), "ffffffff:04:00.0");
}

// Every spelling the BDF validator accepts finds the function its canonical
// spelling finds.
TEST(cuidtstUnprivileged, HandleByBdfAcceptsEverySpellingOfOneFunction) {
  uint32_t count = 0;
  if (amdcuid_get_all_handles(nullptr, &count) != AMDCUID_STATUS_INSUFFICIENT_SIZE)
    GTEST_SKIP() << "no components";
  std::vector<amdcuid_id_t> handles(count);
  ASSERT_EQ(amdcuid_get_all_handles(handles.data(), &count), AMDCUID_STATUS_SUCCESS);
  size_t checked = 0;
  for (const auto& want : handles) {
    char bdf[32] = {};
    uint32_t length = sizeof(bdf);
    if (amdcuid_query_device_property(want, AMDCUID_QUERY_BDF, bdf, &length) !=
        AMDCUID_STATUS_SUCCESS)
      continue;
    amdcuid_device_type_t type = AMDCUID_DEVICE_TYPE_NONE;
    length = sizeof(type);
    ASSERT_EQ(amdcuid_query_device_property(want, AMDCUID_QUERY_DEVICE_TYPE, &type, &length),
              AMDCUID_STATUS_SUCCESS);
    std::string upper = bdf;
    for (auto& c : upper) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    std::string padded = bdf;
    if (padded.find(':') == 4) padded = "0000" + padded;
    for (const std::string& spelling : {std::string(bdf), upper, padded}) {
      amdcuid_id_t got{};
      ASSERT_EQ(amdcuid_get_handle_by_bdf(spelling.c_str(), type, &got), AMDCUID_STATUS_SUCCESS)
          << spelling;
      EXPECT_TRUE(SameId(got, want)) << spelling;
    }
    ++checked;
  }
  if (checked == 0) GTEST_SKIP() << "no PCI component";
}

// A Component Type that is not assigned is a bad argument; an assigned one this
// library discovers nothing of is unsupported, before anything is looked up.
TEST(cuidtstUnprivileged, LookupsAnswerTheTypeBeforeTheAddress) {
  const auto reserved = static_cast<amdcuid_device_type_t>(0xB);
  amdcuid_id_t handle{};
  for (const amdcuid_device_type_t type : {reserved, AMDCUID_DEVICE_TYPE_NONE}) {
    EXPECT_EQ(amdcuid_get_handle_by_dev_path("/sys/devices/system/cpu/cpu0", type, &handle),
              AMDCUID_STATUS_INVALID_ARGUMENT)
        << type;
    EXPECT_EQ(amdcuid_get_handle_by_bdf("0000:00:00.0", type, &handle),
              AMDCUID_STATUS_INVALID_ARGUMENT)
        << type;
    EXPECT_EQ(amdcuid_get_handle_by_fd(0, type, &handle), AMDCUID_STATUS_INVALID_ARGUMENT) << type;
  }
  for (const amdcuid_device_type_t type :
       {AMDCUID_DEVICE_TYPE_STORAGE, AMDCUID_DEVICE_TYPE_OTHER}) {
    EXPECT_EQ(amdcuid_get_handle_by_dev_path("/sys/devices/system/cpu/cpu0", type, &handle),
              AMDCUID_STATUS_UNSUPPORTED)
        << type;
    EXPECT_EQ(amdcuid_get_handle_by_bdf("0000:00:00.0", type, &handle), AMDCUID_STATUS_UNSUPPORTED)
        << type;
    EXPECT_EQ(amdcuid_get_handle_by_fd(0, type, &handle), AMDCUID_STATUS_UNSUPPORTED) << type;
  }
  EXPECT_EQ(amdcuid_generate_hash_key(nullptr), AMDCUID_STATUS_INVALID_ARGUMENT);
}

// A function discovery lists as one type, looked up as another PCI type, is
// the wrong type rather than unsupported or absent.
TEST(cuidtstUnprivileged, HandleByBdfOfAnotherTypeIsTheWrongType) {
  uint32_t count = 0;
  if (amdcuid_get_all_handles(nullptr, &count) != AMDCUID_STATUS_INSUFFICIENT_SIZE)
    GTEST_SKIP() << "no components";
  std::vector<amdcuid_id_t> handles(count);
  ASSERT_EQ(amdcuid_get_all_handles(handles.data(), &count), AMDCUID_STATUS_SUCCESS);
  size_t checked = 0;
  for (const auto& listed : handles) {
    char bdf[32] = {};
    uint32_t length = sizeof(bdf);
    if (amdcuid_query_device_property(listed, AMDCUID_QUERY_BDF, bdf, &length) !=
        AMDCUID_STATUS_SUCCESS)
      continue;
    amdcuid_device_type_t type = AMDCUID_DEVICE_TYPE_NONE;
    length = sizeof(type);
    ASSERT_EQ(amdcuid_query_device_property(listed, AMDCUID_QUERY_DEVICE_TYPE, &type, &length),
              AMDCUID_STATUS_SUCCESS);
    for (const amdcuid_device_type_t other :
         {AMDCUID_DEVICE_TYPE_GPU, AMDCUID_DEVICE_TYPE_NIC, AMDCUID_DEVICE_TYPE_NPU}) {
      if (other == type) continue;
      amdcuid_id_t got{};
      EXPECT_EQ(amdcuid_get_handle_by_bdf(bdf, other, &got), AMDCUID_STATUS_WRONG_DEVICE_TYPE)
          << bdf << " as type " << other;
    }
    ++checked;
  }
  if (checked == 0) GTEST_SKIP() << "no PCI component";
}

// Temporariness is a property of the primary, not of the key: it must be
// answerable where no key is at hand.
TEST(cuidtstUnprivileged, TemporaryCuidNeedsNoKey) {
  bool temporary = true;
  EXPECT_EQ(PrimaryOnlyDevice(false).is_temporary_cuid(&temporary), AMDCUID_STATUS_SUCCESS);
  EXPECT_FALSE(temporary);
  uint8_t machine_id[16];
  if (CuidUtilities::host_machine_id(machine_id) == AMDCUID_STATUS_SUCCESS) {
    EXPECT_EQ(PrimaryOnlyDevice(true).is_temporary_cuid(&temporary), AMDCUID_STATUS_SUCCESS);
    EXPECT_TRUE(temporary);
  }
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

namespace {

void MakeFunction(const std::string& root, const std::string& bdf, const char* pci_class,
                  std::initializer_list<const char*> interfaces = {}) {
  const std::string fn = root + "/pci/" + bdf;
  ASSERT_EQ(mkdir(fn.c_str(), 0700), 0);
  std::ofstream(fn + "/vendor") << "0x1022\n";
  std::ofstream(fn + "/device") << "0x1450\n";
  std::ofstream(fn + "/revision") << "0x00\n";
  std::ofstream(fn + "/class") << pci_class << "\n";
  if (interfaces.size() == 0) return;
  ASSERT_EQ(mkdir((fn + "/net").c_str(), 0700), 0);
  for (const char* iface : interfaces) {
    ASSERT_EQ(mkdir((fn + "/net/" + iface).c_str(), 0700), 0);
    const std::string net = root + "/net/" + iface;
    ASSERT_EQ(mkdir(net.c_str(), 0700), 0);
    ASSERT_EQ(symlink(fn.c_str(), (net + "/device").c_str()), 0);
  }
}

}  // namespace

// Interfaces that share a PCI function, such as switchdev representors, are
// one component. Its device path is its first interface by name.
TEST(cuidtstUnprivileged, NicDiscoveryKeepsOneInterfacePerFunction) {
  const ScopedTempDir dir("cuid_net_");
  ASSERT_FALSE(dir.path().empty());
  const std::string& root = dir.path();
  ASSERT_EQ(mkdir((root + "/net").c_str(), 0700), 0);
  ASSERT_EQ(mkdir((root + "/pci").c_str(), 0700), 0);
  MakeFunction(root, "fffe:fe:1f.0", "0x020000", {"rep0", "eth0"});
  MakeFunction(root, "fffe:fe:1f.1", "0x020000", {"eth1"});

  std::vector<DevicePtr> nics;
  ASSERT_EQ(CuidNic::discover(root + "/pci", root + "/net", nics), AMDCUID_STATUS_SUCCESS);
  ASSERT_EQ(nics.size(), 2u);
  std::string bdf, path;
  ASSERT_EQ(nics[0]->get_bdf(bdf), AMDCUID_STATUS_SUCCESS);
  ASSERT_EQ(nics[0]->get_device_path(path), AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(bdf, "fffe:fe:1f.0");
  EXPECT_EQ(path, root + "/net/eth0");
  ASSERT_EQ(nics[1]->get_bdf(bdf), AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(bdf, "fffe:fe:1f.1");
}

// A NIC is found by its PCI class, so one with no interface in the caller's
// network namespace (another container, vfio, DPDK) is still a component, at
// its PCI path.
TEST(cuidtstUnprivileged, NicDiscoveryNeedsNoInterface) {
  const ScopedTempDir dir("cuid_net_");
  ASSERT_FALSE(dir.path().empty());
  const std::string& root = dir.path();
  ASSERT_EQ(mkdir((root + "/net").c_str(), 0700), 0);
  ASSERT_EQ(mkdir((root + "/pci").c_str(), 0700), 0);
  MakeFunction(root, "fffe:fe:1f.0", "0x020000");

  std::vector<DevicePtr> nics;
  ASSERT_EQ(CuidNic::discover(root + "/pci", root + "/net", nics), AMDCUID_STATUS_SUCCESS);
  ASSERT_EQ(nics.size(), 1u);
  std::string bdf, path;
  ASSERT_EQ(nics[0]->get_bdf(bdf), AMDCUID_STATUS_SUCCESS);
  ASSERT_EQ(nics[0]->get_device_path(path), AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(bdf, "fffe:fe:1f.0");
  EXPECT_EQ(path, root + "/pci/fffe:fe:1f.0");
}

// Only a network or 802.11 controller is a NIC; a Bluetooth function is not.
TEST(cuidtstUnprivileged, NicDiscoveryIgnoresOtherPciClasses) {
  const ScopedTempDir dir("cuid_net_");
  ASSERT_FALSE(dir.path().empty());
  const std::string& root = dir.path();
  ASSERT_EQ(mkdir((root + "/net").c_str(), 0700), 0);
  ASSERT_EQ(mkdir((root + "/pci").c_str(), 0700), 0);
  MakeFunction(root, "fffe:fe:1f.0", "0x020000");
  MakeFunction(root, "fffe:fe:1f.1", "0x030000");
  MakeFunction(root, "fffe:fe:1f.2", "0x0d8000");
  MakeFunction(root, "fffe:fe:1f.3", "0x0d1100");

  std::vector<DevicePtr> nics;
  ASSERT_EQ(CuidNic::discover(root + "/pci", root + "/net", nics), AMDCUID_STATUS_SUCCESS);
  std::vector<std::string> bdfs;
  for (const auto& nic : nics) {
    std::string bdf;
    ASSERT_EQ(nic->get_bdf(bdf), AMDCUID_STATUS_SUCCESS);
    bdfs.push_back(bdf);
  }
  EXPECT_EQ(bdfs, (std::vector<std::string>{"fffe:fe:1f.0", "fffe:fe:1f.2"}));
}

// A lookup by an interface discovery did not keep answers with the function's
// handle, rather than adding a second component with the same CUID.
TEST(cuidtstUnprivileged, NicLookupBySiblingInterfaceFindsTheFunction) {
  const ScopedTempDir dir("cuid_net_");
  ASSERT_FALSE(dir.path().empty());
  const std::string& root = dir.path();
  ASSERT_EQ(mkdir((root + "/net").c_str(), 0700), 0);
  ASSERT_EQ(mkdir((root + "/pci").c_str(), 0700), 0);
  MakeFunction(root, "fffe:fe:1f.0", "0x020000", {"cuidtst0", "cuidtst1"});
  std::vector<DevicePtr> nics;
  ASSERT_EQ(CuidNic::discover(root + "/pci", root + "/net", nics), AMDCUID_STATUS_SUCCESS);
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

// A directory that looks like a PCI network function but is not one discovery
// lists has no CUID, and looking it up adds nothing to the inventory.
TEST(cuidtstUnprivileged, LookupOfAPathDiscoveryDoesNotListAddsNothing) {
  uint8_t machine_id[16];
  if (CuidUtilities::host_machine_id(machine_id) != AMDCUID_STATUS_SUCCESS)
    GTEST_SKIP() << "no machine-id, so no temporary CUID";
  const ScopedTempDir dir("cuid_phantom_");
  ASSERT_FALSE(dir.path().empty());
  const std::string& root = dir.path();
  ASSERT_EQ(mkdir((root + "/pci").c_str(), 0700), 0);
  MakeFunction(root, "0000:ee:1f.7", "0x020000");
  const std::string phantom = root + "/pci/0000:ee:1f.7";

  uint32_t before = 0;
  const amdcuid_status_t listed = amdcuid_get_all_handles(nullptr, &before);
  ASSERT_TRUE(listed == AMDCUID_STATUS_INSUFFICIENT_SIZE || listed == AMDCUID_STATUS_UNSUPPORTED)
      << listed;
  amdcuid_id_t handle{};
  EXPECT_EQ(amdcuid_get_handle_by_dev_path(phantom.c_str(), AMDCUID_DEVICE_TYPE_NIC, &handle),
            AMDCUID_STATUS_DEVICE_NOT_FOUND);
  EXPECT_EQ(amdcuid_get_handle_by_bdf("0000:ee:1f.7", AMDCUID_DEVICE_TYPE_NIC, &handle),
            AMDCUID_STATUS_DEVICE_NOT_FOUND);
  uint32_t after = 0;
  amdcuid_get_all_handles(nullptr, &after);
  EXPECT_EQ(after, before);
}

// A property query answers from the handle's device alone: it does not
// re-derive every other device. Compared against a cold
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
  uint8_t machine_id[16];
  if (CuidUtilities::host_machine_id(machine_id) != AMDCUID_STATUS_SUCCESS)
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

// A temporary PCI function whose input the test changes, as a re-read can.
class SerialDevice : public CuidDevice {
 public:
  SerialDevice(std::string bdf, uint64_t serial) : serial(serial), bdf_(std::move(bdf)) {}
  amdcuid_device_type_t type() const override { return AMDCUID_DEVICE_TYPE_OTHER; }
  amdcuid_status_t get_primary_cuid(amdcuid_primary_id&) const override {
    return AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND;
  }
  amdcuid_status_t get_temporary_input(CuidUtilities::AuxiliaryInput& input) const override {
    if (input_status != AMDCUID_STATUS_SUCCESS) return input_status;
    input = {};
    input.format = CuidUtilities::kAuxFormatPcie;
    input.routing_id = static_cast<uint32_t>(serial);
    input.device_id = 0x74a1;
    input.vendor_id = 0x1002;
    input.component_type = AMDCUID_DEVICE_TYPE_OTHER;
    return AMDCUID_STATUS_SUCCESS;
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
  amdcuid_status_t input_status = AMDCUID_STATUS_SUCCESS;

 private:
  std::string bdf_;
};

}  // namespace

// A component whose input turns malformed after discovery is left out of the
// refreshed inventory; every other component is still listed.
TEST(cuidtstUnprivileged, AMalformedComponentLeavesTheOthersListed) {
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
  for (const amdcuid_status_t failure :
       {AMDCUID_STATUS_INVALID_FORMAT, AMDCUID_STATUS_PCI_ERROR, AMDCUID_STATUS_FILE_ERROR}) {
    b->input_status = failure;
    const amdcuid_status_t rebuilt = mgr.build_cuid_index();
    const auto handles = mgr.get_all_handles();
    EXPECT_EQ(rebuilt, AMDCUID_STATUS_SUCCESS) << failure;
    ASSERT_EQ(handles.size(), 1u) << failure;
    EXPECT_TRUE(SameId(handles[0], da.UUIDv8_representation)) << failure;
  }
  mgr.shutdown();
}

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

// An NPU is an AMD signal-processing or accelerator function bound to amdxdna.
// The same class on another driver (the Sensor Fusion Hub) or vendor is not,
// and a lookup by the function's own sysfs directory keeps its BDF.
TEST(cuidtstUnprivileged, NpuNeedsAmdxdna) {
  const ScopedTempDir dir("cuid_npu_");
  ASSERT_FALSE(dir.path().empty());
  const std::string& root = dir.path();
  ASSERT_EQ(mkdir((root + "/drivers").c_str(), 0700), 0);
  for (const char* driver : {"amdxdna", "amd_sfh"})
    ASSERT_EQ(mkdir((root + "/drivers/" + driver).c_str(), 0700), 0);
  struct Case {
    const char* bdf;
    const char* vendor;
    const char* driver;
    amdcuid_status_t want;
  };
  for (const Case& c : {Case{"fffe:fe:1f.0", "0x1022", "amdxdna", AMDCUID_STATUS_SUCCESS},
                        Case{"fffe:fe:1f.1", "0x1022", "amd_sfh", AMDCUID_STATUS_UNSUPPORTED},
                        Case{"fffe:fe:1f.2", "0x1022", nullptr, AMDCUID_STATUS_UNSUPPORTED},
                        Case{"fffe:fe:1f.3", "0x1002", "amdxdna", AMDCUID_STATUS_UNSUPPORTED}}) {
    const std::string fn = root + "/" + c.bdf;
    ASSERT_EQ(mkdir(fn.c_str(), 0700), 0);
    std::ofstream(fn + "/vendor") << c.vendor << "\n";
    std::ofstream(fn + "/device") << "0x1502\n";
    std::ofstream(fn + "/revision") << "0x00\n";
    std::ofstream(fn + "/class") << "0x118000\n";
    if (c.driver)
      ASSERT_EQ(symlink((root + "/drivers/" + c.driver).c_str(), (fn + "/driver").c_str()), 0);
    amdcuid_npu_info info{};
    EXPECT_EQ(CuidNpu::discover_single(&info, fn), c.want) << c.bdf;
    if (c.want == AMDCUID_STATUS_SUCCESS) EXPECT_EQ(info.bdf, c.bdf);
  }
}
