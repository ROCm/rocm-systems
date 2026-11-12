// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// The temporary CUID: HMAC-SHA256(key = S, msg = the CUID application id),
// truncated to 16 octets and framed as a UUIDv8 with payload bit 117 set.
// Known answers computed outside this library.

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "src/cuid_device.h"
#include "src/cuid_device_manager.h"
#include "src/cuid_gpu.h"
#include "src/cuid_util.h"
#include "test_common.h"

namespace {

constexpr uint8_t kMachineId[16] = {0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
                                    0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef};

std::string Hex(const uint8_t* data, size_t len) {
  static const char kDigits[] = "0123456789abcdef";
  std::string out;
  for (size_t i = 0; i < len; ++i) {
    out.push_back(kDigits[data[i] >> 4]);
    out.push_back(kDigits[data[i] & 0x0F]);
  }
  return out;
}

CuidUtilities::AuxiliaryInput Pcie(const char* bdf, uint8_t revision_id, uint16_t device_id,
                                   uint16_t vendor_id, amdcuid_device_type_t type,
                                   uint16_t unit_id) {
  CuidUtilities::AuxiliaryInput in;
  in.format = CuidUtilities::kAuxFormatPcie;
  CuidUtilities::set_pcie_routing(in, bdf);
  in.revision_id = revision_id;
  in.device_id = device_id;
  in.vendor_id = vendor_id;
  in.component_type = static_cast<uint8_t>(type);
  in.unit_id = unit_id;
  return in;
}

CuidUtilities::AuxiliaryInput Gpu(uint16_t unit_id, const char* bdf = "0000:63:00.0") {
  return Pcie(bdf, 0x00, 0x73A3, 0x1002, AMDCUID_DEVICE_TYPE_GPU, unit_id);
}

std::string S(const CuidUtilities::AuxiliaryInput& in, const uint8_t* machine_id = kMachineId) {
  uint8_t structure[32] = {0};
  CuidUtilities::pack_auxiliary_input(in, machine_id, structure);
  return Hex(structure, sizeof(structure));
}

std::string Temporary(const CuidUtilities::AuxiliaryInput& in,
                      const uint8_t* machine_id = kMachineId) {
  amdcuid_derived_id id = {};
  CuidUtilities::temporary_cuid(in, machine_id, &id);
  return CuidUtilities::get_cuid_as_string(&id.UUIDv8_representation);
}

bool HaveMachineId() {
  uint8_t machine_id[16];
  return CuidUtilities::host_machine_id(machine_id) == AMDCUID_STATUS_SUCCESS;
}

}  // namespace

TEST(cuidtstUnprivileged, TemporaryKatGpu) {
  EXPECT_EQ(S(Gpu(0)), "01000123456789abcdef0123456789abcdef0063000000a37302100200000000");
  EXPECT_EQ(Temporary(Gpu(0)), "df21fafe-884b-89e0-bc2b-f5ee8a5f7dde");
}

// Partitions share their parent's BDF; only UnitID at S 224:239 keeps them
// apart from it and from each other. DPX: partitions 0 and 1, UnitIDs 2 and 3.
TEST(cuidtstUnprivileged, TemporaryKatPartitionsAreDistinct) {
  EXPECT_EQ(S(Gpu(3)), "01000123456789abcdef0123456789abcdef0063000000a37302100203000000");
  EXPECT_EQ(Temporary(Gpu(2)), "bff8449d-a0e5-85c5-8a9d-b57c47533dde");
  EXPECT_EQ(Temporary(Gpu(3)), "c9ce22eb-5353-8733-8785-615e393b6ccc");
  EXPECT_NE(Temporary(Gpu(2)), Temporary(Gpu(0)));
  EXPECT_NE(Temporary(Gpu(3)), Temporary(Gpu(0)));
  EXPECT_NE(Temporary(Gpu(2)), Temporary(Gpu(3)));
}

// The Routing ID keeps only the segment's low 16 bits, so a VMD domain is told
// from domain 0 by the segment's high half at S 240:255.
TEST(cuidtstUnprivileged, TemporaryKatVmdDomainIsDistinct) {
  const auto low = Gpu(0);
  const auto vmd = Gpu(0, "10000:63:00.0");
  EXPECT_EQ(low.routing_id, vmd.routing_id);
  EXPECT_EQ(S(vmd), "01000123456789abcdef0123456789abcdef0063000000a37302100200000100");
  EXPECT_EQ(Temporary(vmd), "ce8f02b6-3e76-8cda-9786-6a729a4afeaf");
  EXPECT_NE(Temporary(vmd), Temporary(low));
}

TEST(cuidtstUnprivileged, TemporaryKatNicCpuAndPlatform) {
  const auto nic = Pcie("0000:41:00.1", 0x01, 0x1658, 0x14E4, AMDCUID_DEVICE_TYPE_NIC, 1);
  EXPECT_EQ(S(nic), "01000123456789abcdef0123456789abcdef01410000015816e4140301000000");
  EXPECT_EQ(Temporary(nic), "d32ba2dc-84f7-8e98-b14c-b7e878e47dfe");

  CuidUtilities::AuxiliaryInput cpu;
  cpu.format = CuidUtilities::kAuxFormatCpu;
  cpu.revision_id = 0x01;
  cpu.device_id = 0x1911;
  cpu.vendor_id = 0x1022;
  cpu.component_type = AMDCUID_DEVICE_TYPE_CPU;
  EXPECT_EQ(S(cpu), "02000123456789abcdef0123456789abcdef0000000001111922100100000000");
  EXPECT_EQ(Temporary(cpu), "908e8d6e-bce7-81df-93dc-ccf675f01ae7");

  CuidUtilities::AuxiliaryInput platform;
  platform.format = CuidUtilities::kAuxFormatCpu;
  platform.component_type = AMDCUID_DEVICE_TYPE_PLATFORM;
  EXPECT_EQ(S(platform), "02000123456789abcdef0123456789abcdef0000000000000000000000000000");
  EXPECT_EQ(Temporary(platform), "77dc6991-eb75-845c-806a-5c82f066b38a");
}

// Over many inputs, the framing gives an RFC 9562 v8 with octet 15 bit 7 set,
// which this library de-frames to payload bit 117, the Auxiliary Value
// Identifier.
TEST(cuidtstUnprivileged, TemporaryCuidIsAValidTemporaryDerivedPayload) {
  std::mt19937_64 rng(20261001);
  uint8_t machine_id[16];
  for (int n = 0; n < 20000; ++n) {
    CuidUtilities::AuxiliaryInput in;
    in.format = static_cast<uint16_t>(rng());
    in.routing_id = static_cast<uint32_t>(rng());
    in.revision_id = static_cast<uint8_t>(rng());
    in.device_id = static_cast<uint16_t>(rng());
    in.vendor_id = static_cast<uint16_t>(rng());
    in.component_type = static_cast<uint8_t>(rng() & 0xF);
    in.unit_id = static_cast<uint16_t>(rng() & 0x1FFF);
    in.segment_high = static_cast<uint16_t>(rng());
    for (auto& octet : machine_id) octet = static_cast<uint8_t>(rng());
    amdcuid_derived_id id = {};
    CuidUtilities::temporary_cuid(in, machine_id, &id);
    const uint8_t* b = id.UUIDv8_representation.bytes;
    ASSERT_EQ(b[6] >> 4, 8) << n;
    ASSERT_EQ(b[8] >> 6, 2) << n;
    ASSERT_TRUE(CuidUtilities::is_constructed(&id.UUIDv8_representation)) << n;
    uint8_t raw[16];
    CuidUtilities::remove_UUIDv8_bits(&id.UUIDv8_representation, raw);
    ASSERT_EQ(std::memcmp(raw, id.raw_bits, 16), 0) << n;
    ASSERT_NE(b[15] & 0x80, 0) << n;
    ASSERT_NE(raw[14] & 0x20, 0) << n << ": payload bit 117";
  }
}

// The machine-id is in S, at octets 2:17 in file order, so it alone separates
// two hosts with the same part in the same slot.
TEST(cuidtstUnprivileged, TemporaryMachineIdIsInS) {
  uint8_t structure[32];
  CuidUtilities::pack_auxiliary_input(Gpu(0), kMachineId, structure);
  EXPECT_EQ(std::memcmp(structure + 2, kMachineId, 16), 0);
  uint8_t other[16];
  std::memcpy(other, kMachineId, 16);
  other[15] ^= 0x01;
  EXPECT_NE(S(Gpu(0), other), S(Gpu(0)));
  EXPECT_NE(Temporary(Gpu(0), other), Temporary(Gpu(0)));
}

// No temporary CUID without a valid machine-id: an empty, "uninitialized" or
// all-zero /etc/machine-id is an image whose identity is not set yet, and the
// D-Bus id beside it is not consulted.
TEST(cuidtstUnprivileged, TemporaryNeedsAValidMachineId) {
  const ScopedTempDir dir("cuid_machine_id_");
  ASSERT_FALSE(dir.path().empty());
  const std::string etc = dir.path() + "/machine-id";
  const std::string dbus = dir.path() + "/dbus-machine-id";
  std::ofstream(dbus) << "0123456789abcdef0123456789abcdef\n";
  uint8_t out[16];
  for (const char* content : {"", "\n", "uninitialized\n", "00000000000000000000000000000000\n",
                              "0123456789abcdef0123456789abcde\n"}) {
    std::ofstream(etc, std::ios::trunc) << content;
    std::memset(out, 0x5A, sizeof(out));
    EXPECT_EQ(CuidUtilities::host_machine_id(etc, dbus, out),
              AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND)
        << "'" << content << "'";
  }
  EXPECT_EQ(CuidUtilities::host_machine_id(dir.path() + "/absent", dir.path() + "/absent2", out),
            AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND);
  ASSERT_EQ(CuidUtilities::host_machine_id(dir.path() + "/absent", dbus, out),
            AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(std::memcmp(out, kMachineId, 16), 0);
}

// One warning for the host, not one per component.
TEST(cuidtstUnprivileged, NoMachineIdWarnsOnce) {
  ::testing::internal::CaptureStderr();
  for (int i = 0; i < 3; ++i) {
    uint8_t out[16];
    EXPECT_EQ(CuidUtilities::host_machine_id("/nonexistent/machine-id", "/nonexistent/dbus", out),
              AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND);
  }
  const std::string err = ::testing::internal::GetCapturedStderr();
  size_t count = 0;
  for (size_t at = err.find("No machine identity"); at != std::string::npos;
       at = err.find("No machine identity", at + 1))
    ++count;
  EXPECT_LE(count, 1u) << err;
}

TEST(cuidtstUnprivileged, BdfWithAVmdSegmentParses) {
  EXPECT_TRUE(CuidUtilities::is_valid_bdf("10000:04:00.0"));
  EXPECT_TRUE(CuidUtilities::is_valid_bdf("ffffffff:04:00.0"));
  EXPECT_FALSE(CuidUtilities::is_valid_bdf("100000000:04:00.0"));
  EXPECT_FALSE(CuidUtilities::is_valid_bdf("000:04:00.0"));
  EXPECT_EQ(CuidUtilities::routing_id_from_bdf("10000:04:00.0"), 0x00000400u);
  EXPECT_EQ(CuidUtilities::segment_high_from_bdf("10000:04:00.0"), 0x0001);
  EXPECT_EQ(CuidUtilities::segment_high_from_bdf("0000:04:00.0"), 0x0000);

  char dir[] = "/tmp/cuid_vmd_XXXXXX";
  ASSERT_NE(mkdtemp(dir), nullptr);
  const std::string link = std::string(dir) + "/device";
  ASSERT_EQ(symlink("../../../pci10000:00/10000:04:00.0", link.c_str()), 0);
  EXPECT_EQ(CuidUtilities::readlink_bdf(link), "10000:04:00.0");
  unlink(link.c_str());
  rmdir(dir);
}

namespace {

// A whole GPU the driver does not name, so temporary.
amdcuid_gpu_info UnnamedGpu() {
  amdcuid_gpu_info info = {};
  info.header.device_type = AMDCUID_DEVICE_TYPE_GPU;
  info.header.fields.gpu.vendor_id = 0x1002;
  info.header.fields.gpu.device_id = 0x73A3;
  info.bdf = "fffe:63:00.0";
  info.render_node = "/nonexistent/cuid-temporary-gpu";
  return info;
}

}  // namespace

// A temporary component has no primary: none is computed or written, and its
// derived CUID is the temporary one with nothing else asked of it.
TEST(cuidtstUnprivileged, TemporaryComponentHasNoPrimary) {
  if (!HaveMachineId()) GTEST_SKIP() << "no machine-id, so no temporary CUID";
  const CuidGpu gpu(UnnamedGpu());
  amdcuid_primary_id primary;
  std::memset(&primary, 0xA5, sizeof(primary));
  const amdcuid_primary_id untouched = primary;
  EXPECT_EQ(gpu.get_primary_cuid(primary), AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND);
  EXPECT_EQ(std::memcmp(&primary, &untouched, sizeof(primary)), 0);

  amdcuid_derived_id derived = {};
  ASSERT_EQ(gpu.get_derived_cuid(derived, nullptr), AMDCUID_STATUS_SUCCESS);
  amdcuid_derived_id expected = {};
  ASSERT_EQ(CuidUtilities::temporary_cuid(Gpu(0, "fffe:63:00.0"), &expected),
            AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(std::memcmp(&derived.UUIDv8_representation, &expected.UUIDv8_representation, 16), 0);
  bool temporary = false;
  ASSERT_EQ(gpu.is_temporary_cuid(&temporary, nullptr), AMDCUID_STATUS_SUCCESS);
  EXPECT_TRUE(temporary);

  // Through the API: every query but the primary answers, and the primary is
  // refused without writing (root) or for want of privilege (anyone else).
  const auto device = std::make_shared<CuidGpu>(UnnamedGpu());
  const amdcuid_id_t handle = derived.UUIDv8_representation;
  CuidDeviceManager& mgr = CuidDeviceManager::instance();
  mgr.shutdown();
  ASSERT_EQ(mgr.index_handle(device, handle), AMDCUID_STATUS_SUCCESS);
  uint16_t vendor_id = 0, device_id = 0, unit_id = 0xFFFF;
  amdcuid_device_type_t type = AMDCUID_DEVICE_TYPE_NONE;
  amdcuid_source_t source = AMDCUID_SOURCE_UNKNOWN;
  bool flag = false;
  char path[128] = {0};
  uint32_t length = sizeof(vendor_id);
  EXPECT_EQ(amdcuid_query_device_property(handle, AMDCUID_QUERY_VENDOR_ID, &vendor_id, &length),
            AMDCUID_STATUS_SUCCESS);
  length = sizeof(device_id);
  EXPECT_EQ(amdcuid_query_device_property(handle, AMDCUID_QUERY_DEVICE_ID, &device_id, &length),
            AMDCUID_STATUS_SUCCESS);
  length = sizeof(unit_id);
  EXPECT_EQ(amdcuid_query_device_property(handle, AMDCUID_QUERY_UNIT_ID, &unit_id, &length),
            AMDCUID_STATUS_SUCCESS);
  length = sizeof(type);
  EXPECT_EQ(amdcuid_query_device_property(handle, AMDCUID_QUERY_DEVICE_TYPE, &type, &length),
            AMDCUID_STATUS_SUCCESS);
  length = sizeof(flag);
  EXPECT_EQ(amdcuid_query_device_property(handle, AMDCUID_QUERY_TEMPORARY_CUID, &flag, &length),
            AMDCUID_STATUS_SUCCESS);
  length = sizeof(source);
  EXPECT_EQ(amdcuid_query_device_property(handle, AMDCUID_QUERY_SOURCE, &source, &length),
            AMDCUID_STATUS_SUCCESS);
  length = sizeof(path);
  EXPECT_EQ(amdcuid_query_device_property(handle, AMDCUID_QUERY_DEVICE_PATH, path, &length),
            AMDCUID_STATUS_SUCCESS);
  amdcuid_id_t refused;
  std::memset(&refused, 0xA5, sizeof(refused));
  const amdcuid_id_t sentinel = refused;
  length = sizeof(refused);
  const amdcuid_status_t primary_status =
      amdcuid_query_device_property(handle, AMDCUID_QUERY_PRIMARY_CUID, &refused, &length);
  mgr.shutdown();

  EXPECT_EQ(vendor_id, 0x1002);
  EXPECT_EQ(device_id, 0x73A3);
  EXPECT_EQ(unit_id, 0);
  EXPECT_EQ(type, AMDCUID_DEVICE_TYPE_GPU);
  EXPECT_TRUE(flag);
  EXPECT_EQ(source, AMDCUID_SOURCE_LIBRARY);
  EXPECT_STREQ(path, "/nonexistent/cuid-temporary-gpu");
  EXPECT_EQ(primary_status,
            geteuid() == 0 ? AMDCUID_STATUS_UNSUPPORTED : AMDCUID_STATUS_PERMISSION_DENIED);
  EXPECT_EQ(std::memcmp(&refused, &sentinel, sizeof(refused)), 0);
}

// A temporary CUID, written out as a string and read back, still names its
// component: the index and every query agree with the value handed out.
TEST(cuidtstUnprivileged, ReverseLookupByTemporaryCuidString) {
  if (!HaveMachineId()) GTEST_SKIP() << "no machine-id, so no temporary CUID";
  const auto device = std::make_shared<CuidGpu>(UnnamedGpu());
  amdcuid_derived_id derived = {};
  ASSERT_EQ(device->get_derived_cuid(derived, nullptr), AMDCUID_STATUS_SUCCESS);
  const std::string text = CuidUtilities::get_cuid_as_string(&derived.UUIDv8_representation);
  amdcuid_id_t parsed = {};
  ASSERT_EQ(CuidUtilities::uuid_string_to_uint8(text, parsed.bytes), AMDCUID_STATUS_SUCCESS);

  CuidDeviceManager& mgr = CuidDeviceManager::instance();
  mgr.shutdown();
  ASSERT_EQ(mgr.index_handle(device, parsed), AMDCUID_STATUS_SUCCESS);
  const bool found = mgr.lookup_by_handle(parsed) == device;
  amdcuid_id_t current = {};
  uint32_t length = sizeof(current);
  const amdcuid_status_t status =
      amdcuid_query_device_property(parsed, AMDCUID_QUERY_DERIVED_CUID, &current, &length);
  mgr.shutdown();
  EXPECT_TRUE(found);
  EXPECT_EQ(status, AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(std::memcmp(&current, &parsed, sizeof(parsed)), 0);

  // This host's own temporary components: each handle, as a string, is the
  // temporary CUID of the properties the library reports for it.
  uint32_t count = 0;
  if (amdcuid_get_all_handles(nullptr, &count) != AMDCUID_STATUS_INSUFFICIENT_SIZE) return;
  std::vector<amdcuid_id_t> handles(count);
  ASSERT_EQ(amdcuid_get_all_handles(handles.data(), &count), AMDCUID_STATUS_SUCCESS);
  for (const auto& handle : handles) {
    bool temporary = false;
    length = sizeof(temporary);
    ASSERT_EQ(
        amdcuid_query_device_property(handle, AMDCUID_QUERY_TEMPORARY_CUID, &temporary, &length),
        AMDCUID_STATUS_SUCCESS);
    if (!temporary) continue;
    amdcuid_id_t again = {};
    ASSERT_EQ(CuidUtilities::uuid_string_to_uint8(amdcuid_id_to_string(handle), again.bytes),
              AMDCUID_STATUS_SUCCESS);
    amdcuid_device_type_t type = AMDCUID_DEVICE_TYPE_NONE;
    length = sizeof(type);
    EXPECT_EQ(amdcuid_query_device_property(again, AMDCUID_QUERY_DEVICE_TYPE, &type, &length),
              AMDCUID_STATUS_SUCCESS);
    if (GpuPartition(again)) continue;
    amdcuid_id_t rebuilt = {};
    ASSERT_TRUE(RebuiltTemporaryCuid(again, rebuilt)) << amdcuid_id_to_string(handle);
    EXPECT_EQ(std::memcmp(&rebuilt, &handle, sizeof(handle)), 0)
        << "type " << type << ": " << amdcuid_id_to_string(handle) << " vs rebuilt "
        << amdcuid_id_to_string(rebuilt);
  }
}
