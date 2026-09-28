// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// UnitID names a sub-unit of a component, not a place. A CPU package is
// published as a whole, so its UnitID is zero; a topology address there gives
// one socket a different primary CUID per discovery path.

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "src/cuid_cpu.h"
#include "src/cuid_util.h"

namespace {

std::string PrimaryOf(const amdcuid_cpu_info& info) {
  CuidCpu cpu(info);
  amdcuid_primary_id primary = {};
  if (cpu.get_primary_cuid(primary) != AMDCUID_STATUS_SUCCESS) return "";
  const char* str = amdcuid_id_to_string(primary.UUIDv8_representation);
  return str ? str : "";
}

}  // namespace

// Passes on a single socket even with an APIC ID in UnitID (the first
// package's lowest is 0); the next case catches that.
TEST(cuidtst, CpuUnitIdIsZeroForAWholePackage) {
  std::vector<DevicePtr> cpus;
  ASSERT_EQ(CuidCpu::discover(cpus), AMDCUID_STATUS_SUCCESS);
  if (cpus.empty()) GTEST_SKIP() << "no CPU discovered";

  for (const auto& device : cpus) {
    uint16_t unit_id = 0xFFFF;
    ASSERT_EQ(device->get_unit_id(unit_id), AMDCUID_STATUS_SUCCESS);
    EXPECT_EQ(unit_id, 0) << "a CPU package reported a non-zero UnitID, which can only be a "
                             "topology address: UnitID names a sub-unit, not a location";
  }
}

TEST(cuidtst, OneSocketHasOneIdentityWhicheverCpuFoundIt) {
  std::vector<DevicePtr> cpus;
  ASSERT_EQ(CuidCpu::discover(cpus), AMDCUID_STATUS_SUCCESS);
  if (cpus.empty()) GTEST_SKIP() << "no CPU discovered";

  std::vector<std::string> paths;
  for (int cpu = 0; cpu < 64 && paths.size() < 8; ++cpu) {
    const std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(cpu);
    if (access(path.c_str(), F_OK) == 0) paths.push_back(path);
  }
  if (paths.size() < 2) GTEST_SKIP() << "fewer than two logical CPUs to compare";

  // Keyed by socket: distinct sockets must differ.
  std::map<uint16_t, std::set<std::string>> identities_by_socket;
  size_t resolved = 0;

  for (const auto& path : paths) {
    amdcuid_cpu_info info = {};
    if (CuidCpu::discover_single(&info, path) != AMDCUID_STATUS_SUCCESS) continue;

    EXPECT_EQ(info.header.fields.cpu.unit_id, 0) << path << " put a logical CPU number in UnitID";

    const std::string primary = PrimaryOf(info);
    if (primary.empty()) continue;  // no PPIN and no SMBIOS UUID, or unprivileged
    identities_by_socket[info.header.fields.cpu.physical_id].insert(primary);
    ++resolved;
  }
  if (resolved < 2) GTEST_SKIP() << "fewer than two logical CPUs resolved to an identity";

  for (const auto& socket : identities_by_socket) {
    EXPECT_EQ(socket.second.size(), 1u)
        << "socket " << socket.first << " reported " << socket.second.size()
        << " different primary CUIDs depending on which logical CPU was used to find it";
  }
}

// Every socket has UnitID 0, so without a fingerprint only the Routing ID
// separates two sockets of one host. Socket 0 keeps the zero Routing ID.
TEST(cuidtst, TwoSocketsGetDistinctTemporaryCuids) {
  amdcuid_cpu_info socket0 = {};
  socket0.header.device_type = AMDCUID_DEVICE_TYPE_CPU;
  socket0.header.fields.cpu.vendor_id = 0x1022;
  socket0.header.fields.cpu.device_id = 0x1A11;
  socket0.header.fields.cpu.revision_id = 0x00;
  amdcuid_cpu_info socket1 = socket0;
  socket1.header.fields.cpu.physical_id = 1;

  amdcuid_primary_id first = {};
  amdcuid_primary_id second = {};
  const amdcuid_status_t status = CuidCpu(socket0).get_auxiliary_primary_cuid(first);
  if (status != AMDCUID_STATUS_SUCCESS) GTEST_SKIP() << "no machine-id, so no temporary CUID";
  ASSERT_EQ(CuidCpu(socket1).get_auxiliary_primary_cuid(second), AMDCUID_STATUS_SUCCESS);
  EXPECT_NE(std::memcmp(first.UUIDv8_representation.bytes, second.UUIDv8_representation.bytes,
                        sizeof(first.UUIDv8_representation.bytes)),
            0)
      << "two sockets of one host share a temporary CUID";

  CuidUtilities::AuxiliaryInput aux;
  aux.format = CuidUtilities::kAuxFormatCpu;
  aux.routing_id = 0;
  aux.device_id = 0x1A11;
  aux.vendor_id = 0x1022;
  aux.component_type = static_cast<uint8_t>(AMDCUID_DEVICE_TYPE_CPU);
  uint64_t fingerprint = 0;
  ASSERT_EQ(CuidUtilities::make_fallback_fingerprint(aux, fingerprint), AMDCUID_STATUS_SUCCESS);
  amdcuid_primary_id expected = {};
  ASSERT_EQ(CuidUtilities::generate_primary_cuid(fingerprint, 0, 0x00, 0x1A11, 0x1022,
                                                 AMDCUID_DEVICE_TYPE_CPU, &expected, true),
            AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(std::memcmp(first.UUIDv8_representation.bytes, expected.UUIDv8_representation.bytes,
                        sizeof(expected.UUIDv8_representation.bytes)),
            0)
      << "socket 0 no longer matches the zero-Routing-ID structure A-CPU pins";
}
