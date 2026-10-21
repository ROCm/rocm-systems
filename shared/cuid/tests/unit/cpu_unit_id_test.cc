// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// UnitID names a sub-unit of a component, not a place. A CPU package is
// published as a whole, so its UnitID is zero; a topology address there gives
// one socket a different primary CUID per discovery path.

#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "src/cuid_cpu.h"
#include "src/cuid_util.h"
#include "test_common.h"

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

  amdcuid_derived_id first = {};
  amdcuid_derived_id second = {};
  const amdcuid_status_t status = CuidCpu(socket0).get_temporary_cuid(first);
  if (status != AMDCUID_STATUS_SUCCESS) GTEST_SKIP() << "no machine-id, so no temporary CUID";
  ASSERT_EQ(CuidCpu(socket1).get_temporary_cuid(second), AMDCUID_STATUS_SUCCESS);
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
  amdcuid_derived_id expected = {};
  ASSERT_EQ(CuidUtilities::temporary_cuid(aux, &expected), AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(std::memcmp(first.UUIDv8_representation.bytes, expected.UUIDv8_representation.bytes,
                        sizeof(expected.UUIDv8_representation.bytes)),
            0)
      << "socket 0 does not match the zero-Routing-ID structure TV-CPU pins";
}

// A package ID that is not a 16-bit decimal is refused, never truncated or
// thrown through the C API.
TEST(cuidtstUnprivileged, CpuLookupRefusesAMalformedPackageId) {
#ifndef __x86_64__
  GTEST_SKIP() << "no CPUID, so no CPU CUID";
#endif
  for (const char* package : {"x1", "70000", "-1", "1x"}) {
    const ScopedTempDir dir("cuid_cpu_");
    ASSERT_FALSE(dir.path().empty());
    ASSERT_EQ(mkdir((dir.path() + "/topology").c_str(), 0700), 0);
    std::ofstream(dir.path() + "/topology/physical_package_id") << package << "\n";
    std::ofstream(dir.path() + "/topology/core_id") << "0\n";
    amdcuid_cpu_info info{};
    amdcuid_status_t status = AMDCUID_STATUS_SUCCESS;
    EXPECT_NO_THROW(status = CuidCpu::discover_single(&info, dir.path())) << package;
    if (status == AMDCUID_STATUS_UNSUPPORTED) GTEST_SKIP() << "CPU vendor has no CUID";
    EXPECT_EQ(status, AMDCUID_STATUS_INVALID_FORMAT) << package;
  }
}

// Packages come from sysfs, each with its own family, model and stepping from
// its first /proc/cpuinfo block. A logical CPU with a bad package ID, or a
// package whose first block is malformed (package 2, even with a good second
// block), is left out; the others are still listed.
TEST(cuidtstUnprivileged, CpuPackagesComeFromSysfsWithTheirOwnSignature) {
#ifndef __x86_64__
  GTEST_SKIP() << "no CPUID, so no CPU CUID";
#endif
  const ScopedTempDir dir("cuid_cpus_");
  ASSERT_FALSE(dir.path().empty());
  const std::string base = dir.path() + "/cpu";
  ASSERT_EQ(mkdir(base.c_str(), 0700), 0);
  for (const auto& c :
       {std::make_pair("cpu0", "0"), std::make_pair("cpu1", "0"), std::make_pair("cpu4", "1"),
        std::make_pair("cpu5", "-1"), std::make_pair("cpu7", "2")}) {
    const std::string cpu = base + "/" + c.first;
    ASSERT_EQ(mkdir(cpu.c_str(), 0700), 0);
    ASSERT_EQ(mkdir((cpu + "/topology").c_str(), 0700), 0);
    std::ofstream(cpu + "/topology/physical_package_id") << c.second << "\n";
  }
  ASSERT_EQ(mkdir((base + "/cpu6").c_str(), 0700), 0);
  const std::string cpuinfo = dir.path() + "/cpuinfo";
  std::ofstream(cpuinfo) << "processor\t: 0\nphysical id\t: 0\ncpu family\t: 25\nmodel\t\t: 1\n"
                            "stepping\t: 1\n\nprocessor\t: 4\nphysical id\t: 1\ncpu family\t: 25\n"
                            "model\t\t: 1\nstepping\t: 2\n\nprocessor\t: 7\nphysical id\t: 2\n"
                            "cpu family\t: 25\nmodel\t\t: x\nstepping\t: 1\n\nprocessor\t: 8\n"
                            "physical id\t: 2\ncpu family\t: 25\nmodel\t\t: 1\nstepping\t: 1\n\n";

  std::vector<DevicePtr> cpus;
  const amdcuid_status_t status = CuidCpu::discover(base, cpuinfo, cpus);
  if (status == AMDCUID_STATUS_UNSUPPORTED) GTEST_SKIP() << "CPU vendor has no CUID";
  ASSERT_EQ(status, AMDCUID_STATUS_SUCCESS);
  ASSERT_EQ(cpus.size(), 2u);
  for (size_t i = 0; i < cpus.size(); ++i) {
    uint16_t package = 0xFFFF, device_id = 0;
    uint8_t stepping = 0;
    std::string path;
    ASSERT_EQ(cpus[i]->get_physical_id(package), AMDCUID_STATUS_SUCCESS);
    ASSERT_EQ(cpus[i]->get_device_id(device_id), AMDCUID_STATUS_SUCCESS);
    ASSERT_EQ(cpus[i]->get_revision_id(stepping), AMDCUID_STATUS_SUCCESS);
    ASSERT_EQ(cpus[i]->get_device_path(path), AMDCUID_STATUS_SUCCESS);
    EXPECT_EQ(package, i);
    EXPECT_EQ(device_id, 0x1901);
    EXPECT_EQ(stepping, i + 1);
    EXPECT_EQ(path, i == 0 ? "/sys/devices/system/cpu/cpu0" : "/sys/devices/system/cpu/cpu4");
  }
}

// The public lookup reports why a CPU path names no component, rather than
// DEVICE_NOT_FOUND for every failure.
TEST(cuidtstUnprivileged, CpuPathLookupReportsTheFailure) {
#ifndef __x86_64__
  GTEST_SKIP() << "no CPUID, so no CPU CUID";
#endif
  const ScopedTempDir dir("cuid_cpu_");
  ASSERT_FALSE(dir.path().empty());
  const std::string cpu = dir.path() + "/cpu99999";
  ASSERT_EQ(mkdir(cpu.c_str(), 0700), 0);
  ASSERT_EQ(mkdir((cpu + "/topology").c_str(), 0700), 0);
  std::ofstream(cpu + "/topology/physical_package_id") << "70000\n";
  std::ofstream(cpu + "/topology/core_id") << "0\n";
  amdcuid_id_t handle{};
  const amdcuid_status_t status =
      amdcuid_get_handle_by_dev_path(cpu.c_str(), AMDCUID_DEVICE_TYPE_CPU, &handle);
  if (status == AMDCUID_STATUS_UNSUPPORTED) GTEST_SKIP() << "CPU vendor has no CUID";
  EXPECT_EQ(status, AMDCUID_STATUS_INVALID_FORMAT);
}
