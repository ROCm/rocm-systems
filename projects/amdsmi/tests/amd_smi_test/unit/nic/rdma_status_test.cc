// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// RDMA info is optional: a tolerated status keeps the NIC with a zeroed
// rdma_dev. Driver/port info is optional too, tolerating NO_DATA alone with no
// zeroing guarantee. Every other failure on any of the five required getters
// drops the NIC. Driven through the test seam, so each getter's status is set
// independently and no NIC hardware is needed.
//
// Coverage stops at populate_amd_ainic_device(). Reaching the public
// amdsmi_get_nic_rdma_dev_info() needs a handle registered with AMDSmiSystem,
// which no seam exposes, so its zeroed-struct contract is unverified here.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>

#include "amd_smi/impl/amd_smi_nic_testing.h"
#include "nic_unit_fixture.h"

namespace {

using amd::smi::AMDSmiAINICDevice;
using amd::smi::nic_set_info_getters_for_testing;
using amd::smi::populate_amd_ainic_device;

// The five getters ahead of the RDMA one; failing any must drop the NIC.
constexpr std::array<size_t, 5> kRequiredGetters = {kBus, kDriver, kAsic, kNuma, kPort};

template <typename T>
bool IsFilledWith(const T& value, unsigned char byte) {
  const auto* bytes = reinterpret_cast<const unsigned char*>(&value);
  return std::all_of(bytes, bytes + sizeof(T), [byte](unsigned char b) { return b == byte; });
}

// Every non-SUCCESS status, paired with the amdsmi_status_t a fatal getter
// failure must surface. Mirrors ainic_status_map in amd_smi_common.h.
struct StatusMapping {
  smi_nic_status_t nic;
  amdsmi_status_t amdsmi;
};

constexpr std::array<StatusMapping, 8> kStatusMappings = {{
    {SMI_NIC_STATUS_ERROR, AMDSMI_STATUS_API_FAILED},
    {SMI_NIC_STATUS_WRONG_PARAM, AMDSMI_STATUS_INVAL},
    {SMI_NIC_STATUS_NOT_FOUND, AMDSMI_STATUS_NOT_FOUND},
    {SMI_NIC_STATUS_NO_RESOURCE, AMDSMI_STATUS_OUT_OF_RESOURCES},
    {SMI_NIC_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_SUPPORTED},
    {SMI_NIC_STATUS_NOT_INIT, AMDSMI_STATUS_NOT_INIT},
    {SMI_NIC_STATUS_NO_DATA, AMDSMI_STATUS_NO_DATA},
    {SMI_NIC_STATUS_DRIVER_NOT_LOADED, AMDSMI_STATUS_DRIVER_NOT_LOADED},
}};

// DRIVER_NOT_LOADED is the highest enumerator, so its value is the count of
// non-SUCCESS statuses. Appending one breaks here rather than silently
// under-covering this table, ainic_status_map, and smi_nic_status_str at once.
static_assert(kStatusMappings.size() == SMI_NIC_STATUS_DRIVER_NOT_LOADED,
              "kStatusMappings must cover every non-SUCCESS smi_nic_status_t");

// The two the RDMA getter is allowed to return without losing the NIC.
bool IsToleratedForRdma(smi_nic_status_t status) {
  return status == SMI_NIC_STATUS_NO_DATA || status == SMI_NIC_STATUS_DRIVER_NOT_LOADED;
}

// kDriver and kPort additionally forgive NO_DATA alone: a fwctl-only NIC has no
// host netdev to query for driver info or link state. Mirrors the carve-outs in
// populate_amd_ainic_device() (amd_smi_system.cc).
bool IsToleratedForDriverOrPort(size_t getter, smi_nic_status_t status) {
  return (((getter == kDriver) || (getter == kPort)) && (status == SMI_NIC_STATUS_NO_DATA));
}

template <size_t N>
bool IsAllZero(const char (&field)[N]) {
  return std::all_of(std::begin(field), std::end(field), [](char c) { return c == '\0'; });
}

// num_rdma_dev alone would still pass if the tolerated path reset only the
// count. Field by field rather than memcmp: `= {}` says nothing about padding.
void ExpectRdmaZeroed(const amdsmi_nic_rdma_devices_info_t& rdma) {
  EXPECT_EQ(rdma.num_rdma_dev, 0);
  for (const amdsmi_nic_rdma_dev_info_t& dev : rdma.rdma_dev_info) {
    EXPECT_TRUE(IsAllZero(dev.rdma_dev));
    EXPECT_TRUE(IsAllZero(dev.node_guid));
    EXPECT_TRUE(IsAllZero(dev.node_type));
    EXPECT_TRUE(IsAllZero(dev.sys_image_guid));
    EXPECT_TRUE(IsAllZero(dev.fw_ver));
    EXPECT_EQ(dev.num_rdma_ports, 0);
    for (const amdsmi_nic_rdma_port_info_t& port : dev.rdma_port_info) {
      EXPECT_TRUE(IsAllZero(port.netdev));
      EXPECT_TRUE(IsAllZero(port.state));
      EXPECT_EQ(port.rdma_port, 0);
      EXPECT_EQ(port.max_mtu, 0);
      EXPECT_EQ(port.active_mtu, 0);
    }
  }
}

// Each sub-struct must carry its own getter's byte, so a swapped
// reinterpret_cast destination among the six near-identical blocks fails here.
// The rdma_dev byte doubles as the check that nothing zeroes it on the success
// path, which would leave it 0 rather than the fill.
TEST_F(NicUnit, AllGettersSucceed) {
  EXPECT_EQ(Populate(), AMDSMI_STATUS_SUCCESS);
  EXPECT_TRUE(IsFilledWith(info_.bus, FillByte(kBus)));
  EXPECT_TRUE(IsFilledWith(info_.driver, FillByte(kDriver)));
  EXPECT_TRUE(IsFilledWith(info_.asic, FillByte(kAsic)));
  EXPECT_TRUE(IsFilledWith(info_.numa, FillByte(kNuma)));
  EXPECT_TRUE(IsFilledWith(info_.port, FillByte(kPort)));
  EXPECT_TRUE(IsFilledWith(info_.rdma_dev, FillByte(kRdma)));
}

// Covers the lookup-miss branches that a status added to smi_nic_interface.h
// but to neither map would take: nic_status_str() must not throw, and
// ainic_to_amdsmi_status() must report the miss.
TEST_F(NicUnit, UnmappedStatusReportsMapError) {
  g_status[kRdma] = static_cast<smi_nic_status_t>(SMI_NIC_STATUS_DRIVER_NOT_LOADED + 1);
  EXPECT_EQ(Populate(), AMDSMI_STATUS_MAP_ERROR);
}

TEST_F(NicUnit, RdmaDriverNotLoadedKeepsNic) {
  g_status[kRdma] = SMI_NIC_STATUS_DRIVER_NOT_LOADED;
  EXPECT_EQ(Populate(), AMDSMI_STATUS_SUCCESS);
  ExpectRdmaZeroed(info_.rdma_dev);
}

TEST_F(NicUnit, RdmaNoDataKeepsNic) {
  g_status[kRdma] = SMI_NIC_STATUS_NO_DATA;
  EXPECT_EQ(Populate(), AMDSMI_STATUS_SUCCESS);
  ExpectRdmaZeroed(info_.rdma_dev);
}

// Unlike the RDMA getter, driver/port never write on this path (they return
// before touching their output struct), so there is no zeroed-struct contract
// to assert here.
TEST_F(NicUnit, DriverNoDataKeepsNic) {
  g_status[kDriver] = SMI_NIC_STATUS_NO_DATA;
  EXPECT_EQ(Populate(), AMDSMI_STATUS_SUCCESS);
}

TEST_F(NicUnit, PortNoDataKeepsNic) {
  g_status[kPort] = SMI_NIC_STATUS_NO_DATA;
  EXPECT_EQ(Populate(), AMDSMI_STATUS_SUCCESS);
}

// Pins the status translation, not just "something went wrong".
TEST_F(NicUnit, RdmaHardFailureMapsToItsAmdsmiStatus) {
  for (const StatusMapping& mapping : kStatusMappings) {
    if (IsToleratedForRdma(mapping.nic)) {
      continue;
    }
    SetUp();
    g_status[kRdma] = mapping.nic;
    EXPECT_EQ(Populate(), mapping.amdsmi) << "smi_nic_status_t=" << mapping.nic;
  }
}

// The required five must drop the NIC for every status, including the two the
// RDMA path forgives, except NO_DATA on driver/port (see
// IsToleratedForDriverOrPort, covered separately by DriverNoDataKeepsNic and
// PortNoDataKeepsNic).
TEST_F(NicUnit, RequiredGetterFailureDropsNicForEveryStatus) {
  for (size_t getter : kRequiredGetters) {
    for (const StatusMapping& mapping : kStatusMappings) {
      if (IsToleratedForDriverOrPort(getter, mapping.nic)) {
        continue;
      }
      SetUp();
      g_status[getter] = mapping.nic;
      EXPECT_EQ(Populate(), mapping.amdsmi)
          << "getter=" << getter << " smi_nic_status_t=" << mapping.nic;
    }
  }
}

// A restore that silently kept the stubs installed would leave the rest of
// amdsmitst discovering NICs through them. With the production getters back,
// a null context fails in smi_get_nic_bus_info before any hardware is touched.
TEST_F(NicUnit, RestoreRebindsProductionGetters) {
  nic_set_info_getters_for_testing(nullptr);
  AMDSmiAINICDevice::AINICInfo info = {};
  EXPECT_EQ(populate_amd_ainic_device(nullptr, 0x1000, info), AMDSMI_STATUS_INVAL);
}

}  // namespace
