// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// RDMA info is optional: a tolerated status keeps the NIC with a zeroed
// rdma_dev. Driver/port info is optional too, tolerating NO_DATA alone with no
// zeroing guarantee. Every other failure on any of the five required getters
// drops the NIC. Driven through the test seam, so each getter's status is set
// independently and no NIC hardware is needed.
//
// The public getters are reached through an AI-NIC registered with AMDSmiSystem.
// Port and RDMA info go through the same stubs on every call; the rest comes
// from what discovery stored.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <memory>
#include <stdexcept>

#include "amd_smi/impl/amd_smi_nic_testing.h"
#include "nic_unit_fixture.h"
#include "test_fixture_utils.h"

namespace {

using amd::smi::AMDSmiAINICDevice;
using amd::smi::nic_set_info_getters_for_testing;
using amd::smi::populate_amd_ainic_device;
using amd::smi::testing::ScopedProcessorRegistration;

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

// An AI-NIC registered with AMDSmiSystem whose discovery copy is all zeros, so a
// stub-filled result from a public getter can only come from a read made when it
// was called. Heap-allocated: the device holds over 600 KB.
class RegisteredNic {
 public:
  explicit RegisteredNic(const AMDSmiAINICDevice::AINICInfo& discovered)
      : nic_(std::make_unique<AMDSmiAINICDevice>(discovered)), registration_(nic_.get()) {}
  RegisteredNic() : RegisteredNic(*std::make_unique<AMDSmiAINICDevice::AINICInfo>()) {}

  amdsmi_processor_handle handle() const { return registration_.handle(); }

 private:
  // Declaration order is setup order; teardown unregisters before amdsmi_shut_down().
  ScopedAmdsmiInit init_;
  std::unique_ptr<AMDSmiAINICDevice> nic_;
  ScopedProcessorRegistration registration_;
};

TEST_F(NicUnit, PortInfoIsReadWhenCalled) {
  RegisteredNic nic;
  auto port = std::make_unique<amdsmi_nic_port_info_t>();
  EXPECT_EQ(amdsmi_get_nic_port_info(nic.handle(), port.get()), AMDSMI_STATUS_SUCCESS);
  EXPECT_TRUE(IsFilledWith(*port, FillByte(kPort)));

  g_status[kPort] = SMI_NIC_STATUS_DRIVER_NOT_LOADED;
  EXPECT_EQ(amdsmi_get_nic_port_info(nic.handle(), port.get()), AMDSMI_STATUS_DRIVER_NOT_LOADED);

  // A fwctl-only NIC has no port: as at discovery, an empty port info.
  g_status[kPort] = SMI_NIC_STATUS_NO_DATA;
  EXPECT_EQ(amdsmi_get_nic_port_info(nic.handle(), port.get()), AMDSMI_STATUS_SUCCESS);
  EXPECT_TRUE(IsFilledWith(*port, 0));
}

// The amdsmi.h contract for a NIC without RDMA, at the public getter: a tolerated
// status is SUCCESS with a zeroed struct, and every other status surfaces.
TEST_F(NicUnit, RdmaDevInfoIsReadWhenCalled) {
  RegisteredNic nic;
  auto rdma = std::make_unique<amdsmi_nic_rdma_devices_info_t>();
  EXPECT_EQ(amdsmi_get_nic_rdma_dev_info(nic.handle(), rdma.get()), AMDSMI_STATUS_SUCCESS);
  EXPECT_TRUE(IsFilledWith(*rdma, FillByte(kRdma)));

  for (const StatusMapping& mapping : kStatusMappings) {
    g_status[kRdma] = mapping.nic;
    const amdsmi_status_t status = amdsmi_get_nic_rdma_dev_info(nic.handle(), rdma.get());
    if (IsToleratedForRdma(mapping.nic)) {
      EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS) << "smi_nic_status_t=" << mapping.nic;
      ExpectRdmaZeroed(*rdma);
    } else {
      EXPECT_EQ(status, mapping.amdsmi) << "smi_nic_status_t=" << mapping.nic;
    }
  }
}

// The whole-struct getter keeps the discovery copy and reads port and RDMA info now.
TEST_F(NicUnit, AinicInfoReadsPortAndRdmaWhenCalled) {
  RegisteredNic nic;
  auto info = std::make_unique<AMDSmiAINICDevice::AINICInfo>();
  std::memset(info.get(), 0x5A, sizeof(*info));
  EXPECT_EQ(amdsmi_get_ainic_info(nic.handle(), info.get()), AMDSMI_STATUS_SUCCESS);
  EXPECT_TRUE(IsFilledWith(info->asic, 0));
  EXPECT_TRUE(IsFilledWith(info->port, FillByte(kPort)));
  EXPECT_TRUE(IsFilledWith(info->rdma_dev, FillByte(kRdma)));

  g_status[kRdma] = SMI_NIC_STATUS_ERROR;
  EXPECT_EQ(amdsmi_get_ainic_info(nic.handle(), info.get()), AMDSMI_STATUS_API_FAILED);
  g_status[kPort] = SMI_NIC_STATUS_NOT_FOUND;
  EXPECT_EQ(amdsmi_get_ainic_info(nic.handle(), info.get()), AMDSMI_STATUS_NOT_FOUND);
}

// ASIC, bus, driver and NUMA info do not change while the NIC is present, so they
// come from what discovery stored; a getter call now would fail.
TEST_F(NicUnit, StaticInfoComesFromDiscovery) {
  ASSERT_EQ(Populate(), AMDSMI_STATUS_SUCCESS);
  RegisteredNic nic(info_);
  g_status.fill(SMI_NIC_STATUS_ERROR);

  amdsmi_nic_asic_info_t asic = {};
  EXPECT_EQ(amdsmi_get_nic_asic_info(nic.handle(), &asic), AMDSMI_STATUS_SUCCESS);
  EXPECT_TRUE(IsFilledWith(asic, FillByte(kAsic)));
  amdsmi_nic_bus_info_t bus = {};
  EXPECT_EQ(amdsmi_get_nic_bus_info(nic.handle(), &bus), AMDSMI_STATUS_SUCCESS);
  EXPECT_TRUE(IsFilledWith(bus, FillByte(kBus)));
  amdsmi_nic_driver_info_t driver = {};
  EXPECT_EQ(amdsmi_get_nic_driver_info(nic.handle(), &driver), AMDSMI_STATUS_SUCCESS);
  EXPECT_TRUE(IsFilledWith(driver, FillByte(kDriver)));
  amdsmi_nic_numa_info_t numa = {};
  EXPECT_EQ(amdsmi_get_nic_numa_info(nic.handle(), &numa), AMDSMI_STATUS_SUCCESS);
  EXPECT_TRUE(IsFilledWith(numa, FillByte(kNuma)));
}

template <typename Info>
smi_nic_status_t Throw(smi_nic_ctx_t, uint64_t, Info*) {
  throw std::runtime_error("NIC read failed");
}

// A read made on every call must return a status instead of letting an exception from the NIC
// library reach a C caller.
TEST_F(NicUnit, ThrowingNicReadIsAStatus) {
  static const amd::smi::nic_info_getters_t kThrowingGetters = [] {
    amd::smi::nic_info_getters_t getters = kStubGetters;
    getters.port = Throw<smi_nic_port_info_t>;
    getters.rdma = Throw<smi_nic_rdma_devices_info_t>;
    return getters;
  }();
  RegisteredNic nic;
  nic_set_info_getters_for_testing(&kThrowingGetters);
  auto port = std::make_unique<amdsmi_nic_port_info_t>();
  EXPECT_EQ(amdsmi_get_nic_port_info(nic.handle(), port.get()), AMDSMI_STATUS_INTERNAL_EXCEPTION);
  auto rdma = std::make_unique<amdsmi_nic_rdma_devices_info_t>();
  EXPECT_EQ(amdsmi_get_nic_rdma_dev_info(nic.handle(), rdma.get()),
            AMDSMI_STATUS_INTERNAL_EXCEPTION);
}

constexpr uint16_t kUnknownMtu = UINT16_MAX;

void SetRdmaPort(amdsmi_nic_rdma_dev_info_t& device, const char* name, uint8_t port_num,
                 const char* state, uint16_t mtu) {
  std::snprintf(device.rdma_dev, sizeof(device.rdma_dev), "%s", name);
  device.num_rdma_ports = 1;
  amdsmi_nic_rdma_port_info_t& port = device.rdma_port_info[0];
  port.rdma_port = port_num;
  std::snprintf(port.state, sizeof(port.state), "%s", state);
  port.max_mtu = mtu;
  port.active_mtu = mtu;
}

// The state read now: rdma0 went down, and rdma1 appeared after discovery.
smi_nic_status_t LiveRdma(smi_nic_ctx_t, uint64_t, smi_nic_rdma_devices_info_t* info) {
  auto* rdma = reinterpret_cast<amdsmi_nic_rdma_devices_info_t*>(info);
  std::memset(rdma, 0, sizeof(*rdma));
  rdma->num_rdma_dev = 2;
  SetRdmaPort(rdma->rdma_dev_info[0], "rdma0", 1, "DOWN", kUnknownMtu);
  SetRdmaPort(rdma->rdma_dev_info[1], "rdma1", 1, "ACTIVE", kUnknownMtu);
  return SMI_NIC_STATUS_SUCCESS;
}

// The MTU probe forks per port, so it runs once; the state is read on every call.
TEST_F(NicUnit, RdmaDevInfoKeepsTheProbedMtu) {
  static const amd::smi::nic_info_getters_t kLiveRdmaGetters = [] {
    amd::smi::nic_info_getters_t getters = kStubGetters;
    getters.rdma = LiveRdma;
    return getters;
  }();
  constexpr uint16_t kProbedMtu = 4096;
  auto discovered = std::make_unique<AMDSmiAINICDevice::AINICInfo>();
  discovered->rdma_dev.num_rdma_dev = 1;
  SetRdmaPort(discovered->rdma_dev.rdma_dev_info[0], "rdma0", 1, "ACTIVE", kProbedMtu);
  RegisteredNic nic(*discovered);
  nic_set_info_getters_for_testing(&kLiveRdmaGetters);

  auto rdma = std::make_unique<amdsmi_nic_rdma_devices_info_t>();
  ASSERT_EQ(amdsmi_get_nic_rdma_dev_info(nic.handle(), rdma.get()), AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(rdma->num_rdma_dev, 2);
  const amdsmi_nic_rdma_port_info_t& known = rdma->rdma_dev_info[0].rdma_port_info[0];
  EXPECT_STREQ(known.state, "DOWN");
  EXPECT_EQ(known.max_mtu, kProbedMtu);
  EXPECT_EQ(known.active_mtu, kProbedMtu);
  const amdsmi_nic_rdma_port_info_t& added = rdma->rdma_dev_info[1].rdma_port_info[0];
  EXPECT_STREQ(added.state, "ACTIVE");
  EXPECT_EQ(added.max_mtu, kUnknownMtu);

  auto info = std::make_unique<AMDSmiAINICDevice::AINICInfo>();
  ASSERT_EQ(amdsmi_get_ainic_info(nic.handle(), info.get()), AMDSMI_STATUS_SUCCESS);
  EXPECT_STREQ(info->rdma_dev.rdma_dev_info[0].rdma_port_info[0].state, "DOWN");
  EXPECT_EQ(info->rdma_dev.rdma_dev_info[0].rdma_port_info[0].max_mtu, kProbedMtu);
}

}  // namespace
