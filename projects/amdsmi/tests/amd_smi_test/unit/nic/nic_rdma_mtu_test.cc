// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * The RDMA MTU probe forks once per port, so amdsmi_init must not run it. The first read that
 * returns an MTU fills it, and later reads reuse the answer.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <memory>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_nic_testing.h"
#include "amd_smi/impl/nic/amd_smi_ainic_device.h"
#include "amd_smi/impl/nic/amd_smi_nic_interface.h"
#include "nic_unit_fixture.h"
#include "test_fixture_utils.h"

namespace {

using amd::smi::AMDSmiAINICDevice;
using amd::smi::testing::ScopedProcessorRegistration;

constexpr auto kUnknownMtu = std::numeric_limits<uint16_t>::max();
constexpr auto kFilledMaxMtu = uint16_t{4096};
constexpr auto kFilledActiveMtu = uint16_t{1024};
constexpr auto kRdmaPorts = uint8_t{2};

int g_mtu_fill_calls = 0;
smi_nic_status_t g_mtu_fill_status = SMI_NIC_STATUS_SUCCESS;

smi_nic_status_t CountingMtuFill(smi_nic_ctx_t, uint64_t, smi_nic_rdma_devices_info_t* info) {
  g_mtu_fill_calls++;
  if (g_mtu_fill_status != SMI_NIC_STATUS_SUCCESS) {
    return g_mtu_fill_status;
  }
  for (auto k = uint8_t{0}; k < info->rdma_dev_info[0].num_rdma_ports; k++) {
    info->rdma_dev_info[0].rdma_port_info[k].max_mtu = kFilledMaxMtu;
    info->rdma_dev_info[0].rdma_port_info[k].active_mtu = kFilledActiveMtu;
  }
  return SMI_NIC_STATUS_SUCCESS;
}

amd::smi::nic_info_getters_t g_getters = {};

// Installs the counting MTU fill and returns a device whose ports still hold the unknown MTU.
std::unique_ptr<AMDSmiAINICDevice::AINICInfo> PrepareRdmaDeviceInfo() {
  g_mtu_fill_calls = 0;
  g_mtu_fill_status = SMI_NIC_STATUS_SUCCESS;
  g_getters = kStubGetters;
  g_getters.rdma_mtu = CountingMtuFill;
  amd::smi::nic_set_info_getters_for_testing(&g_getters);

  auto info = std::make_unique<AMDSmiAINICDevice::AINICInfo>();
  info->rdma_dev.num_rdma_dev = 1;
  info->rdma_dev.rdma_dev_info[0].num_rdma_ports = kRdmaPorts;
  for (auto k = uint8_t{0}; k < kRdmaPorts; k++) {
    info->rdma_dev.rdma_dev_info[0].rdma_port_info[k].max_mtu = kUnknownMtu;
    info->rdma_dev.rdma_dev_info[0].rdma_port_info[k].active_mtu = kUnknownMtu;
  }
  return info;
}

void ExpectFilled(const amdsmi_nic_rdma_devices_info_t& rdma_dev) {
  for (auto k = uint8_t{0}; k < kRdmaPorts; k++) {
    EXPECT_EQ(rdma_dev.rdma_dev_info[0].rdma_port_info[k].max_mtu, kFilledMaxMtu);
    EXPECT_EQ(rdma_dev.rdma_dev_info[0].rdma_port_info[k].active_mtu, kFilledActiveMtu);
  }
}

// The shared smi_get_nic_rdma_dev_info probes the MTU; amdsmi_init must use the variant that does
// not.
TEST_F(NicUnit, RdmaMtuProductionDiscoveryUsesTheGetterWithoutTheMtu) {
  const auto& getters = amd::smi::nic_production_info_getters_for_testing();

  EXPECT_EQ(getters.rdma, smi_get_nic_rdma_dev_info_without_mtu);
  EXPECT_EQ(getters.rdma_mtu, smi_get_nic_rdma_mtu);
}

bool g_was_deferred_during_create = false;
smi_nic_status_t g_create_status = SMI_NIC_STATUS_SUCCESS;

smi_nic_status_t RecordingCreateContext(smi_nic_ctx_t*) {
  g_was_deferred_during_create = smi_nic_is_rdma_mtu_deferred();
  return g_create_status;
}

// Runs the real init path with context creation replaced by a recorder.
void InitWithRecordedCreate(smi_nic_status_t create_status, bool was_deferred_before) {
  g_getters = kStubGetters;
  g_getters.create_context = RecordingCreateContext;
  amd::smi::nic_set_info_getters_for_testing(&g_getters);
  g_create_status = create_status;
  g_was_deferred_during_create = false;
  smi_nic_defer_rdma_mtu(was_deferred_before);

  EXPECT_NE(amdsmi_init(AMDSMI_INIT_AMD_NICS), AMDSMI_STATUS_SUCCESS);
}

TEST_F(NicUnit, RdmaMtuInitDefersTheProbeOnlyWhileTheContextIsCreated) {
  InitWithRecordedCreate(SMI_NIC_STATUS_ERROR, false);

  EXPECT_TRUE(g_was_deferred_during_create);
  EXPECT_FALSE(smi_nic_is_rdma_mtu_deferred());
}

TEST_F(NicUnit, RdmaMtuInitRestoresAPreviousDeferralWhenCreationSucceedsAndDiscoveryFails) {
  InitWithRecordedCreate(SMI_NIC_STATUS_SUCCESS, true);

  EXPECT_TRUE(g_was_deferred_during_create);
  EXPECT_TRUE(smi_nic_is_rdma_mtu_deferred());
  smi_nic_defer_rdma_mtu(false);
}

TEST_F(NicUnit, RdmaMtuPopulateDoesNotFillTheMtu) {
  PrepareRdmaDeviceInfo();
  ASSERT_EQ(Populate(), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(g_mtu_fill_calls, 0);
}

TEST_F(NicUnit, RdmaMtuFirstRdmaInfoReadFillsOnceAndTheSecondReusesIt) {
  auto device_info = PrepareRdmaDeviceInfo();
  ASSERT_EQ(amdsmi_init(AMDSMI_INIT_AMD_NICS), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(g_mtu_fill_calls, 0);
  {
    AMDSmiAINICDevice device(*device_info);
    ScopedProcessorRegistration registration(&device);
    auto rdma_dev = std::make_unique<amdsmi_nic_rdma_devices_info_t>();

    ASSERT_EQ(amdsmi_get_nic_rdma_dev_info(registration.handle(), rdma_dev.get()),
              AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(g_mtu_fill_calls, 1);
    ExpectFilled(*rdma_dev);

    ASSERT_EQ(amdsmi_get_nic_rdma_dev_info(registration.handle(), rdma_dev.get()),
              AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(g_mtu_fill_calls, 1);
    ExpectFilled(*rdma_dev);
  }
  EXPECT_EQ(amdsmi_shut_down(), AMDSMI_STATUS_SUCCESS);
}

TEST_F(NicUnit, RdmaMtuFullNicInfoReadFillsTheMtuToo) {
  auto device_info = PrepareRdmaDeviceInfo();
  ASSERT_EQ(amdsmi_init(AMDSMI_INIT_AMD_NICS), AMDSMI_STATUS_SUCCESS);
  {
    AMDSmiAINICDevice device(*device_info);
    ScopedProcessorRegistration registration(&device);
    auto info = std::make_unique<AMDSmiAINICDevice::AINICInfo>();

    ASSERT_EQ(amdsmi_get_ainic_info(registration.handle(), info.get()), AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(g_mtu_fill_calls, 1);
    ExpectFilled(info->rdma_dev);

    auto rdma_dev = std::make_unique<amdsmi_nic_rdma_devices_info_t>();
    ASSERT_EQ(amdsmi_get_nic_rdma_dev_info(registration.handle(), rdma_dev.get()),
              AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(g_mtu_fill_calls, 1);
  }
  EXPECT_EQ(amdsmi_shut_down(), AMDSMI_STATUS_SUCCESS);
}

TEST_F(NicUnit, RdmaMtuRdmaStatisticsDoNotProbeTheMtu) {
  auto device_info = PrepareRdmaDeviceInfo();
  ASSERT_EQ(amdsmi_init(AMDSMI_INIT_AMD_NICS), AMDSMI_STATUS_SUCCESS);
  {
    AMDSmiAINICDevice device(*device_info);
    ScopedProcessorRegistration registration(&device);
    auto num_stats = uint32_t{0};

    (void)amdsmi_get_nic_rdma_port_statistics(registration.handle(), 0, &num_stats, nullptr);

    EXPECT_EQ(g_mtu_fill_calls, 0);
  }
  EXPECT_EQ(amdsmi_shut_down(), AMDSMI_STATUS_SUCCESS);
}

TEST_F(NicUnit, RdmaMtuAFailedFillKeepsTheMtuUnknownAndIsRetried) {
  auto device_info = PrepareRdmaDeviceInfo();
  g_mtu_fill_status = SMI_NIC_STATUS_ERROR;
  ASSERT_EQ(amdsmi_init(AMDSMI_INIT_AMD_NICS), AMDSMI_STATUS_SUCCESS);
  {
    AMDSmiAINICDevice device(*device_info);
    ScopedProcessorRegistration registration(&device);
    auto rdma_dev = std::make_unique<amdsmi_nic_rdma_devices_info_t>();

    ASSERT_EQ(amdsmi_get_nic_rdma_dev_info(registration.handle(), rdma_dev.get()),
              AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(rdma_dev->rdma_dev_info[0].rdma_port_info[0].max_mtu, kUnknownMtu);
    EXPECT_EQ(rdma_dev->rdma_dev_info[0].rdma_port_info[0].active_mtu, kUnknownMtu);

    g_mtu_fill_status = SMI_NIC_STATUS_SUCCESS;
    ASSERT_EQ(amdsmi_get_nic_rdma_dev_info(registration.handle(), rdma_dev.get()),
              AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(g_mtu_fill_calls, 2);
    ExpectFilled(*rdma_dev);
  }
  EXPECT_EQ(amdsmi_shut_down(), AMDSMI_STATUS_SUCCESS);
}

TEST_F(NicUnit, RdmaMtuANicWithoutRdmaDevicesNeverFills) {
  auto device_info = PrepareRdmaDeviceInfo();
  device_info->rdma_dev.num_rdma_dev = 0;
  ASSERT_EQ(amdsmi_init(AMDSMI_INIT_AMD_NICS), AMDSMI_STATUS_SUCCESS);
  {
    AMDSmiAINICDevice device(*device_info);
    ScopedProcessorRegistration registration(&device);
    auto rdma_dev = std::make_unique<amdsmi_nic_rdma_devices_info_t>();

    ASSERT_EQ(amdsmi_get_nic_rdma_dev_info(registration.handle(), rdma_dev.get()),
              AMDSMI_STATUS_SUCCESS);

    EXPECT_EQ(g_mtu_fill_calls, 0);
  }
  EXPECT_EQ(amdsmi_shut_down(), AMDSMI_STATUS_SUCCESS);
}

}  // namespace
