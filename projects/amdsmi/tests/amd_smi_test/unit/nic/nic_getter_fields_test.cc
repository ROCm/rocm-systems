// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * Each single-field NIC getter returns its own sub-struct of the NIC info and keeps the status
 * codes of amdsmi_get_ainic_info() for a bad handle, a bad pointer and an uninitialised library.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <memory>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_processor.h"
#include "amd_smi/impl/nic/amd_smi_ainic_device.h"
#include "nic_unit_fixture.h"
#include "test_fixture_utils.h"

namespace {

using amd::smi::AMDSmiAINICDevice;
using amd::smi::testing::ScopedProcessorRegistration;

// A fill byte unique to each sub-struct, so a copy of the wrong one is caught.
constexpr auto kAsicFill = 0x11;
constexpr auto kBusFill = 0x22;
constexpr auto kDriverFill = 0x33;
constexpr auto kNumaFill = 0x44;
constexpr auto kPortFill = 0x55;
constexpr auto kRdmaFill = 0x66;

auto make_filled_info() -> std::unique_ptr<AMDSmiAINICDevice::AINICInfo> {
  auto info = std::make_unique<AMDSmiAINICDevice::AINICInfo>();
  std::memset(&info->asic, kAsicFill, sizeof(info->asic));
  std::memset(&info->bus, kBusFill, sizeof(info->bus));
  std::memset(&info->driver, kDriverFill, sizeof(info->driver));
  std::memset(&info->numa, kNumaFill, sizeof(info->numa));
  std::memset(&info->port, kPortFill, sizeof(info->port));
  std::memset(&info->rdma_dev, kRdmaFill, sizeof(info->rdma_dev));
  return info;
}

template <typename Out_t>
auto is_equal_bytes(const Out_t& lhs, const Out_t& rhs) -> bool {
  return (std::memcmp(&lhs, &rhs, sizeof(Out_t)) == 0);
}

// Runs one getter against a registered NIC: value equality, then the error statuses.
template <typename Out_t, typename Getter_t>
void check_getter(Getter_t getter, const Out_t& expected,
                  const AMDSmiAINICDevice::AINICInfo& info) {
  auto out = std::make_unique<Out_t>();
  std::memset(out.get(), 0, sizeof(Out_t));

  EXPECT_EQ(getter(nullptr, out.get()), AMDSMI_STATUS_NOT_INIT);
  ASSERT_EQ(amdsmi_init(AMDSMI_INIT_AMD_NICS), AMDSMI_STATUS_SUCCESS);
  {
    AMDSmiAINICDevice device(info);
    ScopedProcessorRegistration registration(&device);

    EXPECT_EQ(getter(registration.handle(), out.get()), AMDSMI_STATUS_SUCCESS);
    EXPECT_TRUE(is_equal_bytes(*out, expected));
    EXPECT_EQ(getter(registration.handle(), nullptr), AMDSMI_STATUS_INVAL);
    EXPECT_EQ(getter(nullptr, out.get()), AMDSMI_STATUS_INVAL);
  }
  {
    amd::smi::AMDSmiProcessor gpu(AMDSMI_PROCESSOR_TYPE_AMD_GPU);
    ScopedProcessorRegistration registration(&gpu);
    EXPECT_EQ(getter(registration.handle(), out.get()), AMDSMI_STATUS_NOT_SUPPORTED);
  }
  EXPECT_EQ(amdsmi_shut_down(), AMDSMI_STATUS_SUCCESS);
}

TEST_F(NicUnit, AsicInfoIsTheAsicField) {
  const auto info = make_filled_info();
  check_getter(amdsmi_get_nic_asic_info, info->asic, *info);
}

TEST_F(NicUnit, BusInfoIsTheBusField) {
  const auto info = make_filled_info();
  check_getter(amdsmi_get_nic_bus_info, info->bus, *info);
}

TEST_F(NicUnit, DriverInfoIsTheDriverField) {
  const auto info = make_filled_info();
  check_getter(amdsmi_get_nic_driver_info, info->driver, *info);
}

TEST_F(NicUnit, NumaInfoIsTheNumaField) {
  const auto info = make_filled_info();
  check_getter(amdsmi_get_nic_numa_info, info->numa, *info);
}

TEST_F(NicUnit, PortInfoIsThePortField) {
  const auto info = make_filled_info();
  check_getter(amdsmi_get_nic_port_info, info->port, *info);
}

TEST_F(NicUnit, RdmaDevInfoIsTheRdmaField) {
  const auto info = make_filled_info();
  check_getter(amdsmi_get_nic_rdma_dev_info, info->rdma_dev, *info);
}

TEST_F(NicUnit, RdmaPortStatisticsKeepItsStatusCodes) {
  auto info = make_filled_info();
  auto num_stats = uint32_t{0};

  EXPECT_EQ(amdsmi_get_nic_rdma_port_statistics(nullptr, 0, &num_stats, nullptr),
            AMDSMI_STATUS_NOT_INIT);
  ASSERT_EQ(amdsmi_init(AMDSMI_INIT_AMD_NICS), AMDSMI_STATUS_SUCCESS);
  {
    info->rdma_dev = {};
    AMDSmiAINICDevice device(*info);
    ScopedProcessorRegistration registration(&device);

    EXPECT_EQ(amdsmi_get_nic_rdma_port_statistics(registration.handle(), 0, nullptr, nullptr),
              AMDSMI_STATUS_INVAL);
    EXPECT_EQ(amdsmi_get_nic_rdma_port_statistics(nullptr, 0, &num_stats, nullptr),
              AMDSMI_STATUS_INVAL);
    EXPECT_EQ(amdsmi_get_nic_rdma_port_statistics(registration.handle(), 0, &num_stats, nullptr),
              AMDSMI_STATUS_NOT_SUPPORTED);
  }
  {
    amd::smi::AMDSmiProcessor gpu(AMDSMI_PROCESSOR_TYPE_AMD_GPU);
    ScopedProcessorRegistration registration(&gpu);
    EXPECT_EQ(amdsmi_get_nic_rdma_port_statistics(registration.handle(), 0, &num_stats, nullptr),
              AMDSMI_STATUS_NOT_SUPPORTED);
  }
  EXPECT_EQ(amdsmi_shut_down(), AMDSMI_STATUS_SUCCESS);
}

}  // namespace
