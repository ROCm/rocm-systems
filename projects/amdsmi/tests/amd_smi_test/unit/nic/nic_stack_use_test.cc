// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * The AINICInfo of a NIC is over 600 KB. A getter that needs one field of it must not copy the
 * whole struct onto the stack: a caller on a thread with a small stack would overflow it.
 */

#include <gtest/gtest.h>
#include <pthread.h>

#include <cstddef>
#include <cstdint>
#include <memory>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/nic/amd_smi_ainic_device.h"
#include "nic_unit_fixture.h"
#include "test_fixture_utils.h"

namespace {

using amd::smi::AMDSmiAINICDevice;
using amd::smi::testing::ScopedProcessorRegistration;

constexpr auto kSmallStackBytes = size_t{256} * 1024;

// The getters whose only use of the NIC info is its BDF. The status is not the point: reaching
// the end without overflowing the stack is.
void* call_bdf_only_getters(void* arg) {
  const auto handle = static_cast<amdsmi_processor_handle>(arg);
  auto fw_info = amdsmi_nic_fw_info_t{};
  auto telemetry = amdsmi_nic_telemetry_t{};
  auto num_sensors = uint32_t{0};
  auto num_stats = uint32_t{0};

  (void)amdsmi_get_nic_fw_info(handle, &fw_info);
  (void)amdsmi_get_nic_telemetry(handle, &telemetry);
  (void)amdsmi_get_nic_hwmon_sensors(handle, &num_sensors, nullptr);
  (void)amdsmi_get_nic_port_statistics(handle, 0, &num_stats, nullptr);
  return nullptr;
}

// The getters that return one sub-struct of the NIC info. The status is not the point here either.
// The output structs are large themselves, so they live on the heap to leave the stack to the
// getters.
void* call_sub_struct_getters(void* arg) {
  const auto handle = static_cast<amdsmi_processor_handle>(arg);
  auto asic = std::make_unique<amdsmi_nic_asic_info_t>();
  auto bus = std::make_unique<amdsmi_nic_bus_info_t>();
  auto driver = std::make_unique<amdsmi_nic_driver_info_t>();
  auto numa = std::make_unique<amdsmi_nic_numa_info_t>();
  auto port = std::make_unique<amdsmi_nic_port_info_t>();
  auto rdma_dev = std::make_unique<amdsmi_nic_rdma_devices_info_t>();
  auto num_stats = uint32_t{0};

  (void)amdsmi_get_nic_asic_info(handle, asic.get());
  (void)amdsmi_get_nic_bus_info(handle, bus.get());
  (void)amdsmi_get_nic_driver_info(handle, driver.get());
  (void)amdsmi_get_nic_numa_info(handle, numa.get());
  (void)amdsmi_get_nic_port_info(handle, port.get());
  (void)amdsmi_get_nic_rdma_dev_info(handle, rdma_dev.get());
  (void)amdsmi_get_nic_rdma_port_statistics(handle, 0, &num_stats, nullptr);
  return nullptr;
}

void run_on_small_stack(void* (*routine)(void*), amdsmi_processor_handle handle) {
  pthread_attr_t attr;
  ASSERT_EQ(pthread_attr_init(&attr), 0);
  ASSERT_EQ(pthread_attr_setstacksize(&attr, kSmallStackBytes), 0);
  pthread_t thread;
  ASSERT_EQ(pthread_create(&thread, &attr, routine, handle), 0);
  ASSERT_EQ(pthread_join(thread, nullptr), 0);
  pthread_attr_destroy(&attr);
}

TEST_F(NicUnit, SubStructGettersRunOnASmallStack) {
  ASSERT_EQ(amdsmi_init(AMDSMI_INIT_AMD_NICS), AMDSMI_STATUS_SUCCESS);
  {
    auto info = std::make_unique<AMDSmiAINICDevice::AINICInfo>();
    AMDSmiAINICDevice device(*info);
    ScopedProcessorRegistration registration(&device);

    run_on_small_stack(call_sub_struct_getters, registration.handle());
  }
  EXPECT_EQ(amdsmi_shut_down(), AMDSMI_STATUS_SUCCESS);
}

TEST_F(NicUnit, BdfOnlyGettersRunOnASmallStack) {
  ASSERT_EQ(amdsmi_init(AMDSMI_INIT_AMD_NICS), AMDSMI_STATUS_SUCCESS);
  {
    auto info = std::make_unique<AMDSmiAINICDevice::AINICInfo>();
    AMDSmiAINICDevice device(*info);
    ScopedProcessorRegistration registration(&device);

    pthread_attr_t attr;
    ASSERT_EQ(pthread_attr_init(&attr), 0);
    ASSERT_EQ(pthread_attr_setstacksize(&attr, kSmallStackBytes), 0);
    pthread_t thread;
    ASSERT_EQ(pthread_create(&thread, &attr, call_bdf_only_getters, registration.handle()), 0);
    ASSERT_EQ(pthread_join(thread, nullptr), 0);
    pthread_attr_destroy(&attr);
  }
  EXPECT_EQ(amdsmi_shut_down(), AMDSMI_STATUS_SUCCESS);
}

}  // namespace
