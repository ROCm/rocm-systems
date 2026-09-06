// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstring>
#include <string>
#include <vector>

#include "api_test_framework.h"

using amdsmi::test::kInvalidHandle;
using amdsmi::test::kVerbose;

// =====================================================================
// amdsmi_get_nic_processor_handles(socket, uint32_t* count, handles*)
// =====================================================================

TEST_F(NicIntegration, GetNicProcessorHandles_NullCount) {
  if (sockets().empty()) GTEST_SKIP() << "No sockets";
  DISPLAY_AMDSMI_API("amdsmi_get_nic_processor_handles", "socket=0 count=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_nic_processor_handles(sockets()[0], nullptr, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}

TEST_F(NicIntegration, GetNicProcessorHandles_InvalidHandle) {
  uint32_t count = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_nic_processor_handles", "socket=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_nic_processor_handles(nullptr, &count, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}

TEST_F(NicIntegration, GetNicProcessorHandles_AllSockets) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_nic_processor_handles");
  if (sockets().empty()) GTEST_SKIP() << "No sockets";
  for (size_t i = 0; i < sockets().size(); ++i) {
    uint32_t count = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_nic_processor_handles",
                       "socket=" + std::to_string(i) + " count-query", kVerbose);
    amdsmi_status_t err = amdsmi_get_nic_processor_handles(sockets()[i], &count, nullptr);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("socket=" + std::to_string(i) + " count-query", err);
    if (err != AMDSMI_STATUS_SUCCESS || count == 0) continue;
    std::vector<amdsmi_processor_handle> handles(count);
    DISPLAY_AMDSMI_API("amdsmi_get_nic_processor_handles", "socket=" + std::to_string(i) + " fill",
                       kVerbose);
    err = amdsmi_get_nic_processor_handles(sockets()[i], &count, handles.data());
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("socket=" + std::to_string(i) + " fill", err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// =====================================================================
// amdsmi_get_nic_device_bdf(handle, amdsmi_bdf_t* bdf)
// =====================================================================

TEST_F(NicIntegration, GetNicDeviceBdf_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_nic_device_bdf", "nic=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_nic_device_bdf(any_nic(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}

TEST_F(NicIntegration, GetNicDeviceBdf_InvalidHandle) {
  amdsmi_bdf_t bdf;
  memset(&bdf, 0, sizeof(bdf));
  DISPLAY_AMDSMI_API("amdsmi_get_nic_device_bdf", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_nic_device_bdf(kInvalidHandle, &bdf);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}

TEST_F(NicIntegration, GetNicDeviceBdf_AllNics) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_nic_device_bdf");
  if (nics().empty()) GTEST_SKIP() << "No NIC devices";
  for (size_t i = 0; i < nics().size(); ++i) {
    amdsmi_bdf_t bdf;
    memset(&bdf, 0, sizeof(bdf));
    DISPLAY_AMDSMI_API("amdsmi_get_nic_device_bdf", "nic=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_nic_device_bdf(nics()[i], &bdf);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("nic=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// =====================================================================
// amdsmi_get_nic_driver_info(handle, amdsmi_nic_driver_info_t* info)
// =====================================================================

AMDSMI_INTEGRATION_NIC_STRUCT_GETTER(GetNicDriverInfo, amdsmi_get_nic_driver_info,
                                     amdsmi_nic_driver_info_t);

// =====================================================================
// amdsmi_get_nic_asic_info(handle, amdsmi_nic_asic_info_t* info)
// =====================================================================

AMDSMI_INTEGRATION_NIC_STRUCT_GETTER(GetNicAsicInfo, amdsmi_get_nic_asic_info,
                                     amdsmi_nic_asic_info_t);

// =====================================================================
// amdsmi_get_nic_bus_info(handle, amdsmi_nic_bus_info_t* info)
// =====================================================================

AMDSMI_INTEGRATION_NIC_STRUCT_GETTER(GetNicBusInfo, amdsmi_get_nic_bus_info, amdsmi_nic_bus_info_t);

// =====================================================================
// amdsmi_get_nic_numa_info(handle, amdsmi_nic_numa_info_t* info)
// =====================================================================

AMDSMI_INTEGRATION_NIC_STRUCT_GETTER(GetNicNumaInfo, amdsmi_get_nic_numa_info,
                                     amdsmi_nic_numa_info_t);

// =====================================================================
// amdsmi_get_nic_port_info(handle, amdsmi_nic_port_info_t* info)
// =====================================================================

AMDSMI_INTEGRATION_NIC_STRUCT_GETTER(GetNicPortInfo, amdsmi_get_nic_port_info,
                                     amdsmi_nic_port_info_t);

// =====================================================================
// amdsmi_get_nic_rdma_dev_info(handle, amdsmi_nic_rdma_devices_info_t* info)
// =====================================================================

AMDSMI_INTEGRATION_NIC_STRUCT_GETTER(GetNicRdmaDevInfo, amdsmi_get_nic_rdma_dev_info,
                                     amdsmi_nic_rdma_devices_info_t);

// =====================================================================
// amdsmi_get_nic_fw_info(handle, amdsmi_nic_fw_info_t* info)
// =====================================================================

AMDSMI_INTEGRATION_NIC_STRUCT_GETTER(GetNicFwInfo, amdsmi_get_nic_fw_info, amdsmi_nic_fw_info_t);

TEST_F(NicIntegration, DeviceBdf_InvalidHandle) {
  RequireInit();
  amdsmi_bdf_t bdf;
  memset(&bdf, 0, sizeof(bdf));
  DISPLAY_AMDSMI_API("amdsmi_get_nic_device_bdf", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_nic_device_bdf(kInvalidHandle, &bdf);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

TEST_F(NicIntegration, AsicInfo_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_nic_asic_info", "nic=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_nic_asic_info(any_nic(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}

TEST_F(NicIntegration, DeviceBdf_Stable) {
  if (nics().empty()) GTEST_SKIP() << "No NIC devices";
  amdsmi::test::StatusCollector col("amdsmi_get_nic_device_bdf");
  for (size_t i = 0; i < nics().size(); ++i) {
    amdsmi_bdf_t a, b;
    memset(&a, 0, sizeof(a));
    memset(&b, 0, sizeof(b));
    DISPLAY_AMDSMI_API("amdsmi_get_nic_device_bdf", "nic=" + std::to_string(i), kVerbose);
    amdsmi_status_t e1 = amdsmi_get_nic_device_bdf(nics()[i], &a);
    amdsmi_status_t e2 = amdsmi_get_nic_device_bdf(nics()[i], &b);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, e1, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    col.Record("nic=" + std::to_string(i), e1,
               ::amdsmi::test::AmdsmiStatusIsExpected(e1, AMDSMI_STATUS_SUCCESS,
                                                      AMDSMI_STATUS_NOT_SUPPORTED,
                                                      AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
    EXPECT_EQ(e1, e2) << "nic=" << i << " getter status not stable";
    if (e1 == AMDSMI_STATUS_SUCCESS) {
      EXPECT_EQ(a.as_uint, b.as_uint) << "nic=" << i << " BDF not stable across reads";
    }
  }
  col.ExpectNoFailures();
}
