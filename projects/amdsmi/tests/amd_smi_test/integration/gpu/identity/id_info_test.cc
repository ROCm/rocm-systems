/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <cstring>
#include <string>

#include "api_test_framework.h"

using amdsmi::test::kInvalidHandle;
using amdsmi::test::kVerbose;

// ---------------- amdsmi_get_gpu_device_bdf ----------------
TEST_F(GpuIntegration, GetDeviceBdf_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_device_bdf", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_device_bdf(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetDeviceBdf_InvalidHandle) {
  amdsmi_bdf_t bdf;
  memset(&bdf, 0, sizeof(bdf));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_device_bdf", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_device_bdf(kInvalidHandle, &bdf);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetDeviceBdf_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_device_bdf");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_bdf_t bdf;
    memset(&bdf, 0, sizeof(bdf));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_device_bdf", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_device_bdf(gpus()[i], &bdf);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_device_uuid ----------------
TEST_F(GpuIntegration, GetDeviceUuid_NullOutput) {
  unsigned int len = 256;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_device_uuid", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_device_uuid(any_gpu(), &len, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetDeviceUuid_InvalidHandle) {
  unsigned int len = 256;
  char uuid[256];
  memset(uuid, 0, sizeof(uuid));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_device_uuid", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_device_uuid(kInvalidHandle, &len, uuid);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetDeviceUuid_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_device_uuid");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    unsigned int len = 256;
    char uuid[256];
    memset(uuid, 0, sizeof(uuid));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_device_uuid", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_device_uuid(gpus()[i], &len, uuid);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_device_cuid ----------------
TEST_F(GpuIntegration, GetDeviceCuid_NullOutput) {
  unsigned int len = 256;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_device_cuid", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_device_cuid(any_gpu(), &len, nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetDeviceCuid_InvalidHandle) {
  unsigned int len = 256;
  char cuid[256];
  memset(cuid, 0, sizeof(cuid));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_device_cuid", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_device_cuid(kInvalidHandle, &len, cuid);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetDeviceCuid_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_device_cuid");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    unsigned int len = 256;
    char cuid[256];
    memset(cuid, 0, sizeof(cuid));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_device_cuid", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_device_cuid(gpus()[i], &len, cuid);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_enumeration_info ----------------
AMDSMI_INTEGRATION_GPU_STRUCT_GETTER(GetEnumerationInfo, amdsmi_get_gpu_enumeration_info,
                                     amdsmi_enumeration_info_t)

// ---------------- amdsmi_get_gpu_virtualization_mode ----------------
TEST_F(GpuIntegration, GetVirtualizationMode_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_virtualization_mode", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_virtualization_mode(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetVirtualizationMode_InvalidHandle) {
  amdsmi_virtualization_mode_t mode;
  memset(&mode, 0, sizeof(mode));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_virtualization_mode", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_virtualization_mode(kInvalidHandle, &mode);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetVirtualizationMode_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_virtualization_mode");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_virtualization_mode_t mode;
    memset(&mode, 0, sizeof(mode));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_virtualization_mode", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_virtualization_mode(gpus()[i], &mode);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_id ----------------
TEST_F(GpuIntegration, GetGpuId_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_id", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_id(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetGpuId_InvalidHandle) {
  uint16_t id = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_id", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_id(kInvalidHandle, &id);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetGpuId_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_id");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    uint16_t id = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_id", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_id(gpus()[i], &id);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_revision ----------------
TEST_F(GpuIntegration, GetRevision_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_revision", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_revision(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetRevision_InvalidHandle) {
  uint16_t rev = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_revision", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_revision(kInvalidHandle, &rev);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetRevision_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_revision");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    uint16_t rev = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_revision", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_revision(gpus()[i], &rev);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_vendor_name ----------------
TEST_F(GpuIntegration, GetVendorName_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_vendor_name", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_vendor_name(any_gpu(), nullptr, 256);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetVendorName_InvalidHandle) {
  char name[256];
  memset(name, 0, sizeof(name));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_vendor_name", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_vendor_name(kInvalidHandle, name, sizeof(name));
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetVendorName_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_vendor_name");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    char name[256];
    memset(name, 0, sizeof(name));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_vendor_name", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_vendor_name(gpus()[i], name, sizeof(name));
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_vram_vendor ----------------
TEST_F(GpuIntegration, GetVramVendor_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_vram_vendor", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_vram_vendor(any_gpu(), nullptr, 256);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetVramVendor_InvalidHandle) {
  char brand[256];
  memset(brand, 0, sizeof(brand));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_vram_vendor", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_vram_vendor(kInvalidHandle, brand, sizeof(brand));
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetVramVendor_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_vram_vendor");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    char brand[256];
    memset(brand, 0, sizeof(brand));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_vram_vendor", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_vram_vendor(gpus()[i], brand, sizeof(brand));
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_subsystem_id ----------------
TEST_F(GpuIntegration, GetSubsystemId_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_subsystem_id", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_subsystem_id(any_gpu(), nullptr);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetSubsystemId_InvalidHandle) {
  uint16_t id = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_subsystem_id", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_subsystem_id(kInvalidHandle, &id);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetSubsystemId_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_subsystem_id");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    uint16_t id = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_subsystem_id", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_subsystem_id(gpus()[i], &id);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_subsystem_name ----------------
TEST_F(GpuIntegration, GetSubsystemName_NullOutput) {
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_subsystem_name", "gpu=0 out=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_subsystem_name(any_gpu(), nullptr, 256);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetSubsystemName_InvalidHandle) {
  char name[256];
  memset(name, 0, sizeof(name));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_subsystem_name", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_subsystem_name(kInvalidHandle, name, sizeof(name));
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetSubsystemName_AllGpus) {
  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_subsystem_name");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    char name[256];
    memset(name, 0, sizeof(name));
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_subsystem_name", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_subsystem_name(gpus()[i], name, sizeof(name));
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}

// ---------------- amdsmi_get_gpu_xcd_counter ----------------
TEST_F(GpuIntegration, GetXcdCounter_NullOutput) {
  GTEST_SKIP() << "amdsmi_get_gpu_xcd_counter crashes on a null output pointer; proper return "
                  "should be AMDSMI_STATUS_INVAL";
  // Proper contract once fixed:
  //   amdsmi_status_t err = amdsmi_get_gpu_xcd_counter(gpus()[0], nullptr);
  //   AMDSMI_EXPECT_NULL_ARG(err);
}
TEST_F(GpuIntegration, GetXcdCounter_InvalidHandle) {
  uint16_t xcd = 0;
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_xcd_counter", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_xcd_counter(kInvalidHandle, &xcd);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  AMDSMI_EXPECT_INVALID_HANDLE(err);
}
TEST_F(GpuIntegration, GetXcdCounter_AllGpus) {
  GTEST_SKIP() << "amdsmi_get_gpu_xcd_counter returns AMDSMI_STATUS_UNEXPECTED_DATA; root cause "
                  "unknown, under investigation";

  amdsmi::test::StatusCollector amdsmi_col("amdsmi_get_gpu_xcd_counter");
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  for (size_t i = 0; i < gpus().size(); ++i) {
    uint16_t xcd = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_xcd_counter", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_get_gpu_xcd_counter(gpus()[i], &xcd);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    amdsmi_col.RecordPositive("gpu=" + std::to_string(i), err);
  }
  AMDSMI_FINISH_POSITIVE(amdsmi_col);
}
