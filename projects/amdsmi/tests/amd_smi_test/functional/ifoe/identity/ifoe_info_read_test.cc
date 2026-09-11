// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "ifoe_info_read.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <string>

#include "amd_smi/amdsmi.h"
#include "test_common.h"

TestIfoeInfoRead::TestIfoeInfoRead() : TestBase() {
  set_title("AMDSMI IFOE Info Read Test");
  set_description(
      "This test verifies that IFOE information such as "
      "device, fabric telemetry, and fabric info can be read properly.");
}

TestIfoeInfoRead::~TestIfoeInfoRead(void) {}

void TestIfoeInfoRead::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestIfoeInfoRead::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestIfoeInfoRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestIfoeInfoRead::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

static const uint32_t kBufferLen = 80;

static const std::map<amdsmi_virtualization_mode_t, std::string> virtualization_mode_map = {
    {AMDSMI_VIRTUALIZATION_MODE_UNKNOWN, "UNKNOWN"},
    {AMDSMI_VIRTUALIZATION_MODE_BAREMETAL, "BAREMETAL"},
    {AMDSMI_VIRTUALIZATION_MODE_HOST, "HOST"},
    {AMDSMI_VIRTUALIZATION_MODE_GUEST, "GUEST"},
    {AMDSMI_VIRTUALIZATION_MODE_PASSTHROUGH, "PASSTHROUGH"}};

void TestIfoeInfoRead::Run(void) {
  amdsmi_status_t err;
  uint16_t id;
  uint64_t val_ui64;

  char buffer[kBufferLen];

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
    IF_VERB(STANDARD) {
      std::cout << "\t*************************" << std::endl;
      std::cout << "\t**Device index: " << i << std::endl;
    }

    // Get the device ID, name, vendor ID and vendor name for the device
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_id", "gpu=" + std::to_string(i), VERB(STANDARD));
    err = amdsmi_get_gpu_id(processor_handles_[i], &id);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
    if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
      amdsmi_status_t ret;
      // Verify api support checking functionality is working
      DISPLAY_AMDSMI_API("amdsmi_get_gpu_id(nullptr check)", "gpu=" + std::to_string(i),
                         VERB(STANDARD));
      ret = amdsmi_get_gpu_id(processor_handles_[i], nullptr);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_NOT_SUPPORTED);
      ASSERT_EQ(ret, AMDSMI_STATUS_NOT_SUPPORTED);
    } else {
      CHK_ERR_ASRT(err)

      IF_VERB(STANDARD) { std::cout << "\t**Device ID: 0x" << std::hex << id << std::endl; }
      // Verify api support checking functionality is working
      DISPLAY_AMDSMI_API("amdsmi_get_gpu_id(nullptr check)", "gpu=" + std::to_string(i),
                         VERB(STANDARD));
      err = amdsmi_get_gpu_id(processor_handles_[i], nullptr);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
      ASSERT_EQ(err, AMDSMI_STATUS_INVAL);
    }

    // vendor_id, unique_id
    amdsmi_asic_info_t asic_info;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_asic_info", "gpu=0", VERB(STANDARD));
    err = amdsmi_get_gpu_asic_info(processor_handles_[0], &asic_info);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
    CHK_ERR_ASRT(err)

    // physical_acc_id is UALoE-backed; without an active session it stays at
    // the UINT32_MAX sentinel rather than failing amdsmi_get_gpu_asic_info().
    if (asic_info.physical_acc_id == std::numeric_limits<uint32_t>::max()) {
      IF_VERB(STANDARD) {
        std::cout << "\t**Physical Accelerator ID: N/A (not supported on this system)" << std::endl;
      }
    } else {
      IF_VERB(STANDARD) {
        std::cout << "\t**Physical Accelerator ID: " << asic_info.physical_acc_id << std::endl;
      }
    }

    // device name, brand, serial_number
    amdsmi_board_info_t board_info;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_board_info", "gpu=0", VERB(STANDARD));
    err = amdsmi_get_gpu_board_info(processor_handles_[0], &board_info);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
    CHK_ERR_ASRT(err)

    amdsmi_vram_info_t vram_info;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_vram_info", "gpu=" + std::to_string(i), VERB(STANDARD));
    err = amdsmi_get_gpu_vram_info(processor_handles_[i], &vram_info);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
    CHK_ERR_ASRT(err)
    IF_VERB(STANDARD) {
      std::cout << "\t**Device Vram type id: " << vram_info.vram_type << std::endl;
      std::cout << "\t**Device Vram vendor id: " << vram_info.vram_vendor << std::endl;
      std::cout << "\t**Device Vram size: 0x" << std::hex << vram_info.vram_size << " (" << std::dec
                << vram_info.vram_size << ")" << std::endl;
      std::cout << "\t**Device Bit Width: 0x" << std::hex << vram_info.vram_bit_width << " ("
                << std::dec << vram_info.vram_bit_width << ")" << std::endl;
      std::cout << "\t**Device Vram Max Bandwidth: " << vram_info.vram_max_bandwidth << " GB/s"
                << std::endl;
    }

    DISPLAY_AMDSMI_API("amdsmi_get_gpu_vendor_name", "gpu=" + std::to_string(i), VERB(STANDARD));
    err = amdsmi_get_gpu_vendor_name(processor_handles_[i], buffer, kBufferLen);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
    if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
      std::cout << "\t**Device Vendor name string not found on this system." << std::endl;
      // Verify api support checking functionality is working
      DISPLAY_AMDSMI_API("amdsmi_get_gpu_vendor_name(nullptr check)", "gpu=" + std::to_string(i),
                         VERB(STANDARD));
      err = amdsmi_get_gpu_vendor_name(processor_handles_[i], nullptr, kBufferLen);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_NOT_SUPPORTED);
      ASSERT_EQ(err, AMDSMI_STATUS_NOT_SUPPORTED);
    } else {
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) { std::cout << "\t**Device Vendor name: " << buffer << std::endl; }
      // Verify api support checking functionality is working
      DISPLAY_AMDSMI_API("amdsmi_get_gpu_vendor_name(nullptr check)", "gpu=" + std::to_string(i),
                         VERB(STANDARD));
      err = amdsmi_get_gpu_vendor_name(processor_handles_[i], nullptr, kBufferLen);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
      ASSERT_EQ(err, AMDSMI_STATUS_INVAL);
    }

    // Get the device ID, name, vendor ID and vendor name for the sub-device
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_subsystem_id", "gpu=" + std::to_string(i), VERB(STANDARD));
    err = amdsmi_get_gpu_subsystem_id(processor_handles_[i], &id);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
    if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
      // Verify api support checking functionality is working
      DISPLAY_AMDSMI_API("amdsmi_get_gpu_subsystem_id(nullptr check)", "gpu=" + std::to_string(i),
                         VERB(STANDARD));
      err = amdsmi_get_gpu_subsystem_id(processor_handles_[i], nullptr);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_NOT_SUPPORTED);
      ASSERT_EQ(err, AMDSMI_STATUS_NOT_SUPPORTED);
    } else {
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) { std::cout << "\t**Subsystem ID: 0x" << std::hex << id << std::endl; }
      // Verify api support checking functionality is working
      DISPLAY_AMDSMI_API("amdsmi_get_gpu_subsystem_id(nullptr check)", "gpu=" + std::to_string(i),
                         VERB(STANDARD));
      err = amdsmi_get_gpu_subsystem_id(processor_handles_[i], nullptr);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
      ASSERT_EQ(err, AMDSMI_STATUS_INVAL);
    }
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_subsystem_name", "gpu=" + std::to_string(i), VERB(STANDARD));
    err = amdsmi_get_gpu_subsystem_name(processor_handles_[i], buffer, kBufferLen);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
    if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
      std::cout << "\t**Subsystem name string not found on this system." << std::endl;
      // Verify api support checking functionality is working
      DISPLAY_AMDSMI_API("amdsmi_get_gpu_subsystem_name(nullptr check)", "gpu=" + std::to_string(i),
                         VERB(STANDARD));
      err = amdsmi_get_gpu_subsystem_name(processor_handles_[i], nullptr, kBufferLen);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_NOT_SUPPORTED);
      ASSERT_EQ(err, AMDSMI_STATUS_NOT_SUPPORTED);
    } else {
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) { std::cout << "\t**Subsystem name: " << buffer << std::endl; }
      // Verify api support checking functionality is working
      DISPLAY_AMDSMI_API("amdsmi_get_gpu_subsystem_name(nullptr check)", "gpu=" + std::to_string(i),
                         VERB(STANDARD));
      err = amdsmi_get_gpu_subsystem_name(processor_handles_[i], nullptr, kBufferLen);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
      ASSERT_EQ(err, AMDSMI_STATUS_INVAL);
    }

    IF_VERB(STANDARD) {
      std::cout << "\t**Sub-system Vendor ID: 0x" << std::hex << asic_info.subvendor_id
                << std::endl;
    }

    DISPLAY_AMDSMI_API("amdsmi_get_gpu_vendor_name", "gpu=" + std::to_string(i), VERB(STANDARD));
    err = amdsmi_get_gpu_vendor_name(processor_handles_[i], buffer, kBufferLen);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
    if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
      std::cout << "\t**Subsystem Vendor name string not found on this system." << std::endl;
      // Verify api support checking functionality is working
      DISPLAY_AMDSMI_API("amdsmi_get_gpu_vendor_name(nullptr check)", "gpu=" + std::to_string(i),
                         VERB(STANDARD));
      err = amdsmi_get_gpu_vendor_name(processor_handles_[i], nullptr, kBufferLen);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_NOT_SUPPORTED);
      ASSERT_EQ(err, AMDSMI_STATUS_NOT_SUPPORTED);
    } else {
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) { std::cout << "\t**Subsystem Vendor name: " << buffer << std::endl; }
      // Verify api support checking functionality is working
      DISPLAY_AMDSMI_API("amdsmi_get_gpu_vendor_name(nullptr check)", "gpu=" + std::to_string(i),
                         VERB(STANDARD));
      err = amdsmi_get_gpu_vendor_name(processor_handles_[i], nullptr, kBufferLen);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
      ASSERT_EQ(err, AMDSMI_STATUS_INVAL);
    }

    DISPLAY_AMDSMI_API("amdsmi_get_gpu_bdf_id", "gpu=" + std::to_string(i), VERB(STANDARD));
    err = amdsmi_get_gpu_bdf_id(processor_handles_[i], &val_ui64);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
    // Don't check for AMDSMI_STATUS_NOT_SUPPORTED since this should always be
    // supported. It is not based on a sysfs file.
    CHK_ERR_ASRT(err)
    IF_VERB(STANDARD) {
      std::cout << "\t**PCI ID (BDFID): 0x" << std::hex << val_ui64;
      std::cout << " (" << std::dec << val_ui64 << ")" << std::endl;
    }
    // Verify api support checking functionality is working
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_bdf_id(nullptr check)", "gpu=" + std::to_string(i),
                       VERB(STANDARD));
    err = amdsmi_get_gpu_bdf_id(processor_handles_[i], nullptr);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
    ASSERT_EQ(err, AMDSMI_STATUS_INVAL);

    // Verify api support checking functionality is working
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_virtualization_mode(nullptr check)",
                       "gpu=" + std::to_string(i), VERB(STANDARD));
    err = amdsmi_get_gpu_virtualization_mode(processor_handles_[i], nullptr);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
    ASSERT_EQ(err, AMDSMI_STATUS_INVAL);
    amdsmi_virtualization_mode_t vmode;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_virtualization_mode", "gpu=" + std::to_string(i),
                       VERB(STANDARD));
    err = amdsmi_get_gpu_virtualization_mode(processor_handles_[i], &vmode);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED);
    ASSERT_TRUE(err == AMDSMI_STATUS_SUCCESS || err == AMDSMI_STATUS_NOT_SUPPORTED);
    IF_VERB(STANDARD) {
      auto it = virtualization_mode_map.find(vmode);
      if (it != virtualization_mode_map.end()) {
        std::cout << "\t**Virtualization Mode: " << it->second << std::endl;
      } else {
        std::cout << "\t**Virtualization Mode: MAP TYPE UNKNOWN?" << std::endl;
      }
    }
  }
}
