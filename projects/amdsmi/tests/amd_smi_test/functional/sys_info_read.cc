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

#include <stdint.h>
#include <stddef.h>
#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <limits>

#include "amd_smi/amdsmi.h"
#include "sys_info_read.h"

TestSysInfoRead::TestSysInfoRead() : TestBase() {
  set_title("AMDSMI System Info Read Test");
  set_description("This test verifies that system information such as the "
             "BDFID, AMDSMI version, VBIOS version, "
             "vendor_id, unique_id, target_gfx_version, kfd_id, node_id, etc. "
             "can be read properly.");
}

TestSysInfoRead::~TestSysInfoRead(void) {
}

void TestSysInfoRead::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestSysInfoRead::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestSysInfoRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestSysInfoRead::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}


void TestSysInfoRead::Run(void) {
  amdsmi_status_t err;
  uint64_t val_ui64;
  int32_t val_i32;
  amdsmi_version_t ver = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, nullptr};

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
    PrintDeviceHeader(processor_handles_[i]);

    amdsmi_vbios_info_t vbios_info;
    err = amdsmi_get_gpu_vbios_info(processor_handles_[i], &vbios_info);

    if (err != AMDSMI_STATUS_SUCCESS) {
      if ((err == AMDSMI_STATUS_FILE_ERROR) || (err == AMDSMI_STATUS_NOT_SUPPORTED)) {
        IF_VERB(STANDARD) {
          std::cout << "\t**VBIOS read: Not supported on this machine"
                                                                << std::endl;
        }
        // Verify api support checking functionality is working
        err = amdsmi_get_gpu_vbios_info(processor_handles_[i], nullptr);
        ASSERT_EQ(err, AMDSMI_STATUS_NOT_SUPPORTED);
      } else {
        // Verify api support checking functionality is working
        err = amdsmi_get_gpu_vbios_info(processor_handles_[i], nullptr);
        ASSERT_EQ(err, AMDSMI_STATUS_INVAL);

        CHK_ERR_ASRT(err)
      }
    } else {
      IF_VERB(STANDARD) {
        std::cout << "\t**VBIOS Version: "
                << vbios_info.version << std::endl;
      }
    }

    err = amdsmi_get_gpu_bdf_id(processor_handles_[i], &val_ui64);
    CHK_ERR_ASRT(err)
    IF_VERB(STANDARD) {
      std::cout << "\t**PCI ID (BDFID): 0x" << std::hex << val_ui64;
      std::cout << " (" << std::dec << val_ui64 << ")" << std::endl;
    }
    // Verify api support checking functionality is working
    err = amdsmi_get_gpu_bdf_id(processor_handles_[i], nullptr);
    ASSERT_EQ(err, AMDSMI_STATUS_INVAL);

    err = amdsmi_get_gpu_topo_numa_affinity(processor_handles_[i], &val_i32);
    if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
      std::cout << "\t**amdsmi_get_gpu_topo_numa_affinity(): Not supported on this machine"
                << std::endl;
      ASSERT_EQ(err, AMDSMI_STATUS_NOT_SUPPORTED);
    } else  {
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\t**NUMA NODE: 0x" << std::hex << val_i32;
        std::cout << " (" << std::dec << val_i32 << ")" << std::endl;
      }
    }

    // Verify api support checking functionality is working
    err = amdsmi_get_gpu_topo_numa_affinity(processor_handles_[i], nullptr);
    ASSERT_EQ(err, AMDSMI_STATUS_INVAL);

    // vendor_id, unique_id, target_gfx_version
    amdsmi_asic_info_t asic_info = {};
    err = amdsmi_get_gpu_asic_info(processor_handles_[i], &asic_info);
    if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
        std::cout <<
            "\t**amdsmi_dev_unique_id() is not supported"
            " on this machine" << std::endl;
        EXPECT_EQ(asic_info.target_graphics_version, std::numeric_limits<uint64_t>::max());
        // Verify api support checking functionality is working
        err = amdsmi_get_gpu_asic_info(processor_handles_[i], nullptr);
        ASSERT_EQ(err, AMDSMI_STATUS_NOT_SUPPORTED);
    } else {
        if (err == AMDSMI_STATUS_SUCCESS) {
            IF_VERB(STANDARD) {
              std:: cout << "\t**GPU PCIe Vendor : "
                  << asic_info.vendor_name << std::endl;
              std::cout << "\t**Target GFX version: " << std::dec
                        << asic_info.target_graphics_version << "\n";
            }
            EXPECT_EQ(err, AMDSMI_STATUS_SUCCESS);
            EXPECT_NE(asic_info.target_graphics_version, std::numeric_limits<uint64_t>::max());
            // Verify api support checking functionality is working
            err = amdsmi_get_gpu_asic_info(processor_handles_[i], nullptr);
            ASSERT_EQ(err, AMDSMI_STATUS_INVAL);
        } else {
            std::cout << "amdsmi_dev_unique_id_get() failed with error " <<
                                                               err << std::endl;
        }
    }

    // kfd_id, node_id, current_partition_id
    amdsmi_kfd_info_t kfd_info = {};
    err = amdsmi_get_gpu_kfd_info(processor_handles_[i], &kfd_info);
    if (err != AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(kfd_info.kfd_id, std::numeric_limits<uint64_t>::max());
        EXPECT_EQ(kfd_info.node_id, std::numeric_limits<uint32_t>::max());
        EXPECT_EQ(kfd_info.current_partition_id, std::numeric_limits<uint32_t>::max());
    } else {
          IF_VERB(STANDARD) {
            std::cout << "\t**KFD ID: " << std::dec
                      << kfd_info.kfd_id << "\n";
            std::cout << "\t**Node ID: " << std::dec
                      << kfd_info.node_id << "\n";
            std::cout << "\t**Current Parition ID: " << std::dec
                      << kfd_info.current_partition_id << "\n";
          }
          EXPECT_EQ(err, AMDSMI_STATUS_SUCCESS);
          EXPECT_NE(kfd_info.kfd_id, std::numeric_limits<uint64_t>::max());
          EXPECT_NE(kfd_info.node_id, std::numeric_limits<uint32_t>::max());
          EXPECT_NE(kfd_info.current_partition_id, std::numeric_limits<uint32_t>::max());
    }
    // Verify api support checking functionality is working
    err = amdsmi_get_gpu_kfd_info(processor_handles_[i], nullptr);
    ASSERT_EQ(err, AMDSMI_STATUS_INVAL);

  err = amdsmi_get_lib_version(&ver);
  CHK_ERR_ASRT(err)

  ASSERT_TRUE(ver.major != 0xFFFFFFFF && ver.minor != 0xFFFFFFFF &&
              ver.release != 0xFFFFFFFF && ver.build != nullptr);
  IF_VERB(STANDARD) {
    std::cout << "\t**AMD SMI Library version: " << ver.major << "." <<
       ver.minor << "." << ver.release << " (" << ver.build << ")" << std::endl;
  }

    std::cout << std::setbase(10);

    amdsmi_fw_info_t fw_info;
    err = amdsmi_get_fw_info(processor_handles_[i], &fw_info);
    if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
        std::cout << "\t**No FW  " <<
                    " available on this system" << std::endl;
        err = amdsmi_get_fw_info(processor_handles_[i], nullptr);
        ASSERT_EQ(err, AMDSMI_STATUS_NOT_SUPPORTED);
    } else {
        CHK_ERR_ASRT(err)
    }
  }
}
