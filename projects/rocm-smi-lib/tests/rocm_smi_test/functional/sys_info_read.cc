/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */

#include <stdint.h>
#include <stddef.h>

#include <iostream>
#include <string>
#include <limits>

#include "gtest/gtest.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi_test/functional/sys_info_read.h"
#include "rocm_smi_test/test_common.h"
#include "rocm_smi_test/test_utils.h"

TestSysInfoRead::TestSysInfoRead() : TestBase() {
  set_title("RSMI System Info Read Test");
  set_description("This test verifies that system information such as the "
             "BDFID, RSMI version, VBIOS version, etc. can be read properly.");
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
  // rsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}


void TestSysInfoRead::Run(void) {
  rsmi_status_t err;
  uint64_t val_ui64;
  uint32_t val_ui32;
  int32_t val_i32;
  char buffer[80];
  rsmi_version_t ver = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, nullptr};

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
    PrintDeviceHeader(i);

    err = rsmi_dev_vbios_version_get(i, buffer, 80);

    if (err != RSMI_STATUS_SUCCESS) {
      if ((err == RSMI_STATUS_FILE_ERROR) || (err == RSMI_STATUS_NOT_SUPPORTED)) {
        IF_VERB(STANDARD) {
          std::cout << "\t**VBIOS read: Not supported on this machine"
                                                                << std::endl;
        }
        // Verify api support checking functionality is working
        err = rsmi_dev_vbios_version_get(i, nullptr, 80);
        ASSERT_EQ(err, RSMI_STATUS_NOT_SUPPORTED);
      } else {
        // Verify api support checking functionality is working
        err = rsmi_dev_vbios_version_get(i, nullptr, 80);
        ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);

        CHK_ERR_ASRT(err)
      }
    } else {
      IF_VERB(STANDARD) {
        std::cout << "\t**VBIOS Version: " << std::hex << buffer << std::endl;
      }
    }

    err = rsmi_dev_pci_id_get(i, &val_ui64);
    CHK_ERR_ASRT(err)
    IF_VERB(STANDARD) {
      std::cout << "\t**PCI ID (BDFID): 0x" << std::hex << val_ui64;
      std::cout << " (" << std::dec << val_ui64 << ")" << std::endl;
    }
    // Verify api support checking functionality is working
    err = rsmi_dev_pci_id_get(i, nullptr);
    ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);

    err = rsmi_topo_numa_affinity_get(i, &val_i32);
    if (err == RSMI_STATUS_NOT_SUPPORTED) {
      std::cout << "\t**rsmi_topo_numa_affinity_get(): Not supported on this machine"
                << std::endl;
      ASSERT_EQ(err, RSMI_STATUS_NOT_SUPPORTED);
    } else  {
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\t**NUMA NODE: 0x" << std::hex << val_i32;
        std::cout << " (" << std::dec << val_i32 << ")" << std::endl;
      }
    }
    // Verify api support checking functionality is working
    err = rsmi_topo_numa_affinity_get(i, nullptr);
    ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);

    err = rsmi_dev_unique_id_get(i, &val_ui64);
    if (err == RSMI_STATUS_NOT_SUPPORTED) {
        std::cout <<
            "\t**rsmi_dev_unique_id() is not supported"
            " on this machine" << std::endl;
        // Verify api support checking functionality is working
        err = rsmi_dev_unique_id_get(i, nullptr);
        ASSERT_EQ(err, RSMI_STATUS_NOT_SUPPORTED);
    } else {
        if (err == RSMI_STATUS_SUCCESS) {
            IF_VERB(STANDARD) {
                std::cout << "\t**GPU Unique ID : " << std::hex << val_ui64 <<
                std::endl;
            }
            // Verify api support checking functionality is working
            err = rsmi_dev_unique_id_get(i, nullptr);
            ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);
        } else {
            std::cout << "rsmi_dev_unique_id_get() failed with error " <<
                                                               err << std::endl;
        }
    }

    err = rsmi_version_get(&ver);
    CHK_ERR_ASRT(err)

    ASSERT_TRUE(ver.major != 0xFFFFFFFF && ver.minor != 0xFFFFFFFF &&
                               ver.patch != 0xFFFFFFFF && ver.build != nullptr);
    IF_VERB(STANDARD) {
      std::cout << "\t**RocM SMI Library version: " << ver.major << "." <<
         ver.minor << "." << ver.patch << " (" << ver.build << ")" << std::endl;
    }

    std::cout << std::setbase(10);
    for (int x = RSMI_FW_BLOCK_FIRST; x <= RSMI_FW_BLOCK_LAST; ++x) {
      rsmi_fw_block_t block = static_cast<rsmi_fw_block_t>(x);
      err = rsmi_dev_firmware_version_get(i, block, &val_ui64);
      if (err == RSMI_STATUS_NOT_SUPPORTED) {
        std::cout << "\t**No FW block " << NameFromFWEnum(block) <<
                                     " available on this system" << std::endl;

        // Verify api support checking functionality is working
        err = rsmi_dev_firmware_version_get(i, block, nullptr);
        ASSERT_EQ(err, RSMI_STATUS_NOT_SUPPORTED);
        continue;
      }
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\t**FW VERSION for " << NameFromFWEnum(block) <<
                                                ": " << val_ui64 << std::endl;
      }
      // Verify api support checking functionality is working
      err = rsmi_dev_firmware_version_get(i, block, nullptr);
      ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);
    }

    err = rsmi_dev_target_graphics_version_get(i, &val_ui64);
    IF_VERB(STANDARD) {
        std::cout << "\t**Target GFX version: " << std::dec
        << val_ui64 << "\n";
    }
    EXPECT_EQ(err, RSMI_STATUS_SUCCESS);
    EXPECT_NE(val_ui64, std::numeric_limits<uint64_t>::max());
    err = rsmi_dev_target_graphics_version_get(i, nullptr);
    EXPECT_EQ(err, RSMI_STATUS_INVALID_ARGS);

    err = rsmi_dev_guid_get(i, &val_ui64);
    IF_VERB(STANDARD) {
        std::cout << "\t**GUID: " << std::dec
        << val_ui64 << "\n";
    }
    EXPECT_EQ(err, RSMI_STATUS_SUCCESS);
    EXPECT_NE(val_ui64, std::numeric_limits<uint64_t>::max());
    err = rsmi_dev_guid_get(i, nullptr);
    EXPECT_EQ(err, RSMI_STATUS_INVALID_ARGS);

    err = rsmi_dev_node_id_get(i, &val_ui32);
    IF_VERB(STANDARD) {
        std::cout << "\t**Node ID: " << std::dec
        << val_ui32 << "\n";
    }
    EXPECT_EQ(err, RSMI_STATUS_SUCCESS);
    EXPECT_NE(val_ui32, std::numeric_limits<uint32_t>::max());
    err = rsmi_dev_node_id_get(i, nullptr);
    EXPECT_EQ(err, RSMI_STATUS_INVALID_ARGS);

  }
}
