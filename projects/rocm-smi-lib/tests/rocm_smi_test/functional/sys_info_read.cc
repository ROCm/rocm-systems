/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2019, Advanced Micro Devices, Inc.
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

#include "gtest/gtest.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi_test/functional/sys_info_read.h"
#include "rocm_smi_test/test_common.h"

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
  char buffer[80];
  rsmi_version_t ver = {0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, nullptr};

  TestBase::Run();

  for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
    PrintDeviceHeader(i);

    err = rsmi_dev_vbios_version_get(i, buffer, 80);

    if (err != RSMI_STATUS_SUCCESS) {
      if (err == RSMI_STATUS_FILE_ERROR) {
        IF_VERB(STANDARD) {
          std::cout << "\t**VBIOS read: Not supported on this machine"
                                                                << std::endl;
        }
      } else {
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

    err = rsmi_version_get(&ver);
    CHK_ERR_ASRT(err)

    ASSERT_TRUE(ver.major != 0xFFFFFFFF && ver.minor != 0xFFFFFFFF &&
                               ver.patch != 0xFFFFFFFF && ver.build != nullptr);
    IF_VERB(STANDARD) {
      std::cout << "\t**RocM SMI Library version: " << ver.major << "." <<
         ver.minor << "." << ver.patch << " (" << ver.build << ")" << std::endl;
    }
  }
}
