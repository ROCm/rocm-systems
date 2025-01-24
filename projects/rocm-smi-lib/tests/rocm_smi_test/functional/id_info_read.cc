/*
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
#include "rocm_smi_test/functional/id_info_read.h"
#include "rocm_smi_test/test_common.h"

TestIdInfoRead::TestIdInfoRead() : TestBase() {
  set_title("RSMI ID Info Read Test");
  set_description("This test verifies that ID information such as the "
             "device, subsystem and vendor IDs can be read properly.");
}

TestIdInfoRead::~TestIdInfoRead(void) {
}

void TestIdInfoRead::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestIdInfoRead::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestIdInfoRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestIdInfoRead::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // rsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

static const uint32_t kBufferLen = 80;
static const uint32_t kSize5 = 5;

void TestIdInfoRead::Run(void) {
  rsmi_status_t err;
  uint16_t id;
  uint64_t val_ui64;
  uint32_t drm_render_minor;

  char buffer[kBufferLen];
  char fix_market_name[kBufferLen];
  char small_buffer[kSize5];

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
    std::string temp_val = "Hello World111111111111111111111111111111";
    memset(fix_market_name, '\0', kBufferLen);
    temp_val.copy(fix_market_name, kBufferLen-1);
    fix_market_name[kBufferLen-1] = '\0';

    IF_VERB(STANDARD) {
      std::cout << "\t*************************" << std::endl;
      std::cout << "\t**Device index: " << i << std::endl;
    }

    // Get the device ID, name, vendor ID and vendor name for the device
    err = rsmi_dev_id_get(i, &id);
    if (err == RSMI_STATUS_NOT_SUPPORTED) {
      rsmi_status_t ret;
      // Verify api support checking functionality is working
      ret = rsmi_dev_id_get(i, nullptr);
      ASSERT_EQ(ret, RSMI_STATUS_NOT_SUPPORTED);
    } else {
      CHK_ERR_ASRT(err)

      IF_VERB(STANDARD) {
        std::cout << "\t**Device ID: 0x" << std::hex << id << std::endl;
      }
      // Verify api support checking functionality is working
      err = rsmi_dev_id_get(i, nullptr);
      ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);
    }
    // Get device Revision
    err = rsmi_dev_revision_get(i, &id);
    if (err == RSMI_STATUS_NOT_SUPPORTED) {
      rsmi_status_t ret;
      // Verify api support checking functionality is working
      ret = rsmi_dev_revision_get(i, nullptr);
      ASSERT_EQ(ret, RSMI_STATUS_NOT_SUPPORTED);
    } else {
      CHK_ERR_ASRT(err)

      IF_VERB(STANDARD) {
        std::cout << "\t**Dev.Rev.ID: 0x" << std::hex << id << std::endl;
      }
      // Verify api support checking functionality is working
      err = rsmi_dev_revision_get(i, nullptr);
      ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);
    }

    // Verify api support checking functionality is working
    err = rsmi_dev_market_name_get(i, small_buffer, kSize5);
    if (err == RSMI_STATUS_NOT_SUPPORTED) {
      std::cout << "\t**Device Marketing name (libdrm) not found on this system." <<
                                                                    std::endl;
    } else {
      IF_VERB(STANDARD) {
        std::cout << "\t**[Test small_buffer] Device Marketing name"
                  << " (libdrm): " << small_buffer << std::endl;
      }
      ASSERT_EQ(err, RSMI_STATUS_INSUFFICIENT_SIZE);
    }

    err = rsmi_dev_market_name_get(i, nullptr, kBufferLen);
    ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);
    bool shouldCompareMarketValue = false;

    err = rsmi_dev_market_name_get(i, fix_market_name, kBufferLen);
    if (err == RSMI_STATUS_NOT_SUPPORTED) {
      std::cout << "\t**Device Marketing name (libdrm) not found on this system." <<
                                                                    std::endl;
    } else {
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\t**[Test fix_market_name] Device Marketing name"
                  << " (libdrm): " << fix_market_name << std::endl;
      }
      if (err == RSMI_STATUS_SUCCESS) {
        shouldCompareMarketValue = true;
      }
    }

    err = rsmi_dev_market_name_get(i, buffer, kBufferLen);
    if (err == RSMI_STATUS_NOT_SUPPORTED) {
      std::cout << "\t**Device Marketing name (libdrm) not found on this system." <<
                                                                    std::endl;
    } else {
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\t**[buffer] Device Marketing name (libdrm): " << buffer << std::endl;
      }
      (shouldCompareMarketValue && err == RSMI_STATUS_SUCCESS) ? shouldCompareMarketValue = true :
        shouldCompareMarketValue = false;
    }

    if (shouldCompareMarketValue) {
      IF_VERB(STANDARD) {
        std::cout << "\t**fix_market_name: |" << fix_market_name << "|" << std::endl;
        std::cout << "\t**buffer: |" << buffer << "|" << std::endl;
        (strcmp(fix_market_name, buffer) == 0) ?
          (std::cout << "\t**Both fix_market_name & buffer are equal" << std::endl) :
          (std::cout << "\t**Both fix_market_name & buffer not equal" << std::endl);
      }
      ASSERT_STREQ(fix_market_name, buffer);
    }

    err = rsmi_dev_name_get(i, buffer, kBufferLen);
    if (err == RSMI_STATUS_NOT_SUPPORTED) {
      std::cout << "\t**Device Marketing name not found on this system." <<
                                                                    std::endl;
      // Verify api support checking functionality is working
      err = rsmi_dev_name_get(i, nullptr, kBufferLen);
      ASSERT_EQ(err, RSMI_STATUS_NOT_SUPPORTED);
    } else {
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\t**Device Marketing name: " << buffer << std::endl;
      }
      // Verify api support checking functionality is working
      err = rsmi_dev_name_get(i, nullptr, kBufferLen);
      ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);
    }
    err = rsmi_dev_brand_get(i, buffer, kBufferLen);
    if (err == RSMI_STATUS_NOT_SUPPORTED) {
        // Verify api support checking functionality is working
        err = rsmi_dev_brand_get(i, nullptr, kBufferLen);
        ASSERT_EQ(err, RSMI_STATUS_NOT_SUPPORTED);
    } else {
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\t**Device Brand name: " << buffer << std::endl;
      }
      // Verify api support checking functionality is working
      err = rsmi_dev_brand_get(i, nullptr, kBufferLen);
      ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);
    }
    err = rsmi_dev_vram_vendor_get(i, buffer, kBufferLen);
    if (err == RSMI_STATUS_NOT_SUPPORTED) {
      std::cout <<
        "\t**Vram Vendor string not supported on this system." << std::endl;
      err = rsmi_dev_vram_vendor_get(i, nullptr, kBufferLen);
      ASSERT_EQ(err, RSMI_STATUS_NOT_SUPPORTED);
    } else {
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\t**Device Vram Vendor name: " << buffer << std::endl;
      }
      err = rsmi_dev_vram_vendor_get(i, nullptr, kBufferLen);
      ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);
    }
    err = rsmi_dev_vendor_id_get(i, &id);
    if (err == RSMI_STATUS_NOT_SUPPORTED) {
      // Verify api support checking functionality is working
      err = rsmi_dev_vendor_id_get(i, nullptr);
      ASSERT_EQ(err, RSMI_STATUS_NOT_SUPPORTED);
    } else {
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\t**Vendor ID: 0x" << std::hex << id << std::endl;
      }
      // Verify api support checking functionality is working
      err = rsmi_dev_vendor_id_get(i, nullptr);
      ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);
    }
    err = rsmi_dev_drm_render_minor_get(i, &drm_render_minor);
    if (err == RSMI_STATUS_NOT_SUPPORTED) {
      // Verify api support checking functionality is working
      err = rsmi_dev_drm_render_minor_get(i, nullptr);
      ASSERT_EQ(err, RSMI_STATUS_NOT_SUPPORTED);
    } else {
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\t**DRM Render Minor: " << drm_render_minor << std::endl;
      }
      // Verify api support checking functionality is working
      err = rsmi_dev_drm_render_minor_get(i, nullptr);
      ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);
    }
    err = rsmi_dev_vendor_name_get(i, buffer, kBufferLen);
    if (err == RSMI_STATUS_NOT_SUPPORTED) {
      std::cout << "\t**Device Vendor name string not found on this system." <<
                                                                     std::endl;
      // Verify api support checking functionality is working
      err = rsmi_dev_vendor_name_get(i, nullptr, kBufferLen);
      ASSERT_EQ(err, RSMI_STATUS_NOT_SUPPORTED);
    } else {
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\t**Device Vendor name: " << buffer << std::endl;
      }
      // Verify api support checking functionality is working
      err = rsmi_dev_vendor_name_get(i, nullptr, kBufferLen);
      ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);
    }

    // Get the device ID, name, vendor ID and vendor name for the sub-device
    err = rsmi_dev_subsystem_id_get(i, &id);
    if (err == RSMI_STATUS_NOT_SUPPORTED) {
      // Verify api support checking functionality is working
      err = rsmi_dev_subsystem_id_get(i, nullptr);
      ASSERT_EQ(err, RSMI_STATUS_NOT_SUPPORTED);
    } else {
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\t**Subsystem ID: 0x" << std::hex << id << std::endl;
      }
      // Verify api support checking functionality is working
      err = rsmi_dev_subsystem_id_get(i, nullptr);
      ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);
    }
    err = rsmi_dev_subsystem_name_get(i, buffer, kBufferLen);
    if (err == RSMI_STATUS_NOT_SUPPORTED) {
      std::cout << "\t**Subsystem name string not found on this system." <<
                                                                    std::endl;
      // Verify api support checking functionality is working
      err = rsmi_dev_subsystem_name_get(i, nullptr, kBufferLen);
      ASSERT_EQ(err, RSMI_STATUS_NOT_SUPPORTED);
    } else {
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\t**Subsystem name: " << buffer << std::endl;
      }
      // Verify api support checking functionality is working
      err = rsmi_dev_subsystem_name_get(i, nullptr, kBufferLen);
      ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);
    }
    err = rsmi_dev_subsystem_vendor_id_get(i, &id);
    if (err == RSMI_STATUS_NOT_SUPPORTED) {
      // Verify api support checking functionality is working
      err = rsmi_dev_subsystem_vendor_id_get(i, nullptr);
      ASSERT_EQ(err, RSMI_STATUS_NOT_SUPPORTED);
    } else {
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\t**Sub-system Vendor ID: 0x" << std::hex <<
                                                              id << std::endl;
      }
      // Verify api support checking functionality is working
      err = rsmi_dev_subsystem_vendor_id_get(i, nullptr);
      ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);
    }
    err = rsmi_dev_vendor_name_get(i, buffer, kBufferLen);
    if (err == RSMI_STATUS_NOT_SUPPORTED) {
      std::cout <<
           "\t**Subsystem Vendor name string not found on this system." <<
                                                                    std::endl;
     // Verify api support checking functionality is working
     err = rsmi_dev_vendor_name_get(i, nullptr, kBufferLen);
     ASSERT_EQ(err, RSMI_STATUS_NOT_SUPPORTED);
    } else {
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\t**Subsystem Vendor name: " << buffer << std::endl;
      }
      // Verify api support checking functionality is working
      err = rsmi_dev_vendor_name_get(i, nullptr, kBufferLen);
      ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);
    }

    err = rsmi_dev_pci_id_get(i, &val_ui64);
    // Don't check for RSMI_STATUS_NOT_SUPPORTED since this should always be
    // supported. It is not based on a sysfs file.
    CHK_ERR_ASRT(err)
    IF_VERB(STANDARD) {
      std::cout << "\t**PCI ID (BDFID): 0x" << std::hex << val_ui64;
      std::cout << " (" << std::dec << val_ui64 << ")" << std::endl;
    }
    // Verify api support checking functionality is working
    err = rsmi_dev_pci_id_get(i, nullptr);
    ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);

    err = rsmi_dev_serial_number_get(i, buffer, kBufferLen);
    if (err == RSMI_STATUS_NOT_SUPPORTED) {
      // Verify api support checking functionality is working
      err = rsmi_dev_serial_number_get(i, nullptr, kBufferLen);
      ASSERT_EQ(err, RSMI_STATUS_NOT_SUPPORTED);

      std::cout <<
        "\t**Serial Number string not supported on this system." << std::endl;
    } else {
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\t**Device Serial Number:" << buffer << std::endl;
      }
      // Verify api support checking functionality is working
      err = rsmi_dev_serial_number_get(i, nullptr, kBufferLen);
      ASSERT_EQ(err, RSMI_STATUS_INVALID_ARGS);
    }
  }
}
