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

#include <cstdint>

#include <iostream>
#include <string>

#include <gtest/gtest.h>
#include "amd_smi/amdsmi.h"
#include "mem_page_info_read.h"

TestMemPageInfoRead::TestMemPageInfoRead() : TestBase() {
  set_title("AMDSMI Memory Page Info Test");
  set_description("The Memory Page Info. test verifies that we can read "
      "memory page information, and then displays the information read");
}

TestMemPageInfoRead::~TestMemPageInfoRead(void) {
}

void TestMemPageInfoRead::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestMemPageInfoRead::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestMemPageInfoRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestMemPageInfoRead::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

void TestMemPageInfoRead::Run(void) {
  amdsmi_status_t err;
  amdsmi_retired_page_record_t *records;
  uint32_t num_pages;

  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
    PrintDeviceHeader(processor_handles_[i]);

    err = amdsmi_get_gpu_memory_reserved_pages(processor_handles_[i], &num_pages, nullptr);

    if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
      std::cout <<
          "\t**Memory page information is not supported for this device"
                                                                 << std::endl;

      // Verify api support checking functionality is working
      err = amdsmi_get_gpu_memory_reserved_pages(processor_handles_[i], nullptr, nullptr);
      ASSERT_EQ(err, AMDSMI_STATUS_NOT_SUPPORTED);

      continue;
    } else {
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\tNumber of memory page records: " << num_pages <<
                                                                    std::endl;
      }
      // Verify api support checking functionality is working
      err = amdsmi_get_gpu_memory_reserved_pages(processor_handles_[i], nullptr, nullptr);
      ASSERT_EQ(err, AMDSMI_STATUS_INVAL);
    }

    if (num_pages > 0) {
      records = new amdsmi_retired_page_record_t[num_pages];

      assert(records != nullptr);

      err = amdsmi_get_gpu_memory_reserved_pages(processor_handles_[i], &num_pages, records);
      if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
        std::cout << "\t**Getting Memory Page Retirement Status not "
                                     "supported for this device" << std::endl;
        continue;
      } else {
          CHK_ERR_ASRT(err)
      }

      IF_VERB(STANDARD) {
        std::cout.setf(std::ios::hex, std::ios::basefield);
        std::string page_state;

        for (uint32_t p = 0; p < num_pages; ++p) {
          std::cout << "\t\tAddress: 0x" << records[p].page_address;
          std::cout << "  Size: " << records[p].page_size;

          switch (records[p].status) {
            case AMDSMI_MEM_PAGE_STATUS_RESERVED:
              page_state = "Retired";
              break;

            case AMDSMI_MEM_PAGE_STATUS_PENDING:
              page_state = "Pending";
              break;

            case AMDSMI_MEM_PAGE_STATUS_UNRESERVABLE:
              page_state = "Unreservable";
              break;

            default:
              ASSERT_EQ(0, 1) << "Unexpected memory page status";
          }
          std::cout << "  Status: " << page_state << std::endl;
        }
        std::cout.setf(std::ios::dec, std::ios::basefield);
      }
      delete []records;
    } else {
      continue;
    }
  }
}
