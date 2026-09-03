// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mem_page_info_read.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>
#include <string>

#include "amd_smi/amdsmi.h"
#include "test_common.h"

TestMemPageInfoRead::TestMemPageInfoRead() : TestBase() {
  set_title("AMDSMI Memory Page Info Test");
  set_description(
      "The Memory Page Info. test verifies that we can read "
      "memory page information, and then displays the information read");
}

TestMemPageInfoRead::~TestMemPageInfoRead(void) {}

void TestMemPageInfoRead::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestMemPageInfoRead::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

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
  amdsmi_retired_page_record_t* records;
  uint32_t num_pages;

  TestBase::Run();
  PRINT_VERBOSITY();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
    PrintDeviceHeader(processor_handles_[i]);

    DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_reserved_pages", "gpu=" + std::to_string(i),
                       VERB(STANDARD));
    err = amdsmi_get_gpu_memory_reserved_pages(processor_handles_[i], &num_pages, nullptr);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);

    if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
      // Verify api support checking functionality is working
      DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_reserved_pages(nullptr)",
                         "gpu=" + std::to_string(i), VERB(STANDARD));
      err = amdsmi_get_gpu_memory_reserved_pages(processor_handles_[i], nullptr, nullptr);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_NOT_SUPPORTED);
      ASSERT_EQ(err, AMDSMI_STATUS_NOT_SUPPORTED);
      continue;
    } else {
      CHK_ERR_ASRT(err)
      IF_VERB(STANDARD) {
        std::cout << "\tNumber of memory page records: " << num_pages << std::endl;
      }
      // Verify api support checking functionality is working
      DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_reserved_pages(nullptr)",
                         "gpu=" + std::to_string(i), VERB(STANDARD));
      err = amdsmi_get_gpu_memory_reserved_pages(processor_handles_[i], nullptr, nullptr);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
      ASSERT_EQ(err, AMDSMI_STATUS_INVAL);
    }

    if (num_pages > 0) {
      records = new amdsmi_retired_page_record_t[num_pages];

      assert(records != nullptr);

      DISPLAY_AMDSMI_API("amdsmi_get_gpu_memory_reserved_pages", "gpu=" + std::to_string(i),
                         VERB(STANDARD));
      err = amdsmi_get_gpu_memory_reserved_pages(processor_handles_[i], &num_pages, records);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
      if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
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
      delete[] records;
    } else {
      continue;
    }
  }
}
