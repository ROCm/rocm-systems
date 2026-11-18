// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "perf_level_read_write.h"

#include <gtest/gtest.h>
#include <stddef.h>
#include <stdint.h>

#include <iostream>
#include <map>

#include "amd_smi/amdsmi.h"
#include "test_common.h"

TestPerfLevelReadWrite::TestPerfLevelReadWrite() : TestBase() {
  set_title("AMDSMI Performance Level Read/Write Test");
  set_description(
      "The Performance Level tests verify that the performance "
      "level settings can be read and controlled properly.");
}

TestPerfLevelReadWrite::~TestPerfLevelReadWrite(void) {}

void TestPerfLevelReadWrite::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestPerfLevelReadWrite::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestPerfLevelReadWrite::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestPerfLevelReadWrite::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

void TestPerfLevelReadWrite::Run(void) {
  amdsmi_status_t ret;
  amdsmi_dev_perf_level_t pfl, orig_pfl;

  TestBase::Run();
  PRINT_VERBOSITY();
  if (setup_failed_) {
    IF_VERB(STANDARD) { std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl; }
    return;
  }

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    PrintDeviceHeader(processor_handles_[dv_ind]);

    DISPLAY_AMDSMI_API("amdsmi_get_gpu_perf_level", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    ret = amdsmi_get_gpu_perf_level(processor_handles_[dv_ind], &orig_pfl);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
      ASSERT_EQ(ret, AMDSMI_STATUS_NOT_SUPPORTED);
      continue;
    }

    IF_VERB(STANDARD) {
      std::cout << "\t**Original Perf Level:" << GetPerfLevelStr(orig_pfl) << std::endl;
    }

    uint32_t pfl_i = static_cast<uint32_t>(AMDSMI_DEV_PERF_LEVEL_FIRST);
    for (; pfl_i <= static_cast<uint32_t>(AMDSMI_DEV_PERF_LEVEL_LAST); pfl_i++) {
      if (pfl_i == static_cast<uint32_t>(orig_pfl)) {
        continue;
      }

      IF_VERB(STANDARD) {
        std::cout << "Set Performance Level to "
                  << GetPerfLevelStr(static_cast<amdsmi_dev_perf_level_t>(pfl_i)) << " ..."
                  << std::endl;
      }
      DISPLAY_AMDSMI_API("amdsmi_set_gpu_perf_level", "gpu=" + std::to_string(dv_ind),
                         VERB(STANDARD));
      ret = amdsmi_set_gpu_perf_level(processor_handles_[dv_ind],
                                      static_cast<amdsmi_dev_perf_level_t>(pfl_i));
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
      if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
        std::cout << "\t**" << GetPerfLevelStr(static_cast<amdsmi_dev_perf_level_t>(pfl_i))
                  << " returned AMDSMI_STATUS_NOT_SUPPORTED" << std::endl;
      } else {
        CHK_ERR_ASRT(ret)
        DISPLAY_AMDSMI_API("amdsmi_get_gpu_perf_level", "gpu=" + std::to_string(dv_ind),
                           VERB(STANDARD));
        ret = amdsmi_get_gpu_perf_level(processor_handles_[dv_ind], &pfl);
        DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
        CHK_ERR_ASRT(ret)
        IF_VERB(STANDARD) {
          std::cout << "\t**New Perf Level:" << GetPerfLevelStr(pfl) << std::endl;
        }
      }
    }
    IF_VERB(STANDARD) {
      std::cout << "Reset Perf level to " << GetPerfLevelStr(orig_pfl) << " ..." << std::endl;
    }
    DISPLAY_AMDSMI_API("amdsmi_set_gpu_perf_level", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    ret = amdsmi_set_gpu_perf_level(processor_handles_[dv_ind], orig_pfl);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
      continue;
    }
    CHK_ERR_ASRT(ret)
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_perf_level", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    ret = amdsmi_get_gpu_perf_level(processor_handles_[dv_ind], &pfl);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    CHK_ERR_ASRT(ret)

    IF_VERB(STANDARD) { std::cout << "\t**New Perf Level:" << GetPerfLevelStr(pfl) << std::endl; }
  }
}
