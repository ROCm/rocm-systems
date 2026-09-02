// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "perf_determinism.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>
#include <string>

#include "amd_smi/amdsmi.h"
#include "test_common.h"

TestPerfDeterminism::TestPerfDeterminism() : TestBase() {
  set_title("AMDSMI Performance Determinism Test");
  set_description(
      "The Performance Determinism tests verifies "
      "Enabling/Disabling performance determinism mode.");
}

TestPerfDeterminism::~TestPerfDeterminism(void) {}

void TestPerfDeterminism::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestPerfDeterminism::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestPerfDeterminism::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestPerfDeterminism::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

void TestPerfDeterminism::Run(void) {
  amdsmi_status_t err;
  amdsmi_dev_perf_level_t pfl;
  amdsmi_od_volt_freq_data_t odv{};
  amdsmi_status_t ret;
  uint64_t clkvalue(0);
  TestBase::Run();
  PRINT_VERBOSITY();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
    PrintDeviceHeader(processor_handles_[i]);
    // Put back the level this GPU arrived with; it is not always AUTO.
    amdsmi_dev_perf_level_t entry_pfl = AMDSMI_DEV_PERF_LEVEL_AUTO;
    const bool have_entry_pfl =
        amdsmi_get_gpu_perf_level(processor_handles_[i], &entry_pfl) == AMDSMI_STATUS_SUCCESS;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_od_volt_info", "gpu=" + std::to_string(i), VERB(STANDARD));
    err = amdsmi_get_gpu_od_volt_info(processor_handles_[i], &odv);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
    if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) { std::cout << "\t** Not supported on this machine\n"; }
      return;
    } else if (err == AMDSMI_STATUS_SUCCESS) {
      clkvalue = (odv.curr_sclk_range.lower_bound / 1000000) + 50;
    } else {
      IF_VERB(STANDARD) { std::cout << "\t** Unable to retrieve lower bound sclk, continue.. \n"; }
      continue;
    }
    std::cout << "About to rsmi_perf_determinism_mode_set() -->\n";

    DISPLAY_AMDSMI_API("amdsmi_set_gpu_perf_determinism_mode", "gpu=" + std::to_string(i),
                       VERB(STANDARD));
    err = amdsmi_set_gpu_perf_determinism_mode(processor_handles_[i], clkvalue);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
    if (err == AMDSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) { std::cout << "\t**Not supported on this machine" << std::endl; }
      continue;
    } else {
      DISPLAY_AMDSMI_API("amdsmi_get_gpu_perf_level", "gpu=" + std::to_string(i), VERB(STANDARD));
      ret = amdsmi_get_gpu_perf_level(processor_handles_[i], &pfl);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
      CHK_ERR_ASRT(ret)
      IF_VERB(STANDARD) {
        std::cout << "\t**New Perf Level:" << GetPerfLevelStr(pfl) << std::endl;
        std::cout << "\t**SCLK is now set to " << clkvalue << std::endl;
      }

      std::cout << "\t**Resetting performance determinism" << std::endl;
      DISPLAY_AMDSMI_API("amdsmi_set_gpu_perf_level", "gpu=" + std::to_string(i), VERB(STANDARD));
      err = amdsmi_set_gpu_perf_level(processor_handles_[i],
                                      have_entry_pfl ? entry_pfl : AMDSMI_DEV_PERF_LEVEL_AUTO);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS);
      CHK_ERR_ASRT(err)
      DISPLAY_AMDSMI_API("amdsmi_get_gpu_perf_level", "gpu=" + std::to_string(i), VERB(STANDARD));
      ret = amdsmi_get_gpu_perf_level(processor_handles_[i], &pfl);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
      CHK_ERR_ASRT(ret)
      IF_VERB(STANDARD) { std::cout << "\t**New Perf Level:" << GetPerfLevelStr(pfl) << std::endl; }
    }  // END - SET SUPPORTED
  }  // END - DEVICE LOOP
}
