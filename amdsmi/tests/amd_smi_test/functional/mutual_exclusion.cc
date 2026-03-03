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
#include "mutual_exclusion.h"

#define  AMD_SMI_INIT_FLAG_RESRV_TEST1 0x800000000000000  //!< Reserved for test

TestMutualExclusion::TestMutualExclusion() : TestBase() {
  set_title("Mutual Exclusion Test");
  set_description("Verify that AMDSMI only allows 1 process at a time"
    " to access AMDSMI resources (primarily sysfs files). This test has one "
    "process that obtains the mutex that ensures only 1 process accesses a "
      "device's sysfs files at a time, and another process that attempts "
      "to access the device's sysfs files. The second process should fail "
      "in these attempts.");
}

TestMutualExclusion::~TestMutualExclusion(void) {
}

extern amdsmi_status_t rsmi_test_sleep(uint32_t dv_ind, uint32_t seconds);

void TestMutualExclusion::SetUp(void) {
  std::string label;
  amdsmi_status_t ret;

  //   TestBase::SetUp(AMD_SMI_INIT_FLAG_RESRV_TEST1);
  IF_VERB(STANDARD) {
    MakeHeaderStr(kSetupLabel, &label);
    printf("\n\t%s\n", label.c_str());
  }

  sleeper_process_ = false;
  child_ = 0;
  child_ = fork();

  if (child_ != 0) {
    sleeper_process_ = true;  // sleeper_process is parent

    // AMD_SMI_INIT_FLAG_RESRV_TEST1 tells rsmi to fail immediately
    // if it can't get the mutex instead of waiting.
    ret = amdsmi_init(AMD_SMI_INIT_FLAG_RESRV_TEST1);
    if (ret != AMDSMI_STATUS_SUCCESS) {
      setup_failed_ = true;
    }
    ASSERT_EQ(ret, AMDSMI_STATUS_SUCCESS);

    sleep(2);  // Let both processes get through amdsmi_init
  } else {
    sleep(1);  // Let the sleeper process get through amdsmi_init() before
              // this one goes, so it doesn't fail.
    ret = amdsmi_init(AMD_SMI_INIT_FLAG_RESRV_TEST1);
    if (ret != AMDSMI_STATUS_SUCCESS) {
      setup_failed_ = true;
    }
    ASSERT_EQ(ret, AMDSMI_STATUS_SUCCESS);

    sleep(2);  // Let both processes get through amdsmi_init;
  }

  num_monitor_devs_ = num_monitor_devs();

  if (num_monitor_devs_ == 0) {
    std::cout << "No monitor devices found on this machine." << std::endl;
    std::cout << "No ROCm SMI tests can be run." << std::endl;
    setup_failed_ = true;
  }

  return;
}

void TestMutualExclusion::DisplayTestInfo(void) {
  IF_VERB(STANDARD) {
    TestBase::DisplayTestInfo();
  }
}

void TestMutualExclusion::DisplayResults(void) const {
  IF_VERB(STANDARD) {
    TestBase::DisplayResults();
  }
  return;
}

void TestMutualExclusion::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

extern amdsmi_status_t
rsmi_test_sleep(uint32_t dv_ind, uint32_t seconds);

void TestMutualExclusion::Run(void) {
  amdsmi_status_t ret;

  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  if (sleeper_process_) {
    IF_VERB(STANDARD) {
      std::cout << "MUTEX_HOLDER process: started sleeping for 10 seconds..." <<
                                                                     std::endl;
    }
    ret = rsmi_test_sleep(0, 10);
    ASSERT_EQ(ret, AMDSMI_STATUS_SUCCESS);
    IF_VERB(STANDARD) {
      std::cout << "MUTEX_HOLDER process: Sleep process woke up." << std::endl;
    }
    pid_t cpid = wait(nullptr);
    ASSERT_EQ(cpid, child_);
  } else {
    // Both processes should have completed amdsmi_init().
    // let the other process get started on rsmi_test_sleep().
    sleep(2);
    TestBase::Run();
    IF_VERB(STANDARD) {
      std::cout << "TESTER process: verifing that all amdsmi_dev_* functions "
                    "return AMDSMI_STATUS_BUSY because MUTEX_HOLDER process "
                                               "holds the mutex" << std::endl;
    }
    // Try all the device related rsmi calls. They should all fail with
    // AMDSMI_STATUS_BUSY
    // Set dummy values should to working, deterministic values.
    uint16_t dmy_ui16 = 0;
    uint32_t dmy_ui32 = 1;
    uint64_t dmy_ui64 = 0;
    int64_t dmy_i64 = 0;
    char dmy_str[10];
    amdsmi_dev_perf_level_t dmy_perf_lvl;
    amdsmi_frequencies_t dmy_freqs{};
    amdsmi_od_volt_freq_data_t dmy_od_volt{};
    amdsmi_freq_volt_region_t dmy_vlt_reg{};
    amdsmi_error_count_t dmy_err_cnt{};
    amdsmi_ras_err_state_t dmy_ras_err_st;

    // This can be replaced with ASSERT_EQ() once env. stabilizes
#define CHECK_RET(A, B) { \
  if ((A) != (B)) { \
    std::cout << "Expected return value of " << B << \
                               " but got " << A << std::endl; \
    std::cout << "at " << __FILE__ << ":" << __LINE__ << std::endl; \
  } \
}
    ret = amdsmi_get_gpu_id(processor_handles_[0], &dmy_ui16);

    // vendor_id, unique_id
    amdsmi_asic_info_t asic_info;
    ret = amdsmi_get_gpu_asic_info(processor_handles_[0], &asic_info);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);

    // device name, brand, serial_number
    amdsmi_board_info_t board_info;
    ret = amdsmi_get_gpu_board_info(processor_handles_[0], &board_info);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);

    ret = amdsmi_get_gpu_vendor_name(processor_handles_[0], dmy_str, 10);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);
    ret = amdsmi_get_gpu_vram_vendor(processor_handles_[0], dmy_str, 10);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);
    ret = amdsmi_get_gpu_subsystem_id(processor_handles_[0], &dmy_ui16);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);
    ret = amdsmi_get_gpu_bdf_id(processor_handles_[0], &dmy_ui64);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);
    ret = amdsmi_get_gpu_pci_throughput(processor_handles_[0], &dmy_ui64, &dmy_ui64, &dmy_ui64);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);
    ret =  amdsmi_get_gpu_pci_replay_counter(processor_handles_[0], &dmy_ui64);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);
    ret =  amdsmi_set_gpu_pci_bandwidth(processor_handles_[0], 0);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);
    ret = amdsmi_get_gpu_fan_rpms(processor_handles_[0], dmy_ui32, &dmy_i64);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);
    ret = amdsmi_get_gpu_fan_speed(processor_handles_[0], 0, &dmy_i64);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);
    ret = amdsmi_get_gpu_fan_speed_max(processor_handles_[0], 0, &dmy_ui64);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);
    ret =  amdsmi_get_temp_metric(processor_handles_[0], AMDSMI_TEMPERATURE_TYPE_EDGE, AMDSMI_TEMP_CURRENT, &dmy_i64);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);
    ret = amdsmi_reset_gpu_fan(processor_handles_[0], 0);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);
    ret = amdsmi_set_gpu_fan_speed(processor_handles_[0], dmy_ui32, 0);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);
    ret = amdsmi_get_gpu_perf_level(processor_handles_[0], &dmy_perf_lvl);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);
    ret = amdsmi_get_gpu_overdrive_level(processor_handles_[0], &dmy_ui32);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);
    ret =  amdsmi_get_clk_freq(processor_handles_[0], AMDSMI_CLK_TYPE_SYS, &dmy_freqs);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);
    ret =  amdsmi_get_gpu_od_volt_info(processor_handles_[0], &dmy_od_volt);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);
    ret =  amdsmi_get_gpu_od_volt_curve_regions(processor_handles_[0], &dmy_ui32, &dmy_vlt_reg);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);
    ret =  amdsmi_set_clk_freq(processor_handles_[0], AMDSMI_CLK_TYPE_SYS, 0);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);
    ret =  amdsmi_get_gpu_ecc_count(processor_handles_[0], AMDSMI_GPU_BLOCK_UMC, &dmy_err_cnt);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);
    ret =  amdsmi_get_gpu_ecc_enabled(processor_handles_[0], &dmy_ui64);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);
    ret =  amdsmi_get_gpu_ecc_status(processor_handles_[0], AMDSMI_GPU_BLOCK_UMC, &dmy_ras_err_st);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);

    /* Other functions holding device mutexes. Listed for reference.
    amdsmi_dev_sku_get
     amdsmi_set_gpu_od_clk_info
     amdsmi_set_gpu_od_volt_info
    amdsmi_dev_firmware_version_get
    amdsmi_dev_firmware_version_get
    amdsmi_dev_name_get
    amdsmi_dev_brand_get
    amdsmi_get_gpu_vram_vendor
    amdsmi_get_gpu_subsystem_name
    amdsmi_get_gpu_vendor_name
    amdsmi_get_gpu_pci_bandwidth
     amdsmi_set_gpu_pci_bandwidth
    amdsmi_get_gpu_pci_throughput
     amdsmi_get_temp_metric
     amdsmi_get_gpu_volt_metric
    amdsmi_get_gpu_fan_speed
    amdsmi_get_gpu_fan_rpms
    amdsmi_reset_gpu_fan
    amdsmi_set_gpu_fan_speed
    amdsmi_get_gpu_fan_speed_max
     amdsmi_get_gpu_od_volt_info
     amdsmi_get_gpu_metrics_info
     amdsmi_get_gpu_od_volt_curve_regions
    amdsmi_dev_power_max_get
    amdsmi_get_power_ave
    amdsmi_dev_power_cap_get
    amdsmi_dev_power_cap_range_get
     amdsmi_set_power_cap
     amdsmi_get_gpu_power_profile_presets
     amdsmi_set_gpu_power_profile
    amdsmi_get_gpu_memory_total
    amdsmi_get_gpu_memory_usage
    amdsmi_dev_vbios_version_get
    amdsmi_dev_serial_number_get
     amdsmi_get_gpu_pci_replay_counter
    amdsmi_dev_unique_id_get
    amdsmi_gpu_create_counter
     amdsmi_get_gpu_available_counters
    amdsmi_gpu_counter_group_supported
    amdsmi_get_gpu_memory_reserved_pages
    amdsmi_gpu_xgmi_error_status
    amdsmi_reset_gpu_xgmi_error
    amdsmi_dev_xgmi_hive_id_get
    amdsmi_topo_get_link_weight
     amdsmi_set_gpu_event_notification_mask
    amdsmi_init_gpu_event_notification
    amdsmi_stop_gpu_event_notification
    */

    IF_VERB(STANDARD) {
      std::cout << "TESTER process: Finished verifying that all "
                "amdsmi_dev_* functions returned AMDSMI_STATUS_BUSY" << std::endl;
    }
    exit(0);
  }
}
