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
#include <map>

#include "gtest/gtest.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi_test/functional/mutual_exclusion.h"
#include "rocm_smi_test/test_common.h"


TestMutualExclusion::TestMutualExclusion() : TestBase() {
  set_title("Mutual Exclusion Test");
  set_description("Verify that RSMI only allows 1 process at a time"
    " to access RSMI resources (primarily sysfs files). This test has one "
    "process that obtains the mutex that ensures only 1 process accesses a "
      "device's sysfs files at a time, and another process that attempts "
      "to access the device's sysfs files. The second process should fail "
      "in these attempts.");
}

TestMutualExclusion::~TestMutualExclusion(void) {
}

extern rsmi_status_t rsmi_test_sleep(uint32_t dv_ind, uint32_t seconds);

void TestMutualExclusion::SetUp(void) {
  std::string label;
  rsmi_status_t ret;

  //   TestBase::SetUp(RSMI_INIT_FLAG_RESRV_TEST1);
  IF_VERB(STANDARD) {
    MakeHeaderStr(kSetupLabel, &label);
    printf("\n\t%s\n", label.c_str());
  }

  sleeper_process_ = false;
  child_ = 0;
  child_ = fork();

  if (child_ != 0) {
    sleeper_process_ = true;  // sleeper_process is parent

    // RSMI_INIT_FLAG_RESRV_TEST1 tells rsmi to fail immediately
    // if it can't get the mutex instead of waiting.
    ret = rsmi_init(RSMI_INIT_FLAG_RESRV_TEST1);
    if (ret != RSMI_STATUS_SUCCESS) {
      setup_failed_ = true;
    }
    ASSERT_EQ(ret, RSMI_STATUS_SUCCESS);

    sleep(2);  // Let both processes get through rsmi_init
  } else {
    sleep(1);  // Let the sleeper process get through rsmi_init() before
              // this one goes, so it doesn't fail.
    ret = rsmi_init(RSMI_INIT_FLAG_RESRV_TEST1);
    if (ret != RSMI_STATUS_SUCCESS) {
      setup_failed_ = true;
    }
    ASSERT_EQ(ret, RSMI_STATUS_SUCCESS);

    sleep(2);  // Let both processes get through rsmi_init;
  }

  ret = rsmi_num_monitor_devices(&num_monitor_devs_);
  if (ret != RSMI_STATUS_SUCCESS) {
    setup_failed_ = true;
  }
  ASSERT_EQ(ret, RSMI_STATUS_SUCCESS);

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
  // rsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

extern rsmi_status_t
rsmi_test_sleep(uint32_t dv_ind, uint32_t seconds);

void TestMutualExclusion::Run(void) {
  rsmi_status_t ret;

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
    ASSERT_EQ(ret, RSMI_STATUS_SUCCESS);
    IF_VERB(STANDARD) {
      std::cout << "MUTEX_HOLDER process: Sleep process woke up." << std::endl;
    }
    pid_t cpid = wait(nullptr);
    ASSERT_EQ(cpid, child_);
  } else {
    // Both processes should have completed rsmi_init().
    // let the other process get started on rsmi_test_sleep().
    sleep(2);
    TestBase::Run();
    IF_VERB(STANDARD) {
      std::cout << "TESTER process: verifing that all rsmi_dev_* functions "
                    "return RSMI_STATUS_BUSY because MUTEX_HOLDER process "
                                               "holds the mutex" << std::endl;
    }
    // Try all the device related rsmi calls. They should all fail with
    // RSMI_STATUS_BUSY
    // Set dummy values should to working, deterministic values.
    uint16_t dmy_ui16 = 0;
    uint32_t dmy_ui32 = 1;
    int32_t dmy_i32 = 0;
    uint64_t dmy_ui64 = 0;
    int64_t dmy_i64 = 0;
    char dmy_str[10];
    rsmi_dev_perf_level_t dmy_perf_lvl;
    rsmi_frequencies_t dmy_freqs{};
    rsmi_od_volt_freq_data_t dmy_od_volt{};
    rsmi_freq_volt_region_t dmy_vlt_reg{};
    rsmi_error_count_t dmy_err_cnt{};
    rsmi_ras_err_state_t dmy_ras_err_st;

    // This can be replaced with ASSERT_EQ() once env. stabilizes
#define CHECK_RET(A, B) { \
  if ((A) != (B)) { \
    std::cout << "Expected return value of " << B << \
                               " but got " << A << std::endl; \
    std::cout << "at " << __FILE__ << ":" << __LINE__ << std::endl; \
  } \
}
    ret = rsmi_dev_id_get(0, &dmy_ui16);
    CHECK_RET(ret, RSMI_STATUS_BUSY);
    ret = rsmi_dev_vendor_id_get(0, &dmy_ui16);
    CHECK_RET(ret, RSMI_STATUS_BUSY);
    ret = rsmi_dev_name_get(0, dmy_str, 10);
    CHECK_RET(ret, RSMI_STATUS_BUSY);
    ret = rsmi_dev_brand_get(0, dmy_str, 10);
    CHECK_RET(ret, RSMI_STATUS_BUSY);
    ret = rsmi_dev_vendor_name_get(0, dmy_str, 10);
    CHECK_RET(ret, RSMI_STATUS_BUSY);
    ret = rsmi_dev_vram_vendor_get(0, dmy_str, 10);
    CHECK_RET(ret, RSMI_STATUS_BUSY);
    ret = rsmi_dev_serial_number_get(0, dmy_str, 10);
    CHECK_RET(ret, RSMI_STATUS_BUSY);
    ret = rsmi_dev_subsystem_id_get(0, &dmy_ui16);
    CHECK_RET(ret, RSMI_STATUS_BUSY);
    ret = rsmi_dev_subsystem_vendor_id_get(0, &dmy_ui16);
    CHECK_RET(ret, RSMI_STATUS_BUSY);
    ret = rsmi_dev_unique_id_get(0, &dmy_ui64);
    CHECK_RET(ret, RSMI_STATUS_BUSY);
    ret = rsmi_dev_pci_id_get(0, &dmy_ui64);
    CHECK_RET(ret, RSMI_STATUS_BUSY);
    ret = rsmi_dev_pci_throughput_get(0, &dmy_ui64, &dmy_ui64, &dmy_ui64);
    CHECK_RET(ret, RSMI_STATUS_BUSY);
    ret = rsmi_dev_pci_replay_counter_get(0, &dmy_ui64);
    CHECK_RET(ret, RSMI_STATUS_BUSY);
    ret = rsmi_dev_pci_bandwidth_set(0, 0);
    CHECK_RET(ret, RSMI_STATUS_BUSY);
    ret = rsmi_dev_fan_rpms_get(0, dmy_ui32, &dmy_i64);
    CHECK_RET(ret, RSMI_STATUS_BUSY);
    ret = rsmi_dev_fan_speed_get(0, 0, &dmy_i64);
    CHECK_RET(ret, RSMI_STATUS_BUSY);
    ret = rsmi_dev_fan_speed_max_get(0, 0, &dmy_ui64);
    CHECK_RET(ret, RSMI_STATUS_BUSY);
    ret = rsmi_dev_temp_metric_get(0, dmy_ui32, RSMI_TEMP_CURRENT, &dmy_i64);
    CHECK_RET(ret, RSMI_STATUS_BUSY);
    ret = rsmi_dev_fan_reset(0, 0);
    CHECK_RET(ret, RSMI_STATUS_BUSY);
    ret = rsmi_dev_fan_speed_set(0, dmy_ui32, 0);
    CHECK_RET(ret, RSMI_STATUS_BUSY);
    ret = rsmi_dev_perf_level_get(0, &dmy_perf_lvl);
    CHECK_RET(ret, RSMI_STATUS_BUSY);
    ret = rsmi_dev_overdrive_level_get(0, &dmy_ui32);
    CHECK_RET(ret, RSMI_STATUS_BUSY);
    ret = rsmi_dev_gpu_clk_freq_get(0, RSMI_CLK_TYPE_SYS, &dmy_freqs);
    CHECK_RET(ret, RSMI_STATUS_BUSY);
    ret = rsmi_dev_od_volt_info_get(0, &dmy_od_volt);
    CHECK_RET(ret, RSMI_STATUS_BUSY);
    ret = rsmi_dev_od_volt_curve_regions_get(0, &dmy_ui32, &dmy_vlt_reg);
    CHECK_RET(ret, RSMI_STATUS_BUSY);
    ret = rsmi_dev_overdrive_level_set_v1(dmy_i32, 0);
    CHECK_RET(ret, RSMI_STATUS_BUSY);
    ret = rsmi_dev_gpu_clk_freq_set(0, RSMI_CLK_TYPE_SYS, 0);
    CHECK_RET(ret, RSMI_STATUS_BUSY);
    ret = rsmi_dev_ecc_count_get(0, RSMI_GPU_BLOCK_UMC, &dmy_err_cnt);
    CHECK_RET(ret, RSMI_STATUS_BUSY);
    ret = rsmi_dev_ecc_enabled_get(0, &dmy_ui64);
    CHECK_RET(ret, RSMI_STATUS_BUSY);
    ret = rsmi_dev_ecc_status_get(0, RSMI_GPU_BLOCK_UMC, &dmy_ras_err_st);
    CHECK_RET(ret, RSMI_STATUS_BUSY);

    /* Other functions holding device mutexes. Listed for reference.
    rsmi_dev_sku_get
    rsmi_dev_perf_level_set_v1
    rsmi_dev_od_clk_info_set
    rsmi_dev_od_volt_info_set
    rsmi_dev_firmware_version_get
    rsmi_dev_firmware_version_get
    rsmi_dev_name_get
    rsmi_dev_brand_get
    rsmi_dev_vram_vendor_get
    rsmi_dev_subsystem_name_get
    rsmi_dev_drm_render_minor_get
    rsmi_dev_vendor_name_get
    rsmi_dev_pci_bandwidth_get
    rsmi_dev_pci_bandwidth_set
    rsmi_dev_pci_throughput_get
    rsmi_dev_temp_metric_get
    rsmi_dev_volt_metric_get
    rsmi_dev_fan_speed_get
    rsmi_dev_fan_rpms_get
    rsmi_dev_fan_reset
    rsmi_dev_fan_speed_set
    rsmi_dev_fan_speed_max_get
    rsmi_dev_od_volt_info_get
    rsmi_dev_gpu_metrics_info_get
    rsmi_dev_od_volt_curve_regions_get
    rsmi_dev_power_max_get
    rsmi_dev_power_ave_get
    rsmi_dev_power_cap_get
    rsmi_dev_power_cap_range_get
    rsmi_dev_power_cap_set
    rsmi_dev_power_profile_presets_get
    rsmi_dev_power_profile_set
    rsmi_dev_memory_total_get
    rsmi_dev_memory_usage_get
    rsmi_dev_memory_busy_percent_get
    rsmi_dev_busy_percent_get
    rsmi_dev_vbios_version_get
    rsmi_dev_serial_number_get
    rsmi_dev_pci_replay_counter_get
    rsmi_dev_unique_id_get
    rsmi_dev_counter_create
    rsmi_counter_available_counters_get
    rsmi_dev_counter_group_supported
    rsmi_dev_memory_reserved_pages_get
    rsmi_dev_xgmi_error_status
    rsmi_dev_xgmi_error_reset
    rsmi_dev_xgmi_hive_id_get
    rsmi_topo_get_link_weight
    rsmi_event_notification_mask_set
    rsmi_event_notification_init
    rsmi_event_notification_stop
    */

    IF_VERB(STANDARD) {
      std::cout << "TESTER process: Finished verifying that all "
                "rsmi_dev_* functions returned RSMI_STATUS_BUSY" << std::endl;
    }
    exit(0);
  }
}
