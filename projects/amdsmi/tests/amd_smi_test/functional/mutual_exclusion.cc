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
#include "mutual_exclusion.h"

#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <string>
#include <vector>

#include "../test_common.h"
#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_test_flags.h"
#include "amd_smi/impl/amd_smi_utils.h"

TestMutualExclusion::TestMutualExclusion() : TestBase() {
  set_title("Mutual Exclusion Test");
  set_description(
      "Verify that AMDSMI only allows 1 process at a time"
      " to access AMDSMI resources (primarily sysfs files). This test has one "
      "process that obtains the mutex that ensures only 1 process accesses a "
      "device's sysfs files at a time, and another process that attempts "
      "to access the device's sysfs files. The second process should fail "
      "in these attempts.");
}

TestMutualExclusion::~TestMutualExclusion(void) {}

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

  // Cross-process shared memory mutex is required for this test.
  // Must be set before fork() so both processes inherit it and amdsmi_init()
  // uses shm_open/mmap instead of a per-process pthread_mutex_t.
  // Save original value so Close() can restore it after the test.
  const char* orig = getenv("AMDSMI_MUTEX_CROSS_PROCESS");
  orig_cross_process_env_ = orig ? orig : "";
  orig_cross_process_env_was_set_ = (orig != nullptr);
  setenv("AMDSMI_MUTEX_CROSS_PROCESS", "1", 1);

  child_ = fork();
  if (child_ < 0) {
    std::cout << "fork() failed: " << strerror(errno) << std::endl;
    setup_failed_ = true;
    return;
  }

  if (child_ != 0) {
    sleeper_process_ = true;  // sleeper_process is parent

    // AMD_SMI_INIT_FLAG_RESRV_TEST1 tells rsmi to fail immediately
    // if it can't get the mutex instead of waiting.
    DISPLAY_AMDSMI_API("[before sleep] amdsmi_init(AMD_SMI_INIT_FLAG_RESRV_TEST1)", "",
                       VERB(STANDARD));
    ret = amdsmi_init(AMDSMI_INIT_AMD_GPUS | AMD_SMI_INIT_FLAG_RESRV_TEST1);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    if (ret != AMDSMI_STATUS_SUCCESS) {
      setup_failed_ = true;
    }
    ASSERT_EQ(ret, AMDSMI_STATUS_SUCCESS);

    sleep(2);  // Let both processes get through amdsmi_init
  } else {
    sleep(1);  // Let the sleeper process get through amdsmi_init() before
               // this one goes, so it doesn't fail.
    DISPLAY_AMDSMI_API("[after sleep] amdsmi_init(AMD_SMI_INIT_FLAG_RESRV_TEST1)", "",
                       VERB(STANDARD));
    ret = amdsmi_init(AMDSMI_INIT_AMD_GPUS | AMD_SMI_INIT_FLAG_RESRV_TEST1);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    if (ret != AMDSMI_STATUS_SUCCESS) {
      setup_failed_ = true;
    }
    ASSERT_EQ(ret, AMDSMI_STATUS_SUCCESS);

    sleep(2);  // Let both processes get through amdsmi_init;
  }

  // Enumerate sockets and processor handles in each process after amdsmi_init()
  amdsmi_status_t enum_err = amdsmi_get_socket_handles(&socket_count_, nullptr);
  if (enum_err != AMDSMI_STATUS_SUCCESS) {
    setup_failed_ = true;
    return;
  }
  sockets_.resize(socket_count_);
  enum_err = amdsmi_get_socket_handles(&socket_count_, &sockets_[0]);
  if (enum_err != AMDSMI_STATUS_SUCCESS) {
    setup_failed_ = true;
    return;
  }
  num_monitor_devs_ = 0;
  for (uint32_t i = 0; i < socket_count_; i++) {
    uint32_t device_count = 0;
    amdsmi_status_t status = amdsmi_get_processor_handles(sockets_[i], &device_count, nullptr);
    if (status != AMDSMI_STATUS_SUCCESS || device_count == 0) {
      continue;
    }
    std::vector<amdsmi_processor_handle> handles(device_count);
    status = amdsmi_get_processor_handles(sockets_[i], &device_count, &handles[0]);
    if (status != AMDSMI_STATUS_SUCCESS) {
      continue;
    }
    for (uint32_t j = 0; j < device_count && num_monitor_devs_ < MAX_MONITOR_DEVICES; j++) {
      processor_handles_[num_monitor_devs_++] = handles[j];
    }
  }

  if (num_monitor_devs_ == 0) {
    std::cout << "No monitor devices found on this machine." << std::endl;
    std::cout << "No ROCm SMI tests can be run." << std::endl;
    setup_failed_ = true;
  }

  return;
}

void TestMutualExclusion::DisplayTestInfo(void) {
  IF_VERB(STANDARD) { TestBase::DisplayTestInfo(); }
}

void TestMutualExclusion::DisplayResults(void) const {
  IF_VERB(STANDARD) { TestBase::DisplayResults(); }
  return;
}

void TestMutualExclusion::Close() {
  // Shut down first while AMDSMI_MUTEX_CROSS_PROCESS is still set, so that
  // shared_mutex_close() correctly munmaps the shared-memory mutex instead of
  // calling delete on it (which would crash with "free(): invalid pointer").
  TestBase::Close();

  // Restore AMDSMI_MUTEX_CROSS_PROCESS to its original state so subsequent
  // tests in this process are not affected.
  if (orig_cross_process_env_was_set_) {
    setenv("AMDSMI_MUTEX_CROSS_PROCESS", orig_cross_process_env_.c_str(), 1);
  } else {
    unsetenv("AMDSMI_MUTEX_CROSS_PROCESS");
  }
}

extern amdsmi_status_t rsmi_test_sleep(uint32_t dv_ind, uint32_t seconds);

void TestMutualExclusion::Run(void) {
  amdsmi_status_t ret;

  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  if (sleeper_process_) {
    // Block SIGCHLD so that when the child exits, the signal does not interrupt
    // sleep() inside rsmi_test_sleep and cause the mutex to be released early.
    sigset_t sigchld_mask, old_mask;
    sigemptyset(&sigchld_mask);
    sigaddset(&sigchld_mask, SIGCHLD);
    sigprocmask(SIG_BLOCK, &sigchld_mask, &old_mask);

    IF_VERB(STANDARD) {
      std::cout << "MUTEX_HOLDER process: started sleeping for 10 seconds..." << std::endl;
    }
    ret = rsmi_test_sleep(0, 10);
    ASSERT_EQ(ret, AMDSMI_STATUS_SUCCESS);
    IF_VERB(STANDARD) { std::cout << "MUTEX_HOLDER process: Sleep process woke up." << std::endl; }

    // Restore signal mask before wait() so SIGCHLD can be delivered and the
    // child can be reaped normally.
    sigprocmask(SIG_SETMASK, &old_mask, nullptr);
    int child_status = 0;
    pid_t cpid = wait(&child_status);
    ASSERT_EQ(cpid, child_);
    EXPECT_EQ(WEXITSTATUS(child_status), 0) << "TESTER child process reported CHECK_RET failure(s)";
  } else {
    // Both processes should have completed amdsmi_init().
    // let the other process get started on rsmi_test_sleep().
    sleep(2);
    TestBase::Run();
    IF_VERB(STANDARD) {
      std::cout << "TESTER process: verifing that all amdsmi_dev_* functions "
                   "return AMDSMI_STATUS_BUSY because MUTEX_HOLDER process "
                   "holds the mutex"
                << std::endl;
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

// Accepts one or more expected values; passes if actual matches any of them.
#define CHECK_RET(retVal, ...)                                                             \
  {                                                                                        \
    auto _tst_ret_val = (retVal);                                                          \
    std::initializer_list<decltype(_tst_ret_val)> _expected_returns = {__VA_ARGS__};       \
    bool _chk_matched = false;                                                             \
    for (auto _chk_e : _expected_returns) {                                                \
      if (_tst_ret_val == _chk_e) {                                                        \
        _chk_matched = true;                                                               \
        break;                                                                             \
      }                                                                                    \
    }                                                                                      \
    if (!_chk_matched) {                                                                   \
      std::cout << "Expected return value of one of {";                                    \
      bool _chk_first = true;                                                              \
      for (auto _chk_e : _expected_returns) {                                              \
        if (!_chk_first) std::cout << ", ";                                                \
        std::string status = smi_amdgpu_get_status_string(_chk_e, false);                  \
        std::cout << status << " (" << _chk_e << ")";                                      \
        _chk_first = false;                                                                \
      }                                                                                    \
      std::string ret_status = smi_amdgpu_get_status_string(_tst_ret_val, false);          \
      std::cout << "} but got " << ret_status << " (" << _tst_ret_val << ")" << std::endl; \
      std::cout << "at " << __FILE__ << ":" << __LINE__ << std::endl;                      \
    }                                                                                      \
    EXPECT_TRUE(_chk_matched);                                                             \
  }
    // TODO(amdsmi_team): Add more device calls here, we should also check how CPU/NIC/etc handle
    // These are APIs which either need to include:
    // DEVICE_MUTEX or SMIGPUDEVICE_MUTEX (refer to AMDSMI_MUTEX_CROSS_PROCESS references)
    // to safely handle multithreaded/process access, or need to be checked for AMDSMI_STATUS_BUSY
    // if the mutex is held by another process. These are not exhaustive lists, just examples.
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_id", "0", VERB(STANDARD));
    ret = amdsmi_get_gpu_id(processor_handles_[0], &dmy_ui16);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_BUSY);

    DISPLAY_AMDSMI_API("amdsmi_get_temp_metric", "0", VERB(STANDARD));
    ret = amdsmi_get_temp_metric(processor_handles_[0], AMDSMI_TEMPERATURE_TYPE_EDGE,
                                 AMDSMI_TEMP_CURRENT, &dmy_i64);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_BUSY);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);

    // vendor_id, unique_id
    amdsmi_asic_info_t asic_info;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_asic_info", "0", VERB(STANDARD));
    ret = amdsmi_get_gpu_asic_info(processor_handles_[0], &asic_info);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_BUSY);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);

    // device name, brand, serial_number
    amdsmi_board_info_t board_info;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_board_info", "0", VERB(STANDARD));
    ret = amdsmi_get_gpu_board_info(processor_handles_[0], &board_info);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_BUSY);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);

    DISPLAY_AMDSMI_API("amdsmi_get_gpu_vendor_name", "0", VERB(STANDARD));
    ret = amdsmi_get_gpu_vendor_name(processor_handles_[0], dmy_str, 10);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_BUSY);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);

    DISPLAY_AMDSMI_API("amdsmi_get_gpu_vram_vendor", "0", VERB(STANDARD));
    ret = amdsmi_get_gpu_vram_vendor(processor_handles_[0], dmy_str, 10);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_BUSY);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);

    DISPLAY_AMDSMI_API("amdsmi_get_gpu_subsystem_id", "0", VERB(STANDARD));
    ret = amdsmi_get_gpu_subsystem_id(processor_handles_[0], &dmy_ui16);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_BUSY);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);

    DISPLAY_AMDSMI_API("amdsmi_get_gpu_bdf_id", "0", VERB(STANDARD));
    ret = amdsmi_get_gpu_bdf_id(processor_handles_[0], &dmy_ui64);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_BUSY);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);

    DISPLAY_AMDSMI_API("amdsmi_get_gpu_pci_throughput", "0", VERB(STANDARD));
    ret = amdsmi_get_gpu_pci_throughput(processor_handles_[0], &dmy_ui64, &dmy_ui64, &dmy_ui64);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_BUSY);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);

    DISPLAY_AMDSMI_API("amdsmi_get_gpu_pci_replay_counter", "0", VERB(STANDARD));
    ret = amdsmi_get_gpu_pci_replay_counter(processor_handles_[0], &dmy_ui64);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_BUSY);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);

    DISPLAY_AMDSMI_API("amdsmi_set_gpu_pci_bandwidth", "0", VERB(STANDARD));
    ret = amdsmi_set_gpu_pci_bandwidth(processor_handles_[0], 0);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_BUSY,
                          AMDSMI_STATUS_NO_PERM);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY, AMDSMI_STATUS_NO_PERM);

    DISPLAY_AMDSMI_API("amdsmi_get_gpu_fan_rpms", "0", VERB(STANDARD));
    ret = amdsmi_get_gpu_fan_rpms(processor_handles_[0], dmy_ui32, &dmy_i64);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_BUSY);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);

    DISPLAY_AMDSMI_API("amdsmi_get_gpu_fan_speed", "0", VERB(STANDARD));
    ret = amdsmi_get_gpu_fan_speed(processor_handles_[0], 0, &dmy_i64);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_BUSY);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);

    DISPLAY_AMDSMI_API("amdsmi_get_gpu_fan_speed_max", "0", VERB(STANDARD));
    ret = amdsmi_get_gpu_fan_speed_max(processor_handles_[0], 0, &dmy_ui64);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_BUSY);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);

    DISPLAY_AMDSMI_API("amdsmi_reset_gpu_fan", "0", VERB(STANDARD));
    ret = amdsmi_reset_gpu_fan(processor_handles_[0], 0);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_BUSY,
                          AMDSMI_STATUS_NO_PERM);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY, AMDSMI_STATUS_NO_PERM);

    DISPLAY_AMDSMI_API("amdsmi_set_gpu_fan_speed", "0", VERB(STANDARD));
    ret = amdsmi_set_gpu_fan_speed(processor_handles_[0], dmy_ui32, 0);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_BUSY,
                          AMDSMI_STATUS_NO_PERM);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY, AMDSMI_STATUS_NO_PERM);

    DISPLAY_AMDSMI_API("amdsmi_get_gpu_perf_level", "0", VERB(STANDARD));
    ret = amdsmi_get_gpu_perf_level(processor_handles_[0], &dmy_perf_lvl);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_BUSY);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);

    DISPLAY_AMDSMI_API("amdsmi_get_gpu_overdrive_level", "0", VERB(STANDARD));
    ret = amdsmi_get_gpu_overdrive_level(processor_handles_[0], &dmy_ui32);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_BUSY);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);

    DISPLAY_AMDSMI_API("amdsmi_get_clk_freq", "0", VERB(STANDARD));
    ret = amdsmi_get_clk_freq(processor_handles_[0], AMDSMI_CLK_TYPE_SYS, &dmy_freqs);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_BUSY);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);

    DISPLAY_AMDSMI_API("amdsmi_get_gpu_od_volt_info", "0", VERB(STANDARD));
    ret = amdsmi_get_gpu_od_volt_info(processor_handles_[0], &dmy_od_volt);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_BUSY);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);

    DISPLAY_AMDSMI_API("amdsmi_get_gpu_od_volt_curve_regions", "0", VERB(STANDARD));
    ret = amdsmi_get_gpu_od_volt_curve_regions(processor_handles_[0], &dmy_ui32, &dmy_vlt_reg);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_BUSY);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);

    DISPLAY_AMDSMI_API("amdsmi_set_clk_freq", "0", VERB(STANDARD));
    ret = amdsmi_set_clk_freq(processor_handles_[0], AMDSMI_CLK_TYPE_SYS, 0);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_BUSY,
                          AMDSMI_STATUS_NO_PERM);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY, AMDSMI_STATUS_NO_PERM);

    DISPLAY_AMDSMI_API("amdsmi_get_gpu_ecc_count", "0", VERB(STANDARD));
    ret = amdsmi_get_gpu_ecc_count(processor_handles_[0], AMDSMI_GPU_BLOCK_UMC, &dmy_err_cnt);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_BUSY);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);

    DISPLAY_AMDSMI_API("amdsmi_get_gpu_ecc_enabled", "0", VERB(STANDARD));
    ret = amdsmi_get_gpu_ecc_enabled(processor_handles_[0], &dmy_ui64);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_BUSY);
    CHECK_RET(ret, AMDSMI_STATUS_BUSY);

    DISPLAY_AMDSMI_API("amdsmi_get_gpu_ecc_status", "0", VERB(STANDARD));
    ret = amdsmi_get_gpu_ecc_status(processor_handles_[0], AMDSMI_GPU_BLOCK_UMC, &dmy_ras_err_st);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_BUSY);
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
                   "amdsmi_dev_* functions returned AMDSMI_STATUS_BUSY"
                << std::endl;
    }
    std::cout.flush();
    // Use _exit() to terminate immediately without running atexit handlers or
    // static destructors — exit() would trigger gtest/RocmSMI cleanup that
    // was never meant to run in the child process, causing heap corruption.
    // Exit with 1 if any EXPECT_* in this process failed, so the parent can
    // detect and report the failure via WEXITSTATUS.
    _exit(::testing::Test::HasFailure() ? 1 : 0);
  }
}
