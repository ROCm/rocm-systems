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
#include <gtest/gtest.h>

#include <cstdlib>

#include "amd_smi/impl/amd_smi_utils.h"
#include "functional/gpu/clock/frequencies_read.h"
#include "functional/gpu/clock/frequencies_read_write.h"
#include "functional/gpu/events/evt_notif_read_write.h"
#include "functional/gpu/identity/id_info_read.h"
#include "functional/gpu/identity/version_read.h"
#include "functional/gpu/memory/mem_page_info_read.h"
#include "functional/gpu/memory/mem_util_read.h"
#include "functional/gpu/memory/memory_read_write.h"
#include "functional/gpu/metrics/gpu_busy_read.h"
#include "functional/gpu/metrics/gpu_cache_read.h"
#include "functional/gpu/metrics/gpu_metrics_read.h"
#include "functional/gpu/metrics/gpu_partition_metrics_read.h"
#include "functional/gpu/metrics/metrics_counter_read.h"
#include "functional/gpu/metrics/process_info_read.h"
#include "functional/gpu/partition/computepartition_memallocmode_read_write.h"
#include "functional/gpu/partition/computepartition_read_write.h"
#include "functional/gpu/partition/memorypartition_read_write.h"
#include "functional/gpu/pci/pci_read_write.h"
#include "functional/gpu/perf/overdrive_read.h"
#include "functional/gpu/perf/overdrive_read_write.h"
#include "functional/gpu/perf/perf_cntr_read_write.h"
#include "functional/gpu/perf/perf_determinism.h"
#include "functional/gpu/perf/perf_level_read.h"
#include "functional/gpu/perf/perf_level_read_write.h"
#include "functional/gpu/perf/volt_freq_curv_read.h"
#include "functional/gpu/perf/volt_read.h"
#include "functional/gpu/power/power_cap_read_write.h"
#include "functional/gpu/power/power_read.h"
#include "functional/gpu/power/power_read_write.h"
#include "functional/gpu/ras/err_cnt_read.h"
#include "functional/gpu/thermal/fan_read.h"
#include "functional/gpu/thermal/fan_read_write.h"
#include "functional/gpu/thermal/temp_read.h"
#include "functional/gpu/xgmi/xgmi_read_write.h"
#include "functional/ifoe/fabric/fabric_read.h"
#include "functional/ifoe/identity/ifoe_info_read.h"
#include "functional/system/cross_process_serialization.h"
#include "functional/system/hw_topology_read.h"
#include "functional/system/init_shutdown_refcount.h"
#include "functional/system/kfd_atfork_read.h"
#include "functional/system/mutual_exclusion.h"
#include "functional/system/sys_info_read.h"
#include "rocm_smi/rocm_smi_utils.h"
#include "test_base.h"
#include "test_common.h"

static AMDSMITstGlobals* sRSMIGlvalues = nullptr;

static void SetFlags(TestBase* test) {
  assert(sRSMIGlvalues != nullptr);

  test->set_verbosity(sRSMIGlvalues->verbosity);
  test->set_dont_fail(sRSMIGlvalues->dont_fail);
  test->set_init_options(sRSMIGlvalues->init_options);
  test->set_num_iterations(sRSMIGlvalues->num_iterations);
}

static void RunCustomTestProlog(TestBase* test) {
  SetFlags(test);

  if (sRSMIGlvalues->verbosity >= TestBase::VERBOSE_STANDARD) {
    test->DisplayTestInfo();
  }
  test->SetUp();
  test->Run();
}
static void RunCustomTestEpilog(TestBase* tst) {
  if (sRSMIGlvalues->verbosity >= TestBase::VERBOSE_STANDARD) {
    tst->DisplayResults();
  }
  tst->Close();
}

// If the test case one big test, you should use RunGenericTest()
// to run the test case. OTOH, if the test case consists of multiple
// functions to be run as separate tests, follow this pattern:
//   * RunCustomTestProlog(test)  // Run() should contain minimal code
//   * <insert call to actual test function within test case>
//   * RunCustomTestEpilog(test)
static void RunGenericTest(TestBase* test) {
  RunCustomTestProlog(test);
  RunCustomTestEpilog(test);
}

// TEST ENTRY TEMPLATE:
// TEST(rocrtst, Perf_<test name>) {
//  <Test Implementation class> <test_obj>;
//
//  // Copy and modify implementation of RunGenericTest() if you need to deviate
//  // from the standard pattern implemented there.
//  RunGenericTest(&<test_obj>);
// }
TEST(GpuFunctionalReadOnly, TestVersionRead) {
  TestVersionRead tst;
  RunGenericTest(&tst);
}
TEST(GpuFunctionalReadOnly, FanRead) {
  TestFanRead tst;
  RunGenericTest(&tst);
}
TEST(GpuFunctionalReadWrite, FanReadWrite) {
  if (std::getenv("AMDSMI_NON_PRIVILEGED")) GTEST_SKIP_("Skipped in non-privileged mode");
  if (!amd::smi::is_sudo_user()) GTEST_SKIP_("Invalid permission - Must run as super user");
  TestFanReadWrite tst;
  RunGenericTest(&tst);
}
TEST(GpuFunctionalReadOnly, TempRead) {
  TestTempRead tst;
  RunGenericTest(&tst);
}
TEST(GpuFunctionalReadOnly, VoltRead) {
  TestVoltRead tst;
  RunGenericTest(&tst);
}
TEST(GpuFunctionalReadOnly, TestVoltCurvRead) {
  TestVoltCurvRead tst;
  RunGenericTest(&tst);
}
TEST(GpuFunctionalReadOnly, TestPerfLevelRead) {
  if (amd::smi::is_vm_guest()) GTEST_SKIP();
  TestPerfLevelRead tst;
  RunGenericTest(&tst);
}
TEST(GpuFunctionalReadWrite, TestPerfLevelReadWrite) {
  if (std::getenv("AMDSMI_NON_PRIVILEGED")) GTEST_SKIP_("Skipped in non-privileged mode");
  if (amd::smi::is_vm_guest()) GTEST_SKIP();
  if (!amd::smi::is_sudo_user()) GTEST_SKIP_("Invalid permission - Must run as super user");
  TestPerfLevelReadWrite tst;
  RunGenericTest(&tst);
}
TEST(GpuFunctionalReadOnly, TestOverdriveRead) {
  TestOverdriveRead tst;
  RunGenericTest(&tst);
}
TEST(GpuFunctionalReadWrite, TestOverdriveReadWrite) {
  if (std::getenv("AMDSMI_NON_PRIVILEGED")) GTEST_SKIP_("Skipped in non-privileged mode");
  if (!amd::smi::is_sudo_user()) GTEST_SKIP_("Invalid permission - Must run as super user");
  TestOverdriveReadWrite tst;
  RunGenericTest(&tst);
}
TEST(GpuFunctionalReadOnly, TestFrequenciesRead) {
  TestFrequenciesRead tst;
  RunGenericTest(&tst);
}
TEST(GpuFunctionalReadWrite, TestFrequenciesReadWrite) {
  if (std::getenv("AMDSMI_NON_PRIVILEGED")) GTEST_SKIP_("Skipped in non-privileged mode");
  if (!amd::smi::is_sudo_user()) GTEST_SKIP_("Invalid permission - Must run as super user");
  TestFrequenciesReadWrite tst;
  RunGenericTest(&tst);
}
TEST(GpuFunctionalReadWrite, TestPciReadWrite) {
  if (std::getenv("AMDSMI_NON_PRIVILEGED")) GTEST_SKIP_("Skipped in non-privileged mode");
  if (amd::smi::is_vm_guest()) GTEST_SKIP();
  if (!amd::smi::is_sudo_user()) GTEST_SKIP_("Invalid permission - Must run as super user");
  TestPciReadWrite tst;
  RunGenericTest(&tst);
}
TEST(GpuFunctionalReadOnly, TestSysInfoRead) {
  if (amd::smi::is_vm_guest()) GTEST_SKIP();
  TestSysInfoRead tst;
  RunGenericTest(&tst);
}
TEST(GpuFunctionalReadOnly, TestGPUBusyRead) {
  TestGPUBusyRead tst;
  RunGenericTest(&tst);
}
TEST(GpuFunctionalReadOnly, TestPowerRead) {
  if (amd::smi::is_vm_guest()) GTEST_SKIP();
  TestPowerRead tst;
  RunGenericTest(&tst);
}
TEST(GpuFunctionalReadWrite, TestPowerReadWrite) {
  if (std::getenv("AMDSMI_NON_PRIVILEGED")) GTEST_SKIP_("Skipped in non-privileged mode");
  if (amd::smi::is_vm_guest()) GTEST_SKIP();
  if (!amd::smi::is_sudo_user()) GTEST_SKIP_("Invalid permission - Must run as super user");
  TestPowerReadWrite tst;
  RunGenericTest(&tst);
}
TEST(GpuFunctionalReadWrite, TestPowerCapReadWrite) {
  if (std::getenv("AMDSMI_NON_PRIVILEGED")) GTEST_SKIP_("Skipped in non-privileged mode");
  if (amd::smi::is_vm_guest()) GTEST_SKIP();
  if (!amd::smi::is_sudo_user()) GTEST_SKIP_("Invalid permission - Must run as super user");
  TestPowerCapReadWrite tst;
  RunGenericTest(&tst);
}
TEST(GpuFunctionalReadOnly, TestErrCntRead) {
  TestErrCntRead tst;
  RunGenericTest(&tst);
}
TEST(GpuFunctionalReadOnly, TestMemUtilRead) {
  TestMemUtilRead tst;
  RunGenericTest(&tst);
}
TEST(GpuFunctionalReadOnly, TestIdInfoRead) {
  if (amd::smi::is_vm_guest()) GTEST_SKIP();
  TestIdInfoRead tst;
  RunGenericTest(&tst);
}
TEST(GpuFunctionalReadWrite, TestPerfCntrReadWrite) {
  if (std::getenv("AMDSMI_NON_PRIVILEGED")) GTEST_SKIP_("Skipped in non-privileged mode");
  if (!amd::smi::is_sudo_user()) GTEST_SKIP_("Invalid permission - Must run as super user");
  TestPerfCntrReadWrite tst;
  RunGenericTest(&tst);
}
TEST(GpuFunctionalReadOnly, TestProcInfoRead) {
  TestProcInfoRead tst;
  RunGenericTest(&tst);
}
TEST(GpuFunctionalReadOnly, TestHWTopologyRead) {
  TestHWTopologyRead tst;
  RunGenericTest(&tst);
}
TEST(GpuFunctionalReadOnly, TestGpuMetricsRead) {
  TestGpuMetricsRead tst;
  RunGenericTest(&tst);
}
TEST(GpuFunctionalReadOnly, TestGpuPartitionMetricsRead) {
  TestGpuPartitionMetricsRead tst;
  RunGenericTest(&tst);
}
TEST(GpuFunctionalReadOnly, TestMetricsCounterRead) {
  TestMetricsCounterRead tst;
  RunGenericTest(&tst);
}
TEST(GpuFunctionalReadWrite, TestPerfDeterminism) {
  if (std::getenv("AMDSMI_NON_PRIVILEGED")) GTEST_SKIP_("Skipped in non-privileged mode");
  if (!amd::smi::is_sudo_user()) GTEST_SKIP_("Invalid permission - Must run as super user");
  TestPerfDeterminism tst;
  RunGenericTest(&tst);
}
TEST(GpuFunctionalReadWrite, TestXGMIReadWrite) {
  if (std::getenv("AMDSMI_NON_PRIVILEGED")) GTEST_SKIP_("Skipped in non-privileged mode");
  if (!amd::smi::is_sudo_user()) GTEST_SKIP_("Invalid permission - Must run as super user");
  TestXGMIReadWrite tst;
  RunGenericTest(&tst);
}
TEST(GpuFunctionalReadOnly, TestMemPageInfoRead) {
  TestMemPageInfoRead tst;
  RunGenericTest(&tst);
}

TEST(GpuFunctionalReadOnly, TestMutualExclusion) {
  TestMutualExclusion tst;
  SetFlags(&tst);
  tst.DisplayTestInfo();
  tst.SetUp();
  tst.Run();
  RunCustomTestEpilog(&tst);
}

TEST(GpuFunctionalReadWrite, TestComputePartitionReadWrite) {
  if (std::getenv("AMDSMI_NON_PRIVILEGED")) GTEST_SKIP_("Skipped in non-privileged mode");
  if (!amd::smi::is_sudo_user()) GTEST_SKIP_("Invalid permission - Must run as super user");
  TestComputePartitionReadWrite tst;
  RunGenericTest(&tst);
}

TEST(GpuFunctionalReadWrite, TestComputePartitionMemAllocModeReadWrite) {
  if (!amd::smi::is_sudo_user()) GTEST_SKIP_("Invalid permission - Must run as super user");
  TestComputePartitionMemAllocModeReadWrite tst;
  RunGenericTest(&tst);
}

TEST(GpuFunctionalReadWrite, TestMemoryPartitionReadWrite) {
  if (std::getenv("AMDSMI_NON_PRIVILEGED")) GTEST_SKIP_("Skipped in non-privileged mode");
  if (!amd::smi::is_sudo_user()) GTEST_SKIP_("Invalid permission - Must run as super user");
  TestMemoryPartitionReadWrite tst;
  RunGenericTest(&tst);
}

TEST(GpuFunctionalReadWrite, TestEvtNotifReadWrite) {
  if (std::getenv("AMDSMI_NON_PRIVILEGED")) GTEST_SKIP_("Skipped in non-privileged mode");
  if (!amd::smi::is_sudo_user()) GTEST_SKIP_("Invalid permission - Must run as super user");
  TestEvtNotifReadWrite tst;
  RunGenericTest(&tst);
}

TEST(GpuFunctionalReadOnly, TestGPUCacheRead) {
  TestGPUCacheRead tst;
  RunGenericTest(&tst);
}

TEST(GpuFunctionalReadWrite, TestMemoryReadWrite) {
  if (std::getenv("AMDSMI_NON_PRIVILEGED")) GTEST_SKIP_("Skipped in non-privileged mode");
  TestMemoryReadWrite tst;
  RunGenericTest(&tst);
}

TEST(GpuFunctionalReadOnly, TestKfdAtforkRead) {
  TestKfdAtforkRead tst;
  RunGenericTest(&tst);
}

TEST(GpuFunctionalReadOnly, TestFabricRead) {
  TestFabricRead tst;
  RunGenericTest(&tst);
}

TEST(GpuFunctionalReadOnly, TestIfoeInfoRead) {
  TestIfoeInfoRead tst;
  RunGenericTest(&tst);
}

TEST(GpuFunctionalReadOnly, TestCrossProcessSerialization) {
  TestCrossProcessSerialization tst;
  SetFlags(&tst);
  tst.DisplayTestInfo();
  tst.SetUp();
  tst.Run();
  RunCustomTestEpilog(&tst);
}
/*
TEST(GpuFunctionalReadOnly, TestConcurrentInit) {
  TestConcurrentInit tst;
  SetFlags(&tst);
  tst.DisplayTestInfo();
  //  tst.SetUp();   // Avoid extra amdsmi_init
  tst.Run();
  // RunCustomTestEpilog(&tst);  // Avoid extra amdsmi_shut_down
  tst.DisplayResults();
}
*/

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  AMDSMITstGlobals settings;

  // Set some default values
  settings.verbosity = 1;
  settings.monitor_verbosity = 1;
  settings.num_iterations = 1;
  settings.dont_fail = false;
  settings.init_options = 0;

  if (ProcessCmdline(&settings, argc, argv)) {
    return 1;
  }

  sRSMIGlvalues = &settings;
  SetTestVerbosity(settings.verbosity);
  return RUN_ALL_TESTS();
}
