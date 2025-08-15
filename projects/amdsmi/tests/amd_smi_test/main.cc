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


#include "rocm_smi/rocm_smi_utils.h"
#include "amd_smi/impl/amd_smi_utils.h"
#include "test_common.h"
#include "test_base.h"

#include "functional/fan_read.h"
#include "functional/fan_read_write.h"
#include "functional/evt_notif_read_write.h"
#include "functional/perf_cntr_read_write.h"
#include "functional/hw_topology_read.h"
#include "functional/xgmi_read_write.h"
#include "functional/api_support_read.h"
#include "functional/process_info_read.h"
#include "functional/gpu_busy_read.h"
#include "functional/gpu_metrics_read.h"
#include "functional/err_cnt_read.h"
#include "functional/power_read.h"
#include "functional/power_read_write.h"
#include "functional/power_cap_read_write.h"
#include "functional/mem_util_read.h"
#include "functional/mem_page_info_read.h"
#include "functional/frequencies_read.h"
#include "functional/frequencies_read_write.h"
#include "functional/overdrive_read.h"
#include "functional/overdrive_read_write.h"
#include "functional/temp_read.h"
#include "functional/volt_read.h"
#include "functional/volt_freq_curv_read.h"
#include "functional/perf_level_read.h"
#include "functional/perf_level_read_write.h"
#include "functional/pci_read_write.h"
#include "functional/perf_determinism.h"
#include "functional/sys_info_read.h"
#include "functional/id_info_read.h"
#include "functional/metrics_counter_read.h"
#include "functional/version_read.h"
#include "functional/memorypartition_read_write.h"
#include "functional/computepartition_read_write.h"
#include "functional/gpu_cache_read.h"

static AMDSMITstGlobals *sRSMIGlvalues = nullptr;

static void SetFlags(TestBase *test) {
  assert(sRSMIGlvalues != nullptr);

  test->set_verbosity(sRSMIGlvalues->verbosity);
  test->set_dont_fail(sRSMIGlvalues->dont_fail);
  test->set_init_options(sRSMIGlvalues->init_options);
  test->set_num_iterations(sRSMIGlvalues->num_iterations);
}

static void RunCustomTestProlog(TestBase *test) {
  SetFlags(test);

  if (sRSMIGlvalues->verbosity >= TestBase::VERBOSE_STANDARD) {
    test->DisplayTestInfo();
  }
  test->SetUp();
  test->Run();
}
static void RunCustomTestEpilog(TestBase *tst) {
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
static void RunGenericTest(TestBase *test) {
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
TEST(amdsmitstReadOnly, TestVersionRead) {
  TestVersionRead tst;
  RunGenericTest(&tst);
}
TEST(amdsmitstReadOnly, FanRead) {
  TestFanRead tst;
  RunGenericTest(&tst);
}
TEST(amdsmitstReadWrite, FanReadWrite) {
  if (!amd::smi::is_sudo_user()) GTEST_SKIP_("Invalid permission - Must run as super user");
  TestFanReadWrite tst;
  RunGenericTest(&tst);
}
TEST(amdsmitstReadOnly, TempRead) {
  TestTempRead tst;
  RunGenericTest(&tst);
}
TEST(amdsmitstReadOnly, VoltRead) {
  TestVoltRead tst;
  RunGenericTest(&tst);
}
TEST(amdsmitstReadOnly, TestVoltCurvRead) {
  TestVoltCurvRead tst;
  RunGenericTest(&tst);
}
TEST(amdsmitstReadOnly, TestPerfLevelRead) {
  if (amd::smi::is_vm_guest()) GTEST_SKIP();
  TestPerfLevelRead tst;
  RunGenericTest(&tst);
}
TEST(amdsmitstReadWrite, TestPerfLevelReadWrite) {
  if (amd::smi::is_vm_guest()) GTEST_SKIP();
  if (!amd::smi::is_sudo_user()) GTEST_SKIP_("Invalid permission - Must run as super user");
  TestPerfLevelReadWrite tst;
  RunGenericTest(&tst);
}
TEST(amdsmitstReadOnly, TestOverdriveRead) {
  TestOverdriveRead tst;
  RunGenericTest(&tst);
}
TEST(amdsmitstReadWrite, TestOverdriveReadWrite) {
  if (!amd::smi::is_sudo_user()) GTEST_SKIP_("Invalid permission - Must run as super user");
  TestOverdriveReadWrite tst;
  RunGenericTest(&tst);
}
TEST(amdsmitstReadOnly, TestFrequenciesRead) {
  TestFrequenciesRead tst;
  RunGenericTest(&tst);
}
TEST(amdsmitstReadWrite, TestFrequenciesReadWrite) {
  if (!amd::smi::is_sudo_user()) GTEST_SKIP_("Invalid permission - Must run as super user");
  TestFrequenciesReadWrite tst;
  RunGenericTest(&tst);
}
TEST(amdsmitstReadWrite, TestPciReadWrite) {
  if (amd::smi::is_vm_guest()) GTEST_SKIP();
  if (!amd::smi::is_sudo_user()) GTEST_SKIP_("Invalid permission - Must run as super user");
  TestPciReadWrite tst;
  RunGenericTest(&tst);
}
TEST(amdsmitstReadOnly, TestSysInfoRead) {
  if (amd::smi::is_vm_guest()) GTEST_SKIP();
  TestSysInfoRead tst;
  RunGenericTest(&tst);
}
TEST(amdsmitstReadOnly, TestGPUBusyRead) {
  TestGPUBusyRead tst;
  RunGenericTest(&tst);
}
TEST(amdsmitstReadOnly, TestPowerRead) {
  if (amd::smi::is_vm_guest()) GTEST_SKIP();
  TestPowerRead tst;
  RunGenericTest(&tst);
}
TEST(amdsmitstReadWrite, TestPowerReadWrite) {
  if (amd::smi::is_vm_guest()) GTEST_SKIP();
  if (!amd::smi::is_sudo_user()) GTEST_SKIP_("Invalid permission - Must run as super user");
  TestPowerReadWrite tst;
  RunGenericTest(&tst);
}
TEST(amdsmitstReadWrite, TestPowerCapReadWrite) {
  if (amd::smi::is_vm_guest()) GTEST_SKIP();
  if (!amd::smi::is_sudo_user()) GTEST_SKIP_("Invalid permission - Must run as super user");
  TestPowerCapReadWrite tst;
  RunGenericTest(&tst);
}
TEST(amdsmitstReadOnly, TestErrCntRead) {
  TestErrCntRead tst;
  RunGenericTest(&tst);
}
TEST(amdsmitstReadOnly, TestMemUtilRead) {
  TestMemUtilRead tst;
  RunGenericTest(&tst);
}
TEST(amdsmitstReadOnly, TestIdInfoRead) {
  if (amd::smi::is_vm_guest()) GTEST_SKIP();
  TestIdInfoRead tst;
  RunGenericTest(&tst);
}
TEST(amdsmitstReadWrite, TestPerfCntrReadWrite) {
  if (!amd::smi::is_sudo_user()) GTEST_SKIP_("Invalid permission - Must run as super user");
  TestPerfCntrReadWrite tst;
  RunGenericTest(&tst);
}
TEST(amdsmitstReadOnly, TestProcInfoRead) {
  TestProcInfoRead tst;
  RunGenericTest(&tst);
}
TEST(amdsmitstReadOnly, TestHWTopologyRead) {
  TestHWTopologyRead tst;
  RunGenericTest(&tst);
}
TEST(amdsmitstReadOnly, TestGpuMetricsRead) {
  TestGpuMetricsRead tst;
  RunGenericTest(&tst);
}
TEST(amdsmitstReadOnly, TestMetricsCounterRead) {
  TestMetricsCounterRead tst;
  RunGenericTest(&tst);
}
TEST(amdsmitstReadWrite, TestPerfDeterminism) {
  if (!amd::smi::is_sudo_user()) GTEST_SKIP_("Invalid permission - Must run as super user");
  TestPerfDeterminism tst;
  RunGenericTest(&tst);
}
TEST(amdsmitstReadWrite, TestXGMIReadWrite) {
  if (!amd::smi::is_sudo_user()) GTEST_SKIP_("Invalid permission - Must run as super user");
  TestXGMIReadWrite tst;
  RunGenericTest(&tst);
}
TEST(amdsmitstReadOnly, TestMemPageInfoRead) {
  TestMemPageInfoRead tst;
  RunGenericTest(&tst);
}
TEST(amdsmitstReadOnly, TestAPISupportRead) {
  TestAPISupportRead tst;
  RunGenericTest(&tst);
}
/*
TEST(amdsmitstReadOnly, TestMutualExclusion) {
  TestMutualExclusion tst;
  SetFlags(&tst);
  tst.DisplayTestInfo();
  tst.SetUp();
  tst.Run();
  RunCustomTestEpilog(&tst);
}
*/

TEST(amdsmitstReadWrite, TestComputePartitionReadWrite) {
  if (!amd::smi::is_sudo_user()) GTEST_SKIP_("Invalid permission - Must run as super user");
  TestComputePartitionReadWrite tst;
  RunGenericTest(&tst);
}

TEST(amdsmitstReadWrite, TestMemoryPartitionReadWrite) {
  if (!amd::smi::is_sudo_user()) GTEST_SKIP_("Invalid permission - Must run as super user");
  TestMemoryPartitionReadWrite tst;
  RunGenericTest(&tst);
}

TEST(amdsmitstReadWrite, TestEvtNotifReadWrite) {
  if (!amd::smi::is_sudo_user()) GTEST_SKIP_("Invalid permission - Must run as super user");
  TestEvtNotifReadWrite tst;
  RunGenericTest(&tst);
}

TEST(amdsmitstReadOnly, TestGPUCacheRead) {
  TestGPUCacheRead tst;
  RunGenericTest(&tst);
}
/*
TEST(amdsmitstReadOnly, TestConcurrentInit) {
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
  return RUN_ALL_TESTS();
}
