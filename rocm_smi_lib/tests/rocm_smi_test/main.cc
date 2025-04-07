/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2017-2023, Advanced Micro Devices, Inc.
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

#include <string>
#include <vector>
#include <memory>
#include <iostream>

#include "rocm_smi/rocm_smi.h"
#include "gtest/gtest.h"
#include "rocm_smi_test/test_common.h"
#include "rocm_smi_test/test_base.h"
#include "functional/fan_read.h"
#include "functional/fan_read_write.h"
#include "functional/temp_read.h"
#include "functional/volt_read.h"
#include "functional/volt_freq_curv_read.h"
#include "functional/perf_level_read.h"
#include "functional/overdrive_read.h"
#include "functional/frequencies_read.h"
#include "functional/sys_info_read.h"
#include "functional/gpu_busy_read.h"
#include "functional/power_read.h"
#include "functional/overdrive_read_write.h"
#include "functional/perf_level_read_write.h"
#include "functional/frequencies_read_write.h"
#include "functional/pci_read_write.h"
#include "functional/power_read_write.h"
#include "functional/power_cap_read_write.h"
#include "functional/version_read.h"
#include "functional/err_cnt_read.h"
#include "functional/mem_util_read.h"
#include "functional/id_info_read.h"
#include "functional/perf_cntr_read_write.h"
#include "functional/process_info_read.h"
#include "functional/xgmi_read_write.h"
#include "functional/mem_page_info_read.h"
#include "functional/api_support_read.h"
#include "functional/measure_api_execution_time.h"
#include "functional/mutual_exclusion.h"
#include "functional/evt_notif_read_write.h"
#include "functional/init_shutdown_refcount.h"
#include "functional/computepartition_read_write.h"
#include "rocm_smi_test/functional/hw_topology_read.h"
#include "rocm_smi_test/functional/gpu_metrics_read.h"
#include "rocm_smi_test/functional/metrics_counter_read.h"
#include "rocm_smi_test/functional/perf_determinism.h"
#include "functional/memorypartition_read_write.h"

static RSMITstGlobals *sRSMIGlvalues = nullptr;

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
TEST(rsmitstReadOnly, TestVersionRead) {
  TestVersionRead tst;
  RunGenericTest(&tst);
}
TEST(rsmitstReadOnly, TestSysInfoRead) {
  TestSysInfoRead tst;
  RunGenericTest(&tst);
}
TEST(rsmitstReadOnly, FanRead) {
  TestFanRead tst;
  RunGenericTest(&tst);
}
TEST(rsmitstReadWrite, FanReadWrite) {
  TestFanReadWrite tst;
  RunGenericTest(&tst);
}
TEST(rsmitstReadOnly, TempRead) {
  TestTempRead tst;
  RunGenericTest(&tst);
}
TEST(rsmitstReadOnly, VoltRead) {
  TestVoltRead tst;
  RunGenericTest(&tst);
}
TEST(rsmitstReadOnly, TestVoltCurvRead) {
  TestVoltCurvRead tst;
  RunGenericTest(&tst);
}
TEST(rsmitstReadOnly, TestPerfLevelRead) {
  TestPerfLevelRead tst;
  RunGenericTest(&tst);
}
TEST(rsmitstReadWrite, TestComputePartitionReadWrite) {
  TestComputePartitionReadWrite tst;
  RunGenericTest(&tst);
}
TEST(rsmitstReadWrite, TestMemoryPartitionReadWrite) {
  TestMemoryPartitionReadWrite tst;
  RunGenericTest(&tst);
}
TEST(rsmitstReadWrite, TestPerfLevelReadWrite) {
  TestPerfLevelReadWrite tst;
  RunGenericTest(&tst);
}
TEST(rsmitstReadOnly, TestOverdriveRead) {
  TestOverdriveRead tst;
  RunGenericTest(&tst);
}
TEST(rsmitstReadWrite, TestOverdriveReadWrite) {
  TestOverdriveReadWrite tst;
  RunGenericTest(&tst);
}
TEST(rsmitstReadOnly, TestFrequenciesRead) {
  TestFrequenciesRead tst;
  RunGenericTest(&tst);
}
TEST(rsmitstReadWrite, TestFrequenciesReadWrite) {
  TestFrequenciesReadWrite tst;
  RunGenericTest(&tst);
}
TEST(rsmitstReadWrite, TestPciReadWrite) {
  TestPciReadWrite tst;
  RunGenericTest(&tst);
}
TEST(rsmitstReadOnly, TestGPUBusyRead) {
  TestGPUBusyRead tst;
  RunGenericTest(&tst);
}
TEST(rsmitstReadOnly, TestPowerRead) {
  TestPowerRead tst;
  RunGenericTest(&tst);
}
TEST(rsmitstReadWrite, TestPowerReadWrite) {
  TestPowerReadWrite tst;
  RunGenericTest(&tst);
}
TEST(rsmitstReadWrite, TestPowerCapReadWrite) {
  TestPowerCapReadWrite tst;
  RunGenericTest(&tst);
}
TEST(rsmitstReadOnly, TestErrCntRead) {
  TestErrCntRead tst;
  RunGenericTest(&tst);
}
TEST(rsmitstReadOnly, TestMemUtilRead) {
  TestMemUtilRead tst;
  RunGenericTest(&tst);
}
TEST(rsmitstReadOnly, TestIdInfoRead) {
  TestIdInfoRead tst;
  RunGenericTest(&tst);
}
TEST(rsmitstReadWrite, TestPerfCntrReadWrite) {
  TestPerfCntrReadWrite tst;
  RunGenericTest(&tst);
}
TEST(rsmitstReadOnly, TestProcInfoRead) {
  TestProcInfoRead tst;
  RunGenericTest(&tst);
}
TEST(rsmitstReadOnly, TestHWTopologyRead) {
  TestHWTopologyRead tst;
  RunGenericTest(&tst);
}
TEST(rsmitstReadOnly, TestGpuMetricsRead) {
  TestGpuMetricsRead tst;
  RunGenericTest(&tst);
}
TEST(rsmitstReadOnly, TestMetricsCounterRead) {
  TestMetricsCounterRead tst;
  RunGenericTest(&tst);
}
TEST(rsmitstReadWrite, TestPerfDeterminism) {
  TestPerfDeterminism tst;
  RunGenericTest(&tst);
}
TEST(rsmitstReadWrite, TestXGMIReadWrite) {
  TestXGMIReadWrite tst;
  RunGenericTest(&tst);
}
TEST(rsmitstReadOnly, TestMemPageInfoRead) {
  TestMemPageInfoRead tst;
  RunGenericTest(&tst);
}
TEST(rsmitstReadOnly, TestAPISupportRead) {
  TestAPISupportRead tst;
  RunGenericTest(&tst);
}
TEST(rsmitstReadOnly, TestMeasureApiExecutionTime) {
  TestMeasureApiExecutionTime tst;
  RunGenericTest(&tst);
}
TEST(rsmitstReadOnly, TestMutualExclusion) {
  TestMutualExclusion tst;
  SetFlags(&tst);
  tst.DisplayTestInfo();
  tst.SetUp();
  tst.Run();
  RunCustomTestEpilog(&tst);
}
TEST(rsmitstReadWrite, TestEvtNotifReadWrite) {
  TestEvtNotifReadWrite tst;
  RunGenericTest(&tst);
}
TEST(rsmitstReadOnly, Test) {
  TestConcurrentInit tst;
  SetFlags(&tst);
  tst.DisplayTestInfo();
  //  tst.SetUp();   // Avoid extra rsmi_init
  tst.Run();
  // RunCustomTestEpilog(&tst);  // Avoid extra rsmi_shut_down
  tst.DisplayResults();
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  RSMITstGlobals settings;

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
