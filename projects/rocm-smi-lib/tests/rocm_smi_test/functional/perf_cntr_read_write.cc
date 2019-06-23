/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2019, Advanced Micro Devices, Inc.
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
#include <bitset>
#include <string>
#include <algorithm>
#include <vector>

#include "gtest/gtest.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi_test/functional/perf_cntr_read_write.h"
#include "rocm_smi_test/test_common.h"

PerfCntrEvtGrp::PerfCntrEvtGrp(rsmi_event_group_t grp, uint32_t first,
        uint32_t last, std::string name) : grp_(grp), first_evt_(first),
                                                last_evt_(last), name_(name) {
  num_events_ = last_evt_ - first_evt_ + 1;
}

PerfCntrEvtGrp::~PerfCntrEvtGrp() {}

// Add new event groups to test here
#define PC_EVT_GRP(SHRT, NAME) \
    PerfCntrEvtGrp(RSMI_EVNT_GRP_##SHRT, RSMI_EVNT_##SHRT##_FIRST, \
                                                RSMI_EVNT_##SHRT##_LAST, NAME)
static const std::vector<PerfCntrEvtGrp> s_event_groups = {
    PC_EVT_GRP(XGMI, "XGMI")
};

TestPerfCntrReadWrite::TestPerfCntrReadWrite() : TestBase() {
  set_title("RSMI Performance Counter Read/Write Test");
  set_description("The Performance counter tests verify that performance"
                            " counters can be controlled and read properly.");
}

TestPerfCntrReadWrite::~TestPerfCntrReadWrite(void) {
}

void TestPerfCntrReadWrite::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestPerfCntrReadWrite::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestPerfCntrReadWrite::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestPerfCntrReadWrite::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // rsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

#define RSMI_EVNT_ENUM_FIRST(GRP_NAME) RSMI_EVNT_##GRP_NAME##_FIRST
#define RSMI_EVNT_ENUM_LAST(GRP_NAME) RSMI_EVNT_##GRP_NAME##_LAST

// Refactor this to handle different event groups once we have > 1 event group
void
TestPerfCntrReadWrite::testEventsIndividually(uint32_t dv_ind) {
  rsmi_event_handle_t evt_handle;
  rsmi_status_t ret;
  rsmi_counter_value_t val;

  std::cout << "****************************" << std::endl;
  std::cout << "Test each event individually" << std::endl;
  std::cout << "****************************" << std::endl;

  for (PerfCntrEvtGrp grp : s_event_groups) {
    if (rsmi_dev_counter_group_supported(dv_ind, grp.group())
                                               == RSMI_STATUS_NOT_SUPPORTED) {
      continue;
    }

    IF_VERB(STANDARD) {
      std::cout << "Testing Event Group " << grp.name() << std::endl;
    }

    for (uint32_t evnt = grp.first_evt(); evnt <= grp.last_evt(); ++evnt) {
      IF_VERB(STANDARD) {
        std::cout << "\tTesting Event Type " << evnt << std::endl;
      }

      IF_VERB(STANDARD) {
        std::cout << "\t\tCreating event..." << std::endl;
      }
      ret = rsmi_dev_counter_create(dv_ind,
                           static_cast<rsmi_event_type_t>(evnt), &evt_handle);
      CHK_ERR_ASRT(ret)

      IF_VERB(STANDARD) {
        std::cout << "\t\tStart Counting..." << std::endl;
      }
      ret = rsmi_counter_control(evt_handle, RSMI_CNTR_CMD_START, nullptr);
      CHK_ERR_ASRT(ret)

      sleep(1);
      IF_VERB(STANDARD) {
        std::cout << "\t\tStop Counting..." << std::endl;
      }
      ret = rsmi_counter_control(evt_handle, RSMI_CNTR_CMD_STOP, nullptr);
      CHK_ERR_ASRT(ret)

      IF_VERB(STANDARD) {
        std::cout << "\t\tRead Counter..." << std::endl;
      }
      ret = rsmi_counter_read(evt_handle, &val);
      CHK_ERR_ASRT(ret)

      IF_VERB(STANDARD) {
        std::cout << "\t\tSuccessfully read value: " << std::endl;
        std::cout << "\t\t\tValue: " << val.value << std::endl;
        std::cout << "\t\t\tTime Enabled: " << val.time_enabled << std::endl;
        std::cout << "\t\t\tTime Running: " << val.time_running << std::endl;
      }
      IF_VERB(STANDARD) {
        std::cout << "\t\tRe-start Counting..." << std::endl;
      }
      ret = rsmi_counter_control(evt_handle, RSMI_CNTR_CMD_START, nullptr);
      CHK_ERR_ASRT(ret)

      IF_VERB(STANDARD) {
        std::cout << "\t\tRead free-running Counter..." << std::endl;
      }
      ret = rsmi_counter_read(evt_handle, &val);
      CHK_ERR_ASRT(ret)

      IF_VERB(STANDARD) {
        std::cout << "\t\tSuccessfully read value: " << std::endl;
        std::cout << "\t\t\tValue: " << val.value << std::endl;
        std::cout << "\t\t\tTime Enabled: " << val.time_enabled << std::endl;
        std::cout << "\t\t\tTime Running: " << val.time_running << std::endl;
      }
      ret = rsmi_dev_counter_destroy(evt_handle);
      CHK_ERR_ASRT(ret)
    }
  }
}

// Refactor this to handle different event groups once we have > 1 event group
void
TestPerfCntrReadWrite::testEventsSimultaneously(uint32_t dv_ind) {
  rsmi_event_handle_t evt_handle[RSMI_EVNT_XGMI_LAST -
                                                    RSMI_EVNT_XGMI_FIRST + 1];
  rsmi_status_t ret;
  rsmi_counter_value_t val;
  uint32_t avail_counters;

  std::cout << "****************************" << std::endl;
  std::cout << "Test events simultaneously"   << std::endl;
  std::cout << "****************************" << std::endl;

  for (PerfCntrEvtGrp grp : s_event_groups) {
    if (rsmi_dev_counter_group_supported(dv_ind, grp.group())
                                               == RSMI_STATUS_NOT_SUPPORTED) {
      std::cout << "\tEvent Group" << grp.name() <<
                                  " is not supported. Skipping." << std::endl;
      continue;
    }
    IF_VERB(STANDARD) {
      std::cout << "Testing Event Group " << grp.name() << std::endl;
    }

    ret = rsmi_counter_available_counters_get(dv_ind, grp.group(),
                                                               &avail_counters);
    std::cout << "Available Counters: " << avail_counters << std::endl;

    CHK_ERR_ASRT(ret)

    uint32_t tmp;

    for (uint32_t evnt = grp.first_evt(); evnt <= grp.last_evt();
                                                     evnt += avail_counters) {
      IF_VERB(STANDARD) {
        std::cout << "\tTesting Event Type " << evnt << std::endl;
      }
      IF_VERB(STANDARD) {
        std::cout << "\tCreating events..." << std::endl;
      }
      for (uint32_t j = 0; j < avail_counters; ++j) {
        tmp = static_cast<rsmi_event_type_t>(evnt + j);
        IF_VERB(STANDARD) {
          std::cout << "\tEvent Type " << tmp << std::endl;
        }

        ret = rsmi_dev_counter_create(dv_ind,
                     static_cast<rsmi_event_type_t>(tmp), &evt_handle[tmp]);
        CHK_ERR_ASRT(ret)
      }

      IF_VERB(STANDARD) {
        std::cout << "\tStart Counters..." << std::endl;
      }
      uint32_t tmp_cntrs;

      for (uint32_t j = 0; j < avail_counters; ++j) {
        tmp = static_cast<rsmi_event_type_t>(evnt + j);
        ret = rsmi_counter_control(evt_handle[tmp], RSMI_CNTR_CMD_START,
                                                                     nullptr);
        CHK_ERR_ASRT(ret)

        ret = rsmi_counter_available_counters_get(dv_ind, grp.group(),
                                                                  &tmp_cntrs);
        CHK_ERR_ASRT(ret)
        ASSERT_EQ(tmp_cntrs, (avail_counters - j -1));
      }

      sleep(5);
      IF_VERB(STANDARD) {
        std::cout << "\tStop Counters..." << std::endl;
      }
      for (uint32_t j = 0; j < avail_counters; ++j) {
        tmp = static_cast<rsmi_event_type_t>(evnt + j);
        ret = rsmi_counter_control(evt_handle[tmp], RSMI_CNTR_CMD_STOP,
                                                                     nullptr);
        CHK_ERR_ASRT(ret)
      }

      IF_VERB(STANDARD) {
        std::cout << "\tRead Counters..." << std::endl;
      }
      for (uint32_t j = 0; j < avail_counters; ++j) {
        tmp = static_cast<rsmi_event_type_t>(evnt + j);
        ret = rsmi_counter_read(evt_handle[tmp], &val);
        CHK_ERR_ASRT(ret)

        IF_VERB(STANDARD) {
          std::cout << "\tCounter: " << tmp << std::endl;
          std::cout << "\tSuccessfully read value: " << std::endl;
          std::cout << "\t\tValue: " << val.value << std::endl;
          std::cout << "\t\tTime Enabled: " << val.time_enabled << std::endl;
          std::cout << "\t\tTime Running: " << val.time_running << std::endl;
        }
      }
      for (uint32_t j = 0; j < avail_counters; ++j) {
        tmp = static_cast<rsmi_event_type_t>(evnt + j);
        ret = rsmi_dev_counter_destroy(evt_handle[tmp]);
        CHK_ERR_ASRT(ret)
      }
    }
  }
}

void TestPerfCntrReadWrite::Run(void) {
  TestBase::Run();

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    PrintDeviceHeader(dv_ind);

    testEventsIndividually(dv_ind);
    testEventsSimultaneously(dv_ind);
  }
}
