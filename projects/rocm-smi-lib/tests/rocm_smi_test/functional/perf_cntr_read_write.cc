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
#include <bitset>
#include <string>
#include <algorithm>
#include <vector>
#include <memory>

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
    PC_EVT_GRP(XGMI, "XGMI"),
    PC_EVT_GRP(XGMI_DATA_OUT, "XGMI_DATA_OUT")
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

void TestPerfCntrReadWrite::CountEvents(uint32_t dv_ind,
       rsmi_event_type_t evnt, rsmi_counter_value_t *val, int32_t sleep_sec) {
  rsmi_event_handle_t evt_handle;
  rsmi_status_t ret;

  ret = rsmi_dev_counter_create(dv_ind,
                       static_cast<rsmi_event_type_t>(evnt), &evt_handle);
  CHK_ERR_ASRT(ret)

  // Note that rsmi_dev_counter_create() should never return
  // RSMI_STATUS_NOT_SUPPORTED. It will return RSMI_STATUS_OUT_OF_RESOURCES
  // if it is unable to create a counter.
  ret = rsmi_dev_counter_create(dv_ind,
                       static_cast<rsmi_event_type_t>(evnt), nullptr);
  ASSERT_EQ(ret, RSMI_STATUS_INVALID_ARGS);

  ret = rsmi_counter_control(evt_handle, RSMI_CNTR_CMD_START, nullptr);
  if (ret == RSMI_STATUS_NOT_SUPPORTED) {
     std::cout << "rsmi_counter_control() returned "
                                "RSMI_STATUS_NOT_SUPPORTED" << std::endl;
     throw RSMI_STATUS_NOT_SUPPORTED;
  } else {
    CHK_ERR_ASRT(ret)
  }
  sleep(sleep_sec);

  ret = rsmi_counter_read(evt_handle, val);
  CHK_ERR_ASRT(ret)

  IF_VERB(STANDARD) {
    std::cout << "\t\t\tValue: " << val->value << std::endl;
    std::cout << "\t\t\tTime Enabled (nS): " << val->time_enabled << std::endl;
    std::cout << "\t\t\tTime Running (nS): " << val->time_running << std::endl;
    std::cout << "\t\t\tEvents/Second Running: " <<
            val->value/static_cast<float>(val->time_running) << std::endl;
  }
  ret = rsmi_dev_counter_destroy(evt_handle);
  CHK_ERR_ASRT(ret)
}

static const uint64_t kGigByte = 1073741824;  // 1024^3
static const uint64_t kGig = 1000000000;

static const uint64_t kVg20Level1Bandwidth = 23;  // 23 GB/sec


void
TestPerfCntrReadWrite::testEventsIndividually(uint32_t dv_ind) {
  rsmi_status_t ret;
  rsmi_counter_value_t val;
  uint64_t throughput;

  std::cout << "Test events sequentially (device "   <<
                                                 dv_ind << ")" << std::endl;

  auto utiliz = [&](rsmi_event_type_t evt, uint32_t chan) {
    IF_VERB(STANDARD) {
      std::cout << "****************************" << std::endl;
      std::cout << "Test XGMI Link Utilization (channel " <<
                                                     chan << ")" << std::endl;
      std::cout << "****************************" << std::endl;
      std::cout << "Assumed Level 1 Bandwidth: " <<
                                 kVg20Level1Bandwidth << "GB/sec" << std::endl;
    }
    uint32_t tmp_verbosity = verbosity();
    set_verbosity(0);
    for (int i = 0; i < 5; ++i) {
      std::cout << "\t\tPass " << i << ":" << std::endl;

      CountEvents(dv_ind, evt, &val, 1);
      double coll_time_sec = static_cast<double>(val.time_running)/kGig;
      throughput = (val.value * 32)/coll_time_sec;
      std::cout << "\t\t\tCollected events for " << coll_time_sec <<
                                                        " seconds" << std::endl;
      std::cout << "\t\t\tEvents collected: " << val.value << std::endl;
      std::cout << "\t\t\tXGMI throughput: " << throughput <<
                                                   " bytes/second" << std::endl;
      std::cout << "\t\t\tXGMI Channel Utilization: " <<
        100*throughput/static_cast<double>(kVg20Level1Bandwidth*kGigByte) <<
                                                               "%" << std::endl;
      std::cout << "\t\t\t****" << std::endl;
    }
    set_verbosity(tmp_verbosity);
  };


  IF_VERB(STANDARD) {
    std::cout << "****************************" << std::endl;
    std::cout << "Test each event individually" << std::endl;
    std::cout << "****************************" << std::endl;
  }
  for (PerfCntrEvtGrp grp : s_event_groups) {
    ret = rsmi_dev_counter_group_supported(dv_ind, grp.group());
    if (ret == RSMI_STATUS_NOT_SUPPORTED) {
      continue;
    }

    IF_VERB(STANDARD) {
      std::cout << "Testing Event Group " << grp.name() << std::endl;
    }
    if (grp.group() == RSMI_EVNT_GRP_XGMI_DATA_OUT) {
      utiliz(RSMI_EVNT_XGMI_DATA_OUT_0, 0);
      utiliz(RSMI_EVNT_XGMI_DATA_OUT_1, 1);
      utiliz(RSMI_EVNT_XGMI_DATA_OUT_2, 2);
      utiliz(RSMI_EVNT_XGMI_DATA_OUT_3, 3);
      utiliz(RSMI_EVNT_XGMI_DATA_OUT_4, 4);
      utiliz(RSMI_EVNT_XGMI_DATA_OUT_5, 5);
    } else if (grp.group() == RSMI_EVNT_GRP_XGMI) {
      utiliz(RSMI_EVNT_XGMI_1_BEATS_TX, 1);
      utiliz(RSMI_EVNT_XGMI_0_BEATS_TX, 0);
    }
    for (uint32_t evnt = grp.first_evt(); evnt <= grp.last_evt(); ++evnt) {
      IF_VERB(STANDARD) {
        std::cout << "\tTesting Event Type " << evnt << std::endl;
      }
      CountEvents(dv_ind, static_cast<rsmi_event_type_t>(evnt), &val);
    }
  }
}

void
TestPerfCntrReadWrite::testEventsSimultaneously(uint32_t dv_ind) {
  rsmi_status_t ret;
  rsmi_counter_value_t val;
  uint32_t avail_counters;

  IF_VERB(STANDARD) {
    std::cout << "****************************" << std::endl;
    std::cout << "Test events simultaneously (device "   <<
                                                   dv_ind << ")" << std::endl;
    std::cout << "****************************" << std::endl;
  }

  /* This code is a little convoluted. The reason is that it is meant to test
   * having multiple events being used at one time, rather than sequentially
   * handling 1 event at a time.
   */
  for (PerfCntrEvtGrp grp : s_event_groups) {
    ret = rsmi_dev_counter_group_supported(dv_ind, grp.group());
    if (ret == RSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout << "\tEvent Group " << grp.name() <<
                                  " is not supported. Skipping." << std::endl;
      }
      continue;
    }

    IF_VERB(STANDARD) {
      std::cout << "Testing Event Group " << grp.name() << std::endl;
    }

    ret = rsmi_counter_available_counters_get(dv_ind, grp.group(),
                                                             &avail_counters);
    IF_VERB(STANDARD) {
      std::cout << "Available Counters: " << avail_counters << std::endl;
    }
    CHK_ERR_ASRT(ret)

    std::shared_ptr<rsmi_event_handle_t> evt_handle =
                    std::shared_ptr<rsmi_event_handle_t>(
                                     new rsmi_event_handle_t[avail_counters]);

    uint32_t tmp, j;
    uint32_t num_created = 0;

    for (uint32_t evnt = grp.first_evt(); evnt <= grp.last_evt();
                                                     evnt += avail_counters) {
      IF_VERB(STANDARD) {
        std::cout << "\tTesting Event Type " << evnt << std::endl;
      }
      IF_VERB(STANDARD) {
        std::cout << "\tCreating events..." << std::endl;
      }
      for (j = 0; j < avail_counters; ++j) {
        tmp = static_cast<rsmi_event_type_t>(evnt + j);

        if (tmp > grp.last_evt()) {
          break;
        }

        IF_VERB(STANDARD) {
          std::cout << "\tEvent Type " << tmp << std::endl;
        }

        ret = rsmi_dev_counter_create(dv_ind,
                     static_cast<rsmi_event_type_t>(tmp), &evt_handle.get()[j]);
        CHK_ERR_ASRT(ret)
      }
      num_created = j;
      IF_VERB(STANDARD) {
        std::cout << "\tStart Counters..." << std::endl;
      }
      uint32_t tmp_cntrs;

      for (j = 0; j < num_created; ++j) {
        tmp = static_cast<rsmi_event_type_t>(evnt + j);

        ret = rsmi_counter_control(evt_handle.get()[j], RSMI_CNTR_CMD_START,
                                                                     nullptr);
        CHK_ERR_ASRT(ret)

        ret = rsmi_counter_available_counters_get(dv_ind, grp.group(),
                                                                  &tmp_cntrs);
        CHK_ERR_ASRT(ret)
        ASSERT_EQ(tmp_cntrs, (avail_counters - j - 1));
      }

      sleep(1);

      IF_VERB(STANDARD) {
        std::cout << "\tRead Counters..." << std::endl;
      }
      for (j = 0; j < num_created; ++j) {
        tmp = static_cast<rsmi_event_type_t>(evnt + j);

        ret = rsmi_counter_read(evt_handle.get()[j], &val);
        CHK_ERR_ASRT(ret)

        IF_VERB(STANDARD) {
          std::cout << "\tCounter: " << tmp << std::endl;
          std::cout << "\tSuccessfully read value: " << std::endl;
          std::cout << "\t\tValue: " << val.value << std::endl;
          std::cout << "\t\tTime Enabled: " << val.time_enabled << std::endl;
          std::cout << "\t\tTime Running: " << val.time_running << std::endl;
        }
      }
      for (j = 0; j < num_created; ++j) {
        ret = rsmi_dev_counter_destroy(evt_handle.get()[j]);
        CHK_ERR_ASRT(ret)
      }
    }
  }
}

void TestPerfCntrReadWrite::Run(void) {
  TestBase::Run();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    PrintDeviceHeader(dv_ind);
    try {
      testEventsIndividually(dv_ind);
      testEventsSimultaneously(dv_ind);
    } catch(rsmi_status_t r) {
       switch (r) {
         case RSMI_STATUS_NOT_SUPPORTED:
           std::cout << "The performance counter event tried is not "
                                     "supported for this device" << std::endl;
           break;

         default:
           std::cout << "Unexpected exception caught with rsmi "
                                         "return value of " << r << std::endl;
       }
    } catch(...) {
      ASSERT_FALSE("Unexpected exception caught");
    }
  }
}
