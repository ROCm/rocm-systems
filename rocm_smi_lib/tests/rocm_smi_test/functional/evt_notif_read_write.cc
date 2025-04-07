/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2020, Advanced Micro Devices, Inc.
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

#include "gtest/gtest.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi_test/functional/evt_notif_read_write.h"
#include "rocm_smi_test/test_common.h"
#include "rocm_smi_test/test_utils.h"

TestEvtNotifReadWrite::TestEvtNotifReadWrite() : TestBase() {
  set_title("RSMI Event Notification Read/Write Test");
  set_description("The Event Notification Read/Write tests verifies that "
        "we can configure to collect various event types and then read them");
}

TestEvtNotifReadWrite::~TestEvtNotifReadWrite(void) {
}

void TestEvtNotifReadWrite::SetUp(void) {
  TestBase::SetUp();
  return;
}

void TestEvtNotifReadWrite::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestEvtNotifReadWrite::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestEvtNotifReadWrite::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // rsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

void TestEvtNotifReadWrite::Run(void) {
  rsmi_status_t ret;
  uint32_t dv_ind;

  TestBase::Run();
  if (num_monitor_devs() == 0) {
    return;
  }

  if (setup_failed_) {
     IF_VERB(STANDARD) {
        std::cout << "** SetUp Failed for this test. Skipping.**" <<
                                                                    std::endl;
     }
    return;
  }

  rsmi_evt_notification_type_t evt_type = RSMI_EVT_NOTIF_FIRST;
  uint64_t mask = RSMI_EVENT_MASK_FROM_INDEX(evt_type);
  while (evt_type <= RSMI_EVT_NOTIF_LAST) {
    mask |= RSMI_EVENT_MASK_FROM_INDEX(evt_type);
    evt_type = static_cast<rsmi_evt_notification_type_t>(
                                           static_cast<uint32_t>(evt_type)+1);
  }

  for (dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    ret = rsmi_event_notification_init(dv_ind);
    if (ret == RSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout <<
          "Event notification is not supported for this driver version." <<
                                                                    std::endl;
      }
      return;
    }
    ASSERT_EQ(ret, RSMI_STATUS_SUCCESS);
    ret = rsmi_event_notification_mask_set(dv_ind, mask);
    ASSERT_EQ(ret, RSMI_STATUS_SUCCESS);
  }

  rsmi_evt_notification_data_t data[10];
  uint32_t num_elem = 10;
  bool read_again = false;

  ret = rsmi_event_notification_get(10000, &num_elem, data);
  if (ret == RSMI_STATUS_SUCCESS || ret == RSMI_STATUS_INSUFFICIENT_SIZE) {
    EXPECT_LE(num_elem, 10) <<
            "Expected the number of elements found to be <= buffer size (10)";
    IF_VERB(STANDARD) {
      for (uint32_t i = 0; i < num_elem; ++i) {
        std::cout << "\tdv_ind=" << data[i].dv_ind <<
                   "  Type: " << NameFromEvtNotifType(data[i].event) <<
                   "  Mesg: " << data[i].message << std::endl;
        if (data[i].event == RSMI_EVT_NOTIF_GPU_PRE_RESET) {
          read_again = true;
        }
      }
    }
    IF_VERB(STANDARD) {
      if (ret == RSMI_STATUS_INSUFFICIENT_SIZE) {
        std::cout <<
        "\t\tBuffer size is 10, but more than 10 events are available." <<
                                                                    std::endl;
      }
    }
  } else if (ret == RSMI_STATUS_NO_DATA) {
    IF_VERB(STANDARD) {
      std::cout << "\tNo events were collected." << std::endl;
    }
  } else {
    // This should always fail. We want to print out the return code.
    EXPECT_EQ(ret, RSMI_STATUS_SUCCESS) <<
                  "Unexpected return code for rsmi_event_notification_get()";
  }

  // In case GPU Pre reset event was collected in the previous read,
  // read again to get the GPU Post reset event.
  if (read_again) {
    ret = rsmi_event_notification_get(10000, &num_elem, data);
    if (ret == RSMI_STATUS_SUCCESS || ret == RSMI_STATUS_INSUFFICIENT_SIZE) {
      EXPECT_LE(num_elem, 10) <<
              "Expected the number of elements found to be <= buffer size (10)";
      IF_VERB(STANDARD) {
        for (uint32_t i = 0; i < num_elem; ++i) {
          std::cout << "\tdv_ind=" << data[i].dv_ind <<
                     "  Type: " << NameFromEvtNotifType(data[i].event) <<
                     "  Mesg: " << data[i].message << std::endl;
        }
      }
      IF_VERB(STANDARD) {
        if (ret == RSMI_STATUS_INSUFFICIENT_SIZE) {
          std::cout <<
          "\t\tBuffer size is 10, but more than 10 events are available." <<
                                                                    std::endl;
        }
      }
    } else if (ret == RSMI_STATUS_NO_DATA) {
      IF_VERB(STANDARD) {
        std::cout << "\tNo further events were collected." << std::endl;
      }
    } else {
      // This should always fail. We want to print out the return code.
      EXPECT_EQ(ret, RSMI_STATUS_SUCCESS) <<
                  "Unexpected return code for rsmi_event_notification_get()";
    }
  }

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    ret = rsmi_event_notification_stop(dv_ind);
    ASSERT_EQ(ret, RSMI_STATUS_SUCCESS);
  }
}
