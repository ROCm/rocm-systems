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

#include <gtest/gtest.h>
#include "amd_smi/amdsmi.h"
#include "evt_notif_read_write.h"
#include "../test_utils.h"

TestEvtNotifReadWrite::TestEvtNotifReadWrite() : TestBase() {
  set_title("AMDSMI Event Notification Read/Write Test");
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
  // This will close handles opened within amdsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

void TestEvtNotifReadWrite::Run(void) {
  amdsmi_status_t ret;
  uint32_t dv_ind;

  TestBase::Run();
  if (num_monitor_devs() == 0) {
    return;
  }

  if (setup_failed_) {
     IF_VERB(STANDARD) {
        std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
     }
    return;
  }

  amdsmi_evt_notification_type_t evt_type = AMDSMI_EVT_NOTIF_FIRST;
  uint64_t mask = AMDSMI_EVENT_MASK_FROM_INDEX(evt_type);
  while (evt_type <= AMDSMI_EVT_NOTIF_LAST) {
    mask |= AMDSMI_EVENT_MASK_FROM_INDEX(evt_type);
    evt_type = static_cast<amdsmi_evt_notification_type_t>(
                                           static_cast<uint32_t>(evt_type)+1);
  }

  for (dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    ret = amdsmi_init_gpu_event_notification(processor_handles_[dv_ind]);
    if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout <<
          "Event notification is not supported for this driver version." <<
                                                                    std::endl;
      }
      return;
    }
    ASSERT_EQ(ret, AMDSMI_STATUS_SUCCESS);
    ret =  amdsmi_set_gpu_event_notification_mask(processor_handles_[dv_ind], mask);
    ASSERT_EQ(ret, AMDSMI_STATUS_SUCCESS);
  }

  amdsmi_evt_notification_data_t data[10];
  uint32_t num_elem = 10;
  bool read_again = false;

  ret =  amdsmi_get_gpu_event_notification(10000, &num_elem, data);
  if (ret == AMDSMI_STATUS_SUCCESS || ret == AMDSMI_STATUS_INSUFFICIENT_SIZE) {
    EXPECT_LE(num_elem, 10) <<
            "Expected the number of elements found to be <= buffer size (10)";
    IF_VERB(STANDARD) {
      for (uint32_t i = 0; i < num_elem; ++i) {
        std::cout << "\tdv_handle=" << data[i].processor_handle <<
                   "  Type: " << NameFromEvtNotifType(data[i].event) <<
                   "  Mesg: " << data[i].message << std::endl;
        if (data[i].event == AMDSMI_EVT_NOTIF_GPU_PRE_RESET) {
          read_again = true;
        }
      }
    }
    IF_VERB(STANDARD) {
      if (ret == AMDSMI_STATUS_INSUFFICIENT_SIZE) {
        std::cout <<
        "\t\tBuffer size is 10, but more than 10 events are available." <<
                                                                    std::endl;
      }
    }
  } else if (ret == AMDSMI_STATUS_NO_DATA) {
    IF_VERB(STANDARD) {
      std::cout << "\tNo events were collected." << std::endl;
    }
  } else {
    // This should always fail. We want to print out the return code.
    EXPECT_EQ(ret, AMDSMI_STATUS_SUCCESS) <<
                  "Unexpected return code for  amdsmi_get_gpu_event_notification()";
  }

  // In case GPU Pre reset event was collected in the previous read,
  // read again to get the GPU Post reset event.
  if (read_again) {
    ret =  amdsmi_get_gpu_event_notification(10000, &num_elem, data);
    if (ret == AMDSMI_STATUS_SUCCESS || ret == AMDSMI_STATUS_INSUFFICIENT_SIZE) {
      EXPECT_LE(num_elem, 10) <<
              "Expected the number of elements found to be <= buffer size (10)";
      IF_VERB(STANDARD) {
        for (uint32_t i = 0; i < num_elem; ++i) {
          std::cout << "\tdv_handle=" << data[i].processor_handle <<
                     "  Type: " << NameFromEvtNotifType(data[i].event) <<
                     "  Mesg: " << data[i].message << std::endl;
        }
      }
      IF_VERB(STANDARD) {
        if (ret == AMDSMI_STATUS_INSUFFICIENT_SIZE) {
          std::cout <<
          "\t\tBuffer size is 10, but more than 10 events are available." <<
                                                                    std::endl;
        }
      }
    } else if (ret == AMDSMI_STATUS_NO_DATA) {
      IF_VERB(STANDARD) {
        std::cout << "\tNo further events were collected." << std::endl;
      }
    } else {
      // This should always fail. We want to print out the return code.
      EXPECT_EQ(ret, AMDSMI_STATUS_SUCCESS) <<
                  "Unexpected return code for  amdsmi_get_gpu_event_notification()";
    }
  }

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    ret = amdsmi_stop_gpu_event_notification(processor_handles_[dv_ind]);
    ASSERT_EQ(ret, AMDSMI_STATUS_SUCCESS);
  }
}
