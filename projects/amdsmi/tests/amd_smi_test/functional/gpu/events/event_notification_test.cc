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

#include <cstring>
#include <string>
#include <vector>

#include "unit/unit_test_framework.h"

using amdsmi::unittest::kInvalidHandle;
using amdsmi::unittest::kVerbose;

// Event notification workflow: amdsmi_init_gpu_event_notification /
// amdsmi_set_gpu_event_notification_mask / amdsmi_get_gpu_event_notification /
// amdsmi_stop_gpu_event_notification.
// OR of every event type into the notification bit mask.
static uint64_t AllEventsMask() {
  uint64_t mask = 0;
  for (amdsmi_evt_notification_type_t e = AMDSMI_EVT_NOTIF_FIRST; e <= AMDSMI_EVT_NOTIF_LAST;
       e = static_cast<amdsmi_evt_notification_type_t>(static_cast<uint32_t>(e) + 1)) {
    mask |= AMDSMI_EVENT_MASK_FROM_INDEX(e);
  }
  return mask;
}

// ---------------- invalid parameters first (not gated) ----------------
TEST_F(GpuFunctionalReadWrite, InitEventNotification_InvalidHandle) {
  RequireInit();
  DISPLAY_AMDSMI_API("amdsmi_init_gpu_event_notification", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_init_gpu_event_notification(kInvalidHandle);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

TEST_F(GpuFunctionalReadWrite, SetEventMask_InvalidHandle) {
  RequireInit();
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_event_notification_mask", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_gpu_event_notification_mask(kInvalidHandle, AllEventsMask());
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_INIT);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

TEST_F(GpuFunctionalReadWrite, GetEventNotification_NullCount) {
  RequireInit();
  amdsmi_evt_notification_data_t data[4];
  memset(data, 0, sizeof(data));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_event_notification", "num_elem=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_event_notification(0, nullptr, data);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  EXPECT_EQ(err, AMDSMI_STATUS_INVAL);
}

TEST_F(GpuFunctionalReadWrite, StopEventNotification_InvalidHandle) {
  RequireInit();
  DISPLAY_AMDSMI_API("amdsmi_stop_gpu_event_notification", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_stop_gpu_event_notification(kInvalidHandle);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_INIT);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

// ---------------- full init -> set-mask -> collect -> stop workflow ----------------
// init/set/stop allocate and mutate per-device event-notification state, so the
// flow is gated with the shared mutation gate.
TEST_F(GpuFunctionalReadWrite, EventNotification_Workflow) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";

  amdsmi::unittest::StatusCollector col("amdsmi_gpu_event_notification");
  const uint64_t mask = AllEventsMask();
  std::vector<size_t> inited;

  for (size_t i = 0; i < gpus().size(); ++i) {
    DISPLAY_AMDSMI_API("amdsmi_init_gpu_event_notification", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_init_gpu_event_notification(gpus()[i]);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    col.Record("init gpu=" + std::to_string(i), err,
               ::amdsmi::unittest::AmdsmiStatusIsExpected(err, AMDSMI_STATUS_SUCCESS,
                                                          AMDSMI_STATUS_NOT_SUPPORTED,
                                                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
    if (err != AMDSMI_STATUS_SUCCESS) continue;  // no init -> nothing to set/stop
    inited.push_back(i);

    DISPLAY_AMDSMI_API("amdsmi_set_gpu_event_notification_mask",
                       "gpu=" + std::to_string(i) + " mask=all", kVerbose);
    amdsmi_status_t serr = amdsmi_set_gpu_event_notification_mask(gpus()[i], mask);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, serr, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    col.Record("set_mask gpu=" + std::to_string(i), serr,
               ::amdsmi::unittest::AmdsmiStatusIsExpected(serr, AMDSMI_STATUS_SUCCESS,
                                                          AMDSMI_STATUS_NOT_SUPPORTED,
                                                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }

  // Collect any pending events (short timeout; usually none fire during the test).
  if (!inited.empty()) {
    amdsmi_evt_notification_data_t data[16];
    memset(data, 0, sizeof(data));
    uint32_t num_elem = 16;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_event_notification", "timeout=100ms", kVerbose);
    amdsmi_status_t gerr = amdsmi_get_gpu_event_notification(100, &num_elem, data);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, gerr, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NO_DATA, AMDSMI_STATUS_INSUFFICIENT_SIZE,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    col.Record(
        "get_notification", gerr,
        ::amdsmi::unittest::AmdsmiStatusIsExpected(
            gerr, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NO_DATA, AMDSMI_STATUS_INSUFFICIENT_SIZE,
            AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
    if (gerr == AMDSMI_STATUS_SUCCESS || gerr == AMDSMI_STATUS_INSUFFICIENT_SIZE) {
      EXPECT_LE(num_elem, 16u) << "reported more events than the buffer holds";
      for (uint32_t i = 0; i < num_elem && i < 16u; ++i) {
        if (kVerbose) {
          std::cout << "\t  event type=" << static_cast<int>(data[i].event)
                    << " msg=" << data[i].message << std::endl;
        }
      }
    }
  }

  // Restore each device's event state by releasing the notification resources.
  for (size_t i : inited) {
    DISPLAY_AMDSMI_API("amdsmi_stop_gpu_event_notification", "gpu=" + std::to_string(i), kVerbose);
    amdsmi_status_t err = amdsmi_stop_gpu_event_notification(gpus()[i]);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED);
    col.Record("stop gpu=" + std::to_string(i), err,
               ::amdsmi::unittest::AmdsmiStatusIsExpected(err, AMDSMI_STATUS_SUCCESS,
                                                          AMDSMI_STATUS_NOT_SUPPORTED,
                                                          AMDSMI_STATUS_NOT_YET_IMPLEMENTED));
  }

  col.ExpectNoFailures();
}
