// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <cstring>
#include <string>

#include "api_test_framework.h"

using amdsmi::test::kInvalidHandle;
using amdsmi::test::kVerbose;

// Invalid-input coverage for the setter APIs whose valid-input
// behavior is exercised in functional/gpu/events/event_notification_test.cc. Rejecting a bad
// argument never touches the device, so these belong in the integration tier.

// ---------------- invalid parameters first (not gated) ----------------
// OR of every event type into the notification bit mask.
static uint64_t AllEventsMask() {
  uint64_t mask = 0;
  for (amdsmi_evt_notification_type_t e = AMDSMI_EVT_NOTIF_FIRST; e <= AMDSMI_EVT_NOTIF_LAST;
       e = static_cast<amdsmi_evt_notification_type_t>(static_cast<uint32_t>(e) + 1)) {
    mask |= AMDSMI_EVENT_MASK_FROM_INDEX(e);
  }
  return mask;
}

TEST_F(GpuIntegration, InitEventNotification_InvalidHandle) {
  RequireInit();
  DISPLAY_AMDSMI_API("amdsmi_init_gpu_event_notification", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_init_gpu_event_notification(kInvalidHandle);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

TEST_F(GpuIntegration, SetEventMask_InvalidHandle) {
  RequireInit();
  DISPLAY_AMDSMI_API("amdsmi_set_gpu_event_notification_mask", "handle=invalid", kVerbose);
  amdsmi_status_t err = amdsmi_set_gpu_event_notification_mask(kInvalidHandle, AllEventsMask());
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,
                        AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_INIT);
  EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);
}

TEST_F(GpuIntegration, GetEventNotification_NullCount) {
  RequireInit();
  amdsmi_evt_notification_data_t data[4];
  memset(data, 0, sizeof(data));
  DISPLAY_AMDSMI_API("amdsmi_get_gpu_event_notification", "num_elem=nullptr", kVerbose);
  amdsmi_status_t err = amdsmi_get_gpu_event_notification(0, nullptr, data);
  DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL);
  AMDSMI_EXPECT_NULL_ARG(err);
}
