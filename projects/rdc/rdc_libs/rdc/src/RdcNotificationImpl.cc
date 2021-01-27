/*
Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <assert.h>
#include <sys/time.h>
#include <ctime>

#include <unordered_map>
#include <vector>
#include <mutex>  // NOLINT

#include "rdc/rdc.h"
#include "rdc_lib/impl/RdcTelemetryModule.h"
#include "rdc_lib/impl/RdcNotificationImpl.h"
#include "rdc_lib/impl/RsmiUtils.h"
#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/impl/RdcSmiLib.h"
#include "rocm_smi/rocm_smi.h"

namespace amd {
namespace rdc {

static std::unordered_map<rdc_field_t, rsmi_evt_notification_type_t>
                                                rdc_2_rsmi_event_notif_map = {
  {RDC_EVNT_NOTIF_VMFAULT, RSMI_EVT_NOTIF_VMFAULT},
  {RDC_EVNT_NOTIF_FIRST, RSMI_EVT_NOTIF_FIRST},
  {RDC_EVNT_NOTIF_THERMAL_THROTTLE, RSMI_EVT_NOTIF_THERMAL_THROTTLE},
  {RDC_EVNT_NOTIF_PRE_RESET, RSMI_EVT_NOTIF_GPU_PRE_RESET},
  {RDC_EVNT_NOTIF_POST_RESET, RSMI_EVT_NOTIF_GPU_POST_RESET},
};
static std::unordered_map<rsmi_evt_notification_type_t, rdc_field_t>
                                                rsmi_event_notif_2_rdc_map = {
  {RSMI_EVT_NOTIF_VMFAULT, RDC_EVNT_NOTIF_VMFAULT},
  {RSMI_EVT_NOTIF_FIRST, RDC_EVNT_NOTIF_FIRST},
  {RSMI_EVT_NOTIF_THERMAL_THROTTLE, RDC_EVNT_NOTIF_THERMAL_THROTTLE},
  {RSMI_EVT_NOTIF_GPU_PRE_RESET, RDC_EVNT_NOTIF_PRE_RESET},
  {RSMI_EVT_NOTIF_GPU_POST_RESET, RDC_EVNT_NOTIF_POST_RESET},
};

// This const determines space allocated on stack for notification events.
const uint32_t kMaxRSMIEvents = 64;

RdcNotificationImpl::RdcNotificationImpl() {
}

RdcNotificationImpl::~RdcNotificationImpl() {
}

bool
RdcNotificationImpl::is_notification_event(rdc_field_t field) const {
  if (rdc_2_rsmi_event_notif_map.find(field) ==
                                         rdc_2_rsmi_event_notif_map.end()) {
    return false;
  }
  return true;
}

rdc_status_t
RdcNotificationImpl::set_listen_events(const std::vector<RdcFieldKey> fk_arr) {
  rsmi_status_t ret;
  std::map<uint32_t, uint64_t> new_masks;

  for (uint32_t i = 0; i < fk_arr.size(); ++i) {
    if (rdc_2_rsmi_event_notif_map.find(fk_arr[i].second) ==
                                           rdc_2_rsmi_event_notif_map.end()) {
      continue;
    }
    new_masks[fk_arr[i].first] |=
     RSMI_EVENT_MASK_FROM_INDEX(rdc_2_rsmi_event_notif_map[fk_arr[i].second]);
  }

  std::map<uint32_t, uint64_t>::iterator it = new_masks.begin();

  std::lock_guard<std::mutex> guard(notif_mutex_);
  for (; it != new_masks.end(); ++it) {
    if (it->second == gpu_evnt_notif_masks_[it->first]) {
      // No change to mask; nothing to be done
      continue;
    }
    ret = rsmi_event_notification_init(it->first);
    if (ret != RSMI_STATUS_SUCCESS) {
      RDC_LOG(RDC_INFO,
       "rsmi_event_notification_init() returned " << ret << " for device " <<
                                            it->first << ". " << std::endl <<
                               "  Will not listen for events on this device");
      continue;
    }

    ret = rsmi_event_notification_mask_set(it->first, it->second);

    if (ret == RSMI_STATUS_SUCCESS) {
      gpu_evnt_notif_masks_[it->first] = it->second;
      RDC_LOG(RDC_INFO, "Event notification mask for gpu " << it->first <<
                                    "is set to 0x" << std::hex << it->second);
    } else {
      RDC_LOG(RDC_INFO, "rsmi_event_notification_mask_set() returned " << ret
                                              << " for device " << it->first);
      return Rsmi2RdcError(ret);
    }
  }
  return RDC_ST_OK;
}

// Blocking
rdc_status_t
RdcNotificationImpl::listen(rdc_evnt_notification_t *events,
                                  uint32_t *num_events, uint32_t timeout_ms) {
  if (events == nullptr || *num_events == 0) {
    return RDC_ST_BAD_PARAMETER;
  }

  uint32_t f_cnt = std::min(*num_events, kMaxRSMIEvents);
  rsmi_evt_notification_data_t rsmi_events[kMaxRSMIEvents];

  rsmi_status_t ret =
                 rsmi_event_notification_get(timeout_ms, &f_cnt, rsmi_events);

  if (ret != RSMI_STATUS_SUCCESS) {
    return Rsmi2RdcError(ret);
  }
  struct timeval  tv;
  gettimeofday(&tv, NULL);
  uint64_t now = static_cast<uint64_t>(tv.tv_sec)*1000+tv.tv_usec/1000;
  *num_events = f_cnt;

  for (uint32_t i = 0; i < f_cnt; ++i) {
    assert(rsmi_event_notif_2_rdc_map.find(rsmi_events[i].event) !=
                                            rsmi_event_notif_2_rdc_map.end());
    events[i].gpu_id = rsmi_events[i].dv_ind;
    events[i].field.field_id = rsmi_event_notif_2_rdc_map[rsmi_events[i].event];
    events[i].field.status = RDC_ST_OK;
    events[i].field.ts = now;
    events[i].field.type = STRING;
    strncpy_with_null(events[i].field.value.str,
                        rsmi_events[i].message, RDC_MAX_STR_LENGTH);
  }

  return RDC_ST_OK;
}

rdc_status_t
RdcNotificationImpl::stop_listening(uint32_t gpu_id) {
  rsmi_status_t ret;

  ret = rsmi_event_notification_mask_set(gpu_id, 0);
  if (ret != RSMI_STATUS_SUCCESS) {
    RDC_LOG(RDC_INFO, "rsmi_event_notification_mask_set() returned " << ret
                                                 << " for device " << gpu_id);
  }

  ret = rsmi_event_notification_stop(gpu_id);
  if (ret == RSMI_STATUS_SUCCESS) {
    std::lock_guard<std::mutex> guard(notif_mutex_);
    gpu_evnt_notif_masks_[gpu_id] = 0;
  } else {
    RDC_LOG(RDC_INFO, "rsmi_event_notification_stop() returned " << ret
                                                 << " for device " << gpu_id);
  }
  return RDC_ST_OK;
}


}  // namespace rdc
}  // namespace amd
