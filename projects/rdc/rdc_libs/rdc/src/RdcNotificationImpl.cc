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
#include "rdc_lib/impl/RdcNotificationImpl.h"

#include <assert.h>
#include <sys/time.h>

#include <cstdint>
#include <ctime>
#include <mutex>  // NOLINT
#include <unordered_map>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "common/rdc_capabilities.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/impl/SmiUtils.h"

namespace amd {
namespace rdc {

static std::unordered_map<rdc_field_t, amdsmi_evt_notification_type_t> rdc_2_smi_event_notif_map = {
    {RDC_EVNT_NOTIF_VMFAULT, AMDSMI_EVT_NOTIF_VMFAULT},
    {RDC_EVNT_NOTIF_FIRST, AMDSMI_EVT_NOTIF_FIRST},
    {RDC_EVNT_NOTIF_THERMAL_THROTTLE, AMDSMI_EVT_NOTIF_THERMAL_THROTTLE},
    {RDC_EVNT_NOTIF_PRE_RESET, AMDSMI_EVT_NOTIF_GPU_PRE_RESET},
    {RDC_EVNT_NOTIF_POST_RESET, AMDSMI_EVT_NOTIF_GPU_POST_RESET},
    {RDC_EVNT_NOTIF_MIGRATE_START, AMDSMI_EVT_NOTIF_MIGRATE_START},
    {RDC_EVNT_NOTIF_MIGRATE_END, AMDSMI_EVT_NOTIF_MIGRATE_END},
    {RDC_EVNT_NOTIF_PAGE_FAULT_START, AMDSMI_EVT_NOTIF_PAGE_FAULT_START},
    {RDC_EVNT_NOTIF_PAGE_FAULT_END, AMDSMI_EVT_NOTIF_PAGE_FAULT_END},
    {RDC_EVNT_NOTIF_QUEUE_EVICTION, AMDSMI_EVT_NOTIF_QUEUE_EVICTION},
    {RDC_EVNT_NOTIF_QUEUE_RESTORE, AMDSMI_EVT_NOTIF_QUEUE_RESTORE},
    {RDC_EVNT_NOTIF_UNMAP_FROM_GPU, AMDSMI_EVT_NOTIF_UNMAP_FROM_GPU},
    {RDC_EVNT_NOTIF_PROCESS_START, AMDSMI_EVT_NOTIF_PROCESS_START},
    {RDC_EVNT_NOTIF_PROCESS_END, AMDSMI_EVT_NOTIF_PROCESS_END},
};
static std::unordered_map<amdsmi_evt_notification_type_t, rdc_field_t> smi_event_notif_2_rdc_map = {
    {AMDSMI_EVT_NOTIF_VMFAULT, RDC_EVNT_NOTIF_VMFAULT},
    {AMDSMI_EVT_NOTIF_FIRST, RDC_EVNT_NOTIF_FIRST},
    {AMDSMI_EVT_NOTIF_THERMAL_THROTTLE, RDC_EVNT_NOTIF_THERMAL_THROTTLE},
    {AMDSMI_EVT_NOTIF_GPU_PRE_RESET, RDC_EVNT_NOTIF_PRE_RESET},
    {AMDSMI_EVT_NOTIF_GPU_POST_RESET, RDC_EVNT_NOTIF_POST_RESET},
    {AMDSMI_EVT_NOTIF_MIGRATE_START, RDC_EVNT_NOTIF_MIGRATE_START},
    {AMDSMI_EVT_NOTIF_MIGRATE_END, RDC_EVNT_NOTIF_MIGRATE_END},
    {AMDSMI_EVT_NOTIF_PAGE_FAULT_START, RDC_EVNT_NOTIF_PAGE_FAULT_START},
    {AMDSMI_EVT_NOTIF_PAGE_FAULT_END, RDC_EVNT_NOTIF_PAGE_FAULT_END},
    {AMDSMI_EVT_NOTIF_QUEUE_EVICTION, RDC_EVNT_NOTIF_QUEUE_EVICTION},
    {AMDSMI_EVT_NOTIF_QUEUE_RESTORE, RDC_EVNT_NOTIF_QUEUE_RESTORE},
    {AMDSMI_EVT_NOTIF_UNMAP_FROM_GPU, RDC_EVNT_NOTIF_UNMAP_FROM_GPU},
    {AMDSMI_EVT_NOTIF_PROCESS_START, RDC_EVNT_NOTIF_PROCESS_START},
    {AMDSMI_EVT_NOTIF_PROCESS_END, RDC_EVNT_NOTIF_PROCESS_END},
};

// This const determines space allocated on stack for notification events.
const uint32_t kMaxRSMIEvents = 64;

RdcNotificationImpl::RdcNotificationImpl() {}

RdcNotificationImpl::~RdcNotificationImpl() {}

bool RdcNotificationImpl::is_notification_event(rdc_field_t field) const {
  if (rdc_2_smi_event_notif_map.find(field) == rdc_2_smi_event_notif_map.end()) {
    return false;
  }
  return true;
}

rdc_status_t RdcNotificationImpl::set_listen_events(const std::vector<RdcFieldKey> fk_arr) {
  amdsmi_status_t ret;
  std::map<uint32_t, uint64_t> new_masks;

  for (uint32_t i = 0; i < fk_arr.size(); ++i) {
    if (rdc_2_smi_event_notif_map.find(fk_arr[i].second) == rdc_2_smi_event_notif_map.end()) {
      continue;
    }
    new_masks[fk_arr[i].first] |=
        AMDSMI_EVENT_MASK_FROM_INDEX(rdc_2_smi_event_notif_map[fk_arr[i].second]);
  }

  std::map<uint32_t, uint64_t>::iterator it = new_masks.begin();

  std::lock_guard<std::mutex> guard(notif_mutex_);
  for (; it != new_masks.end(); ++it) {
    if (it->second == gpu_evnt_notif_masks_[it->first]) {
      // No change to mask; nothing to be done
      continue;
    }

    // Get processor handle from GPU id
    amdsmi_processor_handle processor_handle;
    ret = get_processor_handle_from_id(it->first, &processor_handle);
    if (ret != AMDSMI_STATUS_SUCCESS) {
      RDC_LOG(RDC_ERROR,
              "Failed to get processor handle for GPU " << it->first << " error: " << ret);
      return Smi2RdcError(ret);
    }

    // Temporarily get DAC capability
    ScopedCapability sc(CAP_DAC_OVERRIDE, CAP_EFFECTIVE);

    if (sc.error()) {
      RDC_LOG(RDC_ERROR, "Failed to acquire required capabilities. Errno " << sc.error());
      return RDC_ST_PERM_ERROR;
    }

    ret = amdsmi_init_gpu_event_notification(processor_handle);
    if (ret != AMDSMI_STATUS_SUCCESS) {
      RDC_LOG(RDC_ERROR, "amdsmi_init_gpu_event_notification() returned "
                             << ret << " for device " << it->first << ". " << std::endl
                             << "  Will not listen for events on this device");
      continue;
    }

    ret = amdsmi_set_gpu_event_notification_mask(processor_handle, it->second);
    // Release DAC capability
    sc.Relinquish();

    if (sc.error()) {
      RDC_LOG(RDC_ERROR, "Failed to relinquish capabilities. Errno " << sc.error());
      return RDC_ST_PERM_ERROR;
    }

    if (ret == AMDSMI_STATUS_SUCCESS) {
      gpu_evnt_notif_masks_[it->first] = it->second;
      RDC_LOG(RDC_INFO, "Event notification mask for gpu " << it->first << "is set to 0x"
                                                           << std::hex << it->second);
    } else {
      RDC_LOG(RDC_INFO, "amdsmi_set_gpu_event_notification_mask() returned "
                            << ret << " for device " << it->first);
      return Smi2RdcError(ret);
    }
  }
  return RDC_ST_OK;
}

// Blocking
rdc_status_t RdcNotificationImpl::listen(rdc_evnt_notification_t* events, uint32_t* num_events,
                                         uint32_t timeout_ms) {
  if (events == nullptr || *num_events == 0) {
    return RDC_ST_BAD_PARAMETER;
  }

  uint32_t f_cnt = std::min(*num_events, kMaxRSMIEvents);
  amdsmi_evt_notification_data_t smi_events[kMaxRSMIEvents];

  amdsmi_status_t ret = amdsmi_get_gpu_event_notification(timeout_ms, &f_cnt, smi_events);

  if (ret != AMDSMI_STATUS_SUCCESS) {
    return Smi2RdcError(ret);
  }
  struct timeval tv;
  gettimeofday(&tv, NULL);
  uint64_t now = static_cast<uint64_t>(tv.tv_sec) * 1000 + tv.tv_usec / 1000;
  *num_events = f_cnt;

  for (uint32_t i = 0; i < f_cnt; ++i) {
    assert(smi_event_notif_2_rdc_map.find(smi_events[i].event) != smi_event_notif_2_rdc_map.end());
    uint32_t gpu_id;
    ret = get_gpu_id_from_processor_handle(smi_events[i].processor_handle, &gpu_id);
    if (ret != AMDSMI_STATUS_SUCCESS) {
      return Smi2RdcError(ret);
    }
    events[i].gpu_id = gpu_id;
    events[i].field.field_id = smi_event_notif_2_rdc_map[smi_events[i].event];
    events[i].field.status = RDC_ST_OK;
    events[i].field.ts = now;
    events[i].field.type = STRING;
    strncpy_with_null(events[i].field.value.str, smi_events[i].message, RDC_MAX_STR_LENGTH);
  }

  return RDC_ST_OK;
}

rdc_status_t RdcNotificationImpl::stop_listening(uint32_t gpu_id) {
  amdsmi_status_t ret;

  // Get processor handle from GPU id
  amdsmi_processor_handle processor_handle;
  ret = get_processor_handle_from_id(gpu_id, &processor_handle);
  if (ret != AMDSMI_STATUS_SUCCESS) {
    RDC_LOG(RDC_ERROR, "Failed to get processor handle for GPU " << gpu_id << " error: " << ret);
    return Smi2RdcError(ret);
  }

  ret = amdsmi_set_gpu_event_notification_mask(processor_handle, 0);
  if (ret != AMDSMI_STATUS_SUCCESS) {
    RDC_LOG(RDC_ERROR, "amdsmi_set_gpu_event_notification_mask() returned " << ret << " for device "
                                                                            << gpu_id);
  }

  ret = amdsmi_stop_gpu_event_notification(processor_handle);
  if (ret == AMDSMI_STATUS_SUCCESS) {
    std::lock_guard<std::mutex> guard(notif_mutex_);
    gpu_evnt_notif_masks_[gpu_id] = 0;
  } else {
    RDC_LOG(RDC_ERROR,
            "amdsmi_stop_gpu_event_notification() returned " << ret << " for device " << gpu_id);
  }
  return RDC_ST_OK;
}

}  // namespace rdc
}  // namespace amd
