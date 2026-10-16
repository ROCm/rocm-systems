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

#include <sys/time.h>

#include <algorithm>
#include <cstdint>
#include <ctime>
#include <mutex>  // NOLINT
#include <set>
#include <unordered_map>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "common/rdc_capabilities.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/impl/SmiUtils.h"

namespace amd {
namespace rdc {

static const std::unordered_map<rdc_field_t, amdsmi_evt_notification_type_t>
    rdc_2_smi_event_notif_map = {
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
static const std::unordered_map<amdsmi_evt_notification_type_t, rdc_field_t>
    smi_event_notif_2_rdc_map = {
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

// The mask of a GPU whose events RDC does not know: AMD SMI could not find the GPU when RDC last
// changed its mask, or the mask write failed. AMD SMI may send some of the GPU's events or none.
// No mask that listeners want has every bit set (RDC knows 13 events), so the GPU's next try
// writes its mask again, or with no listener left writes 0 and stops its events.
static const uint64_t kMaskUnknown = ~uint64_t{0};

RdcNotificationImpl::RdcNotificationImpl() {}

RdcNotificationImpl::~RdcNotificationImpl() {}

bool RdcNotificationImpl::is_notification_event(rdc_field_t field) const {
  if (rdc_2_smi_event_notif_map.find(field) == rdc_2_smi_event_notif_map.end()) {
    return false;
  }
  return true;
}

rdc_status_t RdcNotificationImpl::add_event_listeners(const std::vector<RdcFieldKey>& fields) {
  std::lock_guard<std::mutex> guard(notif_mutex_);
  std::set<uint32_t> gpu_ids;
  for (const auto& field : fields) {
    if (!is_notification_event(field.second)) {
      continue;
    }
    gpus_[field.first].listeners[field.second]++;
    gpu_ids.insert(field.first);
  }
  return apply_masks_locked(gpu_ids);
}

rdc_status_t RdcNotificationImpl::remove_event_listeners(const std::vector<RdcFieldKey>& fields) {
  std::lock_guard<std::mutex> guard(notif_mutex_);
  std::set<uint32_t> gpu_ids;
  for (const auto& field : fields) {
    auto gpu = gpus_.find(field.first);
    if (gpu == gpus_.end()) {
      continue;
    }
    auto listener = gpu->second.listeners.find(field.second);
    if (listener == gpu->second.listeners.end()) {
      continue;
    }
    if (--listener->second == 0) {
      gpu->second.listeners.erase(listener);
    }
    gpu_ids.insert(field.first);
  }
  return apply_masks_locked(gpu_ids);
}

rdc_status_t RdcNotificationImpl::apply_masks_locked(const std::set<uint32_t>& gpu_ids) {
  rdc_status_t result = RDC_ST_OK;
  // Every GPU whose mask is not yet the one its listeners want, not only those in gpu_ids: a GPU
  // that was unavailable is retried. A GPU whose last try failed (e.g. on a kernel without SMI
  // events) is tried again only by a call that names it, since each try logs the failure, or
  // once it has no listener left. Errors go to the call that names the GPU.
  for (auto gpu = gpus_.begin(); gpu != gpus_.end();) {
    const bool named = gpu_ids.count(gpu->first) != 0;
    if (named || !gpu->second.failed || gpu->second.listeners.empty()) {
      rdc_status_t ret = apply_mask_locked(gpu->first, &gpu->second, named);
      if (ret != RDC_ST_OK && result == RDC_ST_OK && named) {
        result = ret;
      }
    }
    if (gpu->second.listeners.empty() && gpu->second.mask == 0) {
      gpu = gpus_.erase(gpu);
    } else {
      ++gpu;
    }
  }
  return result;
}

rdc_status_t RdcNotificationImpl::apply_mask_locked(uint32_t gpu_id, GpuListeners* gpu,
                                                    bool named) {
  uint64_t wanted = 0;
  for (const auto& listener : gpu->listeners) {
    wanted |= AMDSMI_EVENT_MASK_FROM_INDEX(rdc_2_smi_event_notif_map.at(listener.first));
  }
  if (wanted == gpu->mask) {
    return RDC_ST_OK;
  }

  // Get processor handle from GPU id
  amdsmi_processor_handle processor_handle;
  amdsmi_status_t ret = get_processor_handle_from_id(gpu_id, &processor_handle);
  if (ret == AMDSMI_STATUS_NOT_FOUND) {
    // An unavailable GPU sends no events; its mask is written once a later call finds it. Not a
    // failure: every call tries it again.
    RDC_LOG(RDC_DEBUG, "GPU " << gpu_id << " is unavailable; not changing its event mask");
    if (gpu->mask != 0) {
      gpu->mask = kMaskUnknown;
    }
    gpu->failed = false;
    return RDC_ST_OK;
  }
  if (ret != AMDSMI_STATUS_SUCCESS) {
    // Without listeners, every call tries the GPU again, only to stop its events: the tries of
    // calls that do not name it are logged as debug messages.
    RDC_LOG(named || !gpu->listeners.empty() ? RDC_ERROR : RDC_DEBUG,
            "Failed to get processor handle for GPU " << gpu_id << " error: " << ret);
    gpu->failed = true;
    return Smi2RdcError(ret);
  }

  if (wanted == 0) {
    // No listener left. Each call leaves AMD SMI without events for the GPU, and a stop that
    // fails may have closed them anyway, so the next listener initializes them again.
    rdc_status_t result = RDC_ST_OK;
    ret = amdsmi_set_gpu_event_notification_mask(processor_handle, 0);
    if (ret != AMDSMI_STATUS_SUCCESS) {
      RDC_LOG(RDC_ERROR, "amdsmi_set_gpu_event_notification_mask() returned "
                             << ret << " for device " << gpu_id);
      result = Smi2RdcError(ret);
    }
    ret = amdsmi_stop_gpu_event_notification(processor_handle);
    if (ret != AMDSMI_STATUS_SUCCESS) {
      RDC_LOG(RDC_ERROR,
              "amdsmi_stop_gpu_event_notification() returned " << ret << " for device " << gpu_id);
      if (result == RDC_ST_OK) {
        result = Smi2RdcError(ret);
      }
    }
    gpu->mask = 0;
    return result;
  }

  // Temporarily get DAC capability
  ScopedCapability sc(CAP_DAC_OVERRIDE, CAP_EFFECTIVE);

  if (sc.error()) {
    RDC_LOG(RDC_ERROR, "Failed to acquire required capabilities. Errno " << sc.error());
    gpu->failed = true;
    return RDC_ST_PERM_ERROR;
  }

  ret = amdsmi_init_gpu_event_notification(processor_handle);
  if (ret != AMDSMI_STATUS_SUCCESS) {
    RDC_LOG(RDC_ERROR, "amdsmi_init_gpu_event_notification() returned "
                           << ret << " for device " << gpu_id << ". " << std::endl
                           << "  Will not listen for events on this device");
    gpu->failed = true;
    // Release DAC capability
    sc.Relinquish();
    if (sc.error()) {
      RDC_LOG(RDC_ERROR, "Failed to relinquish capabilities. Errno " << sc.error());
      return RDC_ST_PERM_ERROR;
    }
    return RDC_ST_OK;
  }

  ret = amdsmi_set_gpu_event_notification_mask(processor_handle, wanted);
  // Release DAC capability
  sc.Relinquish();

  // The GPU's events are open now. After a failed write, the mask AMD SMI has is not known.
  gpu->mask = ret == AMDSMI_STATUS_SUCCESS ? wanted : kMaskUnknown;
  gpu->failed = ret != AMDSMI_STATUS_SUCCESS;
  // Before the release check, which returns first: a failed write is logged also when the
  // release fails.
  if (ret == AMDSMI_STATUS_SUCCESS) {
    RDC_LOG(RDC_INFO, "Event notification mask for gpu " << gpu_id << " is set to 0x" << std::hex
                                                         << wanted << std::dec);
  } else {
    RDC_LOG(RDC_ERROR, "amdsmi_set_gpu_event_notification_mask() returned " << ret << " for device "
                                                                            << gpu_id);
  }
  if (sc.error()) {
    RDC_LOG(RDC_ERROR, "Failed to relinquish capabilities. Errno " << sc.error());
    return RDC_ST_PERM_ERROR;
  }
  if (ret != AMDSMI_STATUS_SUCCESS) {
    return Smi2RdcError(ret);
  }
  return RDC_ST_OK;
}

// Blocking
rdc_status_t RdcNotificationImpl::listen(rdc_evnt_notification_t* events, uint32_t* num_events,
                                         uint32_t timeout_ms) {
  if (events == nullptr || *num_events == 0) {
    return RDC_ST_BAD_PARAMETER;
  }

  const uint32_t room = std::min(*num_events, kMaxRSMIEvents);
  uint32_t f_cnt = room;
  // Zeroed: an entry that AMD SMI did not write has event 0 and no handle, and is skipped below.
  amdsmi_evt_notification_data_t smi_events[kMaxRSMIEvents] = {};

  amdsmi_status_t ret = amdsmi_get_gpu_event_notification(timeout_ms, &f_cnt, smi_events);

  if (ret != AMDSMI_STATUS_SUCCESS) {
    *num_events = 0;
    return Smi2RdcError(ret);
  }
  if (f_cnt > room) {
    // AMD SMI reports the events it wrote, which fit in the room it was given.
    RDC_LOG(RDC_ERROR, "amdsmi_get_gpu_event_notification() reported "
                           << f_cnt << " events, room for " << room);
    f_cnt = room;
  }
  struct timeval tv;
  gettimeofday(&tv, NULL);
  uint64_t now = static_cast<uint64_t>(tv.tv_sec) * 1000 + tv.tv_usec / 1000;

  uint32_t written = 0;
  for (uint32_t i = 0; i < f_cnt; ++i) {
    // Skip an event that RDC cannot report, and keep the others.
    auto field = smi_event_notif_2_rdc_map.find(smi_events[i].event);
    if (field == smi_event_notif_2_rdc_map.end()) {
      RDC_LOG(RDC_DEBUG, "Skipping event " << smi_events[i].event << ", which RDC does not know");
      continue;
    }
    uint32_t gpu_id;
    ret = get_gpu_id_from_processor_handle(smi_events[i].processor_handle, &gpu_id);
    if (ret != AMDSMI_STATUS_SUCCESS) {
      RDC_LOG(RDC_DEBUG,
              "Skipping event " << smi_events[i].event << " of a GPU that RDC does not list");
      continue;
    }
    rdc_evnt_notification_t& event = events[written++];
    event.gpu_id = gpu_id;
    event.field.field_id = field->second;
    event.field.status = RDC_ST_OK;
    event.field.type = STRING;
    strncpy_with_null(event.field.value.str, smi_events[i].message, RDC_MAX_STR_LENGTH);
  }
  // Each event gets a timestamp of its own, later than the last event's:
  // rdc_field_get_value_since() tells the samples of a field apart by their timestamps.
  std::lock_guard<std::mutex> guard(notif_mutex_);
  for (uint32_t i = 0; i < written; ++i) {
    last_event_ts_ = std::max(now, last_event_ts_ + 1);
    events[i].field.ts = last_event_ts_;
  }
  *num_events = written;

  return RDC_ST_OK;
}

}  // namespace rdc
}  // namespace amd
