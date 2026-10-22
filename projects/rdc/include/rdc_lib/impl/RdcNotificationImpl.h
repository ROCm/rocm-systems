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
#ifndef INCLUDE_RDC_LIB_IMPL_RDCNOTIFICATIONIMPL_H_
#define INCLUDE_RDC_LIB_IMPL_RDCNOTIFICATIONIMPL_H_

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <vector>

#include "rdc/rdc.h"
#include "rdc_lib/RdcNotification.h"
#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {

class RdcNotificationImpl : public RdcNotification {
 public:
  RdcNotificationImpl();
  ~RdcNotificationImpl();

  bool is_notification_event(rdc_field_t field) const override;
  rdc_status_t add_event_listeners(const std::vector<RdcFieldKey>& fields) override;
  rdc_status_t remove_event_listeners(const std::vector<RdcFieldKey>& fields) override;
  // Blocking
  rdc_status_t listen(rdc_evnt_notification_t* events, uint32_t* num_events,
                      uint32_t timeout_ms) override;

 private:
  // One GPU's listeners, and the event mask that RDC last left AMD SMI with for the GPU: 0 for
  // none, all bits set when RDC does not know it (kMaskUnknown).
  struct GpuListeners {
    std::map<rdc_field_t, uint32_t> listeners;  // event field -> listener count
    uint64_t mask = 0;
    // The last try for the GPU failed. Whether calls that do not name the GPU try it depends on
    // this only while it has listeners, and then every try is for a nonzero mask.
    bool failed = false;
  };

  // Gives GPUs the mask their listeners want: those in gpu_ids, and the others unless their last
  // try failed and they still have listeners. Returns the first error for a GPU in gpu_ids.
  rdc_status_t apply_masks_locked(const std::set<uint32_t>& gpu_ids);
  // named: the call names the GPU. A lookup error is logged as an error, except in the tries that
  // calls not naming the GPU make of a GPU without listeners (every call makes one).
  rdc_status_t apply_mask_locked(uint32_t gpu_id, GpuListeners* gpu, bool named);

  std::map<uint32_t, GpuListeners> gpus_;
  uint64_t last_event_ts_ = 0;  // the timestamp listen() gave the last event, in ms
  std::mutex notif_mutex_;
};

}  // namespace rdc
}  // namespace amd

#endif  // INCLUDE_RDC_LIB_IMPL_RDCNOTIFICATIONIMPL_H_
