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

#include <map>
#include <memory>
#include <mutex>
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
  rdc_status_t set_listen_events(const std::vector<RdcFieldKey> fk_arr) override;
  // Blocking
  rdc_status_t listen(rdc_evnt_notification_t* events, uint32_t* num_events,
                      uint32_t timeout_ms) override;
  rdc_status_t stop_listening(uint32_t gpu_id) override;

 private:
  std::map<uint32_t, uint64_t> gpu_evnt_notif_masks_;
  std::mutex notif_mutex_;
};

}  // namespace rdc
}  // namespace amd

#endif  // INCLUDE_RDC_LIB_IMPL_RDCNOTIFICATIONIMPL_H_
