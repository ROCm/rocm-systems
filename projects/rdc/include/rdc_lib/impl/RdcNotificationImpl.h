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
#include <semaphore>
#include <vector>

#include "rdc/rdc.h"
#include "rdc_lib/RdcNotification.h"
#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {

// Maximum concurrent AMD SMI event notification calls
// Increase this value if more concurrent notification calls are needed
static constexpr std::size_t kMaxConcurrentNotifCalls = 4;

// RAII guard for counting_semaphore to ensure release on all exit paths
class ScopedSemaphore {
  std::counting_semaphore<kMaxConcurrentNotifCalls>& sem_;
  bool acquired_;

 public:
  ScopedSemaphore(ScopedSemaphore&&) = delete;
  ScopedSemaphore& operator=(ScopedSemaphore&&) = delete;
  // NOTE: try_acquire() is non-blocking.
  // this can be changed to timeout based try_acquire_for() if needed.
  explicit ScopedSemaphore(std::counting_semaphore<kMaxConcurrentNotifCalls>& sem)
      : sem_(sem), acquired_(sem.try_acquire()) {}
  ~ScopedSemaphore() {
    if (acquired_) sem_.release();
  }
  bool acquired() const { return acquired_; }
  ScopedSemaphore(const ScopedSemaphore&) = delete;
  ScopedSemaphore& operator=(const ScopedSemaphore&) = delete;
};

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
  std::counting_semaphore<kMaxConcurrentNotifCalls> notif_sem_{kMaxConcurrentNotifCalls};
};

}  // namespace rdc
}  // namespace amd

#endif  // INCLUDE_RDC_LIB_IMPL_RDCNOTIFICATIONIMPL_H_
