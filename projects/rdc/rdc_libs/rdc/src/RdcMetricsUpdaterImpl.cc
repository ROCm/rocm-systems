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
#include "rdc_lib/impl/RdcMetricsUpdaterImpl.h"

#include <sys/time.h>

#include <chrono>  // NOLINT(build/c++11)
#include <ctime>
#include <thread>

#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {

RdcMetricsUpdaterImpl::RdcMetricsUpdaterImpl(const RdcWatchTablePtr& watch_table,
                                             const uint32_t check_frequency)
    : watch_table_(watch_table), started_(false), _check_frequency(check_frequency) {}

// The longest one notification listen blocks, and so the longest stop() waits for the
// notification thread. Events that arrive between listens wait in AMD SMI's queue.
static const uint32_t kRdcFieldListenNotifTime_mS = 1000;
static const uint32_t kRdcEventCheck_ms = 1000;

RdcMetricsUpdaterImpl::~RdcMetricsUpdaterImpl() { stop(); }

template <typename Duration>
void RdcMetricsUpdaterImpl::pause(Duration period) {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait_for(lock, period, [this] { return !started_; });
}

void RdcMetricsUpdaterImpl::start() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (started_) {
    return;
  }
  started_ = true;
  notif_updater_ = std::async(std::launch::async, [this]() {
    while (started_) {
      watch_table_->rdc_field_listen_notif(kRdcFieldListenNotifTime_mS);
      pause(std::chrono::milliseconds(kRdcEventCheck_ms));
    }
  });
  updater_ = std::async(std::launch::async, [this]() {
    while (started_) {
      watch_table_->rdc_field_update_all();
      pause(std::chrono::microseconds(_check_frequency));
    }
  });
}

void RdcMetricsUpdaterImpl::stop() {
  {
    // Under the mutex, so a thread between its check and its wait cannot miss the wake-up.
    std::lock_guard<std::mutex> lock(mutex_);
    started_ = false;
  }
  cv_.notify_all();
  // A notification listen in progress ends after its timeout (kRdcFieldListenNotifTime_mS).
  if (updater_.valid()) {
    updater_.wait();
  }
  if (notif_updater_.valid()) {
    notif_updater_.wait();
  }
}

}  // namespace rdc
}  // namespace amd
