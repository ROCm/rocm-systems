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

#include <pthread.h>
#include <sys/time.h>

#include <algorithm>
#include <chrono>  // NOLINT(build/c++11)
#include <ctime>
#include <thread>

#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {

// The longest the update loop sleeps with the first constructor: a watch that nothing
// woke the loop for falls due at most this late.
static constexpr std::chrono::milliseconds kMaxUpdateWait(100);

RdcMetricsUpdaterImpl::RdcMetricsUpdaterImpl(const RdcWatchTablePtr& watch_table,
                                             const uint32_t check_frequency)
    : RdcMetricsUpdaterImpl(watch_table, check_frequency, kMaxUpdateWait) {}

RdcMetricsUpdaterImpl::RdcMetricsUpdaterImpl(const RdcWatchTablePtr& watch_table,
                                             uint32_t check_frequency,
                                             std::chrono::milliseconds max_wait)
    : watch_table_(watch_table),
      started_(false),
      _check_frequency(check_frequency),
      max_wait_(max_wait) {}

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

void RdcMetricsUpdaterImpl::sleep_until_next_update() {
  const uint64_t due_ms = watch_table_->ms_until_next_update();
  const auto period = std::max<std::chrono::microseconds>(
      std::chrono::milliseconds(std::min<uint64_t>(due_ms, max_wait_.count())),
      std::chrono::microseconds(_check_frequency));
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait_for(lock, period, [this] { return !started_ || woken_; });
  woken_ = false;
}

void RdcMetricsUpdaterImpl::wake() {
  {
    // Under the mutex, so a wake() between an update and the sleep after it is not lost.
    std::lock_guard<std::mutex> lock(mutex_);
    woken_ = true;
  }
  cv_.notify_all();
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
    // Named for tools such as top -H, and for the test of how often it wakes up.
    pthread_setname_np(pthread_self(), "rdc_update");
    while (started_) {
      watch_table_->rdc_field_update_all();
      sleep_until_next_update();
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
