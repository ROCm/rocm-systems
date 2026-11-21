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
#ifndef INCLUDE_RDC_LIB_IMPL_RDCMETRICSUPDATERIMPL_H_
#define INCLUDE_RDC_LIB_IMPL_RDCMETRICSUPDATERIMPL_H_

#include <atomic>
#include <chrono>              // NOLINT(build/c++11)
#include <condition_variable>  // NOLINT(build/c++11)
#include <future>              // NOLINT(build/c++11)
#include <memory>
#include <mutex>  // NOLINT(build/c++11)

#include "rdc_lib/RdcMetricsUpdater.h"
#include "rdc_lib/RdcWatchTable.h"

namespace amd {
namespace rdc {

class RdcMetricsUpdaterImpl final : public RdcMetricsUpdater {
 public:
  void start() override;
  // Returns once both threads have finished; start() may be called again after it.
  void stop() override;
  void wake() override;
  // Between two updates the update loop sleeps until the watch table's next update is
  // due, at least check_frequency microseconds and at most max_wait (100 ms with the
  // first constructor), or until wake() or stop().
  explicit RdcMetricsUpdaterImpl(const RdcWatchTablePtr& watch_table,
                                 const uint32_t check_frequency);
  RdcMetricsUpdaterImpl(const RdcWatchTablePtr& watch_table, uint32_t check_frequency,
                        std::chrono::milliseconds max_wait);
  ~RdcMetricsUpdaterImpl();

 private:
  // Sleeps for `period`, or until stop().
  template <typename Duration>
  void pause(Duration period);
  // Sleeps until the next update is due, or until wake() or stop().
  void sleep_until_next_update();

  RdcWatchTablePtr watch_table_;
  std::atomic<bool> started_;
  bool woken_ = false;  // set by wake() under mutex_, cleared by the update loop
  std::mutex mutex_;    // with cv_, wakes the threads from their sleeps
  std::condition_variable cv_;
  std::future<void> updater_;                 // keep the future of updater
  std::future<void> notif_updater_;           // keep the future of notif updater
  const uint32_t _check_frequency;            // shortest sleep between updates, in microseconds
  const std::chrono::milliseconds max_wait_;  // longest sleep between updates
};

}  // namespace rdc
}  // namespace amd

#endif  // INCLUDE_RDC_LIB_IMPL_RDCMETRICSUPDATERIMPL_H_
