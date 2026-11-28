/*
Copyright (c) 2026 - present Advanced Micro Devices, Inc. All rights reserved.

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

// RdcMetricsUpdaterImpl runs the update and notification loops of embedded RDC.
// stop() must return only once both loops have finished, so that nothing calls
// AMD SMI after RDC releases it, and start() must work again after stop(). Between
// updates the update loop sleeps until the watch table's next update is due, but
// no longer than its longest wait, and wake() cuts that sleep short.
// RdcMetricsUpdaterImpl.cc is compiled into this binary with the watch table
// below standing in for RDC's.

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <future>
#include <memory>
#include <mutex>
#include <thread>

#include "rdc_lib/impl/RdcMetricsUpdaterImpl.h"

namespace {

using amd::rdc::RdcMetricsUpdaterImpl;
using std::chrono::milliseconds;

// Counts updates; while held, an update waits for release(). Reports the next
// update as due at once, or after set_next_update_ms().
class TestWatchTable : public amd::rdc::RdcWatchTable {
 public:
  rdc_status_t rdc_field_update_all() override {
    std::unique_lock<std::mutex> lock(mutex_);
    updates_++;
    in_update_ = true;
    cv_.notify_all();
    cv_.wait(lock, [this] { return !held_; });
    in_update_ = false;
    return RDC_ST_OK;
  }
  rdc_status_t rdc_field_listen_notif(uint32_t timeout_ms) override {
    std::lock_guard<std::mutex> lock(mutex_);
    listens_++;
    longest_listen_ms_ = std::max(longest_listen_ms_, timeout_ms);
    cv_.notify_all();
    return RDC_ST_OK;
  }
  uint64_t ms_until_next_update() override {
    std::lock_guard<std::mutex> lock(mutex_);
    return next_update_ms_;
  }

  void hold() {
    std::lock_guard<std::mutex> lock(mutex_);
    held_ = true;
  }
  void release() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      held_ = false;
    }
    cv_.notify_all();
  }
  void set_next_update_ms(uint64_t ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    next_update_ms_ = ms;
  }
  // Waits until an update is running, or more than `count` updates have started.
  bool wait_for_update(int count) {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, std::chrono::seconds(30),
                        [&] { return updates_ > count || (held_ && in_update_); });
  }
  int updates() {
    std::lock_guard<std::mutex> lock(mutex_);
    return updates_;
  }
  bool in_update() {
    std::lock_guard<std::mutex> lock(mutex_);
    return in_update_;
  }
  bool wait_for_listen() {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, std::chrono::seconds(30), [this] { return listens_ > 0; });
  }
  uint32_t longest_listen_ms() {
    std::lock_guard<std::mutex> lock(mutex_);
    return longest_listen_ms_;
  }

  rdc_status_t rdc_job_start_stats(rdc_gpu_group_t, const char[64], uint64_t,
                                   const rdc_gpu_gauges_t&) override {
    return RDC_ST_NOT_SUPPORTED;
  }
  rdc_status_t rdc_job_stop_stats(const char[64], const rdc_gpu_gauges_t&) override {
    return RDC_ST_NOT_SUPPORTED;
  }
  rdc_status_t rdc_job_remove(const char[64]) override { return RDC_ST_NOT_SUPPORTED; }
  rdc_status_t rdc_job_remove_all() override { return RDC_ST_NOT_SUPPORTED; }
  rdc_status_t rdc_field_watch(rdc_gpu_group_t, rdc_field_grp_t, uint64_t, double,
                               uint32_t) override {
    return RDC_ST_NOT_SUPPORTED;
  }
  rdc_status_t rdc_field_unwatch(rdc_gpu_group_t, rdc_field_grp_t) override {
    return RDC_ST_NOT_SUPPORTED;
  }
  rdc_status_t rdc_health_set(rdc_gpu_group_t, unsigned int) override {
    return RDC_ST_NOT_SUPPORTED;
  }
  rdc_status_t rdc_health_get(rdc_gpu_group_t, unsigned int*) override {
    return RDC_ST_NOT_SUPPORTED;
  }
  rdc_status_t rdc_health_check(rdc_gpu_group_t, rdc_health_response_t*) override {
    return RDC_ST_NOT_SUPPORTED;
  }
  rdc_status_t rdc_health_clear(rdc_gpu_group_t) override { return RDC_ST_NOT_SUPPORTED; }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  int updates_ = 0;
  bool in_update_ = false;
  bool held_ = false;
  int listens_ = 0;
  uint32_t longest_listen_ms_ = 0;
  uint64_t next_update_ms_ = 0;
};

TEST(MetricsUpdater, StopWaitsForTheUpdateInProgress) {
  auto table = std::make_shared<TestWatchTable>();
  RdcMetricsUpdaterImpl updater(table, 1000);
  table->hold();
  updater.start();
  ASSERT_TRUE(table->wait_for_update(0)) << "no update started";

  std::atomic<bool> stopped{false};
  std::thread stopper([&] {
    updater.stop();
    stopped = true;
  });
  std::this_thread::sleep_for(milliseconds(300));
  EXPECT_FALSE(stopped) << "stop() returned while an update was still running";
  table->release();
  stopper.join();
  EXPECT_TRUE(stopped);
  EXPECT_FALSE(table->in_update());
}

TEST(MetricsUpdater, StartsAgainAfterStop) {
  auto table = std::make_shared<TestWatchTable>();
  RdcMetricsUpdaterImpl updater(table, 1000);
  updater.start();
  ASSERT_TRUE(table->wait_for_update(0));
  updater.stop();
  int after_stop = table->updates();
  std::this_thread::sleep_for(milliseconds(100));
  EXPECT_EQ(table->updates(), after_stop) << "an update ran after stop() returned";

  // A start() whose threads never see the stop before it waits for them deadlocks.
  std::promise<void> started;
  std::thread starter([&] {
    updater.start();
    started.set_value();
  });
  if (started.get_future().wait_for(std::chrono::seconds(30)) != std::future_status::ready) {
    ADD_FAILURE() << "start() after stop() did not return within 30 s";
    std::_Exit(1);  // the blocked thread cannot be joined
  }
  starter.join();
  EXPECT_TRUE(table->wait_for_update(after_stop)) << "no update after the second start()";
  updater.stop();
}

TEST(MetricsUpdater, DestructorStopsTheLoops) {
  auto table = std::make_shared<TestWatchTable>();
  {
    RdcMetricsUpdaterImpl updater(table, 1000);
    updater.start();
    ASSERT_TRUE(table->wait_for_update(0));
  }
  int after = table->updates();
  std::this_thread::sleep_for(milliseconds(100));
  EXPECT_EQ(table->updates(), after) << "an update ran after the updater was destroyed";
}

// stop() waits for the notification listen in progress, so a listen must be short.
TEST(MetricsUpdater, NotificationListensAreShort) {
  auto table = std::make_shared<TestWatchTable>();
  RdcMetricsUpdaterImpl updater(table, 1000);
  updater.start();
  ASSERT_TRUE(table->wait_for_listen()) << "no notification listen started";
  updater.stop();
  EXPECT_LE(table->longest_listen_ms(), 1000u) << "a listen longer than 1 s delays stop()";
}

// With nothing due, the update loop sleeps for its longest wait (100 ms by default)
// between updates, instead of waking up every check_frequency microseconds.
TEST(MetricsUpdater, SleepsWhileNothingIsDue) {
  auto table = std::make_shared<TestWatchTable>();
  table->set_next_update_ms(60000);
  RdcMetricsUpdaterImpl updater(table, 1000);
  updater.start();
  ASSERT_TRUE(table->wait_for_update(0));
  int first = table->updates();
  std::this_thread::sleep_for(milliseconds(1000));
  int updates = table->updates() - first;
  updater.stop();
  // About 10; a loop that wakes up every millisecond makes about 1000.
  EXPECT_LE(updates, 30) << updates << " updates in 1 s with nothing due";
}

// An update that is due comes on time.
TEST(MetricsUpdater, UpdatesWhenAnUpdateIsDue) {
  auto table = std::make_shared<TestWatchTable>();
  table->set_next_update_ms(20);
  RdcMetricsUpdaterImpl updater(table, 1000);
  updater.start();
  ASSERT_TRUE(table->wait_for_update(0));
  int first = table->updates();
  std::this_thread::sleep_for(milliseconds(1000));
  int updates = table->updates() - first;
  updater.stop();
  // About 50; a loop that always sleeps for 100 ms makes about 10.
  EXPECT_GE(updates, 20) << updates << " updates in 1 s with one due every 20 ms";
}

// wake() runs an update at once, however long the loop would sleep otherwise.
TEST(MetricsUpdater, WakeRunsAnUpdate) {
  auto table = std::make_shared<TestWatchTable>();
  table->set_next_update_ms(60000);
  RdcMetricsUpdaterImpl updater(table, 1000, std::chrono::minutes(1));
  updater.start();
  ASSERT_TRUE(table->wait_for_update(0));
  int first = table->updates();
  updater.wake();
  EXPECT_TRUE(table->wait_for_update(first)) << "no update within 30 s of wake()";
  updater.stop();
}

// A wake() during an update is not lost: another update follows instead of a sleep.
TEST(MetricsUpdater, WakeDuringAnUpdateIsNotLost) {
  auto table = std::make_shared<TestWatchTable>();
  table->set_next_update_ms(60000);
  RdcMetricsUpdaterImpl updater(table, 1000, std::chrono::minutes(1));
  table->hold();
  updater.start();
  ASSERT_TRUE(table->wait_for_update(0)) << "no update started";
  int first = table->updates();
  updater.wake();
  table->release();
  EXPECT_TRUE(table->wait_for_update(first)) << "the wake() during the update was lost";
  updater.stop();
}

// stop() does not wait for the next update to fall due.
TEST(MetricsUpdater, StopDoesNotWaitForTheNextUpdate) {
  auto table = std::make_shared<TestWatchTable>();
  table->set_next_update_ms(60000);
  RdcMetricsUpdaterImpl updater(table, 1000, std::chrono::minutes(1));
  updater.start();
  ASSERT_TRUE(table->wait_for_update(0));
  std::promise<void> stopped;
  std::thread stopper([&] {
    updater.stop();
    stopped.set_value();
  });
  if (stopped.get_future().wait_for(std::chrono::seconds(30)) != std::future_status::ready) {
    ADD_FAILURE() << "stop() did not return within 30 s";
    std::_Exit(1);  // the blocked thread cannot be joined
  }
  stopper.join();
}

}  // namespace
