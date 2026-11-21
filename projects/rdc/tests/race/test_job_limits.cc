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

// Job calls on RdcWatchTableImpl from several threads. A start checks for the
// job and enters it as one step, so two starts of one job ID never both
// succeed; a remove waits for a start under way, so the ID stays usable; and
// a start waits for a stop of its ID under way. Starts at the job limit remove
// stopped jobs while other threads read job stats and update the job fields,
// and the limit holds afterwards.
//
// RdcWatchTableImpl.cc, RdcCacheManagerImpl.cc and RdcLogger.cc are compiled
// into this binary; the groups, metric fetcher, modules, telemetry and
// notifications they use are the fakes below, and the cache can hold a start.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <ctime>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "rdc/rdc.h"
#include "rdc_lib/RdcGroupSettings.h"
#include "rdc_lib/RdcMetricFetcher.h"
#include "rdc_lib/RdcModuleMgr.h"
#include "rdc_lib/RdcNotification.h"
#include "rdc_lib/RdcTelemetry.h"
#include "rdc_lib/impl/RdcCacheManagerImpl.h"
#include "rdc_lib/impl/RdcWatchTableImpl.h"
#include "rdc_lib/rdc_common.h"

// Defined in RdcBootStrap.cc and RdcNotificationImpl.cc, which are not
// compiled in: a race test that compiles either in must drop the copy here.
char* strncpy_with_null(char* dest, const char* src, size_t n) {
  if (n == 0) {
    return dest;
  }
  strncpy(dest, src, n - 1);
  dest[n - 1] = '\0';
  return dest;
}

namespace amd {
namespace rdc {
const uint32_t kMaxRSMIEvents = 64;
}  // namespace rdc
}  // namespace amd

namespace {

constexpr int kMaxJobs = static_cast<int>(amd::rdc::RdcCacheManagerImpl::kMaxJobs);
// Job fields are due for an update every millisecond.
constexpr uint64_t kUpdateFreqUs = 1000;

// Every GPU group holds GPU 0, and the job field group holds one field.
class AnyGroup : public amd::rdc::RdcGroupSettings {
 public:
  rdc_status_t rdc_group_gpu_get_info(rdc_gpu_group_t, rdc_group_info_t* info) override {
    std::memset(info, 0, sizeof(*info));
    info->count = 1;
    info->entity_ids[0] = 0;
    return RDC_ST_OK;
  }
  rdc_status_t rdc_group_field_get_info(rdc_field_grp_t, rdc_field_group_info_t* info) override {
    std::memset(info, 0, sizeof(*info));
    info->count = 1;
    info->field_ids[0] = RDC_FI_GPU_CLOCK;
    return RDC_ST_OK;
  }
  rdc_status_t rdc_group_gpu_create(const char*, rdc_gpu_group_t*) override {
    return RDC_ST_NOT_SUPPORTED;
  }
  rdc_status_t rdc_group_gpu_destroy(rdc_gpu_group_t) override { return RDC_ST_NOT_SUPPORTED; }
  rdc_status_t rdc_group_gpu_add(rdc_gpu_group_t, uint32_t) override {
    return RDC_ST_NOT_SUPPORTED;
  }
  rdc_status_t rdc_group_get_all_ids(rdc_gpu_group_t[], uint32_t*) override {
    return RDC_ST_NOT_SUPPORTED;
  }
  rdc_status_t rdc_group_field_create(uint32_t, rdc_field_t*, const char*,
                                      rdc_field_grp_t*) override {
    return RDC_ST_NOT_SUPPORTED;
  }
  rdc_status_t rdc_group_field_add_field(rdc_field_grp_t, rdc_field_t) override {
    return RDC_ST_NOT_SUPPORTED;
  }
  rdc_status_t rdc_group_field_destroy(rdc_field_grp_t) override { return RDC_ST_NOT_SUPPORTED; }
  rdc_status_t rdc_group_field_get_all_ids(rdc_field_grp_t[], uint32_t*) override {
    return RDC_ST_NOT_SUPPORTED;
  }
};

// The telemetry module below answers every read; nothing reads a GPU here.
class NoReads : public amd::rdc::RdcMetricFetcher {
 public:
  rdc_status_t acquire_smi_handle(RdcFieldKey) override { return RDC_ST_OK; }
  rdc_status_t delete_smi_handle(RdcFieldKey) override { return RDC_ST_OK; }
  rdc_status_t fetch_smi_field(uint32_t, rdc_field_t, rdc_field_value*) override {
    return RDC_ST_NOT_SUPPORTED;
  }
  rdc_status_t bulk_fetch_smi_fields(rdc_gpu_field_t*, uint32_t,
                                     std::vector<rdc_gpu_field_value_t>&) override {
    return RDC_ST_NOT_SUPPORTED;
  }
};

// RDC_FI_GPU_CLOCK, the job field of every group here, reads 100 on any GPU.
class OneField : public amd::rdc::RdcTelemetry {
 public:
  rdc_status_t rdc_telemetry_fields_query(uint32_t field_ids[MAX_NUM_FIELDS],
                                          uint32_t* field_count) override {
    field_ids[0] = RDC_FI_GPU_CLOCK;
    *field_count = 1;
    return RDC_ST_OK;
  }
  rdc_status_t rdc_telemetry_fields_value_get(rdc_gpu_field_t* fields, uint32_t fields_count,
                                              rdc_field_value_f callback,
                                              void* user_data) override {
    std::vector<rdc_gpu_field_value_t> values(fields_count);
    for (uint32_t i = 0; i < fields_count; ++i) {
      values[i].gpu_index = fields[i].gpu_index;
      values[i].field_value.field_id = fields[i].field_id;
      values[i].field_value.status = RDC_ST_OK;
      values[i].field_value.ts = static_cast<uint64_t>(time(nullptr)) * 1000000;
      values[i].field_value.type = INTEGER;
      values[i].field_value.value.l_int = 100;
    }
    return callback(values.data(), fields_count, user_data);
  }
  rdc_status_t rdc_telemetry_fields_watch(rdc_gpu_field_t*, uint32_t) override { return RDC_ST_OK; }
  rdc_status_t rdc_telemetry_fields_unwatch(rdc_gpu_field_t*, uint32_t) override {
    return RDC_ST_OK;
  }
};

class OneModule : public amd::rdc::RdcModuleMgr {
 public:
  amd::rdc::RdcTelemetryPtr get_telemetry_module() override { return telemetry_; }
  amd::rdc::RdcDiagnosticPtr get_diagnostic_module() override { return nullptr; }

 private:
  amd::rdc::RdcTelemetryPtr telemetry_ = std::make_shared<OneField>();
};

class NoEvents : public amd::rdc::RdcNotification {
 public:
  bool is_notification_event(rdc_field_t) const override { return false; }
  rdc_status_t set_listen_events(const std::vector<RdcFieldKey>) override { return RDC_ST_OK; }
  rdc_status_t listen(amd::rdc::rdc_evnt_notification_t*, uint32_t* num_events, uint32_t) override {
    *num_events = 0;
    return RDC_ST_OK;
  }
  rdc_status_t stop_listening(uint32_t) override { return RDC_ST_OK; }
};

// When asked, holds the next job start right after the cache entered the job,
// before the watch table does, or the next job stop right before the cache
// stops the job's stats, after the watch table erased the job. The hold lasts
// until another thread says go (at most a second), and 10 ms more.
class PausingCache : public amd::rdc::RdcCacheManagerImpl {
 public:
  rdc_status_t rdc_job_start_stats(const char job_id[64], const rdc_group_info_t& group,
                                   const rdc_field_group_info_t& finfo,
                                   const rdc_gpu_gauges_t& gpu_gauges) override {
    rdc_status_t result =
        RdcCacheManagerImpl::rdc_job_start_stats(job_id, group, finfo, gpu_gauges);
    if (hold_next_start.exchange(false)) {
      hold();
    }
    return result;
  }
  rdc_status_t rdc_job_stop_stats(const char job_id[64],
                                  const rdc_gpu_gauges_t& gpu_gauge) override {
    if (hold_next_stop.exchange(false)) {
      hold();
    }
    return RdcCacheManagerImpl::rdc_job_stop_stats(job_id, gpu_gauge);
  }

  std::atomic<bool> hold_next_start{false};
  std::atomic<bool> hold_next_stop{false};
  std::atomic<bool> held{false};
  std::atomic<bool> go{false};

 private:
  void hold() {
    held = true;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (!go && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::yield();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
};

class JobRace : public ::testing::Test {
 protected:
  rdc_status_t start(rdc_gpu_group_t group, const std::string& job) {
    return table_.rdc_job_start_stats(group, Id(job).c, kUpdateFreqUs, gauges_);
  }
  rdc_status_t stop(const std::string& job) {
    return table_.rdc_job_stop_stats(Id(job).c, gauges_);
  }
  rdc_status_t remove(const std::string& job) { return table_.rdc_job_remove(Id(job).c); }
  rdc_status_t remove_all() { return table_.rdc_job_remove_all(); }
  void update() { table_.rdc_field_update_all(); }
  PausingCache& cache() { return *cache_; }
  rdc_status_t get(const std::string& job) {
    auto info = std::make_unique<rdc_job_info_t>();
    return cache_->rdc_job_get_stats(Id(job).c, gauges_, info.get());
  }
  // The end of the job's stats: now while it runs.
  uint64_t end_time(const std::string& job) {
    auto info = std::make_unique<rdc_job_info_t>();
    EXPECT_EQ(cache_->rdc_job_get_stats(Id(job).c, gauges_, info.get()), RDC_ST_OK);
    return info->summary.end_time;
  }

 private:
  struct Id {
    explicit Id(const std::string& job) { job.copy(c, sizeof(c) - 1); }
    char c[64] = {};
  };

  // rdc_job_get_stats() needs each GPU's total memory.
  const rdc_gpu_gauges_t gauges_{{{0, RDC_FI_GPU_MEMORY_TOTAL}, 1 << 30}};
  std::shared_ptr<PausingCache> cache_ = std::make_shared<PausingCache>();
  amd::rdc::RdcWatchTableImpl table_{std::make_shared<AnyGroup>(), cache_,
                                     std::make_shared<NoReads>(), std::make_shared<OneModule>(),
                                     std::make_shared<NoEvents>()};
};

TEST_F(JobRace, ConcurrentStartsOfOneJobAdmitOne) {
  constexpr int kThreads = 4;
  for (int round = 0; round < 200; ++round) {
    std::atomic<int> ready{0};
    std::vector<rdc_status_t> results(kThreads);
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
      threads.emplace_back([&, t] {
        ready++;
        while (ready < kThreads) {
          std::this_thread::yield();
        }
        results[t] = start(t + 1, "job");  // a group of its own
      });
    }
    for (auto& thread : threads) {
      thread.join();
    }
    int started = 0;
    for (auto result : results) {
      if (result == RDC_ST_OK) {
        started++;
      } else {
        // Not RDC_ST_CONFLICT: a start that lost left no watch on its group.
        ASSERT_EQ(result, RDC_ST_ALREADY_EXIST) << "round " << round;
      }
    }
    ASSERT_EQ(started, 1) << "round " << round;
    remove("job");
    ASSERT_EQ(get("job"), RDC_ST_NOT_FOUND) << "round " << round;
  }
}

TEST_F(JobRace, RemoveDuringAStartLeavesTheIdUsable) {
  for (bool all : {false, true}) {
    for (int round = 0; round < 10; ++round) {
      cache().held = false;
      cache().go = false;
      cache().hold_next_start = true;
      rdc_status_t started = RDC_ST_UNKNOWN_ERROR;
      std::atomic<bool> start_done{false};
      std::thread starter([&] {
        started = start(1, "job");
        start_done = true;
      });
      // The remove comes while the start holds between the cache and the
      // watch table.
      bool raced = false;
      std::thread remover([&] {
        while (!cache().held && !start_done) {
          std::this_thread::yield();
        }
        raced = !start_done;
        cache().go = true;
        if (all) {
          remove_all();
        } else {
          remove("job");
        }
      });
      starter.join();
      remover.join();
      ASSERT_TRUE(raced) << "round " << round;
      ASSERT_EQ(started, RDC_ST_OK) << "round " << round;
      // The remove waited for the start, so the job is gone and its ID starts
      // again. A remove that erased the job's cache entry under the start
      // left the job running without stats, and the next start of the ID got
      // RDC_ST_ALREADY_EXIST.
      ASSERT_EQ(get("job"), RDC_ST_NOT_FOUND) << "round " << round << (all ? " (all)" : "");
      ASSERT_EQ(start(2, "job"), RDC_ST_OK) << "round " << round << (all ? " (all)" : "");
      remove("job");
    }
  }
}

TEST_F(JobRace, AStartOfAnIdWaitsForItsStop) {
  ASSERT_EQ(start(1, "job"), RDC_ST_OK);
  cache().held = false;
  cache().go = false;
  cache().hold_next_stop = true;
  rdc_status_t stopped = RDC_ST_UNKNOWN_ERROR;
  std::atomic<bool> stop_done{false};
  std::thread stopper([&] {
    stopped = stop("job");
    stop_done = true;
  });
  while (!cache().held && !stop_done) {
    std::this_thread::yield();
  }
  const bool raced = !stop_done;
  // The start comes while the stop holds between the watch table and the
  // cache.
  rdc_status_t started = RDC_ST_UNKNOWN_ERROR;
  std::thread starter([&] {
    cache().go = true;
    started = start(2, "job");
  });
  stopper.join();
  starter.join();
  ASSERT_TRUE(raced);
  ASSERT_EQ(stopped, RDC_ST_OK);
  ASSERT_EQ(started, RDC_ST_OK);
  // The start waited for the stop, so the new job runs: its stats end now. A
  // stop that went on under the start ended the new job's stats instead.
  const uint64_t stopped_by = static_cast<uint64_t>(std::time(nullptr));
  std::this_thread::sleep_for(std::chrono::milliseconds(1500));
  EXPECT_GT(end_time("job"), stopped_by);
  remove("job");
}

TEST_F(JobRace, StartsAtTheLimitRemoveStoppedJobsWhileStatsAreRead) {
  for (int i = 0; i < kMaxJobs; ++i) {
    ASSERT_EQ(start(1, "old" + std::to_string(i)), RDC_ST_OK);
    ASSERT_EQ(stop("old" + std::to_string(i)), RDC_ST_OK);
  }
  std::atomic<bool> done{false};
  std::thread reader([&] {
    while (!done) {
      rdc_status_t result = get("job");
      EXPECT_TRUE(result == RDC_ST_OK || result == RDC_ST_NOT_FOUND) << result;
    }
  });
  // "job" starts again and again; once stopped, it is among the jobs that the
  // other thread's starts may remove.
  std::thread restarts([&] {
    for (int i = 0; i < 300; ++i) {
      EXPECT_EQ(start(2, "job"), RDC_ST_OK);
      EXPECT_EQ(stop("job"), RDC_ST_OK);
    }
  });
  std::thread new_jobs([&] {
    for (int i = 0; i < 300; ++i) {
      EXPECT_EQ(start(3, "new" + std::to_string(i)), RDC_ST_OK);
      EXPECT_EQ(stop("new" + std::to_string(i)), RDC_ST_OK);
    }
  });
  // The updater feeds the running jobs' stats, as rdcd's update thread does.
  std::thread updater([&] {
    while (!done) {
      update();
    }
  });
  restarts.join();
  new_jobs.join();
  done = true;
  reader.join();
  updater.join();

  // The limit still holds: kMaxJobs running jobs replace every stopped one,
  // and one more does not start.
  for (int i = 0; i < kMaxJobs; ++i) {
    ASSERT_EQ(start(100 + i, "running" + std::to_string(i)), RDC_ST_OK) << i;
  }
  EXPECT_EQ(start(1000, "one_more"), RDC_ST_MAX_LIMIT);
}

}  // namespace
