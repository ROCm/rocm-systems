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

// Watches of destroyed groups, samples for every job and health watch that holds them, the job
// stats that process events and power samples make, and what the cache keeps of events and
// returns from rdc_field_get_value_since(). Needs no GPU: the test builds
// RdcWatchTableImpl.cc, RdcGroupSettingsImpl.cc and RdcCacheManagerImpl.cc itself instead of
// linking librdc, with a telemetry module that records what it is asked, and links only
// librdc_bootstrap.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#include "rdc/rdc.h"
#include "rdc_lib/RdcGroupSettings.h"
#include "rdc_lib/RdcMetricFetcher.h"
#include "rdc_lib/RdcModuleMgr.h"
#include "rdc_lib/RdcNotification.h"
#include "rdc_lib/RdcPartition.h"
#include "rdc_lib/RdcTelemetry.h"
#include "rdc_lib/impl/RdcCacheManagerImpl.h"
#include "rdc_lib/impl/RdcGroupSettingsImpl.h"
#include "rdc_lib/impl/RdcWatchTableImpl.h"

// Defined in RdcNotificationImpl.cc, which this test does not build. Only
// rdc_field_listen_notif() reads it.
const uint32_t amd::rdc::kMaxRSMIEvents = 64;

namespace {

uint64_t now_ms() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

// Supports the given fields, reads each of them as `value`, and records the fields it reads and
// the fields it is told to stop watching.
class RecordingTelemetry : public amd::rdc::RdcTelemetry {
 public:
  rdc_status_t rdc_telemetry_fields_query(uint32_t field_ids[MAX_NUM_FIELDS],
                                          uint32_t* field_count) override {
    std::copy(supported.begin(), supported.end(), field_ids);
    *field_count = supported.size();
    return RDC_ST_OK;
  }
  rdc_status_t rdc_telemetry_fields_value_get(rdc_gpu_field_t* fields, uint32_t fields_count,
                                              rdc_field_value_f callback,
                                              void* user_data) override {
    read.clear();
    std::vector<rdc_gpu_field_value_t> values(fields_count);
    for (uint32_t i = 0; i < fields_count; i++) {
      read.emplace_back(fields[i].gpu_index, fields[i].field_id);
      values[i] = {};
      values[i].gpu_index = fields[i].gpu_index;
      values[i].field_value.field_id = fields[i].field_id;
      values[i].field_value.status = RDC_ST_OK;
      values[i].field_value.type = INTEGER;
      values[i].field_value.ts = now_ms();
      values[i].field_value.value.l_int = value;
    }
    if (fields_count > 0) callback(values.data(), fields_count, user_data);
    return RDC_ST_OK;
  }
  rdc_status_t rdc_telemetry_fields_watch(rdc_gpu_field_t*, uint32_t) override { return RDC_ST_OK; }
  rdc_status_t rdc_telemetry_fields_unwatch(rdc_gpu_field_t* fields,
                                            uint32_t fields_count) override {
    for (uint32_t i = 0; i < fields_count; i++) {
      unwatched.emplace_back(fields[i].gpu_index, fields[i].field_id);
    }
    return RDC_ST_OK;
  }

  std::vector<uint32_t> supported;
  std::vector<RdcFieldKey> read;       // the fields of the last read, in order
  std::vector<RdcFieldKey> unwatched;  // every field it was told to stop watching
  int64_t value = 7000000;             // 7 W as a power usage
};

class TelemetryOnly : public amd::rdc::RdcModuleMgr {
 public:
  amd::rdc::RdcTelemetryPtr get_telemetry_module() override { return telemetry; }
  amd::rdc::RdcDiagnosticPtr get_diagnostic_module() override { return nullptr; }

  std::shared_ptr<RecordingTelemetry> telemetry = std::make_shared<RecordingTelemetry>();
};

// The first values of a health watch: every field reads 0.
class ZeroFetcher : public amd::rdc::RdcMetricFetcher {
 public:
  rdc_status_t acquire_smi_handle(RdcFieldKey) override { return RDC_ST_OK; }
  rdc_status_t delete_smi_handle(RdcFieldKey) override { return RDC_ST_OK; }
  rdc_status_t fetch_smi_field(uint32_t, rdc_field_t field_id, rdc_field_value* value) override {
    *value = {};
    value->field_id = field_id;
    value->type = INTEGER;
    value->ts = now_ms();
    return RDC_ST_OK;
  }
  rdc_status_t bulk_fetch_smi_fields(rdc_gpu_field_t*, uint32_t,
                                     std::vector<rdc_gpu_field_value_t>&) override {
    return RDC_ST_NOT_SUPPORTED;
  }
};

// The event fields are the VM fault and the process start and end. Records the listeners that
// go, and listen() returns the queued events.
class RecordingEvents : public amd::rdc::RdcNotification {
 public:
  bool is_notification_event(rdc_field_t field) const override {
    return field == RDC_EVNT_NOTIF_VMFAULT || field == RDC_EVNT_NOTIF_PROCESS_START ||
           field == RDC_EVNT_NOTIF_PROCESS_END;
  }
  rdc_status_t add_event_listeners(const std::vector<RdcFieldKey>&) override { return RDC_ST_OK; }
  rdc_status_t remove_event_listeners(const std::vector<RdcFieldKey>& fields) override {
    removed.insert(removed.end(), fields.begin(), fields.end());
    return RDC_ST_OK;
  }
  rdc_status_t listen(amd::rdc::rdc_evnt_notification_t* events, uint32_t* num_events,
                      uint32_t) override {
    const uint32_t count = std::min<uint32_t>(*num_events, queued.size());
    std::copy(queued.begin(), queued.begin() + count, events);
    queued.erase(queued.begin(), queued.begin() + count);
    *num_events = count;
    return RDC_ST_OK;
  }

  std::vector<RdcFieldKey> removed;
  std::vector<amd::rdc::rdc_evnt_notification_t> queued;
};

// Every GPU index is a GPU with one partition.
class OnePartition : public amd::rdc::RdcPartition {
 public:
  rdc_status_t rdc_instance_profile_get_impl(uint32_t, rdc_instance_resource_type_t,
                                             rdc_resource_profile_t*) override {
    return RDC_ST_NOT_SUPPORTED;
  }
  rdc_status_t rdc_get_num_partition_impl(uint32_t, uint16_t* num_partition) override {
    *num_partition = 1;
    return RDC_ST_OK;
  }
};

// The watch table with a real group table and cache, and a telemetry module that supports the
// job fields and the PCIe replay count.
class WatchTable : public ::testing::Test {
 protected:
  void SetUp() override {
    rdc_field_group_info_t job_fields = {};
    ASSERT_EQ(groups_->rdc_group_field_get_info(amd::rdc::JOB_FIELD_ID, &job_fields), RDC_ST_OK);
    auto& supported = modules_->telemetry->supported;
    supported.assign(job_fields.field_ids, job_fields.field_ids + job_fields.count);
    supported.push_back(RDC_HEALTH_PCIE_REPLAY_COUNT);
    for (uint32_t gpu : {0u, 1u}) {
      gauges_[{gpu, RDC_FI_GPU_MEMORY_TOTAL}] = 1ull << 34;
    }
  }

  rdc_gpu_group_t group(const char* name, std::initializer_list<uint32_t> gpus) {
    rdc_gpu_group_t id = 0;
    EXPECT_EQ(groups_->rdc_group_gpu_create(name, &id), RDC_ST_OK);
    for (uint32_t gpu : gpus) {
      EXPECT_EQ(groups_->rdc_group_gpu_add(id, gpu), RDC_ST_OK);
    }
    return id;
  }

  rdc_field_grp_t field_group(rdc_field_t field) {
    rdc_field_grp_t id = 0;
    EXPECT_EQ(groups_->rdc_group_field_create(1, &field, "watched", &id), RDC_ST_OK);
    return id;
  }

  // The fields that one rdc_field_update_all() reads, sorted. The first update also cleans up.
  std::vector<RdcFieldKey> update_all() {
    modules_->telemetry->read.clear();
    EXPECT_EQ(watch_table_.rdc_field_update_all(), RDC_ST_OK);
    std::vector<RdcFieldKey> read = modules_->telemetry->read;
    std::sort(read.begin(), read.end());
    return read;
  }

  // Queues an event for the next rdc_field_listen_notif().
  void deliver(uint32_t gpu, rdc_field_t field, const std::string& text, uint64_t ts) {
    amd::rdc::rdc_evnt_notification_t event = {};
    event.gpu_id = gpu;
    event.field.field_id = field;
    event.field.status = RDC_ST_OK;
    event.field.type = STRING;
    event.field.ts = ts;
    strncpy(event.field.value.str, text.c_str(), sizeof(event.field.value.str) - 1);
    events_->queued.push_back(event);
  }

  // The texts of the event samples that the cache holds for the field, oldest first.
  std::vector<std::string> cached_events(uint32_t gpu, rdc_field_t field) {
    std::vector<std::string> texts;
    uint64_t since = 0;
    for (int i = 0; i < 100; i++) {
      rdc_field_value value = {};
      uint64_t next = 0;
      if (cache_->rdc_field_get_value_since(gpu, field, since, &next, &value) != RDC_ST_OK) break;
      texts.emplace_back(value.value.str);
      since = next;
    }
    return texts;
  }

  std::shared_ptr<amd::rdc::RdcGroupSettingsImpl> groups_ =
      std::make_shared<amd::rdc::RdcGroupSettingsImpl>(std::make_shared<OnePartition>());
  std::shared_ptr<amd::rdc::RdcCacheManagerImpl> cache_ =
      std::make_shared<amd::rdc::RdcCacheManagerImpl>();
  std::shared_ptr<TelemetryOnly> modules_ = std::make_shared<TelemetryOnly>();
  std::shared_ptr<RecordingEvents> events_ = std::make_shared<RecordingEvents>();
  amd::rdc::RdcWatchTableImpl watch_table_{groups_, cache_, std::make_shared<ZeroFetcher>(),
                                           modules_, events_};
  rdc_gpu_gauges_t gauges_;
};

TEST_F(WatchTable, UnwatchWorksWhileAnotherWatchedGroupIsGone) {
  const rdc_gpu_group_t kept = group("kept", {0});
  const rdc_gpu_group_t gone = group("gone", {1});
  const rdc_field_grp_t util = field_group(RDC_FI_GPU_UTIL);
  ASSERT_EQ(watch_table_.rdc_field_watch(kept, util, 1, 1, 1), RDC_ST_OK);
  ASSERT_EQ(watch_table_.rdc_field_watch(gone, util, 1, 1, 1), RDC_ST_OK);
  ASSERT_EQ(groups_->rdc_group_gpu_destroy(gone), RDC_ST_OK);

  EXPECT_EQ(watch_table_.rdc_field_unwatch(kept, util), RDC_ST_OK);
  EXPECT_EQ(watch_table_.rdc_field_unwatch(gone, util), RDC_ST_OK);
}

TEST_F(WatchTable, UnwatchWorksWhileAnotherWatchedFieldGroupIsGone) {
  const rdc_gpu_group_t gpus = group("gpus", {0});
  const rdc_field_grp_t util = field_group(RDC_FI_GPU_UTIL);
  const rdc_field_grp_t power = field_group(RDC_FI_POWER_USAGE);
  ASSERT_EQ(watch_table_.rdc_field_watch(gpus, util, 1, 1, 1), RDC_ST_OK);
  ASSERT_EQ(watch_table_.rdc_field_watch(gpus, power, 1, 1, 1), RDC_ST_OK);
  ASSERT_EQ(groups_->rdc_group_field_destroy(power), RDC_ST_OK);

  EXPECT_EQ(watch_table_.rdc_field_unwatch(gpus, util), RDC_ST_OK);
}

TEST_F(WatchTable, FieldsOfADestroyedGroupAreNoLongerRead) {
  const rdc_gpu_group_t kept = group("kept", {0});
  const rdc_gpu_group_t gone = group("gone", {1});
  const rdc_field_grp_t util = field_group(RDC_FI_GPU_UTIL);
  ASSERT_EQ(watch_table_.rdc_field_watch(kept, util, 1, 1, 1), RDC_ST_OK);
  ASSERT_EQ(watch_table_.rdc_field_watch(gone, util, 1, 1, 1), RDC_ST_OK);
  ASSERT_EQ(groups_->rdc_group_gpu_destroy(gone), RDC_ST_OK);

  // The first update reads both GPUs, then cleans up, which ends the watch of the destroyed group.
  update_all();
  EXPECT_EQ(update_all(), std::vector<RdcFieldKey>{RdcFieldKey(0, RDC_FI_GPU_UTIL)});
  EXPECT_EQ(modules_->telemetry->unwatched,
            std::vector<RdcFieldKey>{RdcFieldKey(1, RDC_FI_GPU_UTIL)});
}

TEST_F(WatchTable, EventListenersOfADestroyedGroupGo) {
  const rdc_gpu_group_t gone = group("gone", {1});
  const rdc_field_grp_t faults = field_group(RDC_EVNT_NOTIF_VMFAULT);
  ASSERT_EQ(watch_table_.rdc_field_watch(gone, faults, 1, 1, 1), RDC_ST_OK);
  ASSERT_EQ(groups_->rdc_group_gpu_destroy(gone), RDC_ST_OK);

  update_all();
  EXPECT_EQ(events_->removed, std::vector<RdcFieldKey>{RdcFieldKey(1, RDC_EVNT_NOTIF_VMFAULT)});
}

TEST_F(WatchTable, UnwatchLeavesTheFrequencyOfTheRemainingWatch) {
  const rdc_gpu_group_t hourly = group("hourly", {0});
  const rdc_gpu_group_t always = group("always", {0});
  const rdc_field_grp_t util = field_group(RDC_FI_GPU_UTIL);
  ASSERT_EQ(watch_table_.rdc_field_watch(hourly, util, 3600000000ull, 1, 1), RDC_ST_OK);
  ASSERT_EQ(watch_table_.rdc_field_watch(always, util, 1, 1, 1), RDC_ST_OK);
  EXPECT_EQ(update_all(), std::vector<RdcFieldKey>{RdcFieldKey(0, RDC_FI_GPU_UTIL)});

  ASSERT_EQ(watch_table_.rdc_field_unwatch(always, util), RDC_ST_OK);
  // Read a moment ago, the field is not due again for an hour.
  EXPECT_TRUE(update_all().empty());
}

TEST_F(WatchTable, JobOfADestroyedGroupStops) {
  const rdc_gpu_group_t gone = group("gone", {0});
  char job[64] = "job";
  ASSERT_EQ(watch_table_.rdc_job_start_stats(gone, job, 1, gauges_), RDC_ST_OK);
  ASSERT_EQ(groups_->rdc_group_gpu_destroy(gone), RDC_ST_OK);

  EXPECT_EQ(watch_table_.rdc_job_stop_stats(job, gauges_), RDC_ST_OK);
}

TEST_F(WatchTable, EveryJobOnAGpuGetsItsSamples) {
  const rdc_gpu_group_t a = group("a", {0});
  const rdc_gpu_group_t b = group("b", {0, 1});
  char job_a[64] = "job-a";
  char job_b[64] = "job-b";
  ASSERT_EQ(watch_table_.rdc_job_start_stats(a, job_a, 1, gauges_), RDC_ST_OK);
  ASSERT_EQ(watch_table_.rdc_job_start_stats(b, job_b, 1, gauges_), RDC_ST_OK);

  update_all();
  for (const char* job : {job_a, job_b}) {
    auto info = std::make_unique<rdc_job_info_t>();
    ASSERT_EQ(cache_->rdc_job_get_stats(job, gauges_, info.get()), RDC_ST_OK);
    // The one power sample of GPU 0: 7 W.
    EXPECT_EQ(info->gpus[0].power_usage.max_value, 7u) << job;
    EXPECT_EQ(info->gpus[0].power_usage.min_value, 7u) << job;
  }
}

TEST_F(WatchTable, EveryJobOnAGpuGetsItsProcessEvents) {
  const rdc_gpu_group_t a = group("a", {0});
  const rdc_gpu_group_t b = group("b", {0, 1});
  char job_a[64] = "job-a";
  char job_b[64] = "job-b";
  ASSERT_EQ(watch_table_.rdc_job_start_stats(a, job_a, 1, gauges_), RDC_ST_OK);
  ASSERT_EQ(watch_table_.rdc_job_start_stats(b, job_b, 1, gauges_), RDC_ST_OK);

  amd::rdc::rdc_evnt_notification_t event = {};
  event.gpu_id = 0;
  event.field.field_id = RDC_EVNT_NOTIF_PROCESS_START;
  event.field.status = RDC_ST_OK;
  event.field.type = STRING;
  event.field.ts = now_ms();
  strncpy(event.field.value.str, "PID: 42  task: p", sizeof(event.field.value.str) - 1);
  events_->queued.push_back(event);
  ASSERT_EQ(watch_table_.rdc_field_listen_notif(0), RDC_ST_OK);

  for (const char* job : {job_a, job_b}) {
    auto info = std::make_unique<rdc_job_info_t>();
    ASSERT_EQ(cache_->rdc_job_get_stats(job, gauges_, info.get()), RDC_ST_OK);
    ASSERT_EQ(info->num_processes, 1u) << job;
    EXPECT_EQ(info->processes[0].pid, 42u) << job;
  }
}

TEST_F(WatchTable, EveryHealthWatchOnAGpuGetsItsSamples) {
  const rdc_gpu_group_t a = group("a", {0});
  const rdc_gpu_group_t b = group("b", {0});
  ASSERT_EQ(watch_table_.rdc_health_set(a, RDC_HEALTH_WATCH_PCIE), RDC_ST_OK);
  ASSERT_EQ(watch_table_.rdc_health_set(b, RDC_HEALTH_WATCH_PCIE), RDC_ST_OK);

  // The first values were 0; the telemetry module reads 7000000 replays.
  update_all();
  for (rdc_gpu_group_t g : {a, b}) {
    rdc_field_value latest = {};
    ASSERT_EQ(cache_->rdc_health_get_values(g, 0, RDC_HEALTH_PCIE_REPLAY_COUNT, 0, UINT64_MAX,
                                            nullptr, &latest),
              RDC_ST_OK);
    EXPECT_EQ(latest.value.l_int, 7000000) << "group " << g;
  }
}

TEST_F(WatchTable, EventSamplesFollowTheLimitsOfTheirWatch) {
  const rdc_gpu_group_t gpus = group("gpus", {0});
  const rdc_field_grp_t faults = field_group(RDC_EVNT_NOTIF_VMFAULT);
  // Samples are kept for 60 s, and 2 at most.
  ASSERT_EQ(watch_table_.rdc_field_watch(gpus, faults, 1000000, 60, 2), RDC_ST_OK);
  const uint64_t now = now_ms();
  deliver(0, RDC_EVNT_NOTIF_VMFAULT, "old", now - 120000);
  for (int i = 1; i <= 4; i++) {
    deliver(0, RDC_EVNT_NOTIF_VMFAULT, "fault " + std::to_string(i), now + i);
  }
  ASSERT_EQ(watch_table_.rdc_field_listen_notif(0), RDC_ST_OK);

  update_all();
  EXPECT_EQ(cached_events(0, RDC_EVNT_NOTIF_VMFAULT),
            (std::vector<std::string>{"fault 3", "fault 4"}));
}

TEST_F(WatchTable, EventsThatNoWatchHoldsAreNotCached) {
  const rdc_gpu_group_t gpus = group("gpus", {0});
  char job[64] = "job";
  ASSERT_EQ(watch_table_.rdc_job_start_stats(gpus, job, 1, gauges_), RDC_ST_OK);
  deliver(0, RDC_EVNT_NOTIF_PROCESS_START, "PID: 42  task: p", now_ms());
  ASSERT_EQ(watch_table_.rdc_field_listen_notif(0), RDC_ST_OK);

  // The job lists the process; no watch holds the event field, so the cache keeps nothing.
  auto info = std::make_unique<rdc_job_info_t>();
  ASSERT_EQ(cache_->rdc_job_get_stats(job, gauges_, info.get()), RDC_ST_OK);
  EXPECT_EQ(info->num_processes, 1u);
  EXPECT_TRUE(cached_events(0, RDC_EVNT_NOTIF_PROCESS_START).empty());
}

TEST_F(WatchTable, EventsAfterTheUnwatchAreNotCached) {
  const rdc_gpu_group_t gpus = group("gpus", {0});
  const rdc_field_grp_t faults = field_group(RDC_EVNT_NOTIF_VMFAULT);
  ASSERT_EQ(watch_table_.rdc_field_watch(gpus, faults, 1000000, 60, 10), RDC_ST_OK);
  const uint64_t now = now_ms();
  deliver(0, RDC_EVNT_NOTIF_VMFAULT, "watched", now);
  ASSERT_EQ(watch_table_.rdc_field_listen_notif(0), RDC_ST_OK);
  ASSERT_EQ(watch_table_.rdc_field_unwatch(gpus, faults), RDC_ST_OK);
  deliver(0, RDC_EVNT_NOTIF_VMFAULT, "unwatched", now + 1);
  ASSERT_EQ(watch_table_.rdc_field_listen_notif(0), RDC_ST_OK);

  // The samples of the watch stay until they age out; none is added after the unwatch.
  EXPECT_EQ(cached_events(0, RDC_EVNT_NOTIF_VMFAULT), (std::vector<std::string>{"watched"}));
}

TEST_F(WatchTable, EventFieldsAreNotRead) {
  const rdc_gpu_group_t gpus = group("gpus", {0});
  ASSERT_EQ(watch_table_.rdc_field_watch(gpus, field_group(RDC_EVNT_NOTIF_VMFAULT), 1, 60, 2),
            RDC_ST_OK);
  ASSERT_EQ(watch_table_.rdc_field_watch(gpus, field_group(RDC_FI_GPU_UTIL), 1, 60, 2), RDC_ST_OK);
  EXPECT_EQ(update_all(), std::vector<RdcFieldKey>{RdcFieldKey(0, RDC_FI_GPU_UTIL)});
}

// A job on GPU 0 whose field group holds only the power usage, in the cache alone.
class JobStats : public ::testing::Test {
 protected:
  void SetUp() override {
    rdc_group_info_t gpus = {};
    gpus.count = 1;
    gpus.entity_ids[0] = 0;
    rdc_field_group_info_t fields = {};
    fields.count = 1;
    fields.field_ids[0] = RDC_FI_POWER_USAGE;
    gauges_[{0, RDC_FI_GPU_MEMORY_TOTAL}] = 1ull << 34;
    ASSERT_EQ(cache_.rdc_job_start_stats(job_, gpus, fields, gauges_), RDC_ST_OK);
  }

  void process_event(rdc_field_t field, const char* text) {
    rdc_field_value value = {};
    value.field_id = field;
    value.type = STRING;
    value.ts = now_ms();
    strncpy(value.value.str, text, sizeof(value.value.str) - 1);
    cache_.rdc_update_job_stats(0, job_, value);
  }

  void power(uint64_t ts_ms, int64_t microwatts) {
    rdc_field_value value = {};
    value.field_id = RDC_FI_POWER_USAGE;
    value.type = INTEGER;
    value.ts = ts_ms;
    value.value.l_int = microwatts;
    cache_.rdc_update_job_stats(0, job_, value);
  }

  std::unique_ptr<rdc_job_info_t> stats() {
    auto info = std::make_unique<rdc_job_info_t>();
    EXPECT_EQ(cache_.rdc_job_get_stats(job_, gauges_, info.get()), RDC_ST_OK);
    return info;
  }

  amd::rdc::RdcCacheManagerImpl cache_;
  char job_[64] = "job";
  rdc_gpu_gauges_t gauges_;
};

TEST_F(JobStats, ProcessEventWithoutAPidIsIgnored) {
  process_event(RDC_EVNT_NOTIF_PROCESS_START, "unexpected text");
  EXPECT_EQ(stats()->num_processes, 0u);
}

TEST_F(JobStats, ProcessWithoutATaskNameIsKept) {
  process_event(RDC_EVNT_NOTIF_PROCESS_START, "PID: 77  task: ");
  auto info = stats();
  ASSERT_EQ(info->num_processes, 1u);
  EXPECT_EQ(info->processes[0].pid, 77u);
}

TEST_F(JobStats, AJobListsItsFirstProcessesAndLogsTheRestOnce) {
  testing::internal::CaptureStdout();
  for (uint32_t pid = 1; pid <= RDC_MAX_NUM_PROCESSES_STATUS + 2; pid++) {
    process_event(RDC_EVNT_NOTIF_PROCESS_START,
                  ("PID: " + std::to_string(pid) + "  task: p").c_str());
  }
  const std::string log = testing::internal::GetCapturedStdout();

  auto info = stats();
  ASSERT_EQ(info->num_processes, static_cast<uint32_t>(RDC_MAX_NUM_PROCESSES_STATUS));
  EXPECT_EQ(info->processes[RDC_MAX_NUM_PROCESSES_STATUS - 1].pid,
            static_cast<uint32_t>(RDC_MAX_NUM_PROCESSES_STATUS));
  const std::string dropped = "more than " + std::to_string(RDC_MAX_NUM_PROCESSES_STATUS);
  const size_t first = log.find(dropped);
  ASSERT_NE(first, std::string::npos) << log;
  EXPECT_EQ(log.find(dropped, first + 1), std::string::npos) << log;
}

TEST_F(JobStats, EnergyIgnoresASampleThatGoesBack) {
  power(1000, 100000000);  // 100 W from 1 s
  power(2000, 100000000);  // 100 J
  power(1500, 100000000);  // older than the last one
  power(3000, 100000000);  // 100 J more
  EXPECT_EQ(stats()->gpus[0].energy_consumed, 200u);
}

TEST_F(JobStats, EnergyOfAHugePowerReadingDoesNotOverflow) {
  power(1000, 100000000000000000);  // 1e11 W
  power(2000, 100000000000000000);  // 1e11 J in a second; 1e20 as an integer product
  EXPECT_EQ(stats()->gpus[0].energy_consumed, 100000000000u);
}

TEST_F(JobStats, EnergyIgnoresANegativePowerReading) {
  power(1000, 5000000);
  power(2000, -1);
  power(3000, 5000000);  // 10 J: 5 W for the 2 s since the last valid sample
  EXPECT_EQ(stats()->gpus[0].energy_consumed, 10u);
}

TEST_F(JobStats, EnergyKeepsFractionsOfAJoule) {
  power(1000, 5000000);  // 5 W
  for (uint64_t ts = 1100; ts <= 2000; ts += 100) {
    power(ts, 5000000);  // 0.5 J each
  }
  EXPECT_EQ(stats()->gpus[0].energy_consumed, 5u);
}

rdc_field_value util_sample(uint64_t ts, int64_t value) {
  rdc_field_value sample = {};
  sample.field_id = RDC_FI_GPU_UTIL;
  sample.status = RDC_ST_OK;
  sample.type = INTEGER;
  sample.ts = ts;
  sample.value.l_int = value;
  return sample;
}

// The values that a client loop over rdc_field_get_value_since() gets, from since 0 on; every
// next_since must be later than the sample it came with.
std::vector<int64_t> values_since(amd::rdc::RdcCacheManagerImpl& cache) {
  std::vector<int64_t> values;
  uint64_t since = 0;
  for (int i = 0; i < 10; i++) {
    rdc_field_value value = {};
    uint64_t next = 0;
    if (cache.rdc_field_get_value_since(0, RDC_FI_GPU_UTIL, since, &next, &value) != RDC_ST_OK) {
      break;
    }
    values.push_back(value.value.l_int);
    EXPECT_GT(next, value.ts);
    since = next;
  }
  return values;
}

TEST(Cache, ValueSinceReturnsEachSampleOnce) {
  amd::rdc::RdcCacheManagerImpl cache;
  for (int64_t i = 0; i < 3; i++) {
    cache.rdc_update_cache(0, util_sample(1000 + i, i));
  }
  EXPECT_EQ(values_since(cache), (std::vector<int64_t>{0, 1, 2}));
}

TEST(Cache, ValueSinceSkipsASampleThatSharesItsMillisecond) {
  amd::rdc::RdcCacheManagerImpl cache;
  cache.rdc_update_cache(0, util_sample(1000, 1));
  cache.rdc_update_cache(0, util_sample(1000, 2));
  cache.rdc_update_cache(0, util_sample(1001, 3));
  EXPECT_EQ(values_since(cache), (std::vector<int64_t>{1, 3}));
}

TEST(Cache, WarnsOnceAboveItsSampleThreshold) {
  amd::rdc::RdcCacheManagerImpl cache(10);
  testing::internal::CaptureStdout();
  for (uint64_t ts = 1; ts <= 30; ts++) {
    cache.rdc_update_cache(0, util_sample(ts, 1));
  }
  const std::string log = testing::internal::GetCapturedStdout();
  const size_t first = log.find("more than 10 samples");
  ASSERT_NE(first, std::string::npos) << log;
  EXPECT_EQ(log.find("more than 10 samples", first + 1), std::string::npos) << log;
}

TEST(Cache, CountsOnlyTheSamplesItHolds) {
  amd::rdc::RdcCacheManagerImpl cache(10);
  const uint64_t now = now_ms();
  testing::internal::CaptureStdout();
  // 10 samples are not more than 10. Eviction leaves 2 of them, and 8 more make 10 again.
  for (uint64_t i = 0; i < 10; i++) {
    cache.rdc_update_cache(0, util_sample(now + i, 1));
  }
  EXPECT_EQ(cache.evict_cache(0, RDC_FI_GPU_UTIL, 2, 3600), RDC_ST_OK);
  for (uint64_t i = 10; i < 18; i++) {
    cache.rdc_update_cache(0, util_sample(now + i, 1));
  }
  const std::string log = testing::internal::GetCapturedStdout();
  EXPECT_EQ(log.find("more than 10 samples"), std::string::npos) << log;
}

}  // namespace
