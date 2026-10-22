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

// Health watches: what rdc_health_set() and rdc_health_clear() leave behind, and what
// rdc_health_check() reports. Needs no GPU: the test builds RdcWatchTableImpl.cc,
// RdcGroupSettingsImpl.cc and RdcCacheManagerImpl.cc itself instead of linking librdc, reads
// GPU values from a stub, and links only librdc_bootstrap.

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "rdc/rdc.h"
#include "rdc_lib/RdcMetricFetcher.h"
#include "rdc_lib/RdcModuleMgr.h"
#include "rdc_lib/RdcNotification.h"
#include "rdc_lib/RdcPartition.h"
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

// GPU values for the health checks. A field without a value cannot be read, like a field the
// GPU does not support.
class StubFetcher : public amd::rdc::RdcMetricFetcher {
 public:
  rdc_status_t acquire_smi_handle(RdcFieldKey) override { return RDC_ST_OK; }
  rdc_status_t delete_smi_handle(RdcFieldKey) override { return RDC_ST_OK; }
  rdc_status_t fetch_smi_field(uint32_t gpu_index, rdc_field_t field_id,
                               rdc_field_value* value) override {
    if (on_fetch) on_fetch();
    auto it = values.find({gpu_index, field_id});
    if (it == values.end()) return RDC_ST_SMI_ERROR;
    *value = {};
    value->field_id = field_id;
    value->type = INTEGER;
    value->ts = now_ms() - age_ms;
    value->value.l_int = it->second;
    return RDC_ST_OK;
  }
  rdc_status_t bulk_fetch_smi_fields(rdc_gpu_field_t*, uint32_t,
                                     std::vector<rdc_gpu_field_value_t>&) override {
    return RDC_ST_NOT_SUPPORTED;
  }

  std::map<RdcFieldKey, int64_t> values;
  uint64_t age_ms = 0;             // how old a value is when it is read
  std::function<void()> on_fetch;  // runs before every read
};

// No telemetry module: the watch table keeps every field of a watch and calls no module.
class NoModules : public amd::rdc::RdcModuleMgr {
 public:
  amd::rdc::RdcTelemetryPtr get_telemetry_module() override { return nullptr; }
  amd::rdc::RdcDiagnosticPtr get_diagnostic_module() override { return nullptr; }
};

// The health fields are not event fields, so nothing listens for events.
class NoEvents : public amd::rdc::RdcNotification {
 public:
  bool is_notification_event(rdc_field_t) const override { return false; }
  rdc_status_t add_event_listeners(const std::vector<RdcFieldKey>&) override { return RDC_ST_OK; }
  rdc_status_t remove_event_listeners(const std::vector<RdcFieldKey>&) override {
    return RDC_ST_OK;
  }
  rdc_status_t listen(amd::rdc::rdc_evnt_notification_t*, uint32_t*, uint32_t) override {
    return RDC_ST_NOT_SUPPORTED;
  }
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

std::string not_run(const std::string& check) {
  return "The " + check + " health check did not complete (status " +
         std::to_string(RDC_ST_SMI_ERROR) + ").";
}

// A GPU group with GPU 0, and the watch table with a real group table and cache.
class Health : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(groups_->rdc_group_gpu_create("health", &group_), RDC_ST_OK);
    ASSERT_EQ(groups_->rdc_group_gpu_add(group_, 0), RDC_ST_OK);
  }

  // The field groups that rdc_health_set() created.
  std::vector<rdc_field_grp_t> health_field_groups() {
    rdc_field_grp_t ids[RDC_MAX_NUM_FIELD_GROUPS];
    uint32_t count = 0;
    EXPECT_EQ(groups_->rdc_group_field_get_all_ids(ids, &count), RDC_ST_OK);
    std::vector<rdc_field_grp_t> found;
    for (uint32_t i = 0; i < count; i++) {
      rdc_field_group_info_t info = {};
      if (groups_->rdc_group_field_get_info(ids[i], &info) == RDC_ST_OK &&
          std::string(info.group_name) == "health-field-group") {
        found.push_back(ids[i]);
      }
    }
    return found;
  }

  // Whether the cache holds health values of the group's GPU 0 for the field.
  bool cached(rdc_field_t field) {
    rdc_field_value value;
    return cache_->rdc_health_get_values(group_, 0, field, 0, 0, &value, nullptr) == RDC_ST_OK;
  }

  rdc_status_t check() { return watch_table_.rdc_health_check(group_, response_.get()); }

  // The incident messages of the last check, in order.
  std::vector<std::string> messages() const {
    std::vector<std::string> msgs;
    for (uint32_t i = 0; i < response_->incidents_count; i++) {
      msgs.emplace_back(response_->incidents[i].error.msg);
    }
    return msgs;
  }

  std::shared_ptr<StubFetcher> fetcher_ = std::make_shared<StubFetcher>();
  std::shared_ptr<amd::rdc::RdcGroupSettingsImpl> groups_ =
      std::make_shared<amd::rdc::RdcGroupSettingsImpl>(std::make_shared<OnePartition>());
  std::shared_ptr<amd::rdc::RdcCacheManagerImpl> cache_ =
      std::make_shared<amd::rdc::RdcCacheManagerImpl>();
  amd::rdc::RdcWatchTableImpl watch_table_{groups_, cache_, fetcher_, std::make_shared<NoModules>(),
                                           std::make_shared<NoEvents>()};
  rdc_gpu_group_t group_ = 0;
  // About 260 KB: too big for the stack of a sanitizer build.
  std::unique_ptr<rdc_health_response_t> response_ = std::make_unique<rdc_health_response_t>();
};

TEST_F(Health, ClearOfADestroyedGroupDropsTheWatch) {
  fetcher_->values = {{{0, RDC_HEALTH_PCIE_REPLAY_COUNT}, 0}};
  ASSERT_EQ(watch_table_.rdc_health_set(group_, RDC_HEALTH_WATCH_PCIE), RDC_ST_OK);
  ASSERT_EQ(health_field_groups().size(), 1u);
  ASSERT_TRUE(cached(RDC_HEALTH_PCIE_REPLAY_COUNT));
  ASSERT_EQ(groups_->rdc_group_gpu_destroy(group_), RDC_ST_OK);

  // The clear drops the field group, the cached values and the watch although the group is
  // gone. It returns RDC_ST_GROUP_NOT_FOUND when ending the field watch needs the group, and
  // RDC_ST_OK when it does not.
  const rdc_status_t cleared = watch_table_.rdc_health_clear(group_);
  EXPECT_TRUE(cleared == RDC_ST_GROUP_NOT_FOUND || cleared == RDC_ST_OK) << cleared;
  EXPECT_TRUE(health_field_groups().empty());
  EXPECT_FALSE(cached(RDC_HEALTH_PCIE_REPLAY_COUNT));
  EXPECT_EQ(check(), RDC_ST_NOT_FOUND);
}

TEST_F(Health, SetReplacesAWatchThatCannotStop) {
  fetcher_->values = {{{0, RDC_HEALTH_PCIE_REPLAY_COUNT}, 0},
                      {{0, RDC_HEALTH_THERMAL_THROTTLE_TIME}, 0}};
  ASSERT_EQ(watch_table_.rdc_health_set(group_, RDC_HEALTH_WATCH_PCIE), RDC_ST_OK);
  // Without its field group, the field watch of the PCIe watch cannot stop.
  const std::vector<rdc_field_grp_t> pcie = health_field_groups();
  ASSERT_EQ(pcie.size(), 1u);
  ASSERT_EQ(groups_->rdc_group_field_destroy(pcie[0]), RDC_ST_OK);

  ASSERT_EQ(watch_table_.rdc_health_set(group_, RDC_HEALTH_WATCH_THERMAL), RDC_ST_OK);
  unsigned int components = 0;
  ASSERT_EQ(watch_table_.rdc_health_get(group_, &components), RDC_ST_OK);
  EXPECT_EQ(components, static_cast<unsigned int>(RDC_HEALTH_WATCH_THERMAL));
  EXPECT_EQ(health_field_groups().size(), 1u);
}

TEST_F(Health, SetThatFailsLeavesNothingBehind) {
  fetcher_->values = {{{0, RDC_HEALTH_PCIE_REPLAY_COUNT}, 0}};
  // Another client destroys the group while rdc_health_set() reads the first values, so the
  // field watch fails.
  fetcher_->on_fetch = [this] { groups_->rdc_group_gpu_destroy(group_); };
  EXPECT_EQ(watch_table_.rdc_health_set(group_, RDC_HEALTH_WATCH_PCIE), RDC_ST_GROUP_NOT_FOUND);
  EXPECT_TRUE(health_field_groups().empty());
  EXPECT_FALSE(cached(RDC_HEALTH_PCIE_REPLAY_COUNT));
}

TEST_F(Health, SetReadsTheOtherFieldsWhenOneCannotBeRead) {
  // The PCIe replay count cannot be read; the thermal value can.
  fetcher_->values = {{{0, RDC_HEALTH_THERMAL_THROTTLE_TIME}, 0}};
  ASSERT_EQ(watch_table_.rdc_health_set(group_, RDC_HEALTH_WATCH_PCIE | RDC_HEALTH_WATCH_THERMAL),
            RDC_ST_OK);
  EXPECT_TRUE(cached(RDC_HEALTH_THERMAL_THROTTLE_TIME));

  ASSERT_EQ(check(), RDC_ST_OK);
  EXPECT_EQ(messages(), std::vector<std::string>{not_run("PCIe")});
}

TEST_F(Health, CheckReportsAComponentItCannotRead) {
  fetcher_->values = {{{0, RDC_HEALTH_PCIE_REPLAY_COUNT}, 0}};
  ASSERT_EQ(watch_table_.rdc_health_set(group_, RDC_HEALTH_WATCH_PCIE), RDC_ST_OK);
  fetcher_->values.clear();

  ASSERT_EQ(check(), RDC_ST_OK);
  EXPECT_EQ(response_->overall_health, RDC_HEALTH_RESULT_WARN);
  ASSERT_EQ(response_->incidents_count, 1u);
  const rdc_health_incidents_t& incident = response_->incidents[0];
  EXPECT_EQ(incident.gpu_index, 0u);
  EXPECT_EQ(incident.component, RDC_HEALTH_WATCH_PCIE);
  EXPECT_EQ(incident.health, RDC_HEALTH_RESULT_WARN);
  EXPECT_EQ(incident.error.code, static_cast<uint32_t>(RDC_FR_HEALTH_CHECK_NOT_RUN));
  EXPECT_EQ(std::string(incident.error.msg), not_run("PCIe"));
}

TEST_F(Health, CheckReportsEveryComponentItCannotRead) {
  const unsigned int all = RDC_HEALTH_WATCH_PCIE | RDC_HEALTH_WATCH_XGMI | RDC_HEALTH_WATCH_MEM |
                           RDC_HEALTH_WATCH_EEPROM | RDC_HEALTH_WATCH_THERMAL |
                           RDC_HEALTH_WATCH_POWER;
  // No field can be read, not even for the first values.
  ASSERT_EQ(watch_table_.rdc_health_set(group_, all), RDC_ST_OK);

  ASSERT_EQ(check(), RDC_ST_OK);
  EXPECT_EQ(response_->overall_health, RDC_HEALTH_RESULT_WARN);
  EXPECT_EQ(messages(),
            (std::vector<std::string>{not_run("PCIe"), not_run("XGMI"), not_run("memory"),
                                      not_run("EEPROM"), not_run("thermal"), not_run("power")}));
}

TEST_F(Health, CheckReadsTheOtherFieldsWhenOneCannotBeRead) {
  fetcher_->values = {{{0, RDC_HEALTH_PCIE_REPLAY_COUNT}, 0},
                      {{0, RDC_HEALTH_THERMAL_THROTTLE_TIME}, 0}};
  // The first values are two minutes old, so the thermal check, which compares with the
  // values of the last minute, needs the values that rdc_health_check() reads first.
  fetcher_->age_ms = 120 * 1000;
  ASSERT_EQ(watch_table_.rdc_health_set(group_, RDC_HEALTH_WATCH_PCIE | RDC_HEALTH_WATCH_THERMAL),
            RDC_ST_OK);
  fetcher_->age_ms = 0;
  fetcher_->values.erase({0, RDC_HEALTH_PCIE_REPLAY_COUNT});

  ASSERT_EQ(check(), RDC_ST_OK);
  EXPECT_EQ(messages(), std::vector<std::string>{not_run("PCIe")});
}

TEST_F(Health, CheckCountsTheIncreaseOfACounter) {
  fetcher_->values = {{{0, RDC_HEALTH_PCIE_REPLAY_COUNT}, 100},
                      {{0, RDC_HEALTH_THERMAL_THROTTLE_TIME}, 1000},
                      {{0, RDC_HEALTH_POWER_THROTTLE_TIME}, 1000}};
  ASSERT_EQ(watch_table_.rdc_health_set(
                group_, RDC_HEALTH_WATCH_PCIE | RDC_HEALTH_WATCH_THERMAL | RDC_HEALTH_WATCH_POWER),
            RDC_ST_OK);
  fetcher_->values = {{{0, RDC_HEALTH_PCIE_REPLAY_COUNT}, 120},
                      {{0, RDC_HEALTH_THERMAL_THROTTLE_TIME}, 1500},
                      {{0, RDC_HEALTH_POWER_THROTTLE_TIME}, 1700}};

  ASSERT_EQ(check(), RDC_ST_OK);
  EXPECT_EQ(messages(),
            (std::vector<std::string>{
                "Detected 20 PCIe replays per minute exceeding the max limit 8.",
                "Detected 500 clock throttling due to thermal violation in the last minute.",
                "Detected 700 clock throttling due to power violation in the last minute."}));
}

TEST_F(Health, CheckCountsFromZeroAfterACounterReset) {
  fetcher_->values = {{{0, RDC_HEALTH_PCIE_REPLAY_COUNT}, 100},
                      {{0, RDC_HEALTH_THERMAL_THROTTLE_TIME}, 1000},
                      {{0, RDC_HEALTH_POWER_THROTTLE_TIME}, 1000}};
  ASSERT_EQ(watch_table_.rdc_health_set(
                group_, RDC_HEALTH_WATCH_PCIE | RDC_HEALTH_WATCH_THERMAL | RDC_HEALTH_WATCH_POWER),
            RDC_ST_OK);
  // A GPU reset restarts the counters from zero.
  fetcher_->values = {{{0, RDC_HEALTH_PCIE_REPLAY_COUNT}, 5},
                      {{0, RDC_HEALTH_THERMAL_THROTTLE_TIME}, 10},
                      {{0, RDC_HEALTH_POWER_THROTTLE_TIME}, 20}};

  ASSERT_EQ(check(), RDC_ST_OK);
  // 5 PCIe replays are within the limit of 8 a minute.
  EXPECT_EQ(messages(),
            (std::vector<std::string>{
                "Detected 10 clock throttling due to thermal violation in the last minute.",
                "Detected 20 clock throttling due to power violation in the last minute."}));
}

TEST_F(Health, CheckCountsRetiredPagesFromZeroAfterAReset) {
  fetcher_->values = {{{0, RDC_FI_ECC_UNCORRECT_TOTAL}, 0},
                      {{0, RDC_HEALTH_RETIRED_PAGE_NUM}, 5},
                      {{0, RDC_HEALTH_PENDING_PAGE_NUM}, 0},
                      {{0, RDC_HEALTH_RETIRED_PAGE_LIMIT}, 100}};
  ASSERT_EQ(watch_table_.rdc_health_set(group_, RDC_HEALTH_WATCH_MEM), RDC_ST_OK);
  // The retired-page count restarts after the RAS table of the GPU is reset.
  fetcher_->values[{0, RDC_HEALTH_RETIRED_PAGE_NUM}] = 3;

  ASSERT_EQ(check(), RDC_ST_OK);
  EXPECT_EQ(response_->overall_health, RDC_HEALTH_RESULT_FAIL);
  EXPECT_EQ(messages(),
            std::vector<std::string>{"Detected 3 retired pages more than one in the last week."});
}

}  // namespace
