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

// rdc_health_set/rdc_health_check driven by a scripted metric fetcher, so the
// probe, the XGMI_WAFL fallback and the memory sub-checks run without MI300.

#include <gtest/gtest.h>

#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcMetricFetcher.h"
#include "rdc_lib/RdcModuleMgr.h"
#include "rdc_lib/RdcNotification.h"
#include "rdc_lib/RdcPartition.h"
#include "rdc_lib/impl/RdcCacheManagerImpl.h"
#include "rdc_lib/impl/RdcGroupSettingsImpl.h"
#include "rdc_lib/impl/RdcWatchTableImpl.h"

namespace {

using amd::rdc::rdc_evnt_notification_t;

class FakeFetcher : public amd::rdc::RdcMetricFetcher {
 public:
  struct Reading {
    rdc_status_t status;
    int64_t value;
  };

  void set(uint32_t gpu, rdc_field_t field, rdc_status_t status, int64_t value = 0) {
    readings_[{gpu, field}] = {status, value};
  }
  int fetches(uint32_t gpu, rdc_field_t field) const {
    auto it = fetches_.find({gpu, field});
    return it == fetches_.end() ? 0 : it->second;
  }

  rdc_status_t acquire_smi_handle(RdcFieldKey) override { return RDC_ST_OK; }
  rdc_status_t delete_smi_handle(RdcFieldKey) override { return RDC_ST_OK; }
  rdc_status_t bulk_fetch_smi_fields(rdc_gpu_field_t*, uint32_t,
                                     std::vector<rdc_gpu_field_value_t>&) override {
    return RDC_ST_NOT_SUPPORTED;
  }
  rdc_status_t fetch_smi_field(uint32_t gpu, rdc_field_t field, rdc_field_value* value) override {
    fetches_[{gpu, field}]++;
    auto it = readings_.find({gpu, field});
    Reading r = (it == readings_.end()) ? Reading{RDC_ST_NOT_SUPPORTED, 0} : it->second;
    *value = {};
    value->field_id = field;
    value->ts = 1;
    value->type = INTEGER;
    value->status = r.status;
    value->value.l_int = r.value;
    return r.status == RDC_ST_OK ? RDC_ST_OK : RDC_ST_SMI_ERROR;
  }

 private:
  std::map<RdcFieldKey, Reading> readings_;
  std::map<RdcFieldKey, int> fetches_;
};

class SinglePartition : public amd::rdc::RdcPartition {
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

class NoModules : public amd::rdc::RdcModuleMgr {
 public:
  amd::rdc::RdcTelemetryPtr get_telemetry_module() override { return nullptr; }
  amd::rdc::RdcDiagnosticPtr get_diagnostic_module() override { return nullptr; }
};

class NoNotifications : public amd::rdc::RdcNotification {
 public:
  bool is_notification_event(rdc_field_t) const override { return false; }
  rdc_status_t set_listen_events(const std::vector<RdcFieldKey>) override { return RDC_ST_OK; }
  rdc_status_t listen(rdc_evnt_notification_t*, uint32_t*, uint32_t) override { return RDC_ST_OK; }
  rdc_status_t stop_listening(uint32_t) override { return RDC_ST_OK; }
};

class CaptureLog {
 public:
  CaptureLog() : saved_(std::cout.rdbuf(out_.rdbuf())) {}
  ~CaptureLog() { std::cout.rdbuf(saved_); }

  bool has_error(const std::string& text) const {
    std::istringstream lines(out_.str());
    for (std::string line; std::getline(lines, line);) {
      if (line.find(" ERROR ") != std::string::npos && line.find(text) != std::string::npos) {
        return true;
      }
    }
    return false;
  }

 private:
  std::ostringstream out_;
  std::streambuf* saved_;
};

class HealthWatchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    fetcher_ = std::make_shared<FakeFetcher>();
    groups_ = std::make_shared<amd::rdc::RdcGroupSettingsImpl>(std::make_shared<SinglePartition>());
    table_ = std::make_unique<amd::rdc::RdcWatchTableImpl>(
        groups_, std::make_shared<amd::rdc::RdcCacheManagerImpl>(), fetcher_,
        std::make_shared<NoModules>(), std::make_shared<NoNotifications>());
    ASSERT_EQ(RDC_ST_OK, groups_->rdc_group_gpu_create("health", &group_));
    ASSERT_EQ(RDC_ST_OK, groups_->rdc_group_gpu_add(group_, kGpu));
  }

  rdc_health_response_t check() {
    rdc_health_response_t response = {};
    EXPECT_EQ(RDC_ST_OK, table_->rdc_health_check(group_, &response));
    return response;
  }

  static constexpr uint32_t kGpu = 0;
  std::shared_ptr<FakeFetcher> fetcher_;
  std::shared_ptr<amd::rdc::RdcGroupSettingsImpl> groups_;
  std::unique_ptr<amd::rdc::RdcWatchTableImpl> table_;
  rdc_gpu_group_t group_ = 0;
};

TEST_F(HealthWatchTest, XgmiFallsBackToWaflWhereLegacyStatusIsUnreadable) {
  fetcher_->set(kGpu, RDC_HEALTH_XGMI_ERROR, RDC_ST_BAD_PARAMETER);
  fetcher_->set(kGpu, RDC_FI_ECC_XGMI_WAFL_UE, RDC_ST_OK, 0);
  {
    CaptureLog log;
    ASSERT_EQ(RDC_ST_OK, table_->rdc_health_set(group_, RDC_HEALTH_WATCH_XGMI));
    EXPECT_FALSE(log.has_error("is not supported on GPU"));
  }

  EXPECT_EQ(0u, check().incidents_count);
  EXPECT_EQ(1, fetcher_->fetches(kGpu, RDC_HEALTH_XGMI_ERROR));

  fetcher_->set(kGpu, RDC_FI_ECC_XGMI_WAFL_UE, RDC_ST_OK, 1);
  rdc_health_response_t single = check();
  ASSERT_EQ(1u, single.incidents_count);
  EXPECT_EQ(RDC_HEALTH_WATCH_XGMI, single.incidents[0].component);
  EXPECT_EQ(RDC_FR_XGMI_SINGLE_ERROR, single.incidents[0].error.code);
  EXPECT_EQ(RDC_HEALTH_RESULT_FAIL, single.overall_health);

  fetcher_->set(kGpu, RDC_FI_ECC_XGMI_WAFL_UE, RDC_ST_OK, 3);
  rdc_health_response_t multiple = check();
  ASSERT_EQ(1u, multiple.incidents_count);
  EXPECT_EQ(RDC_FR_XGMI_MULTIPLE_ERROR, multiple.incidents[0].error.code);
}

TEST_F(HealthWatchTest, ReadableLegacyXgmiStatusNeverTouchesFallback) {
  fetcher_->set(kGpu, RDC_HEALTH_XGMI_ERROR, RDC_ST_OK, AMDSMI_XGMI_STATUS_NO_ERRORS);
  fetcher_->set(kGpu, RDC_FI_ECC_XGMI_WAFL_UE, RDC_ST_OK, 5);
  ASSERT_EQ(RDC_ST_OK, table_->rdc_health_set(group_, RDC_HEALTH_WATCH_XGMI));
  EXPECT_EQ(0u, check().incidents_count);

  fetcher_->set(kGpu, RDC_HEALTH_XGMI_ERROR, RDC_ST_OK, AMDSMI_XGMI_STATUS_ERROR);
  rdc_health_response_t response = check();
  ASSERT_EQ(1u, response.incidents_count);
  EXPECT_EQ(RDC_FR_XGMI_SINGLE_ERROR, response.incidents[0].error.code);
  EXPECT_EQ(0, fetcher_->fetches(kGpu, RDC_FI_ECC_XGMI_WAFL_UE));
}

TEST_F(HealthWatchTest, UnsupportedFieldIsProbedOnceAndNotRefetched) {
  fetcher_->set(kGpu, RDC_HEALTH_PCIE_REPLAY_COUNT, RDC_ST_OK, 0);
  fetcher_->set(kGpu, RDC_HEALTH_THERMAL_THROTTLE_TIME, RDC_ST_NOT_SUPPORTED);
  ASSERT_EQ(RDC_ST_OK,
            table_->rdc_health_set(group_, RDC_HEALTH_WATCH_PCIE | RDC_HEALTH_WATCH_THERMAL));
  check();
  check();
  EXPECT_EQ(1, fetcher_->fetches(kGpu, RDC_HEALTH_THERMAL_THROTTLE_TIME));
  EXPECT_LT(1, fetcher_->fetches(kGpu, RDC_HEALTH_PCIE_REPLAY_COUNT));
}

TEST_F(HealthWatchTest, TransientProbeFailureStaysWatched) {
  fetcher_->set(kGpu, RDC_HEALTH_PCIE_REPLAY_COUNT, RDC_ST_UNKNOWN_ERROR);
  ASSERT_EQ(RDC_ST_OK, table_->rdc_health_set(group_, RDC_HEALTH_WATCH_PCIE));
  int probed = fetcher_->fetches(kGpu, RDC_HEALTH_PCIE_REPLAY_COUNT);
  check();
  EXPECT_LT(probed, fetcher_->fetches(kGpu, RDC_HEALTH_PCIE_REPLAY_COUNT));
}

TEST_F(HealthWatchTest, NothingReadableLeavesNoWatch) {
  {
    CaptureLog log;
    EXPECT_EQ(RDC_ST_NOT_SUPPORTED, table_->rdc_health_set(group_, RDC_HEALTH_WATCH_XGMI));
    EXPECT_TRUE(log.has_error("is not supported on GPU"));
  }
  unsigned int components = 1;
  EXPECT_EQ(RDC_ST_OK, table_->rdc_health_get(group_, &components));
  EXPECT_EQ(0u, components);
  rdc_health_response_t response = {};
  EXPECT_EQ(RDC_ST_NOT_FOUND, table_->rdc_health_check(group_, &response));
}

TEST_F(HealthWatchTest, MemoryCheckEvaluatesRetiredPagesWhenPendingIsUnreadable) {
  fetcher_->set(kGpu, RDC_FI_ECC_UNCORRECT_TOTAL, RDC_ST_OK, 0);
  fetcher_->set(kGpu, RDC_HEALTH_PENDING_PAGE_NUM, RDC_ST_NOT_SUPPORTED);
  fetcher_->set(kGpu, RDC_HEALTH_RETIRED_PAGE_NUM, RDC_ST_OK, 10);
  fetcher_->set(kGpu, RDC_HEALTH_RETIRED_PAGE_LIMIT, RDC_ST_OK, 5);
  ASSERT_EQ(RDC_ST_OK, table_->rdc_health_set(group_, RDC_HEALTH_WATCH_MEM));

  rdc_health_response_t response = check();
  bool over_limit = false;
  for (unsigned int i = 0; i < response.incidents_count; i++) {
    over_limit |= response.incidents[i].error.code == RDC_FR_RETIRED_PAGES_LIMIT;
  }
  EXPECT_TRUE(over_limit);
}

}  // namespace
