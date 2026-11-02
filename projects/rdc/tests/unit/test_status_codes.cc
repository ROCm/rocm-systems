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

// Statuses that RDC returns for a GPU group that does not exist, and the status
// strings. Needs no GPU: the test builds RdcWatchTableImpl.cc and
// RdcGroupSettingsImpl.cc itself, because librdc initializes AMD SMI when it is
// loaded, and links only librdc_bootstrap.

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>

#include "rdc/rdc.h"
#include "rdc_lib/RdcNotification.h"
#include "rdc_lib/impl/RdcGroupSettingsImpl.h"
#include "rdc_lib/impl/RdcWatchTableImpl.h"

// Defined in RdcNotificationImpl.cc, which this test does not build. Only
// rdc_field_listen_notif() reads it.
const uint32_t amd::rdc::kMaxRSMIEvents = 64;

namespace {

constexpr rdc_gpu_group_t kMissingGroup = 4242;
constexpr unsigned int kComponents = RDC_HEALTH_WATCH_PCIE | RDC_HEALTH_WATCH_MEM;

class HealthStatus : public ::testing::Test {
 protected:
  uint32_t field_group_count() {
    rdc_field_grp_t ids[RDC_MAX_NUM_FIELD_GROUPS];
    uint32_t count = 0;
    EXPECT_EQ(groups_->rdc_group_field_get_all_ids(ids, &count), RDC_ST_OK);
    return count;
  }

  std::shared_ptr<amd::rdc::RdcGroupSettingsImpl> groups_ =
      std::make_shared<amd::rdc::RdcGroupSettingsImpl>(nullptr);
  // The calls under test return before they use the cache, the metric fetcher, the
  // modules or the notifications, so those stay null.
  amd::rdc::RdcWatchTableImpl watch_table_{groups_, nullptr, nullptr, nullptr, nullptr};
};

TEST_F(HealthStatus, SetOnMissingGroupLeavesNoFieldGroup) {
  const uint32_t before = field_group_count();
  EXPECT_EQ(watch_table_.rdc_health_set(kMissingGroup, kComponents), RDC_ST_GROUP_NOT_FOUND);
  EXPECT_EQ(field_group_count(), before);
}

TEST_F(HealthStatus, GetOnMissingGroupReturnsGroupNotFound) {
  unsigned int components = kComponents;
  EXPECT_EQ(watch_table_.rdc_health_get(kMissingGroup, &components), RDC_ST_GROUP_NOT_FOUND);
}

TEST_F(HealthStatus, GetOnGroupWithoutWatchesReturnsNoComponents) {
  rdc_gpu_group_t group_id = 0;
  ASSERT_EQ(groups_->rdc_group_gpu_create("no-watches", &group_id), RDC_ST_OK);
  unsigned int components = kComponents;
  EXPECT_EQ(watch_table_.rdc_health_get(group_id, &components), RDC_ST_OK);
  EXPECT_EQ(components, 0u);
}

TEST(StatusString, DescribesEveryStatus) {
  // RDC_ST_FLDGROUP_NOT_FOUND is the last status before RDC_ST_UNKNOWN_ERROR.
  for (uint32_t status = RDC_ST_OK; status <= RDC_ST_FLDGROUP_NOT_FOUND; ++status) {
    EXPECT_STRNE(rdc_status_string(static_cast<rdc_status_t>(status)), "Unknown")
        << "status " << status;
  }
  EXPECT_STREQ(rdc_status_string(static_cast<rdc_status_t>(RDC_ST_FLDGROUP_NOT_FOUND + 1)),
               "Unknown")
      << "A status after RDC_ST_FLDGROUP_NOT_FOUND has a string; extend the loop bound to it.";
  EXPECT_STRNE(rdc_status_string(RDC_ST_UNKNOWN_ERROR), "Unknown");
}

}  // namespace
