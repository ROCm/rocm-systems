// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "evt_notif_read_write.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "test_common.h"
#include "test_utils.h"

namespace {

// One reset can yield several records on every device; the read loop drains what does not fit.
constexpr uint32_t kEventsPerDevice = 20;
// Only a guard against an endless event stream; the loop normally ends on NO_DATA.
constexpr int kMaxReads = 100;
constexpr int kFirstReadTimeoutMs = 10000;
constexpr int kDrainReadTimeoutMs = 1000;

constexpr uint64_t AllEventsMask() {
  uint64_t mask = 0;
  for (uint32_t evt = AMDSMI_EVT_NOTIF_FIRST; evt <= AMDSMI_EVT_NOTIF_LAST; ++evt) {
    mask |= AMDSMI_EVENT_MASK_FROM_INDEX(evt);
  }
  return mask;
}

// Subscribes to every defined event type; the single place to change if that set changes.
constexpr uint64_t kAllEventsMask = AllEventsMask();

// amdsmi_reset_gpu() is destructive, so generating reset events is opt-in.
bool ResetRequested() {
  const char* env = std::getenv("AMDSMI_TEST_TRIGGER_RESET");
  return env != nullptr && std::string(env) == "1";
}

// One reset covers every device in an XGMI hive, so only one device per hive needs one.
std::vector<uint32_t> ResetTargets(const amdsmi_processor_handle* handles, uint32_t count) {
  std::vector<uint32_t> targets;
  std::set<uint64_t> seen_hives;
  for (uint32_t i = 0; i < count; ++i) {
    amdsmi_xgmi_info_t info = {};
    const bool in_hive =
        amdsmi_get_xgmi_info(handles[i], &info) == AMDSMI_STATUS_SUCCESS && info.xgmi_hive_id != 0;
    if (!in_hive || seen_hives.insert(info.xgmi_hive_id).second) {
      targets.push_back(i);
    }
  }
  return targets;
}

}  // namespace

TestEvtNotifReadWrite::TestEvtNotifReadWrite() : TestBase() {
  set_title("AMDSMI Event Notification Read/Write Test");
  set_description(
      "The Event Notification Read/Write tests verifies that "
      "we can configure to collect various event types and then read them");
}

TestEvtNotifReadWrite::~TestEvtNotifReadWrite(void) {}

void TestEvtNotifReadWrite::SetUp(void) {
  TestBase::SetUp();
  return;
}

void TestEvtNotifReadWrite::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestEvtNotifReadWrite::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestEvtNotifReadWrite::Close() {
  // This will close handles opened within amdsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

void TestEvtNotifReadWrite::Run(void) {
  amdsmi_status_t ret;
  uint32_t dv_ind;

  TestBase::Run();
  PRINT_VERBOSITY();
  if (num_monitor_devs() == 0) {
    return;
  }

  if (setup_failed_) {
    IF_VERB(STANDARD) { std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl; }
    return;
  }

  const bool trigger_reset = ResetRequested();

  for (dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    DISPLAY_AMDSMI_API("amdsmi_init_gpu_event_notification", "gpu=" + std::to_string(dv_ind),
                       VERB(STANDARD));
    ret = amdsmi_init_gpu_event_notification(processor_handles_[dv_ind]);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
      return;
    }
    ASSERT_EQ(ret, AMDSMI_STATUS_SUCCESS);
    std::ostringstream mask_desc;
    mask_desc << "gpu=" << dv_ind << ", mask=0x" << std::hex << std::uppercase << kAllEventsMask;
    DISPLAY_AMDSMI_API("amdsmi_set_gpu_event_notification_mask", mask_desc.str(), VERB(STANDARD));
    ret = amdsmi_set_gpu_event_notification_mask(processor_handles_[dv_ind], kAllEventsMask);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    ASSERT_EQ(ret, AMDSMI_STATUS_SUCCESS);
  }

  if (trigger_reset) {
    const std::vector<uint32_t> targets = ResetTargets(processor_handles_, num_monitor_devs());
    IF_VERB(STANDARD) {
      if (targets.size() < num_monitor_devs()) {
        std::cout << "\tResetting " << targets.size() << " of " << num_monitor_devs()
                  << " devices; the others share an XGMI hive and are reset with it." << std::endl;
      }
    }
    for (uint32_t target : targets) {
      DISPLAY_AMDSMI_API("amdsmi_reset_gpu", "gpu=" + std::to_string(target), VERB(STANDARD));
      ret = amdsmi_reset_gpu(processor_handles_[target]);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
      ASSERT_EQ(ret, AMDSMI_STATUS_SUCCESS);
    }
  } else {
    IF_VERB(STANDARD) {
      std::cout << "\tSKIPPED amdsmi_reset_gpu: set AMDSMI_TEST_TRIGGER_RESET=1 to generate "
                   "reset events."
                << std::endl;
    }
  }

  std::map<amdsmi_processor_handle, std::set<amdsmi_evt_notification_type_t>> seen_events;
  const uint32_t max_events = kEventsPerDevice * num_monitor_devs();
  std::vector<amdsmi_evt_notification_data_t> data(max_events);

  // Event records carry a raw handle; map it back to the gpu index used elsewhere in the output.
  auto gpu_label = [&](amdsmi_processor_handle handle) {
    for (uint32_t idx = 0; idx < num_monitor_devs(); ++idx) {
      if (processor_handles_[idx] == handle) {
        return std::to_string(idx);
      }
    }
    return std::string("?");
  };

  // A get can return as soon as the first event lands (e.g. PRE_RESET), so keep reading
  // until the stream goes quiet.
  for (int read = 0; read < kMaxReads; ++read) {
    uint32_t num_elem = max_events;
    const int timeout_ms = (read == 0) ? kFirstReadTimeoutMs : kDrainReadTimeoutMs;
    DISPLAY_AMDSMI_API("amdsmi_get_gpu_event_notification",
                       "timeout_ms=" + std::to_string(timeout_ms), VERB(STANDARD));
    ret = amdsmi_get_gpu_event_notification(timeout_ms, &num_elem, data.data());
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_INSUFFICIENT_SIZE, AMDSMI_STATUS_NO_DATA);
    if (ret == AMDSMI_STATUS_NO_DATA) {
      IF_VERB(STANDARD) {
        std::cout << (read == 0 ? "\tNo events were collected."
                                : "\tNo further events were collected.")
                  << std::endl;
      }
      break;
    }
    if (ret != AMDSMI_STATUS_SUCCESS && ret != AMDSMI_STATUS_INSUFFICIENT_SIZE) {
      // This should always fail. We want to print out the return code.
      EXPECT_EQ(ret, AMDSMI_STATUS_SUCCESS)
          << "Unexpected return code for  amdsmi_get_gpu_event_notification()";
      break;
    }
    EXPECT_LE(num_elem, max_events)
        << "Expected the number of elements found to be <= buffer size (" << max_events << ")";
    for (uint32_t i = 0; i < num_elem && i < max_events; ++i) {
      seen_events[data[i].processor_handle].insert(data[i].event);
      IF_VERB(STANDARD) {
        std::cout << "\tdv_handle=" << data[i].processor_handle
                  << "[gpu=" << gpu_label(data[i].processor_handle) << "]"
                  << "\t Type: " << NameFromEvtNotifType(data[i].event)
                  << "\t Msg:  " << data[i].message << std::endl;
      }
    }
    IF_VERB(STANDARD) {
      if (ret == AMDSMI_STATUS_INSUFFICIENT_SIZE) {
        std::cout << "\t\tBuffer size is " << max_events << ", but more events are available."
                  << std::endl;
      }
    }
  }

  // A forced reset must produce both reset events on every device.
  if (trigger_reset) {
    for (dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
      const auto& events = seen_events[processor_handles_[dv_ind]];
      EXPECT_EQ(events.count(AMDSMI_EVT_NOTIF_GPU_PRE_RESET), 1u)
          << "gpu " << dv_ind << ": GPU_PRE_RESET not received after amdsmi_reset_gpu";
      EXPECT_EQ(events.count(AMDSMI_EVT_NOTIF_GPU_POST_RESET), 1u)
          << "gpu " << dv_ind << ": GPU_POST_RESET not received after amdsmi_reset_gpu";
    }
  }

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    DISPLAY_AMDSMI_API("amdsmi_stop_gpu_event_notification", "", VERB(STANDARD));
    ret = amdsmi_stop_gpu_event_notification(processor_handles_[dv_ind]);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    ASSERT_EQ(ret, AMDSMI_STATUS_SUCCESS);
  }
}
