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

// RdcNotificationImpl against fake AMD SMI event calls: which events listen() reports, and the
// event mask that each GPU gets from its listeners; and RdcWatchTableImpl's listeners: a watch of
// event fields, and a job start that records its listeners while a stop of the job runs. The
// test compiles RdcNotificationImpl.cc, RdcWatchTableImpl.cc and RdcGroupSettingsImpl.cc and
// defines the AMD SMI calls, the SmiUtils.cc lookups and ModifyCapability() they make, so it
// needs neither librdc nor amd_smi nor a GPU; librdc_bootstrap provides the logger,
// strncpy_with_null() and the entity codec.

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "common/rdc_capabilities.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcCacheManager.h"
#include "rdc_lib/RdcModuleMgr.h"
#include "rdc_lib/RdcNotification.h"
#include "rdc_lib/RdcPartition.h"
#include "rdc_lib/RdcTelemetry.h"
#include "rdc_lib/impl/RdcGroupSettingsImpl.h"
#include "rdc_lib/impl/RdcNotificationImpl.h"
#include "rdc_lib/impl/RdcWatchTableImpl.h"
#include "rdc_lib/impl/SmiUtils.h"

namespace {

constexpr uint32_t kGpus = 4;

// What the fake AMD SMI does and saw.
struct FakeSmi {
  amdsmi_status_t lookup[kGpus];       // get_processor_handle_from_id()
  uint32_t lookups[kGpus];             // calls of get_processor_handle_from_id()
  amdsmi_status_t init_status[kGpus];  // amdsmi_init_gpu_event_notification()
  amdsmi_status_t mask_status[kGpus];  // amdsmi_set_gpu_event_notification_mask()
  amdsmi_status_t stop_status[kGpus];  // amdsmi_stop_gpu_event_notification()
  int capability_error;                // ModifyCapability() when enabling
  int relinquish_error;                // ModifyCapability() when disabling
  // When set, get_processor_handle_from_id()'s lookups ("lookup 0") and ModifyCapability()'s
  // raises and drops ("raise", "drop") go into calls too.
  bool trace;
  uint64_t mask[kGpus];            // the last mask written successfully
  std::vector<std::string> calls;  // "init 0", "mask 0 0x1", "stop 0", in order

  // amdsmi_get_gpu_event_notification(): the status, the events written (all of them with
  // ignore_room, else at most the room given) and the count reported (the number written unless
  // set).
  amdsmi_status_t get_status;
  std::vector<amdsmi_evt_notification_data_t> written;
  bool ignore_room;
  int64_t reported;
};
FakeSmi g_smi;

void reset_fake() {
  for (uint32_t i = 0; i < kGpus; ++i) {
    g_smi.lookup[i] = AMDSMI_STATUS_SUCCESS;
    g_smi.lookups[i] = 0;
    g_smi.init_status[i] = AMDSMI_STATUS_SUCCESS;
    g_smi.mask_status[i] = AMDSMI_STATUS_SUCCESS;
    g_smi.stop_status[i] = AMDSMI_STATUS_SUCCESS;
    g_smi.mask[i] = 0;
  }
  g_smi.capability_error = 0;
  g_smi.relinquish_error = 0;
  g_smi.trace = false;
  g_smi.calls.clear();
  g_smi.get_status = AMDSMI_STATUS_SUCCESS;
  g_smi.written.clear();
  g_smi.ignore_room = false;
  g_smi.reported = -1;
}

amdsmi_processor_handle handle_of(uint32_t gpu) {
  return reinterpret_cast<amdsmi_processor_handle>(uintptr_t{0x1000} + gpu);
}

// The GPU of a handle, or kGpus for one that is not a GPU's.
uint32_t gpu_of(amdsmi_processor_handle h) {
  const uintptr_t v = reinterpret_cast<uintptr_t>(h);
  return v >= 0x1000 && v < 0x1000 + kGpus ? static_cast<uint32_t>(v - 0x1000) : kGpus;
}

std::string hex(uint64_t v) {
  char buf[32];
  snprintf(buf, sizeof(buf), "0x%llx", static_cast<unsigned long long>(v));
  return buf;
}

uint64_t bit(amdsmi_evt_notification_type_t event) { return AMDSMI_EVENT_MASK_FROM_INDEX(event); }

amdsmi_evt_notification_data_t event_of(uint32_t gpu, uint32_t event, const char* message) {
  amdsmi_evt_notification_data_t e;
  memset(&e, 0, sizeof(e));
  e.processor_handle =
      gpu < kGpus ? handle_of(gpu) : reinterpret_cast<amdsmi_processor_handle>(uintptr_t{0xDEAD});
  e.event = static_cast<amdsmi_evt_notification_type_t>(event);
  snprintf(e.message, sizeof(e.message), "%s", message);
  return e;
}

}  // namespace

extern "C" {

amdsmi_status_t amdsmi_init_gpu_event_notification(amdsmi_processor_handle h) {
  const uint32_t gpu = gpu_of(h);
  g_smi.calls.push_back("init " + std::to_string(gpu));
  return g_smi.init_status[gpu];
}

amdsmi_status_t amdsmi_set_gpu_event_notification_mask(amdsmi_processor_handle h, uint64_t mask) {
  const uint32_t gpu = gpu_of(h);
  g_smi.calls.push_back("mask " + std::to_string(gpu) + " " + hex(mask));
  if (g_smi.mask_status[gpu] == AMDSMI_STATUS_SUCCESS) {
    g_smi.mask[gpu] = mask;
  }
  return g_smi.mask_status[gpu];
}

amdsmi_status_t amdsmi_stop_gpu_event_notification(amdsmi_processor_handle h) {
  const uint32_t gpu = gpu_of(h);
  g_smi.calls.push_back("stop " + std::to_string(gpu));
  return g_smi.stop_status[gpu];
}

amdsmi_status_t amdsmi_get_gpu_event_notification(int, uint32_t* num_elem,
                                                  amdsmi_evt_notification_data_t* data) {
  if (g_smi.get_status != AMDSMI_STATUS_SUCCESS) {
    return g_smi.get_status;
  }
  // A well-behaved library writes at most *num_elem events. With ignore_room it writes past them,
  // but within the kMaxRSMIEvents entries of the buffer that listen() hands it.
  const uint32_t size = static_cast<uint32_t>(g_smi.written.size());
  const uint32_t n =
      g_smi.ignore_room ? std::min(size, amd::rdc::kMaxRSMIEvents) : std::min(*num_elem, size);
  for (uint32_t i = 0; i < n; ++i) {
    data[i] = g_smi.written[i];
  }
  *num_elem = g_smi.reported >= 0 ? static_cast<uint32_t>(g_smi.reported) : n;
  return AMDSMI_STATUS_SUCCESS;
}

}  // extern "C"

namespace amd {
namespace rdc {

rdc_status_t Smi2RdcError(amdsmi_status_t status) {
  return status == AMDSMI_STATUS_SUCCESS ? RDC_ST_OK : RDC_ST_UNKNOWN_ERROR;
}

amdsmi_status_t get_processor_handle_from_id(uint32_t gpu_id, amdsmi_processor_handle* handle) {
  if (gpu_id >= kGpus) {
    return AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS;
  }
  ++g_smi.lookups[gpu_id];
  if (g_smi.trace) {
    g_smi.calls.push_back("lookup " + std::to_string(gpu_id));
  }
  if (g_smi.lookup[gpu_id] == AMDSMI_STATUS_SUCCESS) {
    *handle = handle_of(gpu_id);
  }
  return g_smi.lookup[gpu_id];
}

amdsmi_status_t get_gpu_id_from_processor_handle(amdsmi_processor_handle handle,
                                                 uint32_t* gpu_index) {
  const uint32_t gpu = gpu_of(handle);
  if (gpu == kGpus || g_smi.lookup[gpu] != AMDSMI_STATUS_SUCCESS) {
    return AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS;
  }
  *gpu_index = gpu;
  return AMDSMI_STATUS_SUCCESS;
}

int ModifyCapability(cap_value_t cap, cap_flag_t type, bool enable) {
  // The only capability that RdcNotificationImpl may raise.
  EXPECT_EQ(cap, CAP_DAC_OVERRIDE);
  EXPECT_EQ(type, CAP_EFFECTIVE);
  if (g_smi.trace) {
    g_smi.calls.push_back(enable ? "raise" : "drop");
  }
  return enable ? g_smi.capability_error : g_smi.relinquish_error;
}

}  // namespace rdc
}  // namespace amd

namespace {

using amd::rdc::rdc_evnt_notification_t;
using amd::rdc::RdcNotificationImpl;

class Notifications : public ::testing::Test {
 protected:
  void SetUp() override { reset_fake(); }

  // listen() with room for `room` events, in an array with `extra` more entries after them.
  rdc_status_t listen(uint32_t room, uint32_t* count, uint32_t extra = 8) {
    events_.assign(room + extra, rdc_evnt_notification_t{});
    for (auto& e : events_) {
      e.gpu_id = 0xDEAD;
    }
    *count = room;
    return notifications_.listen(events_.data(), count, 0);
  }

  // The calls the fake saw since the last take().
  std::vector<std::string> take() {
    std::vector<std::string> calls;
    calls.swap(g_smi.calls);
    return calls;
  }

  RdcNotificationImpl notifications_;
  std::vector<rdc_evnt_notification_t> events_;
};

// Whether the logger, which writes to stdout, wrote an error line that contains text.
bool logged_error(const std::string& out, const std::string& text) {
  std::istringstream lines(out);
  for (std::string line; std::getline(lines, line);) {
    if (line.find(" ERROR ") != std::string::npos && line.find(text) != std::string::npos) {
      return true;
    }
  }
  return false;
}

TEST_F(Notifications, ListenReadsNoMoreEventsThanTheLibraryWrote) {
  // A library that reports more events than it had room for: RDC uses only those it wrote.
  g_smi.written = {event_of(0, AMDSMI_EVT_NOTIF_VMFAULT, "zero"),
                   event_of(1, AMDSMI_EVT_NOTIF_THERMAL_THROTTLE, "one")};
  g_smi.reported = 1000;
  uint32_t count = 0;
  ASSERT_EQ(listen(4, &count), RDC_ST_OK);
  ASSERT_EQ(count, 2u);
  EXPECT_EQ(events_[0].gpu_id, 0u);
  EXPECT_EQ(events_[0].field.field_id, RDC_EVNT_NOTIF_VMFAULT);
  EXPECT_STREQ(events_[0].field.value.str, "zero");
  EXPECT_EQ(events_[1].gpu_id, 1u);
  EXPECT_EQ(events_[1].field.field_id, RDC_EVNT_NOTIF_THERMAL_THROTTLE);
  EXPECT_STREQ(events_[1].field.value.str, "one");
  for (size_t i = 2; i < events_.size(); ++i) {
    EXPECT_EQ(events_[i].gpu_id, 0xDEADu) << "entry " << i << " written";
  }
}

TEST_F(Notifications, ListenReportsNoMoreEventsThanItHasRoomFor) {
  // A library that writes and reports more events than it had room for.
  g_smi.ignore_room = true;
  for (uint32_t i = 0; i < 6; ++i) {
    g_smi.written.push_back(event_of(i % 2, AMDSMI_EVT_NOTIF_VMFAULT, "x"));
  }
  uint32_t count = 0;
  ASSERT_EQ(listen(4, &count), RDC_ST_OK);
  EXPECT_EQ(count, 4u);
  for (size_t i = 4; i < events_.size(); ++i) {
    EXPECT_EQ(events_[i].gpu_id, 0xDEADu) << "entry " << i << " written";
  }
}

TEST_F(Notifications, ListenIgnoresEntriesTheLibraryDidNotWrite) {
  // The first read leaves four events where the second one's go (the second listen()'s buffer
  // reuses the first one's stack, so this depends on the compiler's stack layout); the second
  // library reports four but writes two.
  for (uint32_t i = 0; i < 4; ++i) {
    g_smi.written.push_back(event_of(1, AMDSMI_EVT_NOTIF_VMFAULT, "first"));
  }
  uint32_t count = 0;
  ASSERT_EQ(listen(64, &count), RDC_ST_OK);
  ASSERT_EQ(count, 4u);
  g_smi.written.resize(2);
  g_smi.reported = 4;
  ASSERT_EQ(listen(64, &count), RDC_ST_OK);
  EXPECT_EQ(count, 2u);
}

TEST_F(Notifications, ListenSkipsEventsOfKindsRdcDoesNotKnow) {
  g_smi.written = {event_of(0, AMDSMI_EVT_NOTIF_VMFAULT, "known"), event_of(0, 0, "none"),
                   event_of(1, 14, "newer"), event_of(1, AMDSMI_EVT_NOTIF_PROCESS_START, "start")};
  uint32_t count = 0;
  ASSERT_EQ(listen(64, &count), RDC_ST_OK);
  ASSERT_EQ(count, 2u);
  EXPECT_EQ(events_[0].field.field_id, RDC_EVNT_NOTIF_VMFAULT);
  EXPECT_EQ(events_[1].gpu_id, 1u);
  EXPECT_EQ(events_[1].field.field_id, RDC_EVNT_NOTIF_PROCESS_START);
  EXPECT_STREQ(events_[1].field.value.str, "start");
}

TEST_F(Notifications, ListenSkipsEventsOfGpusRdcDoesNotList) {
  g_smi.lookup[2] = AMDSMI_STATUS_NOT_FOUND;  // RDC's table holds no handle for GPU 2
  g_smi.written = {event_of(0, AMDSMI_EVT_NOTIF_VMFAULT, "zero"),
                   event_of(kGpus, AMDSMI_EVT_NOTIF_VMFAULT, "unknown handle"),
                   event_of(2, AMDSMI_EVT_NOTIF_VMFAULT, "unavailable"),
                   event_of(1, AMDSMI_EVT_NOTIF_VMFAULT, "one")};
  uint32_t count = 0;
  ASSERT_EQ(listen(64, &count), RDC_ST_OK);
  ASSERT_EQ(count, 2u);
  EXPECT_EQ(events_[0].gpu_id, 0u);
  EXPECT_STREQ(events_[0].field.value.str, "zero");
  EXPECT_EQ(events_[1].gpu_id, 1u);
  EXPECT_STREQ(events_[1].field.value.str, "one");
}

TEST_F(Notifications, ListenGivesEveryEventItsOwnLaterTimestamp) {
  // Two events of one kind on one GPU, read together: rdc_field_get_value_since() tells them
  // apart only by their timestamps.
  std::vector<uint64_t> ts;
  for (int call = 0; call < 2; ++call) {
    g_smi.written = {event_of(0, AMDSMI_EVT_NOTIF_VMFAULT, "a"),
                     event_of(0, AMDSMI_EVT_NOTIF_VMFAULT, "b"),
                     event_of(1, AMDSMI_EVT_NOTIF_VMFAULT, "c")};
    uint32_t count = 0;
    ASSERT_EQ(listen(64, &count), RDC_ST_OK);
    ASSERT_EQ(count, 3u);
    for (uint32_t i = 0; i < count; ++i) {
      ts.push_back(events_[i].field.ts);
    }
  }
  for (size_t i = 1; i < ts.size(); ++i) {
    EXPECT_LT(ts[i - 1], ts[i]) << "event " << i;
  }
}

TEST_F(Notifications, ListenReportsNoEventOnAnError) {
  g_smi.get_status = AMDSMI_STATUS_NO_DATA;
  uint32_t count = 0;
  EXPECT_NE(listen(64, &count), RDC_ST_OK);
  EXPECT_EQ(count, 0u);
}

TEST_F(Notifications, ListenersOfTwoWatchesAddUp) {
  ASSERT_EQ(notifications_.add_event_listeners({{0, RDC_EVNT_NOTIF_VMFAULT}}), RDC_ST_OK);
  EXPECT_EQ(g_smi.mask[0], bit(AMDSMI_EVT_NOTIF_VMFAULT));
  // A job's process events do not replace the watch's page faults.
  ASSERT_EQ(notifications_.add_event_listeners(
                {{0, RDC_EVNT_NOTIF_PROCESS_START}, {0, RDC_EVNT_NOTIF_PROCESS_END}}),
            RDC_ST_OK);
  EXPECT_EQ(g_smi.mask[0], bit(AMDSMI_EVT_NOTIF_VMFAULT) | bit(AMDSMI_EVT_NOTIF_PROCESS_START) |
                               bit(AMDSMI_EVT_NOTIF_PROCESS_END));
  // Removing the job's keeps the watch's, without stopping the GPU's events.
  take();
  ASSERT_EQ(notifications_.remove_event_listeners(
                {{0, RDC_EVNT_NOTIF_PROCESS_START}, {0, RDC_EVNT_NOTIF_PROCESS_END}}),
            RDC_ST_OK);
  EXPECT_EQ(take(), (std::vector<std::string>{"init 0", "mask 0 0x1"}));
  // The last listener's removal writes 0 and stops.
  ASSERT_EQ(notifications_.remove_event_listeners({{0, RDC_EVNT_NOTIF_VMFAULT}}), RDC_ST_OK);
  EXPECT_EQ(take(), (std::vector<std::string>{"mask 0 0x0", "stop 0"}));
}

TEST_F(Notifications, ListenersOfOneEventAreCounted) {
  const std::vector<RdcFieldKey> vm = {{0, RDC_EVNT_NOTIF_VMFAULT}};
  // A field that is not an event adds no listener and makes no call.
  ASSERT_EQ(notifications_.add_event_listeners({{0, RDC_FI_GPU_TEMP}}), RDC_ST_OK);
  EXPECT_EQ(take(), std::vector<std::string>{});
  // Removing a listener that was never added changes nothing, and is not counted against a
  // later one.
  ASSERT_EQ(notifications_.remove_event_listeners(vm), RDC_ST_OK);
  ASSERT_EQ(notifications_.remove_event_listeners(vm), RDC_ST_OK);
  EXPECT_EQ(take(), std::vector<std::string>{});
  ASSERT_EQ(notifications_.add_event_listeners(vm), RDC_ST_OK);
  ASSERT_EQ(notifications_.add_event_listeners(vm), RDC_ST_OK);
  EXPECT_EQ(take(), (std::vector<std::string>{"init 0", "mask 0 0x1"}));
  ASSERT_EQ(notifications_.remove_event_listeners(vm), RDC_ST_OK);
  EXPECT_EQ(take(), std::vector<std::string>{}) << "one listener is left";
  ASSERT_EQ(notifications_.remove_event_listeners(vm), RDC_ST_OK);
  EXPECT_EQ(take(), (std::vector<std::string>{"mask 0 0x0", "stop 0"}));
}

TEST_F(Notifications, AFailedStopLetsTheNextListenerEnableTheEventsAgain) {
  const std::vector<RdcFieldKey> vm = {{0, RDC_EVNT_NOTIF_VMFAULT}};
  ASSERT_EQ(notifications_.add_event_listeners(vm), RDC_ST_OK);
  g_smi.stop_status[0] = AMDSMI_STATUS_UNKNOWN_ERROR;
  take();
  EXPECT_NE(notifications_.remove_event_listeners(vm), RDC_ST_OK);
  EXPECT_EQ(take(), (std::vector<std::string>{"mask 0 0x0", "stop 0"}));
  // The stop may have closed the GPU's events: the next listener opens them and writes the mask.
  g_smi.stop_status[0] = AMDSMI_STATUS_SUCCESS;
  ASSERT_EQ(notifications_.add_event_listeners(vm), RDC_ST_OK);
  EXPECT_EQ(take(), (std::vector<std::string>{"init 0", "mask 0 0x1"}));
}

TEST_F(Notifications, AFailedMaskWriteOfOneGpuStillSetsTheOthers) {
  g_smi.mask_status[0] = AMDSMI_STATUS_UNKNOWN_ERROR;
  EXPECT_NE(notifications_.add_event_listeners(
                {{0, RDC_EVNT_NOTIF_VMFAULT}, {1, RDC_EVNT_NOTIF_VMFAULT}}),
            RDC_ST_OK);
  EXPECT_EQ(g_smi.mask[1], bit(AMDSMI_EVT_NOTIF_VMFAULT));
  // Only a call that names GPU 0 tries it again: each try logs its failure.
  take();
  EXPECT_EQ(notifications_.add_event_listeners({{2, RDC_EVNT_NOTIF_VMFAULT}}), RDC_ST_OK);
  EXPECT_EQ(take(), (std::vector<std::string>{"init 2", "mask 2 0x1"}));
  // That call writes the mask again, also for the same events.
  g_smi.mask_status[0] = AMDSMI_STATUS_SUCCESS;
  EXPECT_EQ(notifications_.add_event_listeners({{0, RDC_EVNT_NOTIF_VMFAULT}}), RDC_ST_OK);
  EXPECT_EQ(take(), (std::vector<std::string>{"init 0", "mask 0 0x1"}));
}

TEST_F(Notifications, AnErrorOfAGpuTheCallDoesNotNameIsNotItsError) {
  // GPU 0 is found after its listener came: the next call, for GPU 1, writes GPU 0's mask.
  g_smi.lookup[0] = AMDSMI_STATUS_NOT_FOUND;
  EXPECT_EQ(notifications_.add_event_listeners({{0, RDC_EVNT_NOTIF_VMFAULT}}), RDC_ST_OK);
  g_smi.lookup[0] = AMDSMI_STATUS_SUCCESS;
  g_smi.mask_status[0] = AMDSMI_STATUS_UNKNOWN_ERROR;
  EXPECT_EQ(notifications_.add_event_listeners({{1, RDC_EVNT_NOTIF_VMFAULT}}), RDC_ST_OK);
  EXPECT_EQ(take(), (std::vector<std::string>{"init 0", "mask 0 0x1", "init 1", "mask 1 0x1"}));
  // The write failed: only a call that names GPU 0 tries it again.
  EXPECT_EQ(notifications_.add_event_listeners({{2, RDC_EVNT_NOTIF_VMFAULT}}), RDC_ST_OK);
  EXPECT_EQ(take(), (std::vector<std::string>{"init 2", "mask 2 0x1"}));
}

TEST_F(Notifications, AFailedGpuWithoutListenersIsStoppedByAnyCall) {
  // GPU 0's write fails, and so does its lookup when its last listener goes.
  g_smi.mask_status[0] = AMDSMI_STATUS_UNKNOWN_ERROR;
  const std::vector<RdcFieldKey> vm = {{0, RDC_EVNT_NOTIF_VMFAULT}};
  EXPECT_NE(notifications_.add_event_listeners(vm), RDC_ST_OK);
  g_smi.mask_status[0] = AMDSMI_STATUS_SUCCESS;
  g_smi.lookup[0] = AMDSMI_STATUS_UNKNOWN_ERROR;
  EXPECT_NE(notifications_.remove_event_listeners(vm), RDC_ST_OK);
  g_smi.lookup[0] = AMDSMI_STATUS_SUCCESS;
  take();
  // Once its lookup works, any call stops its events, which its init opened.
  EXPECT_EQ(notifications_.add_event_listeners({{1, RDC_EVNT_NOTIF_VMFAULT}}), RDC_ST_OK);
  EXPECT_EQ(take(), (std::vector<std::string>{"mask 0 0x0", "stop 0", "init 1", "mask 1 0x1"}));
}

TEST_F(Notifications, AGpuWhoseMaskWriteFailedIsStoppedWhenItsListenersGo) {
  // Its events were opened, so they are closed even though no mask was written.
  g_smi.mask_status[0] = AMDSMI_STATUS_UNKNOWN_ERROR;
  const std::vector<RdcFieldKey> vm = {{0, RDC_EVNT_NOTIF_VMFAULT}};
  EXPECT_NE(notifications_.add_event_listeners(vm), RDC_ST_OK);
  EXPECT_EQ(take(), (std::vector<std::string>{"init 0", "mask 0 0x1"}));
  g_smi.mask_status[0] = AMDSMI_STATUS_SUCCESS;
  EXPECT_EQ(notifications_.remove_event_listeners(vm), RDC_ST_OK);
  EXPECT_EQ(take(), (std::vector<std::string>{"mask 0 0x0", "stop 0"}));
}

TEST_F(Notifications, ARelinquishFailureIsAnErrorButTheMaskIsKept) {
  g_smi.relinquish_error = 1;
  EXPECT_EQ(notifications_.add_event_listeners({{0, RDC_EVNT_NOTIF_VMFAULT}}), RDC_ST_PERM_ERROR);
  EXPECT_EQ(g_smi.mask[0], bit(AMDSMI_EVT_NOTIF_VMFAULT));
  // AMD SMI has GPU 0's mask: later calls do not write it again.
  g_smi.relinquish_error = 0;
  take();
  EXPECT_EQ(notifications_.add_event_listeners({{1, RDC_EVNT_NOTIF_VMFAULT}}), RDC_ST_OK);
  EXPECT_EQ(take(), (std::vector<std::string>{"init 1", "mask 1 0x1"}));
}

TEST_F(Notifications, ARelinquishFailureAfterAFailedInitIsAnError) {
  // The init failure alone is not an error; a capability left raised is.
  g_smi.init_status[0] = AMDSMI_STATUS_NOT_SUPPORTED;
  g_smi.relinquish_error = 1;
  EXPECT_EQ(notifications_.add_event_listeners({{0, RDC_EVNT_NOTIF_VMFAULT}}), RDC_ST_PERM_ERROR);
  EXPECT_EQ(take(), (std::vector<std::string>{"init 0"}));
}

TEST_F(Notifications, ACapabilityFailureIsAnError) {
  g_smi.capability_error = 1;
  EXPECT_EQ(notifications_.add_event_listeners({{0, RDC_EVNT_NOTIF_VMFAULT}}), RDC_ST_PERM_ERROR);
  EXPECT_EQ(take(), std::vector<std::string>{});
  g_smi.capability_error = 0;
  EXPECT_EQ(notifications_.add_event_listeners({{1, RDC_EVNT_NOTIF_VMFAULT}}), RDC_ST_OK);
  EXPECT_EQ(g_smi.mask[0], 0u) << "only a call that names GPU 0 tries it again";
  EXPECT_EQ(notifications_.add_event_listeners({{0, RDC_EVNT_NOTIF_PROCESS_START}}), RDC_ST_OK);
  EXPECT_EQ(g_smi.mask[0], bit(AMDSMI_EVT_NOTIF_VMFAULT) | bit(AMDSMI_EVT_NOTIF_PROCESS_START));
}

TEST_F(Notifications, OnlyTheInitAndTheMaskWriteRaiseTheCapability) {
  struct TraceOn {
    TraceOn() { g_smi.trace = true; }
    ~TraceOn() { g_smi.trace = false; }
  } trace_on;
  {
    // Constructing and destroying an RdcNotificationImpl raises nothing.
    RdcNotificationImpl fresh;
  }
  EXPECT_EQ(take(), std::vector<std::string>{});
  const std::vector<RdcFieldKey> vm = {{0, RDC_EVNT_NOTIF_VMFAULT}};
  ASSERT_EQ(notifications_.add_event_listeners(vm), RDC_ST_OK);
  // Raised after the lookup, for the init and the mask write; Relinquish() and the end of its
  // scope each drop it.
  EXPECT_EQ(take(), (std::vector<std::string>{"lookup 0", "raise", "init 0", "mask 0 0x1", "drop",
                                              "drop"}));
  // Calls that leave the mask as it is make no call at all, not even the lookup.
  ASSERT_EQ(notifications_.add_event_listeners(vm), RDC_ST_OK);
  ASSERT_EQ(notifications_.remove_event_listeners(vm), RDC_ST_OK);
  EXPECT_EQ(take(), std::vector<std::string>{});
  // The last listener goes: mask 0 and stop, without the capability.
  ASSERT_EQ(notifications_.remove_event_listeners(vm), RDC_ST_OK);
  EXPECT_EQ(take(), (std::vector<std::string>{"lookup 0", "mask 0 0x0", "stop 0"}));
  // A failed init and a failed raise leave no capability raised and raise it no more. The end of
  // the scope drops it also after a raise that failed.
  g_smi.init_status[3] = AMDSMI_STATUS_NOT_SUPPORTED;
  EXPECT_EQ(notifications_.add_event_listeners({{3, RDC_EVNT_NOTIF_VMFAULT}}), RDC_ST_OK);
  EXPECT_EQ(take(), (std::vector<std::string>{"lookup 3", "raise", "init 3", "drop", "drop"}));
  g_smi.capability_error = 1;
  EXPECT_EQ(notifications_.add_event_listeners({{3, RDC_EVNT_NOTIF_PROCESS_START}}),
            RDC_ST_PERM_ERROR);
  EXPECT_EQ(take(), (std::vector<std::string>{"lookup 3", "raise", "drop"}));
  g_smi.capability_error = 0;
  // A release that fails, after a failed init or after the mask write, raises nothing more.
  g_smi.relinquish_error = 1;
  EXPECT_EQ(notifications_.add_event_listeners({{3, RDC_EVNT_NOTIF_VMFAULT}}), RDC_ST_PERM_ERROR);
  EXPECT_EQ(take(), (std::vector<std::string>{"lookup 3", "raise", "init 3", "drop", "drop"}));
  EXPECT_EQ(notifications_.add_event_listeners(vm), RDC_ST_PERM_ERROR);
  EXPECT_EQ(take(), (std::vector<std::string>{"lookup 0", "raise", "init 0", "mask 0 0x1", "drop",
                                              "drop"}));
  g_smi.relinquish_error = 0;
  // A lookup that fails, for an unavailable GPU or with an error, raises nothing. GPU 1, still
  // unavailable, is looked up again by the second call.
  g_smi.lookup[1] = AMDSMI_STATUS_NOT_FOUND;
  EXPECT_EQ(notifications_.add_event_listeners({{1, RDC_EVNT_NOTIF_VMFAULT}}), RDC_ST_OK);
  EXPECT_EQ(take(), (std::vector<std::string>{"lookup 1"}));
  g_smi.lookup[2] = AMDSMI_STATUS_UNKNOWN_ERROR;
  EXPECT_NE(notifications_.add_event_listeners({{2, RDC_EVNT_NOTIF_VMFAULT}}), RDC_ST_OK);
  EXPECT_EQ(take(), (std::vector<std::string>{"lookup 1", "lookup 2"}));
  // Reading events raises nothing: neither the events it reports nor those it skips, of a kind RDC
  // does not know or of a GPU it does not list (GPU 1's lookup fails).
  g_smi.written = {event_of(0, AMDSMI_EVT_NOTIF_VMFAULT, "zero"), event_of(0, 0, "none"),
                   event_of(1, AMDSMI_EVT_NOTIF_VMFAULT, "one"),
                   event_of(3, AMDSMI_EVT_NOTIF_PROCESS_START, "three")};
  uint32_t count = 0;
  ASSERT_EQ(listen(4, &count), RDC_ST_OK);
  EXPECT_EQ(count, 2u);
  EXPECT_EQ(take(), std::vector<std::string>{});
  // Nor does a read whose count AMD SMI over-reports, which is logged as an error, or a read that
  // fails.
  g_smi.reported = 1000;
  testing::internal::CaptureStdout();
  const rdc_status_t clamped = listen(1, &count);
  const std::string out = testing::internal::GetCapturedStdout();
  ASSERT_EQ(clamped, RDC_ST_OK);
  EXPECT_EQ(count, 1u);
  EXPECT_TRUE(logged_error(out, "reported 1000 events, room for 1")) << out;
  g_smi.get_status = AMDSMI_STATUS_UNKNOWN_ERROR;
  EXPECT_NE(listen(4, &count), RDC_ST_OK);
  EXPECT_EQ(take(), std::vector<std::string>{});
}

TEST_F(Notifications, ALookupErrorIsAnError) {
  g_smi.lookup[0] = AMDSMI_STATUS_UNKNOWN_ERROR;
  EXPECT_NE(notifications_.add_event_listeners({{0, RDC_EVNT_NOTIF_VMFAULT}}), RDC_ST_OK);
  g_smi.lookup[0] = AMDSMI_STATUS_SUCCESS;
  // Only a call that names GPU 0 tries it again.
  EXPECT_EQ(notifications_.add_event_listeners({{1, RDC_EVNT_NOTIF_VMFAULT}}), RDC_ST_OK);
  EXPECT_EQ(take(), (std::vector<std::string>{"init 1", "mask 1 0x1"}));
  EXPECT_EQ(notifications_.add_event_listeners({{0, RDC_EVNT_NOTIF_PROCESS_START}}), RDC_ST_OK);
  EXPECT_EQ(take(), (std::vector<std::string>{"init 0", "mask 0 0x801"}));
}

const char kLookupError0[] = "Failed to get processor handle for GPU 0 ";

TEST_F(Notifications, ALookupErrorIsLoggedAsAnErrorOnlyByACallThatNamesTheGpu) {
  const std::vector<RdcFieldKey> vm = {{0, RDC_EVNT_NOTIF_VMFAULT}};
  ASSERT_EQ(notifications_.add_event_listeners(vm), RDC_ST_OK);
  g_smi.lookup[0] = AMDSMI_STATUS_UNKNOWN_ERROR;
  // The call that removes GPU 0's last listener names it.
  testing::internal::CaptureStdout();
  rdc_status_t status = notifications_.remove_event_listeners(vm);
  std::string out = testing::internal::GetCapturedStdout();
  EXPECT_NE(status, RDC_ST_OK);
  EXPECT_TRUE(logged_error(out, kLookupError0)) << out;
  // A call that does not name GPU 0 tries it again, to stop its events, and logs no error.
  const uint32_t lookups = g_smi.lookups[0];
  testing::internal::CaptureStdout();
  status = notifications_.add_event_listeners({{1, RDC_EVNT_NOTIF_VMFAULT}});
  out = testing::internal::GetCapturedStdout();
  EXPECT_EQ(status, RDC_ST_OK);
  EXPECT_FALSE(logged_error(out, kLookupError0)) << out;
  EXPECT_GT(g_smi.lookups[0], lookups);
}

TEST_F(Notifications, ALookupErrorOfAGpuWithListenersIsLoggedAsAnErrorByAnyCall) {
  // GPU 0 has a listener and no mask: it was unavailable, which is not a failure.
  g_smi.lookup[0] = AMDSMI_STATUS_NOT_FOUND;
  ASSERT_EQ(notifications_.add_event_listeners({{0, RDC_EVNT_NOTIF_VMFAULT}}), RDC_ST_OK);
  g_smi.lookup[0] = AMDSMI_STATUS_UNKNOWN_ERROR;
  // A call that does not name GPU 0 tries it, and logs its lookup error as an error.
  testing::internal::CaptureStdout();
  const rdc_status_t status = notifications_.add_event_listeners({{1, RDC_EVNT_NOTIF_VMFAULT}});
  const std::string out = testing::internal::GetCapturedStdout();
  EXPECT_EQ(status, RDC_ST_OK);
  EXPECT_TRUE(logged_error(out, kLookupError0)) << out;
}

TEST_F(Notifications, AFailedMaskWriteIsLoggedAlsoWhenTheReleaseFails) {
  g_smi.mask_status[0] = AMDSMI_STATUS_UNKNOWN_ERROR;
  g_smi.relinquish_error = 1;
  testing::internal::CaptureStdout();
  const rdc_status_t status = notifications_.add_event_listeners({{0, RDC_EVNT_NOTIF_VMFAULT}});
  const std::string out = testing::internal::GetCapturedStdout();
  EXPECT_EQ(status, RDC_ST_PERM_ERROR);
  EXPECT_TRUE(logged_error(out, "amdsmi_set_gpu_event_notification_mask() returned")) << out;
  EXPECT_TRUE(logged_error(out, "Failed to relinquish capabilities")) << out;
}

TEST_F(Notifications, AnInitFailureIsNotAnError) {
  // As on a kernel without SMI events: watches and job starts keep working.
  g_smi.init_status[0] = AMDSMI_STATUS_NOT_SUPPORTED;
  EXPECT_EQ(notifications_.add_event_listeners({{0, RDC_EVNT_NOTIF_VMFAULT}}), RDC_ST_OK);
  EXPECT_EQ(take(), (std::vector<std::string>{"init 0"}));
  // Only a call that names GPU 0 tries it again: each try logs the failure.
  EXPECT_EQ(notifications_.add_event_listeners({{1, RDC_EVNT_NOTIF_VMFAULT}}), RDC_ST_OK);
  EXPECT_EQ(take(), (std::vector<std::string>{"init 1", "mask 1 0x1"}));
  g_smi.init_status[0] = AMDSMI_STATUS_SUCCESS;
  EXPECT_EQ(notifications_.add_event_listeners({{0, RDC_EVNT_NOTIF_PROCESS_START}}), RDC_ST_OK);
  EXPECT_EQ(take(), (std::vector<std::string>{"init 0", "mask 0 0x801"}));
}

TEST_F(Notifications, AnUnavailableGpuGetsItsMaskOnceFound) {
  g_smi.lookup[0] = AMDSMI_STATUS_NOT_FOUND;
  EXPECT_EQ(notifications_.add_event_listeners({{0, RDC_EVNT_NOTIF_VMFAULT}}), RDC_ST_OK);
  EXPECT_EQ(take(), std::vector<std::string>{});
  g_smi.lookup[0] = AMDSMI_STATUS_SUCCESS;
  EXPECT_EQ(notifications_.add_event_listeners({{1, RDC_EVNT_NOTIF_VMFAULT}}), RDC_ST_OK);
  EXPECT_EQ(take(), (std::vector<std::string>{"init 0", "mask 0 0x1", "init 1", "mask 1 0x1"}));
}

TEST_F(Notifications, AFailedGpuThatWasUnavailableIsTriedByAnyCallOnceFound) {
  // An unavailable GPU is not a failure: the last try of GPU 0 found it unavailable.
  g_smi.capability_error = 1;
  EXPECT_EQ(notifications_.add_event_listeners({{0, RDC_EVNT_NOTIF_VMFAULT}}), RDC_ST_PERM_ERROR);
  g_smi.capability_error = 0;
  g_smi.lookup[0] = AMDSMI_STATUS_NOT_FOUND;
  EXPECT_EQ(notifications_.add_event_listeners({{0, RDC_EVNT_NOTIF_PROCESS_START}}), RDC_ST_OK);
  g_smi.lookup[0] = AMDSMI_STATUS_SUCCESS;
  take();
  EXPECT_EQ(notifications_.add_event_listeners({{1, RDC_EVNT_NOTIF_VMFAULT}}), RDC_ST_OK);
  EXPECT_EQ(take(), (std::vector<std::string>{"init 0", "mask 0 0x801", "init 1", "mask 1 0x1"}));
}

TEST_F(Notifications, AGpuUnavailableWhileItHadListenersIsNotStoppedOnceFound) {
  // Its events were never opened: nothing to stop once it is found.
  g_smi.lookup[0] = AMDSMI_STATUS_NOT_FOUND;
  EXPECT_EQ(notifications_.add_event_listeners({{0, RDC_EVNT_NOTIF_VMFAULT}}), RDC_ST_OK);
  EXPECT_EQ(notifications_.remove_event_listeners({{0, RDC_EVNT_NOTIF_VMFAULT}}), RDC_ST_OK);
  g_smi.lookup[0] = AMDSMI_STATUS_SUCCESS;
  EXPECT_EQ(notifications_.add_event_listeners({{1, RDC_EVNT_NOTIF_VMFAULT}}), RDC_ST_OK);
  EXPECT_EQ(take(), (std::vector<std::string>{"init 1", "mask 1 0x1"}));
}

TEST_F(Notifications, AGpuLostAfterItsMaskWasWrittenIsWrittenAgainOnceFound) {
  ASSERT_EQ(notifications_.add_event_listeners({{0, RDC_EVNT_NOTIF_VMFAULT}}), RDC_ST_OK);
  g_smi.lookup[0] = AMDSMI_STATUS_NOT_FOUND;
  // Its listeners change while it is lost, back to the same events.
  EXPECT_EQ(notifications_.add_event_listeners({{0, RDC_EVNT_NOTIF_PROCESS_START}}), RDC_ST_OK);
  EXPECT_EQ(notifications_.remove_event_listeners({{0, RDC_EVNT_NOTIF_PROCESS_START}}), RDC_ST_OK);
  take();
  g_smi.lookup[0] = AMDSMI_STATUS_SUCCESS;
  EXPECT_EQ(notifications_.add_event_listeners({{1, RDC_EVNT_NOTIF_VMFAULT}}), RDC_ST_OK);
  EXPECT_EQ(take(), (std::vector<std::string>{"init 0", "mask 0 0x1", "init 1", "mask 1 0x1"}));
  // And with no listener left once found: written 0 and stopped.
  g_smi.lookup[0] = AMDSMI_STATUS_NOT_FOUND;
  EXPECT_EQ(notifications_.remove_event_listeners({{0, RDC_EVNT_NOTIF_VMFAULT}}), RDC_ST_OK);
  g_smi.lookup[0] = AMDSMI_STATUS_SUCCESS;
  take();
  EXPECT_EQ(notifications_.add_event_listeners({{2, RDC_EVNT_NOTIF_VMFAULT}}), RDC_ST_OK);
  EXPECT_EQ(take(), (std::vector<std::string>{"mask 0 0x0", "stop 0", "init 2", "mask 2 0x1"}));
}

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

// A telemetry module that supports no field and records the sizes of the watch and unwatch
// lists it gets.
class RecordingTelemetry : public amd::rdc::RdcTelemetry {
 public:
  rdc_status_t rdc_telemetry_fields_query(uint32_t[MAX_NUM_FIELDS],
                                          uint32_t* field_count) override {
    *field_count = 0;
    return RDC_ST_OK;
  }
  rdc_status_t rdc_telemetry_fields_value_get(rdc_gpu_field_t*, uint32_t, rdc_field_value_f,
                                              void*) override {
    return RDC_ST_OK;
  }
  rdc_status_t rdc_telemetry_fields_watch(rdc_gpu_field_t*, uint32_t fields_count) override {
    watches.push_back(fields_count);
    return RDC_ST_OK;
  }
  rdc_status_t rdc_telemetry_fields_unwatch(rdc_gpu_field_t*, uint32_t fields_count) override {
    unwatches.push_back(fields_count);
    return RDC_ST_OK;
  }

  std::vector<uint32_t> watches;
  std::vector<uint32_t> unwatches;
};

class TelemetryOnly : public amd::rdc::RdcModuleMgr {
 public:
  amd::rdc::RdcTelemetryPtr get_telemetry_module() override { return telemetry; }
  amd::rdc::RdcDiagnosticPtr get_diagnostic_module() override { return nullptr; }

  std::shared_ptr<RecordingTelemetry> telemetry = std::make_shared<RecordingTelemetry>();
};

TEST(WatchTable, AWatchOfEventFieldsOnlyListensWithoutTelemetryCalls) {
  reset_fake();
  auto groups = std::make_shared<amd::rdc::RdcGroupSettingsImpl>(std::make_shared<OnePartition>());
  auto modules = std::make_shared<TelemetryOnly>();
  // The watch and the unwatch use neither the cache nor the metric fetcher.
  amd::rdc::RdcWatchTableImpl watch_table{groups, nullptr, nullptr, modules,
                                          std::make_shared<RdcNotificationImpl>()};
  rdc_gpu_group_t group_id = 0;
  ASSERT_EQ(groups->rdc_group_gpu_create("events", &group_id), RDC_ST_OK);
  ASSERT_EQ(groups->rdc_group_gpu_add(group_id, 0), RDC_ST_OK);
  ASSERT_EQ(groups->rdc_group_gpu_add(group_id, 1), RDC_ST_OK);
  rdc_field_t vm_fault = RDC_EVNT_NOTIF_VMFAULT;
  rdc_field_grp_t field_group_id = 0;
  ASSERT_EQ(groups->rdc_group_field_create(1, &vm_fault, "vm", &field_group_id), RDC_ST_OK);

  ASSERT_EQ(watch_table.rdc_field_watch(group_id, field_group_id, 1000000, 0, 0), RDC_ST_OK);
  EXPECT_EQ(g_smi.mask[0], bit(AMDSMI_EVT_NOTIF_VMFAULT));
  EXPECT_EQ(g_smi.mask[1], bit(AMDSMI_EVT_NOTIF_VMFAULT));
  ASSERT_EQ(watch_table.rdc_field_unwatch(group_id, field_group_id), RDC_ST_OK);
  EXPECT_EQ(g_smi.mask[0], 0u);
  EXPECT_EQ(g_smi.mask[1], 0u);
  // No telemetry field is left to watch: the lists were empty, and their first entry's address
  // (&v[0]) was undefined behavior.
  EXPECT_EQ(modules->telemetry->watches, std::vector<uint32_t>{});
  EXPECT_EQ(modules->telemetry->unwatches, std::vector<uint32_t>{});
}

// Where a thread under test waits until the test lets it go. Waits end after 10 s on the steady
// clock, so that a broken test fails instead of hanging. (Each wait for the condition variable
// is on the system clock: ThreadSanitizer, as of GCC 11, does not see it unlock its mutex in a
// wait on the steady clock.)
class Gate {
 public:
  // The thread under test: says that it arrived, and waits.
  void pass() {
    std::unique_lock<std::mutex> lock(mutex_);
    arrived_ = true;
    cv_.notify_all();
    wait(&lock, [this] { return open_; });
  }
  // The test: waits until a thread arrived.
  bool arrived() {
    std::unique_lock<std::mutex> lock(mutex_);
    return wait(&lock, [this] { return arrived_; });
  }
  void open() {
    std::lock_guard<std::mutex> lock(mutex_);
    open_ = true;
    cv_.notify_all();
  }

 private:
  bool wait(std::unique_lock<std::mutex>* lock, const std::function<bool()>& done) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (!done() && std::chrono::steady_clock::now() < end) {
      cv_.wait_until(*lock, std::chrono::system_clock::now() + std::chrono::milliseconds(100));
    }
    return done();
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  bool arrived_ = false;
  bool open_ = false;
};

// Group settings whose first look-up of one group's GPUs waits at a gate.
class HoldingGroups : public amd::rdc::RdcGroupSettingsImpl {
 public:
  HoldingGroups() : RdcGroupSettingsImpl(std::make_shared<OnePartition>()) {}
  rdc_status_t rdc_group_gpu_get_info(rdc_gpu_group_t group_id, rdc_group_info_t* info) override {
    if (group_id == held_group && !held.exchange(true)) {
      gate.pass();
    }
    return RdcGroupSettingsImpl::rdc_group_gpu_get_info(group_id, info);
  }

  rdc_gpu_group_t held_group = 0;
  std::atomic<bool> held{false};
  Gate gate;
};

// Counts the listeners of each (GPU, event field) pair. The add of one GPU's listeners, and the
// removal of another GPU's, wait at gates.
class HoldingNotification : public amd::rdc::RdcNotification {
 public:
  bool is_notification_event(rdc_field_t field) const override {
    return RDC_EVNT_IS_NOTIF_FIELD(field);
  }
  rdc_status_t add_event_listeners(const std::vector<RdcFieldKey>& fields) override {
    if (!fields.empty() && fields[0].first == held_add_gpu) {
      add_gate.pass();
    }
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& field : fields) {
      listeners_[field]++;
    }
    return RDC_ST_OK;
  }
  rdc_status_t remove_event_listeners(const std::vector<RdcFieldKey>& fields) override {
    if (!fields.empty() && fields[0].first == held_remove_gpu) {
      remove_gate.pass();
    }
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& field : fields) {
      if (listeners_[field] > 0) {
        listeners_[field]--;
      }
    }
    return RDC_ST_OK;
  }
  rdc_status_t listen(rdc_evnt_notification_t*, uint32_t* num_events, uint32_t) override {
    *num_events = 0;
    return RDC_ST_OK;
  }
  uint32_t listeners(uint32_t gpu, rdc_field_t field) {
    std::lock_guard<std::mutex> lock(mutex_);
    return listeners_[{gpu, field}];
  }

  uint32_t held_add_gpu = 0;
  uint32_t held_remove_gpu = 0;
  Gate add_gate;
  Gate remove_gate;

 private:
  std::mutex mutex_;
  std::map<RdcFieldKey, uint32_t> listeners_;
};

// A cache with nothing in it, in which jobs start and stop.
class EmptyCache : public amd::rdc::RdcCacheManager {
 public:
  rdc_status_t rdc_field_get_latest_value(uint32_t, rdc_field_t, rdc_field_value*) override {
    return RDC_ST_NOT_FOUND;
  }
  rdc_status_t rdc_field_get_value_since(uint32_t, rdc_field_t, uint64_t, uint64_t*,
                                         rdc_field_value*) override {
    return RDC_ST_NOT_FOUND;
  }
  rdc_status_t rdc_update_cache(uint32_t, const rdc_field_value&) override { return RDC_ST_OK; }
  rdc_status_t evict_cache(uint32_t, rdc_field_t, uint64_t, double) override { return RDC_ST_OK; }
  std::string get_cache_stats() override { return ""; }
  rdc_status_t rdc_job_get_stats(const char[64], const rdc_gpu_gauges_t&,
                                 rdc_job_info_t*) override {
    return RDC_ST_NOT_FOUND;
  }
  rdc_status_t rdc_job_start_stats(const char[64], const rdc_group_info_t&,
                                   const rdc_field_group_info_t&,
                                   const rdc_gpu_gauges_t&) override {
    return RDC_ST_OK;
  }
  rdc_status_t rdc_job_stop_stats(const char[64], const rdc_gpu_gauges_t&) override {
    return RDC_ST_OK;
  }
  rdc_status_t rdc_update_job_stats(uint32_t, const std::string&, const rdc_field_value&) override {
    return RDC_ST_OK;
  }
  rdc_status_t rdc_job_remove(const char[64]) override { return RDC_ST_OK; }
  rdc_status_t rdc_job_remove_all() override { return RDC_ST_OK; }
  rdc_status_t rdc_health_set(rdc_gpu_group_t, uint32_t, const rdc_field_value&) override {
    return RDC_ST_OK;
  }
  rdc_status_t rdc_health_get_values(rdc_gpu_group_t, uint32_t, rdc_field_t, uint64_t, uint64_t,
                                     rdc_field_value*, rdc_field_value*) override {
    return RDC_ST_NOT_FOUND;
  }
  rdc_status_t rdc_health_clear(rdc_gpu_group_t) override { return RDC_ST_OK; }
  rdc_status_t rdc_update_health_stats(rdc_gpu_group_t, uint32_t, const rdc_field_value&) override {
    return RDC_ST_OK;
  }
};

TEST(JobListeners, AStartThatRecordsWhileAStopRunsLeavesNoListeners) {
  // Two starts of job j, on GPU 0 and on GPU 1 (nothing keeps two starts of one job apart), and
  // a stop of j, ordered with gates: the start on GPU 1 records its listeners after the stop
  // released those of the start on GPU 0, before the stop erased the job.
  auto groups = std::make_shared<HoldingGroups>();
  auto notifications = std::make_shared<HoldingNotification>();
  // Job starts and stops use no metric fetcher.
  amd::rdc::RdcWatchTableImpl watch_table{groups, std::make_shared<EmptyCache>(), nullptr,
                                          std::make_shared<TelemetryOnly>(), notifications};
  rdc_gpu_group_t gpu0 = 0;
  rdc_gpu_group_t gpu1 = 0;
  ASSERT_EQ(groups->rdc_group_gpu_create("gpu0", &gpu0), RDC_ST_OK);
  ASSERT_EQ(groups->rdc_group_gpu_add(gpu0, 0), RDC_ST_OK);
  ASSERT_EQ(groups->rdc_group_gpu_create("gpu1", &gpu1), RDC_ST_OK);
  ASSERT_EQ(groups->rdc_group_gpu_add(gpu1, 1), RDC_ST_OK);
  groups->held_group = gpu1;
  notifications->held_add_gpu = 1;
  notifications->held_remove_gpu = 0;
  const rdc_gpu_gauges_t gauges;
  rdc_status_t second_start = RDC_ST_UNKNOWN_ERROR;
  rdc_status_t stop = RDC_ST_UNKNOWN_ERROR;
  std::thread starter;
  std::thread stopper;
  // On any failure below, let both threads go and wait for them.
  struct Finally {
    std::function<void()> run;
    ~Finally() { run(); }
  } finally{[&] {
    groups->gate.open();
    notifications->add_gate.open();
    notifications->remove_gate.open();
    if (starter.joinable()) {
      starter.join();
    }
    if (stopper.joinable()) {
      stopper.join();
    }
  }};

  // The start on GPU 1 finds no job j and waits.
  starter = std::thread(
      [&] { second_start = watch_table.rdc_job_start_stats(gpu1, "j", 1000000, gauges); });
  ASSERT_TRUE(groups->gate.arrived());
  // The start on GPU 0 runs, and records its listeners.
  ASSERT_EQ(watch_table.rdc_job_start_stats(gpu0, "j", 1000000, gauges), RDC_ST_OK);
  // The start on GPU 1 adds its listeners, and waits before it records them.
  groups->gate.open();
  ASSERT_TRUE(notifications->add_gate.arrived());
  // The stop releases GPU 0's listeners, and waits before its unwatch.
  stopper = std::thread([&] { stop = watch_table.rdc_job_stop_stats("j", gauges); });
  ASSERT_TRUE(notifications->remove_gate.arrived());
  // The start on GPU 1 records its listeners: job j is still in the table.
  notifications->add_gate.open();
  starter.join();
  notifications->remove_gate.open();
  stopper.join();

  EXPECT_EQ(second_start, RDC_ST_OK);
  EXPECT_EQ(stop, RDC_ST_OK);
  for (rdc_field_t field : {RDC_EVNT_NOTIF_PROCESS_START, RDC_EVNT_NOTIF_PROCESS_END}) {
    EXPECT_EQ(notifications->listeners(0, field), 0u) << field;
    EXPECT_EQ(notifications->listeners(1, field), 0u) << field << ": the stop left them";
  }
}

}  // namespace
