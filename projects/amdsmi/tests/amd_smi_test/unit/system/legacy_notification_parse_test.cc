// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstring>
#include <sstream>
#include <string>

#include "../../../../rocm_smi/include/rocm_smi/rocm_smi.h"

namespace amd::smi {
extern "C" bool amdsmi_test_parse_legacy_notification(const std::string& line,
                                                      rsmi_evt_notification_data_t* data_item);
extern "C" void amdsmi_test_copy_legacy_string(const std::string& source, char* destination,
                                               size_t capacity);
}  // namespace amd::smi

TEST(SystemUnit, LegacyNotificationRejectsTruncatedPayloads) {
  rsmi_evt_notification_data_t notification{};
  EXPECT_FALSE(amd::smi::amdsmi_test_parse_legacy_notification("1 123:", &notification));
  EXPECT_FALSE(amd::smi::amdsmi_test_parse_legacy_notification("2 ff:", &notification));
  EXPECT_FALSE(amd::smi::amdsmi_test_parse_legacy_notification("5 123 -4 @", &notification));
}

TEST(SystemUnit, LegacyNotificationRejectsInvalidNumericFields) {
  rsmi_evt_notification_data_t notification{};
  for (const char* line :
       {"", "nothex", "0junk", "ffjunk", "1 100000000:task", "1 -1:task", "2 10000000000000000:1",
        "2 -1:1", "2 ff:1junk", "5 9223372036854775808 -4 @10(20) a->b c:d 1"}) {
    EXPECT_FALSE(amd::smi::amdsmi_test_parse_legacy_notification(line, &notification)) << line;
  }
}

TEST(SystemUnit, LegacyNotificationKeepsDestinationOnFailure) {
  rsmi_evt_notification_data_t notification;
  std::memset(&notification, 'X', sizeof(notification));
  const std::string before(reinterpret_cast<const char*>(&notification), sizeof(notification));
  EXPECT_FALSE(amd::smi::amdsmi_test_parse_legacy_notification("2 ff:", &notification));
  EXPECT_EQ(std::string(reinterpret_cast<const char*>(&notification), sizeof(notification)),
            before);
}

TEST(SystemUnit, LegacyNotificationPreservesEventFormats) {
  struct Example {
    rsmi_evt_notification_type_t event;
    const char* payload;
    const char* expected;
  };
  const Example examples[] = {
      {RSMI_EVT_NOTIF_NONE, "", "Event type None received"},
      {RSMI_EVT_NOTIF_VMFAULT, "ff:worker", "PID: 255  task name: worker"},
      {RSMI_EVT_NOTIF_THERMAL_THROTTLE, "ffffffffffffffff:a",
       "bitmask: 0xffffffffffffffff  counter: 0xa"},
      {RSMI_EVT_NOTIF_GPU_PRE_RESET, "a reset cause",
       "reset sequence number: 10  reset cause: reset cause"},
      {RSMI_EVT_NOTIF_GPU_POST_RESET, "a", "reset sequence number: 10"},
      {RSMI_EVT_NOTIF_EVENT_MIGRATE_START, "123 -4 @10(20) a->b c:d -1",
       "nd: 123  pid: 4  start: 0xa  size: 0x14  from: 0xa  to: 0xb  prefetch_loc: 0xc  "
       "preferred_loc: 0xd  migrate_trigger: -1"},
      {RSMI_EVT_NOTIF_EVENT_MIGRATE_END, "123 -4 @10(20) a->b 2 3",
       "nd: 123  pid: 4  start: 0xa  size: 0x14  from: 0xa  to: 0xb  migrate_trigger: 2  "
       "error_code: 3"},
      {RSMI_EVT_NOTIF_EVENT_PAGE_FAULT_START, "123 -4 @ab(c) R",
       "ns: 123  pid: 4  addr: 0xab  node: 0xc  rw: R"},
      {RSMI_EVT_NOTIF_EVENT_PAGE_FAULT_END, "123 -4 @ab(c) M",
       "ns: 123  pid: 4  addr: 0xab  node: 0xc  migrate_udpate: M"},
      {RSMI_EVT_NOTIF_EVENT_QUEUE_EVICTION, "123-4 c 2",
       "ns: 123  pid: 4  node: 0xc  evict_trigger: 2"},
      {RSMI_EVT_NOTIF_EVENT_QUEUE_RESTORE, "123-4 c R",
       "ns: 123  pid: 4  node: 0xc  rescheduled: R"},
      {RSMI_EVT_NOTIF_EVENT_UNMAP_FROM_GPU, "123 -4 @ab(cd) e 2",
       "ns: 123  pid: 4  addr: 0xab  size: 0xcd  node: 0xe  unmap_trigger: 2"},
      {RSMI_EVT_NOTIF_EVENT_PROCESS_START, "ff worker", "PID: 255  task: worker"},
      {RSMI_EVT_NOTIF_EVENT_PROCESS_END, "invalid", "PID:UNKNOWN  task:UNKNOWN"},
  };
  for (const auto& example : examples) {
    rsmi_evt_notification_data_t notification{};
    std::ostringstream line;
    line << std::hex << example.event << ' ' << example.payload << '\n';
    ASSERT_TRUE(amd::smi::amdsmi_test_parse_legacy_notification(line.str(), &notification))
        << line.str();
    EXPECT_EQ(notification.event, example.event);
    EXPECT_STREQ(notification.message, example.expected);
  }
}

TEST(SystemUnit, LegacyNotificationTerminatesReusedMessageBuffer) {
  rsmi_evt_notification_data_t notification;
  std::memset(&notification, 'X', sizeof(notification));
  std::ostringstream line;
  line << std::hex << RSMI_EVT_NOTIF_GPU_PRE_RESET << " 1 " << std::string(512, 'a');
  ASSERT_TRUE(amd::smi::amdsmi_test_parse_legacy_notification(line.str(), &notification));
  EXPECT_EQ(notification.message[sizeof(notification.message) - 1], '\0');
  EXPECT_EQ(std::strlen(notification.message), sizeof(notification.message) - 1);
}

TEST(SystemUnit, LegacyNotificationTerminatesAndPadsBoundedStrings) {
  char destination[5] = {'X', 'X', 'X', 'X', '!'};
  amd::smi::amdsmi_test_copy_legacy_string("abcdef", destination, 4);
  EXPECT_EQ(std::string(destination, 4), std::string("abc\0", 4));
  EXPECT_EQ(destination[4], '!');
  amd::smi::amdsmi_test_copy_legacy_string("a", destination, 4);
  EXPECT_EQ(destination[2], '\0');
  EXPECT_EQ(destination[3], '\0');
  amd::smi::amdsmi_test_copy_legacy_string(std::string("a\0b", 3), destination, 4);
  EXPECT_EQ(destination[2], '\0');
  amd::smi::amdsmi_test_copy_legacy_string("abc", destination, 1);
  EXPECT_EQ(destination[0], '\0');
  destination[0] = '!';
  amd::smi::amdsmi_test_copy_legacy_string("abc", destination, 0);
  EXPECT_EQ(destination[0], '!');
}