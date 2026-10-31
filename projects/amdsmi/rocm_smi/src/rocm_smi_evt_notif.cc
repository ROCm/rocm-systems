// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocm_smi/rocm_smi_evt_notif.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>

#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_logger.h"

namespace amd::smi {

namespace {

constexpr const char* kNotAvailable = "N/A";
constexpr int kProcessPayloadFieldCount = 2;  // pid and task name

// Counts the values a scanf format stores, which is also the number of sscanf arguments.
constexpr int CountConversions(const char* format) {
  int count = 0;
  for (const char* p = format; *p != '\0'; ++p) {
    if (*p != '%') {
      continue;
    }
    ++p;
    if (*p == '\0') {
      break;
    }
    if (*p != '%' && *p != '*') {  // "%%" is a literal percent, "%*" stores nothing
      ++count;
    }
  }
  return count;
}
static_assert(CountConversions("%x:%255[^\n]\n") == 2, "two conversions");
static_assert(CountConversions("%d%%") == 1, "%% is not a conversion");
static_assert(CountConversions("%*d %d") == 1, "%* is not counted");

// Formats the fields of one KFD event line. sscanf converts fields left to right and returns how
// many it converted, so field n is valid only if n <= that count; later fields read N/A. Add
// fields in the order of sscanf's arguments. Older kernels omit trailing fields: mark those
// optional. Every sscanf conversion needs one Add...() call; BuildAndLogIfShort() reports a
// mismatch. To add or change an event, see the note above FormatEvtNotifMessage().
class EvtMessageBuilder {
 public:
  // `sscanf_result` is sscanf's return value: the fields it converted, or EOF.
  // `format_conversions` is the number of values the sscanf format stores.
  EvtMessageBuilder(int sscanf_result, int format_conversions)
      : fields_converted_(sscanf_result), format_conversions_(format_conversions) {}

  template <typename T>
  EvtMessageBuilder& AddDecimal(const char* label, T value) {
    return AddCheckedField(label, std::to_string(value));
  }

  template <typename T>
  EvtMessageBuilder& AddHex(const char* label, T value) {
    std::ostringstream os;
    os << "0x" << std::hex << value;
    return AddCheckedField(label, os.str());
  }

  EvtMessageBuilder& AddChar(const char* label, char value) {
    return AddCheckedField(label, std::string(1, value));
  }

  // For text the kernel may leave empty, such as a task name: never N/A and never required.
  EvtMessageBuilder& AddText(const char* label, const char* value) {
    ++fields_added_;
    AppendLabelAndValue(label, value);
    return *this;
  }

  // Fields added after this call may be missing without being an error.
  EvtMessageBuilder& MarkRemainingOptional() {
    remaining_optional_ = true;
    return *this;
  }

  // Logs the payload if sscanf converted fewer fields than are required, then returns the text.
  // If the Add...() calls do not match the format, this is a coding error: it is logged and
  // reported as the message instead of printing values under the wrong labels.
  std::string BuildAndLogIfShort(const char* event_name, const std::string& payload) const {
    if (fields_added_ != format_conversions_) {
      std::ostringstream ss;
      ss << "Internal error: " << event_name << " format has " << format_conversions_
         << " conversions but " << fields_added_ << " fields were added";
      const std::string error_text = ss.str();  // LOG_ERROR empties the stream
      LOG_ERROR(ss);
      return error_text;
    }
    if (fields_converted_ < fields_required_) {
      std::ostringstream ss;
      ss << "Unrecognized " << event_name << " payload (" << fields_converted_ << " of "
         << fields_required_ << " fields): " << payload;
      LOG_ERROR(ss);
    }
    return text_;
  }

 private:
  EvtMessageBuilder& AddCheckedField(const char* label, const std::string& value) {
    ++fields_added_;
    if (!remaining_optional_) {
      fields_required_ = fields_added_;
    }
    const bool was_converted = fields_added_ <= fields_converted_;
    AppendLabelAndValue(label, was_converted ? value : std::string(kNotAvailable));
    return *this;
  }

  void AppendLabelAndValue(const char* label, const std::string& value) {
    if (!text_.empty()) {
      text_ += "  ";
    }
    text_ += label;
    text_ += ": ";
    text_ += value;
  }

  const int fields_converted_;
  const int format_conversions_;
  int fields_added_ = 0;
  int fields_required_ = 0;
  bool remaining_optional_ = false;
  std::string text_;
};

}  // namespace

// Runs sscanf with a literal format (so the compiler still checks the argument types) and starts
// a builder that also knows how many values that format stores.
#define EVT_SCAN(message, format, ...) \
  EvtMessageBuilder(sscanf((message), format, __VA_ARGS__), CountConversions(format))

// Each sscanf format below mirrors the matching KFD_EVENT_FMT_* macro in the kernel's
// kfd_ioctl.h; widths and bases (%lx is hex, 64-bit) must match it.
//
// To add or change an event: declare the variable, update the sscanf format and its arguments, and
// add the matching Add...() call in the same position as the argument (the builder numbers fields
// by call order). A field that older kernels do not send goes after MarkRemainingOptional(). Then
// update the event's test in kfd_event_formats_test.cc.
std::string FormatEvtNotifMessage(uint32_t event, const std::string& payload) {
  const char* message = payload.c_str();

  switch (event) {
    case RSMI_EVT_NOTIF_NONE:
      return "Event type None received";
    case RSMI_EVT_NOTIF_VMFAULT: {
      uint32_t pid = 0;
      char task_name[MAX_EVENT_NOTIFICATION_MSG_SIZE];
      memset(task_name, '\0', MAX_EVENT_NOTIFICATION_MSG_SIZE);

      EvtMessageBuilder builder = EVT_SCAN(message, "%x:%255[^\n]\n", &pid, task_name);
      return builder.AddDecimal("PID", pid)
          .AddText("task name", task_name)
          .BuildAndLogIfShort("VMFAULT", payload);
    }
    case RSMI_EVT_NOTIF_THERMAL_THROTTLE: {
      uint64_t bitmask = 0;
      uint64_t counter = 0;

      EvtMessageBuilder builder =
          EVT_SCAN(message, "%" PRIx64 ":%" PRIx64 "\n", &bitmask, &counter);
      return builder.AddHex("bitmask", bitmask)
          .AddHex("counter", counter)
          .BuildAndLogIfShort("THERMAL_THROTTLE", payload);
    }
    case RSMI_EVT_NOTIF_GPU_PRE_RESET: {
      uint32_t reset_seq_num = 0;
      char reset_cause[MAX_EVENT_NOTIFICATION_MSG_SIZE];
      memset(reset_cause, '\0', MAX_EVENT_NOTIFICATION_MSG_SIZE);

      EvtMessageBuilder builder = EVT_SCAN(message, "%x %255[^\n]\n", &reset_seq_num, reset_cause);
      return builder.AddDecimal("reset sequence number", reset_seq_num)
          .AddText("reset cause", reset_cause)
          .BuildAndLogIfShort("GPU_PRE_RESET", payload);
    }
    case RSMI_EVT_NOTIF_GPU_POST_RESET: {
      uint32_t reset_seq_num = 0;

      // The cause that follows the sequence number is not reported for a post-reset.
      EvtMessageBuilder builder = EVT_SCAN(message, "%x %*[^\n]\n", &reset_seq_num);
      return builder.AddDecimal("reset sequence number", reset_seq_num)
          .BuildAndLogIfShort("GPU_POST_RESET", payload);
    }
    case RSMI_EVT_NOTIF_EVENT_MIGRATE_START: {
      int64_t ns = 0;
      int32_t pid = 0;
      uint64_t start = 0;
      uint64_t size = 0;
      uint32_t from = 0;
      uint32_t to = 0;
      uint32_t prefetch_loc = 0;
      uint32_t preferred_loc = 0;
      int32_t migrate_trigger = 0;

      EvtMessageBuilder builder = EVT_SCAN(
          message, "%" PRId64 " -%d @%" PRIx64 "(%" PRIx64 ") %x->%x %x:%x %d\n", &ns, &pid, &start,
          &size, &from, &to, &prefetch_loc, &preferred_loc, &migrate_trigger);
      return builder.AddDecimal("ns", ns)
          .AddDecimal("pid", pid)
          .AddHex("start", start)
          .AddHex("size", size)
          .AddHex("from", from)
          .AddHex("to", to)
          .AddHex("prefetch_loc", prefetch_loc)
          .AddHex("preferred_loc", preferred_loc)
          .AddDecimal("migrate_trigger", migrate_trigger)
          .BuildAndLogIfShort("MIGRATE_START", payload);
    }
    case RSMI_EVT_NOTIF_EVENT_MIGRATE_END: {
      int64_t ns = 0;
      int32_t pid = 0;
      uint64_t start = 0;
      uint64_t size = 0;
      uint32_t from = 0;
      uint32_t to = 0;
      int32_t migrate_trigger = 0;
      int32_t error_code = 0;

      EvtMessageBuilder builder =
          EVT_SCAN(message, "%" PRId64 " -%d @%" PRIx64 "(%" PRIx64 ") %x->%x %d %d\n", &ns, &pid,
                   &start, &size, &from, &to, &migrate_trigger, &error_code);
      return builder.AddDecimal("ns", ns)
          .AddDecimal("pid", pid)
          .AddHex("start", start)
          .AddHex("size", size)
          .AddHex("from", from)
          .AddHex("to", to)
          .AddDecimal("migrate_trigger", migrate_trigger)
          .MarkRemainingOptional()  // kernels before error_code was added do not send it
          .AddDecimal("error_code", error_code)
          .BuildAndLogIfShort("MIGRATE_END", payload);
    }
    case RSMI_EVT_NOTIF_EVENT_PAGE_FAULT_START: {
      int64_t ns = 0;
      int32_t pid = 0;
      uint64_t addr = 0;
      uint32_t node = 0;
      char rw = '\0';

      EvtMessageBuilder builder =
          EVT_SCAN(message, "%" PRId64 " -%d @%" PRIx64 "(%x) %c\n", &ns, &pid, &addr, &node, &rw);
      return builder.AddDecimal("ns", ns)
          .AddDecimal("pid", pid)
          .AddHex("addr", addr)
          .AddHex("node", node)
          .AddChar("rw", rw)
          .BuildAndLogIfShort("PAGE_FAULT_START", payload);
    }
    case RSMI_EVT_NOTIF_EVENT_PAGE_FAULT_END: {
      int64_t ns = 0;
      int32_t pid = 0;
      uint64_t addr = 0;
      uint32_t node = 0;
      char migrate_update = '\0';

      EvtMessageBuilder builder = EVT_SCAN(message, "%" PRId64 " -%d @%" PRIx64 "(%x) %c\n", &ns,
                                           &pid, &addr, &node, &migrate_update);
      return builder.AddDecimal("ns", ns)
          .AddDecimal("pid", pid)
          .AddHex("addr", addr)
          .AddHex("node", node)
          .AddChar("migrate_update", migrate_update)
          .BuildAndLogIfShort("PAGE_FAULT_END", payload);
    }
    case RSMI_EVT_NOTIF_EVENT_QUEUE_EVICTION: {
      int64_t ns = 0;
      int32_t pid = 0;
      uint32_t node = 0;
      int32_t evict_trigger = 0;

      EvtMessageBuilder builder =
          EVT_SCAN(message, "%" PRId64 " -%d %x %d\n", &ns, &pid, &node, &evict_trigger);
      return builder.AddDecimal("ns", ns)
          .AddDecimal("pid", pid)
          .AddHex("node", node)
          .AddDecimal("evict_trigger", evict_trigger)
          .BuildAndLogIfShort("QUEUE_EVICTION", payload);
    }
    case RSMI_EVT_NOTIF_EVENT_QUEUE_RESTORE: {
      int64_t ns = 0;
      int32_t pid = 0;
      uint32_t node = 0;
      char rescheduled = '\0';

      EvtMessageBuilder builder =
          EVT_SCAN(message, "%" PRId64 " -%d %x %c\n", &ns, &pid, &node, &rescheduled);
      return builder.AddDecimal("ns", ns)
          .AddDecimal("pid", pid)
          .AddHex("node", node)
          .MarkRemainingOptional()  // kernels before the rescheduled flag do not send it
          .AddChar("rescheduled", rescheduled)
          .BuildAndLogIfShort("QUEUE_RESTORE", payload);
    }
    case RSMI_EVT_NOTIF_EVENT_UNMAP_FROM_GPU: {
      int64_t ns = 0;
      int32_t pid = 0;
      uint64_t addr = 0;
      uint64_t size = 0;
      uint32_t node = 0;
      int32_t unmap_trigger = 0;

      EvtMessageBuilder builder =
          EVT_SCAN(message, "%" PRId64 " -%d @%" PRIx64 "(%" PRIx64 ") %x %d\n", &ns, &pid, &addr,
                   &size, &node, &unmap_trigger);
      return builder.AddDecimal("ns", ns)
          .AddDecimal("pid", pid)
          .AddHex("addr", addr)
          .AddHex("size", size)
          .AddHex("node", node)
          .AddDecimal("unmap_trigger", unmap_trigger)
          .BuildAndLogIfShort("UNMAP_FROM_GPU", payload);
    }
    case RSMI_EVT_NOTIF_EVENT_PROCESS_START:
    case RSMI_EVT_NOTIF_EVENT_PROCESS_END: {
      uint32_t pid = 0;
      char task[MAX_EVENT_NOTIFICATION_MSG_SIZE];
      memset(task, '\0', MAX_EVENT_NOTIFICATION_MSG_SIZE);

      if (sscanf(message, "%x %255[^\n]", &pid, task) != kProcessPayloadFieldCount) {
        LOG_ERROR("Failed to parse process event payload");
        return "PID:UNKNOWN  task:UNKNOWN";
      }
      return "PID: " + std::to_string(pid) + "  task: " + task;
    }
    default:
      return "Unknown event received";
  }
}

#undef EVT_SCAN

}  // namespace amd::smi
