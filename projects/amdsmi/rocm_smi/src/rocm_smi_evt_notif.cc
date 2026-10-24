// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocm_smi/rocm_smi_evt_notif.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <sstream>

#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_logger.h"

namespace amd::smi {

// Each sscanf format below mirrors the matching KFD_EVENT_FMT_* macro in the kernel's
// kfd_ioctl.h; widths and bases (%lx is hex, 64-bit) must match it.
std::string FormatEvtNotifMessage(uint32_t event, const std::string& payload) {
  const char* message = payload.c_str();
  std::stringstream final_message;

  switch (event) {
    case RSMI_EVT_NOTIF_NONE:
      return "Event type None received";
    case RSMI_EVT_NOTIF_VMFAULT: {
      uint32_t pid;
      char task_name[MAX_EVENT_NOTIFICATION_MSG_SIZE];
      memset(task_name, '\0', MAX_EVENT_NOTIFICATION_MSG_SIZE);

      sscanf(message, "%x:%255[^\n]\n", &pid, task_name);
      final_message << "PID: " << std::to_string(pid).c_str() << "  task name: " << task_name;
    } break;
    case RSMI_EVT_NOTIF_THERMAL_THROTTLE: {
      uint64_t bitmask;
      uint64_t counter;

      sscanf(message, "%" PRIx64 ":%" PRIx64 "\n", &bitmask, &counter);
      final_message << "bitmask: 0x" << std::hex << bitmask << "  counter: 0x" << std::hex
                    << counter;
    } break;
    case RSMI_EVT_NOTIF_GPU_PRE_RESET: {
      uint32_t reset_seq_num;
      char reset_cause[MAX_EVENT_NOTIFICATION_MSG_SIZE];
      memset(reset_cause, '\0', MAX_EVENT_NOTIFICATION_MSG_SIZE);

      sscanf(message, "%x %255[^\n]\n", &reset_seq_num, reset_cause);
      final_message << "reset sequence number: " << std::to_string(reset_seq_num).c_str()
                    << "  reset cause: " << reset_cause;
    } break;
    case RSMI_EVT_NOTIF_GPU_POST_RESET: {
      uint32_t reset_seq_num;
      char tmp[MAX_EVENT_NOTIFICATION_MSG_SIZE];

      sscanf(message, "%x %255[^\n]\n", &reset_seq_num, tmp);
      final_message << "reset sequence number: " << std::to_string(reset_seq_num).c_str();
    } break;
    case RSMI_EVT_NOTIF_EVENT_MIGRATE_START: {
      int64_t ns;
      int32_t pid;
      uint64_t start;
      uint64_t size;
      uint32_t from;
      uint32_t to;
      uint32_t prefetch_loc;
      uint32_t preferred_loc;
      int32_t migrate_trigger;

      sscanf(message, "%" PRId64 " -%d @%" PRIx64 "(%" PRIx64 ") %x->%x %x:%x %d\n", &ns, &pid,
             &start, &size, &from, &to, &prefetch_loc, &preferred_loc, &migrate_trigger);
      final_message << "ns: " << std::to_string(ns).c_str()
                    << "  pid: " << std::to_string(pid).c_str() << "  start: 0x" << std::hex
                    << start << "  size: 0x" << std::hex << size << "  from: 0x" << std::hex << from
                    << "  to: 0x" << std::hex << to << "  prefetch_loc: 0x" << std::hex
                    << prefetch_loc << "  preferred_loc: 0x" << std::hex << preferred_loc
                    << "  migrate_trigger: " << std::to_string(migrate_trigger).c_str();
    } break;
    case RSMI_EVT_NOTIF_EVENT_MIGRATE_END: {
      int64_t ns;
      int32_t pid;
      uint64_t start;
      uint64_t size;
      uint32_t from;
      uint32_t to;
      int32_t migrate_trigger;
      int32_t error_code;

      sscanf(message, "%" PRId64 " -%d @%" PRIx64 "(%" PRIx64 ") %x->%x %d %d\n", &ns, &pid, &start,
             &size, &from, &to, &migrate_trigger, &error_code);
      final_message << "ns: " << std::to_string(ns).c_str()
                    << "  pid: " << std::to_string(pid).c_str() << "  start: 0x" << std::hex
                    << start << "  size: 0x" << std::hex << size << "  from: 0x" << std::hex << from
                    << "  to: 0x" << std::hex << to
                    << "  migrate_trigger: " << std::to_string(migrate_trigger).c_str()
                    << "  error_code: " << std::to_string(error_code).c_str();
    } break;
    case RSMI_EVT_NOTIF_EVENT_PAGE_FAULT_START: {
      int64_t ns;
      int32_t pid;
      uint64_t addr;
      uint32_t node;
      char rw = '\0';

      sscanf(message, "%" PRId64 " -%d @%" PRIx64 "(%x) %c\n", &ns, &pid, &addr, &node, &rw);
      final_message << "ns: " << std::to_string(ns).c_str()
                    << "  pid: " << std::to_string(pid).c_str() << "  addr: 0x" << std::hex << addr
                    << "  node: 0x" << std::hex << node << "  rw: " << rw;
    } break;
    case RSMI_EVT_NOTIF_EVENT_PAGE_FAULT_END: {
      int64_t ns;
      int32_t pid;
      uint64_t addr;
      uint32_t node;
      char migrate_update = '\0';

      sscanf(message, "%" PRId64 " -%d @%" PRIx64 "(%x) %c\n", &ns, &pid, &addr, &node,
             &migrate_update);
      final_message << "ns: " << std::to_string(ns).c_str()
                    << "  pid: " << std::to_string(pid).c_str() << "  addr: 0x" << std::hex << addr
                    << "  node: 0x" << std::hex << node << "  migrate_update: " << migrate_update;
    } break;
    case RSMI_EVT_NOTIF_EVENT_QUEUE_EVICTION: {
      int64_t ns;
      int32_t pid;
      uint32_t node;
      int32_t evict_trigger;

      sscanf(message, "%" PRId64 " -%d %x %d\n", &ns, &pid, &node, &evict_trigger);
      final_message << "ns: " << std::to_string(ns).c_str()
                    << "  pid: " << std::to_string(pid).c_str() << "  node: 0x" << std::hex << node
                    << "  evict_trigger: " << std::to_string(evict_trigger).c_str();
    } break;
    case RSMI_EVT_NOTIF_EVENT_QUEUE_RESTORE: {
      int64_t ns;
      int32_t pid;
      uint32_t node;
      char rescheduled = '\0';

      sscanf(message, "%" PRId64 " -%d %x %c\n", &ns, &pid, &node, &rescheduled);
      final_message << "ns: " << std::to_string(ns).c_str()
                    << "  pid: " << std::to_string(pid).c_str() << "  node: 0x" << std::hex << node
                    << "  rescheduled: " << rescheduled;
    } break;
    case RSMI_EVT_NOTIF_EVENT_UNMAP_FROM_GPU: {
      int64_t ns;
      int32_t pid;
      uint64_t addr;
      uint64_t size;
      uint32_t node;
      int32_t unmap_trigger;

      sscanf(message, "%" PRId64 " -%d @%" PRIx64 "(%" PRIx64 ") %x %d\n", &ns, &pid, &addr, &size,
             &node, &unmap_trigger);
      final_message << "ns: " << std::to_string(ns).c_str()
                    << "  pid: " << std::to_string(pid).c_str() << "  addr: 0x" << std::hex << addr
                    << "  size: 0x" << std::hex << size << "  node: 0x" << std::hex << node
                    << "  unmap_trigger: " << std::to_string(unmap_trigger).c_str();
    } break;
    case RSMI_EVT_NOTIF_EVENT_PROCESS_START:
    case RSMI_EVT_NOTIF_EVENT_PROCESS_END: {
      uint32_t pid;
      char task[MAX_EVENT_NOTIFICATION_MSG_SIZE];
      int rc = sscanf(message, "%x %255[^\n]", &pid, task);
      if (rc == 2) {
        final_message << "PID: " << pid << "  task: " << task;
      } else {
        LOG_ERROR("Failed to parse process event payload");
        final_message << "PID:UNKNOWN  task:UNKNOWN";
      }
    } break;
    default:
      return "Unknown event received";
  }

  return final_message.str();
}

}  // namespace amd::smi
