////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2022-2022, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#include "core/inc/svm_profiler.h"

#include <stdint.h>
#include <algorithm>
#include <sys/eventfd.h>
#include <poll.h>

#include "hsakmt/hsakmt.h"

#include "core/util/utils.h"
#include "core/inc/runtime.h"
#include "core/inc/agent.h"
#include "core/inc/amd_gpu_agent.h"
#include "core/util/os.h"

namespace rocr {
namespace AMD {

static const char* smi_event_string(uint32_t event) {
  static const char* strings[] = {"NONE",
                                  "VMFAULT",
                                  "THERMAL_THROTTLE",
                                  "GPU_PRE_RESET",
                                  "GPU_POST_RESET",
                                  "MIGRATE_START",
                                  "MIGRATE_END",
                                  "PAGE_FAULT_START",
                                  "PAGE_FAULT_END",
                                  "QUEUE_EVICTION",
                                  "QUEUE_RESTORE",
                                  "UNMAP_FROM_GPU",
                                  "UNKNOWN"};

  event = std::min<uint32_t>(event, sizeof(strings) / sizeof(char*) - 1);
  return strings[event];
}

static const char* smi_migrate_string(uint32_t trigger) {
  static const char* strings[] = {"PREFETCH",
                                  "PAGEFAULT_GPU",
                                  "PAGEFAULT_CPU",
                                  "TTM_EVICTION",
                                  "UNKNOWN"};

  trigger = std::min<uint32_t>(trigger, sizeof(strings) / sizeof(char*) - 1);
  return strings[trigger];
}

static const char* smi_eviction_string(uint32_t trigger) {
  static const char* strings[] = {"SVM",
                                  "USERPTR",
                                  "TTM",
                                  "SUSPEND",
                                  "CRIU_CHECKPOINT",
                                  "CRIU_RESTORE",
                                  "UNKNOWN"};

  trigger = std::min<uint32_t>(trigger, sizeof(strings) / sizeof(char*) - 1);
  return strings[trigger];
}

static const char* smi_unmap_string(uint32_t trigger) {
  static const char* strings[] = {"MMU_NOTIFY",
                                  "MMU_NOTIFY_MIGRATE",
                                  "UNMAP_FROM_CPU",
                                  "UNKNOWN"};

  trigger = std::min<uint32_t>(trigger, sizeof(strings) / sizeof(char*) - 1);
  return strings[trigger];
}

void SvmProfileControl::PollSmiRun(void* _profileControl) {
  SvmProfileControl* profileControl = (SvmProfileControl*)_profileControl;

  profileControl->PollSmi();
}

void SvmProfileControl::PollSmi() {
  if (core::Runtime::runtime_singleton_->flag().svm_profile().empty()) {
    return;
  }
  FILE* logFile = fopen(core::Runtime::runtime_singleton_->flag().svm_profile().c_str(), "a");
  if (logFile == NULL) {
    return;
  }
  MAKE_NAMED_SCOPE_GUARD(logGuard, [&]() { fclose(logFile); });

  std::vector<pollfd> files;
  files.resize(core::Runtime::runtime_singleton_->gpu_agents().size() + 1);
  files[0].fd = event;
  files[0].events = POLLIN;
  files[0].revents = 0;

  HSAuint64 events = 0;
  events = HSA_SMI_EVENT_MASK_FROM_INDEX(HSA_SMI_EVENT_MIGRATE_START) |
      HSA_SMI_EVENT_MASK_FROM_INDEX(HSA_SMI_EVENT_MIGRATE_END) |
      HSA_SMI_EVENT_MASK_FROM_INDEX(HSA_SMI_EVENT_PAGE_FAULT_START) |
      HSA_SMI_EVENT_MASK_FROM_INDEX(HSA_SMI_EVENT_PAGE_FAULT_END) |
      HSA_SMI_EVENT_MASK_FROM_INDEX(HSA_SMI_EVENT_QUEUE_EVICTION) |
      HSA_SMI_EVENT_MASK_FROM_INDEX(HSA_SMI_EVENT_QUEUE_RESTORE) |
      HSA_SMI_EVENT_MASK_FROM_INDEX(HSA_SMI_EVENT_UNMAP_FROM_GPU);

  for (int i = 0; i < core::Runtime::runtime_singleton_->gpu_agents().size(); i++) {
    auto err = HSAKMT_CALL(hsaKmtOpenSMI(core::Runtime::runtime_singleton_->gpu_agents()[i]->node_id(),
                             &files[i + 1].fd));
    assert(err == HSAKMT_STATUS_SUCCESS);
    files[i + 1].events = POLLIN;
    files[i + 1].revents = 0;
    // Enable collecting masked events.
    auto wrote = write(files[i + 1].fd, &events, sizeof(events));
    assert(wrote == sizeof(events));
  }
  MAKE_NAMED_SCOPE_GUARD(smiGuard, [&]() {
    for (int i = 1; i < files.size(); i++) {
      close(files[i].fd);
    }
  });

  std::vector<std::string> smi_records;
  smi_records.resize(core::Runtime::runtime_singleton_->gpu_agents().size() + 1);
  char buffer[HSA_SMI_EVENT_MSG_SIZE + 1];

  auto format_agent = [this](uint32_t gpuid) {
    std::string ret;
    core::Agent* agent = core::Runtime::runtime_singleton_->agent_by_gpuid(gpuid);
    if (agent->device_type() == core::Agent::kAmdCpuDevice)
      return std::string("CPU");
    else
      return format("GPU%u(%p)", ((AMD::GpuAgent*)agent)->enumeration_index(),
                    agent->public_handle());
  };

  while (!exit) {
    int ready = poll(&files[0], files.size(), -1);
    if (ready < 1) {
      assert(false && "poll failed!");
      return;
    }

    for (int i = 1; i < files.size(); i++) {
      if (files[i].revents & POLLIN) {
        memset(buffer, 0, sizeof(buffer));
        auto len = read(files[i].fd, buffer, sizeof(buffer) - 1);
        if (len > 0) {
          buffer[len] = '\0';
          // printf("%s\n", buffer);
          // fprintf(logFile, "%s\n", buffer);

          smi_records[i] += buffer;

          while (true) {
            size_t pos = smi_records[i].find('\n');
            if (pos == std::string::npos) break;

            std::string line = smi_records[i].substr(0, pos);
            smi_records[i].erase(0, pos + 1);

            const char* cursor;
            cursor = line.c_str();

            // Event records follow the format:
            // event_id timestamp -pid event_specific_info trigger
            // timestamp, pid, and trigger are in dec.  All other are hex.
            // event_specific substring is listed for each event type.
            // See kfd_ioctl.h for more info.
            int event_id;
            uint64_t time;
            int pid;
            int offset = 0;
            int args = sscanf(cursor, "%x %lu -%u%n", &event_id, &time, &pid, &offset);
            assert(args == 3 && "Parsing error!");

            std::string detail;
            cursor += offset + 1;
            switch (event_id) {
              //@addr(size) from->to prefetch_location:preferred_location
              case HSA_SMI_EVENT_MIGRATE_START: {
                uint64_t addr;
                uint32_t size;
                uint32_t from, to;
                uint32_t trigger = 0;
                uint32_t fetch, pref;
                args = sscanf(cursor, "@%lx(%x) %x->%x %x:%x %u", &addr, &size, &from, &to, &fetch,
                              &pref, &trigger);
                assert(args == 7 && "Parsing error!");

                addr *= 4096;
                size *= 4096;

                std::string from_agent = format_agent(from);
                std::string to_agent = format_agent(to);
                std::string range = format("[%p, %p]", addr, addr + size - 1);
                std::string cause = smi_migrate_string(trigger);
                detail = cause + " " + from_agent + "->" + to_agent + " " + range;
                break;
              }
              //@addr(size) from->to
              case HSA_SMI_EVENT_MIGRATE_END: {
                uint64_t addr;
                uint32_t size;
                uint32_t from, to;
                uint32_t trigger;
                args = sscanf(cursor, "@%lx(%x) %x->%x %u", &addr, &size, &from, &to, &trigger);
                assert(args == 5 && "Parsing error!");

                addr *= 4096;
                size *= 4096;

                std::string from_agent = format_agent(from);
                std::string to_agent = format_agent(to);
                std::string range = format("[%p, %p]", addr, addr + size - 1);
                std::string cause = smi_migrate_string(trigger);
                detail = cause + " " + from_agent + "->" + to_agent + " " + range;
                break;
              }
              //@addr(gpu_id) W/R
              case HSA_SMI_EVENT_PAGE_FAULT_START: {
                uint64_t addr;
                uint32_t gpuid;
                char mode;
                args = sscanf(cursor, "@%lx(%x) %c", &addr, &gpuid, &mode);

                addr *= 4096;

                assert(args == 3 && "Parsing error!");
                std::string agent = format_agent(gpuid);
                std::string range = std::to_string(addr);
                std::string cause = (mode == 'W') ? "Write" : "Read";
                detail = cause + " " + agent + " " + range;
                break;
              }
              //@addr(gpu_id) M/U  (migration / page table update)
              case HSA_SMI_EVENT_PAGE_FAULT_END: {
                uint64_t addr;
                uint32_t gpuid;
                char mode;
                args = sscanf(cursor, "@%lx(%x) %c", &addr, &gpuid, &mode);
                assert(args == 3 && "Parsing error!");

                addr *= 4096;

                std::string agent = format_agent(gpuid);
                std::string range = std::to_string(addr);
                std::string cause = (mode == 'M') ? "Migration" : "Map";
                detail = cause + " " + agent + " " + range;
                break;
              }
              // gpu_id
              case HSA_SMI_EVENT_QUEUE_EVICTION: {
                uint32_t gpuid;
                uint32_t trigger;
                args = sscanf(cursor, "%x %u", &gpuid, &trigger);
                assert(args == 2 && "Parsing error!");
                std::string agent = format_agent(gpuid);
                std::string cause = smi_eviction_string(trigger);
                detail = cause + " " + agent;
                break;
              }
              // gpu_id
              case HSA_SMI_EVENT_QUEUE_RESTORE: {
                uint32_t gpuid;
                args = sscanf(cursor, "%x", &gpuid);
                assert(args == 1 && "Parsing error!");
                std::string agent = format_agent(gpuid);
                detail = agent;
                break;
              }
              //@addr(size) gpu_id
              case HSA_SMI_EVENT_UNMAP_FROM_GPU: {
                uint64_t addr;
                uint32_t size;
                uint32_t gpuid;
                uint32_t trigger;
                args = sscanf(cursor, "@%lx(%x) %x %u", &addr, &size, &gpuid, &trigger);
                assert(args == 4 && "Parsing error!");

                addr *= 4096;
                size *= 4096;

                std::string gpu = format_agent(gpuid);
                std::string range = format("[%p, %p]", addr, addr + size - 1);
                std::string cause = smi_unmap_string(trigger);
                detail = cause + " " + gpu + " " + range;
                break;
              }
              default:;
            }

            std::string record = std::string("ROCr HMM event: ") + std::to_string(time) + " " +
                smi_event_string(event_id) + " " + detail;
            // printf("%s\n", record.c_str());
            fprintf(logFile, "%s\n", record.c_str());
          }
        } else {
          auto err = errno;
          const char* msg = strerror(err);
          // printf("ROCr HMM event error: Read returned %ld, %s (%d)\n", len, msg, err);
          fprintf(logFile, "ROCr HMM event error: Read returned %ld, %s (%d)\n", len, msg, err);
        }
        files[i].revents = 0;
      }
    }
    if (files[0].revents & POLLIN) return;
  }
}

SvmProfileControl::SvmProfileControl() : event(-1), exit(false) {
  event = eventfd(0, EFD_CLOEXEC);
  if (event == -1) return;

  poll_smi_thread_ = os::CreateThread(PollSmiRun, (void*)this);
  if (poll_smi_thread_ == NULL) {
    assert(false && "Poll SMI thread creation error.");
    return;
  }
}

SvmProfileControl::~SvmProfileControl() {
  if (event != -1) eventfd_write(event, 1);
  if (poll_smi_thread_ != NULL) {
    exit = true;
    os::WaitForThread(poll_smi_thread_);
    os::CloseThread(poll_smi_thread_);
    poll_smi_thread_ = NULL;
  }
  close(event);
}

template <typename... Args>
std::string SvmProfileControl::format(const char* format, Args... args) {
  int len = snprintf(&format_buffer[0], format_buffer.size(), format, args...);
  if (len + 1 > format_buffer.size()) {
    format_buffer.resize(len + 1);
    snprintf(&format_buffer[0], format_buffer.size(), format, args...);
  }
  return std::string(&format_buffer[0]);
}
 
} // namespace AMD
} // namespace rocr
