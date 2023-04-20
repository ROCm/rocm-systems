/* Copyright (c) 2022 Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "rocmtool.h"

#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <optional>
#include <thread>

#include "src/core/hardware/hsa_info.h"
#include "src/core/hsa/hsa_support.h"
#include "src/core/hsa/queues/queue.h"
#include "src/utils/helper.h"
#include "src/utils/logger.h"
#include "src/core/memory/generic_buffer.h"

#define ASSERTM(exp, msg) assert(((void)msg, exp))

extern std::mutex sessions_pending_signal_lock;

static inline uint32_t GetTid() { return syscall(__NR_gettid); }

namespace rocmtools {

rocmtool* rocmtool_obj;

// Constructor of rocmtool
// Takes the buffer size, a buffer callback function and a buffer flush
// interval to allocate a buffer pool using GenericStorage Also takes the
// replay mode (application replay/kernel replay/user replay) to set the replay
// mode for the rocmtool class object
rocmtool::rocmtool() : current_session_id_(rocprofiler_session_id_t{0}) {}

// Destructor of rocmtool
// deletes the buffer pool
// Iterates over its session map and resets each session in its internal
// session map and clears them from the map. Pops labels from the range stack
// and deletes the stack.
rocmtool::~rocmtool() {
  // {
  //   std::lock_guard<std::mutex> lock(session_map_lock_);
  //   if (!sessions_.empty()) {
  //     // TODO(aelwazir): throw an exception user need to destroy all created
  //     // session (document)
  //     // fatal("Error: Sessions are not destroyed yet!");
  //     sessions_.clear();
  //   }
  // }
  Counter::ClearBasicCounters();
}

bool rocmtool::FindAgent(rocprofiler_agent_id_t agent_id) { return true; }
size_t rocmtool::GetAgentInfoSize(rocprofiler_agent_info_kind_t kind, rocprofiler_agent_id_t agent_id) {
  return 0;
}
const char* rocmtool::GetAgentInfo(rocprofiler_agent_info_kind_t kind,
                                   rocprofiler_agent_id_t agent_id) {
  return "";
}

// TODO(aelwazir): Implement Queue Query
bool rocmtool::FindQueue(rocprofiler_queue_id_t queue_id) { return true; }
size_t rocmtool::GetQueueInfoSize(rocprofiler_queue_info_kind_t kind, rocprofiler_queue_id_t queue_id) {
  return 0;
}
const char* rocmtool::GetQueueInfo(rocprofiler_queue_info_kind_t kind,
                                   rocprofiler_queue_id_t queue_id) {
  return "";
}

bool rocmtool::FindSession(rocprofiler_session_id_t session_id) {
  std::lock_guard<std::mutex> lock(session_map_lock_);
  return sessions_.find(session_id.handle) != sessions_.end();
}

rocprofiler_session_id_t rocmtool::CreateSession(rocprofiler_replay_mode_t replay_mode) {
  rocprofiler_session_id_t session_id = rocprofiler_session_id_t{GenerateUniqueSessionId()};
  {
    std::lock_guard<std::mutex> lock(session_map_lock_);
    sessions_.emplace(session_id.handle, new Session(replay_mode, session_id));
  }
  return session_id;
}

void rocmtool::DestroySession(rocprofiler_session_id_t session_id) {
  while (GetCurrentActiveInterruptSignalsCount() != 0) {
  }

  // if (GetSession(session_id)->GetTracer()) {
  //   GetSession(session_id)->GetTracer().reset();
  //   GetSession(session_id)
  //       ->GetBuffer(
  //           GetSession(session_id)
  //               ->GetFilter(GetSession(session_id)->GetFilterIdWithKind(ROCPROFILER_API_TRACE))
  //               .GetBufferId())
  //       .reset();
  // }

  {
    std::lock_guard<std::mutex> lock(session_map_lock_);
    ASSERTM(sessions_.find(session_id.handle) != sessions_.end(),
            "Error: Couldn't find a created session with given id");
    delete sessions_.at(session_id.handle);
    sessions_.erase(session_id.handle);
  }
}

bool rocmtool::FindDeviceProfilingSession(rocprofiler_session_id_t session_id) {
  std::lock_guard<std::mutex> lock(device_profiling_session_map_lock_);
  return dev_profiling_sessions_.find(session_id.handle) != dev_profiling_sessions_.end();
}

rocprofiler_session_id_t rocmtool::CreateDeviceProfilingSession(std::vector<std::string> counters,
                                                              int cpu_agent_index,
                                                              int gpu_agent_index) {
  rocprofiler_session_id_t session_id;
  {
    std::lock_guard<std::mutex> lock(device_profiling_session_map_lock_);

    hsa_agent_t cpu_agent;
    hsa_agent_t gpu_agent;
    find_hsa_agent_cpu(cpu_agent_index, &cpu_agent);
    find_hsa_agent_gpu(gpu_agent_index, &gpu_agent);

    dev_profiling_sessions_.emplace(
        session_id.handle,
        new DeviceProfileSession(counters, cpu_agent, gpu_agent, &session_id.handle));
  }
  return session_id;
}

void rocmtool::DestroyDeviceProfilingSession(rocprofiler_session_id_t session_id) {
  {
    std::lock_guard<std::mutex> lock(device_profiling_session_map_lock_);
    ASSERTM(dev_profiling_sessions_.find(session_id.handle) != dev_profiling_sessions_.end(),
            "Error: Couldn't find a created session with given id");
    delete dev_profiling_sessions_.at(session_id.handle);
    dev_profiling_sessions_.erase(session_id.handle);
  }
}

DeviceProfileSession* rocmtool::GetDeviceProfilingSession(rocprofiler_session_id_t session_id) {
  std::lock_guard<std::mutex> lock(device_profiling_session_map_lock_);
  assert(dev_profiling_sessions_.find(session_id.handle) != dev_profiling_sessions_.end() &&
         "Error: Can't find the session!");
  return dev_profiling_sessions_.at(session_id.handle);
}

bool rocmtool::HasActiveSession() { return GetCurrentSessionId().handle > 0; }
bool rocmtool::IsActiveSession(rocprofiler_session_id_t session_id) {
  return (GetCurrentSessionId().handle == session_id.handle);
}

// Get the session by its id
// Looks up the session object for an input session id in the internal map.
// If a given session id doesn't exist, it throws an assertion.
// If a session object exists for the given session id, the session object is
// returned.
Session* rocmtool::GetSession(rocprofiler_session_id_t session_id) {
  std::lock_guard<std::mutex> lock(session_map_lock_);
  assert(sessions_.find(session_id.handle) != sessions_.end() && "Error: Can't find the session!");
  return sessions_.at(session_id.handle);
}

// Get Current Session ID
rocprofiler_session_id_t rocmtool::GetCurrentSessionId() { return current_session_id_; }

void rocmtool::SetCurrentActiveSession(rocprofiler_session_id_t session_id) {
  current_session_id_ = session_id;
}

uint64_t rocmtool::GetUniqueRecordId() { return records_counter_.fetch_add(1); }

uint64_t rocmtool::GetUniqueKernelDispatchId() {
  return kernel_dispatch_counter_.fetch_add(1, std::memory_order_relaxed);
}

size_t rocmtool::GetKernelInfoSize(rocprofiler_kernel_info_kind_t kind,
                                   rocprofiler_kernel_id_t kernel_id) {
  switch (kind) {
    case ROCPROFILER_KERNEL_NAME:
      return GetKernelNameUsingDispatchID(kernel_id.handle).size();
    default:
      warning("The provided Kernel Kind is not yet supported!");
      return 0;
  }
}
const char* rocmtool::GetKernelInfo(rocprofiler_kernel_info_kind_t kind,
                                    rocprofiler_kernel_id_t kernel_id) {
  switch (kind) {
    case ROCPROFILER_KERNEL_NAME:
      return strdup(GetKernelNameUsingDispatchID(kernel_id.handle).c_str());
    default:
      warning("The provided Kernel Kind is not yet supported!");
      return "";
  }
}

// TODO(aelwazir): To be implemented
bool rocmtool::CheckFilterData(rocprofiler_filter_kind_t filter_kind,
                               rocprofiler_filter_data_t filter_data) {
  return true;
}

// End of ROCMTool Class

rocmtool* GetROCMToolObj() { return rocmtool_obj; }

void InitROCMToolObj() { rocmtool_obj = new rocmtool; }
void ResetROCMToolObj() {
  delete rocmtool_obj;
  // if (rocmtool_obj) rocmtool_obj.reset();
}

rocprofiler_timestamp_t GetCurrentTimestamp() { return hsa_support::GetCurrentTimestampNS(); }

rocprofiler_status_t IterateCounters(rocprofiler_counters_info_callback_t counters_info_callback) {
  if (hsa_support::IterateCounters(counters_info_callback)) return ROCPROFILER_STATUS_SUCCESS;
  return ROCPROFILER_STATUS_ERROR;
}

}  // namespace rocmtools
