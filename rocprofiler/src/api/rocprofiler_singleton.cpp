
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

#include "rocprofiler_singleton.h"

#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <cassert>
#include <optional>
#include <thread>

#include "src/core/hardware/hsa_info.h"
#include "src/core/hsa/hsa_support.h"
#include "src/core/hsa/queues/queue.h"
#include "src/utils/helper.h"
#include "src/utils/logger.h"
#include "src/core/memory/generic_buffer.h"
#include "src/utils/filesystem.hpp"

namespace fs = rocprofiler::common::filesystem;
#define ASSERTM(exp, msg) assert(((void)msg, exp))

extern std::mutex sessions_pending_signal_lock;

static inline uint32_t GetTid() { return syscall(__NR_gettid); }

namespace rocprofiler {



// Constructor of ROCProfiler
// Takes the buffer size, a buffer callback function and a buffer flush
// interval to allocate a buffer pool using GenericStorage Also takes the
// replay mode (application replay/kernel replay/user replay) to set the replay
// mode for the rocprofiler class object
ROCProfiler_Singleton::ROCProfiler_Singleton() {
  fs::path sysfs_nodes_path = "/sys/class/kfd/kfd/topology/nodes";
  fs::directory_entry dirp("/sys/class/kfd/kfd/topology/nodes");
  if (!fs::exists(sysfs_nodes_path))
    rocprofiler::fatal("Could not opendir `%s'", sysfs_nodes_path.c_str());
  for (auto const& dirp_entry : fs::directory_iterator{dirp}) {
    fs::path node_path = dirp_entry.path();
    long long node_id = std::stoll(dirp_entry.path().stem().string());
    fs::path gpu_path = node_path / "gpu_id";
    std::ifstream gpu_id_file(gpu_path.c_str());
    std::string gpu_id_str;
    if (gpu_id_file.is_open()) {
      gpu_id_file >> gpu_id_str;
      if (!gpu_id_str.empty()) {
        long long gpu_id = std::stoll(gpu_id_str);
        if (gpu_id > 0) {
          Agent::DeviceInfo deviceInfo(node_id, gpu_id);
          // Since it is in static initializer, so its protected
          agent_device_map_.emplace(deviceInfo.getGPUId(), deviceInfo);
        }
      }
    }
  }
}

// Destructor of rocprofiler_singleton
// deletes the buffer pool
// Iterates over its session map and resets each session in its internal
// session map and clears them from the map. Pops labels from the range stack
// and deletes the stack.
ROCProfiler_Singleton::~ROCProfiler_Singleton() {
  {
    std::lock_guard<std::mutex> lock(session_map_lock_);
    if (!sessions_.empty()) {
      for (auto& session : sessions_) {
        if (session.second) delete session.second;
      }
      sessions_.clear();
    }
  }
  Counter::ClearBasicCounters();
}
ROCProfiler_Singleton& ROCProfiler_Singleton::GetInstance() {
  static ROCProfiler_Singleton* rocprofiler_singleton_instance =
      new ROCProfiler_Singleton;
  return *rocprofiler_singleton_instance;
}
bool ROCProfiler_Singleton::FindAgent(rocprofiler_agent_id_t agent_id) { return true; }
size_t ROCProfiler_Singleton::GetAgentInfoSize(rocprofiler_agent_info_kind_t kind,
                                               rocprofiler_agent_id_t agent_id) {
  return 0;
}
const char* ROCProfiler_Singleton::GetAgentInfo(rocprofiler_agent_info_kind_t kind,
                                                rocprofiler_agent_id_t agent_id) {
  return "";
}

// TODO(aelwazir): Implement Queue Query
bool ROCProfiler_Singleton::FindQueue(rocprofiler_queue_id_t queue_id) { return true; }
size_t ROCProfiler_Singleton::GetQueueInfoSize(rocprofiler_queue_info_kind_t kind,
                                               rocprofiler_queue_id_t queue_id) {
  return 0;
}
const char* ROCProfiler_Singleton::GetQueueInfo(rocprofiler_queue_info_kind_t kind,
                                                rocprofiler_queue_id_t queue_id) {
  return "";
}

bool ROCProfiler_Singleton::FindSession(rocprofiler_session_id_t session_id) {
  std::lock_guard<std::mutex> lock(session_map_lock_);
  return sessions_.find(session_id.handle) != sessions_.end();
}

rocprofiler_session_id_t ROCProfiler_Singleton::CreateSession(
    rocprofiler_replay_mode_t replay_mode) {
  rocprofiler_session_id_t session_id = rocprofiler_session_id_t{GenerateUniqueSessionId()};
  {
    std::lock_guard<std::mutex> lock(session_map_lock_);
    sessions_.emplace(session_id.handle, new Session(replay_mode, session_id));
  }
  return session_id;
}

void ROCProfiler_Singleton::DestroySession(rocprofiler_session_id_t session_id) {
  {
    std::lock_guard<std::mutex> lock(session_map_lock_);
    ASSERTM(sessions_.find(session_id.handle) != sessions_.end(),
            "Error: Couldn't find a created session with given id");
    if (sessions_.at(session_id.handle)) delete sessions_.at(session_id.handle);
    sessions_.erase(session_id.handle);
  }
}
profiler_serializer_t& ROCProfiler_Singleton::GetSerializer() { return profiler_serializer; }
bool ROCProfiler_Singleton::FindDeviceProfilingSession(rocprofiler_session_id_t session_id) {
  std::lock_guard<std::mutex> lock(device_profiling_session_map_lock_);
  return dev_profiling_sessions_.find(session_id.handle) != dev_profiling_sessions_.end();
}

rocprofiler_session_id_t ROCProfiler_Singleton::CreateDeviceProfilingSession(
    std::vector<std::string> counters, int cpu_agent_index, int gpu_agent_index) {
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

void ROCProfiler_Singleton::DestroyDeviceProfilingSession(rocprofiler_session_id_t session_id) {
  {
    std::lock_guard<std::mutex> lock(device_profiling_session_map_lock_);
    ASSERTM(dev_profiling_sessions_.find(session_id.handle) != dev_profiling_sessions_.end(),
            "Error: Couldn't find a created session with given id");
    delete dev_profiling_sessions_.at(session_id.handle);
    dev_profiling_sessions_.erase(session_id.handle);
  }
}

DeviceProfileSession* ROCProfiler_Singleton::GetDeviceProfilingSession(
    rocprofiler_session_id_t session_id) {
  std::lock_guard<std::mutex> lock(device_profiling_session_map_lock_);
  assert(dev_profiling_sessions_.find(session_id.handle) != dev_profiling_sessions_.end() &&
         "Error: Can't find the session!");
  return dev_profiling_sessions_.at(session_id.handle);
}

bool ROCProfiler_Singleton::HasActiveSession() { return GetCurrentSessionId().handle > 0; }
bool ROCProfiler_Singleton::IsActiveSession(rocprofiler_session_id_t session_id) {
  return (GetCurrentSessionId().handle == session_id.handle);
}

// Get the session by its id
// Looks up the session object for an input session id in the internal map.
// If a given session id doesn't exist, it throws an assertion.
// If a session object exists for the given session id, the session object is
// returned.
Session* ROCProfiler_Singleton::GetSession(rocprofiler_session_id_t session_id) {
  std::lock_guard<std::mutex> lock(session_map_lock_);
  assert(sessions_.find(session_id.handle) != sessions_.end() && "Error: Can't find the session!");
  return sessions_.at(session_id.handle);
}

// Get Current Session ID
rocprofiler_session_id_t ROCProfiler_Singleton::GetCurrentSessionId() {
  return current_session_id_;
}

void ROCProfiler_Singleton::SetCurrentActiveSession(rocprofiler_session_id_t session_id) {
  current_session_id_ = session_id;
}

uint64_t ROCProfiler_Singleton::GetUniqueRecordId() { return records_counter_.fetch_add(1); }

uint64_t ROCProfiler_Singleton::GetUniqueKernelDispatchId() {
  return kernel_dispatch_counter_.fetch_add(1, std::memory_order_relaxed);
}

size_t ROCProfiler_Singleton::GetKernelInfoSize(rocprofiler_kernel_info_kind_t kind,
                                                rocprofiler_kernel_id_t kernel_id) {
  switch (kind) {
    case ROCPROFILER_KERNEL_NAME:
      return GetKernelNameUsingDispatchID(kernel_id.handle).size();
    default:
      warning("The provided Kernel Kind is not yet supported!");
      return 0;
  }
}
const char* ROCProfiler_Singleton::GetKernelInfo(rocprofiler_kernel_info_kind_t kind,
                                                 rocprofiler_kernel_id_t kernel_id) {
  switch (kind) {
    case ROCPROFILER_KERNEL_NAME:
      return strdup(GetKernelNameUsingDispatchID(kernel_id.handle).c_str());
    default:
      warning("The provided Kernel Kind is not yet supported!");
      return "";
  }
}

const Agent::DeviceInfo& ROCProfiler_Singleton::GetDeviceInfo(uint64_t gpu_id) {
  std::lock_guard<std::mutex> info_map_lock(agent_device_map_mutex_);
  auto it = agent_device_map_.find(gpu_id);
  if (it == agent_device_map_.end())
    rocprofiler::fatal("Device Info is not found for the given id:%ld", gpu_id);
  return it->second;
}

// TODO(aelwazir): To be implemented
bool ROCProfiler_Singleton::CheckFilterData(rocprofiler_filter_kind_t filter_kind,
                                            rocprofiler_filter_data_t filter_data) {
  return true;
}

rocprofiler_timestamp_t ROCProfiler_Singleton::timestamp_ns() {

  static uint64_t sys_clock_period = 0;
  struct timespec ts;
  // We are not full on memory model. At worst each CPU can call it once.
  // We don't care for this variable's value in global memory as long as we have
  // non-zero in the CPU cache.
  if (sys_clock_period == 0) {
    clock_getres(CLOCK_BOOTTIME, &ts);
    sys_clock_period = (uint64_t(ts.tv_sec) * 1000000000 + uint64_t(ts.tv_nsec));
  }

  clock_gettime(CLOCK_BOOTTIME, &ts);
  uint64_t time = (uint64_t(ts.tv_sec) * 1000000000 + uint64_t(ts.tv_nsec));
  if (sys_clock_period != 1)
    return rocprofiler_timestamp_t{time / sys_clock_period};
  else
    return rocprofiler_timestamp_t{time};
}

rocprofiler_status_t IterateCounters(rocprofiler_counters_info_callback_t counters_info_callback) {
  if (hsa_support_IterateCounters(counters_info_callback)) return ROCPROFILER_STATUS_SUCCESS;
  return ROCPROFILER_STATUS_ERROR;
}

// End of ROCProfiler_Singleton Class

}  // namespace rocprofiler
