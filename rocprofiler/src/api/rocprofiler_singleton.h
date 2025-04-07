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

#ifndef SRC_TOOLS_ROCPROFILER_SINGLETON_H_
#define SRC_TOOLS_ROCPROFILER_SINGLETON_H_

#include <hsa/hsa_ven_amd_aqlprofile.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stack>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "src/core/session/session.h"
#include "src/core/session/device_profiling.h"
#include "src/core/hardware/hsa_info.h"
#include "src/core/hsa/queues/queue.h"
namespace rocprofiler {
  /*This is a profiler serializer. It should be instantiated
 only once for the profiler. The following is the
 description of each field.
 1. dispatch_queue - The queue to which the currently dispatched kernel
            belongs to.
            At any given time, in serialization only one kernel
            can be executing.
 2. dispatch_ready- It is a software data structure which holds
            the queues which have a kernel ready to be dispatched.
            This stores the queues in FIFO order.
 3. serializer_mutex - The mutex is used for thread synchronization
            while accessing the singleton instance of this structure.
 Currently, in case of profiling kernels are serialized by default.
*/
struct profiler_serializer_t {
  queue::Queue* dispatch_queue{nullptr};
  std::vector<rocprofiler::queue::Queue*> dispatch_ready;
  std::mutex serializer_mutex;
};


class ROCProfiler_Singleton {
 public:
  ROCProfiler_Singleton(const ROCProfiler_Singleton&) = delete;
  ROCProfiler_Singleton& operator=(const ROCProfiler_Singleton&) = delete;
  static ROCProfiler_Singleton& GetInstance();

  bool FindAgent(rocprofiler_agent_id_t agent_id);
  size_t GetAgentInfoSize(rocprofiler_agent_info_kind_t kind, rocprofiler_agent_id_t agent_id);
  const char* GetAgentInfo(rocprofiler_agent_info_kind_t kind, rocprofiler_agent_id_t agent_id);

  bool FindQueue(rocprofiler_queue_id_t queue_id);
  size_t GetQueueInfoSize(rocprofiler_queue_info_kind_t kind, rocprofiler_queue_id_t queue_id);
  const char* GetQueueInfo(rocprofiler_queue_info_kind_t kind, rocprofiler_queue_id_t queue_id);

  bool FindKernel(rocprofiler_kernel_id_t kernel_id);
  size_t GetKernelInfoSize(rocprofiler_kernel_info_kind_t kind, rocprofiler_kernel_id_t kernel_id);
  const char* GetKernelInfo(rocprofiler_kernel_info_kind_t kind, rocprofiler_kernel_id_t kernel_id);

  // Session
  rocprofiler_session_id_t CreateSession(rocprofiler_replay_mode_t replay_mode);
  void DestroySession(rocprofiler_session_id_t session_id);
  bool HasActiveSession();
  rocprofiler_session_id_t GetCurrentSessionId();
  void SetCurrentActiveSession(rocprofiler_session_id_t session_id);
  bool FindSession(rocprofiler_session_id_t session_id);
  bool IsActiveSession(rocprofiler_session_id_t session_id);
  Session* GetSession(rocprofiler_session_id_t session_id);

  // Device Profiling Session
  bool FindDeviceProfilingSession(rocprofiler_session_id_t session_id);
  rocprofiler_session_id_t CreateDeviceProfilingSession(std::vector<std::string> counters,
                                                        int cpu_agent_index, int gpu_agent_index);
  void DestroyDeviceProfilingSession(rocprofiler_session_id_t session_id);
  DeviceProfileSession* GetDeviceProfilingSession(rocprofiler_session_id_t session_id);
  profiler_serializer_t& GetSerializer();

  // Generic
  bool CheckFilterData(rocprofiler_filter_kind_t filter_kind,
                       rocprofiler_filter_data_t filter_data);
  uint64_t GetUniqueRecordId();
  uint64_t GetUniqueKernelDispatchId();
  const Agent::DeviceInfo& GetDeviceInfo(uint64_t gpu_id);
  rocprofiler_timestamp_t timestamp_ns();
 private:
  std::atomic<rocprofiler_session_id_t> current_session_id_{rocprofiler_session_id_t{0}};
  std::mutex session_map_lock_;
  std::map<uint64_t, Session*> sessions_;
  std::atomic<uint64_t> records_counter_{0};
  std::mutex device_profiling_session_map_lock_;
  std::map<uint64_t, DeviceProfileSession*> dev_profiling_sessions_;
  std::mutex agent_device_map_mutex_;
  std::unordered_map<uint64_t, Agent::DeviceInfo> agent_device_map_;
  ROCProfiler_Singleton();
  ~ROCProfiler_Singleton();
   profiler_serializer_t profiler_serializer;
  /*
   * XXX: Associating PC samples with a running kernel requires an identifier
   * that will be unique across all kernel executions.  It is not enough to use
   * the name of a kernel or the address of a kernel object, as these will be
   * identical if the same kernel is dispatched twice.  Currently, this
   * identifier is written to the `reserved2` field of the dispatch packet when
   * its launch is intercepted, but this could change: a future version of
   * ROCProfiler may instead attempt to identify a kernel by a key with high
   * _probability_ of uniqueness: for example, a combination of the kernel's
   * name, the queue ID to which it was dispatched, and the offset of the queue
   * write pointer is likely sufficient to associate PC samples with a running
   * kernel and have the PC sample records consumed by the user-provided async
   * callback before the write pointer wraps to the same position in the ring
   * buffer.
   */
  std::atomic<uint64_t> kernel_dispatch_counter_{1};
};

rocprofiler_status_t IterateCounters(rocprofiler_counters_info_callback_t counters_info_callback);
}  // namespace rocprofiler

#endif  // SRC_TOOLS_ROCPROFILER_SINGLETON_H_
