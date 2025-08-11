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

#ifndef SRC_CORE_HSA_QUEUES_QUEUE_H_
#define SRC_CORE_HSA_QUEUES_QUEUE_H_

#include "rocprofiler.h"

#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <hsa/amd_hsa_kernel_code.h>

#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <condition_variable>
#include <shared_mutex>
#include "src/core/session/profiler/profiler.h"

namespace rocprofiler {


void AddKernelName(uint64_t handle, std::string kernel_name);
void RemoveKernelName(uint64_t handle);
void AddKernelNameWithDispatchID(std::string name, uint64_t id);
std::string GetKernelNameUsingDispatchID(uint64_t given_id);
std::string GetKernelNameFromKsymbols(uint64_t handle);
// uint32_t GetCurrentActiveInterruptSignalsCount();

namespace queue {

/* The enum here represents the
state of the queue destruction.
1. normal-The queue  destructor is not initiated.
2. to_destroy - The queue destructor has been initiated.
3. done_destroy - The async handler has been unregistered
and the destructor can now complete.
*/
enum is_destroy {
 normal=0,
 to_destroy=1,
 done_destroy=2
};

class Queue {
 public:
  Queue(const hsa_agent_t cpu_agent, const hsa_agent_t gpu_agent,
         hsa_queue_t* queue);
  ~Queue();

  static void WriteInterceptor(const void* packets, uint64_t pkt_count, uint64_t user_pkt_index,
                               void* data, hsa_amd_queue_intercept_packet_writer writer);
  static bool ATTWriteInterceptor(const void* packets, uint64_t pkt_count, uint64_t user_pkt_index,
                               void* data, hsa_amd_queue_intercept_packet_writer writer);
  static bool ATTSingleWriteInterceptor(const void* packets, uint64_t pkt_count, uint64_t user_pkt_index,
                               void* data, hsa_amd_queue_intercept_packet_writer writer);
  static bool ATTContiguousWriteInterceptor(const void* packets, uint64_t pkt_count, uint64_t user_pkt_index,
                               void* data, hsa_amd_queue_intercept_packet_writer writer);
  hsa_queue_t* GetCurrentInterceptQueue();
  hsa_agent_t GetGPUAgent();
  hsa_agent_t GetCPUAgent();
  uint64_t GetQueueID();
  static void PrintCounters();
  std::mutex qw_mutex;
  enum is_destroy state;
  std::condition_variable  cv_ready_signal;
  hsa_signal_t GetReadySignal();
  hsa_signal_t GetBlockSignal();

  static void ResetSessionID(rocprofiler_session_id_t id = rocprofiler_session_id_t{0});
  static bool CheckNeededProfileConfigs();
 private:
  static std::shared_mutex session_id_mutex;
  static rocprofiler_session_id_t session_id;

  hsa_agent_t cpu_agent_;
  hsa_agent_t gpu_agent_;
  hsa_queue_t* intercept_queue_;
  hsa_signal_t block_signal_;
  hsa_signal_t ready_signal_;

  bool unreg_async_handler_{false};
  hsa_status_t pmcCallback(hsa_ven_amd_aqlprofile_info_type_t info_type,
                           hsa_ven_amd_aqlprofile_info_data_t* info_data, void* data);
};

struct queue_info_session_t {
  hsa_agent_t agent;
  rocprofiler_session_id_t session_id;
  uint64_t queue_id;
  size_t writer_id;
  hsa_signal_t interrupt_signal;
  uint64_t gpu_index;
  size_t xcc_count;
  hsa_signal_t block_signal;
};

void AddRecordCounters(rocprofiler_record_profiler_t* record, const pending_signal_t& pending);

void CheckPacketReqiurements();

}  // namespace queue
}  // namespace rocprofiler

#endif  // SRC_CORE_HSA_QUEUES_QUEUE_H_
