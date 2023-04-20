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

#include "src/core/session/profiler/profiler.h"

namespace rocmtools {

void InitKsymbols();
void FinitKsymbols();
void AddKernelName(uint64_t handle, std::string kernel_name);
void RemoveKernelName(uint64_t handle);
void AddKernelNameWithDispatchID(std::string name, uint64_t id);
std::string GetKernelNameUsingDispatchID(uint64_t given_id);
std::string GetKernelNameFromKsymbols(uint64_t handle);
uint32_t GetCurrentActiveInterruptSignalsCount();

namespace queue {

class Queue {
 public:
  Queue(const hsa_agent_t& cpu_agent, const hsa_agent_t& gpu_agent, uint32_t size,
        hsa_queue_type32_t type,
        void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data), void* data,
        uint32_t private_segment_size, uint32_t group_segment_size, hsa_queue_t** queue);
  ~Queue() {}

  hsa_queue_t* GetCurrentInterceptQueue();
  hsa_agent_t GetGPUAgent();
  hsa_agent_t GetCPUAgent();
  uint64_t GetQueueID();
  static void PrintCounters();
  std::mutex qw_mutex;

 private:
  std::mutex mutex_;
  hsa_agent_t cpu_agent_;
  hsa_agent_t gpu_agent_;
  hsa_queue_t* original_queue_;
  hsa_queue_t* intercept_queue_;

  hsa_status_t pmcCallback(hsa_ven_amd_aqlprofile_info_type_t info_type,
                           hsa_ven_amd_aqlprofile_info_data_t* info_data, void* data);
};

struct queue_info_session_t {
  hsa_agent_t agent;
  rocprofiler_session_id_t session_id;
  uint64_t queue_id;
  uint32_t writer_id;
  hsa_signal_t interrupt_signal;
};

void AddRecordCounters(rocprofiler_record_profiler_t* record, const pending_signal_t& pending);

void InitializePools(hsa_agent_t cpu_agent);

}  // namespace queue
}  // namespace rocmtools

#endif  // SRC_CORE_HSA_QUEUES_QUEUE_H_
