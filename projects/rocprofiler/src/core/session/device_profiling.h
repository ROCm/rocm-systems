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

#ifndef SRC_CORE_SESSION_DEVICE_PROFILING_H_
#define SRC_CORE_SESSION_DEVICE_PROFILING_H_

#include "rocprofiler.h"
#include "src/core/hsa/packets/packets_generator.h"
#include <mutex>
// #include "src/core/counters/rdc/rdc_metrics.h"
#include "src/core/counters/metrics/metrics.h"
#include "src/core/counters/metrics/eval_metrics.h"


namespace rocprofiler {

class DeviceProfileSession {
 public:
  void StartSession();
  void PollMetrics(rocprofiler_device_profile_metric_t* data);
  void StopSession();

  DeviceProfileSession(std::vector<std::string> counters, hsa_agent_t cpu_agent,
                       hsa_agent_t gpu_agent, uint64_t* session_id);

  ~DeviceProfileSession();

 private:
  bool createQueue();
  bool generatePackets();
  bool readPmcCounters();

  static hsa_queue_t* getQueue(hsa_agent_t);

  uint64_t session_id_;

  std::vector<std::string> profiling_data_;

  hsa_agent_t cpu_agent_;
  hsa_agent_t gpu_agent_;

  static std::map<uint64_t, hsa_queue_t*> agent_queue_map_;
  static std::mutex agent_queue_map_mutex_;

  Packet::packet_t start_packet_;
  Packet::packet_t stop_packet_;
  Packet::packet_t read_packet_;

  MetricsDict* metrics_dict_;
  std::vector<const Metric*> metrics_list_;
  std::map<std::string, results_t*> results_map_;
  std::vector<event_t> events_list_;
  std::vector<results_t*> results_list_;

  hsa_signal_t completion_signal_;
  hsa_signal_t start_signal_;
  hsa_signal_t stop_signal_;

  // TODO: remove this or do actual cleanup
  hsa_ven_amd_aqlprofile_profile_t* profile_;
};

bool find_hsa_agent_cpu(uint64_t index, hsa_agent_t* agent);
bool find_hsa_agent_gpu(uint64_t index, hsa_agent_t* agent);


}  // namespace rocprofiler


#endif  // SRC_CORE_SESSION_DEVICE_PROFILING_H_
