/******************************************************************************
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*******************************************************************************/

#ifndef SRC_CORE_METRICS_EVALMETRICS_H_
#define SRC_CORE_METRICS_EVALMETRICS_H_

#include <vector>
#include <map>
#include "metrics.h"
#include <hsa/hsa.h>

#include <atomic>
#include <set>

namespace rocprofiler {

class results_t {
 public:
  results_t(std::string in_name, event_t in_event, int xcc_count)
      : name(in_name), val_double(0), event(in_event) { xcc_vals.reserve(xcc_count); }
  std::string name;
  double val_double;
  std::vector<double> xcc_vals;
  event_t event;
};

typedef struct {
  packet_t* start_packet;
  packet_t* stop_packet;
  packet_t* read_packet;
  rocprofiler::MetricsDict* metrics_dict;
  std::vector<const rocprofiler::Metric*> metrics_list;
  std::map<std::string, rocprofiler::results_t*> results_map;
  std::vector<rocprofiler::results_t*> results_list;
  std::vector<event_t> events_list;
  hsa_agent_t gpu_agent;
  hsa_signal_t begin_signal;
  std::atomic<bool> begin_completed{false};
} profiling_context_t;

namespace metrics {
bool ExtractMetricEvents(
    std::vector<std::string>& metric_names, hsa_agent_t gpu_agent, MetricsDict* metrics_dict,
    std::map<std::string, results_t*>& results_map, std::vector<event_t>& events_list,
    std::vector<results_t*>& results_list,
    std::map<std::pair<uint32_t, uint32_t>, uint64_t>& event_to_max_block_count,
    std::map<std::string, std::set<std::string>>& metrics_counters);


bool GetCounterData(hsa_ven_amd_aqlprofile_profile_t* profile, hsa_agent_t gpu_agent,
                    std::vector<results_t*>& results_list);

bool GetMetricsData(std::map<std::string, results_t*>& results_map,
                    std::vector<const Metric*>& metrics_list, uint64_t kernel_duration = 0);

void GetCountersAndMetricResultsByXcc(uint32_t xcc_index, std::vector<results_t*>& results_list,
                                      std::map<std::string, results_t*>& results_map,
                                      std::vector<const Metric*>& metrics_list,
                                      uint64_t kernel_duration = 0);

}  // namespace metrics
}  // namespace rocprofiler

#endif  // SRC_CORE_METRICS_EVALMETRICS_H_