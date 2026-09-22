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

#ifndef SRC_CORE_GROUP_SET_H_
#define SRC_CORE_GROUP_SET_H_

#include <stdio.h>
#include <map>
#include <vector>

#include "core/metrics.h"
#include "util/exception.h"
#include "util/hsa_rsrc_factory.h"

namespace rocprofiler {

// Block descriptor
struct block_des_t {
  uint32_t id;
  uint32_t index;
};

// block_des_t less-then functor
struct lt_block_des {
  bool operator()(const block_des_t& a1, const block_des_t& a2) const {
    return (a1.id < a2.id) || ((a1.id == a2.id) && (a1.index < a2.index));
  }
};

// Block status
struct block_status_t {
  uint32_t max_counters;
  uint32_t counter_index;
  uint32_t group_index;
};

// Metrics set class
class MetricsGroup {
 public:
  // Info map type
  typedef std::map<std::string, const Metric*> info_map_t;
  // Blocks map type
  typedef std::map<block_des_t, block_status_t, lt_block_des> blocks_map_t;

  MetricsGroup(const util::AgentInfo* agent_info) : agent_info_(agent_info) {
    metrics_ = MetricsDict::Create(agent_info);
    if (metrics_ == NULL) EXC_RAISING(HSA_STATUS_ERROR, "MetricsDict create failed");
  }

  void Print(FILE* file) const {
    for (const Metric* metric : metrics_vec_) {
      fprintf(file, " %s", metric->GetName().c_str());
      fflush(stdout);
    }
    fprintf(file, "\n");
    fflush(stdout);
  }

  static const Metric* GetMetric(const MetricsDict* metrics, const std::string& name) {
    // Metric object
    const Metric* metric = metrics->Get(name);
    if (metric == NULL) EXC_RAISING(HSA_STATUS_ERROR, "input metric '" << name << "' is not found");
    return metric;
  }

  static const Metric* GetMetric(const MetricsDict* metrics, const rocprofiler_feature_t* info) {
    // Metrics name
    const char* name = info->name;
    if (name == NULL) EXC_RAISING(HSA_STATUS_ERROR, "input feature name is NULL");
    const Metric* metric = GetMetric(metrics, name);
#if 0
    std::cout << "    " << name << (metric->GetExpr() ? " = " + metric->GetExpr()->String() : " counter") << std::endl;
#endif
    return metric;
  }

  // Add metric
  bool AddMetric(const rocprofiler_feature_t* info) { return AddMetric(GetMetric(metrics_, info)); }

  bool AddMetric(const Metric* metric) {
    // Blocks utilization delta
    blocks_map_t blocks_delta;

    // Process metrics counters
    const counters_vec_t& counters_vec = metric->GetCounters();
    if (counters_vec.empty())
      EXC_RAISING(HSA_STATUS_ERROR, "bad metric '" << metric->GetName() << "' is empty");

    for (const counter_t* counter : counters_vec) {
      const event_t* event = &(counter->event);

      // For metrics expressions checking that there is no the same counter in the input metrics
      // and also that the counter wasn't registered already by another input metric expression
      if (info_map_.find(counter->name) != info_map_.end()) continue;

      const block_des_t block_des = {event->block_name, event->block_index};
      auto ret = blocks_map_.insert({block_des, {}});
      block_status_t& block_status = ret.first->second;
      if (ret.second == true) {
        profile_t query = {};
        query.agent = agent_info_->dev_id;
        query.type = HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_PMC;
        query.events = event;

        uint32_t block_counters;
        hsa_status_t status =
            util::HsaRsrcFactory::Instance().AqlProfileApi()->hsa_ven_amd_aqlprofile_get_info(
                &query, HSA_VEN_AMD_AQLPROFILE_INFO_BLOCK_COUNTERS, &block_counters);
        if (status != HSA_STATUS_SUCCESS) AQL_EXC_RAISING(status, "get block_counters info");
        block_status.max_counters = block_counters;
      }

      ret = blocks_delta.insert({block_des, block_status});
      block_status_t& delta_status = ret.first->second;
      delta_status.counter_index += 1;
      if (delta_status.counter_index > delta_status.max_counters) return false;
    }

    // Register metric
    metrics_vec_.push_back(metric);
    info_map_[metric->GetName()] = metric;
    for (const counter_t* counter : counters_vec) {
      if (info_map_.find(counter->name) == info_map_.end())
        info_map_[counter->name] = NewCounterInfo(counter->name);
    }
    for (const auto& entry : blocks_delta) {
      blocks_map_[entry.first] = entry.second;
    }

    return true;
  }

 private:
  const Metric* NewCounterInfo(const std::string& name) const { return GetMetric(metrics_, name); }

  // Agent info
  const util::AgentInfo* const agent_info_;
  // Metrics dictionary
  const MetricsDict* metrics_;
  // Info map
  info_map_t info_map_;
  // Blocks map
  blocks_map_t blocks_map_;
  // Metrics vector
  std::vector<const Metric*> metrics_vec_;
};

// Metrics groups class
class MetricsGroupSet {
 public:
  MetricsGroupSet(const util::AgentInfo* agent_info, const rocprofiler_feature_t* info_array,
                  const uint32_t info_count)
      : agent_info_(agent_info) {
    metrics_ = MetricsDict::Create(agent_info);
    if (metrics_ == NULL) EXC_RAISING(HSA_STATUS_ERROR, "MetricsDict create failed");
    Initialize(info_array, info_count);
  }

  ~MetricsGroupSet() {
    for (auto* group : groups_) delete group;
  }

  uint32_t GetSize() const { return groups_.size(); }

  void Print(FILE* file) const {
    for (const auto* group : groups_) {
      fprintf(stdout, " pmc : ");
      fflush(stdout);
      group->Print(file);
    }
  }

 private:
  void Initialize(const rocprofiler_feature_t* info_array, const uint32_t info_count) {
    std::multimap<uint32_t, const Metric*, std::greater<uint32_t> > input_metrics;
    for (unsigned i = 0; i < info_count; ++i) {
      const rocprofiler_feature_t* info = &info_array[i];
      if (info->kind != ROCPROFILER_FEATURE_KIND_METRIC) continue;
      const Metric* metric = MetricsGroup::GetMetric(metrics_, info);
      const uint32_t counters_num = metric->GetCounters().size();
      input_metrics.insert({counters_num, metric});

      if (MetricsGroup(agent_info_).AddMetric(metric) == false) {
        AQL_EXC_RAISING(HSA_STATUS_ERROR,
                        "Metric '" << metric->GetName() << "' doesn't fit in one group");
      }
    }
#if 0
    for (const auto& entry : input_metrics) {
      printf("%u %s\n", entry.first, entry.second->GetName().c_str());
    }
#endif
    auto end = input_metrics.end();
    while (!input_metrics.empty()) {
      MetricsGroup* group = NextGroup();
      auto it = input_metrics.begin();
      do {
        auto curr = it++;
        const Metric* metric = curr->second;
        if (group->AddMetric(metric) == true) {
          input_metrics.erase(curr);
        }
      } while (it != end);
    }
  }

  MetricsGroup* NextGroup() {
    groups_.push_back(new MetricsGroup(agent_info_));
    return groups_.back();
  }

  // Agent info
  const util::AgentInfo* const agent_info_;
  // Metrics dictionary
  const MetricsDict* metrics_;
  // Metrics group vector
  std::vector<MetricsGroup*> groups_;
};

}  // namespace rocprofiler

#endif  // SRC_CORE_GROUP_SET_H_
