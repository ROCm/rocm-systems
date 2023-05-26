#include "eval_metrics.h"
#include "src/utils/helper.h"
#include "src/core/hsa/hsa_common.h"
#include <set>
#include <math.h>

using namespace rocmtools;


struct block_des_t {
  uint32_t id;
  uint32_t index;
};
struct lt_block_des {
  bool operator()(const block_des_t& a1, const block_des_t& a2) const {
    return (a1.id < a2.id) || ((a1.id == a2.id) && (a1.index < a2.index));
  }
};

struct block_status_t {
  uint32_t max_counters;
  uint32_t counter_index;
  uint32_t group_index;
};

typedef struct {
  std::vector<results_t*>* results;
  size_t index;
  uint32_t single_xcc_buff_size;
} callback_data_t;

static inline bool IsEventMatch(const hsa_ven_amd_aqlprofile_event_t& event1,
                                const hsa_ven_amd_aqlprofile_event_t& event2) {
  return (event1.block_name == event2.block_name) && (event1.block_index == event2.block_index) &&
      (event1.counter_id == event2.counter_id);
}

hsa_status_t pmcCallback(hsa_ven_amd_aqlprofile_info_type_t info_type,
                         hsa_ven_amd_aqlprofile_info_data_t* info_data, void* data) {
  hsa_status_t status = HSA_STATUS_SUCCESS;
  callback_data_t* passed_data = reinterpret_cast<callback_data_t*>(data);
  passed_data->index += 1;

  for (auto data_it = passed_data->results->begin(); data_it != passed_data->results->end();
       ++data_it) {
    if (info_type == HSA_VEN_AMD_AQLPROFILE_INFO_PMC_DATA) {
      if (IsEventMatch(info_data->pmc_data.event, (*data_it)->event)) {
        uint32_t xcc_index = floor(passed_data->index / passed_data->single_xcc_buff_size);
        (*data_it)->xcc_vals[xcc_index] += info_data->pmc_data.result;  // stores event result from each xcc separately
        (*data_it)->val_double += info_data->pmc_data.result;           // stores accumulated event result from all xccs
      }
    }
  }
  return status;
}


template <class Map> class MetricArgs : public xml::args_cache_t {
 public:
  MetricArgs(const Map* map) : map_(map) {}
  ~MetricArgs() {}
  bool Lookup(const std::string& name, double& result) const {
    results_t* counter_result = NULL;
    auto it = map_->find(name);
    if (it == map_->end()) std::cout << "var '" << name << "' is not found" << std::endl;
    counter_result = it->second;
    if (counter_result) {
      result = counter_result->val_double;
    } else
      std::cout << "var '" << name << "' info is NULL" << std::endl;
    return (counter_result != NULL);
  }

 private:
  const Map* map_;
};

static std::mutex extract_metric_events_lock;

bool metrics::ExtractMetricEvents(
    std::vector<std::string>& metric_names, hsa_agent_t gpu_agent, MetricsDict* metrics_dict,
    std::map<std::string, results_t*>& results_map, std::vector<event_t>& events_list,
    std::vector<results_t*>& results_list,
    std::map<std::pair<uint32_t, uint32_t>, uint64_t>& event_to_max_block_count,
    std::map<std::string, std::set<std::string>>& metrics_counters) {
  std::map<block_des_t, block_status_t, lt_block_des> groups_map;

  /* brief:
      results_map holds the result objects for each metric name(basic or derived)
      events_list holds the list of unique events from all the metrics entered
      results_list holds the result objects for each event (which means, basic counters only)
  */
  try {
    uint32_t xcc_count = rocmtools::hsa_support::GetAgentInfo(gpu_agent.handle).getXccCount();
    for (size_t i = 0; i < metric_names.size(); i++) {
      counters_vec_t counters_vec;
      // TODO: saurabh
      //   const Metric* metric = metrics_dict->GetMetricByName(metric_names[i]);
      const Metric* metric = metrics_dict->Get(metric_names[i]);
      if (metric == nullptr) {
          Agent::AgentInfo& agentInfo = rocmtools::hsa_support::GetAgentInfo(gpu_agent.handle);
          fatal("input metric'%s' not supported on this hardware: %s ", metric_names[i].c_str(),
          agentInfo.getName().data());

      }

      // adding result object for derived metric
      std::lock_guard<std::mutex> lock(extract_metric_events_lock);
      if (results_map.find(metric_names[i]) == results_map.end()) {
        results_map[metric_names[i]] = new results_t(metric_names[i], {}, xcc_count);
      }  // else {
         //  continue;
      // }

      counters_vec = metric->GetCounters();
      if (counters_vec.empty())
        rocmtools::fatal("bad metric '%s' is empty", metric_names[i].c_str());

      for (const counter_t* counter : counters_vec) {
        results_t* result = nullptr;
        if (metric->GetExpr()) {
          metrics_counters[metric->GetName()].insert(counter->name);
          // add this counter event only if it wasn't repeated before
          if (results_map.find(counter->name) != results_map.end()) {
            // std::cout << "Metric : " << metric->GetName() << " has " << counter->name
            //           << " which is already part of the results map!" << std::endl;
            // continue;
            result = results_map.at(counter->name);
          } else {
            // result object for base metric
            // std::cout << "Metric : " << metric->GetName() << " : " << counter->name << std::endl;
            result = new results_t(counter->name, {}, xcc_count);  // TODO: set correct initial value
            results_map[counter->name] = result;
          }
        } else {
          // std::cout << "Counter : " << metric->GetName() << " : " << counter->name << std::endl;
          result = results_map.at(counter->name);
        }
        // std::cout << "General Counter : " << metric->GetName() << " : " << counter->name <<
        // std::endl;
        const event_t* event = &(counter->event);
        const block_des_t block_des = {event->block_name, event->block_index};
        auto ret = groups_map.insert({block_des, {}});
        block_status_t& block_status = ret.first->second;
        if (block_status.max_counters == 0) {
          hsa_ven_amd_aqlprofile_profile_t query = {};
          query.agent = gpu_agent;
          query.type = HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_PMC;
          query.events = event;

          uint32_t max_block_counters;
          hsa_status_t status = hsa_ven_amd_aqlprofile_get_info(
              &query, HSA_VEN_AMD_AQLPROFILE_INFO_BLOCK_COUNTERS, &max_block_counters);
          if (status != HSA_STATUS_SUCCESS) fatal("get block_counters info failed");
          block_status.max_counters = max_block_counters;
        }

        // std::cout << "Counter: " << result->name << ", Block Index: " <<
        // counter->event.block_index
        //           << ", Counter ID: " << counter->event.counter_id
        //           << "\nBlock Status Group Index: " << block_status.group_index
        //           << ", Block Max Counters: " << block_status.max_counters
        //           << ", Block Status Counter ID: " << block_status.counter_index << std::endl;

        if (block_status.counter_index >= block_status.max_counters) {
          rocmtools::fatal("Metrics specified have exceeded HW limits!");
          return false;
        }
        block_status.counter_index += 1;
        events_list.push_back(counter->event);
        result->event = counter->event;
        results_list.push_back(result);
        event_to_max_block_count.emplace(
            std::make_pair(static_cast<uint32_t>(counter->event.block_name),
                           static_cast<uint32_t>(counter->event.block_index)),
            block_status.max_counters);
      }
    }
  } catch (std::string ex) {
    std::cout << ex << std::endl;
    abort();
  }

  return true;
}


bool metrics::GetCounterData(hsa_ven_amd_aqlprofile_profile_t* profile, hsa_agent_t gpu_agent,
                             std::vector<results_t*>& results_list) {
  uint32_t xcc_count = rocmtools::hsa_support::GetAgentInfo(gpu_agent.handle).getXccCount();
  uint32_t single_xcc_buff_size = profile->output_buffer.size /(sizeof(uint64_t) * xcc_count);
  callback_data_t callback_data{&results_list, 0, single_xcc_buff_size};
  hsa_status_t status = hsa_ven_amd_aqlprofile_iterate_data(profile, pmcCallback, &callback_data);
  return (status == HSA_STATUS_SUCCESS);
}

bool metrics::GetMetricsData(std::map<std::string, results_t*>& results_map,
                             std::vector<const Metric*>& metrics_list) {
  MetricArgs<std::map<std::string, results_t*>> args(&results_map);
  for (auto& metric : metrics_list) {
    const xml::Expr* expr = metric->GetExpr();
    if (expr) {
      auto it = results_map.find(metric->GetName());
      if (it == results_map.end()) rocmtools::fatal("metric results not found ");
      results_t* res = it->second;
      res->val_double = expr->Eval(args);
    }
  }

  return true;
}

void metrics::GetCountersAndMetricResultsByXcc(uint32_t xcc_index, std::vector<results_t*>& results_list,
                                 std::map<std::string, results_t*>& results_map,
                                 std::vector<const Metric*>& metrics_list){
    for(auto it = results_list.begin(); it != results_list.end(); it++){
      (*it)->val_double = (*it)->xcc_vals[xcc_index]; // set val_double to hold value for specific xcc
    }

    for(auto it = results_map.begin(); it != results_map.end(); it++){
      it->second->val_double = it->second->xcc_vals[xcc_index]; // set val_double to hold value for specific xcc
    }

    GetMetricsData(results_map, metrics_list);
}
