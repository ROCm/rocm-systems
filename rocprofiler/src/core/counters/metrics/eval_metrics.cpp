#include "eval_metrics.h"
#include "src/utils/helper.h"
#include "src/core/hsa/hsa_support.h"
#include <cstdint>
#include <exception>
#include <set>
#include <utility>
#include <math.h>
#include <sys/types.h>

using namespace rocprofiler;


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

struct aqlprofile_event_t : public hsa_ven_amd_aqlprofile_event_t
{
  bool operator==(const aqlprofile_event_t& other) const {
    return this->block_name == other.block_name &&
           this->block_index == other.block_index &&
           this->counter_id ==other.counter_id;
  }
};

template <>
struct std::hash<aqlprofile_event_t>
{
  std::size_t operator()(const aqlprofile_event_t& k) const {
    return (int(k.block_name)<<20) ^ (int(k.counter_id)<<10) ^ int(k.block_index);
  }
};

typedef struct {
  std::unordered_map<aqlprofile_event_t, results_t*> results;
  size_t index;
} callback_data_t;

hsa_status_t pmcCallback(hsa_ven_amd_aqlprofile_info_type_t info_type,
                         hsa_ven_amd_aqlprofile_info_data_t* info_data, void* data) {
  if (info_type != HSA_VEN_AMD_AQLPROFILE_INFO_PMC_DATA) return HSA_STATUS_SUCCESS;
  callback_data_t* passed_data = reinterpret_cast<callback_data_t*>(data);

  auto it = passed_data->results.find(aqlprofile_event_t{info_data->pmc_data.event});
  if (it == passed_data->results.end())
    return HSA_STATUS_ERROR;

  auto* res = it->second;
  res->xcc_vals.push_back(info_data->pmc_data.result);
  res->val_double += info_data->pmc_data.result;
  passed_data->index += 1;

  return HSA_STATUS_SUCCESS;
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
    HSASupport_Singleton& hsasupport_singleton = HSASupport_Singleton::GetInstance();
    size_t xcc_count = hsasupport_singleton.GetHSAAgentInfo(gpu_agent.handle).GetDeviceInfo().getXccCount();


    for (size_t i = 0; i < metric_names.size(); i++) {
      counters_vec_t counters_vec;
      // TODO: saurabh
      //   const Metric* metric = metrics_dict->GetMetricByName(metric_names[i]);
      const Metric* metric = metrics_dict->Get(metric_names[i]);
      if (metric == nullptr) {
          HSAAgentInfo& agentInfo = HSASupport_Singleton::GetInstance().GetHSAAgentInfo(gpu_agent.handle);
          fatal("input metric'%s' not supported on this hardware: %s ", metric_names[i].c_str(),
          agentInfo.GetDeviceInfo().getName().data());

      }

      // adding result object for derived metric
      std::lock_guard<std::mutex> lock(extract_metric_events_lock);

      if (metric_names[i].compare("KERNEL_DURATION") == 0) {
        if (results_map.find(metric_names[i]) == results_map.end()) {
          results_map[metric_names[i]] = new results_t(metric_names[i], {}, xcc_count);
        }
        continue;
      }
      counters_vec = metric->GetCounters();
      if (counters_vec.empty())
        rocprofiler::fatal("bad metric '%s' is empty", metric_names[i].c_str());

      if (metric->GetExpr() && results_map.find(metric_names[i]) == results_map.end()) {
        results_map[metric_names[i]] = new results_t(metric_names[i], {}, xcc_count);
        for (const counter_t* counter : counters_vec)
          metrics_counters[metric->GetName()].insert(counter->name);
      }

      for (const counter_t* counter : counters_vec) {
        if (results_map.find(counter->name) != results_map.end()) continue;

        results_t* result = new results_t(counter->name, {}, xcc_count);
        results_map[counter->name] = result;

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

        if (block_status.counter_index >= block_status.max_counters) {
          std::cerr << "Error: "
                    << std::string_view(counter->name)
                           .substr(0, std::string_view(counter->name).find("_"))
                    << " exceeded hardware block counters limit (" << block_status.max_counters
                    << ")" << std::endl;
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
  size_t gpu_xcc_count = HSASupport_Singleton::GetInstance().GetHSAAgentInfo(gpu_agent.handle).GetDeviceInfo().getXccCount();

  callback_data_t callback_data{};
  for (auto* res : results_list)
    callback_data.results[aqlprofile_event_t{res->event}] = res;
  hsa_status_t status = hsa_ven_amd_aqlprofile_iterate_data(profile, pmcCallback, &callback_data);

  for (auto& data : results_list)
  {
    size_t xcc_count = (data->event.block_name != HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_UMC) ? gpu_xcc_count : 1;
    std::vector<double> xcc_results = std::move(data->xcc_vals);
    data->xcc_vals.clear();
    data->xcc_vals.reserve(xcc_count);
    size_t split = (xcc_results.size()+xcc_count-1) / xcc_count;

    // Exploit the fact AQLprofile calls counter in order per XCC
    for (size_t xcc=0; xcc < xcc_count; xcc++)
    {
      double accumulated = 0;
      for (size_t s=xcc*split; s<std::min(xcc_results.size(), (xcc+1)*split); s++)
        accumulated += xcc_results.at(s);
      data->xcc_vals.push_back(accumulated);
    }
  }

  return (status == HSA_STATUS_SUCCESS);
}

bool metrics::GetMetricsData(std::map<std::string, results_t*>& results_map,
                             std::vector<const Metric*>& metrics_list, uint64_t kernel_duration) {
  MetricArgs<std::map<std::string, results_t*>> args(&results_map);
  for (auto& metric : metrics_list) {
    const xml::Expr* expr = metric->GetExpr();
    if (expr) {
      auto it = results_map.find(metric->GetName());
      if (it == results_map.end()) rocprofiler::fatal("metric results not found ");
      results_t* res = it->second;
      if (metric->GetName().compare("KERNEL_DURATION") == 0) {
        res->val_double = kernel_duration;
        continue;
      }
      res->val_double = expr->Eval(args);
    }
  }

  return true;
}

void metrics::GetCountersAndMetricResultsByXcc(uint32_t xcc_index,
                                               std::vector<results_t*>& results_list,
                                               std::map<std::string, results_t*>& results_map,
                                               std::vector<const Metric*>& metrics_list,
                                               uint64_t kernel_duration) {
  for (auto* it : results_list) // set val_double to hold value for specific xcc
    it->val_double = it->xcc_vals[xcc_index];

  for (auto& [str, it] : results_map)
    it->val_double = it->xcc_vals[xcc_index];

  GetMetricsData(results_map, metrics_list, kernel_duration);
}
