#ifndef SRC_CORE_METRICS_H_
#define SRC_CORE_METRICS_H_

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <fstream>
#include <iostream>
#include <thread>
#include <map>
#include <vector>

#include "core/types.h"
#include "util/exception.h"
#include "util/hsa_rsrc_factory.h"
#include "xml/expr.h"
#include "xml/xml.h"

namespace rocprofiler {
struct counter_t {
  std::string name;
  event_t event;
};
typedef std::vector<const counter_t*> counters_vec_t;

class Metric {
 public:
  Metric(const std::string& name) : name_(name) {}
  virtual ~Metric() {}
  std::string GetName() const { return name_; }
  virtual void GetCounters(counters_vec_t& vec) const = 0;
  counters_vec_t GetCounters() const {
    counters_vec_t counters;
    GetCounters(counters);
    return counters;
  }
  virtual const xml::Expr* GetExpr() const = 0;

 private:
  const std::string name_;
};

class BaseMetric : public Metric {
 public:
  BaseMetric(const std::string& name, const counter_t& counter) : Metric(name), counter_(counter) {}
  void GetCounters(counters_vec_t& vec) const { vec.push_back(&counter_); }
  const xml::Expr* GetExpr() const { return NULL; }

 private:
  const counter_t counter_;
};

class ExprMetric : public Metric {
 public:
  ExprMetric(const std::string& name, const counters_vec_t& counters, const xml::Expr* expr)
      : Metric(name), counters_(counters), expr_(expr) {}
  void GetCounters(counters_vec_t& vec) const {
    vec.insert(vec.end(), counters_.begin(), counters_.end());
  }
  const xml::Expr* GetExpr() const { return expr_; }

 private:
  const counters_vec_t counters_;
  const xml::Expr* expr_;
};

class MetricsDict {
 public:
  typedef std::map<std::string, const Metric*> cache_t;
  typedef cache_t::const_iterator const_iterator_t;
  typedef std::map<std::string, MetricsDict*> map_t;
  typedef std::mutex mutex_t;

  class ExprCache : public xml::expr_cache_t {
   public:
    ExprCache(const cache_t* cache) : cache_(cache) {}
    bool Lookup(const std::string& name, std::string& result) const {
      bool ret = false;
      auto it = cache_->find(name);
      if (it != cache_->end()) {
        ret = true;
        const rocprofiler::ExprMetric* expr_metric =
            dynamic_cast<const rocprofiler::ExprMetric*>(it->second);
        if (expr_metric) result = expr_metric->GetExpr()->GetStr();
      }
      return ret;
    }

   private:
    const cache_t* const cache_;
  };

  static MetricsDict* Create(const util::AgentInfo* agent_info) {
    std::lock_guard<mutex_t> lck(mutex_);
    if (map_ == NULL) map_ = new map_t;
    auto ret = map_->insert({agent_info->gfxip, NULL});
    if (ret.second) ret.first->second = new MetricsDict(agent_info);
    return ret.first->second;
  }

  static void Destroy() {
    if (map_ != NULL) {
      for (auto& entry : *map_) delete entry.second;
      delete map_;
      map_ = NULL;
    }
  }

  const Metric* Get(const std::string& name) {
    const Metric* metric = NULL;

    auto it = cache_.find(name);
    if (it != cache_.end()) metric = it->second;
    else {
      const std::size_t pos = name.find(':');
      if (pos != std::string::npos) {
        std::string block_name = name.substr(0, pos);
        const std::string event_str = name.substr(pos + 1);

        uint32_t block_index = 0;
        bool indexed = false;
        const std::size_t pos1 = block_name.find('[');
        if (pos1 != std::string::npos) {
          const std::size_t pos2 = block_name.find(']');
          if (pos2 == std::string::npos) EXC_RAISING(HSA_STATUS_ERROR, "Malformed metric name '" << name << "'");
          block_name = name.substr(0, pos1);
          const std::string block_index_str = name.substr(pos1 + 1, pos2 - (pos1 + 1)); 
          block_index = atol(block_index_str.c_str());
          indexed = true;
        }

        const hsa_ven_amd_aqlprofile_id_query_t query = Translate(agent_info_, block_name);
        const hsa_ven_amd_aqlprofile_block_name_t block_id = (hsa_ven_amd_aqlprofile_block_name_t)query.id;
        if ((query.instance_count > 1) && (indexed == false)) EXC_RAISING(HSA_STATUS_ERROR, "Malformed indexed metric name '" << name << "'");
        const uint32_t event_id = atol(event_str.c_str());
        const counter_t counter = {name, {block_id, block_index, event_id}};
        metric = new BaseMetric(name, counter);
      }
    }

    return metric;
  }

  uint32_t Size() const { return cache_.size(); }
  const_iterator_t Begin() const { return cache_.begin(); }
  const_iterator_t End() const { return cache_.end(); }

 private:
  MetricsDict(const util::AgentInfo* agent_info) : xml_(NULL), agent_info_(agent_info) {
    const char* xml_name = getenv("ROCP_METRICS");
    if (xml_name != NULL) {
      xml_ = xml::Xml::Create(xml_name);
      if (xml_ == NULL) EXC_RAISING(HSA_STATUS_ERROR, "metrics .xml open error '" << xml_name << "'");
      xml_->AddConst("top.const.metric", "NUM_SIMDS", 64);
      xml_->AddConst("top.const.metric", "NUM_SHADER_ENGINES", 4);
      std::cout << "ROCProfiler: importing '" << xml_name << "':" << std::endl;
      ImportMetrics(agent_info, "const");
      ImportMetrics(agent_info, agent_info->gfxip);
      ImportMetrics(agent_info, "global");
    }
  }

  ~MetricsDict() {
    xml::Xml::Destroy(xml_);
    for (auto& entry : cache_) delete entry.second;
  }

  static hsa_ven_amd_aqlprofile_id_query_t Translate(const util::AgentInfo* agent_info, const std::string& block_name) {
    hsa_ven_amd_aqlprofile_profile_t profile;
    profile.agent = agent_info->dev_id;
    hsa_ven_amd_aqlprofile_id_query_t query = {block_name.c_str(), 0, 0};
    hsa_status_t status =
        util::HsaRsrcFactory::Instance().AqlProfileApi()->hsa_ven_amd_aqlprofile_get_info(
            &profile, HSA_VEN_AMD_AQLPROFILE_INFO_BLOCK_ID, &query);
    if (status != HSA_STATUS_SUCCESS) AQL_EXC_RAISING(HSA_STATUS_ERROR, "ImportMetrics: bad block name '" << block_name << "'");
    return query;
  }

  void ImportMetrics(const util::AgentInfo* agent_info, const std::string& scope) {
    auto scope_list = xml_->GetNodes("top." + scope + ".metric");
    if (!scope_list.empty()) {
      std::cout << "  " << scope_list.size() << " " << scope << " metrics found" << std::endl;

      for (auto node : scope_list) {
        const std::string name = node->opts["name"];
        const std::string expr_str = node->opts["expr"];
        std::string descr = node->opts["descr"];
        if (descr.empty()) descr = (expr_str.empty()) ? name : expr_str;

        if (expr_str.empty()) {
          const std::string block_name = node->opts["block"];
          const std::string event_str = node->opts["event"];
          const uint32_t event_id = atol(event_str.c_str());

          const hsa_ven_amd_aqlprofile_id_query_t query = Translate(agent_info, block_name);
          const hsa_ven_amd_aqlprofile_block_name_t block_id = (hsa_ven_amd_aqlprofile_block_name_t)query.id;
          if (query.instance_count > 1) {
            for (unsigned block_index = 0; block_index < query.instance_count; ++block_index) {
              std::ostringstream full_name;
              full_name << name << '[' << block_index << ']';
              std::ostringstream alias;
              alias << block_name << "[" << block_index << "]:" << event_str;
              const counter_t counter = {full_name.str(), {block_id, block_index, event_id}};
              AddMetric(full_name.str(), alias.str(), counter);
            }
          } else {
            const std::string alias = block_name + ":" + event_str;
            const counter_t counter = {name, {block_id, 0, event_id}};
            AddMetric(name, alias, counter);
          }
        } else {
          xml::Expr* expr_obj = new xml::Expr(expr_str, new ExprCache(&cache_));
          std::cout << "# " << descr << std::endl;
          std::cout << name << "=" << expr_obj->String() << "\n" << std::endl;
          counters_vec_t counters_vec;
          for (const std::string var : expr_obj->GetVars()) {
            auto it = cache_.find(var);
            if (it == cache_.end())
              EXC_RAISING(HSA_STATUS_ERROR, "Bad metric '" << name << "', var '" << var
                                                           << "' is not found");
            it->second->GetCounters(counters_vec);
          }
          AddMetric(name, counters_vec, expr_obj);
        }
      }
    }
  }

  const Metric* AddMetric(const std::string& name, const std::string& /*alias*/, const counter_t& counter) {
    const Metric* metric = NULL;
    const auto ret = cache_.insert({name, NULL});
    if (ret.second) {
      metric = new BaseMetric(name, counter);
      ret.first->second = metric;
    } else EXC_RAISING(HSA_STATUS_ERROR, "metric redefined '" << name << "'");
#if 0
    if (alias != name) {
      if (cache_.find(alias) != cache_.end()) EXC_RAISING(HSA_STATUS_ERROR, "metric alias/name interference '" << alias << "'");
      const auto ret = aliases_.insert({alias, name});
      if (!ret.second) EXC_RAISING(HSA_STATUS_ERROR, "metric alias redefined '" << alias << "'");
    }
#endif
    return metric;
  }

  const Metric* AddMetric(const std::string& name, const counters_vec_t& counters_vec, const xml::Expr* expr_obj) {
    const Metric* metric = NULL;
    const auto ret = cache_.insert({name, NULL});
    if (ret.second) {
      metric = new ExprMetric(name, counters_vec, expr_obj);
      ret.first->second = metric;
    } else EXC_RAISING(HSA_STATUS_ERROR, "expr-metric redefined '" << name << "'");
    return metric;
  }

#if 0
  std::string UnAlias(const std::string& name) const {
    auto it = aliases_.find(name);
    return (it != aliases_.end()) ? it->second : name;
  }
#endif

  void Print() {
    for (auto& v : cache_) {
      const Metric* metric = v.second;
      counters_vec_t counters_vec;
      printf("> Metric '%s'\n", metric->GetName().c_str());
      metric->GetCounters(counters_vec);
      for (auto c : counters_vec) {
        printf("  counter %s, b(%u), i (%u), e (%u)\n", c->name.c_str(), c->event.block_name, c->event.block_index, c->event.counter_id);
      }
    }
  }

  xml::Xml* xml_;
  const util::AgentInfo* agent_info_;
  cache_t cache_;
#if 0
  std::map<std::string, std::string> aliases_;
#endif

  static map_t* map_;
  static mutex_t mutex_;
};

}  // namespace rocprofiler

#endif  // SRC_CORE_METRICS_H_
