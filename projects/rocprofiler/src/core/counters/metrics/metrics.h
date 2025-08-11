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

#ifndef SRC_CORE_COUNTERS_METRICS_METRICS_H_
#define SRC_CORE_COUNTERS_METRICS_METRICS_H_

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dlfcn.h>  // for dladdr

#include <fstream>
#include <iostream>
#include <thread>
#include <list>
#include <map>
#include <vector>
#include <mutex>

#include "types.h"
#include "exception.h"
#include "expr.h"
#include "xml.h"
#include <mutex>
#include <unordered_set>
#include "src/core/hardware/hsa_info.h"
#include "src/core/hsa/hsa_support.h"
#include "src/utils/filesystem.hpp"


namespace fs = rocprofiler::common::filesystem;
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
  ~ExprMetric() { delete expr_; }
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

  static MetricsDict* Create(const rocprofiler::HSAAgentInfo* agent_info) {
    std::lock_guard<mutex_t> lck(mutex_);
    if (map_ == NULL) map_ = new map_t;
    std::string name = std::string(agent_info->GetDeviceInfo().getName());
    auto ret = map_->insert({name, NULL});
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

  const Metric* Get(const std::string& name) const {
    const Metric* metric = NULL;

    auto it = cache_.find(name);
    if (it != cache_.end())
      metric = it->second;
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
          if (pos2 == std::string::npos)
            EXC_RAISING(HSA_STATUS_ERROR, "Malformed metric name '" << name << "'");
          block_name = name.substr(0, pos1);
          const std::string block_index_str = name.substr(pos1 + 1, pos2 - (pos1 + 1));
          block_index = atol(block_index_str.c_str());
          indexed = true;
        }

        const hsa_ven_amd_aqlprofile_id_query_t query = Translate(agent_info_, block_name);
        const hsa_ven_amd_aqlprofile_block_name_t block_id =
            (hsa_ven_amd_aqlprofile_block_name_t)query.id;
        if ((query.instance_count > 1) && (indexed == false))
          EXC_RAISING(HSA_STATUS_ERROR, "Malformed indexed metric name '" << name << "'");
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

  std::string GetAgentName() const { return agent_name_; }

  xml::Xml::nodes_t GetNodes() const {
    auto nodes_vec = GetNodes(agent_name_);
    auto global_vec = GetNodes("global");
    nodes_vec.insert(nodes_vec.end(), global_vec.begin(), global_vec.end());
    return nodes_vec;
  }

 private:
  xml::Xml::nodes_t GetNodes(const std::string& scope) const {
    return (xml_ != NULL) ? xml_->GetNodes("top." + scope + ".metric") : xml::Xml::nodes_t();
  }

  MetricsDict(const rocprofiler::HSAAgentInfo* agent_info) : xml_(NULL), agent_info_(agent_info) {
    std::string xml_name = []() {

      if (const char* path = getenv("ROCPROFILER_METRICS_PATH"); path != nullptr) return path;
      return "";
    }();
    if (xml_name.empty()) {
      Dl_info dl_info;
      if (dladdr(reinterpret_cast<const void*>(MetricsDict::Destroy), &dl_info) != 0)
        xml_name = fs::path(dl_info.dli_fname).remove_filename() /
            "../libexec/rocprofiler/counters/derived_counters.xml";
    }
    xml_ = xml::Xml::Create(xml_name);
    if (xml_ == NULL) EXC_RAISING(HSA_STATUS_ERROR, "metrics .xml open error '" << xml_name << "'");
    xml_->AddConst("top.const.metric", "MAX_WAVE_SIZE", agent_info->GetDeviceInfo().getMaxWaveSize());
    xml_->AddConst("top.const.metric", "CU_NUM", agent_info->GetDeviceInfo().getCUCount());
    xml_->AddConst("top.const.metric", "XCC_NUM", agent_info->GetDeviceInfo().getXccCount());
    xml_->AddConst("top.const.metric", "SIMD_NUM",
                   agent_info->GetDeviceInfo().getSimdCountPerCU() * agent_info->GetDeviceInfo().getCUCount());
    xml_->AddConst("top.const.metric", "SE_NUM", agent_info->GetDeviceInfo().getShaderEngineCount());
    xml_->AddConst("top.const.metric", "LDS_BANKS", 32);
    ImportMetrics(agent_info, "const");
    agent_name_ = agent_info->GetDeviceInfo().getName();

    if (agent_name_.find(':') != std::string::npos)  // Remove compiler flags from the agent_name
      agent_name_ = agent_name_.substr(0, agent_name_.find(':'));

    std::unordered_set<std::string> supported_agent_names = {
        "gfx906",  "gfx908", "gfx90a",    // Vega
        "gfx940",  "gfx941", "gfx942",    // Mi300
        "gfx1030", "gfx1031", "gfx1032",  // Navi2x
        "gfx1100", "gfx1101", "gfx1102",  // Navi3x
        "gfx1150", "gfx1151",
        "gfx1200", "gfx1201",             // Navi4x
    };
    if (supported_agent_names.find(agent_name_) != supported_agent_names.end()) {
      ImportMetrics(agent_info, agent_name_);
    } else {
      agent_name_ = agent_info->GetDeviceInfo().getGfxip();
      ImportMetrics(agent_info, agent_name_);
    }
    ImportMetrics(agent_info, "global");
  }

  ~MetricsDict() {
    xml::Xml::Destroy(xml_);
    for (auto& entry : cache_) delete entry.second;
  }

  static hsa_ven_amd_aqlprofile_id_query_t Translate(const rocprofiler::HSAAgentInfo* agent_info,
                                                     const std::string& block_name) {
    hsa_ven_amd_aqlprofile_profile_t profile{};
    profile.agent = hsa_agent_t{agent_info->getHandle()};
    hsa_ven_amd_aqlprofile_id_query_t query = {block_name.c_str(), 0, 0};
    hsa_status_t status =
        hsa_ven_amd_aqlprofile_get_info(&profile, HSA_VEN_AMD_AQLPROFILE_INFO_BLOCK_ID, &query);
    if (status != HSA_STATUS_SUCCESS)
      AQL_EXC_RAISING(HSA_STATUS_ERROR, "ImportMetrics: bad block name '" << block_name << "'");
    return query;
  }

  void ImportMetrics(const rocprofiler::HSAAgentInfo* agent_info, const std::string& scope) {
    auto arr = xml_->GetNodes("top." + scope + ".metric");
    xml::Xml::node_list_t metrics_list(arr.begin(), arr.end());
    uint32_t metrics_number = metrics_list.size();
    bool do_lookup = true;
    if (!metrics_list.empty()) {
      uint32_t it_number = metrics_number;
      auto it = metrics_list.begin();
      auto end = metrics_list.end();
      while (it != end) {
        auto node = *it;
        const std::string name = node->opts["name"];
        const std::string expr_str = node->opts["expr"];
        std::string descr = node->opts["descr"];
        if (descr.empty()) descr = (expr_str.empty()) ? name : expr_str;

        if (expr_str.empty()) {
          const std::string block_name = node->opts["block"];
          const std::string event_str = node->opts["event"];
          const uint32_t event_id = atol(event_str.c_str());

          const hsa_ven_amd_aqlprofile_id_query_t query = Translate(agent_info, block_name);
          const hsa_ven_amd_aqlprofile_block_name_t block_id =
              (hsa_ven_amd_aqlprofile_block_name_t)query.id;
          if (query.instance_count > 1) {
            for (unsigned block_index = 0; block_index < query.instance_count; ++block_index) {
              std::ostringstream full_name;
              full_name << name << '[' << block_index << ']';
              std::ostringstream block_insance;
              block_insance << block_name << "[" << block_index << "]";
              std::ostringstream alias;
              alias << block_insance.str() << ":" << event_str;
              const counter_t counter = {full_name.str(), {block_id, block_index, event_id}};
              AddMetric(full_name.str(), alias.str(), counter);
            }
          } else {
            const std::string alias = block_name + ":" + event_str;
            const counter_t counter = {name, {block_id, 0, event_id}};
            AddMetric(name, alias, counter);
          }
        } else {
          xml::Expr* expr_obj = NULL;
          try {
            expr_obj = new xml::Expr(expr_str, new ExprCache(&cache_));
          } catch (const xml::exception_t& exc) {
            if (do_lookup) {
              metrics_list.push_back(node);
            } else {
              std::cerr << "Error: " << exc.what() << std::endl;
              abort();
            }
          }
          if (expr_obj) {
#if 0
            std::cout << "# " << descr << std::endl;
            std::cout << name << "=" << expr_obj->String() << "\n" << std::endl;
#endif
            counters_vec_t counters_vec;
            for (const std::string& var : expr_obj->GetVars()) {
              auto it = cache_.find(var);
              if (it == cache_.end()) {
                EXC_RAISING(HSA_STATUS_ERROR,
                            "Bad metric '" << name << "', var '" << var << "' is not found");
              }
              it->second->GetCounters(counters_vec);
            }
            AddMetric(name, counters_vec, expr_obj);
          }
        }

        auto cur = it++;
        metrics_list.erase(cur);
        if (--it_number == 0) {
          it_number = metrics_list.size();
          if (it_number < metrics_number) {
            metrics_number = it_number;
          } else if (it_number == metrics_number) {
            do_lookup = false;
          } else {
            EXC_RAISING(HSA_STATUS_ERROR, "Internal error");
          }
        }
      }
    }
  }

  const Metric* AddMetric(const std::string& name, const std::string& /*alias*/,
                          const counter_t& counter) {
    const Metric* metric = NULL;
    const auto ret = cache_.insert({name, NULL});
    if (ret.second) {
      metric = new BaseMetric(name, counter);
      ret.first->second = metric;
    } else
      EXC_RAISING(HSA_STATUS_ERROR, "metric redefined '" << name << "'");
    return metric;
  }

  const Metric* AddMetric(const std::string& name, const counters_vec_t& counters_vec,
                          const xml::Expr* expr_obj) {
    const Metric* metric = NULL;
    const auto ret = cache_.insert({name, NULL});
    if (ret.second) {
      metric = new ExprMetric(name, counters_vec, expr_obj);
      ret.first->second = metric;
    } else
      EXC_RAISING(HSA_STATUS_ERROR, "expr-metric redefined '" << name << "'");
    return metric;
  }

  void Print() {
    for (auto& v : cache_) {
      const Metric* metric = v.second;
      counters_vec_t counters_vec;
      printf("> Metric '%s'\n", metric->GetName().c_str());
      metric->GetCounters(counters_vec);
      for (auto c : counters_vec) {
        printf("  counter %s, b(%u), i (%u), e (%u)\n", c->name.c_str(), c->event.block_name,
               c->event.block_index, c->event.counter_id);
      }
    }
  }

  xml::Xml* xml_;
  const rocprofiler::HSAAgentInfo* agent_info_;
  std::string agent_name_;
  cache_t cache_;

  static map_t* map_;
  static mutex_t mutex_;
};

}  // namespace rocprofiler

#endif  // SRC_CORE_COUNTERS_METRICS_METRICS_H_
