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

#ifndef SRC_CORE_CONTEXT_H_
#define SRC_CORE_CONTEXT_H_

#include "rocprofiler.h"

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <unistd.h>  // usleep
#include <atomic>
#include <list>
#include <map>
#include <mutex>
#include <vector>

#include "core/group_set.h"
#include "core/metrics.h"
#include "core/profile.h"
#include "core/queue.h"
#include "core/types.h"
#include "util/exception.h"
#include "util/hsa_rsrc_factory.h"
#include "util/logger.h"

namespace rocprofiler {
struct rocprofiler_contex_t;
class Context;

inline unsigned align_size(unsigned size, unsigned alignment) {
  return ((size + alignment - 1) & ~(alignment - 1));
}

// Metrics arguments
template <class Map> class MetricArgs : public xml::args_cache_t {
 public:
  MetricArgs(const Map& map) : map_(map) {}
  bool Lookup(const std::string& name, double& result) const {
    rocprofiler_feature_t* info = NULL;
    auto it = map_.find(name);
    if (it == map_.end()) EXC_RAISING(HSA_STATUS_ERROR, "var '" << name << "' is not found");
    info = it->second;
    if (info) {
      result = info->data.result_int64;
      if (info->data.kind == ROCPROFILER_DATA_KIND_UNINIT)
        EXC_RAISING(HSA_STATUS_ERROR, "var '" << name << "' is uninitialized");
      if (info->data.kind != ROCPROFILER_DATA_KIND_INT64)
        EXC_RAISING(HSA_STATUS_ERROR, "var '" << name << "' is of incompatible type, not INT64");
    } else
      EXC_RAISING(HSA_STATUS_ERROR, "var '" << name << "' info is NULL");
    return (info != NULL);
  }

 private:
  const Map& map_;
};

// Profiling group
class Group {
 public:
  typedef uint32_t refs_t;
  typedef std::atomic<refs_t> atomic_refs_t;

  Group(const util::AgentInfo* agent_info, Context* context, const uint32_t& index)
      : pmc_profile_(agent_info),
        trace_profile_(agent_info),
        n_profiles_(0),
        refs_(1),
        context_(context),
        index_(index),
        barrier_signal_{},
        dispatch_signal_{},
        orig_signal_{},
        record_{} {}

  void Insert(const profile_info_t& info) {
    const rocprofiler_feature_kind_t kind = info.rinfo->kind;
    info_vector_.push_back(info.rinfo);
    switch (kind) {
      case ROCPROFILER_FEATURE_KIND_METRIC:
        pmc_profile_.Insert(info);
        break;
      case ROCPROFILER_FEATURE_KIND_TRACE:
        trace_profile_.Insert(info);
        break;
      default:
        EXC_RAISING(HSA_STATUS_ERROR, "bad rocprofiler feature kind (" << kind << ")");
    }
  }

  hsa_status_t Finalize(const bool is_concurrent = false) {
    hsa_status_t status =
        pmc_profile_.Finalize(start_vector_, stop_vector_, read_vector_, is_concurrent);
    if (status == HSA_STATUS_SUCCESS) {
      status = trace_profile_.Finalize(start_vector_, stop_vector_, read_vector_, is_concurrent);
    }
    if (status == HSA_STATUS_SUCCESS) {
      if (!pmc_profile_.Empty()) ++n_profiles_;
      if (!trace_profile_.Empty()) ++n_profiles_;
    }
    return status;
  }

  void GetProfiles(profile_vector_t& vec) {
    pmc_profile_.GetProfiles(vec);
    trace_profile_.GetProfiles(vec);
  }

  void GetTraceProfiles(profile_vector_t& vec) { trace_profile_.GetProfiles(vec); }

  info_vector_t& GetInfoVector() { return info_vector_; }
  const pkt_vector_t& GetStartVector() const { return start_vector_; }
  const pkt_vector_t& GetStopVector() const { return stop_vector_; }
  const pkt_vector_t& GetReadVector() const { return read_vector_; }
  Context* GetContext() { return context_; }
  uint32_t GetIndex() const { return index_; }

  void SetBarrierSignal(const hsa_signal_t& signal) { barrier_signal_ = signal; }
  hsa_signal_t& GetBarrierSignal() { return barrier_signal_; }
  void SetDispatchSignal(const hsa_signal_t& signal) { dispatch_signal_ = signal; }
  hsa_signal_t& GetDispatchSignal() { return dispatch_signal_; }
  void SetOrigSignal(const hsa_signal_t& signal) { orig_signal_ = signal; }
  const hsa_signal_t& GetOrigSignal() const { return orig_signal_; }
  rocprofiler_dispatch_record_t* GetRecord() { return &record_; }

  atomic_refs_t* AtomicRefsCount() { return reinterpret_cast<atomic_refs_t*>(&refs_); }
  void ResetRefsCount() { AtomicRefsCount()->store(n_profiles_, std::memory_order_release); }
  void IncrRefsCount() { AtomicRefsCount()->fetch_add(1, std::memory_order_acq_rel); }
  uint32_t FetchDecrRefsCount() {
    return AtomicRefsCount()->fetch_sub(1, std::memory_order_acq_rel);
  }

 private:
  PmcProfile pmc_profile_;
  TraceProfile trace_profile_;
  info_vector_t info_vector_;
  pkt_vector_t start_vector_;
  pkt_vector_t stop_vector_;
  pkt_vector_t read_vector_;
  uint32_t n_profiles_;
  refs_t refs_;
  Context* const context_;
  const uint32_t index_;
  // completion signal of after-dispatch barrier
  hsa_signal_t barrier_signal_;
  // completion signal kernel packet dispatch
  hsa_signal_t dispatch_signal_;
  hsa_signal_t orig_signal_;
  rocprofiler_dispatch_record_t record_;
};

// Profiling context
class Context {
 public:
  typedef std::map<std::string, rocprofiler_feature_t*> info_map_t;

  static void Create(Context* obj, const util::AgentInfo* agent_info, Queue* queue,
                     rocprofiler_feature_t* info, const uint32_t info_count,
                     rocprofiler_handler_t handler, void* handler_arg) {
    new (obj) Context(agent_info, queue, info, info_count, handler, handler_arg);
    obj->Construct(agent_info, queue, info, info_count, handler, handler_arg);
  }

  static void Release(Context* obj) { obj->Destruct(); }

  static Context* Create(const util::AgentInfo* agent_info, Queue* queue,
                         rocprofiler_feature_t* info, const uint32_t info_count,
                         rocprofiler_handler_t handler, void* handler_arg) {
    Context* obj = new Context(agent_info, queue, info, info_count, handler, handler_arg);
    if (obj == NULL) EXC_RAISING(HSA_STATUS_ERROR, "allocation error");
    try {
      obj->Construct(agent_info, queue, info, info_count, handler, handler_arg);
    } catch (std::exception& e) {
      std::cerr << e.what() << std::endl;
      delete obj;
      obj = NULL;
      std::cerr << "Error: Context Create failed" << std::endl;
      abort();
    }
    return obj;
  }

  static void Destroy(Context* obj) {
    if (obj != NULL) delete obj;
  }

  void Reset(const uint32_t& group_index) { set_[group_index].ResetRefsCount(); }

  uint32_t GetGroupCount() const { return set_.size(); }

  inline rocprofiler_group_t GetGroupDescr(Group* g) {
    rocprofiler::info_vector_t& info_vector = g->GetInfoVector();
    rocprofiler_group_t group = {};
    group.index = g->GetIndex();
    group.context = reinterpret_cast<rocprofiler_t*>(this);
    group.features = &info_vector[0];
    group.feature_count = info_vector.size();
    return group;
  }
  inline rocprofiler_group_t GetGroupDescr(const uint32_t& index) {
    rocprofiler_group_t group = {};
    if (set_.empty()) {
      group.context = reinterpret_cast<rocprofiler_t*>(this);
    } else {
      group = GetGroupDescr(&set_[index]);
    }
    return group;
  }

  const pkt_vector_t& StartPackets(const uint32_t& group_index) const {
    return set_[group_index].GetStartVector();
  }
  const pkt_vector_t& StopPackets(const uint32_t& group_index) const {
    return set_[group_index].GetStopVector();
  }
  const pkt_vector_t& ReadPackets(const uint32_t& group_index) const {
    return set_[group_index].GetReadVector();
  }

  void Start(const uint32_t& group_index, Queue* const queue = NULL) {
    const pkt_vector_t& start_packets = StartPackets(group_index);
    Queue* const submit_queue = (queue != NULL) ? queue : queue_;
    submit_queue->Submit(&start_packets[0], start_packets.size());
  }
  void Stop(const uint32_t& group_index, Queue* const queue = NULL) {
    const pkt_vector_t& stop_packets = StopPackets(group_index);
    Queue* const submit_queue = (queue != NULL) ? queue : queue_;
    submit_queue->Submit(&stop_packets[0], stop_packets.size());
  }
  void Read(const uint32_t& group_index, Queue* const queue = NULL) {
    const pkt_vector_t& read_packets = ReadPackets(group_index);
    if (read_packets.size() == 0) EXC_RAISING(HSA_STATUS_ERROR, "Read API disabled");
    Queue* const submit_queue = (queue != NULL) ? queue : queue_;
    submit_queue->Submit(&read_packets[0], read_packets.size());
  }
  void Submit(const uint32_t& group_index, const packet_t* packet, Queue* const queue = NULL) {
    Queue* const submit_queue = (queue != NULL) ? queue : queue_;
    Start(group_index, submit_queue);
    submit_queue->Submit(packet);
    Stop(group_index, submit_queue);
  }

  struct callback_data_t {
    const profile_t* profile;
    info_vector_t* info_vector;
    size_t index;
    char* ptr;
    size_t single_xcc_buff_size;
    size_t cb_invocation_count; 
  };

  void RestoreSignals(const profile_tuple_t& tuple) {
    hsa_rsrc_->HsaApi()->hsa_signal_store_screlease(tuple.dispatch_signal, 1);
    if (k_concurrent_) {
      hsa_rsrc_->HsaApi()->hsa_signal_store_screlease(tuple.read_signal, 1);
      hsa_rsrc_->HsaApi()->hsa_signal_store_screlease(tuple.barrier_signal, 1);
    }
  }

  void GetData(const uint32_t& group_index) {
    const profile_vector_t profile_vector = GetProfiles(group_index);
    for (auto& tuple : profile_vector) {
      // Wait for stop packet to complete
      hsa_rsrc_->SignalWaitRestore(tuple.completion_signal, 1);
      // Restore other signals
      RestoreSignals(tuple);
      for (rocprofiler_feature_t* rinfo : *(tuple.info_vector)){
        rinfo->data.kind = ROCPROFILER_DATA_KIND_UNINIT;
        rinfo->data.result_int64 = 0;
      }
      size_t xcc_count = agent_info_->xcc_num;
      size_t single_xcc_buff_size = tuple.profile->output_buffer.size / (sizeof(uint64_t) * xcc_count);
      callback_data_t callback_data{tuple.profile, tuple.info_vector, tuple.info_vector->size(),
                                    NULL, single_xcc_buff_size, 0};
      const hsa_status_t status =
          api_->hsa_ven_amd_aqlprofile_iterate_data(tuple.profile, DataCallback, &callback_data);
      if (status != HSA_STATUS_SUCCESS) AQL_EXC_RAISING(status, "context iterate data failed");
    }
  }

  void GetMetricsData() const {
    const MetricArgs<info_map_t> args(info_map_);
    for (const auto& v : metrics_map_) {
      const std::string& name = v.first;
      const Metric* metric = v.second;
      const xml::Expr* expr = metric->GetExpr();
      if (expr) {
        auto it = info_map_.find(name);
        if (it == info_map_.end())
          EXC_RAISING(HSA_STATUS_ERROR,
                      "metric '" << name << "', rocprofiler info is not found " << this);
        rocprofiler_feature_t* info = it->second;
        info->data.result_double = expr->Eval(args);
        info->data.kind = ROCPROFILER_DATA_KIND_DOUBLE;
      }
    }
  }

  void IterateTraceData(rocprofiler_trace_data_callback_t callback, void* data) {
    profile_vector_t profile_vector;
    set_[0].GetTraceProfiles(profile_vector);
    for (auto& tuple : profile_vector) {
      if (pcsmp_mode_) const_cast<profile_t*>(tuple.profile)->event_count = UINT32_MAX;
      const hsa_status_t status =
          api_->hsa_ven_amd_aqlprofile_iterate_data(tuple.profile, callback, data);
      if (status != HSA_STATUS_SUCCESS) AQL_EXC_RAISING(status, "context iterate data failed");
    }
  }

  static bool Handler(hsa_signal_value_t value, void* arg) {
    Group* group = reinterpret_cast<Group*>(arg);
    Context* context = group->GetContext();
    auto r = group->FetchDecrRefsCount();
    if (r == 1) {
      const rocprofiler_group_t group_descr = context->GetGroupDescr(group);
      context->handler_(group_descr, context->handler_arg_);
    }
    return false;
  }

  hsa_agent_t GetAgent() const { return agent_; }
  Group* GetGroup(const uint32_t& index) { return &set_[index]; }
  rocprofiler_handler_t GetHandler(void** arg) const {
    *arg = handler_arg_;
    return handler_;
  }

  // Concurrent profiling mode
  static bool k_concurrent_;

 private:
  Context(const util::AgentInfo* agent_info, Queue* queue, rocprofiler_feature_t* info,
          const uint32_t info_count, rocprofiler_handler_t handler, void* handler_arg)
      : agent_(agent_info->dev_id),
        agent_info_(agent_info),
        queue_(queue),
        hsa_rsrc_(&util::HsaRsrcFactory::Instance()),
        api_(hsa_rsrc_->AqlProfileApi()),
        metrics_(NULL),
        handler_(handler),
        handler_arg_(handler_arg),
        pcsmp_mode_(false) {}

  ~Context() { Destruct(); }

  void Destruct() {
    for (const auto& v : info_map_) {
      const std::string& name = v.first;
      const rocprofiler_feature_t* info = v.second;
      if ((info->kind == ROCPROFILER_FEATURE_KIND_METRIC) &&
          (metrics_map_.find(name) == metrics_map_.end())) {
        delete info;
      }
    }
  }

  void Construct(const util::AgentInfo* agent_info, Queue* queue, rocprofiler_feature_t* info,
                 const uint32_t info_count, rocprofiler_handler_t handler, void* handler_arg) {
    if (info_count == 0) {
      set_.push_back(Group(agent_info_, this, 0));
      return;
    }

    metrics_ = MetricsDict::Create(agent_info);
    if (metrics_ == NULL) EXC_RAISING(HSA_STATUS_ERROR, "MetricsDict create failed");

    if (Initialize(info, info_count) == false) {
      fprintf(stdout, "\nInput metrics out of HW limit. Proposed metrics group set:\n");
      fflush(stdout);
      MetricsGroupSet(agent_info, info, info_count).Print(stdout);
      fprintf(stdout, "\n");
      fflush(stdout);
      EXC_RAISING(HSA_STATUS_ERROR, "Metrics list exceeds HW limits");
    }
    Finalize();

    if (handler != NULL) {
      for (unsigned group_index = 0; group_index < set_.size(); ++group_index) {
        set_[group_index].ResetRefsCount();
        const profile_vector_t profile_vector = GetProfiles(group_index);
        for (auto& tuple : profile_vector) {
          set_[group_index].SetDispatchSignal(tuple.dispatch_signal);
          set_[group_index].SetBarrierSignal(tuple.barrier_signal);
          // Handler for stop packet completion
          hsa_amd_signal_async_handler(tuple.completion_signal, HSA_SIGNAL_CONDITION_LT, 1, Handler,
                                       &set_[group_index]);
        }
      }
    }
  }

  // Initialize rocprofiler context
  bool Initialize(rocprofiler_feature_t* info_array, const uint32_t info_count) {
    // Register input features to not duplicate by features referencing
    for (unsigned i = 0; i < info_count; ++i) {
      rocprofiler_feature_t* info = &info_array[i];
      const rocprofiler_feature_kind_t kind = info->kind;
      const char* name = info->name;
      if (kind == ROCPROFILER_FEATURE_KIND_METRIC) {
        if (name == NULL) EXC_RAISING(HSA_STATUS_ERROR, "metric name is NULL");
        info_map_[name] = info;
        auto ret = metrics_map_.insert({name, NULL});
        if (!ret.second)
          EXC_RAISING(HSA_STATUS_ERROR,
                      "input metric '" << name << "' is registered more then once");
      }
    }

    // Adding zero group, always present
    if (info_count) set_.push_back(Group(agent_info_, this, 0));

    // Processing input features
    for (unsigned i = 0; i < info_count; ++i) {
      rocprofiler_feature_t* info = &info_array[i];
      const rocprofiler_feature_kind_t kind = info->kind;
      const char* name = info->name;

      if (kind == ROCPROFILER_FEATURE_KIND_METRIC) {  // Processing metrics features
        const Metric* metric = metrics_->Get(name);
        if (metric == NULL)
          EXC_RAISING(HSA_STATUS_ERROR,
                      "input metric '"
                          << name << "' is not supported on this hardware: " << agent_info_->name);
#if 0
        std::cout << "    " << name << (metric->GetExpr() ? " = " + metric->GetExpr()->String() : " counter") << std::endl;
#endif

        metrics_map_[name] = metric;
        counters_vec_t counters_vec = metric->GetCounters();
        if (counters_vec.empty())
          EXC_RAISING(HSA_STATUS_ERROR, "bad metric '" << name << "' is empty");

        for (const counter_t* counter : counters_vec) {
          // For metrics expressions checking that there is no the same counter in the input metrics
          // and also that the counter wasn't registered already by another input metric expression
          if (metric->GetExpr()) {
            if (info_map_.find(counter->name) != info_map_.end()) {
              continue;
            } else {
              info = NewCounterInfo(counter);
              info_map_[info->name] = info;
            }
          }

          const event_t* event = &(counter->event);
          const block_des_t block_des = {event->block_name, event->block_index};
          auto ret = groups_map_.insert({block_des, {}});
          block_status_t& block_status = ret.first->second;
          if (block_status.max_counters == 0) {
            profile_t query = {};
            query.agent = agent_;
            query.type = HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_PMC;
            query.events = event;

            uint32_t block_counters;
            hsa_status_t status = api_->hsa_ven_amd_aqlprofile_get_info(
                &query, HSA_VEN_AMD_AQLPROFILE_INFO_BLOCK_COUNTERS, &block_counters);
            if (status != HSA_STATUS_SUCCESS) AQL_EXC_RAISING(status, "get block_counters info");
            block_status.max_counters = block_counters;
          }
          if (block_status.counter_index >= block_status.max_counters) {
            return false;

            block_status.counter_index = 0;
            block_status.group_index += 1;
          }
          block_status.counter_index += 1;
          if (block_status.group_index >= set_.size()) {
            set_.push_back(Group(agent_info_, this, block_status.group_index));
          }
          const uint32_t group_index = block_status.group_index;
          set_[group_index].Insert(profile_info_t{event, NULL, 0, info});
        }
      } else if (kind & ROCPROFILER_FEATURE_KIND_TRACE) {  // Processing traces features
        info->kind = ROCPROFILER_FEATURE_KIND_TRACE;

        const event_t* event = NULL;
        if (kind & ROCPROFILER_FEATURE_KIND_PCSMP_MOD) {  // PC sampling
          pcsmp_mode_ = true;
        } else if (kind & ROCPROFILER_FEATURE_KIND_SPM_MOD) {  // SPM trace
          const Metric* metric = metrics_->Get(name);
          if (metric == NULL)
            EXC_RAISING(HSA_STATUS_ERROR, "input metric '" << name << "' is not found");
          counters_vec_t counters_vec = metric->GetCounters();
          if (counters_vec.size() != 1)
            EXC_RAISING(HSA_STATUS_ERROR, "trace bad metric '" << name << "' is not base counter");
          const counter_t* counter = counters_vec[0];
          event = &(counter->event);
        }
        set_[0].Insert(profile_info_t{event, info->parameters, info->parameter_count, info});
      } else {
        EXC_RAISING(HSA_STATUS_ERROR, "bad rocprofiler feature kind (" << kind << ")");
      }
    }

    return true;
  }

  void Finalize() {
    for (unsigned index = 0; index < set_.size(); ++index) {
      const hsa_status_t status = set_[index].Finalize(k_concurrent_);
      if (status != HSA_STATUS_SUCCESS) EXC_RAISING(status, "context finalize failed");
    }
  }

  // Getting profling packets
  profile_vector_t GetProfiles(const uint32_t& index) {
    profile_vector_t vec;
    if (index >= set_.size()) {
      EXC_RAISING(HSA_STATUS_ERROR, "index exceeding the maximum " << set_.size());
    }
    set_[index].GetProfiles(vec);
    return vec;
  }

  static hsa_status_t DataCallback(hsa_ven_amd_aqlprofile_info_type_t ainfo_type,
                                   hsa_ven_amd_aqlprofile_info_data_t* ainfo_data, void* data) {
    hsa_status_t status = HSA_STATUS_SUCCESS;
    callback_data_t* callback_data = reinterpret_cast<callback_data_t*>(data);
    const profile_t* profile = callback_data->profile;
    info_vector_t& info_vector = *(callback_data->info_vector);
    uint32_t index = callback_data->index;
    const uint32_t sample_id = ainfo_data->sample_id;
    if (info_vector.size() == index) {
      index = 0;
    } else {
      if (sample_id == 0) index += 1;
    }
    if(callback_data->cb_invocation_count++ % callback_data->single_xcc_buff_size == 0)
      index = 0;
    callback_data->index = index;

    if (index < info_vector.size()) {
      rocprofiler_feature_t* const rinfo = info_vector[index];
      rinfo->data.kind = ROCPROFILER_DATA_KIND_UNINIT;

      if (ainfo_type == HSA_VEN_AMD_AQLPROFILE_INFO_PMC_DATA) {
        rinfo->data.result_int64 += ainfo_data->pmc_data.result;
        rinfo->data.kind = ROCPROFILER_DATA_KIND_INT64;
      } else if (ainfo_type == HSA_VEN_AMD_AQLPROFILE_INFO_TRACE_DATA) {
        if (rinfo->data.result_bytes.copy) {
          const bool trace_local = TraceProfile::IsLocal();
          util::HsaRsrcFactory* hsa_rsrc = &util::HsaRsrcFactory::Instance();
          if (sample_id == 0) {
            const uint32_t output_buffer_size = profile->output_buffer.size;
            const uint32_t output_buffer_size64 = profile->output_buffer.size / sizeof(uint64_t);
            const util::AgentInfo* agent_info = hsa_rsrc->GetAgentInfo(profile->agent);
            void* ptr = (trace_local) ? hsa_rsrc->AllocateSysMemory(agent_info, output_buffer_size)
                                      : calloc(output_buffer_size64, sizeof(uint64_t));
            rinfo->data.result_bytes.size = output_buffer_size;
            rinfo->data.result_bytes.ptr = ptr;
            callback_data->ptr = reinterpret_cast<char*>(ptr);
          }
          char* result_bytes_ptr = reinterpret_cast<char*>(rinfo->data.result_bytes.ptr);
          const char* end = result_bytes_ptr + rinfo->data.result_bytes.size;
          const char* src = reinterpret_cast<char*>(ainfo_data->trace_data.ptr);
          uint32_t size = ainfo_data->trace_data.size;
          char* ptr = callback_data->ptr;
          uint32_t* header = reinterpret_cast<uint32_t*>(ptr);
          char* dest = ptr + sizeof(*header);

          if ((dest + size) >= end) {
            if (dest < end)
              size = end - dest;
            else
              EXC_RAISING(HSA_STATUS_ERROR, "Trace data out of output buffer");
          }

          bool suc = true;
          if (trace_local) {
            suc = hsa_rsrc->Memcpy(profile->agent, dest, src, size);
          } else {
            memcpy(dest, src, size);
          }
          if (suc) {
            *header = size;
            callback_data->ptr = dest + align_size(size, sizeof(uint32_t));
            rinfo->data.result_bytes.instance_count = sample_id + 1;
            rinfo->data.kind = ROCPROFILER_DATA_KIND_BYTES;
          } else
            EXC_RAISING(HSA_STATUS_ERROR,
                        "Agent Memcpy failed, dst(" << (void*)dest << ") src(" << (void*)src
                                                    << ") size(" << size << ")");
        } else {
          if (sample_id == 0) {
            rinfo->data.result_bytes.ptr = profile->output_buffer.ptr;
            rinfo->data.result_bytes.size = profile->output_buffer.size;
            rinfo->data.result_bytes.instance_count = UINT32_MAX;
          }

          rinfo->data.result_bytes.instance_count += 1;
          rinfo->data.kind = ROCPROFILER_DATA_KIND_BYTES;
        }
      } else {
        EXC_RAISING(HSA_STATUS_ERROR, "unknown data type = " << ainfo_type);
      }
    } else
      status = HSA_STATUS_ERROR;

    return status;
  }

  rocprofiler_feature_t* NewCounterInfo(const counter_t* counter) {
    rocprofiler_feature_t* info = new rocprofiler_feature_t{};
    info->kind = ROCPROFILER_FEATURE_KIND_METRIC;
    info->name = counter->name.c_str();
    return info;
  }

  // GPU handel
  const hsa_agent_t agent_;
  const util::AgentInfo* agent_info_;
  // Profiling queue
  Queue* queue_;
  // HSA resources factory
  util::HsaRsrcFactory* hsa_rsrc_;
  // aqlprofile API table
  const pfn_t* api_;
  // Profile group set
  std::vector<Group> set_;
  // Metrics dictionary
  const MetricsDict* metrics_;
  // Groups map
  std::map<block_des_t, block_status_t, lt_block_des> groups_map_;
  // Info map
  info_map_t info_map_;
  // Metrics map
  std::map<std::string, const Metric*> metrics_map_;
  // Context completion handler
  rocprofiler_handler_t handler_;
  void* handler_arg_;

  // PC sampling mode
  bool pcsmp_mode_;
};

#define CONTEXT_INSTANTIATE() bool rocprofiler::Context::k_concurrent_ = false;

}  // namespace rocprofiler

#endif  // SRC_CORE_CONTEXT_H_
