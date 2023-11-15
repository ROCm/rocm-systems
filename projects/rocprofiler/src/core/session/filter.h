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

#ifndef SRC_CORE_SESSION_FILTER_H_
#define SRC_CORE_SESSION_FILTER_H_

#include <string>
#include <variant>
#include <vector>

#include "rocprofiler.h"

#define ASSERTM(exp, msg) assert(((void)msg, exp))

namespace rocprofiler {

class Filter {
 public:
  typedef std::variant<
    std::vector<std::string>,
    uint32_t*,
    std::vector<std::pair<uint64_t,uint64_t>>
  > filter_property_variant_t;
  Filter(rocprofiler_filter_id_t id, rocprofiler_filter_kind_t filter_kind,
         rocprofiler_filter_data_t filter_data, uint64_t data_count);
  ~Filter();

  rocprofiler_filter_id_t GetId();

  void SetBufferId(rocprofiler_buffer_id_t buffer_id);
  rocprofiler_buffer_id_t GetBufferId();
  bool HasBuffer();

  rocprofiler_filter_kind_t GetKind();

  std::vector<std::string> GetCounterData();
  std::vector<rocprofiler_tracer_activity_domain_t> GetTraceData();
  std::vector<rocprofiler_att_parameter_t> GetAttParametersData();
  void SetCallback(rocprofiler_sync_callback_t& callback);
  rocprofiler_sync_callback_t& GetCallback();
  bool HasCallback();

  void SetProperty(rocprofiler_filter_property_t property);
  filter_property_variant_t GetProperty(rocprofiler_filter_property_kind_t kind);

  size_t GetPropertiesCount(rocprofiler_filter_property_kind_t kind);
  rocprofiler_spm_parameter_t* GetSpmParameterData();
  rocprofiler_counters_sampler_parameters_t GetCountersSamplerParameterData();

 private:
  rocprofiler_filter_id_t id_;
  rocprofiler_filter_kind_t kind_;
  rocprofiler_buffer_id_t buffer_id_{0};

  std::vector<std::string> agent_names_;           // GPU name filter
  std::vector<std::string> hsa_tracer_api_calls_;  // HSA API Functions
  std::vector<std::string> hip_tracer_api_calls_;  // HIP API Functions
  std::vector<std::string> kernel_names_;          // HIP/HSA API Functions
  uint32_t dispatch_range_[2];                     // Kernel Dispatches OR API Range

  std::vector<std::string> profiler_counter_names_;                // Counter Names to collect
  std::vector<rocprofiler_tracer_activity_domain_t> tracer_apis_;  // ROCTX/HIP/HSA API
  rocprofiler_spm_parameter_t* spm_parameter_;                     // spm parameter
  std::vector<rocprofiler_att_parameter_t> att_parameters_;        // ATT Parameters
  rocprofiler_counters_sampler_parameters_t
      counters_sampler_parameters_;  // sampled counters parameters
  std::vector<std::pair<uint64_t,uint64_t>> dispatch_id_filter_;

  bool has_sync_callback_{false};
  rocprofiler_sync_callback_t callback_;
};

}  // namespace rocprofiler

#endif  // SRC_CORE_SESSION_FILTER_H_
