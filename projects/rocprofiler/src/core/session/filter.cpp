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

#include "src/core/session/filter.h"

#include <mutex>

#include "src/utils/helper.h"

namespace rocprofiler {

Filter::Filter(rocprofiler_filter_id_t id, rocprofiler_filter_kind_t filter_kind,
               rocprofiler_filter_data_t filter_data, uint64_t data_count)
    : id_(id), kind_(filter_kind) {
  switch (filter_kind) {
    case ROCPROFILER_DISPATCH_TIMESTAMPS_COLLECTION: {
      break;
    }
    case ROCPROFILER_COUNTERS_COLLECTION: {
      profiler_counter_names_.clear();
      for (uint32_t j = 0; j < data_count; j++) {
        profiler_counter_names_.emplace_back(filter_data.counters_names[j]);
      }
      break;
    }
    case ROCPROFILER_PC_SAMPLING_COLLECTION: {
      break;
    }
    case ROCPROFILER_ATT_TRACE_COLLECTION: {
      att_parameters_.clear();
      profiler_counter_names_.clear();

      for (uint32_t j = 0; j < data_count; j++) {
        if (filter_data.att_parameters[j].parameter_name != ROCPROFILER_ATT_PERFCOUNTER_NAME) {
          att_parameters_.emplace_back(filter_data.att_parameters[j]);
        } else {
          profiler_counter_names_.emplace_back(filter_data.att_parameters[j].counter_name);
        }
      }
      break;
    }
    case ROCPROFILER_SPM_COLLECTION: {
      spm_parameter_ = filter_data.spm_parameters;
      break;
    }
    case ROCPROFILER_API_TRACE: {
      tracer_apis_.clear();
      for (uint32_t j = 0; j < data_count; j++) {
        tracer_apis_.emplace_back(filter_data.trace_apis[j]);
      }
      break;
    }
    case ROCPROFILER_COUNTERS_SAMPLER: {
      counters_sampler_parameters_ = filter_data.counters_sampler_parameters;
      break;
    }
    default: {
      warning(
          "Error: ROCProfiler filter specified is not supported for "
          "profiler mode!\n");
      break;
    }
  }
}

Filter::~Filter() {}

rocprofiler_filter_id_t Filter::GetId() { return id_; }

void Filter::SetBufferId(rocprofiler_buffer_id_t buffer_id) { buffer_id_ = buffer_id; }
rocprofiler_buffer_id_t Filter::GetBufferId() { return buffer_id_; }
bool Filter::HasBuffer() { return (buffer_id_.value > 0); }

rocprofiler_filter_kind_t Filter::GetKind() { return kind_; }

std::mutex counter_data_lock;
std::vector<std::string> Filter::GetCounterData() {
  if (kind_ == ROCPROFILER_COUNTERS_COLLECTION || kind_ == ROCPROFILER_ATT_TRACE_COLLECTION) {
    std::lock_guard<std::mutex> lock(counter_data_lock);
    return profiler_counter_names_;
  }
  fatal(
      "Error: ROCProfiler filter specified is not supported for "
      "Counter Collection Filter!\n");
}

std::vector<rocprofiler_tracer_activity_domain_t> Filter::GetTraceData() {
  if (kind_ == ROCPROFILER_API_TRACE) {
    return tracer_apis_;
  }
  fatal(
      "Error: ROCProfiler filter specified is not supported for "
      "profiler mode!\n");
}

std::vector<rocprofiler_att_parameter_t> Filter::GetAttParametersData() {
  if (kind_ == ROCPROFILER_ATT_TRACE_COLLECTION) {
    return att_parameters_;
  }
  fatal(
      "Error: ROCProfiler filter specified is not supported for "
      "ATT tracing mode!\n");
}

rocprofiler_spm_parameter_t* Filter::GetSpmParameterData() {
  if (kind_ == ROCPROFILER_SPM_COLLECTION) {
    return spm_parameter_;
  }
  fatal(
      "Error: ROCProfiler filter specified is not supported for "
      "SPM collection  mode!\n");
}

rocprofiler_counters_sampler_parameters_t Filter::GetCountersSamplerParameterData() {
  if (kind_ == ROCPROFILER_COUNTERS_SAMPLER) {
    return counters_sampler_parameters_;
  }
  fatal(
      "Error: ROCProfiler filter specified is not supported for "
      "Counters sampler mode!\n");
}

void Filter::SetProperty(rocprofiler_filter_property_t property) {
  switch (property.kind) {
    case ROCPROFILER_FILTER_HSA_TRACER_API_FUNCTIONS: {
      if (kind_ == ROCPROFILER_API_TRACE) {
        hsa_tracer_api_calls_.clear();
        for (uint32_t j = 0; j < property.data_count; j++)
          hsa_tracer_api_calls_.emplace_back(property.hsa_functions_names[j]);
      } else {
        throw(ROCPROFILER_STATUS_ERROR_SESSION_FILTER_DATA_MISMATCH);
      }
      break;
    }
    case ROCPROFILER_FILTER_HIP_TRACER_API_FUNCTIONS: {
      if (kind_ == ROCPROFILER_API_TRACE) {
        hip_tracer_api_calls_.clear();
        for (uint32_t j = 0; j < property.data_count; j++)
          hip_tracer_api_calls_.emplace_back(property.hip_functions_names[j]);
      } else {
        throw(ROCPROFILER_STATUS_ERROR_SESSION_FILTER_DATA_MISMATCH);
      }
      break;
    }
    case ROCPROFILER_FILTER_GPU_NAME: {
      if (kind_ == ROCPROFILER_COUNTERS_COLLECTION ||
          kind_ == ROCPROFILER_DISPATCH_TIMESTAMPS_COLLECTION) {
        agent_names_.clear();
        for (uint32_t j = 0; j < property.data_count; j++)
          agent_names_.emplace_back(property.name_regex[j]);
      } else {
        throw(ROCPROFILER_STATUS_ERROR_SESSION_FILTER_DATA_MISMATCH);
      }
      break;
    }
    case ROCPROFILER_FILTER_RANGE: {
      if (kind_ == ROCPROFILER_COUNTERS_COLLECTION ||
          kind_ == ROCPROFILER_DISPATCH_TIMESTAMPS_COLLECTION) {
        dispatch_range_[0] = property.range[0];
        dispatch_range_[1] = property.range[1];
      } else {
        throw(ROCPROFILER_STATUS_ERROR_SESSION_FILTER_DATA_MISMATCH);
      }
      break;
    }
    case ROCPROFILER_FILTER_KERNEL_NAMES: {
      if (kind_ == ROCPROFILER_COUNTERS_COLLECTION ||
          kind_ == ROCPROFILER_DISPATCH_TIMESTAMPS_COLLECTION ||
          kind_ == ROCPROFILER_ATT_TRACE_COLLECTION) {
        kernel_names_.clear();
        for (uint32_t j = 0; j < property.data_count; j++)
          kernel_names_.emplace_back(property.name_regex[j]);
      } else {
        throw(ROCPROFILER_STATUS_ERROR_SESSION_FILTER_DATA_MISMATCH);
      }
      break;
    }
    case ROCPROFILER_FILTER_DISPATCH_IDS:
      dispatch_id_filter_.clear();
      for (uint32_t j = 0; j < property.data_count; j++)
        dispatch_id_filter_.push_back({property.dispatch_ids[j].start, property.dispatch_ids[j].end});
      break;
    default:
      break;
      // TODO(aelwazir): Check for empty property
      // warning(
      //     "Error: ROCProfiler filter specified is not supported for "
      //     "profiler mode!\n");
  }
}
Filter::filter_property_variant_t Filter::GetProperty(rocprofiler_filter_property_kind_t kind) {
  filter_property_variant_t property;
  switch (kind) {
    case ROCPROFILER_FILTER_GPU_NAME: {
      property = agent_names_;
      break;
    }
    case ROCPROFILER_FILTER_RANGE: {
      property = static_cast<uint32_t*>(dispatch_range_);
      break;
    }
    case ROCPROFILER_FILTER_KERNEL_NAMES: {
      property = kernel_names_;
      break;
    }
    case ROCPROFILER_FILTER_HSA_TRACER_API_FUNCTIONS: {
      property = hsa_tracer_api_calls_;
      break;
    }
    case ROCPROFILER_FILTER_HIP_TRACER_API_FUNCTIONS: {
      property = hip_tracer_api_calls_;
      break;
    }
    case ROCPROFILER_FILTER_DISPATCH_IDS: {
      property = dispatch_id_filter_;
      break;
    }
    default:
      fatal(
          "Error: ROCProfiler filter specified is not supported for the given "
          "kind!");
      break;
  }
  return property;
}

void Filter::SetCallback(rocprofiler_sync_callback_t& callback) {
  callback_ = callback;
  has_sync_callback_ = true;
}

bool Filter::HasCallback() { return has_sync_callback_; }

rocprofiler_sync_callback_t& Filter::GetCallback() { return callback_; }

size_t Filter::GetPropertiesCount(rocprofiler_filter_property_kind_t kind) {
  switch (kind) {
    case ROCPROFILER_FILTER_GPU_NAME: {
      return agent_names_.size();
    }
    case ROCPROFILER_FILTER_RANGE: {
      return 2;
    }
    case ROCPROFILER_FILTER_KERNEL_NAMES: {
      return kernel_names_.size();
    }
    case ROCPROFILER_FILTER_HSA_TRACER_API_FUNCTIONS: {
      return hsa_tracer_api_calls_.size();
    }
    case ROCPROFILER_FILTER_HIP_TRACER_API_FUNCTIONS: {
      return hip_tracer_api_calls_.size();
    }
    case ROCPROFILER_FILTER_DISPATCH_IDS: {
      return dispatch_id_filter_.size();
    }
  }
  fatal(
      "Error: ROCProfiler filter specified is not supported for the given "
      "kind!");
}

}  // namespace rocprofiler
