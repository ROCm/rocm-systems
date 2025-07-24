// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "rdc_modules/rdc_rocp/RdcRocpCounterSampler.h"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "rdc_lib/RdcLogger.h"

template <typename Callable>
void RocprofilerCall(Callable&& callable, const std::string& msg, const char* file, int line) {
  auto result = callable();
  if (result != ROCPROFILER_STATUS_SUCCESS) {
    std::string status_msg = rocprofiler_get_status_string(result);
    RDC_LOG(RDC_ERROR, "[CALL][" << file << ":" << line << "] " << msg << " failed with error code "
                                 << result << ": " << status_msg << std::endl);
    std::stringstream errmsg{};
    errmsg << "[CALL][" << file << ":" << line << "] " << msg << " failure (" << status_msg << ")";
    throw std::runtime_error(errmsg.str());
  }
}

namespace amd {
namespace rdc {

std::vector<std::shared_ptr<CounterSampler>> CounterSampler::samplers_;

std::vector<std::shared_ptr<CounterSampler>>& CounterSampler::get_samplers() { return samplers_; }

CounterSampler::CounterSampler(rocprofiler_agent_id_t agent) : agent_(agent) {
  // Setup context (should only be done once per agent)
  RocprofilerCall([&]() { return rocprofiler_create_context(&ctx_); }, "context creation failed",
                  __FILE__, __LINE__);

  RocprofilerCall(
      [&]() {
        return rocprofiler_configure_device_counting_service(
            ctx_, {.handle = 0}, agent,
            [](rocprofiler_context_id_t context_id, rocprofiler_agent_id_t,
               rocprofiler_device_counting_agent_cb_t cb, void* user_data) {
              if (user_data) {
                auto* sampler = static_cast<CounterSampler*>(user_data);
                sampler->set_profile(context_id, cb);
              }
            },
            this);
      },
      "Could not setup buffered service", __FILE__, __LINE__);
}

CounterSampler::~CounterSampler() { ctx_ = {}; }

const std::string& CounterSampler::decode_record_name(
    const rocprofiler_record_counter_t& rec) const {
  static auto roc_counters = [this]() {
    auto name_to_id = CounterSampler::get_supported_counters(agent_);
    std::map<uint64_t, std::string> id_to_name;
    for (const auto& [name, id] : name_to_id) {
      id_to_name.emplace(id.handle, name);
    }
    return id_to_name;
  }();
  rocprofiler_counter_id_t counter_id = {.handle = 0};
  rocprofiler_query_record_counter_id(rec.id, &counter_id);

  auto it = roc_counters.find(counter_id.handle);
  if (it == roc_counters.end()) {
    RDC_LOG(RDC_ERROR, "Error: Counter handle " << counter_id.handle
                                                << " not found in roc_counters." << std::endl);
    throw std::runtime_error("Counter handle not found in roc_counters");
  }

  return it->second;
}

std::unordered_map<std::string, size_t> CounterSampler::get_record_dimensions(
    const rocprofiler_record_counter_t& rec) {
  std::unordered_map<std::string, size_t> out;
  rocprofiler_counter_id_t counter_id = {.handle = 0};
  rocprofiler_query_record_counter_id(rec.id, &counter_id);
  auto dims = get_counter_dimensions(counter_id);

  for (auto& dim : dims) {
    size_t pos = 0;
    rocprofiler_query_record_dimension_position(rec.id, dim.id, &pos);
    out.emplace(dim.name, pos);
  }
  return out;
}

void CounterSampler::sample_counter_values(const std::vector<std::string>& counters,
                                           std::vector<rocprofiler_record_counter_t>& out,
                                           uint64_t duration) {
  auto counter_cached = cached_counter_.find(counters);
  if (counter_cached == cached_counter_.end()) {
    size_t expected_size = 0;
    rocprofiler_counter_config_id_t counter = {};
    std::vector<rocprofiler_counter_id_t> gpu_counters;
    auto roc_counters = get_supported_counters(agent_);
    for (const auto& counter : counters) {
      auto it = roc_counters.find(counter);
      if (it == roc_counters.end()) {
        RDC_LOG(RDC_ERROR, "Counter " << counter << " not found\n");
        continue;
      }
      gpu_counters.push_back(it->second);
      expected_size += get_counter_size(it->second);
    }
    RocprofilerCall(
        [&]() {
          return rocprofiler_create_counter_config(agent_, gpu_counters.data(), gpu_counters.size(),
                                                   &counter);
        },
        "Could not create counter", __FILE__, __LINE__);
    cached_counter_.emplace(counters, counter);
    counter_sizes_.emplace(counter.handle, expected_size);
    counter_cached = cached_counter_.find(counters);
  }

  if (counter_sizes_.find(counter_cached->second.handle) == counter_sizes_.end()) {
    RDC_LOG(RDC_ERROR, "Error: Profile handle " << counter_cached->second.handle
                                                << " not found in counter_sizes_." << std::endl);
    throw std::runtime_error("Profile handle not found in counter_sizes_");
  }
  out.resize(counter_sizes_.at(counter_cached->second.handle));
  counter_ = counter_cached->second;
  rocprofiler_start_context(ctx_);
  size_t out_size = out.size();
  // Wait for sampling window to collect metrics
  usleep(duration);
  rocprofiler_sample_device_counting_service(ctx_, {}, ROCPROFILER_COUNTER_FLAG_NONE, out.data(),
                                             &out_size);
  rocprofiler_stop_context(ctx_);
  out.resize(out_size);
}

std::vector<rocprofiler_agent_v0_t> CounterSampler::get_available_agents() {
  std::vector<rocprofiler_agent_v0_t> agents;
  rocprofiler_query_available_agents_cb_t iterate_cb = [](rocprofiler_agent_version_t agents_ver,
                                                          const void** agents_arr,
                                                          size_t num_agents, void* udata) {
    if (agents_ver != ROCPROFILER_AGENT_INFO_VERSION_0)
      throw std::runtime_error{"unexpected rocprofiler agent version"};
    auto* agents_v = static_cast<std::vector<rocprofiler_agent_v0_t>*>(udata);
    for (size_t i = 0; i < num_agents; ++i) {
      const auto* rocp_agent = static_cast<const rocprofiler_agent_v0_t*>(agents_arr[i]);
      if (rocp_agent->type == ROCPROFILER_AGENT_TYPE_GPU) agents_v->emplace_back(*rocp_agent);
    }
    return ROCPROFILER_STATUS_SUCCESS;
  };

  RocprofilerCall(
      [&]() {
        return rocprofiler_query_available_agents(
            ROCPROFILER_AGENT_INFO_VERSION_0, iterate_cb, sizeof(rocprofiler_agent_t),
            const_cast<void*>(static_cast<const void*>(&agents)));
      },
      "query available agents", __FILE__, __LINE__);
  return agents;
}

void CounterSampler::set_profile(rocprofiler_context_id_t ctx,
                                 rocprofiler_device_counting_agent_cb_t cb) const {
  if (counter_.handle != 0) {
    cb(ctx, counter_);
  }
}

size_t CounterSampler::get_counter_size(rocprofiler_counter_id_t counter) {
  rocprofiler_counter_info_v1_t info;
  rocprofiler_query_counter_info(counter, ROCPROFILER_COUNTER_INFO_VERSION_1,
                                 static_cast<void*>(&info));
  return info.dimensions_instances_count;
}

std::unordered_map<std::string, rocprofiler_counter_id_t> CounterSampler::get_supported_counters(
    rocprofiler_agent_id_t agent) {
  std::unordered_map<std::string, rocprofiler_counter_id_t> out;
  std::vector<rocprofiler_counter_id_t> gpu_counters;

  RocprofilerCall(
      [&]() {
        return rocprofiler_iterate_agent_supported_counters(
            agent,
            [](rocprofiler_agent_id_t, rocprofiler_counter_id_t* counters, size_t num_counters,
               void* user_data) {
              std::vector<rocprofiler_counter_id_t>* vec =
                  static_cast<std::vector<rocprofiler_counter_id_t>*>(user_data);
              for (size_t i = 0; i < num_counters; i++) {
                vec->push_back(counters[i]);
              }
              return ROCPROFILER_STATUS_SUCCESS;
            },
            static_cast<void*>(&gpu_counters));
      },
      "Could not fetch supported counters", __FILE__, __LINE__);
  for (auto& counter : gpu_counters) {
    rocprofiler_counter_info_v0_t version;
    RocprofilerCall(
        [&]() {
          return rocprofiler_query_counter_info(counter, ROCPROFILER_COUNTER_INFO_VERSION_0,
                                                static_cast<void*>(&version));
        },
        "Could not query info for counter", __FILE__, __LINE__);
    out.emplace(version.name, counter);
  }
  return out;
}

std::vector<rocprofiler_counter_record_dimension_info_t> CounterSampler::get_counter_dimensions(
    rocprofiler_counter_id_t counter) {
  rocprofiler_counter_info_v1_t info;
  RocprofilerCall(
      [&]() {
        return rocprofiler_query_counter_info(counter, ROCPROFILER_COUNTER_INFO_VERSION_1,
                                              static_cast<void*>(&info));
      },
      "Could not query info for counter", __FILE__, __LINE__);
  return std::vector<rocprofiler_counter_record_dimension_info_t>{
      *info.dimensions, *info.dimensions + info.dimensions_count};
}

int tool_init(rocprofiler_client_finalize_t, void*) {
  // Get the agents available on the device
  auto agents = CounterSampler::get_available_agents();
  if (agents.empty()) {
    RDC_LOG(RDC_ERROR, "No agents found\n");
    return -1;
  }

  for (auto agent : agents) {
    CounterSampler::get_samplers().push_back(std::make_shared<CounterSampler>(agent.id));
  }

  // no errors
  return 0;
}

void tool_fini(void* user_data) {
  auto* output_stream = static_cast<std::ostream*>(user_data);
  *output_stream << std::flush;
  if (output_stream != &std::cout && output_stream != &std::cerr) delete output_stream;
}

extern "C" rocprofiler_tool_configure_result_t* rocprofiler_configure(uint32_t version,
                                                                      const char* runtime_version,
                                                                      uint32_t priority,
                                                                      rocprofiler_client_id_t* id) {
  // set the client name
  id->name = "CounterClientSample";

  // compute major/minor/patch version info
  uint32_t major = version / 10000;
  uint32_t minor = (version % 10000) / 100;
  uint32_t patch = version % 100;

  // generate info string
  auto info = std::stringstream{};
  info << id->name << " (priority=" << priority << ") is using rocprofiler-sdk v" << major << "."
       << minor << "." << patch << " (" << runtime_version << ")";

  std::clog << info.str() << std::endl;

  std::ostream* output_stream = nullptr;
  std::string filename = "counter_collection.log";
  if (auto* outfile = getenv("ROCPROFILER_SAMPLE_OUTPUT_FILE"); outfile) filename = outfile;
  if (filename == "stdout")
    output_stream = &std::cout;
  else if (filename == "stderr")
    output_stream = &std::cerr;
  else
    output_stream = new std::ofstream{filename};

  // create configure data
  static auto cfg =
      rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t), &tool_init,
                                          &tool_fini, static_cast<void*>(output_stream)};

  // return pointer to configure data
  return &cfg;
}

}  // namespace rdc
}  // namespace amd
