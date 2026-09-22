/* Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

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
 THE SOFTWARE. */

#include "rocprofiler.h"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <iostream>
#include <mutex>
#include <string_view>
#include <utility>
#include <thread>
#include <unordered_set>

#include <cxxabi.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include "json/include/nlohmann/json.hpp"
#include "rocprofiler_plugin.h"
#include "../utils.h"

#include "src/utils/filesystem.hpp"

namespace fs = rocprofiler::common::filesystem;


namespace {

struct TraceCategoryArgs {
  std::string name;
};

struct TraceActivity {
  std::string phase;
  uint64_t category;
  std::string name;
  std::string category_str;
  std::string timestamp;
  std::string duration;
  uint64_t thread_id;
  uint64_t correlation_id;
};

struct TraceCategory {
  TraceCategoryArgs args;
  std::string phase;
  uint64_t category;
  int sort_index;
};

enum TraceFlowCategory {
  ROCPROFILER_DATA_FLOW_START = 0,
  ROCPROFILER_DATA_FLOW_END = 1,
};

class TraceFlow {
 public:
  TraceFlow(uint64_t timestamp, std::string category_str, uint64_t category, uint64_t thread_id,
            uint64_t id);
  uint64_t getTimestamp();
  std::string getCategoryStr();
  uint64_t getID();
  uint64_t getCategory();
  uint64_t getThreadID();
  void setTimestamp(uint64_t timestamp);
  void setID(uint64_t id);

 private:
  uint64_t timestamp_ = 0;
  std::string category_str_;
  uint64_t id_;
  uint64_t category_;
  uint64_t thread_id_;
};

TraceFlow::TraceFlow(uint64_t timestamp, std::string category_str, uint64_t category,
                     uint64_t thread_id, uint64_t id)
    : timestamp_(timestamp),
      category_str_(std::move(category_str)),
      category_(category),
      thread_id_(thread_id),
      id_(id) {}

uint64_t TraceFlow::getTimestamp() { return timestamp_; }
std::string TraceFlow::getCategoryStr() { return category_str_; }
uint64_t TraceFlow::getID() { return id_; }
uint64_t TraceFlow::getCategory() { return category_; }
uint64_t TraceFlow::getThreadID() { return thread_id_; }
void TraceFlow::setTimestamp(uint64_t timestamp) { timestamp_ = timestamp; }
void TraceFlow::setID(uint64_t id) { id_ = id; }

std::mutex writing_lock{};

std::string process_name;

std::string get_kernel_name(rocprofiler_record_profiler_t& profiler_record) {
  std::string kernel_name = "";
  size_t name_length = 1;
  CHECK_ROCPROFILER(rocprofiler_query_kernel_info_size(ROCPROFILER_KERNEL_NAME,
                                                       profiler_record.kernel_id, &name_length));
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#pragma GCC diagnostic ignored "-Wstringop-overread"
  if (name_length > 1) {
    const char* kernel_name_c = nullptr;
    CHECK_ROCPROFILER(rocprofiler_query_kernel_info(ROCPROFILER_KERNEL_NAME,
                                                    profiler_record.kernel_id, &kernel_name_c));
    if (kernel_name_c && strlen(kernel_name_c) > 1)
      kernel_name = rocprofiler::truncate_name(rocprofiler::cxx_demangle(kernel_name_c));
  }
#pragma GCC diagnostic pop
  return kernel_name;
}

class json_plugin_t {
 public:
  json_plugin_t() {
    is_valid_ = true;

    const char* rocprofiler_trace_period = getenv("ROCPROFILER_TRACE_PERIOD");
    if (rocprofiler_trace_period) trace_period_enabled_ = true;

    const char* enable_data_flow_str = getenv("ROCPROFILER_DISABLE_JSON_DATA_FLOWS");
    if (enable_data_flow_str) {
      if (std::string_view(enable_data_flow_str).find("ON") != std::string::npos) {
        enable_data_flow_ = false;
      }
      if (std::string_view(enable_data_flow_str).find("1") != std::string::npos) {
        enable_data_flow_ = false;
      }
      if (std::string_view(enable_data_flow_str).find("OFF") != std::string::npos) {
        enable_data_flow_ = true;
      }
      if (std::string_view(enable_data_flow_str).find("0") != std::string::npos) {
        enable_data_flow_ = true;
      }
    }

    const char* output_dir = getenv("OUTPUT_PATH");
    const char* temp_file_name = getenv("OUT_FILE_NAME");
    std::string output_file_name = temp_file_name ? std::string(temp_file_name) + "_" : "";

    if (output_dir == nullptr) output_dir = "./";

    output_prefix_ = output_dir;
    if (!fs::is_directory(fs::status(output_prefix_))) {
      if (!stream_.fail()) rocprofiler::warning("Cannot open output directory '%s'", output_dir);
      stream_.setstate(std::ios_base::failbit);
      return;
    }
    txt_output_prefix_ = output_prefix_;
    output_prefix_.append(output_file_name + std::to_string(GetPid()) + "_output.json");
    txt_output_prefix_.append(output_file_name + std::to_string(GetPid()) + "_output");
    // file.open(output_prefix_.string());

    // This flush technique won't work we need to join JSON files
    // flush_thread_check_.exchange(true, std::memory_order_acquire);

    // char* flush_interval = getenv("ROCPROFILER_FLUSH_INTERVAL");
    // uint64_t rocprofiler_flush_interval = 10;
    // char* end;
    // if (flush_interval) rocprofiler_flush_interval = std::stoi(flush_interval);

    // flush_thread_ = std::thread(
    //     [&](uint64_t rocprofiler_flush_interval) {
    //       while (flush_thread_check_.load(std::memory_order_acquire)) {
    //         writing_lock.lock();
    //         ExportTraceEventsToJSON();
    //         writing_lock.unlock();
    //         usleep(rocprofiler_flush_interval);
    //       }
    //     },
    //     rocprofiler_flush_interval);
  }

  void delete_json_plugin() {
    if (is_valid_) {
      // flush_thread_check_.exchange(false, std::memory_order_acquire);
      // flush_thread_.join();
      ExportTraceEventsToJSON(output_prefix_.string());
      const char* flame_graph_env = getenv("ROCPROFILER_ENABLE_FLAME_GRAPH");
      if (flame_graph_env &&
          (std::string_view(flame_graph_env).find("1") != std::string::npos ||
           std::string_view(flame_graph_env).find("ON") != std::string::npos))
        ExportTraceEventsForFlameGraph(txt_output_prefix_.string());
      // if (file.is_open()) {
      //   file.close();
      // }
    }
  }

  void ExportTraceEventsToJSON(const std::string& path) {
    nlohmann::json j;
    j["traceEvents"] = nlohmann::json::array();

    if (!trace_categories_check.load(std::memory_order_acquire)) {
      for (const auto& event : trace_categories) {
        nlohmann::json args;
        args["name"] = event.args.name;
        j["traceEvents"].push_back({{"args", args},
                                    {"ph", event.phase},
                                    {"pid", event.category},
                                    {"name", "process_name"},
                                    {"sort_index", event.sort_index}});
      }
      trace_categories_check.exchange(true, std::memory_order_acquire);
    }

    std::sort(trace_copy_activities_.begin(), trace_copy_activities_.end(),
              [](auto& a, auto& b) { return (a.timestamp < b.timestamp); });

    std::sort(trace_events_.begin(), trace_events_.end(),
              [](auto& a, auto& b) { return (a.timestamp < b.timestamp); });

    std::sort(trace_gpu_activities_.begin(), trace_gpu_activities_.end(),
              [](auto& a, auto& b) { return (a.timestamp < b.timestamp); });

    std::sort(trace_unknown_activities_.begin(), trace_unknown_activities_.end(),
              [](auto& a, auto& b) { return (a.timestamp < b.timestamp); });

    for (const auto& event : trace_copy_activities_) {
      nlohmann::json args;
      args["cid"] = event.correlation_id;
      j["traceEvents"].push_back({{"name", event.name},
                                  {"ph", event.phase},
                                  {"ts", event.timestamp},
                                  {"tid", event.thread_id},
                                  {"pid", event.category},
                                  {"dur", event.duration},
                                  {"cat", event.category_str},
                                  {"args", args}});
    }

    for (const auto& event : trace_unknown_activities_) {
      nlohmann::json args;
      args["cid"] = event.correlation_id;
      j["traceEvents"].push_back({{"name", event.name},
                                  {"ph", event.phase},
                                  {"ts", event.timestamp},
                                  {"tid", event.thread_id},
                                  {"pid", event.category},
                                  {"dur", event.duration},
                                  {"cat", event.category_str},
                                  {"args", args}});
    }

    for (const auto& event : trace_events_) {
      nlohmann::json args;
      args["cid"] = event.correlation_id;
      j["traceEvents"].push_back({{"name", event.name},
                                  {"ph", event.phase},
                                  {"ts", event.timestamp},
                                  {"tid", event.thread_id},
                                  {"pid", event.category},
                                  {"dur", event.duration},
                                  {"cat", event.category_str},
                                  {"args", args}});
    }

    for (const auto& event : trace_gpu_activities_) {
      nlohmann::json args;
      args["cid"] = event.correlation_id;
      j["traceEvents"].push_back({{"name", event.name},
                                  {"ph", event.phase},
                                  {"ts", event.timestamp},
                                  {"tid", event.thread_id},
                                  {"pid", event.category},
                                  {"dur", event.duration},
                                  {"cat", event.category_str},
                                  {"args", args}});
    }

    if (enable_data_flow_) {
      for (auto data_flow : trace_data_flows_) {
        if (data_flow.second.size() > 1) {
          uint64_t id = trace_flow_counter.fetch_add(1, std::memory_order_acquire);
          // The following workaround to overcome the timestamp clock issues in Mi300X
          std::string df0 = "s";
          std::string df1 = "t";
          if (data_flow.second[0].getTimestamp() > data_flow.second[1].getTimestamp()) {
            df1 = "s";
            df0 = "t";
          }
          j["traceEvents"].push_back({{"id", id},
                                      {"ph", df0},
                                      {"ts", data_flow.second[0].getTimestamp()},
                                      {"cat", "DataFlow"},
                                      {"pid", data_flow.second[0].getCategory()},
                                      {"tid", data_flow.second[0].getThreadID()},
                                      {"name", "dep"}});
          j["traceEvents"].push_back({{"id", id},
                                      {"ph", df1},
                                      {"ts", data_flow.second[1].getTimestamp()},
                                      {"cat", "DataFlow"},
                                      {"pid", data_flow.second[1].getCategory()},
                                      {"tid", data_flow.second[1].getThreadID()},
                                      {"name", "dep"}});
        }
      }
    }

    std::ofstream file(path);
    if (file.is_open()) {
      file << j.dump(2) << std::endl;  // Pretty print with 4 spaces
      file.close();
    }
    // trace_gpu_activities_.clear();
    // trace_copy_activities_.clear();
    // trace_events_.clear();
    // trace_data_flows_.clear();
  }

  void ExportTraceEventsForFlameGraph(const std::string& file_path) {
    uint64_t sample_rate = 10;
    const char* flame_graph_sample_rate_env = getenv("ROCPROFILER_FLAME_GRAPH_SAMPLE_RATE");
    if (flame_graph_sample_rate_env) sample_rate = std::stoull(flame_graph_sample_rate_env);
    std::thread kernels_graph = std::thread([&]() {
      const char* flame_graph_enable_kernels = getenv("ROCPROFILER_FLAME_GRAPH_ENABLE_KERNELS");
      if (flame_graph_enable_kernels &&
          (std::string_view(flame_graph_enable_kernels).find("0") != std::string::npos ||
           std::string_view(flame_graph_enable_kernels).find("OFF") != std::string::npos))
        return;
      uint64_t kernels_sample_rate = sample_rate;
      const char* flame_graph_sample_rate_kernels_env =
          getenv("ROCPROFILER_FLAME_GRAPH_KERNELS_SAMPLE_RATE");
      if (flame_graph_sample_rate_kernels_env)
        kernels_sample_rate = std::stoull(flame_graph_sample_rate_kernels_env);
      std::ofstream kernels_file(file_path + "_kernels.txt");
      if (!kernels_file.is_open()) {
        std::cerr << "Failed to open file for writing: " << file_path << std::endl;
        return;
      }

      for (const auto& event : trace_gpu_activities_) {
        // Convert duration to sample count (for simplicity, assume 1 sample per microsecond)
        uint64_t duration = std::stoul(event.duration);
        for (uint64_t i = 0; i < duration; i += kernels_sample_rate) {
          kernels_file << event.name << ";" << event.name << i << " " << kernels_sample_rate
                       << "\n";  // Simple example
        }
      }

      kernels_file.close();
    });

    std::thread copy_graph = std::thread([&]() {
      const char* flame_graph_enable_copy = getenv("ROCPROFILER_FLAME_GRAPH_ENABLE_MEM_COPY");
      if (flame_graph_enable_copy &&
          (std::string_view(flame_graph_enable_copy).find("0") != std::string::npos ||
           std::string_view(flame_graph_enable_copy).find("OFF") != std::string::npos))
        return;
      uint64_t copy_sample_rate = sample_rate;
      const char* flame_graph_sample_rate_copy_env =
          getenv("ROCPROFILER_FLAME_GRAPH_MEM_COPY_SAMPLE_RATE");
      if (flame_graph_sample_rate_copy_env)
        copy_sample_rate = std::stoull(flame_graph_sample_rate_copy_env);
      std::ofstream copy_file(file_path + "_mem_copies.txt");
      if (!copy_file.is_open()) {
        std::cerr << "Failed to open file for writing: " << file_path << std::endl;
        return;
      }

      for (const auto& event : trace_copy_activities_) {
        uint64_t duration = std::stoul(event.duration);
        for (uint64_t i = 0; i < duration; i += copy_sample_rate) {
          copy_file << event.name << ";" << event.name << i << " " << copy_sample_rate << "\n";
        }
      }

      copy_file.close();
    });

    std::thread api_graph = std::thread([&]() {
      const char* flame_graph_enable_api = getenv("ROCPROFILER_FLAME_GRAPH_ENABLE_API");
      if (flame_graph_enable_api &&
          (std::string_view(flame_graph_enable_api).find("0") != std::string::npos ||
           std::string_view(flame_graph_enable_api).find("OFF") != std::string::npos))
        return;
      uint64_t api_sample_rate = sample_rate;
      const char* flame_graph_sample_rate_api_env =
          getenv("ROCPROFILER_FLAME_GRAPH_API_SAMPLE_RATE");
      if (flame_graph_sample_rate_api_env)
        api_sample_rate = std::stoull(flame_graph_sample_rate_api_env);
      std::ofstream api_file(file_path + "_api.txt");
      if (!api_file.is_open()) {
        std::cerr << "Failed to open file for writing: " << file_path << std::endl;
        return;
      }

      for (const auto& event : trace_events_) {
        // Convert duration to sample count (for simplicity, assume 1 sample per microsecond)
        uint64_t duration = std::stoul(event.duration);
        for (uint64_t i = 0; i < duration; i += api_sample_rate) {
          api_file << event.name << ";" << event.name << i << " " << api_sample_rate
                   << "\n";  // Simple example
        }
      }

      api_file.close();
    });

    kernels_graph.join();
    copy_graph.join();
    api_graph.join();
  }

  void LogGpuActivityTrace(const std::string& name, const std::string& category_str,
                           uint64_t timestamp, uint64_t thread_id, uint64_t category,
                           uint64_t duration, uint64_t correlation_id) {
    if (duration == 0) duration = 1;
    trace_gpu_activities_.push_back({"X", category, name, category_str, std::to_string(timestamp),
                                     std::to_string(duration), thread_id, correlation_id});
  }

  void LogCopyActivityTrace(const std::string& name, const std::string& category_str,
                            uint64_t timestamp, uint64_t thread_id, uint64_t category,
                            uint64_t duration, uint64_t correlation_id) {
    if (duration == 0) duration = 1;
    trace_copy_activities_.push_back({"X", category, name, category_str, std::to_string(timestamp),
                                      std::to_string(duration), thread_id, correlation_id});
  }

  void LogCopyUnknownTrace(const std::string& name, const std::string& category_str,
                           uint64_t timestamp, uint64_t thread_id, uint64_t category,
                           uint64_t duration, uint64_t correlation_id) {
    if (duration == 0) duration = 1;
    trace_unknown_activities_.push_back({"X", category, name, category_str,
                                         std::to_string(timestamp), std::to_string(duration),
                                         thread_id, correlation_id});
  }

  void LogAPITrace(const std::string& name, const std::string& category_str,
                   uint64_t start_timestamp, uint64_t end_timestamp, uint64_t thread_id,
                   uint64_t category, uint64_t correlation_id) {
    uint64_t duration = (end_timestamp - start_timestamp);
    if (duration == 0) duration = 1;
    trace_events_.push_back({"X", category, name, category_str, std::to_string(start_timestamp),
                             std::to_string(duration), thread_id, correlation_id});
  }

  void AddDataFlowEvent(uint64_t timestamp, TraceFlowCategory type, uint64_t category,
                        uint64_t thread_id, uint64_t correlation_id) {
    if (enable_data_flow_) {
      if (type == ROCPROFILER_DATA_FLOW_START) {
        trace_data_flows_[correlation_id].emplace_back(
            TraceFlow(timestamp, "s", category, thread_id, correlation_id));
      } else {
        trace_data_flows_[correlation_id].emplace_back(
            TraceFlow(timestamp, "t", category, thread_id, correlation_id));
      }
    }
  }

  // We may need to use this in the Args of the trace
  const char* GetDomainName(rocprofiler_tracer_activity_domain_t domain) {
    switch (domain) {
      case ACTIVITY_DOMAIN_ROCTX:
        return "ROCTX_DOMAIN";
        break;
      case ACTIVITY_DOMAIN_HIP_API:
        return "HIP_API_DOMAIN";
        break;
      case ACTIVITY_DOMAIN_HIP_OPS:
        return "HIP_OPS_DOMAIN";
        break;
      case ACTIVITY_DOMAIN_HSA_API:
        return "HSA_API_DOMAIN";
        break;
      case ACTIVITY_DOMAIN_HSA_OPS:
        return "HSA_OPS_DOMAIN";
        break;
      case ACTIVITY_DOMAIN_HSA_EVT:
        return "HSA_EVT_DOMAIN";
        break;
      default:
        return "";
    }
  }

  int FlushProfilerRecord(rocprofiler_record_profiler_t profiler_record,
                          rocprofiler_session_id_t session_id) {
    std::lock_guard<std::mutex> lock(writing_lock);

    const uint64_t device_id = profiler_record.gpu_id.handle;
    const uint64_t queue_id = profiler_record.queue_id.handle;
    const uint64_t correlation_id = profiler_record.correlation_id.value;

    // Taken from rocprofiler: The size hasn't changed in  recent past
    static const uint32_t lds_block_size = 128 * 4;

    std::string full_kernel_name = get_kernel_name(profiler_record);

    uint64_t start_timestamp = profiler_record.timestamps.begin.value / 1000;
    uint64_t end_timestamp = profiler_record.timestamps.end.value / 1000;
    uint64_t duration = end_timestamp - start_timestamp;

    LogGpuActivityTrace(full_kernel_name, "gpu", start_timestamp, device_id + 1, 2, duration,
                        correlation_id);
    AddDataFlowEvent(start_timestamp, ROCPROFILER_DATA_FLOW_END, 2, device_id + 1, correlation_id);

    // We need to add the information below in the Args of the trace
    // TRACE_EVENT_BEGIN("KERNELS", perfetto::DynamicString(full_kernel_name.c_str()), queue_track,
    //                   profiler_record.timestamps.begin.value, "Full Kernel Name",
    //                   full_kernel_name.c_str(), "Agent ID", device_id, "Queue ID",
    //                   profiler_record.queue_id.handle, "GRD",
    //                   profiler_record.kernel_properties.grid_size, "WGR",
    //                   profiler_record.kernel_properties.workgroup_size, "LDS",
    //                   (((profiler_record.kernel_properties.lds_size + (lds_block_size - 1)) &
    //                     ~(lds_block_size - 1))),
    //                   "SCR", profiler_record.kernel_properties.scratch_size, "Arch. VGPR",
    //                   profiler_record.kernel_properties.arch_vgpr_count, "Accumulation Vgpr",
    //                   profiler_record.kernel_properties.accum_vgpr_count, "SGPR",
    //                   profiler_record.kernel_properties.sgpr_count, "Wave Size",
    //                   profiler_record.kernel_properties.wave_size, "Signal",
    //                   profiler_record.kernel_properties.signal_handle,
    //                   perfetto::Flow::ProcessScoped(correlation_id));

    // For Counters
    if (!profiler_record.counters) return 0;

    for (uint64_t i = 0; i < profiler_record.counters_count.value; i++) {
      if (profiler_record.counters[i].counter_handler.handle == 0) continue;
      // We need to add the counter values below in the Args of the trace
      // TRACE_COUNTER("COUNTERS", counters_track, profiler_record.timestamps.begin.value,
      //               profiler_record.counters[i].value.value);
      // // Added an extra zero event for maintaining start-end of the counter
      // TRACE_COUNTER("COUNTERS", counters_track, profiler_record.timestamps.end.value, 0);
    }

    return 0;
  }

  int FlushTracerRecord(rocprofiler_record_tracer_t tracer_record,
                        rocprofiler_session_id_t session_id) {
    std::lock_guard<std::mutex> lock(writing_lock);
    uint64_t device_id = tracer_record.agent_id.handle;
    const char* operation_name_c = nullptr;
    // ROCTX domain Operation ID doesn't have a name
    // It depends on the user input of the roctx functions.
    // ROCTX message is the tracer_record.name
    if (tracer_record.domain != ACTIVITY_DOMAIN_ROCTX) {
      CHECK_ROCPROFILER(rocprofiler_query_tracer_operation_name(
          tracer_record.domain, tracer_record.operation_id, &operation_name_c));
      if (!operation_name_c) operation_name_c = "Unknown Operation";
    }
    uint64_t start_timestamp = tracer_record.timestamps.begin.value / 1000;
    uint64_t end_timestamp = tracer_record.timestamps.end.value / 1000;
    uint64_t duration = end_timestamp - start_timestamp;
    uint64_t correlation_id = tracer_record.correlation_id.value;
    std::string roctx_message;
    uint64_t roctx_id = 0;
    uint64_t thread_id = tracer_record.thread_id.value;
    switch (tracer_record.domain) {
      case ACTIVITY_DOMAIN_ROCTX: {
        roctx_id = tracer_record.external_id.id;
        roctx_message = tracer_record.name ? tracer_record.name : "";
        LogAPITrace((!roctx_message.empty() ? roctx_message : ""), "CPU", start_timestamp,
                    end_timestamp, thread_id, 1, correlation_id);
        break;
      }
      case ACTIVITY_DOMAIN_HSA_API: {
        LogAPITrace(operation_name_c, "CPU", start_timestamp, end_timestamp, thread_id, 1,
                    correlation_id);
        AddDataFlowEvent(start_timestamp, ROCPROFILER_DATA_FLOW_START, 1, thread_id,
                         correlation_id);
        break;
      }
      case ACTIVITY_DOMAIN_HIP_API: {
        LogAPITrace(operation_name_c, "CPU", start_timestamp, end_timestamp, thread_id, 1,
                    correlation_id);
        AddDataFlowEvent(start_timestamp, ROCPROFILER_DATA_FLOW_START, 1, thread_id,
                         correlation_id);
        if (trace_period_enabled_) found_correlation_ids_.insert(correlation_id);
        break;
      }
      case ACTIVITY_DOMAIN_EXT_API: {
        printf("Warning: External API is not supported!\n");
        break;
      }
      case ACTIVITY_DOMAIN_HIP_OPS: {
        if (trace_period_enabled_ &&
            found_correlation_ids_.find(correlation_id) == found_correlation_ids_.end())
          break;
        std::size_t pos = std::string::npos;
        if (tracer_record.name) {
          auto kernel_name_it = kernel_names_map.find(tracer_record.name);
          if (kernel_name_it == kernel_names_map.end()) {
            kernel_name_it =
                kernel_names_map
                    .emplace(
                        tracer_record.name,
                        rocprofiler::truncate_name(rocprofiler::cxx_demangle(tracer_record.name)))
                    .first;
          }
          LogGpuActivityTrace(kernel_name_it->second, "GPU", start_timestamp, device_id + 1, 2,
                              duration, correlation_id);
          AddDataFlowEvent(start_timestamp, ROCPROFILER_DATA_FLOW_END, 2, device_id + 1,
                           tracer_record.correlation_id.value);
        } else {
          // MEM Copies are not correlated to GPUs, so they need a special track
          pos = operation_name_c ? std::string_view(operation_name_c).find("Copy")
                                 : std::string::npos;
          std::string category_str = "COPY";
          uint64_t category = 3;
          uint64_t thread_id_json = 1;

          if (std::string::npos == pos) {
            LogCopyUnknownTrace(operation_name_c, "HIPBLITKERNELS", start_timestamp, device_id, 4,
                                duration, correlation_id);
            AddDataFlowEvent(start_timestamp, ROCPROFILER_DATA_FLOW_END, 4, device_id,
                             correlation_id);
          } else {
            LogCopyActivityTrace(operation_name_c, category_str, start_timestamp, thread_id_json,
                                 category, duration, correlation_id);
            AddDataFlowEvent(start_timestamp, ROCPROFILER_DATA_FLOW_END, category, thread_id_json,
                             correlation_id);
          }
        }
        break;
      }
      case ACTIVITY_DOMAIN_HSA_OPS: {
        LogCopyActivityTrace(operation_name_c, "COPY", start_timestamp, 0, 3, duration,
                             correlation_id);
        AddDataFlowEvent(start_timestamp, ROCPROFILER_DATA_FLOW_END, 3, 0, correlation_id);
        break;
      }
      default: {
        rocprofiler::warning("Ignored record for domain %d", tracer_record.domain);
        break;
      }
    }
    return 0;
  }

  int WriteBufferRecords(const rocprofiler_record_header_t* begin,
                         const rocprofiler_record_header_t* end,
                         rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t buffer_id) {
    while (begin < end) {
      if (!begin) return 0;
      switch (begin->kind) {
        case ROCPROFILER_PROFILER_RECORD: {
          rocprofiler_record_profiler_t* profiler_record =
              const_cast<rocprofiler_record_profiler_t*>(
                  reinterpret_cast<const rocprofiler_record_profiler_t*>(begin));
          FlushProfilerRecord(*profiler_record, session_id);
          break;
        }
        case ROCPROFILER_TRACER_RECORD: {
          rocprofiler_record_tracer_t* tracer_record = const_cast<rocprofiler_record_tracer_t*>(
              reinterpret_cast<const rocprofiler_record_tracer_t*>(begin));
          FlushTracerRecord(*tracer_record, session_id);
          break;
        }
        default:
          break;
      }
      rocprofiler_next_record(begin, &begin, session_id, buffer_id);
    }
    return 0;
  }

  bool IsValid() const { return is_valid_; }

 private:
  bool is_valid_{false};
  fs::path output_prefix_;
  fs::path txt_output_prefix_;

  std::atomic<bool> flush_thread_check_{false};
  std::thread flush_thread_;

  std::ofstream stream_;
  std::unordered_map<std::string, std::string> kernel_names_map;

  std::atomic<uint64_t> trace_flow_counter{0};
  std::vector<TraceActivity> trace_events_;
  std::vector<TraceActivity> trace_gpu_activities_;
  std::vector<TraceActivity> trace_copy_activities_;
  std::vector<TraceActivity> trace_unknown_activities_;

  std::map<uint64_t, std::vector<TraceFlow>> trace_data_flows_;

  TraceCategory trace_categories[4] = {{{"CPU"}, "M", 1, 0},
                                       {{"GPU"}, "M", 2, 1},
                                       {{"COPY"}, "M", 3, 2},
                                       {{"HIPBLITKERNELS"}, "M", 4, 3}};
  std::atomic<bool> trace_categories_check{false};
  bool enable_data_flow_ = true;

  std::unordered_set<uint64_t> found_correlation_ids_;
  bool trace_period_enabled_ = false;
};

json_plugin_t* json_plugin = nullptr;

}  // namespace

int rocprofiler_plugin_initialize(uint32_t rocprofiler_major_version,
                                  uint32_t rocprofiler_minor_version, void* data) {
  if (rocprofiler_major_version != ROCPROFILER_VERSION_MAJOR ||
      rocprofiler_minor_version > ROCPROFILER_VERSION_MINOR)
    return -1;

  std::lock_guard<std::mutex> lock(writing_lock);
  if (json_plugin != nullptr) return -1;

  json_plugin = new json_plugin_t();
  if (json_plugin->IsValid()) {
    writing_lock.unlock();
    return 0;
  }

  delete json_plugin;
  json_plugin = nullptr;
  return -1;
}

void rocprofiler_plugin_finalize() {
  std::lock_guard<std::mutex> lock(writing_lock);
  if (!json_plugin) return;
  json_plugin->delete_json_plugin();
  delete json_plugin;
  json_plugin = nullptr;
}

ROCPROFILER_EXPORT int rocprofiler_plugin_write_buffer_records(
    const rocprofiler_record_header_t* begin, const rocprofiler_record_header_t* end,
    rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t buffer_id) {
  if (!json_plugin || !json_plugin->IsValid()) return -1;
  return json_plugin->WriteBufferRecords(begin, end, session_id, buffer_id);
}

ROCPROFILER_EXPORT int rocprofiler_plugin_write_record(rocprofiler_record_tracer_t record) {
  if (record.header.id.handle == 0) return 0;

  if (!json_plugin || !json_plugin->IsValid()) return -1;
  return json_plugin->FlushTracerRecord(record, rocprofiler_session_id_t{0});
}
