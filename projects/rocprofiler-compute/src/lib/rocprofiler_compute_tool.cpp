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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/*
This is a native tool for rocprofiler-compute to collect counters data for GPU
kernel dispatches using the rocprofiler-sdk public API. This C++ tool is
compiled into a shared object with hipcc/amdclang++ and dynamically links to the
rocprofiler-sdk library. The shared object is injected using the LD_PRELOAD
environment variable so that rocprofiler-sdk services can be configured before
the GPU workload starts executing.

An experimental feature for attach/detach scenarios is also provided.

Code Flow:

1. Entry point - rocprofiler_configure():
    - Parses ROCPROF environment variables to configure profiling.
    - Sets up tool metadata and logging.
    - Returns pointers to tool_init() and tool_fini() functions.

2. Tool Initialization - tool_init():
    - Creates a profiling context.
    - Subscribes to dispatch tracing and counting services by providing function
callbacks.
    - Starts the profiling context.

3. Kernel registration callback - tool_tracing_callback():
    - Invoked when a kernel is registered.
    - Stores the kernel name to kernel id mapping.
    - Determines which kernel names/ids to target for profiling based on ROCPROF
environment variables.

4. Kernel dispatch callback - dispatch_callback():
    - Invoked before a kernel dispatch is enqueued.
    - Decides whether to profile this dispatch.
    - If profiling is required, creates or fetches from cache a counter profile
for the agent and returns a pointer to it.
    - The counter profile dictates which counters to collect for this dispatch.

5. Kernel dispatch record callback - record_callback():
    - Invoked after a kernel dispatch is completed.
    - Receives the collected counter records.
    - Stores the counter records in tool data for later processing.

6. Tool Finalization - tool_fini():
    - Called when the application is terminating.
    - Stops the profiling context.
    - Processes and writes the collected counter records to the output file.
    - Cleans up resources.
*/

#include "helper.hpp"

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#define ROCPROFILER_CALL(result, msg)                                          \
  {                                                                            \
    rocprofiler_status_t CHECKSTATUS = result;                                 \
    if (CHECKSTATUS != ROCPROFILER_STATUS_SUCCESS) {                           \
      std::string status_msg = rocprofiler_get_status_string(CHECKSTATUS);     \
      std::cerr << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] "     \
                << msg << " failed with error code " << CHECKSTATUS << ": "    \
                << status_msg << std::endl;                                    \
      std::stringstream errmsg{};                                              \
      errmsg << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] "        \
             << msg " failure (" << status_msg << ")";                         \
      throw std::runtime_error(errmsg.str());                                  \
    }                                                                          \
  }

namespace {

// Multiplexing modes enum
enum class iteration_multiplexing_mode_t { DISABLED, SIMPLE, KERNEL, LAUNCH };

// Kernel dispatch info struct for iteration multiplexing
struct kernel_dispatch_info_t {
  uint64_t kernel_id;
  uint64_t queue_id;
  rocprofiler_dim3_t workgroup_size;
  rocprofiler_dim3_t grid_size;
  uint32_t LDS_memory_size;

  // Overload operator< for strict weak ordering
  bool operator<(const kernel_dispatch_info_t other) const {
    // Compare based on kernel_id first, then queue_id, then workgroup_size,
    // then grid_size, and finally LDS_memory_size
    return std::tie(kernel_id, queue_id, workgroup_size.x, workgroup_size.y,
                    workgroup_size.z, grid_size.x, grid_size.y, grid_size.z,
                    LDS_memory_size) <
           std::tie(other.kernel_id, other.queue_id, other.workgroup_size.x,
                    other.workgroup_size.y, other.workgroup_size.z,
                    other.grid_size.x, other.grid_size.y, other.grid_size.z,
                    other.LDS_memory_size);
  }
};

// Iteration multiplexing data struct
struct iteration_multiplexing_dispatch_record_t {
  std::size_t config;
  std::map<uint64_t, std::size_t> kernel_config;
  std::map<kernel_dispatch_info_t, std::size_t> dispatch_config;
};

// Struct to store a single counter info record
struct counter_info_record_t {
  uint64_t dispatch_id;
  uint64_t agent_id;
  uint64_t kernel_id;
  uint32_t LDS_memory_size;
  uint64_t counter_id;
  std::string counter_name;
  double counter_value;
};

// Tool data struct, now includes a vector of counter_info_record_t
struct tool_data_t {
  std::mutex mut{};
  std::string output_filename{};
  std::unordered_map<uint64_t, std::string> counter_id_name_map{};
  std::string requested_counters{};
  std::string kernel_filter_include_regex{};
  std::vector<std::pair<uint64_t, uint64_t>> kernel_filter_ranges{};
  std::vector<counter_info_record_t> counter_records;
  std::set<uint64_t> target_kernel_ids{};
  iteration_multiplexing_mode_t iteration_multiplexing_mode{
      iteration_multiplexing_mode_t::DISABLED};
};

using kernel_symbol_data_t =
    rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;

rocprofiler_context_id_t &get_client_ctx() {
  static rocprofiler_context_id_t ctx{0};
  return ctx;
}

iteration_multiplexing_mode_t
iteration_multiplexing_mode(const std::string &mode) {
  // if (mode == "simple")
  //   return iteration_multiplexing_mode_t::SIMPLE;
  // else
  if (mode == "kernel")
    return iteration_multiplexing_mode_t::KERNEL;
  else if (mode == "kernel_launch_params")
    return iteration_multiplexing_mode_t::LAUNCH;
  else
    return iteration_multiplexing_mode_t::DISABLED;
}

void record_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                     rocprofiler_counter_record_t *record_data,
                     size_t record_count,
                     rocprofiler_user_data_t /* user_data */,
                     void *callback_data_args) {
  auto *tool_data_ptr =
      static_cast<std::unique_ptr<tool_data_t> *>(callback_data_args);
  tool_data_t *tool;
  {
    std::lock_guard<std::mutex> lock(tool_data_ptr->get()->mut);
    tool = tool_data_ptr->get();
  }

  // For each counter, write: dispatch_id, counter_id, counter_name,
  // counter_value
  for (size_t i = 0; i < record_count; ++i) {
    rocprofiler_counter_id_t counter_id{};
    ROCPROFILER_CALL(
        rocprofiler_query_record_counter_id(record_data[i].id, &counter_id),
        "query record counter id");

    // Store the counter info record in tool_data
    counter_info_record_t record{dispatch_data.dispatch_info.dispatch_id,
                                 dispatch_data.dispatch_info.agent_id.handle,
                                 dispatch_data.dispatch_info.kernel_id,
                                 dispatch_data.dispatch_info.group_segment_size,
                                 counter_id.handle,
                                 tool->counter_id_name_map[counter_id.handle],
                                 record_data[i].counter_value};
    {
      std::lock_guard<std::mutex> lock(tool->mut);
      tool->counter_records.push_back(std::move(record));
    }
  }
}

/**
 * Callback from rocprofiler when a code object is loaded.
 * We use this to get record kernel names as they are registered.
 */
void tool_tracing_callback(rocprofiler_callback_tracing_record_t record,
                           rocprofiler_user_data_t * /*user_data*/,
                           void *callback_data) {
  if (record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD &&
      record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
      record.operation ==
          ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER) {
    auto *data = static_cast<kernel_symbol_data_t *>(record.payload);
    int demangle_status = 0;
    auto kernel_name =
        helper_utils::cxa_demangle(data->kernel_name, &demangle_status);
    kernel_name = helper_utils::truncate_name(kernel_name);

    // check if regex can be found in kernel name matches regex from tool data,
    // if matches store kernel id
    auto *tool_data_ptr =
        static_cast<std::unique_ptr<tool_data_t> *>(callback_data);
    auto *tool = tool_data_ptr->get();
    // Lock before modifying target_kernel_ids
    std::lock_guard<std::mutex> lock(tool->mut);
    if (!tool->kernel_filter_include_regex.empty()) {
      try {
        std::regex re(tool->kernel_filter_include_regex);
        if (!kernel_name.empty() && std::regex_search(kernel_name, re)) {
          tool->target_kernel_ids.insert(data->kernel_id);
        }
      } catch (const std::regex_error &e) {
        std::cerr
            << "[rocprofiler-compute] [" << __FUNCTION__
            << "] ERROR: Invalid regex in ROCPROF_KERNEL_FILTER_INCLUDE_REGEX: "
            << tool->kernel_filter_include_regex << " : " << e.what()
            << std::endl;
      }
    }
    // If no regex specified, collect for all kernels
    else {
      tool->target_kernel_ids.insert(data->kernel_id);
    }
  }
}

/**
 * Checks if the given kernel dispatch should be targeted for profiling.
 * Returns true if the kernel_id is in the set of target_kernel_ids (if
 * non-empty), and if the kernel_iteration (1-based index) matches the
 * kernel_filter_range (if specified).
 *
 * @param tool Pointer to the tool_data_t structure containing profiling
 * configuration.
 * @param kernel_id The kernel ID of the dispatch.
 * @param kernel_iteration The 1-based index of this kernel_id's dispatch (first
 * dispatch is 1).
 * @return true if the dispatch should be profiled, false otherwise.
 */
bool is_targetted_dispatch(const tool_data_t *tool, uint64_t kernel_id,
                           uint64_t kernel_iteration) {
  // If target_kernel_ids is non-empty, only allow those kernel_ids
  if (!tool->target_kernel_ids.empty() &&
      !tool->target_kernel_ids.count(kernel_id))
    return false;

  // If kernel_filter_ranges is set, check if kernel_iteration is in any of the
  // specified ranges
  if (!tool->kernel_filter_ranges.empty())
    return std::any_of(tool->kernel_filter_ranges.begin(),
                       tool->kernel_filter_ranges.end(),
                       [kernel_iteration](const auto &range) {
                         return kernel_iteration >= range.first &&
                                kernel_iteration <= range.second;
                       });

  // If no filter ranges are specified, or all checks passed, profile this
  // dispatch
  return true;
}

/**
 * @brief Creates a counter collection profile for performance monitoring on a
 * specific GPU agent.
 *
 * This function parses the requested counters from the tool configuration,
 * validates them against the counters supported by the target GPU agent, and
 * creates a rocprofiler counter configuration for collecting the available
 * requested counters during dispatch profiling.
 *
 * @param tool Pointer to tool data containing the requested counters string and
 * counter mappings
 * @param dispatch_data Dispatch counting service data containing agent
 * information for the target GPU
 *
 * @return rocprofiler_counter_config_id_t A valid counter configuration profile
 * ID that can be used for counter collection, or an invalid profile (handle =
 * 0) if creation fails
 *
 * @details
 * The function performs the following operations:
 * 1. Parses the requested counters from tool->requested_counters string
 * (format: "prefix:counter1 counter2 ...")
 * 2. Queries all counters supported by the specified GPU agent
 * 3. Filters the supported counters to match only those requested
 * 4. Logs warnings for any requested counters that are not supported by the
 * agent
 * 5. Creates rocprofiler counter configurations for the valid
 * counters
 *
 * @note If no counters are requested or none of the requested counters are
 * supported, an empty profile may be created. Unsupported counters are logged
 * as warnings but do not cause the function to fail.
 */
void create_counter_collection_profile(
    tool_data_t *tool, rocprofiler_agent_id_t agent_id,
    std::unordered_map<uint64_t, std::vector<rocprofiler_counter_config_id_t>>
        &profile_cache) {
  // get counters to collect
  std::set<std::set<std::string>> counters_to_collect;
  for (const std::string &counters_str :
       helper_utils::split_by_regex(tool->requested_counters, "[,]")) {
    if (!counters_str.empty()) {
      auto pos = counters_str.find(':');
      if (pos != std::string::npos) {
        std::istringstream ss(counters_str.substr(pos + 1));
        std::set<std::string> counters;
        for (std::string token; ss >> token;) {
          counters.insert(token);
        }
        counters_to_collect.insert(counters);
      }
    }
  }

  // Get available counters for this agent
  std::vector<rocprofiler_counter_id_t> gpu_counters;
  ROCPROFILER_CALL(
      rocprofiler_iterate_agent_supported_counters(
          agent_id,
          [](rocprofiler_agent_id_t, rocprofiler_counter_id_t *counters,
             size_t num_counters, void *user_data) {
            std::vector<rocprofiler_counter_id_t> *vec =
                static_cast<std::vector<rocprofiler_counter_id_t> *>(user_data);
            for (size_t i = 0; i < num_counters; i++) {
              vec->push_back(counters[i]);
            }
            return ROCPROFILER_STATUS_SUCCESS;
          },
          static_cast<void *>(&gpu_counters)),
      "fetch supported counters");

  std::vector<std::string> gpu_counter_names;
  std::map<std::string, rocprofiler_counter_id_t> gpu_counter_map;
  for (auto &counter : gpu_counters) {
    rocprofiler_counter_info_v0_t info;
    ROCPROFILER_CALL(rocprofiler_query_counter_info(
                         counter, ROCPROFILER_COUNTER_INFO_VERSION_0,
                         static_cast<void *>(&info)),
                     "query counter info");
    gpu_counter_names.push_back(std::string(info.name));
    gpu_counter_map.insert({std::string(info.name), counter});
  }

  // Identify counters requested to collect which are available
  std::vector<std::vector<std::string>> collect_counter_names;
  std::vector<std::vector<rocprofiler_counter_id_t>> collect_counters;
  std::vector<std::string> unsupported_counters;
  for (const auto &counters : counters_to_collect) {
    std::vector<std::string> counter_names;
    std::vector<rocprofiler_counter_id_t> counter_ids;
    for (const auto &counter_name : counters) {
      if (std::find(gpu_counter_names.begin(), gpu_counter_names.end(),
                    counter_name) != gpu_counter_names.end()) {
        counter_names.push_back(counter_name);
        counter_ids.push_back(gpu_counter_map[counter_name]);
        tool->counter_id_name_map[gpu_counter_map[counter_name].handle] =
            counter_name;
      } else {
        unsupported_counters.push_back(counter_name);
      }
    }
    collect_counter_names.push_back(counter_names);
    collect_counters.push_back(counter_ids);
  }

  if (!unsupported_counters.empty()) {
    std::clog << "\033[33m[rocprofiler-compute] [" << __FUNCTION__
              << "] WARNING: Requested counters not available: ";
    for (size_t i = 0; i < unsupported_counters.size(); ++i) {
      std::clog << unsupported_counters[i];
      if (i + 1 < unsupported_counters.size())
        std::clog << ", ";
    }
    std::clog << "\033[0m" << std::endl;
  }

  // Create a profile cache for the agent
  std::vector<rocprofiler_counter_config_id_t> profiles{};
  // Create a collection profile for the counters
  for (auto &collect_counters_one_iter : collect_counters) {
    rocprofiler_counter_config_id_t profile = {.handle = 0};
    ROCPROFILER_CALL(rocprofiler_create_counter_config(
                         agent_id, collect_counters_one_iter.data(),
                         collect_counters_one_iter.size(), &profile),
                     "construct profile cfg");
    profiles.push_back(profile);
    profile_cache[agent_id.handle] = profiles;
  }
}

/**
 * Callback from rocprofiler when an kernel dispatch is enqueued into the HSA
 * queue. rocprofiler_counter_config_id_t* is a return to specify what counters
 * to collect for this dispatch (dispatch_packet).
 * We store profile in a cache to prevent constructing many identical
 * profiles. We first check the cache to see if we have already constructed a
 * profile for the agent. If we have, return it. Otherwise, construct a new
 * profile.
 */
void dispatch_callback(
    rocprofiler_dispatch_counting_service_data_t dispatch_data,
    rocprofiler_counter_config_id_t *config,
    rocprofiler_user_data_t * /*user_data*/, void *callback_data_args) {

  auto kernel_id = dispatch_data.dispatch_info.kernel_id;
  auto agent_id = dispatch_data.dispatch_info.agent_id.handle;

  // create static map of kernel_id to number of dispatches (zero indexed) and
  // update it
  static std::unordered_map<uint64_t, uint64_t> kernel_id_iteration_map{};
  static std::shared_mutex kernel_id_iteration_mutex;
  uint64_t kernel_iteration = 0;
  {
    // Acquire unique lock for update and ensure map is updated correctly
    std::unique_lock<std::shared_mutex> lock(kernel_id_iteration_mutex);
    auto &iter = kernel_id_iteration_map[kernel_id];
    iter += 1;
    kernel_iteration = iter;
  }

  // static cast tool
  auto *tool_data_ptr =
      static_cast<std::unique_ptr<tool_data_t> *>(callback_data_args);
  tool_data_t *tool;
  {
    std::lock_guard<std::mutex> lock(tool_data_ptr->get()->mut);
    tool = tool_data_ptr->get();
  }

  // kernel filtering
  if (!is_targetted_dispatch(tool, kernel_id, kernel_iteration)) {
    return;
  }

  static std::shared_mutex m_mutex = {};
  static std::unordered_map<uint64_t,
                            std::vector<rocprofiler_counter_config_id_t>>
      profile_cache = {};
  static std::unordered_map<uint64_t, iteration_multiplexing_dispatch_record_t>
      iteration_multiplexing_data = {};

  // check cache for existing profile for this agent
  auto search_profile_cache = [&]() {
    if (auto pos = profile_cache.find(agent_id); pos != profile_cache.end())
      return true;
    return false;
  };

  auto set_config_from_cache = [&]() {
    if (tool->iteration_multiplexing_mode !=
            iteration_multiplexing_mode_t::DISABLED &&
        iteration_multiplexing_data.find(agent_id) ==
            iteration_multiplexing_data.end()) {
      // First time setting up iteration multiplexing data for this agent
      iteration_multiplexing_data[agent_id] =
          iteration_multiplexing_dispatch_record_t{};
      if (tool->iteration_multiplexing_mode ==
          iteration_multiplexing_mode_t::SIMPLE) {
        iteration_multiplexing_data[agent_id].config =
            -1; // so first increment sets to 0
      }
    }

    kernel_dispatch_info_t dispatch_info{
        dispatch_data.dispatch_info.kernel_id,
        dispatch_data.dispatch_info.queue_id.handle,
        dispatch_data.dispatch_info.workgroup_size,
        dispatch_data.dispatch_info.grid_size,
        dispatch_data.dispatch_info.group_segment_size};
    switch (tool->iteration_multiplexing_mode) {
    case iteration_multiplexing_mode_t::DISABLED:
      *config = profile_cache[agent_id][0];
      return;

    case iteration_multiplexing_mode_t::SIMPLE:
      iteration_multiplexing_data[agent_id].config =
          (iteration_multiplexing_data[agent_id].config + 1) %
          profile_cache[agent_id].size();
      *config =
          profile_cache[agent_id][iteration_multiplexing_data[agent_id].config];
      return;

    case iteration_multiplexing_mode_t::KERNEL:
      if (iteration_multiplexing_data[agent_id].kernel_config.find(kernel_id) ==
          iteration_multiplexing_data[agent_id].kernel_config.end()) {
        // First time seeing this kernel_id for this agent
        iteration_multiplexing_data[agent_id].kernel_config[kernel_id] =
            -1; // so first increment sets to 0
      }
      iteration_multiplexing_data[agent_id].kernel_config[kernel_id] =
          (iteration_multiplexing_data[agent_id].kernel_config[kernel_id] + 1) %
          profile_cache[agent_id].size();
      *config = profile_cache[agent_id][iteration_multiplexing_data[agent_id]
                                            .kernel_config[kernel_id]];
      return;

    case iteration_multiplexing_mode_t::LAUNCH:
      if (iteration_multiplexing_data[agent_id].dispatch_config.find(
              dispatch_info) ==
          iteration_multiplexing_data[agent_id].dispatch_config.end()) {
        // First time seeing this dispatch_info for this agent
        iteration_multiplexing_data[agent_id].dispatch_config[dispatch_info] =
            -1; // so first increment sets to 0
      }
      iteration_multiplexing_data[agent_id].dispatch_config[dispatch_info] =
          (iteration_multiplexing_data[agent_id]
               .dispatch_config[dispatch_info] +
           1) %
          profile_cache[agent_id].size();
      *config = profile_cache[agent_id][iteration_multiplexing_data[agent_id]
                                            .dispatch_config[dispatch_info]];
      return;

    default:
      throw std::runtime_error("[" + std::string(__FUNCTION__) +
                               "] Unsupported iteration multiplexing mode");
    }
  };

  {
    auto rlock = std::shared_lock{m_mutex};
    if ((tool->iteration_multiplexing_mode ==
         iteration_multiplexing_mode_t::DISABLED) &&
        search_profile_cache()) {
      *config = profile_cache[agent_id][0];
      return;
    }
  }

  // get write lock to update cache
  auto wlock = std::unique_lock{m_mutex};
  if (search_profile_cache()) {
    set_config_from_cache();
    return;
  }

  create_counter_collection_profile(tool, dispatch_data.dispatch_info.agent_id,
                                    profile_cache);

  // Return the profile to collect those counters for this dispatch
  set_config_from_cache();
}

int tool_init(rocprofiler_client_finalize_t, void *user_data) {
  std::clog << "[rocprofiler-compute] In tool init\n";
  ROCPROFILER_CALL(rocprofiler_create_context(&get_client_ctx()),
                   "context creation");

  ROCPROFILER_CALL(rocprofiler_configure_callback_dispatch_counting_service(
                       get_client_ctx(), dispatch_callback, user_data,
                       record_callback, user_data),
                   "setup counting service");
  ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
                       get_client_ctx(),
                       ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT, nullptr, 0,
                       tool_tracing_callback, user_data),
                   "setup code object tracing service");
  ROCPROFILER_CALL(rocprofiler_start_context(get_client_ctx()),
                   "start context");

  return 0;
}

void generate_output(tool_data_t *tool_data) {
  // Dispatches before the kernel to be filtered was registered may have been
  // profiled. Remove any records whose kernel id does not match the
  // target_kernel_ids
  if (!tool_data->target_kernel_ids.empty()) {
    tool_data->counter_records.erase(
        std::remove_if(tool_data->counter_records.begin(),
                       tool_data->counter_records.end(),
                       [tool_data](const counter_info_record_t &record) {
                         return tool_data->target_kernel_ids.find(
                                    record.kernel_id) ==
                                tool_data->target_kernel_ids.end();
                       }),
        tool_data->counter_records.end());
  }
  if (tool_data->counter_records.empty()) {
    return;
  }
  // Write collected counter records and clean up
  if (!tool_data->output_filename.empty()) {
    std::ofstream ofs(tool_data->output_filename);
    if (!ofs.is_open()) {
      std::cerr << "Failed to open output file: " << tool_data->output_filename
                << std::endl;
      return;
    }
    // Write header at the beginning of the file
    ofs << "dispatch_id,gpu_id,kernel_id,lds_per_workgroup,"
           "counter_id,counter_name,counter_value\n";
    for (const auto &r : tool_data->counter_records)
      ofs << r.dispatch_id << ',' << r.agent_id << "," << r.kernel_id << ','
          << r.LDS_memory_size << ',' << r.counter_id << ',' << r.counter_name
          << ',' << r.counter_value << '\n';
    ofs.flush();
    std::clog << "[rocprofiler-compute] [" << __FUNCTION__
              << "] Counter collection data has been written to: "
              << tool_data->output_filename << std::endl;
  }
}

void tool_fini(void *user_data) {
  assert(user_data);
  std::clog << "[rocprofiler-compute] In tool fini\n";
  rocprofiler_stop_context(get_client_ctx());

  auto *tool_data_ptr = static_cast<std::unique_ptr<tool_data_t> *>(user_data);
  generate_output(tool_data_ptr->get());

  delete tool_data_ptr;
}

} // namespace

std::unique_ptr<tool_data_t>
create_tool_data(rocprofiler_client_id_t * /*id*/) {
  auto tool_data = std::make_unique<tool_data_t>();

  // Generate a unique output filename using the process ID
  std::string base_filename =
      std::to_string(getpid()) + "_native_counter_collection.csv";

  // Require ROCPROF_OUTPUT_PATH to be set, otherwise error out
  std::string filename;
  const char *output_path = getenv("ROCPROF_OUTPUT_PATH");
  if (!output_path || !*output_path) {
    throw std::runtime_error(
        "ROCPROF_OUTPUT_PATH environment variable must be set");
  }
  filename = output_path;
  if (filename.back() != '/')
    filename += '/';
  // Use the generated base filename along with ROCPROF_OUTPUT_PATH
  filename += base_filename;

  tool_data->output_filename = filename;

  // Store ROCPROF env. vars. in tool_data

  // ROCPROF_COUNTERS env. var. is a string like "pmc: counter1 counter2 ..."
  if (const char *v = getenv("ROCPROF_COUNTERS"))
    tool_data->requested_counters = v;

  if (const char *v = getenv("ROCPROF_ITERATION_MULTIPLEXING"))
    tool_data->iteration_multiplexing_mode = iteration_multiplexing_mode(v);

  // ROCPROF_KERNEL_FILTER_INCLUDE_REGEX env. var. is a regex string like
  // kernel_name_1|kernel_name_2|... Used to collect counters only for kernels
  // with names matching the regex
  if (const char *v = getenv("ROCPROF_KERNEL_FILTER_INCLUDE_REGEX"))
    tool_data->kernel_filter_include_regex = v;

  // ROCPROF_KERNEL_FILTER_RANGE env. var. is a string like "[4,7-9,...]"
  if (const char *v = getenv("ROCPROF_KERNEL_FILTER_RANGE")) {
    // Remove square brackets at the ends if present
    std::string v_str = v;
    if (!v_str.empty() && v_str.front() == '[')
      v_str.erase(0, 1);
    if (!v_str.empty() && v_str.back() == ']')
      v_str.pop_back();
    v = v_str.c_str();
    // Parse the range string into vector of pairs
    std::istringstream ss(v);
    for (std::string token; std::getline(ss, token, ',');) {
      size_t dash_pos = token.find('-');
      try {
        if (dash_pos == std::string::npos) {
          // single number
          uint64_t num = std::stoull(token);
          tool_data->kernel_filter_ranges.emplace_back(num, num);
        } else {
          // range of numbers
          uint64_t start = std::stoull(token.substr(0, dash_pos));
          uint64_t end = std::stoull(token.substr(dash_pos + 1));
          tool_data->kernel_filter_ranges.emplace_back(start, end);
        }
      } catch (const std::invalid_argument &) {
        std::cerr << "[rocprofiler-compute] [" << __FUNCTION__
                  << "] ERROR: Invalid entry in ROCPROF_KERNEL_FILTER_RANGE: "
                  << token << std::endl;
      }
    }
  }

  return tool_data;
}

rocprofiler_tool_configure_result_t *
rocprofiler_configure(uint32_t version, const char *runtime_version,
                      uint32_t priority, rocprofiler_client_id_t *id) {
  // set the client name
  id->name = "[rocprofiler-compute]";

  // compute major/minor/patch version info
  uint32_t major = version / 10000;
  uint32_t minor = (version % 10000) / 100;
  uint32_t patch = version % 100;

  // generate info string
  auto info = std::stringstream{};
  info << id->name << " [" << __FUNCTION__ << "] (priority=" << priority
       << ") is using rocprofiler-sdk v" << major << "." << minor << "."
       << patch << " (" << runtime_version << ")";

  std::clog << info.str() << std::endl;

  // init tool data
  auto tool_data = create_tool_data(id);

  // create configure data
  static auto cfg = rocprofiler_tool_configure_result_t{
      sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini,
      static_cast<void *>(
          new std::unique_ptr<tool_data_t>(std::move(tool_data)))};

  // return pointer to configure data
  return &cfg;
}
