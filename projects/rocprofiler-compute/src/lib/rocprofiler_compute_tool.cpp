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

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <cxxabi.h>
#include <fstream>
#include <iostream>
#include <mutex>
#include <random>
#include <regex>
#include <set>
#include <shared_mutex>
#include <sstream>
#include <string>
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

// Struct to store a single counter info record
struct counter_info_record_t {
  uint64_t dispatch_id;
  uint64_t kernel_id;
  uint64_t counter_id;
  std::string counter_name;
  double counter_value;
};

// Tool data struct, now includes a vector of counter_info_record_t
struct tool_data_t {
  std::mutex mut{};
  std::ostream *output_stream{nullptr};
  std::unordered_map<uint64_t, std::string> counter_id_name_map{};
  std::string requested_counters{};
  std::string kernel_filter_include_regex{};
  std::vector<std::pair<uint64_t, uint64_t>> kernel_filter_ranges{};
  std::vector<counter_info_record_t> counter_records;
  std::set<uint64_t> target_kernel_ids{};
};

using kernel_symbol_data_t =
    rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;

rocprofiler_context_id_t &get_client_ctx() {
  static rocprofiler_context_id_t ctx{0};
  return ctx;
}

void record_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                     rocprofiler_counter_record_t *record_data,
                     size_t record_count,
                     rocprofiler_user_data_t /* user_data */,
                     void *callback_data_args) {
  tool_data_t *tool;
  {
    std::lock_guard<std::mutex> lock(
        static_cast<tool_data_t *>(callback_data_args)->mut);
    tool = static_cast<tool_data_t *>(callback_data_args);
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
                                 dispatch_data.dispatch_info.kernel_id,
                                 counter_id.handle,
                                 tool->counter_id_name_map[counter_id.handle],
                                 record_data[i].counter_value};
    {
      std::lock_guard<std::mutex> lock(tool->mut);
      tool->counter_records.push_back(std::move(record));
    }
  }
}

// The function extracts the kernel name from
// input string. By using the iterators it finds the
// window in the string which contains only the kernel name.
// For example 'Foo<int, float>::foo(a[], int (int))' -> 'foo'
std::string truncate_name(std::string_view name) {
  auto rit = name.rbegin();
  auto rend = name.rend();
  uint32_t counter = 0;
  char open_token = 0;
  char close_token = 0;
  while (rit != rend) {
    if (counter == 0) {
      switch (*rit) {
      case ')':
        counter = 1;
        open_token = ')';
        close_token = '(';
        break;
      case '>':
        counter = 1;
        open_token = '>';
        close_token = '<';
        break;
      case ']':
        counter = 1;
        open_token = ']';
        close_token = '[';
        break;
      case ' ':
        ++rit;
        continue;
      }
      if (counter == 0)
        break;
    } else {
      if (*rit == open_token)
        counter++;
      if (*rit == close_token)
        counter--;
    }
    ++rit;
  }
  auto rbeg = rit;
  while ((rit != rend) && (*rit != ' ') && (*rit != ':'))
    rit++;
  return std::string{name.substr(rend - rit, rit - rbeg)};
}

std::string cxa_demangle(std::string_view _mangled_name, int *_status) {
  // return the mangled since there is no buffer
  if (_mangled_name.empty()) {
    *_status = -2;
    return std::string{};
  }

  auto _demangled_name = std::string{_mangled_name};

  // PARAMETERS to __cxa_demangle
  //  mangled_name:
  //      A NULL-terminated character string containing the name to be
  //      demangled.
  //  buffer:
  //      A region of memory, allocated with malloc, of *length bytes, into
  //      which the demangled name is stored. If output_buffer is not long
  //      enough, it is expanded using realloc. output_buffer may instead be
  //      NULL; in that case, the demangled name is placed in a region of memory
  //      allocated with malloc.
  //  _buflen:
  //      If length is non-NULL, the length of the buffer containing the
  //      demangled name is placed in *length.
  //  status:
  //      *status is set to one of the following values
  size_t _demang_len = 0;
  char *_demang = abi::__cxa_demangle(_demangled_name.c_str(), nullptr,
                                      &_demang_len, _status);
  switch (*_status) {
  //  0 : The demangling operation succeeded.
  // -1 : A memory allocation failure occurred.
  // -2 : mangled_name is not a valid name under the C++ ABI mangling rules.
  // -3 : One of the arguments is invalid.
  case 0: {
    if (_demang)
      _demangled_name = std::string{_demang};
    break;
  }
  case -1: {
    std::clog << "memory allocation failure occurred demangling "
              << _demangled_name << std::endl;
    break;
  }
  case -2: {
    break;
  }
  case -3: {
    std::clog << "Invalid argument in: (\"" << _demangled_name
              << "\", nullptr, nullptr, " << static_cast<void *>(_status) << ")"
              << std::endl;
    break;
  }
  default:
    break;
  };

  // if it "demangled" but the length is zero, set the status to -2
  if (_demang_len == 0 && *_status == 0)
    *_status = -2;

  // free allocated buffer
  ::free(_demang);
  return _demangled_name;
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
    auto kernel_name = cxa_demangle(data->kernel_name, &demangle_status);
    kernel_name = truncate_name(kernel_name);

    // check if regex can be found in kernel name matches regex from tool data,
    // if matches store kernel id
    auto *tool = static_cast<tool_data_t *>(callback_data);
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
 * Callback from rocprofiler when an kernel dispatch is enqueued into the HSA
 * queue. rocprofiler_counter_config_id_t* is a return to specify what counters
 * to collect for this dispatch (dispatch_packet).
 */
void dispatch_callback(
    rocprofiler_dispatch_counting_service_data_t dispatch_data,
    rocprofiler_counter_config_id_t *config,
    rocprofiler_user_data_t * /*user_data*/, void *callback_data_args) {
  /**
   * This simple example uses the same profile counter set for all agents.
   * We store profile in a cache to prevent constructing many identical
   * profiles. We first check the cache to see if we have already constructed a
   * profile for the agent. If we have, return it. Otherwise, construct a new
   * profile.
   */

  auto kernel_id = dispatch_data.dispatch_info.kernel_id;

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
  tool_data_t *tool;
  {
    std::lock_guard<std::mutex> lock(
        static_cast<tool_data_t *>(callback_data_args)->mut);
    tool = static_cast<tool_data_t *>(callback_data_args);
  }

  // kernel filtering
  if (!is_targetted_dispatch(tool, kernel_id, kernel_iteration)) {
    return;
  }

  static std::shared_mutex m_mutex = {};
  static std::unordered_map<uint64_t, rocprofiler_counter_config_id_t>
      profile_cache = {};

  // check cache for existing profile for this agent
  auto search_cache = [&]() {
    if (auto pos =
            profile_cache.find(dispatch_data.dispatch_info.agent_id.handle);
        pos != profile_cache.end()) {
      *config = pos->second;
      return true;
    }
    return false;
  };
  {
    auto rlock = std::shared_lock{m_mutex};
    if (search_cache())
      return;
  }

  // get write lock to update cache
  auto wlock = std::unique_lock{m_mutex};
  if (search_cache())
    return;

  // get counters to collect
  std::set<std::string> counters_to_collect;
  const std::string &counters_str = tool->requested_counters;
  if (!counters_str.empty()) {
    auto pos = counters_str.find(':');
    if (pos != std::string::npos) {
      std::istringstream ss(counters_str.substr(pos + 1));
      for (std::string token; ss >> token;)
        counters_to_collect.insert(token);
    }
  }

  // Get available counters for this agent
  std::vector<rocprofiler_counter_id_t> gpu_counters;
  ROCPROFILER_CALL(
      rocprofiler_iterate_agent_supported_counters(
          dispatch_data.dispatch_info.agent_id,
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

  // Identify counters requested to collect which are available
  std::vector<rocprofiler_counter_id_t> collect_counters;
  std::vector<std::string> collect_counters_names;
  for (auto &counter : gpu_counters) {
    rocprofiler_counter_info_v0_t info;
    ROCPROFILER_CALL(rocprofiler_query_counter_info(
                         counter, ROCPROFILER_COUNTER_INFO_VERSION_0,
                         static_cast<void *>(&info)),
                     "query counter info");
    if (counters_to_collect.count(std::string(info.name)) > 0) {
      collect_counters.push_back(counter);
      collect_counters_names.push_back(std::string(info.name));
      tool->counter_id_name_map[counter.handle] = std::string(info.name);
    }
  }

  // Log unsupported counters in a concise, comma-separated line
  std::vector<std::string> unsupported_counters;
  for (const auto &requested : counters_to_collect) {
    if (std::find(collect_counters_names.begin(), collect_counters_names.end(),
                  requested) == collect_counters_names.end()) {
      unsupported_counters.push_back(requested);
    }
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

  // Create a collection profile for the counters
  rocprofiler_counter_config_id_t profile = {.handle = 0};
  ROCPROFILER_CALL(
      rocprofiler_create_counter_config(dispatch_data.dispatch_info.agent_id,
                                        collect_counters.data(),
                                        collect_counters.size(), &profile),
      "construct profile cfg");

  // cache the profile for this agent
  profile_cache.emplace(dispatch_data.dispatch_info.agent_id.handle, profile);
  // Return the profile to collect those counters for this dispatch
  *config = profile;
}

int tool_init(rocprofiler_client_finalize_t, void *user_data) {
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

  // Write collected counter records and clean up
  if (auto *os = tool_data->output_stream) {
    for (const auto &r : tool_data->counter_records)
      *os << r.dispatch_id << ',' << r.counter_id << ',' << r.counter_name
          << ',' << r.counter_value << '\n';
    os->flush();
    if (os != &std::cout && os != &std::cerr)
      delete os;
  }
}

void tool_fini(void *user_data) {
  assert(user_data);
  std::clog << "In tool fini\n";
  rocprofiler_stop_context(get_client_ctx());

  auto *tool_data = static_cast<tool_data_t *>(user_data);
  generate_output(tool_data);

  delete tool_data;
}

} // namespace

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

  auto *tool_data = new tool_data_t{};

  // Generate a unique output filename using a random hex string (no libuuid
  // dependency)
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<uint32_t> dis(0, 0xFFFFFFFF);
  std::stringstream filename_ss;
  filename_ss << std::hex << dis(gen);
  std::string base_filename =
      "counter_collection_" + filename_ss.str().substr(0, 8) + ".csv";

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

  // Set output stream to file
  auto *ofs = new std::ofstream{filename};
  if (!ofs->is_open()) {
    delete ofs;
    throw std::runtime_error("Failed to open output file: " + filename);
  }
  tool_data->output_stream = ofs;

  // Write header at the beginning of the file
  *tool_data->output_stream
      << "dispatch_id,counter_id,counter_name,counter_value\n";
  tool_data->output_stream->flush();

  // Write to clog the path of the logging file
  std::clog << id->name << " [" << __FUNCTION__
            << "] Logging counter collection to: " << filename << std::endl;

  // Store ROCPROF env. vars. in tool_data

  // ROCPROF_COUNTERS env. var. is a string like "pmc: counter1 counter2 ..."
  if (const char *v = getenv("ROCPROF_COUNTERS"))
    tool_data->requested_counters = v;

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

  // create configure data
  static auto cfg = rocprofiler_tool_configure_result_t{
      sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini,
      static_cast<void *>(tool_data)};

  // return pointer to configure data
  return &cfg;
}

#if __has_include(<rocprofiler-sdk/experimental/registration.h>)
#include <rocprofiler-sdk/experimental/registration.h>

int tool_attach(rocprofiler_client_detach_t /*detach_func*/,
                rocprofiler_context_id_t *context_ids,
                uint64_t context_ids_length, void * /*tool_data*/) {
  for (uint64_t i = 0; i < context_ids_length; ++i) {
    if (int status = 0;
        rocprofiler_context_is_active(context_ids[i], &status) ==
            ROCPROFILER_STATUS_SUCCESS &&
        status == 0) {
      std::clog << "Attach mode: starting context ID " << context_ids[i].handle;
      ROCPROFILER_CALL(rocprofiler_start_context(context_ids[i]),
                       "failed to start received context");
    }
  }

  return 0;
}

void tool_detach(void *user_data) {
  std::clog << "In tool detach\n";

  auto *tool_data = static_cast<tool_data_t *>(user_data);
  generate_output(tool_data);
}

rocprofiler_tool_configure_attach_result_t *rocprofiler_configure_attach(
    uint32_t /*version*/, const char * /*runtime_version*/,
    uint32_t /*priority*/, rocprofiler_client_id_t * /*id*/) {
  // This function is called right after rocprofiler_configure with the same
  // parameters. The data returned is only used when attaching to a running
  // process.

  std::clog << "[rocprofiler-compute] [" << __FUNCTION__
            << "] Attach mode configuration called." << std::endl;
  // create configure data using experimental struct with attach/detach support
  static auto cfg = rocprofiler_tool_configure_attach_result_t{
      sizeof(rocprofiler_tool_configure_attach_result_t), &tool_attach,
      &tool_detach, nullptr};

  // return pointer to configure data
  return &cfg;
}

#endif
