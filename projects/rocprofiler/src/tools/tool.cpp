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

#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include "hsa_prof_str.h"
#include <hip/hip_runtime.h>
#include <hip/amd_detail/hip_prof_str.h>

#include "rocprofiler.h"
#include "rocprofiler_plugin.h"
#include <sys/syscall.h>
#include <unistd.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <bits/stdc++.h>


#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <csetjmp>
#include <exception>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <vector>
#include <mutex>
#include <thread>
#include <chrono>

#include "utils/helper.h"
#include "trace_buffer.h"

namespace fs = std::experimental::filesystem;

// Macro to check ROCProfiler calls status
#define CHECK_ROCPROFILER(call)                                                                    \
  do {                                                                                             \
    if ((call) != ROCPROFILER_STATUS_SUCCESS)                                                      \
      rocprofiler::fatal("Error: ROCProfiler API Call Error!");                                      \
  } while (false)
TRACE_BUFFER_INSTANTIATE();
namespace {

struct shmd_t {
  int command;
};
static const char* amd_sys_session_id;
static int shm_fd_sn = -1;
struct shmd_t* shmd;

std::thread wait_for_start_shm;
std::atomic<bool> amd_sys_handler{false};
std::atomic<bool> session_created{false};

[[maybe_unused]] static rocprofiler_session_id_t session_id;
static std::vector<rocprofiler_filter_id_t> filter_ids;
static std::vector<rocprofiler_buffer_id_t> buffer_ids;

void warning(const std::string& msg) { std::cerr << msg << std::endl; }

class rocprofiler_plugin_t {
 public:
  rocprofiler_plugin_t(const std::string& plugin_path) {
    plugin_handle_ = dlopen(plugin_path.c_str(), RTLD_LAZY);
    if (plugin_handle_ == nullptr) {
      warning(std::string("Warning: dlopen for ") + plugin_path + " failed: " + dlerror());
      return;
    }

    rocprofiler_plugin_write_buffer_records_ =
        reinterpret_cast<decltype(rocprofiler_plugin_write_buffer_records)*>(
            dlsym(plugin_handle_, "rocprofiler_plugin_write_buffer_records"));
    if (!rocprofiler_plugin_write_buffer_records_) return;
    rocprofiler_plugin_write_record_ = reinterpret_cast<decltype(rocprofiler_plugin_write_record)*>(
        dlsym(plugin_handle_, "rocprofiler_plugin_write_record"));
    if (!rocprofiler_plugin_write_record_) return;
    rocprofiler_plugin_finalize_ = reinterpret_cast<decltype(rocprofiler_plugin_finalize)*>(
        dlsym(plugin_handle_, "rocprofiler_plugin_finalize"));
    if (!rocprofiler_plugin_finalize_) return;
    if (auto* initialize = reinterpret_cast<decltype(rocprofiler_plugin_initialize)*>(
            dlsym(plugin_handle_, "rocprofiler_plugin_initialize"));
        initialize != nullptr)
      valid_ = initialize(ROCPROFILER_VERSION_MAJOR, ROCPROFILER_VERSION_MINOR) == 0;
  }

  ~rocprofiler_plugin_t() {
    if (is_valid()) rocprofiler_plugin_finalize_();
    if (plugin_handle_ != nullptr) dlclose(plugin_handle_);
  }

  bool is_valid() const { return valid_; }

  template <typename... Args> auto write_callback_record(Args... args) const {
    assert(is_valid());
    return rocprofiler_plugin_write_record_(std::forward<Args>(args)...);
  }
  template <typename... Args> auto write_buffer_records(Args... args) const {
    assert(is_valid());
    return rocprofiler_plugin_write_buffer_records_(std::forward<Args>(args)...);
  }

 private:
  bool valid_{false};
  void* plugin_handle_;

  decltype(rocprofiler_plugin_finalize)* rocprofiler_plugin_finalize_;
  decltype(rocprofiler_plugin_write_buffer_records)* rocprofiler_plugin_write_buffer_records_;
  decltype(rocprofiler_plugin_write_record)* rocprofiler_plugin_write_record_;
};

std::optional<rocprofiler_plugin_t> plugin;

struct hsa_api_trace_entry_t {
  std::atomic<uint32_t> valid;
  rocprofiler_record_tracer_t record;
  hsa_api_data_t api_data;

  hsa_api_trace_entry_t(rocprofiler_record_tracer_t tracer_record, const hsa_api_data_t* data)
      : valid(rocprofiler::TRACE_ENTRY_INIT) {
    record = tracer_record;
    api_data = *data;
    record.api_data_handle.handle = &api_data;
  }
  ~hsa_api_trace_entry_t() {}
};

struct roctx_trace_entry_t {
  std::atomic<rocprofiler::TraceEntryState> valid;
  rocprofiler_record_tracer_t record;

  roctx_trace_entry_t(rocprofiler_record_tracer_t tracer_record, const char* roctx_message_str)
      : valid(rocprofiler::TRACE_ENTRY_INIT) {
    record = tracer_record;
    record.name = roctx_message_str ? strdup(roctx_message_str) : nullptr;
    record.api_data_handle.handle = roctx_message_str;
  }
  ~roctx_trace_entry_t() {
    if (record.name != nullptr) free(const_cast<char*>(record.name));
  }
};

struct hip_api_trace_entry_t {
  std::atomic<uint32_t> valid;
  rocprofiler_record_tracer_t record;

  union {
    hip_api_data_t api_data;
  };

  hip_api_trace_entry_t(rocprofiler_record_tracer_t tracer_record, const char* kernel_name_str,
                        const hip_api_data_t* data)
      : valid(rocprofiler::TRACE_ENTRY_INIT) {
    record = tracer_record;

    api_data = *data;
    record.api_data_handle.handle = &api_data;
    record.name = kernel_name_str ? strdup(kernel_name_str) : nullptr;
  }
  ~hip_api_trace_entry_t() {
    if (record.name != nullptr) free(const_cast<char*>(record.name));
  }
};

rocprofiler::TraceBuffer<hip_api_trace_entry_t> hip_api_buffer(
    "HIP API", 0x200000, [](hip_api_trace_entry_t* entry) {
      assert(plugin && "plugin is not initialized");
      plugin->write_callback_record(entry->record);
    });
rocprofiler::TraceBuffer<hsa_api_trace_entry_t> hsa_api_buffer(
    "HSA API", 0x200000, [](hsa_api_trace_entry_t* entry) {
      assert(plugin && "plugin is not initialized");
      plugin->write_callback_record(entry->record);
    });
rocprofiler::TraceBuffer<roctx_trace_entry_t> roctx_trace_buffer(
    "rocTX API", 0x200000, [](roctx_trace_entry_t* entry) {
      assert(plugin && "plugin is not initialized");
      plugin->write_callback_record(entry->record);
    });

}  // namespace

uint64_t getFlushIntervalFromEnv() {
  const char* path = getenv("ROCPROFILER_FLUSH_INTERVAL");
  if (path) return std::stoll(std::string(path), nullptr, 0);
  return 0;
}

std::vector<std::string> GetCounterNames() {
  std::vector<std::string> counters;
  const char* line_c_str = getenv("ROCPROFILER_COUNTERS");
  if (line_c_str) {
    std::string line = line_c_str;
    // skip commented lines
    auto found = line.find_first_not_of(" \t");
    if (found != std::string::npos) {
      if (line[found] == '#') return {};
    }
    if (line.find("pmc") == std::string::npos) return counters;
    char seperator = ' ';
    std::string::size_type prev_pos = 0, pos = line.find(seperator, prev_pos);
    prev_pos = ++pos;
    if (pos != std::string::npos) {
      while ((pos = line.find(seperator, pos)) != std::string::npos) {
        std::string substring(line.substr(prev_pos, pos - prev_pos));
        if (substring.length() > 0 && substring != ":") {
          counters.push_back(substring);
        }
        prev_pos = ++pos;
      }
      if (!line.substr(prev_pos, pos - prev_pos).empty()) {
        counters.push_back(line.substr(prev_pos, pos - prev_pos));
      }
    }
  }
  return counters;
}

typedef std::tuple<std::vector<std::pair<rocprofiler_att_parameter_name_t, uint32_t>>,
                   std::vector<std::string>, std::vector<std::string>>
    att_parsed_input_t;

att_parsed_input_t GetATTParams() {
  std::vector<std::pair<rocprofiler_att_parameter_name_t, uint32_t>> parameters;
  std::vector<std::string> kernel_names;
  std::vector<std::string> counters_names;
  const char* path = getenv("COUNTERS_PATH");

  // List of parameters the user can set. Maxvalue is unused.
  std::unordered_map<std::string, rocprofiler_att_parameter_name_t> ATT_PARAM_NAMES{};

  ATT_PARAM_NAMES["att: TARGET_CU"] = ROCPROFILER_ATT_COMPUTE_UNIT_TARGET;
  ATT_PARAM_NAMES["SE_MASK"] = ROCPROFILER_ATT_SE_MASK;
  ATT_PARAM_NAMES["SIMD_MASK"] = ROCPROFILER_ATT_MAXVALUE;
  ATT_PARAM_NAMES["PERFCOUNTER_ID"] = ROCPROFILER_ATT_PERFCOUNTER;
  ATT_PARAM_NAMES["PERFCOUNTER"] = ROCPROFILER_ATT_PERFCOUNTER_NAME;
  ATT_PARAM_NAMES["PERFCOUNTERS_COL_PERIOD"] = ROCPROFILER_ATT_MAXVALUE;
  ATT_PARAM_NAMES["KERNEL"] = ROCPROFILER_ATT_MAXVALUE;
  ATT_PARAM_NAMES["REDUCED_MEMORY"] = ROCPROFILER_ATT_MAXVALUE;

  // Default values used for token generation.
  std::unordered_map<std::string, uint32_t> default_params = {
      {"ATT_MASK", 0x3F01}, {"TOKEN_MASK", 0x344B}, {"TOKEN_MASK2", 0xFFFFFFF}};

  bool started_att_counters = false;

  if (!path) return {parameters, kernel_names, counters_names};

  std::string line;
  std::ifstream trace_file(path);
  if (!trace_file.is_open()) {
    std::cout << "Unable to open att trace file." << std::endl;
    return {parameters, kernel_names, counters_names};
  }

  while (getline(trace_file, line)) {
    if (line.find("//") != std::string::npos)
      line = line.substr(0, line.find("//"));  // Remove comments

    auto pos = line.find('=');
    if (pos == std::string::npos) continue;

    std::string param_name = line.substr(0, pos);
    uint32_t param_value;

    if (param_name == "att: TARGET_CU") started_att_counters = true;
    if (!started_att_counters) continue;

    if (param_name == "KERNEL") {
      kernel_names.push_back(line.substr(pos + 1));
      continue;
    } else if (param_name == "PERFCOUNTER") {
      counters_names.push_back(line.substr(pos + 1));
      continue;
    } else {                                                     // param_value is a number
      try {
        auto hexa_pos = line.find("0x", pos);                    // Is it hex?
        if (hexa_pos != std::string::npos)
          param_value = stoi(line.substr(hexa_pos + 2), 0, 16);  // hexadecimal
        else
          param_value = stoi(line.substr(pos + 1), 0, 10);       // decimal
      } catch (...) {
        printf("Error: Invalid parameter value %s - (%s)\n",
               line.substr(pos + 1, line.size()).c_str(), line.c_str());
        exit(1);
      }
    }

    if (param_name == "PERFCOUNTERS_COL_PERIOD") {
      default_params["TOKEN_MASK"] |= 0x4000;
      param_value = ((param_value & 0x1F) << 8) | 0xFFFF00FF;
      parameters.push_back(std::make_pair(ROCPROFILER_ATT_PERF_CTRL, param_value));
      continue;
    } else if (param_name == "SIMD_MASK") {
      default_params["ATT_MASK"] &= ~0xF00;
      default_params["ATT_MASK"] |= (param_value << 8) & 0xF00;
      continue;
    } else if (param_name == "att: TARGET_CU") {
      default_params["ATT_MASK"] &= ~0xF;
      default_params["ATT_MASK"] |= param_value & 0xF;
    } else if (param_name == "PERFCOUNTER_ID") {
      param_value = param_value | (param_value ? (0xF << 24) : 0);
    } else if (param_name == "REDUCED_MEMORY") {
      default_params["TOKEN_MASK2"] = 0;
      continue;
    }

    if (ATT_PARAM_NAMES.find(param_name) != ATT_PARAM_NAMES.end()) {
      parameters.push_back(std::make_pair(ATT_PARAM_NAMES[param_name], param_value));
      try {
        default_params.erase(param_name);
      } catch (...) {
      };
    } else {
      printf("Error: Invalid parameter name: %s  - (%s)\nList of available params:\n",
             param_name.c_str(), line.c_str());
      for (auto& name : ATT_PARAM_NAMES) printf("%s\n", name.first.c_str());
    }
  }
  trace_file.close();

  if (!started_att_counters) return {parameters, kernel_names, counters_names};

  ATT_PARAM_NAMES["ATT_MASK"] = ROCPROFILER_ATT_MASK;
  ATT_PARAM_NAMES["TOKEN_MASK"] = ROCPROFILER_ATT_TOKEN_MASK;
  ATT_PARAM_NAMES["TOKEN_MASK2"] = ROCPROFILER_ATT_TOKEN_MASK2;

  for (auto& param : default_params)
    parameters.push_back(std::make_pair(ATT_PARAM_NAMES[param.first], param.second));

  // If no kernel names were provided, collect them all.
  // Empty string always returns true for "str.find()".
  if (kernel_names.size() == 0) kernel_names.push_back("");

  return {parameters, kernel_names, counters_names};
}

void finish() {
  for ([[maybe_unused]] rocprofiler_buffer_id_t buffer_id : buffer_ids) {
    CHECK_ROCPROFILER(rocprofiler_flush_data(session_id, buffer_id));
  }
  if (amd_sys_handler.load(std::memory_order_release)) {
    amd_sys_handler.exchange(false, std::memory_order_release);
    wait_for_start_shm.join();
    shm_unlink(std::to_string(*amd_sys_session_id).c_str());
  }
  if (session_created.load(std::memory_order_relaxed)) {
    session_created.exchange(false, std::memory_order_release);
    rocprofiler::TraceBufferBase::FlushAll();
    CHECK_ROCPROFILER(rocprofiler_terminate_session(session_id));
  }
}

// load plugins
void plugins_load() {
  // Load output plugin
  if (Dl_info dl_info; dladdr((void*)plugins_load, &dl_info) != 0) {
    const char* plugin_name = getenv("ROCPROFILER_PLUGIN_LIB");
    if (plugin_name == nullptr) {
      plugin_name = "libfile_plugin.so";
    }
    if (!plugin.emplace(fs::path(dl_info.dli_fname).replace_filename(plugin_name)).is_valid()) {
      plugin.reset();
    }
  }
}
/*
 * A callback function for synchronous trace records.
 * This function queries the api infoemation and populates the
 * api_trace buffer and adds it to the trace buffer.
 */
void sync_api_trace_callback(rocprofiler_record_tracer_t tracer_record,
                             rocprofiler_session_id_t session_id) {
  if (tracer_record.domain == ACTIVITY_DOMAIN_HIP_API) {
    size_t kernel_name_size = 0;
    char* kernel_name_c = nullptr;
    CHECK_ROCPROFILER(rocprofiler_query_hip_tracer_api_data_info_size(
        session_id, ROCPROFILER_HIP_KERNEL_NAME, tracer_record.api_data_handle,
        tracer_record.operation_id, &kernel_name_size));
    if (kernel_name_size > 1) {
      CHECK_ROCPROFILER(rocprofiler_query_hip_tracer_api_data_info(
          session_id, ROCPROFILER_HIP_KERNEL_NAME, tracer_record.api_data_handle,
          tracer_record.operation_id, &kernel_name_c));
    }
    char* data = nullptr;
    size_t size = 0;
    CHECK_ROCPROFILER(rocprofiler_query_hip_tracer_api_data_info_size(
        session_id, ROCPROFILER_HIP_API_DATA, tracer_record.api_data_handle,
        tracer_record.operation_id, &size));
    if (size > 0)
      CHECK_ROCPROFILER(rocprofiler_query_hip_tracer_api_data_info(
          session_id, ROCPROFILER_HIP_API_DATA, tracer_record.api_data_handle,
          tracer_record.operation_id, &data));
    hip_api_data_t* hip_api_data = reinterpret_cast<hip_api_data_t*>(data);
    if (tracer_record.phase == ROCPROFILER_PHASE_ENTER) {
      rocprofiler_timestamp_t timestamp;
      CHECK_ROCPROFILER(rocprofiler_get_timestamp(&timestamp));
      *hip_api_data->phase_data = timestamp.value;
      tracer_record.timestamps = rocprofiler_record_header_timestamp_t{.begin = timestamp};
    } else {
      rocprofiler_timestamp_t timestamp;
      CHECK_ROCPROFILER(rocprofiler_get_timestamp(&timestamp));
      tracer_record.timestamps = rocprofiler_record_header_timestamp_t{
          .begin = rocprofiler_timestamp_t{reinterpret_cast<uint64_t>(hip_api_data->phase_data)},
          .end = timestamp};
    }
    hip_api_trace_entry_t& entry = hip_api_buffer.Emplace(
        tracer_record, (const char*)kernel_name_c ? strdup(kernel_name_c) : nullptr, hip_api_data);
    entry.valid.store(rocprofiler::TRACE_ENTRY_COMPLETE, std::memory_order_release);
  }
  if (tracer_record.domain == ACTIVITY_DOMAIN_HSA_API) {
    char* data = nullptr;
    size_t size = 0;
    CHECK_ROCPROFILER(rocprofiler_query_hsa_tracer_api_data_info_size(
        session_id, ROCPROFILER_HSA_API_DATA, tracer_record.api_data_handle,
        tracer_record.operation_id, &size));
    CHECK_ROCPROFILER(rocprofiler_query_hsa_tracer_api_data_info(
        session_id, ROCPROFILER_HSA_API_DATA, tracer_record.api_data_handle,
        tracer_record.operation_id, &data));
    hsa_api_data_t* hsa_api_data = reinterpret_cast<hsa_api_data_t*>(data);
    if (tracer_record.phase == ROCPROFILER_PHASE_ENTER) {
      rocprofiler_timestamp_t timestamp;
      CHECK_ROCPROFILER(rocprofiler_get_timestamp(&timestamp));
      *hsa_api_data->phase_data = timestamp.value;
      tracer_record.timestamps = rocprofiler_record_header_timestamp_t{.begin = timestamp};
    } else {
      rocprofiler_timestamp_t timestamp;
      CHECK_ROCPROFILER(rocprofiler_get_timestamp(&timestamp));
      tracer_record.timestamps = rocprofiler_record_header_timestamp_t{
          .begin = rocprofiler_timestamp_t{reinterpret_cast<uint64_t>(hsa_api_data->phase_data)},
          .end = timestamp};
    }
    hsa_api_trace_entry_t& entry = hsa_api_buffer.Emplace(tracer_record, hsa_api_data);
    entry.valid.store(rocprofiler::TRACE_ENTRY_COMPLETE, std::memory_order_release);
  }
  if (tracer_record.domain == ACTIVITY_DOMAIN_ROCTX) {
    size_t roctx_message_size = 0;
    char* roctx_message_str = nullptr;
    CHECK_ROCPROFILER(rocprofiler_query_roctx_tracer_api_data_info_size(
        session_id, ROCPROFILER_ROCTX_MESSAGE, tracer_record.api_data_handle,
        tracer_record.operation_id, &roctx_message_size));
    if (roctx_message_size > 1) {
      roctx_message_str = (char*)malloc(roctx_message_size * sizeof(char));
      CHECK_ROCPROFILER(rocprofiler_query_roctx_tracer_api_data_info(
          session_id, ROCPROFILER_ROCTX_MESSAGE, tracer_record.api_data_handle,
          tracer_record.operation_id, &roctx_message_str));
      if (roctx_message_str)
        roctx_message_str ? std::string(strdup(roctx_message_str)).c_str() : nullptr;
    }
    rocprofiler_timestamp_t timestamp;
    CHECK_ROCPROFILER(rocprofiler_get_timestamp(&timestamp));
    tracer_record.timestamps = rocprofiler_record_header_timestamp_t{.begin = timestamp};
    roctx_trace_entry_t& entry = roctx_trace_buffer.Emplace(tracer_record, roctx_message_str);
    entry.valid.store(rocprofiler::TRACE_ENTRY_COMPLETE, std::memory_order_release);
  }
}

void wait_for_amdsys() {
  while (amd_sys_handler.load(std::memory_order_relaxed)) {
    shm_fd_sn = shm_open(amd_sys_session_id, O_RDONLY, 0666);
    if (shm_fd_sn < 0) {
      continue;
    }
    shmd = reinterpret_cast<struct shmd_t*>(mmap(0, 1024, PROT_READ, MAP_SHARED, shm_fd_sn, 0));
    bool flag{false};
    if (shmd && (sizeof(shmd->command) == sizeof(int))) {
      switch (shmd->command) {
        // Start
        case 4: {
          printf("AMDSYS:: Starting Tools Session...\n");
          CHECK_ROCPROFILER(rocprofiler_start_session(session_id));
          session_created.exchange(true, std::memory_order_release);
          break;
        }
        // Stop
        case 5: {
          if (session_created.load(std::memory_order_relaxed)) {
            printf("AMDSYS:: Stopping Tools Session...\n");
            session_created.exchange(false, std::memory_order_release);
            CHECK_ROCPROFILER(rocprofiler_terminate_session(session_id));
            for ([[maybe_unused]] rocprofiler_buffer_id_t buffer_id : buffer_ids) {
              CHECK_ROCPROFILER(rocprofiler_flush_data(session_id, buffer_id));
            }
          }
          break;
        }
        // Exit
        case 6: {
          printf("AMDSYS:: Exiting the Application..\n");
          if (session_created.load(std::memory_order_relaxed)) {
            printf("AMDSYS:: Stopping Tools Session...\n");
            session_created.exchange(false, std::memory_order_release);
            CHECK_ROCPROFILER(rocprofiler_terminate_session(session_id));
            for ([[maybe_unused]] rocprofiler_buffer_id_t buffer_id : buffer_ids) {
              CHECK_ROCPROFILER(rocprofiler_flush_data(session_id, buffer_id));
            }
          }
          amd_sys_handler.exchange(false, std::memory_order_release);
          flag = true;
        }
      }
    }
    shm_unlink(amd_sys_session_id);
    if (flag) break;
  }
}

static int info_callback(const rocprofiler_counter_info_t info, const char* gpu_name,
                         uint32_t gpu_index) {
  fprintf(stdout, "\n  %s:%u : %s : %s\n", gpu_name, gpu_index, info.name, info.description);
  if (info.expression != nullptr) {
    fprintf(stdout, "      %s = %s\n", info.name, info.expression);
  } else {
    if (info.instances_count > 1) fprintf(stdout, "[0-%u]", info.instances_count - 1);
    fprintf(stdout, " : %s\n", info.description);
    fprintf(stdout, "      block %s can only handle %u counters at a time\n", info.block_name,
            info.block_counters);
  }
  fflush(stdout);
  return 1;
}

extern "C" {

// The HSA_AMD_TOOL_PRIORITY variable must be a constant value type
// initialized by the loader itself, not by code during _init. 'extern const'
// seems do that although that is not a guarantee.
ROCPROFILER_EXPORT extern const uint32_t HSA_AMD_TOOL_PRIORITY = 1025;

/**
@brief Callback function called upon loading the HSA.
The function updates the core api table function pointers to point to the
interceptor functions in this file.
*/
ROCPROFILER_EXPORT bool OnLoad(void* table, uint64_t runtime_version, uint64_t failed_tool_count,
                               const char* const* failed_tool_names) {
  if (rocprofiler_version_major() != ROCPROFILER_VERSION_MAJOR ||
      rocprofiler_version_minor() < ROCPROFILER_VERSION_MINOR) {
    warning("the ROCProfiler API version is not compatible with this tool");
    return true;
  }

  std::atexit(finish);

  amd_sys_session_id = getenv("ROCPROFILER_ENABLE_AMDSYS");
  if (amd_sys_session_id != nullptr) {
    printf("AMDSYS Session Started!\n");
    wait_for_start_shm = std::thread{wait_for_amdsys};
    amd_sys_handler.exchange(true, std::memory_order_release);
  }

  CHECK_ROCPROFILER(rocprofiler_initialize());

  // Printing out info
  char* info_symb = getenv("ROCPROFILER_COUNTER_LIST");
  if (info_symb != nullptr) {
    if (*info_symb == 'b')
      printf("Basic HW counters:\n");
    else
      printf("Derived metrics:\n");
    CHECK_ROCPROFILER(rocprofiler_iterate_counters(info_callback));
    exit(1);
  }

  // load the plugins
  plugins_load();


  std::vector<rocprofiler_tracer_activity_domain_t> apis_requested;

  if (getenv("ROCPROFILER_HIP_API_TRACE")) apis_requested.emplace_back(ACTIVITY_DOMAIN_HIP_API);
  if (getenv("ROCPROFILER_HIP_ACTIVITY_TRACE"))
    apis_requested.emplace_back(ACTIVITY_DOMAIN_HIP_OPS);
  if (getenv("ROCPROFILER_HSA_API_TRACE")) apis_requested.emplace_back(ACTIVITY_DOMAIN_HSA_API);
  if (getenv("ROCPROFILER_HSA_ACTIVITY_TRACE"))
    apis_requested.emplace_back(ACTIVITY_DOMAIN_HSA_OPS);
  if (getenv("ROCPROFILER_ROCTX_TRACE")) apis_requested.emplace_back(ACTIVITY_DOMAIN_ROCTX);

  std::vector<std::string> counters = GetCounterNames();
  std::vector<const char*> counters_;

  if (counters.size() > 0) {
    printf("ROCProfilerV2: Collecting the following counters:\n");
    for (size_t i = 0; i < counters.size(); i++) {
      counters_.emplace_back(counters.at(i).c_str());
      printf("- %s\n", counters_.back());
    }
  }
  // ATT Parameters
  std::vector<rocprofiler_att_parameter_t> parameters;
  std::vector<std::pair<rocprofiler_att_parameter_name_t, uint32_t>> params;
  std::vector<std::string> kernel_names;
  std::vector<std::string> att_counters_names;
  std::tie(params, kernel_names, att_counters_names) = GetATTParams();

  for (auto& kv_pair : params)
    parameters.emplace_back(rocprofiler_att_parameter_t{kv_pair.first, kv_pair.second});
  for (std::string& name : att_counters_names) {
    rocprofiler_att_parameter_t param;
    param.parameter_name = ROCPROFILER_ATT_PERFCOUNTER_NAME;
    param.counter_name = name.c_str();
    parameters.emplace_back(param);
  }

  CHECK_ROCPROFILER(rocprofiler_create_session(ROCPROFILER_KERNEL_REPLAY_MODE, &session_id));

  bool want_pc_sampling = getenv("ROCPROFILER_PC_SAMPLING");

  std::vector<rocprofiler_filter_kind_t> filters_requested;
  if (((counters.size() == 0 && parameters.size() == 0) && (apis_requested.size() == 0 || getenv("ROCPROFILER_KERNEL_TRACE")))
      || want_pc_sampling /* PC sampling needs a profiler, even it's doing
                             nothing */)
    filters_requested.emplace_back(ROCPROFILER_DISPATCH_TIMESTAMPS_COLLECTION);
  if (want_pc_sampling) {
    filters_requested.emplace_back(ROCPROFILER_PC_SAMPLING_COLLECTION);
  }
  if (counters.size() > 0) filters_requested.emplace_back(ROCPROFILER_COUNTERS_COLLECTION);
  if (apis_requested.size() > 0) filters_requested.emplace_back(ROCPROFILER_API_TRACE);
  if (parameters.size() > 0) filters_requested.emplace_back(ROCPROFILER_ATT_TRACE_COLLECTION);

  for (rocprofiler_filter_kind_t filter_kind : filters_requested) {
    switch (filter_kind) {
      case ROCPROFILER_COUNTERS_COLLECTION: {
        rocprofiler_buffer_id_t buffer_id;
        CHECK_ROCPROFILER(rocprofiler_create_buffer(
            session_id,
            [](const rocprofiler_record_header_t* record,
               const rocprofiler_record_header_t* end_record, rocprofiler_session_id_t session_id,
               rocprofiler_buffer_id_t buffer_id) {
              if (plugin) plugin->write_buffer_records(record, end_record, session_id, buffer_id);
            },
            1 << 20, &buffer_id));
        buffer_ids.emplace_back(buffer_id);
        printf("Enabling Counter Collection\n");
        rocprofiler_filter_id_t filter_id;
        [[maybe_unused]] rocprofiler_filter_property_t property = {};
        CHECK_ROCPROFILER(rocprofiler_create_filter(
            session_id, filter_kind, rocprofiler_filter_data_t{.counters_names = &counters_[0]},
            counters_.size(), &filter_id, property));
        CHECK_ROCPROFILER(rocprofiler_set_filter_buffer(session_id, filter_id, buffer_id));
        filter_ids.emplace_back(filter_id);
        break;
      }
      case ROCPROFILER_DISPATCH_TIMESTAMPS_COLLECTION: {
        rocprofiler_buffer_id_t buffer_id;
        CHECK_ROCPROFILER(rocprofiler_create_buffer(
            session_id,
            [](const rocprofiler_record_header_t* record,
               const rocprofiler_record_header_t* end_record, rocprofiler_session_id_t session_id,
               rocprofiler_buffer_id_t buffer_id) {
              if (plugin) plugin->write_buffer_records(record, end_record, session_id, buffer_id);
            },
            1 << 20, &buffer_id));
        buffer_ids.emplace_back(buffer_id);
        rocprofiler_filter_id_t filter_id;
        [[maybe_unused]] rocprofiler_filter_property_t property = {};
        CHECK_ROCPROFILER(rocprofiler_create_filter(
            session_id, filter_kind, rocprofiler_filter_data_t{}, 0, &filter_id, property));
        CHECK_ROCPROFILER(rocprofiler_set_filter_buffer(session_id, filter_id, buffer_id));
        filter_ids.emplace_back(filter_id);
        break;
      }
      case ROCPROFILER_API_TRACE: {
        rocprofiler_buffer_id_t buffer_id;
        CHECK_ROCPROFILER(rocprofiler_create_buffer(
            session_id,
            [](const rocprofiler_record_header_t* record,
               const rocprofiler_record_header_t* end_record, rocprofiler_session_id_t session_id,
               rocprofiler_buffer_id_t buffer_id) {
              if (plugin) plugin->write_buffer_records(record, end_record, session_id, buffer_id);
            },
            1 << 20, &buffer_id));
        buffer_ids.emplace_back(buffer_id);
        printf("Enabling API Tracing\n");
        rocprofiler_filter_id_t filter_id;
        [[maybe_unused]] rocprofiler_filter_property_t property = {};
        CHECK_ROCPROFILER(rocprofiler_create_filter(session_id, filter_kind,
                                                    rocprofiler_filter_data_t{&apis_requested[0]},
                                                    apis_requested.size(), &filter_id, property));
        CHECK_ROCPROFILER(rocprofiler_set_filter_buffer(session_id, filter_id, buffer_id));
        CHECK_ROCPROFILER(rocprofiler_set_api_trace_sync_callback(session_id, filter_id,
                                                                  sync_api_trace_callback));
        filter_ids.emplace_back(filter_id);
        break;
      }
      case ROCPROFILER_ATT_TRACE_COLLECTION: {
        rocprofiler_buffer_id_t buffer_id;
        CHECK_ROCPROFILER(rocprofiler_create_buffer(
            session_id,
            [](const rocprofiler_record_header_t* record,
               const rocprofiler_record_header_t* end_record, rocprofiler_session_id_t session_id,
               rocprofiler_buffer_id_t buffer_id) {
              if (plugin) plugin->write_buffer_records(record, end_record, session_id, buffer_id);
            },
            1 << 20, &buffer_id));
        buffer_ids.emplace_back(buffer_id);
        printf("Enabling ATT Tracing\n");
        rocprofiler_filter_id_t filter_id;

        std::vector<const char*> kernel_names_c;
        for (auto& name : kernel_names) kernel_names_c.push_back(name.data());

        rocprofiler_filter_property_t property = {};
        property.kind = ROCPROFILER_FILTER_KERNEL_NAMES;
        property.data_count = kernel_names_c.size();
        property.name_regex = kernel_names_c.data();

        CHECK_ROCPROFILER(
            rocprofiler_create_filter(session_id, ROCPROFILER_ATT_TRACE_COLLECTION,
                                      rocprofiler_filter_data_t{.att_parameters = &parameters[0]},
                                      parameters.size(), &filter_id, property));
        CHECK_ROCPROFILER(rocprofiler_set_filter_buffer(session_id, filter_id, buffer_id));
        filter_ids.emplace_back(filter_id);
        break;
      }
      case ROCPROFILER_PC_SAMPLING_COLLECTION: {
        rocprofiler_buffer_id_t buffer_id;
        CHECK_ROCPROFILER(rocprofiler_create_buffer(
            session_id,
            [](const rocprofiler_record_header_t* record,
               const rocprofiler_record_header_t* end_record, rocprofiler_session_id_t session_id,
               rocprofiler_buffer_id_t buffer_id) {
              if (plugin) plugin->write_buffer_records(record, end_record, session_id, buffer_id);
            },
            1 << 20, &buffer_id));
        buffer_ids.emplace_back(buffer_id);
        puts("Enabling PC sampling");
        rocprofiler_filter_id_t filter_id;
        [[maybe_unused]] rocprofiler_filter_property_t property = {};
        CHECK_ROCPROFILER(rocprofiler_create_filter(
            session_id, filter_kind, rocprofiler_filter_data_t{}, 0, &filter_id, property));
        CHECK_ROCPROFILER(rocprofiler_set_filter_buffer(session_id, filter_id, buffer_id));
        filter_ids.emplace_back(filter_id);
        break;
      }
      default:
        warning("Not available for profiling or tracing");
    }
  }

  if (getenv("ROCPROFILER_ENABLE_AMDSYS") == nullptr) {
    CHECK_ROCPROFILER(rocprofiler_start_session(session_id));
    session_created.exchange(true, std::memory_order_release);
  }
  return true;
}

/**
@brief Callback function upon unloading the HSA.
*/
ROCPROFILER_EXPORT void OnUnload() { printf("\n\nTool is getting unloaded\n\n"); }

}  // extern "C"
