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
#include "core/session/tracer/src/roctracer.h"
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
#include "core/session/att/att.h"
#include "src/utils/filesystem.hpp"


struct PluginHeaderPacket {
  std::string plugin_path;
  void* userdata;
};

#define SLEEP_CYCLE_LENGTH 100l

namespace fs = rocprofiler::common::filesystem;

// Macro to check ROCProfiler calls status
#define CHECK_ROCPROFILER(call)                                                                    \
  do {                                                                                             \
    if ((call) != ROCPROFILER_STATUS_SUCCESS)                                                      \
      {std::cerr << "ERROR IN " << __LINE__ << std::endl; abort(); } \
  } while (false)
TRACE_BUFFER_INSTANTIATE();
namespace {

struct shmd_t {
  int command;
};
static const char* roc_sys_session_id;

uint64_t flush_interval, trace_time_length, trace_delay, trace_interval;

std::thread wait_for_start_shm, flush_thread, trace_period_thread;
std::atomic<bool> roc_sys_handler{false};
std::atomic<bool> session_created{false};
std::atomic<bool> trace_period_thread_control{false};
std::atomic<bool> flush_thread_control{false};
std::atomic<bool> rocprof_started{false};

[[maybe_unused]] static rocprofiler_session_id_t session_id;
static std::vector<rocprofiler_filter_id_t> filter_ids;
static std::vector<rocprofiler_buffer_id_t> buffer_ids;
static std::vector<const char*> counter_names;

void warning(const std::string& msg) { std::cerr << msg << std::endl; }

class rocprofiler_plugin_t {
 public:
  rocprofiler_plugin_t(const PluginHeaderPacket& data) {
    plugin_handle_ = dlopen(data.plugin_path.c_str(), RTLD_LAZY);
    if (plugin_handle_ == nullptr) {
      warning(std::string("Warning: dlopen for ") + data.plugin_path + " failed: " + dlerror());
      return;
    }

    rocprofiler_plugin_write_buffer_records_ =
        reinterpret_cast<decltype(rocprofiler_plugin_write_buffer_records)*>(
            dlsym(plugin_handle_, "rocprofiler_plugin_write_buffer_records"));
    if (!rocprofiler_plugin_write_buffer_records_) return;
    rocprofiler_plugin_write_record_ = reinterpret_cast<decltype(rocprofiler_plugin_write_record)*>(
        dlsym(plugin_handle_, "rocprofiler_plugin_write_record"));
    if (!rocprofiler_plugin_write_record_) return;
    rocprofiler_plugin_finalize_fn = reinterpret_cast<decltype(rocprofiler_plugin_finalize)*>(
        dlsym(plugin_handle_, "rocprofiler_plugin_finalize"));
    if (!rocprofiler_plugin_finalize_fn) return;

    initialize_fn = reinterpret_cast<decltype(initialize_fn)>(dlsym(plugin_handle_, "rocprofiler_plugin_initialize"));
    initialize(data.userdata);
  }

  void initialize(void* userdata)
  {
    if (initialize_fn)
      valid_ |= initialize_fn(ROCPROFILER_VERSION_MAJOR, ROCPROFILER_VERSION_MINOR, userdata) == 0;
  }

  ~rocprofiler_plugin_t() {
    if (is_valid()) rocprofiler_plugin_finalize_fn();
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

  void setPluginName(std::string name) { plugin_name_ = name; }

  std::string& getPluginName() { return plugin_name_; }

  decltype(rocprofiler_plugin_finalize)* rocprofiler_plugin_finalize_fn;

 private:
  bool valid_{false};
  void* plugin_handle_;
  std::string plugin_name_;
  decltype(rocprofiler_plugin_initialize)* initialize_fn = nullptr;

  decltype(rocprofiler_plugin_write_buffer_records)* rocprofiler_plugin_write_buffer_records_;
  decltype(rocprofiler_plugin_write_record)* rocprofiler_plugin_write_record_;
};

rocprofiler_plugin_t* plugin = nullptr;

}  // namespace

void getFlushIntervalFromEnv() {
  const char* path = getenv("ROCPROFILER_FLUSH_INTERVAL");
  if (path)
    flush_interval = std::stoll(std::string(path), nullptr, 0);
  else
    flush_interval = 0;
}

void getTracePeriodFromEnv() {
  trace_time_length = 0;
  trace_interval = INT_MAX;
  trace_delay = 0;
  const char* path = getenv("ROCPROFILER_TRACE_PERIOD");
  if (path) {
    try {
      std::string str = path;
      size_t first_pos = str.find(':');
      size_t second_pos = str.rfind(':');
      if (first_pos == second_pos) second_pos = std::string::npos;  // Second ':' does not exists

      trace_delay = std::stoll(str.substr(0, first_pos), nullptr, 0);
      trace_time_length =
          std::stoll(str.substr(first_pos + 1, second_pos), nullptr, 0);  // can throw
      if (second_pos < str.size() - 1)
        trace_interval = std::stoll(str.substr(second_pos + 1), nullptr, 0);
      if (trace_interval < trace_time_length) throw std::exception();
    } catch (std::exception& e) {
      std::cout << "Invalid trace period format: " << path << '\n';
    }
    std::cout << "Setting delay:" << trace_delay << ", length:" << trace_time_length
              << ", interval:" << trace_interval << std::endl;
  }
}

std::vector<std::string> GetCounterNames() {
  std::vector<std::string> counters;
  const char* line_c_str = getenv("ROCPROFILER_COUNTERS");
  if (line_c_str) {
    std::string line = line_c_str;
    rocprofiler::validate_counters_format(counters, line);
  }
  return counters;
}

struct att_parsed_input_t {
  std::vector<std::pair<rocprofiler_att_parameter_name_t, uint32_t>> params{};
  std::vector<std::string> kernel_names{};
  std::vector<std::string> counters_names{};
  std::vector<std::pair<uint64_t, uint64_t>> dispatch_ids{};
  rocprofiler::att_header_packet_t header{.raw = 0};
};

static int GetMpRank() {
  std::vector<const char*> mpivars = {"MPI_RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK"};
  for (const char* envvar : mpivars)
    if (const char* env = getenv(envvar)) return atoi(env);
  return -1;
}

att_parsed_input_t GetATTParams() {
  att_parsed_input_t ret{};
  const char* path = getenv("COUNTERS_PATH");
  if (!path) return ret;

  // List of parameters the user can set. Maxvalue is unused.
  std::unordered_map<std::string, rocprofiler_att_parameter_name_t> ATT_PARAM_NAMES{};

  ATT_PARAM_NAMES["TARGET_CU"] = ROCPROFILER_ATT_COMPUTE_UNIT;
  ATT_PARAM_NAMES["SE_MASK"] = ROCPROFILER_ATT_SE_MASK;
  ATT_PARAM_NAMES["VMID_MASK"] = ROCPROFILER_ATT_VMID_MASK;
  ATT_PARAM_NAMES["SIMD_SELECT"] = ROCPROFILER_ATT_SIMD_SELECT;

  ATT_PARAM_NAMES["PERFCOUNTER_ID"] = ROCPROFILER_ATT_PERFCOUNTER;
  ATT_PARAM_NAMES["PERFCOUNTER"] = ROCPROFILER_ATT_PERFCOUNTER_NAME;
  ATT_PARAM_NAMES["PERFCOUNTER_MASK"] = ROCPROFILER_ATT_PERF_MASK;
  ATT_PARAM_NAMES["PERFCOUNTERS_CTRL"] = ROCPROFILER_ATT_PERF_CTRL;
  ATT_PARAM_NAMES["OCCUPANCY"] = ROCPROFILER_ATT_OCCUPANCY;

  ATT_PARAM_NAMES["KERNEL"] = ROCPROFILER_ATT_MAXVALUE;
  ATT_PARAM_NAMES["BUFFER_SIZE"] = ROCPROFILER_ATT_BUFFER_SIZE;
  ATT_PARAM_NAMES["ISA_CAPTURE_MODE"] = ROCPROFILER_ATT_CAPTURE_MODE;

  ATT_PARAM_NAMES["LEGACY_ATT_MASK"] = ROCPROFILER_ATT_MASK;
  ATT_PARAM_NAMES["LEGACY_TOKEN_MASK"] = ROCPROFILER_ATT_TOKEN_MASK;
  ATT_PARAM_NAMES["LEGACY_TOKEN_MASK2"] = ROCPROFILER_ATT_TOKEN_MASK2;

  // Default values used for token generation.
  std::unordered_map<std::string, uint32_t> default_params = {
      {"SE_MASK", 0x111111},       // One every 4 SEs, by default
      {"SIMD_SELECT", 0x3},        // 0x3 works for both gfx9 and Navi
      {"BUFFER_SIZE", 0xA000000},  // 160MB
      {"ISA_CAPTURE_MODE", static_cast<uint32_t>(ROCPROFILER_CAPTURE_COPY_MEMORY)}};

  std::ifstream trace_file(path);
  if (!trace_file.is_open()) {
    std::cout << "Unable to open att trace file." << std::endl;
    return ret;
  }

  ret.header.enable = 1;
  ret.header.DSIMDM = default_params["SIMD_SELECT"];
  int MPI_RANK = GetMpRank();

  bool started_att_counters = false;
  std::string line;
  while (getline(trace_file, line)) {
    if (line.find("//") != std::string::npos)
      line = line.substr(0, line.find("//"));  // Remove comments

    std::string param_name;
    {
      auto pos = line.find('=');
      if (pos == std::string::npos) continue;

      param_name = line.substr(0, pos);
      for (auto& c : param_name)
        c = (char)toupper(c);  // So we don't have to worry about lowercase inputs
      line = line.substr(pos + 1);
    }

    if (param_name.find("ATT") != std::string::npos &&
        param_name.find("TARGET_CU") != std::string::npos) {
      started_att_counters = true;  // Means we'll do ATT
      param_name = "TARGET_CU";     // To cover different variations
    }
    if (!started_att_counters) continue;

    if (param_name == "REMOVE_HEADER_PACKET") {
      ret.header.enable = 0;
      continue;
    } else if (param_name == "KERNEL") {
      ret.kernel_names.push_back(line);
      continue;
    } else if (param_name == "PERFCOUNTER") {
      ret.counters_names.push_back(line);
      continue;
    } else if (param_name == "DISPATCH") {
      size_t comma = line.find(',');
      int id = stoi(line.substr(0, comma));
      int rank = (comma < line.size() - 1) ? stoi(line.substr(comma + 1)) : 0;

      if (MPI_RANK < 0 || rank == MPI_RANK)  // Only add ID if rank matches the one in input.txt
        ret.dispatch_ids.push_back({id, id});
      continue;
    } else if (param_name == "DISPATCH_RANGE") {
      size_t comma = line.find(',');
      int start = stoi(line.substr(0, comma));
      int end = (comma < line.size() - 1) ? stoi(line.substr(comma + 1)) : start;

      ret.dispatch_ids.push_back({start, end});
      continue;
    }
    // param_value is a number
    uint32_t param_value;
    try {
      auto hexa_pos = line.find("0x");  // Is it hex?
      if (hexa_pos != std::string::npos)
        param_value = stoul(line.substr(hexa_pos + 2), 0, 16);  // hexadecimal
      else
        param_value = stoul(line, 0, 10);  // decimal
    } catch (...) {
      printf("Error: Invalid parameter value %s\n", line.c_str());
      continue;
    }

    if (ATT_PARAM_NAMES.find(param_name) != ATT_PARAM_NAMES.end()) {
      ret.params.push_back(std::make_pair(ATT_PARAM_NAMES[param_name], param_value));
      try {
        default_params.erase(param_name);
      } catch (...) {
      };
    } else {
      printf("Error: Invalid parameter name: %s  - (%s)\nList of available params:\n",
             param_name.c_str(), line.c_str());
      for (auto& name : ATT_PARAM_NAMES) printf("%s\n", name.first.c_str());
    }

    if (param_name.find("TARGET_CU") != std::string::npos)
      ret.header.DCU = param_value;
    else if (param_name.find("SIMD_SELECT") != std::string::npos)
      ret.header.DSIMDM = param_value;
  }

  if (!started_att_counters) return {};

  for (auto& param : default_params)
    ret.params.push_back(std::make_pair(ATT_PARAM_NAMES[param.first], param.second));

  return ret;
}

std::mutex finish_lock{};

void finish()
{
  std::lock_guard<std::mutex> lock(finish_lock);

  if (!rocprof_started.load()) return;

  trace_period_thread_control.store(false);
  if (trace_period_thread.joinable())
    trace_period_thread.join();

  flush_thread_control.store(false);
  if (flush_thread.joinable())
    flush_thread.join();

  for ([[maybe_unused]] rocprofiler_buffer_id_t buffer_id : buffer_ids)
    CHECK_ROCPROFILER(rocprofiler_flush_data(session_id, buffer_id));

  roc_sys_handler.store(false);
  if (wait_for_start_shm.joinable())
    wait_for_start_shm.join();

  bool expect = true;
  if (session_created.compare_exchange_strong(expect, false)) {
    rocprofiler::TraceBufferBase::FlushAll();
    CHECK_ROCPROFILER(rocprofiler_terminate_session(session_id));
  }

  if (plugin->is_valid()) plugin->rocprofiler_plugin_finalize_fn();
  rocprof_started.exchange(false);
}

static bool env_var_search(std::string& s) {
  std::smatch m;
  std::regex e("(.*)\\%\\q\\{([^}]+)\\}(.*)");
  std::regex_match(s, m, e);

  if (m.size() != 4) return false;

  while (m.size() == 4) {
    const char* envvar = getenv(m[2].str().c_str());
    if (!envvar) envvar = "";
    s = m[1].str() + envvar + m[3].str();
    std::regex_match(s, m, e);
  };

  return true;
}

static void env_var_replace(const char* env_name) {
  if (!env_name) return;
  const char* env = getenv(env_name);
  if (!env) return;

  std::string new_env(env);
  if (env_var_search(new_env)) setenv(env_name, new_env.c_str(), 1);
}

// load plugins
void plugins_load(void* userdata) {
  // Load output plugin
  if (Dl_info dl_info; dladdr((void*)plugins_load, &dl_info) != 0) {
    const char* plugin_name = getenv("ROCPROFILER_PLUGIN_LIB");
    if (plugin_name == nullptr) {
      if (getenv("OUTPUT_PATH") || getenv("OUT_FILE_NAME"))
        plugin_name = "libfile_plugin.so";
      else
        plugin_name = "libcli_plugin.so";
    }

    bool bIsATT = std::string_view(plugin_name) == "libatt_plugin.so";
    if (!bIsATT) env_var_replace("OUTPUT_PATH");
    env_var_replace("OUT_FILE_NAME");

    std::string out_path = getenv("OUTPUT_PATH") ? getenv("OUTPUT_PATH") : "";

    if (out_path.size() && !bIsATT) {
      try {
        rocprofiler::common::filesystem::create_directories(out_path);
      } catch (...) {
      }
      out_path = out_path + '/';
    }
    if (out_path.size() && getenv("ROCPROFILER_COUNTERS")) {
      std::ofstream(out_path + "pmc.txt", std::ios::app)
          << std::string(getenv("ROCPROFILER_COUNTERS")) << '\n';
    }

    PluginHeaderPacket header{
        .plugin_path = fs::path(dl_info.dli_fname).replace_filename(plugin_name),
        .userdata = userdata};
    plugin = new rocprofiler_plugin_t{header};
    if (!plugin->is_valid()) {
      delete plugin;
      plugin = nullptr;
    }
    if (plugin)
      plugin->setPluginName(plugin_name);
  }
}

/*
 * A callback function for synchronous trace records.
 * This function queries the api information and populates the
 * api_trace buffer and adds it to the trace buffer.
 */
void sync_api_trace_callback(rocprofiler_record_tracer_t tracer_record,
                             rocprofiler_session_id_t session_id) {
  if (tracer_record.domain == ACTIVITY_DOMAIN_HIP_API) {
    if (tracer_record.phase == ROCPROFILER_PHASE_ENTER) {
      rocprofiler_timestamp_t timestamp;
      CHECK_ROCPROFILER(rocprofiler_get_timestamp(&timestamp));
      *(const_cast<hip_api_data_t*>(tracer_record.api_data.hip)->phase_data) = timestamp.value;
    } else {
      rocprofiler_timestamp_t timestamp;
      CHECK_ROCPROFILER(rocprofiler_get_timestamp(&timestamp));
      tracer_record.timestamps = rocprofiler_record_header_timestamp_t{
          .begin = rocprofiler_timestamp_t{*tracer_record.api_data.hip->phase_data},
          .end = timestamp};
      if (plugin)
        plugin->write_callback_record(tracer_record);
    }
  }
  if (tracer_record.domain == ACTIVITY_DOMAIN_HSA_API) {
    if (tracer_record.phase == ROCPROFILER_PHASE_ENTER) {
      rocprofiler_timestamp_t timestamp;
      CHECK_ROCPROFILER(rocprofiler_get_timestamp(&timestamp));
      *(const_cast<hsa_api_data_t*>(tracer_record.api_data.hsa)->phase_data) = timestamp.value;
    } else {
      rocprofiler_timestamp_t timestamp;
      CHECK_ROCPROFILER(rocprofiler_get_timestamp(&timestamp));
      tracer_record.timestamps = rocprofiler_record_header_timestamp_t{
          .begin = rocprofiler_timestamp_t{*tracer_record.api_data.hip->phase_data},
          .end = timestamp};
      if (plugin)
        plugin->write_callback_record(tracer_record);
    }
  }
  if (tracer_record.domain == ACTIVITY_DOMAIN_ROCTX) {
    rocprofiler_timestamp_t timestamp;
    CHECK_ROCPROFILER(rocprofiler_get_timestamp(&timestamp));
    tracer_record.timestamps =
        rocprofiler_record_header_timestamp_t{timestamp, rocprofiler_timestamp_t{0}};
    tracer_record.operation_id.id = tracer_record.api_data.roctx->args.id;
    if (plugin)
      plugin->write_callback_record(tracer_record);
  }
}

void wait_for_rocsys()
{
  using namespace std::chrono_literals;
  int64_t run_increment_counter = -1;

  while (roc_sys_handler.load())
  {
    run_increment_counter++;
    int shm_fd_sn = shm_open(roc_sys_session_id, O_RDONLY, 0666);
    std::this_thread::sleep_for(10ms);
    if (shm_fd_sn < 0) {
      continue;
    }
    shmd_t* shmd = reinterpret_cast<struct shmd_t*>(mmap(0, 1024, PROT_READ, MAP_SHARED, shm_fd_sn, 0));
    msync(shmd, sizeof(shmd->command), MS_SYNC | MS_INVALIDATE);
    bool flag{false};

    if (shmd && (sizeof(shmd->command) == sizeof(int)) && run_increment_counter > 0)
    {
      switch (shmd->command) {
        // Start
        case 4: {
          printf("ROCSYS:: Starting Tools Session...\n");

          bool expect = false;
          if (session_created.compare_exchange_strong(expect, true))
          {
            plugin->initialize(nullptr);
            CHECK_ROCPROFILER(rocprofiler_start_session(session_id));
          }
          break;
        }
        // Stop
        case 5: {
          printf("ROCSYS:: Stopping Tools Session...\n");

          bool expect = true;
          if (session_created.compare_exchange_strong(expect, false))
          {
            CHECK_ROCPROFILER(rocprofiler_terminate_session(session_id));
            for ([[maybe_unused]] rocprofiler_buffer_id_t buffer_id : buffer_ids) {
              CHECK_ROCPROFILER(rocprofiler_flush_data(session_id, buffer_id));
            }
            rocprofiler::TraceBufferBase::FlushAll();
          }
          break;
        }
        // Exit
        case 6: {
          printf("ROCSYS:: Exiting Tools Session...\n");

          bool expect = true;
          if (session_created.compare_exchange_strong(expect, false))
          {
            CHECK_ROCPROFILER(rocprofiler_terminate_session(session_id));
            for ([[maybe_unused]] rocprofiler_buffer_id_t buffer_id : buffer_ids) {
              CHECK_ROCPROFILER(rocprofiler_flush_data(session_id, buffer_id));
            }
            rocprofiler::TraceBufferBase::FlushAll();
          }
          roc_sys_handler.store(false);
          flag = true;
        }
      }
    }
    shm_unlink(roc_sys_session_id);
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

// Sleeps thread for amount of time without hanging
void sleep_while_condition(int64_t time_length, std::atomic<bool>& condition) {
  int64_t time_slept = 0;
  while (time_slept < time_length && condition.load()) {
    int64_t sleep_amount = std::min(SLEEP_CYCLE_LENGTH, time_length - time_slept);
    time_slept += sleep_amount;
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_amount));
  }
}

void flush_interval_func() {
  while (flush_thread_control.load()) {
    sleep_while_condition(flush_interval, flush_thread_control);
    for ([[maybe_unused]] rocprofiler_buffer_id_t buffer_id : buffer_ids) {
      CHECK_ROCPROFILER(rocprofiler_flush_data(session_id, buffer_id));
    }
    rocprofiler::TraceBufferBase::FlushAll();
  }
}

void trace_period_func() {
  using namespace std::chrono;
  sleep_while_condition(trace_delay, trace_period_thread_control);

  int64_t num_sleeps = 0;
  auto start_time = system_clock::now();

  while (trace_period_thread_control.load()) {
    num_sleeps += 1;
    CHECK_ROCPROFILER(rocprofiler_start_session(session_id));
    session_created.exchange(true);

    sleep_while_condition(trace_time_length, trace_period_thread_control);

    session_created.exchange(false);
    rocprofiler::TraceBufferBase::FlushAll();
    CHECK_ROCPROFILER(rocprofiler_terminate_session(session_id));

    if (trace_interval >= INT_MAX) break;

    auto miliElapsed = duration_cast<milliseconds>(system_clock::now() - start_time).count();
    sleep_while_condition(num_sleeps * trace_interval - miliElapsed, trace_period_thread_control);
  }
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
  rocprof_started.exchange(true);

  std::atexit(finish);

  roc_sys_session_id = getenv("ROCPROFILER_ENABLE_ROCSYS");
  if (roc_sys_session_id != nullptr) {
    printf("ROCSYS Session Created!\n");
    wait_for_start_shm = std::thread{wait_for_rocsys};
    roc_sys_handler.exchange(true);
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

  std::vector<std::string> counters = GetCounterNames();

  if (counters.size() > 0) {
    printf("ROCProfilerV2: Collecting the following counters:\n");
    for (size_t i = 0; i < counters.size(); i++) {
      counter_names.emplace_back(counters.at(i).c_str());
      printf("- %s\n", counter_names.back());
    }
  }

  std::vector<rocprofiler_tracer_activity_domain_t> apis_requested;

  if (getenv("ROCPROFILER_HIP_API_TRACE")) apis_requested.emplace_back(ACTIVITY_DOMAIN_HIP_API);
  if (getenv("ROCPROFILER_HSA_API_TRACE")) apis_requested.emplace_back(ACTIVITY_DOMAIN_HSA_API);
  if (getenv("ROCPROFILER_HSA_ACTIVITY_TRACE"))
    apis_requested.emplace_back(ACTIVITY_DOMAIN_HSA_OPS);
  else if (getenv("ROCPROFILER_HIP_ACTIVITY_TRACE"))
    apis_requested.emplace_back(ACTIVITY_DOMAIN_HIP_OPS);
  if (getenv("ROCPROFILER_ROCTX_TRACE")) apis_requested.emplace_back(ACTIVITY_DOMAIN_ROCTX);

  // ATT Parameters
  att_parsed_input_t att_params = GetATTParams();
  std::vector<rocprofiler_att_parameter_t> parameters;

  // load the plugins
  plugins_load(att_params.params.size() ? (void*)att_params.header.raw : (void*)&counter_names);

  for (auto& kv_pair : att_params.params)
    parameters.emplace_back(rocprofiler_att_parameter_t{kv_pair.first, kv_pair.second});
  for (std::string& name : att_params.counters_names) {
    rocprofiler_att_parameter_t param;
    param.parameter_name = ROCPROFILER_ATT_PERFCOUNTER_NAME;
    param.counter_name = name.c_str();
    parameters.emplace_back(param);
  }

  getFlushIntervalFromEnv();
  getTracePeriodFromEnv();

  CHECK_ROCPROFILER(rocprofiler_create_session(ROCPROFILER_NONE_REPLAY_MODE, &session_id));

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
              if (plugin)
                plugin->write_buffer_records(record, end_record, session_id, buffer_id);
            },
            1 << 20, &buffer_id));
        buffer_ids.emplace_back(buffer_id);
        printf("Enabling Counter Collection\n");
        rocprofiler_filter_id_t filter_id;
        [[maybe_unused]] rocprofiler_filter_property_t property = {};
        CHECK_ROCPROFILER(rocprofiler_create_filter(
            session_id, filter_kind, rocprofiler_filter_data_t{.counters_names = &counter_names[0]},
            counter_names.size(), &filter_id, property));
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
              if (plugin)
                plugin->write_buffer_records(record, end_record, session_id, buffer_id);
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
              if (plugin)
                plugin->write_buffer_records(record, end_record, session_id, buffer_id);
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
        if (plugin->getPluginName().find("json") != std::string::npos)
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
              if (plugin)
                plugin->write_buffer_records(record, end_record, session_id, buffer_id);
            },
            1 << 20, &buffer_id));
        buffer_ids.emplace_back(buffer_id);
        printf("Enabling ATT Tracing\n");

        rocprofiler_filter_id_t filter_id;
        rocprofiler_filter_property_t property = {};
        std::vector<const char*> kernel_names_c;

        if (att_params.dispatch_ids.size()) {  // Correlation ID filter
          property.kind = ROCPROFILER_FILTER_DISPATCH_IDS;
          property.data_count = att_params.dispatch_ids.size();
          property.dispatch_ids =
              reinterpret_cast<decltype(property.dispatch_ids)>(att_params.dispatch_ids.data());
        } else {  // Kernel names filter
          for (auto& name : att_params.kernel_names) kernel_names_c.push_back(name.data());

          property.kind = ROCPROFILER_FILTER_KERNEL_NAMES;
          property.data_count = kernel_names_c.size();
          property.name_regex = kernel_names_c.data();
        }
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
              if (plugin)
                plugin->write_buffer_records(record, end_record, session_id, buffer_id);
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

  // Flush buffers every given interval
  if (flush_interval > 0) {
    flush_thread_control.exchange(true);
    flush_thread = std::thread{flush_interval_func};
  }

  // Let session run for a given period of time
  if (trace_time_length > 0) {
    trace_period_thread_control.exchange(true);
    trace_period_thread = std::thread{trace_period_func};
  } else if (roc_sys_session_id == nullptr) {
    CHECK_ROCPROFILER(rocprofiler_start_session(session_id));
    session_created.exchange(true);
  }
  return true;
}

/**
@brief Callback function upon unloading the HSA.
*/
ROCPROFILER_EXPORT void OnUnload() { finish(); }

}  // extern "C"
