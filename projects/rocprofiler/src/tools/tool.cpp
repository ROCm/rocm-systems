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

#define ROCPROFILER_V2

#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <rocprofiler.h>
#include <rocprofiler_plugin.h>
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

#include "rocprofiler.h"
#include "utils/helper.h"

namespace fs = std::experimental::filesystem;

// Macro to check ROCMTools calls status
#define CHECK_ROCMTOOLS(call)                                                                      \
  do {                                                                                             \
    if ((call) != ROCPROFILER_STATUS_SUCCESS) rocmtools::fatal("Error: ROCMTools API Call Error!");  \
  } while (false)

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

void finish() {
  if (amd_sys_handler.load(std::memory_order_release)) {
    amd_sys_handler.exchange(false, std::memory_order_release);
    wait_for_start_shm.join();
    shm_unlink(std::to_string(*amd_sys_session_id).c_str());
  }
  if (session_created.load(std::memory_order_relaxed)) {
    session_created.exchange(false, std::memory_order_release);
    CHECK_ROCMTOOLS(rocprofiler_terminate_session(session_id));
    for ([[maybe_unused]] rocprofiler_buffer_id_t buffer_id : buffer_ids) {
      CHECK_ROCMTOOLS(rocprofiler_flush_data(session_id, buffer_id));
    }
  }

  // CHECK_ROCMTOOLS(rocprofiler_destroy_session(session_id));
  // CHECK_ROCMTOOLS(rocprofiler_finalize());
}

// load plugins
void plugins_load() {
  // Load output plugin
  if (Dl_info dl_info; dladdr((void*)plugins_load, &dl_info) != 0) {
    const char* plugin_name = getenv("ROCPROFILER_PLUGIN_LIB");
    if (plugin_name == nullptr) {
      if (fs::path(dl_info.dli_fname).string().find("build") != std::string::npos) {
        plugin_name = "libfile_plugin.so";
      } else {
        plugin_name = "rocmtools/libfile_plugin.so";
      }
    }
    if (!plugin.emplace(fs::path(dl_info.dli_fname).replace_filename(plugin_name)).is_valid()) {
      plugin.reset();
    }
  }
}

void plugin_write_record(rocprofiler_record_tracer_t record, rocprofiler_session_id_t session_id) {
  if (plugin) plugin->write_callback_record(record, session_id);
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
          CHECK_ROCMTOOLS(rocprofiler_start_session(session_id));
          session_created.exchange(true, std::memory_order_release);
          break;
        }
        // Stop
        case 5: {
          if (session_created.load(std::memory_order_relaxed)) {
            printf("AMDSYS:: Stopping Tools Session...\n");
            session_created.exchange(false, std::memory_order_release);
            CHECK_ROCMTOOLS(rocprofiler_terminate_session(session_id));
            for ([[maybe_unused]] rocprofiler_buffer_id_t buffer_id : buffer_ids) {
              CHECK_ROCMTOOLS(rocprofiler_flush_data(session_id, buffer_id));
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
            CHECK_ROCMTOOLS(rocprofiler_terminate_session(session_id));
            for ([[maybe_unused]] rocprofiler_buffer_id_t buffer_id : buffer_ids) {
              CHECK_ROCMTOOLS(rocprofiler_flush_data(session_id, buffer_id));
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
ROCPROFILER_EXPORT bool OnLoad(HsaApiTable* table, uint64_t runtime_version,
                             uint64_t failed_tool_count, const char* const* failed_tool_names) {
  if (rocprofiler_version_major() != ROCPROFILER_VERSION_MAJOR ||
      rocprofiler_version_minor() < ROCPROFILER_VERSION_MINOR) {
    warning("the ROCMTools API version is not compatible with this tool");
    return true;
  }

  std::atexit(finish);

  amd_sys_session_id = getenv("ROCPROFILER_ENABLE_AMDSYS");
  if (amd_sys_session_id != nullptr) {
    printf("AMDSYS Session Started!\n");
    wait_for_start_shm = std::thread{wait_for_amdsys};
    amd_sys_handler.exchange(true, std::memory_order_release);
  }

  CHECK_ROCMTOOLS(rocprofiler_initialize());

  // Printing out info
  char* info_symb = getenv("ROCPROFILER_COUNTER_LIST");
  if (info_symb != nullptr) {
    if (*info_symb == 'b')
      printf("Basic HW counters:\n");
    else
      printf("Derived metrics:\n");
    CHECK_ROCMTOOLS(rocprofiler_iterate_counters(info_callback));
    exit(1);
  }

  // load the plugins
  plugins_load();


  std::vector<rocprofiler_tracer_activity_domain_t> apis_requested;

  if (getenv("ROCPROFILER_HIP_API_TRACE")) apis_requested.emplace_back(ACTIVITY_DOMAIN_HIP_API);
  if (getenv("ROCPROFILER_HIP_ACTIVITY_TRACE")) apis_requested.emplace_back(ACTIVITY_DOMAIN_HIP_OPS);
  if (getenv("ROCPROFILER_HSA_API_TRACE")) apis_requested.emplace_back(ACTIVITY_DOMAIN_HSA_API);
  if (getenv("ROCPROFILER_HSA_ACTIVITY_TRACE")) apis_requested.emplace_back(ACTIVITY_DOMAIN_HSA_OPS);
  if (getenv("ROCPROFILER_ROCTX_TRACE")) apis_requested.emplace_back(ACTIVITY_DOMAIN_ROCTX);

  std::vector<std::string> counters = GetCounterNames();
  std::vector<const char*> counters_;

  if (counters.size() > 0) {
    printf("ROCMTools: Collecting the following counters:\n");
    for (size_t i = 0; i < counters.size(); i++) {
      counters_.emplace_back(counters.at(i).c_str());
      printf("- %s\n", counters_.back());
    }
  }

  CHECK_ROCMTOOLS(rocprofiler_create_session(ROCPROFILER_KERNEL_REPLAY_MODE, &session_id));

  bool want_pc_sampling = getenv("ROCPROFILER_PC_SAMPLING");

  std::vector<rocprofiler_filter_kind_t> filters_requested;
  if ((counters.size() == 0 && (apis_requested.size() == 0 || getenv("ROCPROFILER_KERNEL_TRACE")))
      || want_pc_sampling /* PC sampling needs a profiler, even it's doing
                             nothing */)
    filters_requested.emplace_back(ROCPROFILER_DISPATCH_TIMESTAMPS_COLLECTION);
  if (want_pc_sampling) {
    filters_requested.emplace_back(ROCPROFILER_PC_SAMPLING_COLLECTION);
  }
  if (counters.size() > 0) filters_requested.emplace_back(ROCPROFILER_COUNTERS_COLLECTION);
  if (apis_requested.size() > 0) filters_requested.emplace_back(ROCPROFILER_API_TRACE);

  rocprofiler_buffer_id_t buffer_id;
  CHECK_ROCMTOOLS(rocprofiler_create_buffer(
      session_id,
      [](const rocprofiler_record_header_t* record, const rocprofiler_record_header_t* end_record,
         rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t buffer_id) {
        if (plugin) plugin->write_buffer_records(record, end_record, session_id, buffer_id);
      },
      0x9999, &buffer_id));
  buffer_ids.emplace_back(buffer_id);

  rocprofiler_buffer_id_t buffer_id_1;
  CHECK_ROCMTOOLS(rocprofiler_create_buffer(
      session_id,
      [](const rocprofiler_record_header_t* record, const rocprofiler_record_header_t* end_record,
         rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t buffer_id_1) {
        if (plugin) plugin->write_buffer_records(record, end_record, session_id, buffer_id_1);
      },
      0x9999, &buffer_id_1));
  buffer_ids.emplace_back(buffer_id_1);

  for (rocprofiler_filter_kind_t filter_kind : filters_requested) {
    switch (filter_kind) {
      case ROCPROFILER_COUNTERS_COLLECTION: {
        printf("Enabling Counter Collection\n");
        rocprofiler_filter_id_t filter_id;
        [[maybe_unused]] rocprofiler_filter_property_t property = {};
        CHECK_ROCMTOOLS(rocprofiler_create_filter(
            session_id, filter_kind, rocprofiler_filter_data_t{.counters_names = &counters_[0]},
            counters_.size(), &filter_id, property));
        CHECK_ROCMTOOLS(rocprofiler_set_filter_buffer(session_id, filter_id, buffer_id_1));
        filter_ids.emplace_back(filter_id);
        break;
      }
      case ROCPROFILER_DISPATCH_TIMESTAMPS_COLLECTION: {
        rocprofiler_filter_id_t filter_id;
        [[maybe_unused]] rocprofiler_filter_property_t property = {};
        CHECK_ROCMTOOLS(rocprofiler_create_filter(session_id, filter_kind, rocprofiler_filter_data_t{},
                                                0, &filter_id, property));
        CHECK_ROCMTOOLS(rocprofiler_set_filter_buffer(session_id, filter_id, buffer_id_1));
        filter_ids.emplace_back(filter_id);
        break;
      }
      case ROCPROFILER_API_TRACE: {
        printf("Enabling API Tracing\n");
        rocprofiler_filter_id_t filter_id;
        [[maybe_unused]] rocprofiler_filter_property_t property = {};
        CHECK_ROCMTOOLS(rocprofiler_create_filter(session_id, filter_kind,
                                                rocprofiler_filter_data_t{&apis_requested[0]},
                                                apis_requested.size(), &filter_id, property));
        CHECK_ROCMTOOLS(rocprofiler_set_filter_buffer(session_id, filter_id, buffer_id));
        CHECK_ROCMTOOLS(
            rocprofiler_set_api_trace_sync_callback(session_id, filter_id, plugin_write_record));
        filter_ids.emplace_back(filter_id);
        break;
      }
      case ROCPROFILER_PC_SAMPLING_COLLECTION: {
        puts("Enabling PC sampling");
        rocprofiler_filter_id_t filter_id;
        [[maybe_unused]] rocprofiler_filter_property_t property = {};
        CHECK_ROCMTOOLS(rocprofiler_create_filter(session_id, filter_kind,
                                                rocprofiler_filter_data_t{},
                                                0, &filter_id, property));
        CHECK_ROCMTOOLS(rocprofiler_set_filter_buffer(session_id, filter_id, buffer_id));
        filter_ids.emplace_back(filter_id);
        break;
      }
      default:
        warning("Not available for profiling or tracing");
    }
  }

  if (getenv("ROCPROFILER_ENABLE_AMDSYS") == nullptr) {
    CHECK_ROCMTOOLS(rocprofiler_start_session(session_id));
    session_created.exchange(true, std::memory_order_release);
  }
  return true;
}

/**
@brief Callback function upon unloading the HSA.
*/
ROCPROFILER_EXPORT void OnUnload() { printf("\n\nTool is getting unloaded\n\n"); }

}  // extern "C"
