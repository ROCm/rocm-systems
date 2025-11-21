/* Copyright (c) 2018-2022 Advanced Micro Devices, Inc.

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

#include <roctracer_ext.h>
#include <roctracer_hip.h>
#include <roctracer_hsa.h>
#include <roctracer_plugin.h>
#include <roctracer_roctx.h>

#include <atomic>
#include <cassert>
#include <chrono>
#include <experimental/filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <stack>
#include <utility>
#include <vector>
#include <variant>

#include <cxxabi.h> /* kernel name demangling */
#include <dirent.h>
#include <dlfcn.h>
#include <hsa/hsa_api_trace.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h> /* SYS_xxx definitions */
#include <sys/types.h>
#include <unistd.h> /* usleep */

#include "debug.h"
#include "loader.h"
#include "trace_buffer.h"
#include "xml.h"

void initialize() __attribute__((constructor(101)));

namespace fs = std::experimental::filesystem;

// Macro to check ROC-tracer calls status
#define CHECK_ROCTRACER(call)                                                                      \
  do {                                                                                             \
    if ((call) != ROCTRACER_STATUS_SUCCESS) {                                                      \
      fatal(#call " failed: %s", roctracer_error_string());                                        \
    }                                                                                              \
  } while (false)

TRACE_BUFFER_INSTANTIATE();

namespace {

inline roctracer_timestamp_t timestamp_ns() {
  roctracer_timestamp_t timestamp;
  CHECK_ROCTRACER(roctracer_get_timestamp(&timestamp));
  return timestamp;
}

std::vector<std::string> hsa_api_vec;
std::vector<std::string> hip_api_vec;

bool trace_roctx = false;
bool trace_hsa_api = false;
bool trace_hsa_activity = false;
bool trace_hip_api = false;
bool trace_hip_activity = false;
bool trace_pcs = false;

uint32_t GetPid() {
  static uint32_t pid = syscall(__NR_getpid);
  return pid;
}
uint32_t GetTid() {
  static thread_local uint32_t tid = syscall(__NR_gettid);
  return tid;
}

size_t GetBufferSize() {
  auto bufSize = getenv("ROCTRACER_BUFFER_SIZE");
  // Default size if not set
  if (!bufSize) return 0x200000;
  return std::stoll({bufSize});
}

// Tracing control thread
uint32_t control_delay_us = 0;
uint32_t control_len_us = 0;
uint32_t control_dist_us = 0;
std::thread* trace_period_thread = nullptr;
std::atomic_bool trace_period_stop = false;
void trace_period_fun() {
  std::this_thread::sleep_for(std::chrono::microseconds(control_delay_us));
  do {
    roctracer_start();
    if (trace_period_stop) {
      roctracer_stop();
      break;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(control_len_us));
    roctracer_stop();
    if (trace_period_stop) break;
    std::this_thread::sleep_for(std::chrono::microseconds(control_dist_us));
  } while (!trace_period_stop);
}

// Flushing control thread
uint32_t control_flush_us = 0;
std::thread* flush_thread = nullptr;
std::atomic_bool stop_flush_thread = false;


void flush_thr_fun() {
  while (!stop_flush_thread) {
    CHECK_ROCTRACER(roctracer_flush_activity());
    roctracer::TraceBufferBase::FlushAll();
    std::this_thread::sleep_until(std::chrono::steady_clock::now() +
                                  std::chrono::microseconds(control_flush_us));
  }
}

class roctracer_plugin_t {
 public:
  roctracer_plugin_t(const std::string& plugin_path) {
    plugin_handle_ = dlopen(plugin_path.c_str(), RTLD_LAZY);
    if (plugin_handle_ == nullptr) {
      warning("dlopen(\"%s\") failed: %s", plugin_path.c_str(), dlerror());
      return;
    }

    roctracer_plugin_write_callback_record_ =
        reinterpret_cast<decltype(roctracer_plugin_write_callback_record)*>(
            dlsym(plugin_handle_, "roctracer_plugin_write_callback_record"));
    if (!roctracer_plugin_write_callback_record_) return;

    roctracer_plugin_write_activity_records_ =
        reinterpret_cast<decltype(roctracer_plugin_write_activity_records)*>(
            dlsym(plugin_handle_, "roctracer_plugin_write_activity_records"));
    if (!roctracer_plugin_write_activity_records_) return;

    roctracer_plugin_finalize_ = reinterpret_cast<decltype(roctracer_plugin_finalize)*>(
        dlsym(plugin_handle_, "roctracer_plugin_finalize"));
    if (!roctracer_plugin_finalize_) return;

    if (auto* initialize = reinterpret_cast<decltype(roctracer_plugin_initialize)*>(
            dlsym(plugin_handle_, "roctracer_plugin_initialize"));
        initialize != nullptr)
      valid_ = initialize(ROCTRACER_VERSION_MAJOR, ROCTRACER_VERSION_MINOR) == 0;
  }

  ~roctracer_plugin_t() {
    if (is_valid()) roctracer_plugin_finalize_();
    if (plugin_handle_ != nullptr) dlclose(plugin_handle_);
  }

  bool is_valid() const { return valid_; }

  template <typename... Args> auto write_callback_record(Args... args) const {
    assert(is_valid());
    return roctracer_plugin_write_callback_record_(std::forward<Args>(args)...);
  }
  template <typename... Args> auto write_activity_records(Args... args) const {
    assert(is_valid());
    return roctracer_plugin_write_activity_records_(std::forward<Args>(args)...);
  }

 private:
  bool valid_{false};
  void* plugin_handle_;

  decltype(roctracer_plugin_finalize)* roctracer_plugin_finalize_;
  decltype(roctracer_plugin_write_callback_record)* roctracer_plugin_write_callback_record_;
  decltype(roctracer_plugin_write_activity_records)* roctracer_plugin_write_activity_records_;
};

std::optional<roctracer_plugin_t> plugin;

}  // namespace

///////////////////////////////////////////////////////////////////////////////////////////////////////
// rocTX annotation tracing

struct roctx_trace_entry_t {
  std::atomic<roctracer::TraceEntryState> valid;
  roctracer_record_t record;
  union {
    roctx_api_data_t data;
  };

  roctx_trace_entry_t(uint32_t cid, roctracer_timestamp_t time, uint32_t pid, uint32_t tid,
                      roctx_range_id_t rid, const char* message)
      : valid(roctracer::TRACE_ENTRY_INIT) {
    record.domain = ACTIVITY_DOMAIN_ROCTX;
    record.op = cid;
    record.kind = 0;
    record.begin_ns = time;
    record.end_ns = 0;
    record.process_id = pid;
    record.thread_id = tid;
    data.args.message = message != nullptr ? strdup(message) : nullptr;
    data.args.id = rid;
  }
  ~roctx_trace_entry_t() {
    if (data.args.message != nullptr) free(const_cast<char*>(data.args.message));
  }
};

roctracer::TraceBuffer<roctx_trace_entry_t> roctx_trace_buffer(
    "rocTX API", GetBufferSize(), [](roctx_trace_entry_t* entry) {
      assert(plugin && "plugin is not initialized");
      plugin->write_callback_record(&entry->record, &entry->data);
    });

// rocTX callback function
void roctx_api_callback(uint32_t domain, uint32_t cid, const void* callback_data,
                        void* /* user_arg */) {
  const roctx_api_data_t* data = reinterpret_cast<const roctx_api_data_t*>(callback_data);

  roctx_trace_entry_t& entry = roctx_trace_buffer.Emplace(cid, timestamp_ns(), GetPid(), GetTid(),
                                                          data->args.id, data->args.message);
  entry.valid.store(roctracer::TRACE_ENTRY_COMPLETE, std::memory_order_release);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// HSA API tracing

struct hsa_api_trace_entry_t {
  std::atomic<uint32_t> valid;
  roctracer_record_t record;
  union {
    hsa_api_data_t data;
  };

  hsa_api_trace_entry_t(uint32_t cid, roctracer_timestamp_t begin, roctracer_timestamp_t end,
                        uint32_t pid, uint32_t tid, const hsa_api_data_t& hsa_api_data)
      : valid(roctracer::TRACE_ENTRY_INIT) {
    record.domain = ACTIVITY_DOMAIN_HSA_API;
    record.op = cid;
    record.kind = 0;
    record.begin_ns = begin;
    record.end_ns = end;
    record.process_id = pid;
    record.thread_id = tid;
    data = hsa_api_data;
  }
  ~hsa_api_trace_entry_t() {}
};

roctracer::TraceBuffer<hsa_api_trace_entry_t> hsa_api_trace_buffer(
    "HSA API", GetBufferSize(), [](hsa_api_trace_entry_t* entry) {
      assert(plugin && "plugin is not initialized");
      plugin->write_callback_record(&entry->record, &entry->data);
    });

// HSA API callback function

void hsa_api_callback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg) {
  (void)arg;
  const hsa_api_data_t* data = reinterpret_cast<const hsa_api_data_t*>(callback_data);
  if (data->phase == ACTIVITY_API_PHASE_ENTER) {
    *data->phase_data = timestamp_ns();
  } else {
    const roctracer_timestamp_t begin_timestamp = *data->phase_data;
    const roctracer_timestamp_t end_timestamp =
        (cid == HSA_API_ID_hsa_shut_down) ? begin_timestamp : timestamp_ns();
    hsa_api_trace_entry_t& entry = hsa_api_trace_buffer.Emplace(cid, begin_timestamp, end_timestamp,
                                                                GetPid(), GetTid(), *data);
    entry.valid.store(roctracer::TRACE_ENTRY_COMPLETE, std::memory_order_release);
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// HIP API tracing

struct hip_api_trace_entry_t {
  std::atomic<uint32_t> valid;
  roctracer_record_t record;
  union {
    hip_api_data_t data;
  };

  hip_api_trace_entry_t(uint32_t cid, roctracer_timestamp_t begin, roctracer_timestamp_t end,
                        uint32_t pid, uint32_t tid, const hip_api_data_t& hip_api_data,
                        const char* name)
      : valid(roctracer::TRACE_ENTRY_INIT) {
    record.domain = ACTIVITY_DOMAIN_HIP_API;
    record.op = cid;
    record.kind = 0;
    record.begin_ns = begin;
    record.end_ns = end;
    record.process_id = pid;
    record.thread_id = tid;
    data = hip_api_data;
    record.kernel_name = name ? strdup(name) : nullptr;
  }

  ~hip_api_trace_entry_t() {
    if (record.kernel_name != nullptr) free(const_cast<char*>(record.kernel_name));
  }
};

static std::string getKernelNameMultiKernelMultiDevice(hipLaunchParams* launchParamsList,
                                                       int numDevices) {
  std::stringstream name_str;
  for (int i = 0; i < numDevices; ++i) {
    if (launchParamsList[i].func != nullptr) {
      name_str << roctracer::HipLoader::Instance().KernelNameRefByPtr(launchParamsList[i].func)
               << ":"
               << roctracer::HipLoader::Instance().GetStreamDeviceId(launchParamsList[i].stream)
               << ";";
    }
  }
  return name_str.str();
}

template <typename... Ts> struct Overloaded : Ts... {
  using Ts::operator()...;
};
template <class... Ts> Overloaded(Ts...) -> Overloaded<Ts...>;


static std::optional<std::string> getKernelName(uint32_t cid, const hip_api_data_t* data) {
  std::variant<const void*, hipFunction_t> function;
  switch (cid) {
    case HIP_API_ID_hipExtLaunchMultiKernelMultiDevice: {
      return getKernelNameMultiKernelMultiDevice(
          data->args.hipExtLaunchMultiKernelMultiDevice.launchParamsList,
          data->args.hipExtLaunchMultiKernelMultiDevice.numDevices);
    }
    case HIP_API_ID_hipLaunchCooperativeKernelMultiDevice: {
      return getKernelNameMultiKernelMultiDevice(
          data->args.hipLaunchCooperativeKernelMultiDevice.launchParamsList,
          data->args.hipLaunchCooperativeKernelMultiDevice.numDevices);
    }
    case HIP_API_ID_hipLaunchKernel: {
      function = data->args.hipLaunchKernel.function_address;
      break;
    }
    case HIP_API_ID_hipExtLaunchKernel: {
      function = data->args.hipExtLaunchKernel.function_address;
      break;
    }
    case HIP_API_ID_hipLaunchCooperativeKernel: {
      function = data->args.hipLaunchCooperativeKernel.f;
      break;
    }
    case HIP_API_ID_hipLaunchByPtr: {
      function = data->args.hipLaunchByPtr.hostFunction;
      break;
    }
    case HIP_API_ID_hipGraphAddKernelNode: {
      function = data->args.hipGraphAddKernelNode.pNodeParams->func;
      break;
    }
    case HIP_API_ID_hipGraphExecKernelNodeSetParams: {
      function = data->args.hipGraphExecKernelNodeSetParams.pNodeParams->func;
      break;
    }
    case HIP_API_ID_hipGraphKernelNodeSetParams: {
      function = data->args.hipGraphKernelNodeSetParams.pNodeParams->func;
      break;
    }
    case HIP_API_ID_hipModuleLaunchKernel: {
      function = data->args.hipModuleLaunchKernel.f;
      break;
    }
    case HIP_API_ID_hipExtModuleLaunchKernel: {
      function = data->args.hipExtModuleLaunchKernel.f;
      break;
    }
    case HIP_API_ID_hipHccModuleLaunchKernel: {
      function = data->args.hipHccModuleLaunchKernel.f;
      break;
    }
    default:
      return {};
  }
  return std::visit(
      Overloaded{
          [](const void* func) {
            return roctracer::HipLoader::Instance().KernelNameRefByPtr(func);
          },
          [](hipFunction_t func) { return roctracer::HipLoader::Instance().KernelNameRef(func); },
      },
      function);
}

roctracer::TraceBuffer<hip_api_trace_entry_t> hip_api_trace_buffer(
    "HIP API", GetBufferSize(), [](hip_api_trace_entry_t* entry) {
      assert(plugin && "plugin is not initialized");
      plugin->write_callback_record(&entry->record, &entry->data);
    });

void hip_api_callback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg) {
  (void)arg;
  const hip_api_data_t* data = reinterpret_cast<const hip_api_data_t*>(callback_data);
  const roctracer_timestamp_t timestamp = timestamp_ns();
  std::optional<std::string> kernel_name;

  if (data->phase == ACTIVITY_API_PHASE_ENTER) {
    *data->phase_data = timestamp;
  } else {
    // Post init of HIP APU args
    hipApiArgsInit((hip_api_id_t)cid, const_cast<hip_api_data_t*>(data));
    kernel_name = getKernelName(cid, data);
    hip_api_trace_entry_t& entry =
        hip_api_trace_buffer.Emplace(cid, *data->phase_data, timestamp, GetPid(), GetTid(), *data,
                                     kernel_name ? kernel_name->c_str() : nullptr);
    entry.valid.store(roctracer::TRACE_ENTRY_COMPLETE, std::memory_order_release);
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Input parser
std::string normalize_token(const std::string& token, bool not_empty, const std::string& label) {
  const std::string space_chars_set = " \t";
  const size_t first_pos = token.find_first_not_of(space_chars_set);
  size_t norm_len = 0;
  std::string error_str = "none";
  if (first_pos != std::string::npos) {
    const size_t last_pos = token.find_last_not_of(space_chars_set);
    if (last_pos == std::string::npos)
      error_str = "token string error: \"" + token + "\"";
    else {
      const size_t end_pos = last_pos + 1;
      if (end_pos <= first_pos)
        error_str = "token string error: \"" + token + "\"";
      else
        norm_len = end_pos - first_pos;
    }
  }
  if (((first_pos != std::string::npos) && (norm_len == 0)) ||
      ((first_pos == std::string::npos) && not_empty)) {
    error("normalize_token error: %s", error_str.c_str());
  }
  return (norm_len != 0) ? token.substr(first_pos, norm_len) : std::string("");
}

int get_xml_array(const xml::Xml::level_t* node, const std::string& field, const std::string& delim,
                  std::vector<std::string>* vec, const char* label = nullptr) {
  int parse_iter = 0;
  const auto& opts = node->opts;
  auto it = opts.find(field);
  if (it != opts.end()) {
    const std::string& array_string = it->second;
    if (label != nullptr) std::cout << label << field << " = " << array_string << std::endl;
    size_t pos1 = 0;
    size_t string_len = array_string.length();
    while (pos1 < string_len) {
      // set pos2 such that it also handles case of multiple delimiter options.
      // For example-  "hipLaunchKernel, hipExtModuleLaunchKernel, hipMemsetAsync"
      // in this example delimiters are ' ' and also ','
      size_t pos2 = array_string.find_first_of(delim, pos1);
      bool found = (pos2 != std::string::npos);
      size_t token_len = (pos2 != std::string::npos) ? pos2 - pos1 : string_len - pos1;
      std::string token = array_string.substr(pos1, token_len);
      std::string norm_str = normalize_token(token, found, "get_xml_array");
      if (norm_str.length() != 0) vec->push_back(norm_str);
      if (!found) break;
      // update pos2 such that it represents the first non-delimiter character
      // in case multiple delimiters are specified in variable 'delim'
      pos1 = array_string.find_first_not_of(delim, pos2);
      ++parse_iter;
    }
  }
  return parse_iter;
}

// Allocating tracing pool
void open_tracing_pool() {
  if (roctracer_default_pool() == nullptr) {
    roctracer_properties_t properties{};
    properties.buffer_size = GetBufferSize();
    properties.buffer_callback_fun = [](const char* begin, const char* end, void* /* arg */) {
      assert(plugin && "plugin is not initialized");
      plugin->write_activity_records(reinterpret_cast<const roctracer_record_t*>(begin),
                                     reinterpret_cast<const roctracer_record_t*>(end));
    };
    CHECK_ROCTRACER(roctracer_open_pool(&properties));
  }
}

// Flush tracing pool
void close_tracing_pool() {
  if (roctracer_pool_t* pool = roctracer_default_pool(); pool != nullptr) {
    CHECK_ROCTRACER(roctracer_flush_activity_expl(pool));
    CHECK_ROCTRACER(roctracer_close_pool_expl(pool));
  }
}

// tool library is loaded
static bool is_loaded = false;

// tool unload method
void tool_unload() {
  if (is_loaded == false) return;
  is_loaded = false;

  if (flush_thread) {
    stop_flush_thread = true;
    flush_thread->join();
    delete flush_thread;
    flush_thread = nullptr;
  }

  if (trace_period_thread) {
    trace_period_stop = true;
    trace_period_thread->join();
    delete trace_period_thread;
    trace_period_thread = nullptr;
  }

  if (trace_roctx) {
    CHECK_ROCTRACER(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_ROCTX));
  }
  if (trace_hsa_api) {
    CHECK_ROCTRACER(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HSA_API));
  }
  if (trace_hsa_activity || trace_pcs) {
    CHECK_ROCTRACER(roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HSA_OPS));
  }
  if (trace_hip_api || trace_hip_activity) {
    CHECK_ROCTRACER(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API));
    CHECK_ROCTRACER(roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_API));
    CHECK_ROCTRACER(roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_OPS));
  }

  // Flush tracing pool
  close_tracing_pool();
  roctracer::TraceBufferBase::FlushAll();
}

// tool load method
void tool_load() {
  if (is_loaded == true) return;
  is_loaded = true;

  // API traces switches
  const char* trace_domain = getenv("ROCTRACER_DOMAIN");
  if (trace_domain != nullptr) {
    // ROCTX domain
    if (std::string(trace_domain).find("roctx") != std::string::npos) {
      trace_roctx = true;
    }

    // HSA/HIP domains enabling
    if (std::string(trace_domain).find("hsa") != std::string::npos) {
      trace_hsa_api = true;
      trace_hsa_activity = true;
    }
    if (std::string(trace_domain).find("hip") != std::string::npos) {
      trace_hip_api = true;
      trace_hip_activity = true;
    }
    if (std::string(trace_domain).find("sys") != std::string::npos) {
      trace_hsa_api = true;
      trace_hip_api = true;
      trace_hip_activity = true;
    }

    // PC sampling enabling
    if (std::string(trace_domain).find("pcs") != std::string::npos) {
      trace_pcs = true;
    }
  }

  std::cout << "ROCtracer (" << std::dec << GetPid() << "):";

  // XML input
  const char* xml_name = getenv("ROCP_INPUT");
  if (xml_name != nullptr) {
    xml::Xml* xml = xml::Xml::Create(xml_name);
    if (xml == nullptr) error("input file not found '%s'", xml_name);

    bool found = false;
    for (const auto* entry : xml->GetNodes("top.trace")) {
      auto it = entry->opts.find("name");
      if (it == entry->opts.end()) error("trace name is missing");
      const std::string& name = it->second;

      std::vector<std::string> api_vec;
      for (const auto* node : entry->nodes) {
        if (node->tag != "parameters")
          error("trace node is not supported '%s:%%%s'", name.c_str(), node->tag.c_str());
        get_xml_array(node, "api", ", ",
                      &api_vec);  // delimiter options given as both spaces and commas (' ' and ',')
        break;
      }

      if (name == "rocTX") {
        found = true;
        trace_roctx = true;
      }
      if (name == "HSA") {
        found = true;
        trace_hsa_api = true;
        hsa_api_vec = api_vec;
      }
      if (name == "GPU") {
        found = true;
        trace_hsa_activity = true;
      }
      if (name == "HIP") {
        found = true;
        trace_hip_api = true;
        trace_hip_activity = true;
        hip_api_vec = api_vec;
      }
    }

    if (found) std::cout << " input from \"" << xml_name << "\"";
  }
  std::cout << std::endl;

  // Disable HIP activity if HSA activity was set
  if (trace_hsa_activity == true) trace_hip_activity = false;

  // Enable rpcTX callbacks
  if (trace_roctx) {
    // initialize HSA tracing
    std::cout << "    rocTX-trace()" << std::endl;
    CHECK_ROCTRACER(
        roctracer_enable_domain_callback(ACTIVITY_DOMAIN_ROCTX, roctx_api_callback, nullptr));
  }

  const char* ctrl_str = getenv("ROCP_CTRL_RATE");
  if (ctrl_str != nullptr) {
    uint32_t ctrl_delay = 0;
    uint32_t ctrl_len = 0;
    uint32_t ctrl_rate = 0;

    if (sscanf(ctrl_str, "%d:%d:%d", &ctrl_delay, &ctrl_len, &ctrl_rate) != 3 ||
        ctrl_len > ctrl_rate)
      error("invalid ROCP_CTRL_RATE variable (ctrl_delay:ctrl_len:ctrl_rate)");

    control_dist_us = ctrl_rate - ctrl_len;
    control_len_us = ctrl_len;
    control_delay_us = ctrl_delay;

    roctracer_stop();

    if (ctrl_delay != UINT32_MAX) {
      std::cout << "ROCtracer: trace control: delay(" << ctrl_delay << "us), length(" << ctrl_len
                << "us), rate(" << ctrl_rate << "us)" << std::endl;
      trace_period_thread = new std::thread(trace_period_fun);
    } else {
      std::cout << "ROCtracer: trace start disabled" << std::endl;
    }
  }

  const char* flush_str = getenv("ROCP_FLUSH_RATE");
  if (flush_str != nullptr) {
    sscanf(flush_str, "%d", &control_flush_us);
    if (control_flush_us == 0) error("invalid control flush rate value '%s'", flush_str);

    std::cout << "ROCtracer: trace control flush rate(" << control_flush_us << "us)" << std::endl;
    flush_thread = new std::thread(flush_thr_fun);
  }
}

extern "C" {

// The HSA_AMD_TOOL_PRIORITY variable must be a constant value type initialized by the loader
// itself, not by code during _init. 'extern const' seems to do that although that is not a
// guarantee.
ROCTRACER_EXPORT extern const uint32_t HSA_AMD_TOOL_PRIORITY = 1050;

// HSA-runtime tool on-load method
ROCTRACER_EXPORT bool OnLoad(HsaApiTable* table, uint64_t runtime_version,
                             uint64_t failed_tool_count, const char* const* failed_tool_names) {
  if (roctracer_version_major() != ROCTRACER_VERSION_MAJOR ||
      roctracer_version_minor() < ROCTRACER_VERSION_MINOR) {
    warning("the ROCtracer API version is not compatible with this tool");
    return true;
  }

  // Load output plugin
  const char* plugin_name = getenv("ROCTRACER_PLUGIN_LIB");
  if (plugin_name == nullptr) plugin_name = "libfile_plugin.so";
  if (Dl_info dl_info; dladdr((void*)tool_load, &dl_info) != 0) {
    if (!plugin.emplace(fs::path(dl_info.dli_fname).replace_filename(plugin_name)).is_valid())
      plugin.reset();
  }

  tool_load();

  // OnUnload may not be called if the ROC runtime is not shutdown by the client
  // application before exiting, so register an atexit handler to unload the tool.
  std::atexit(tool_unload);

  // Enable HSA API callbacks/activity
  if (trace_hsa_api) {
    std::ostringstream out;
    out << "    HSA-trace(";
    if (hsa_api_vec.size() != 0) {
      out << "-*";
      for (unsigned i = 0; i < hsa_api_vec.size(); ++i) {
        uint32_t cid = HSA_API_ID_NUMBER;
        const char* api = hsa_api_vec[i].c_str();
        if (roctracer_op_code(ACTIVITY_DOMAIN_HSA_API, api, &cid, nullptr) ==
                ROCTRACER_STATUS_SUCCESS &&
            roctracer_enable_op_callback(ACTIVITY_DOMAIN_HSA_API, cid, hsa_api_callback, nullptr) ==
                ROCTRACER_STATUS_SUCCESS)
          out << ' ' << api;
        else
          warning("Unable to enable HSA_API tracing for invalid operation %s", api);
      }
    } else {
      CHECK_ROCTRACER(
          roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HSA_API, hsa_api_callback, nullptr));
      out << "*";
    }
    std::cout << out.str() << ')' << std::endl;
  }

  // Enable HSA GPU activity
  if (trace_hsa_activity) {
    // Allocating tracing pool
    open_tracing_pool();

    std::cout << "    HSA-activity-trace()" << std::endl;
    CHECK_ROCTRACER(roctracer_enable_op_activity(ACTIVITY_DOMAIN_HSA_OPS, HSA_OP_ID_COPY));
  }

  // Enable HIP API callbacks/activity
  if (trace_hip_api || trace_hip_activity) {
    std::ostringstream out;
    out << "    HIP-trace(";
    // Allocating tracing pool
    open_tracing_pool();

    // Enable tracing
    if (trace_hip_api) {
      if (hip_api_vec.size() != 0) {
        out << "-*";
        for (unsigned i = 0; i < hip_api_vec.size(); ++i) {
          uint32_t cid = HIP_API_ID_NONE;
          const char* api = hip_api_vec[i].c_str();
          if (roctracer_op_code(ACTIVITY_DOMAIN_HIP_API, api, &cid, nullptr) ==
                  ROCTRACER_STATUS_SUCCESS &&
              roctracer_enable_op_callback(ACTIVITY_DOMAIN_HIP_API, cid, hip_api_callback,
                                           nullptr) == ROCTRACER_STATUS_SUCCESS)
            out << ' ' << api;
          else
            warning("Unable to enable HIP_API tracing for invalid operation %s", api);
        }
      } else {
        CHECK_ROCTRACER(
            roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HIP_API, hip_api_callback, nullptr));
        out << "*";
      }
    }

    if (trace_hip_activity) {
      CHECK_ROCTRACER(roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HIP_OPS));
    }
    std::cout << out.str() << ')' << std::endl;
  }

  // Enable PC sampling
  if (trace_pcs) {
    std::cout << "    PCS-trace()" << std::endl;
    open_tracing_pool();
    CHECK_ROCTRACER(roctracer_enable_op_activity(ACTIVITY_DOMAIN_HSA_OPS, HSA_OP_ID_RESERVED1));
  }
  return true;
}

// HSA-runtime on-unload method
ROCTRACER_EXPORT void OnUnload() { tool_unload(); }

}  // extern "C"

void initialize() {
  tool_load();
}
