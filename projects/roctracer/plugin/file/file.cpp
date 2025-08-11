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

#include "debug.h"

#include <roctracer_ext.h>
#include <roctracer_hip.h>
#include <roctracer_hsa.h>
#include <roctracer_plugin.h>
#include <roctracer_roctx.h>

#include <cstddef>
#include <cstdint>
#include <experimental/filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>

#include <amd_comgr/amd_comgr.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include <cassert>

// Macro to check ROCtracer calls status
#define CHECK_ROCTRACER(call)                                                                      \
  do {                                                                                             \
    if ((call) != 0) fatal("%s", roctracer_error_string());                                        \
  } while (false)

namespace fs = std::experimental::filesystem;

namespace {

uint32_t GetPid() {
  static uint32_t pid = syscall(__NR_getpid);
  return pid;
}

/* The function extracts the kernel name from
input string. By using the iterators it finds the
window in the string which contains only the kernel name.
For example 'Foo<int, float>::foo(a[], int (int))' -> 'foo'*/
std::string truncate_name(const std::string& name) {
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
      if (counter == 0) break;
    } else {
      if (*rit == open_token) counter++;
      if (*rit == close_token) counter--;
    }
    ++rit;
  }
  auto rbeg = rit;
  while ((rit != rend) && (*rit != ' ') && (*rit != ':')) rit++;
  return name.substr(rend - rit, rit - rbeg);
}

#define amd_comgr_(call)                                                                           \
  do {                                                                                             \
    if (amd_comgr_status_t status = amd_comgr_##call; status != AMD_COMGR_STATUS_SUCCESS) {        \
      const char* reason = "";                                                                     \
      amd_comgr_status_string(status, &reason);                                                    \
      fatal(#call " failed: %s", reason);                                                          \
    }                                                                                              \
  } while (false)

// C++ symbol demangle
std::string cxx_demangle(const std::string& symbol) {
  amd_comgr_data_t mangled_data;
  amd_comgr_(create_data(AMD_COMGR_DATA_KIND_BYTES, &mangled_data));
  amd_comgr_(set_data(mangled_data, symbol.size(), symbol.data()));

  amd_comgr_data_t demangled_data;
  amd_comgr_(demangle_symbol_name(mangled_data, &demangled_data));

  size_t demangled_size = 0;
  amd_comgr_(get_data(demangled_data, &demangled_size, nullptr));

  std::string demangled_str;
  demangled_str.resize(demangled_size);
  amd_comgr_(get_data(demangled_data, &demangled_size, demangled_str.data()));

  amd_comgr_(release_data(mangled_data));
  amd_comgr_(release_data(demangled_data));
  return demangled_str;
}

class file_plugin_t {
 private:
  class output_file_t {
   public:
    output_file_t(std::string name) : name_(std::move(name)) {}

    std::string name() const { return name_; }

    template <typename T> std::ostream& operator<<(T&& value) {
      if (!is_open()) open();
      return stream_ << std::forward<T>(value);
    }

    std::ostream& operator<<(std::ostream& (*func)(std::ostream&)) {
      if (!is_open()) open();
      return stream_ << func;
    }

    void open() {
      // If the stream is already in the failed state, there's no need to try to open the file.
      if (fail()) return;

      const char* output_dir = getenv("ROCP_OUTPUT_DIR");

      if (output_dir == nullptr) {
        stream_.copyfmt(std::cout);
        stream_.clear(std::cout.rdstate());
        stream_.basic_ios<char>::rdbuf(std::cout.rdbuf());
        return;
      }

      fs::path output_prefix(output_dir);
      if (!fs::is_directory(fs::status(output_prefix))) {
        if (!stream_.fail()) warning("Cannot open output directory '%s'", output_dir);
        stream_.setstate(std::ios_base::failbit);
        return;
      }

      std::stringstream ss;
      ss << GetPid() << "_" << name_;
      stream_.open(output_prefix / ss.str());
    }

    bool is_open() const { return stream_.is_open(); }
    bool fail() const { return stream_.fail(); }

   private:
    const std::string name_;
    std::ofstream stream_;
  };

  output_file_t* get_output_file(uint32_t domain, uint32_t op = 0) {
    switch (domain) {
      case ACTIVITY_DOMAIN_ROCTX:
        return &roctx_file_;
      case ACTIVITY_DOMAIN_HSA_API:
        return &hsa_api_file_;
      case ACTIVITY_DOMAIN_HIP_API:
        return &hip_api_file_;
      case ACTIVITY_DOMAIN_HIP_OPS:
        return &hip_activity_file_;
      case ACTIVITY_DOMAIN_HSA_OPS:
        if (op == HSA_OP_ID_COPY) {
          return &hsa_async_copy_file_;
        } else if (op == HSA_OP_ID_RESERVED1) {
          return &pc_sample_file_;
        }
      default:
        assert(!"domain/op not supported!");
        break;
    }
    return nullptr;
  }

 public:
  file_plugin_t() {
    // Dumping HSA handles for agents
    output_file_t hsa_handles("hsa_handles.txt");

    [[maybe_unused]] hsa_status_t status = hsa_iterate_agents(
        [](hsa_agent_t agent, void* user_data) {
          auto* file = static_cast<decltype(hsa_handles)*>(user_data);
          hsa_device_type_t type;

          if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type) != HSA_STATUS_SUCCESS)
            return HSA_STATUS_ERROR;

          *file << std::hex << std::showbase << agent.handle << " agent "
                << ((type == HSA_DEVICE_TYPE_CPU) ? "cpu" : "gpu") << "\n";
          return HSA_STATUS_SUCCESS;
        },
        &hsa_handles);
    assert(status == HSA_STATUS_SUCCESS && "failed to iterate HSA agents");
    if (hsa_handles.fail()) {
      warning("Cannot write to '%s'", hsa_handles.name().c_str());
      return;
    }

    // App begin timestamp begin_ts_file.txt
    output_file_t begin_ts("begin_ts_file.txt");

    roctracer_timestamp_t app_begin_timestamp;
    CHECK_ROCTRACER(roctracer_get_timestamp(&app_begin_timestamp));
    begin_ts << std::dec << app_begin_timestamp << "\n";
    if (begin_ts.fail()) {
      warning("Cannot write to '%s'", begin_ts.name().c_str());
      return;
    }

    valid_ = true;
  }

  int write_callback_record(const roctracer_record_t* record, const void* callback_data) {
    std::stringstream ss;
    output_file_t* output_file{nullptr};
    switch (record->domain) {
      case ACTIVITY_DOMAIN_ROCTX: {
        const roctx_api_data_t* data = reinterpret_cast<const roctx_api_data_t*>(callback_data);
        output_file = get_output_file(ACTIVITY_DOMAIN_ROCTX);
        ss << std::dec << record->begin_ns << " " << record->process_id << ":" << record->thread_id
           << " " << record->op << ":" << data->args.id << ":\""
           << (data->args.message ? data->args.message : "") << "\""
           << "\n";
        *output_file << ss.str();
        break;
      }
      case ACTIVITY_DOMAIN_HSA_API: {
        const hsa_api_data_t* data = reinterpret_cast<const hsa_api_data_t*>(callback_data);
        output_file = get_output_file(ACTIVITY_DOMAIN_HSA_API);
        ss << std::dec << record->begin_ns << ":"
           << ((record->op == HSA_API_ID_hsa_shut_down) ? record->begin_ns : record->end_ns) << " "
           << record->process_id << ":" << record->thread_id << " "
           << hsa_api_data_pair_t(record->op, *data) << " :" << std::dec << data->correlation_id
           << "\n";
        *output_file << ss.str();
        break;
      }
      case ACTIVITY_DOMAIN_HIP_API: {
        const hip_api_data_t* data = reinterpret_cast<const hip_api_data_t*>(callback_data);

        std::string kernel_name;
        if (record->kernel_name) {
          static bool truncate = []() {
            const char* env_var = getenv("ROCP_TRUNCATE_NAMES");
            return env_var && std::atoi(env_var) != 0;
          }();
          kernel_name = cxx_demangle(record->kernel_name);
          if (truncate) kernel_name = truncate_name(kernel_name);
          kernel_name = " kernel=" + kernel_name;
        }

        output_file = get_output_file(ACTIVITY_DOMAIN_HIP_API);
        ss << std::dec << record->begin_ns << ":" << record->end_ns << " " << record->process_id
           << ":" << record->thread_id << " " << hipApiString((hip_api_id_t)record->op, data)
           << kernel_name << " :" << std::dec << data->correlation_id << "\n";
        *output_file << ss.str();
        break;
      }
      default:
        warning("write_callback_record: ignored record for domain %d", record->domain);
        break;
    }

    return (output_file && output_file->fail()) ? -1 : 0;
  }

  int write_activity_records(const roctracer_record_t* begin, const roctracer_record_t* end) {
    while (begin != end) {
      std::stringstream ss;
      output_file_t* output_file{nullptr};
      const char* name = roctracer_op_string(begin->domain, begin->op, begin->kind);

      switch (begin->domain) {
        case ACTIVITY_DOMAIN_HIP_OPS: {
          // The post-processing script cannot handle HIP ops without a correlation ID. The
          // correlation ID is needed to connect the record to a HIP stream and originating thread.
          // The script could be modified to handle ops without correlation IDs, but for backward
          // compatibilty, we are simply dropping the records here.
          if (begin->correlation_id == 0) break;

          output_file = get_output_file(ACTIVITY_DOMAIN_HIP_OPS);
          ss << std::dec << begin->begin_ns << ":" << begin->end_ns << " " << begin->device_id
             << ":" << begin->queue_id << " "
             << ((begin->op == HIP_OP_ID_DISPATCH && begin->kernel_name != nullptr)
                     ? truncate_name(cxx_demangle(begin->kernel_name))
                     : name)
             << ":" << begin->correlation_id << ":" << GetPid() << "\n";
          *output_file << ss.str();
          break;
        }
        case ACTIVITY_DOMAIN_HSA_OPS:
          output_file = get_output_file(ACTIVITY_DOMAIN_HSA_OPS, begin->op);
          if (begin->op == HSA_OP_ID_COPY) {
            ss << std::dec << begin->begin_ns << ":" << begin->end_ns
               << " async-copy:" << begin->correlation_id << ":" << GetPid() << "\n";
            *output_file << ss.str();
            break;
          } else if (begin->op == HSA_OP_ID_RESERVED1) {
            ss << std::dec << begin->pc_sample.se << " " << begin->pc_sample.cycle << " "
               << std::hex << std::showbase << begin->pc_sample.pc << " " << name << "\n";
            *output_file << ss.str();
            break;
          }
          [[fallthrough]];
        default: {
          warning("write_activity_records: ignored activity for domain %d", begin->domain);
          break;
        }
      }
      if (output_file && output_file->fail()) return -1;
      CHECK_ROCTRACER(roctracer_next_record(begin, &begin));
    }
    return 0;
  }

  bool is_valid() const { return valid_; }

 private:
  bool valid_{false};

  output_file_t roctx_file_{"roctx_trace.txt"}, hsa_api_file_{"hsa_api_trace.txt"},
      hip_api_file_{"hip_api_trace.txt"}, hip_activity_file_{"hcc_ops_trace.txt"},
      hsa_async_copy_file_{"async_copy_trace.txt"}, pc_sample_file_{"pcs_trace.txt"};
};

file_plugin_t* file_plugin = nullptr;

}  // namespace

ROCTRACER_EXPORT int roctracer_plugin_initialize(uint32_t roctracer_major_version,
                                                 uint32_t roctracer_minor_version) {
  if (roctracer_major_version != ROCTRACER_VERSION_MAJOR ||
      roctracer_minor_version < ROCTRACER_VERSION_MINOR)
    return -1;

  if (file_plugin != nullptr) return -1;

  file_plugin = new file_plugin_t();
  if (file_plugin->is_valid()) return 0;

  // The plugin failed to initialied, destroy it and return an error.
  delete file_plugin;
  file_plugin = nullptr;
  return -1;
}

ROCTRACER_EXPORT void roctracer_plugin_finalize() {
  if (!file_plugin) return;
  delete file_plugin;
  file_plugin = nullptr;
}

ROCTRACER_EXPORT int roctracer_plugin_write_callback_record(const roctracer_record_t* record,
                                                            const void* callback_data) {
  if (!file_plugin || !file_plugin->is_valid()) return -1;
  return file_plugin->write_callback_record(record, callback_data);
}

ROCTRACER_EXPORT int roctracer_plugin_write_activity_records(const roctracer_record_t* begin,
                                                             const roctracer_record_t* end) {
  if (!file_plugin || !file_plugin->is_valid()) return -1;
  return file_plugin->write_activity_records(begin, end);
}
