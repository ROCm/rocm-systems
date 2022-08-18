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

#include <cxxabi.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

// Macro to check ROCtracer calls status
#define CHECK_ROCTRACER(call)                                                                      \
  do {                                                                                             \
    if ((call) != 0) fatal("%s\n", roctracer_error_string());                                      \
  } while (false)

namespace fs = std::experimental::filesystem;

namespace {

void warning(const char* format, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 1, 2)))
#endif /* defined (__GNUC__) */
    ;

void fatal [[noreturn]] (const char* format, ...)
#if defined(__GNUC__)
__attribute__((format(printf, 1, 2)))
#endif /* defined (__GNUC__) */
;

void warning(const char* format, ...) {
  va_list va;
  va_start(va, format);
  vfprintf(stderr, format, va);
  va_end(va);
}

void fatal(const char* format, ...) {
  va_list va;
  va_start(va, format);
  vfprintf(stderr, format, va);
  va_end(va);
  abort();
}

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

// C++ symbol demangle
std::string cxx_demangle(const std::string& symbol) {
  int status;
  char* demangled = abi::__cxa_demangle(symbol.c_str(), nullptr, nullptr, &status);
  if (status != 0) return symbol;
  std::string ret(demangled);
  free(demangled);
  return ret;
}

class file_plugin_t {
 private:
  class output_file_t {
   public:
    output_file_t(std::string name) : name_(std::move(name)) {}
    ~output_file_t() { close(); }

    std::string path() const {
      std::stringstream ss;
      ss << GetPid() << "_" << name_;
      return output_prefix_ / ss.str();
    }

    template <typename T> output_file_t& operator<<(T&& value) {
      if (!is_open()) open();
      stream() << std::forward<T>(value);
      return *this;
    }

    output_file_t& operator<<(std::ostream& (*func)(std::ostream&)) {
      if (!is_open()) open();
      stream() << func;
      return *this;
    }

    bool is_open() const { return output_prefix_.empty() || stream_.is_open(); }
    bool fail() const { return stream().fail(); }

    void open() {
      if (output_prefix_.empty()) return;
      stream_.open(path());
    }
    void close() {
      if (stream_.is_open()) stream_.close();
    }

   private:
    std::ostream& stream() { return output_prefix_.empty() ? std::cout : stream_; }
    const std::ostream& stream() const {
      return output_prefix_.empty() ? const_cast<const std::ostream&>(std::cout) : stream_;
    }

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
      case ACTIVITY_DOMAIN_EXT_API:
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
                << ((type == HSA_DEVICE_TYPE_CPU) ? "cpu" : "gpu") << std::endl;
          return HSA_STATUS_SUCCESS;
        },
        &hsa_handles);
    assert(status == HSA_STATUS_SUCCESS && "failed to iterate HSA agents");
    if (hsa_handles.fail()) {
      warning("Cannot write to '%s'\n", hsa_handles.path().c_str());
      return;
    }

    // App begin timestamp begin_ts_file.txt
    output_file_t begin_ts("begin_ts_file.txt");

    roctracer_timestamp_t app_begin_timestamp;
    CHECK_ROCTRACER(roctracer_get_timestamp(&app_begin_timestamp));
    begin_ts << std::dec << app_begin_timestamp << std::endl;
    if (begin_ts.fail()) {
      warning("Cannot write to '%s'\n", begin_ts.path().c_str());
      return;
    }
    is_valid_ = true;
  }

  int write_callback_record(const roctracer_record_t* record, const void* callback_data) {
    output_file_t* output_file{nullptr};
    switch (record->domain) {
      case ACTIVITY_DOMAIN_ROCTX: {
        const roctx_api_data_t* data = reinterpret_cast<const roctx_api_data_t*>(callback_data);
        output_file = get_output_file(ACTIVITY_DOMAIN_ROCTX);
        *output_file << std::dec << record->begin_ns << " " << record->process_id << ":"
                     << record->thread_id << " " << record->op << ":" << data->args.id << ":\""
                     << (data->args.message ? data->args.message : "") << "\"" << std::endl;
        break;
      }
      case ACTIVITY_DOMAIN_HSA_API: {
        const hsa_api_data_t* data = reinterpret_cast<const hsa_api_data_t*>(callback_data);
        output_file = get_output_file(ACTIVITY_DOMAIN_HSA_API);
        *output_file << std::dec << record->begin_ns << ":"
                     << ((record->op == HSA_API_ID_hsa_shut_down) ? record->begin_ns
                                                                  : record->end_ns)
                     << " " << record->process_id << ":" << record->thread_id << " "
                     << hsa_api_data_pair_t(record->op, *data) << " :" << data->correlation_id
                     << std::endl;
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
        *output_file << std::dec << record->begin_ns << ":" << record->end_ns << " "
                     << record->process_id << ":" << record->thread_id << " "
                     << hipApiString((hip_api_id_t)record->op, data) << kernel_name << " :"
                     << data->correlation_id << std::endl;
        break;
      }
      case ACTIVITY_DOMAIN_EXT_API: {
        output_file = get_output_file(ACTIVITY_DOMAIN_EXT_API);
        *output_file << std::dec << record->begin_ns << ":" << record->end_ns << " "
                     << record->process_id << ":" << record->thread_id << " MARK"
                     << "(name(" << record->mark_message << "))" << std::endl;
        break;
        [[fallthrough]];
      }
      default:
        warning("write_callback_record: ignored record for domain %d", record->domain);
        break;
    }

    return (output_file && output_file->fail()) ? -1 : 0;
  }

  int write_activity_records(const roctracer_record_t* begin, const roctracer_record_t* end) {
    while (begin != end) {
      output_file_t* output_file{nullptr};
      const char* name = roctracer_op_string(begin->domain, begin->op, begin->kind);

      switch (begin->domain) {
        case ACTIVITY_DOMAIN_HIP_OPS: {
          output_file = get_output_file(ACTIVITY_DOMAIN_HIP_OPS);
          *output_file << std::dec << begin->begin_ns << ":" << begin->end_ns << " "
                       << begin->device_id << ":" << begin->queue_id << " " << name << ":"
                       << begin->correlation_id << ":" << GetPid() << std::endl;
          break;
        }
        case ACTIVITY_DOMAIN_HSA_OPS:
          output_file = get_output_file(ACTIVITY_DOMAIN_HSA_OPS, begin->op);
          if (begin->op == HSA_OP_ID_COPY) {
            *output_file << std::dec << begin->begin_ns << ":" << begin->end_ns
                         << " async-copy:" << begin->correlation_id << " " << GetPid() << std::endl;
            break;
          } else if (begin->op == HSA_OP_ID_RESERVED1) {
            *output_file << std::dec << begin->pc_sample.se << " " << begin->pc_sample.cycle << " "
                         << std::hex << std::showbase << begin->pc_sample.pc << " " << name
                         << std::endl;
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

  bool is_valid() const { return is_valid_; }

 private:
  bool is_valid_{false};

  output_file_t roctx_file_{"roctx_trace.txt"}, hsa_api_file_{"hsa_api_trace.txt"},
      hip_api_file_{"hip_api_trace.txt"}, hip_activity_file_{"hcc_ops_trace.txt"},
      hsa_async_copy_file_{"async_copy_trace.txt"}, pc_sample_file_{"pcs_trace.txt"};

  static fs::path output_prefix_;
};

fs::path file_plugin_t::output_prefix_ = []() {
  const char* output_dir = getenv("ROCP_OUTPUT_DIR");
  if (output_dir == nullptr) return "";

  if (!fs::is_directory(fs::status(output_dir)))
    fatal("Cannot open output directory specified by ROCP_OUTPUT_DIR!\n");
  return output_dir;
}();

}  // namespace

std::unique_ptr<file_plugin_t> file_plugin;

int roctracer_plugin_initialize(uint32_t roctracer_major_version,
                                uint32_t roctracer_minor_version) {
  if (roctracer_major_version != ROCTRACER_VERSION_MAJOR ||
      roctracer_minor_version < ROCTRACER_VERSION_MINOR)
    return -1;

  if (file_plugin) return -1;
  file_plugin = std::make_unique<file_plugin_t>();
  if (!file_plugin->is_valid()) {
    file_plugin.reset();
    return -1;
  }
  return 0;
}

void roctracer_plugin_finalize() { file_plugin.reset(); }

int roctracer_plugin_write_callback_record(const roctracer_record_t* record,
                                           const void* callback_data) {
  if (!file_plugin || !file_plugin->is_valid()) return -1;
  return file_plugin->write_callback_record(record, callback_data);
}

int roctracer_plugin_write_activity_records(const roctracer_record_t* begin,
                                            const roctracer_record_t* end) {
  if (!file_plugin || !file_plugin->is_valid()) return -1;
  return file_plugin->write_activity_records(begin, end);
}
