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

#include <cxxabi.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <hsa/hsa.h>
#include <mutex>
#include <unordered_map>

#include "rocprofiler.h"
#include "rocprofiler_plugin.h"
#include "../utils.h"

#include "src/utils/filesystem.hpp"

namespace fs = rocprofiler::common::filesystem;

namespace {

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

class file_plugin_t {
 private:
  enum class output_type_t { COUNTER, TRACER, PC_SAMPLING };

  class output_file_t {
   public:
    output_file_t(std::string name, bool bOpenOnInit = false) : name_(std::move(name)) {
      if (bOpenOnInit) open();
    }

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
      // If the stream is already in the failed state, there's no need to try
      // to open the file.
      if (fail()) return;

      const char* output_dir = getenv("OUTPUT_PATH");
      std::string output_file_name = getenv("OUT_FILE_NAME") ? getenv("OUT_FILE_NAME") : "";

      if (output_dir == nullptr && getenv("OUT_FILE_NAME") == nullptr) {
        stream_.copyfmt(std::cout);
        stream_.clear(std::cout.rdstate());
        stream_.basic_ios<char>::rdbuf(std::cout.rdbuf());
        bPrintToStdout = true;
        return;
      }
      if (output_dir == nullptr) output_dir = ".";

      fs::path output_prefix(output_dir);
      if (!fs::is_directory(fs::status(output_prefix))) {
        if (!stream_.fail()) rocprofiler::warning("Cannot open output directory '%s'", output_dir);
        stream_.setstate(std::ios_base::failbit);
        return;
      }

      output_file_name = replace_MPI_macros(output_file_name);

      std::stringstream ss;
      ss << name_ << "_" << ((output_file_name.empty()) ? std::to_string(GetPid()) : "")
         << output_file_name << ".csv";
      std::cout << "Results File: " << output_prefix / ss.str() << std::endl;
      stream_.open(output_prefix / ss.str());
    }

    bool is_open() const { return stream_.is_open(); }
    bool fail() const { return stream_.fail(); }
    bool isStdOut() const { return bPrintToStdout; }

    // Returns a string with the MPI %macro replaced with the corresponding envvar
    std::string replace_MPI_macros(std::string output_file_name) {
      std::unordered_map<const char*, const char*> MPI_BUILTINS = {
          {"MPI_RANK", "%rank"},
          {"OMPI_COMM_WORLD_RANK", "%rank"},
          {"MV2_COMM_WORLD_RANK", "%rank"}};

      for (const auto& [envvar, key] : MPI_BUILTINS) {
        size_t key_find = output_file_name.rfind(key);
        if (key_find == std::string::npos) continue;  // Does not contain a %?rank var

        const char* env_var_set = getenv(envvar);
        if (env_var_set == nullptr) continue;  // MPI_COMM_WORLD_x var is does not exist

        int rank = atoi(env_var_set);
        output_file_name = output_file_name.substr(0, key_find) + std::to_string(rank) +
            output_file_name.substr(key_find + std::string(key).size());
      }

      return output_file_name;
    }

   private:
    const std::string name_;
    std::ofstream stream_;
    bool bPrintToStdout = false;
  };

  output_file_t* get_output_file(output_type_t output_type, uint32_t domain = 0) {
    switch (output_type) {
      case output_type_t::COUNTER:
        return &output_file_;
      case output_type_t::TRACER:
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
            return &hsa_async_copy_file_;
          default:
            assert(!"domain/op not supported!");
            break;
        }
        break;
      case output_type_t::PC_SAMPLING:
        return &pc_sample_file_;
    }
    return nullptr;
  }

 public:
  file_plugin_t(void* data) {
    if (data) counter_names_ = GetCounterNames();

    valid_ = true;
  }

  void WriteHeader(output_type_t type, rocprofiler_tracer_activity_domain_t domain) {
    output_file_t* output_file;
    switch (domain) {
      case ACTIVITY_DOMAIN_HSA_API: {
        if (hsa_api_header_written_.load(std::memory_order_relaxed)) return;
        output_file = get_output_file(output_type_t::TRACER, ACTIVITY_DOMAIN_HSA_API);
        *output_file << "Domain,Function,Start_Timestamp,End_Timestamp,Correlation_ID" << std::endl;
        *output_file << std::endl;
        hsa_api_header_written_.exchange(true, std::memory_order_release);
        return;
      }
      case ACTIVITY_DOMAIN_HIP_API: {
        if (hip_api_header_written_.load(std::memory_order_relaxed)) return;
        output_file = get_output_file(output_type_t::TRACER, ACTIVITY_DOMAIN_HIP_API);
        *output_file << "Domain,Function,Start_Timestamp,End_Timestamp,Correlation_ID" << std::endl;
        *output_file << std::endl;
        hip_api_header_written_.exchange(true, std::memory_order_release);
        return;
      }
      case ACTIVITY_DOMAIN_ROCTX: {
        if (roctx_header_written_.load(std::memory_order_relaxed)) return;
        output_file = get_output_file(output_type_t::TRACER, ACTIVITY_DOMAIN_ROCTX);
        *output_file << "Domain,ROCTX_ID,Message,Timestamp" << std::endl;
        *output_file << std::endl;
        roctx_header_written_.exchange(true, std::memory_order_release);
        return;
      }
      case ACTIVITY_DOMAIN_HSA_OPS: {
        if (hsa_async_copy_header_written_.load(std::memory_order_relaxed)) return;
        output_file = get_output_file(output_type_t::TRACER, ACTIVITY_DOMAIN_HSA_OPS);
        *output_file << "Domain,Operation,Start_Timestamp,Stop_Timestamp,Correlation_ID"
                     << std::endl;
        *output_file << std::endl;
        hsa_async_copy_header_written_.exchange(true, std::memory_order_release);
        return;
      }
      case ACTIVITY_DOMAIN_HIP_OPS: {
        if (hip_activity_header_written_.load(std::memory_order_relaxed)) return;
        output_file = get_output_file(output_type_t::TRACER, ACTIVITY_DOMAIN_HIP_OPS);
        *output_file << "Domain,Operation,Kernel_Name,Start_Timestamp,Stop_Timestamp,"
                        "Correlation_ID"
                     << std::endl;
        *output_file << std::endl;
        hip_activity_header_written_.exchange(true, std::memory_order_release);
        return;
      }
      default: {
        if (type == output_type_t::COUNTER) {
          if (kernel_dispatches_header_written_.load(std::memory_order_relaxed)) return;
          output_file = get_output_file(output_type_t::COUNTER);

          *output_file << "Index,KernelName,gpu-id,queue-id,queue-index,pid,tid,grd,wgr,lds,scr,"
                          "arch_vgpr,accum_vgpr,sgpr,wave_size";
          if (counter_names_.size() > 0) {
            for (uint32_t i = 0; i < counter_names_.size(); i++)
              *output_file << "," << counter_names_[i];
          }
          *output_file << ",DispatchNs,BeginNs,EndNs,CompleteNs";
          *output_file << std::endl;
          *output_file << std::endl;
          kernel_dispatches_header_written_.exchange(true, std::memory_order_release);
          return;
        } else if (type == output_type_t::PC_SAMPLING) {
          if (pc_sample_header_written_.load(std::memory_order_relaxed)) return;
          output_file = get_output_file(output_type_t::PC_SAMPLING);
          *output_file << "Dispatch_ID,Timestamp,GPU_ID,PC_Sample,Shader_Engines" << std::endl;
          *output_file << std::endl;
          pc_sample_header_written_.exchange(true, std::memory_order_release);
          return;
        }
        return;
      }
    }
  }

  std::mutex writing_lock;

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

  void FlushTracerRecord(rocprofiler_record_tracer_t tracer_record,
                         rocprofiler_session_id_t session_id,
                         rocprofiler_buffer_id_t buffer_id = rocprofiler_buffer_id_t{0}) {
    std::lock_guard<std::mutex> lock(writing_lock);
    if (tracer_record.timestamps.end.value <= 0 && tracer_record.domain != ACTIVITY_DOMAIN_ROCTX)
      return;
    WriteHeader(output_type_t::TRACER, tracer_record.domain);
    std::string roctx_message;
    if (tracer_record.domain == ACTIVITY_DOMAIN_ROCTX && tracer_record.name) {
      roctx_message = tracer_record.name;
    }

    const char* operation_name_c = nullptr;
    // ROCTX domain Operation ID doesn't have a name
    // It depends on the user input of the roctx functions.
    // ROCTX message is the tracer_record.name
    if (tracer_record.domain != ACTIVITY_DOMAIN_ROCTX) {
      CHECK_ROCPROFILER(rocprofiler_query_tracer_operation_name(
          tracer_record.domain, tracer_record.operation_id, &operation_name_c));
    }
    output_file_t* output_file = get_output_file(output_type_t::TRACER, tracer_record.domain);
    *output_file << GetDomainName(tracer_record.domain);
    if (tracer_record.domain == ACTIVITY_DOMAIN_ROCTX && tracer_record.external_id.id >= 0)
      *output_file << "," << tracer_record.external_id.id;
    if (tracer_record.domain == ACTIVITY_DOMAIN_ROCTX) {
      if (roctx_message.size() > 1)
        *output_file << ",\"" << roctx_message << "\"";
      else
        *output_file << ",";
    }
    if (operation_name_c) *output_file << ",\"" << operation_name_c << "\"";
    if (tracer_record.name && tracer_record.domain != ACTIVITY_DOMAIN_ROCTX) {
      *output_file << ",\"" << rocprofiler::truncate_name(rocprofiler::cxx_demangle(tracer_record.name)) << "\"";
    } else if (tracer_record.domain == ACTIVITY_DOMAIN_HIP_OPS) {
      *output_file << ",";
    }
    if (tracer_record.domain != ACTIVITY_DOMAIN_ROCTX) {
      *output_file << "," << tracer_record.timestamps.begin.value << ","
                   << tracer_record.timestamps.end.value;
      *output_file << "," << tracer_record.correlation_id.value;
    } else {
      *output_file << "," << tracer_record.timestamps.begin.value;
    }
    *output_file << std::endl;
  }

  void FlushProfilerRecord(const rocprofiler_record_profiler_t* profiler_record,
                           rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t buffer_id) {
    std::lock_guard<std::mutex> lock(writing_lock);
    WriteHeader(output_type_t::COUNTER, ACTIVITY_DOMAIN_NUMBER);
    size_t name_length = 0;
    output_file_t* output_file{nullptr};
    output_file = get_output_file(output_type_t::COUNTER);
    CHECK_ROCPROFILER(rocprofiler_query_kernel_info_size(ROCPROFILER_KERNEL_NAME,
                                                         profiler_record->kernel_id, &name_length));
    // Taken from rocprofiler: The size hasn't changed in  recent past
    static const uint32_t lds_block_size = 128 * 4;
    const char* kernel_name_c = nullptr;
    if (name_length > 1) {
      CHECK_ROCPROFILER(rocprofiler_query_kernel_info(ROCPROFILER_KERNEL_NAME,
                                                      profiler_record->kernel_id, &kernel_name_c));
    }
    *output_file << std::to_string(profiler_record->header.id.handle) << ",";
    std::string kernel_name = "";
    if (name_length > 1) {
      kernel_name = rocprofiler::truncate_name(rocprofiler::cxx_demangle(kernel_name_c));
      std::string key = "\"";
      std::size_t found = kernel_name.rfind(key);
      while (found != std::string::npos) {
        kernel_name.replace(found, key.length(), "'");
        found = kernel_name.rfind(key, found - 1);
      }
    }
    *output_file << "\"" << kernel_name << "\",";
    *output_file << std::to_string(profiler_record->gpu_id.handle) << ","
                 << std::to_string(profiler_record->queue_id.handle) << ","
                 << std::to_string(profiler_record->queue_idx.value) << ","
                 << std::to_string(GetPid()) << ","
                 << std::to_string(profiler_record->thread_id.value) << ","
                 << std::to_string(profiler_record->kernel_properties.grid_size) << ","
                 << std::to_string(profiler_record->kernel_properties.workgroup_size) << ","
                 << std::to_string(
                        ((profiler_record->kernel_properties.lds_size + (lds_block_size - 1)) &
                         ~(lds_block_size - 1)))
                 << "," << std::to_string(profiler_record->kernel_properties.scratch_size) << ","
                 << std::to_string(profiler_record->kernel_properties.arch_vgpr_count) << ","
                 << std::to_string(profiler_record->kernel_properties.accum_vgpr_count) << ","
                 << std::to_string(profiler_record->kernel_properties.sgpr_count) << ","
                 << std::to_string(profiler_record->kernel_properties.wave_size);

    // For Counters
    if (profiler_record->counters) {
      for (uint64_t i = 0; i < profiler_record->counters_count.value; i++) {
        if (profiler_record->counters[i].counter_handler.handle > 0) {
          *output_file << "," << std::to_string(profiler_record->counters[i].value.value);
        }
      }
    }
    *output_file << ",0,"
                 << std::to_string(profiler_record->timestamps.begin.value) << ","
                 << std::to_string(profiler_record->timestamps.end.value) << ",0";
    *output_file << '\n';
    if (kernel_name_c) {
      free(const_cast<char*>(kernel_name_c));
    }
  }

  void FlushPCSamplingRecord(const rocprofiler_record_pc_sample_t* pc_sampling_record) {
    WriteHeader(output_type_t::PC_SAMPLING, ACTIVITY_DOMAIN_NUMBER);
    output_file_t* output_file{nullptr};
    output_file = get_output_file(output_type_t::PC_SAMPLING);
    const auto& sample = pc_sampling_record->pc_sample;
    *output_file << sample.dispatch_id.value << "," << sample.timestamp.value << ","
                 << sample.gpu_id.handle << "," << std::hex << std::showbase << sample.pc << ","
                 << sample.se << std::endl;
  }
  int WriteBufferRecords(const rocprofiler_record_header_t* begin,
                         const rocprofiler_record_header_t* end,
                         rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t buffer_id) {
    while (begin < end) {
      if (!begin) return 0;
      switch (begin->kind) {
        case ROCPROFILER_PROFILER_RECORD: {
          const rocprofiler_record_profiler_t* profiler_record =
              reinterpret_cast<const rocprofiler_record_profiler_t*>(begin);
          FlushProfilerRecord(profiler_record, session_id, buffer_id);
          break;
        }
        case ROCPROFILER_TRACER_RECORD: {
          rocprofiler_record_tracer_t* tracer_record = const_cast<rocprofiler_record_tracer_t*>(
              reinterpret_cast<const rocprofiler_record_tracer_t*>(begin));
          FlushTracerRecord(*tracer_record, session_id, buffer_id);
          break;
        }
        case ROCPROFILER_ATT_TRACER_RECORD: {
          break;
        }
        case ROCPROFILER_PC_SAMPLING_RECORD: {
          [[deprecated("PC Sampling is deprecated")]]
          const rocprofiler_record_pc_sample_t* pc_sampling_record =
              reinterpret_cast<const rocprofiler_record_pc_sample_t*>(begin);
          FlushPCSamplingRecord(pc_sampling_record);
          break;
        }
        default:
          break;
      }
      rocprofiler_next_record(begin, &begin, session_id, buffer_id);
    }
    return 0;
  }

  bool is_valid() const { return valid_; }

 private:
  bool valid_{false};
  std::vector<std::string> counter_names_;

  std::atomic<bool> roctx_header_written_{false}, hsa_api_header_written_{false},
      hip_api_header_written_{false}, hip_activity_header_written_{false},
      hsa_async_copy_header_written_{false}, pc_sample_header_written_{false},
      kernel_dispatches_header_written_{false};

  output_file_t roctx_file_{"roctx_trace"}, hsa_api_file_{"hsa_api_trace"},
      hip_api_file_{"hip_api_trace"}, hip_activity_file_{"hcc_ops_trace"},
      hsa_async_copy_file_{"async_copy_trace"}, pc_sample_file_{"pcs_trace"},
      output_file_{"results"};
};

file_plugin_t* file_plugin = nullptr;

}  // namespace

ROCPROFILER_EXPORT int rocprofiler_plugin_initialize(uint32_t rocprofiler_major_version,
                                                     uint32_t rocprofiler_minor_version,
                                                     void* data) {
  if (rocprofiler_major_version != ROCPROFILER_VERSION_MAJOR ||
      rocprofiler_minor_version < ROCPROFILER_VERSION_MINOR)
    return -1;

  if (file_plugin != nullptr) return -1;

  file_plugin = new file_plugin_t(data);
  if (file_plugin->is_valid()) return 0;

  // The plugin failed to initialized, destroy it and return an error.
  delete file_plugin;
  file_plugin = nullptr;
  return -1;
}

ROCPROFILER_EXPORT void rocprofiler_plugin_finalize() {
  if (!file_plugin) return;
  delete file_plugin;
  file_plugin = nullptr;
}

ROCPROFILER_EXPORT int rocprofiler_plugin_write_buffer_records(
    const rocprofiler_record_header_t* begin, const rocprofiler_record_header_t* end,
    rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t buffer_id) {
  if (!file_plugin || !file_plugin->is_valid()) return -1;
  return file_plugin->WriteBufferRecords(begin, end, session_id, buffer_id);
}

ROCPROFILER_EXPORT int rocprofiler_plugin_write_record(rocprofiler_record_tracer_t record) {
  if (!file_plugin || !file_plugin->is_valid()) return -1;
  if (record.header.id.handle == 0) return 0;
  file_plugin->FlushTracerRecord(record, rocprofiler_session_id_t{0}, rocprofiler_buffer_id_t{0});
  return 0;
}
