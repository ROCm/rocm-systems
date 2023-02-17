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

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <hsa/hsa.h>
#include <mutex>
#include <sys/stat.h>

#include "rocprofiler.h"
#include "rocprofiler_plugin.h"
#include "../utils.h"

namespace {

class att_plugin_t {
 public:
  att_plugin_t() {}

  std::mutex writing_lock;
  bool is_valid_{true};

  inline bool att_file_exists(const std::string& name) {
    struct stat buffer;
    return stat(name.c_str(), &buffer) == 0;
  }

  bool IsValid() const { return is_valid_; }

  void FlushATTRecord(const rocprofiler_record_att_tracer_t* att_tracer_record,
                      rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t buffer_id) {
    std::lock_guard<std::mutex> lock(writing_lock);

    if (!att_tracer_record) {
      printf("No att data buffer received\n");
      return;
    }

    size_t name_length;
    CHECK_ROCPROFILER(rocprofiler_query_kernel_info_size(ROCPROFILER_KERNEL_NAME,
                                                       att_tracer_record->kernel_id, &name_length));
    const char* kernel_name_c = static_cast<const char*>(malloc(name_length * sizeof(char)));
    CHECK_ROCPROFILER(rocprofiler_query_kernel_info(ROCPROFILER_KERNEL_NAME,
                                                  att_tracer_record->kernel_id, &kernel_name_c));

    std::string name_demangled = rocmtools::truncate_name(rocmtools::cxx_demangle(kernel_name_c));

    // Get the number of shader engine traces
    int se_num = att_tracer_record->shader_engine_data_count;
    std::string outpath;
    if (getenv("OUTPUT_PATH") == nullptr) {
      outpath = "";
    } else {
      outpath = std::string(getenv("OUTPUT_PATH")) + "/";
    }
    // Find if this filename already exists. If so, increment vname.
    int file_iteration = -1;
    bool bIncrementVersion = true;
    while (bIncrementVersion) {
      file_iteration += 1;
      std::string fss = name_demangled + "_v" + std::to_string(file_iteration);
      bIncrementVersion = att_file_exists(outpath + fss + "_kernel.txt");
    }

    std::string fname =
        outpath + name_demangled + "_v" + std::to_string(file_iteration) + "_kernel.txt";
    std::ofstream(fname.c_str()) << name_demangled << ": " << kernel_name_c << '\n';

    // iterate over each shader engine att trace
    for (int i = 0; i < se_num; i++) {
      if (!att_tracer_record->shader_engine_data &&
          !att_tracer_record->shader_engine_data[i].buffer_ptr)
        continue;
      printf("--------------collecting data for shader_engine %d---------------\n", i);
      rocprofiler_record_se_att_data_t* se_att_trace = &att_tracer_record->shader_engine_data[i];
      uint32_t size = se_att_trace->buffer_size;
      const char* data_buffer_ptr = reinterpret_cast<char*>(se_att_trace->buffer_ptr);

      // dump data in binary format
      std::ostringstream oss;
      oss << outpath + name_demangled << "_v" << file_iteration << "_se" << i << ".att";
      std::ofstream out(oss.str().c_str(), std::ios::binary);
      if (out.is_open()) {
        out.write((char*)data_buffer_ptr, size);
        out.close();
      } else {
        std::cerr << "\t" << __FUNCTION__ << " Failed to open file: " << oss.str().c_str() << '\n';
      }
    }
  }

  int WriteBufferRecords(const rocprofiler_record_header_t* begin,
                         const rocprofiler_record_header_t* end,
                         rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t buffer_id) {
    while (begin < end) {
      if (!begin) return 0;
      switch (begin->kind) {
        case ROCPROFILER_PROFILER_RECORD:
        case ROCPROFILER_TRACER_RECORD:
        case ROCPROFILER_PC_SAMPLING_RECORD:
        case ROCPROFILER_SPM_RECORD:
          printf("Invalid record Kind: %d", begin->kind);
          break;

        case ROCPROFILER_ATT_TRACER_RECORD: {
          rocprofiler_record_att_tracer_t* att_record =
              const_cast<rocprofiler_record_att_tracer_t*>(
                  reinterpret_cast<const rocprofiler_record_att_tracer_t*>(begin));
          FlushATTRecord(att_record, session_id, buffer_id);
          break;
        }
      }
      rocprofiler_next_record(begin, &begin, session_id, buffer_id);
    }

    return 0;
  }

 private:
};

att_plugin_t* att_plugin = nullptr;

}  // namespace

ROCPROFILER_EXPORT int rocprofiler_plugin_initialize(uint32_t rocprofiler_major_version,
                                                     uint32_t rocprofiler_minor_version) {
  if (rocprofiler_major_version != ROCPROFILER_VERSION_MAJOR ||
      rocprofiler_minor_version < ROCPROFILER_VERSION_MINOR)
    return -1;

  if (att_plugin != nullptr) return -1;

  att_plugin = new att_plugin_t();
  if (att_plugin->IsValid()) return 0;

  // The plugin failed to initialied, destroy it and return an error.
  delete att_plugin;
  att_plugin = nullptr;
  return -1;
}

ROCPROFILER_EXPORT void rocprofiler_plugin_finalize() {
  if (!att_plugin) return;
  delete att_plugin;
  att_plugin = nullptr;
}

ROCPROFILER_EXPORT int rocprofiler_plugin_write_buffer_records(
    const rocprofiler_record_header_t* begin, const rocprofiler_record_header_t* end,
    rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t buffer_id) {
  if (!att_plugin || !att_plugin->IsValid()) return -1;
  return att_plugin->WriteBufferRecords(begin, end, session_id, buffer_id);
}

ROCPROFILER_EXPORT int rocprofiler_plugin_write_record(rocprofiler_record_tracer_t record,
                                                       rocprofiler_session_id_t session_id) {
  if (!att_plugin || !att_plugin->IsValid()) return -1;
  if (record.header.id.handle == 0) return 0;
  return 0;
}
