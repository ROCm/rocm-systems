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

#define ATT_FILENAME_MAXBYTES 90

namespace {

class att_plugin_t {
 public:
  att_plugin_t() {
    std::vector<const char*> mpivars = {"MPI_RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK"};

    for (const char* envvar : mpivars)
      if (const char* env = getenv(envvar)) {
        MPI_RANK = atoi(env);
        MPI_ENABLE = true;
        break;
      }
  }

  bool MPI_ENABLE = false;
  int MPI_RANK = 0;
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
    CHECK_ROCPROFILER(rocprofiler_query_kernel_info_size(
        ROCPROFILER_KERNEL_NAME, att_tracer_record->kernel_id, &name_length));
    const char* kernel_name_c = static_cast<const char*>(malloc(name_length * sizeof(char)));
    CHECK_ROCPROFILER(rocprofiler_query_kernel_info(ROCPROFILER_KERNEL_NAME,
                                                    att_tracer_record->kernel_id, &kernel_name_c));

    std::string name_demangled =
        rocprofiler::truncate_name(rocprofiler::cxx_demangle(kernel_name_c));

    if (name_demangled.size() > ATT_FILENAME_MAXBYTES)  // Limit filename size
      name_demangled = name_demangled.substr(0, ATT_FILENAME_MAXBYTES);

    std::string outfilepath = ".";
    if (const char* env = getenv("OUTPUT_PATH")) outfilepath = std::string(env);

    outfilepath.reserve(outfilepath.size() + 128);  // Max filename size
    outfilepath += '/' + name_demangled;
    if (MPI_ENABLE) outfilepath += "_rank" + std::to_string(MPI_RANK);
    outfilepath += "_v";

    // Find if this filename already exists. If so, increment vname.
    int file_iteration = 0;
    while (att_file_exists(outfilepath + std::to_string(file_iteration) + "_kernel.txt"))
      file_iteration += 1;

    outfilepath += std::to_string(file_iteration);
    auto dispatch_id = att_tracer_record->header.id.handle;

    std::string fname = outfilepath + "_kernel.txt";
    std::ofstream(fname.c_str()) << name_demangled << " dispatch[" << dispatch_id << "] GPU["
                                 << att_tracer_record->gpu_id.handle << "]: " << kernel_name_c
                                 << '\n';

    // iterate over each shader engine att trace
    int se_num = att_tracer_record->shader_engine_data_count;
    for (int i = 0; i < se_num; i++) {
      if (!att_tracer_record->shader_engine_data ||
          !att_tracer_record->shader_engine_data[i].buffer_ptr)
        continue;
      printf("--------------collecting data for shader_engine %d---------------\n", i);
      rocprofiler_record_se_att_data_t* se_att_trace = &att_tracer_record->shader_engine_data[i];
      const char* data_buffer_ptr = reinterpret_cast<char*>(se_att_trace->buffer_ptr);

      // dump data in binary format
      std::ofstream out(outfilepath + "_se" + std::to_string(i) + ".att", std::ios::binary);
      if (out.is_open())
        out.write((char*)data_buffer_ptr, se_att_trace->buffer_size);
      else
        std::cerr << "ATT Failed to open file: " << outfilepath << "_se" << i << ".att\n";
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
        case ROCPROFILER_COUNTERS_SAMPLER_RECORD:
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
                                                     uint32_t rocprofiler_minor_version,
                                                     void* data) {
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

ROCPROFILER_EXPORT int rocprofiler_plugin_write_record(rocprofiler_record_tracer_t record) {
  if (!att_plugin || !att_plugin->IsValid()) return -1;
  if (record.header.id.handle == 0) return 0;
  return 0;
}
