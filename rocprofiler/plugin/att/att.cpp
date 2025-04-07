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
#include <regex>

#include "rocprofiler.h"
#include "rocprofiler_plugin.h"
#include "../utils.h"
#include "../../src/core/session/att/att_header.h"

#include "src/utils/filesystem.hpp"

#define ATT_FILENAME_MAXBYTES 90
#define TEST_INVALID_KERNEL size_t(-1)

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

class att_plugin_t {
 public:
  att_plugin_t(void* data) {
    std::vector<const char*> mpivars = {"MPI_RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK"};

    for (const char* envvar : mpivars)
    if (const char* env = getenv(envvar)) {
      MPI_RANK = atoi(env);
      MPI_ENABLE = true;
      break;
    }

    header.raw = reinterpret_cast<uint64_t>(data);
    header.reserved = 0x11;
  }

  bool MPI_ENABLE = false;
  int MPI_RANK = 0;
  static std::mutex writing_lock;
  bool is_valid_{true};
  rocprofiler::att_header_packet_t header{.raw = 0};
  std::string output_dir = ".";

  bool CheckAddrMatches(uint64_t kernel_addr, uint64_t base_address, uint64_t size)
  {
    return (kernel_addr >= base_address) && (kernel_addr < base_address + size);
  }

  void InitOutputDir()
  {
    static bool bIsInit = false;
    if (bIsInit) return;
    bIsInit = true;

    if (const char* env = getenv("OUTPUT_PATH")) output_dir = std::string(env);
    env_var_search(output_dir);

    if (!output_dir.size()) return;

    try {
        rocprofiler::common::filesystem::create_directories(output_dir);
    } catch (...) {}
    output_dir += '/';
  }

  inline bool att_file_exists(const std::string& name) {
    struct stat buffer;
    return stat(name.c_str(), &buffer) == 0;
  }

  bool IsValid() const { return is_valid_; }

  int FlushATTRecord(const rocprofiler_record_att_tracer_t* att_tracer_record,
                     rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t buffer_id) {

    if (!att_tracer_record) return ROCPROFILER_STATUS_ERROR;
    InitOutputDir();

    std::string kernel_name_mangled;
    // Found problem with rocprofiler API for invalid kernel_ids;
    if (att_tracer_record->kernel_id.handle != TEST_INVALID_KERNEL) {
      size_t name_length;
      CHECK_ROCPROFILER(rocprofiler_query_kernel_info_size(
          ROCPROFILER_KERNEL_NAME, att_tracer_record->kernel_id, &name_length));
      const char* kernel_name_c = nullptr;
      CHECK_ROCPROFILER(rocprofiler_query_kernel_info(
          ROCPROFILER_KERNEL_NAME, att_tracer_record->kernel_id, &kernel_name_c));

      assert(kernel_name_c && "Rocprofv2 returned an invalid kernel name");

      kernel_name_mangled = std::string(kernel_name_c);
      free(const_cast<char*>(kernel_name_c));
    } else {  // Temporary. Adding a valid string.
      kernel_name_mangled = "test_kernel";
    }


    std::string name_demangled =
        rocprofiler::truncate_name(rocprofiler::cxx_demangle(kernel_name_mangled));

    if (name_demangled.size() > ATT_FILENAME_MAXBYTES)  // Limit filename size
      name_demangled = name_demangled.substr(0, ATT_FILENAME_MAXBYTES);

    std::string outfilepath = output_dir + '/' + name_demangled;
    outfilepath.reserve(output_dir.size() + 128);  // Max filename size
    if (MPI_ENABLE) outfilepath += "_rank" + std::to_string(MPI_RANK);
    outfilepath += "_v";

    // Find if this filename already exists. If so, increment vname.
    int file_iteration = 0;
    while (att_file_exists(outfilepath + std::to_string(file_iteration) + "_kernel.txt"))
      file_iteration += 1;

    outfilepath += std::to_string(file_iteration);
    auto writer_id = att_tracer_record->writer_id;

    std::string fname = outfilepath + "_kernel.txt";
    std::ofstream kernel_txt_file((outfilepath + "_kernel.txt").c_str());
    kernel_txt_file << name_demangled << " dispatch[" << writer_id << "] GPU["
                    << att_tracer_record->gpu_id.handle << "]: " << kernel_name_mangled
                    << '\n';

    // iterate over each shader engine att trace
    header.navi = !(att_tracer_record->intercept_list.userdata & 0x1);
    int se_num = att_tracer_record->shader_engine_data_count;
    for (int i = 0; i < se_num; i++)
    {
      if (!att_tracer_record->shader_engine_data ||
          !att_tracer_record->shader_engine_data[i].buffer_ptr)
        continue;
      printf("--------------collecting data for shader_engine %d---------------\n", i);
      header.SEID = i;
      rocprofiler_record_se_att_data_t* se_att_trace = &att_tracer_record->shader_engine_data[i];
      char* data_buffer_ptr = reinterpret_cast<char*>(se_att_trace->buffer_ptr);

      // dump data in binary format
      std::ofstream out(outfilepath + "_se" + std::to_string(i) + ".att", std::ios::binary);
      if (!out.is_open()) {
        std::cerr << "ATT Failed to open file: " << outfilepath << "_se" << i << ".att\n";
        return ROCPROFILER_STATUS_ERROR;
      }
      if (header.enable && !header.navi)
        out.write((const char*)&header, sizeof(header.raw));
      out.write(data_buffer_ptr, se_att_trace->buffer_size);
    }

    for (size_t i = 0; i < att_tracer_record->intercept_list.count; i++)
    {
      const auto& symbol = att_tracer_record->intercept_list.symbols[i];
      if (!symbol.filepath) continue;

      std::string sfilepath(symbol.filepath);
      bool bCopiedData = symbol.data && symbol.data_size;

      if (bCopiedData)
      {
        auto pos = sfilepath.find("://");
        auto rpos = sfilepath.rfind('/');

        if (pos == std::string::npos || pos+3 >= sfilepath.size()) continue;

        std::string type(sfilepath.begin(), sfilepath.begin()+pos);
        std::string cut(sfilepath.begin()+rpos+1, sfilepath.end());
        sfilepath = type + cut + ".out";
      }

      kernel_txt_file << std::hex << "0x" << symbol.base_address << " 0x" << symbol.mem_size
                      << ' ' << std::dec << symbol.att_marker_id << ' ' << sfilepath << '\n';

      sfilepath = output_dir + '/' + sfilepath;
      if (!bCopiedData || att_file_exists(sfilepath)) continue;

      std::ofstream isafile(sfilepath, std::ios::binary);
      if (!isafile.is_open()) {
        std::cerr << "Could not open file: " << sfilepath << std::endl;
        return ROCPROFILER_STATUS_ERROR;
      }

      isafile.write(symbol.data, symbol.data_size);
    }

    return ROCPROFILER_STATUS_SUCCESS;
  }

  int WriteBufferRecords(const rocprofiler_record_header_t* begin,
                         const rocprofiler_record_header_t* end,
                         rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t buffer_id) {
    while (begin < end) {
      if (!begin) return ROCPROFILER_STATUS_ERROR;
      switch (begin->kind) {
        case ROCPROFILER_PROFILER_RECORD:
        case ROCPROFILER_TRACER_RECORD:
        case ROCPROFILER_PC_SAMPLING_RECORD:
        case ROCPROFILER_SPM_RECORD:
        case ROCPROFILER_COUNTERS_SAMPLER_RECORD:
          rocprofiler::warning("Invalid record Kind: %d\n", begin->kind);
          break;

        case ROCPROFILER_ATT_TRACER_RECORD: {
          rocprofiler_record_att_tracer_t* att_record =
              const_cast<rocprofiler_record_att_tracer_t*>(
                  reinterpret_cast<const rocprofiler_record_att_tracer_t*>(begin));
          FlushATTRecord(att_record, session_id, buffer_id);
          break;
        }
      }
      int status = rocprofiler_next_record(begin, &begin, session_id, buffer_id);
      if (status != ROCPROFILER_STATUS_SUCCESS) return status;
    }

    return ROCPROFILER_STATUS_SUCCESS;
  }

 private:
};

att_plugin_t* att_plugin = nullptr;
std::mutex att_plugin_t::writing_lock;

ROCPROFILER_EXPORT int rocprofiler_plugin_initialize(uint32_t rocprofiler_major_version,
                                                     uint32_t rocprofiler_minor_version,
                                                     void* data) {
  if (rocprofiler_major_version != ROCPROFILER_VERSION_MAJOR ||
      rocprofiler_minor_version < ROCPROFILER_VERSION_MINOR)
    return ROCPROFILER_STATUS_ERROR;

  std::lock_guard<std::mutex> lock(att_plugin_t::writing_lock);
  if (att_plugin != nullptr) return ROCPROFILER_STATUS_ERROR;

  att_plugin = new att_plugin_t(data);
  if (att_plugin->IsValid()) return ROCPROFILER_STATUS_SUCCESS;

  // The plugin failed to initialied, destroy it and return an error.
  delete att_plugin;
  att_plugin = nullptr;
  return ROCPROFILER_STATUS_ERROR;
}

ROCPROFILER_EXPORT void rocprofiler_plugin_finalize() {
  std::lock_guard<std::mutex> lock(att_plugin_t::writing_lock);
  if (!att_plugin) return;
  delete att_plugin;
  att_plugin = nullptr;
}

ROCPROFILER_EXPORT int rocprofiler_plugin_write_buffer_records(
    const rocprofiler_record_header_t* begin, const rocprofiler_record_header_t* end,
    rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t buffer_id) {
  std::lock_guard<std::mutex> lock(att_plugin_t::writing_lock);
  if (!att_plugin || !att_plugin->IsValid()) return ROCPROFILER_STATUS_ERROR;
  return att_plugin->WriteBufferRecords(begin, end, session_id, buffer_id);
}

ROCPROFILER_EXPORT int rocprofiler_plugin_write_record(rocprofiler_record_tracer_t record) {
  std::lock_guard<std::mutex> lock(att_plugin_t::writing_lock);
  if (!att_plugin || !att_plugin->IsValid()) return ROCPROFILER_STATUS_ERROR;
  if (record.header.id.handle == 0) return ROCPROFILER_STATUS_SUCCESS;
  return ROCPROFILER_STATUS_SUCCESS;
}
