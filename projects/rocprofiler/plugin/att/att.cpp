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
#include "code_printing.hpp"
#include "../../src/core/session/att/att.h"

#define ATT_FILENAME_MAXBYTES 90
#define TEST_INVALID_KERNEL size_t(-1)

namespace {
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

    isa_mode = static_cast<decltype(isa_mode)>(header.isadumpmode);
    header.isadumpmode = 0;
  }

  bool MPI_ENABLE = false;
  int MPI_RANK = 0;
  std::mutex writing_lock;
  bool is_valid_{true};
  rocprofiler::att_header_packet_t header{.raw = 0};
  rocprofiler::rocprofiler_att_isa_dump_mode isa_mode = rocprofiler::ISA_MODE_DUMP_ALL;

  bool CheckAddrMatches(uint64_t kernel_addr, uint64_t base_address, uint64_t size)
  {
    if (isa_mode == rocprofiler::ISA_MODE_DUMP_ALL)
      return true;
    return (kernel_addr >= base_address) && (kernel_addr < base_address + size);
  }

  inline bool att_file_exists(const std::string& name) {
    struct stat buffer;
    return stat(name.c_str(), &buffer) == 0;
  }

  bool IsValid() const { return is_valid_; }

  int FlushATTRecord(const rocprofiler_record_att_tracer_t* att_tracer_record,
                     rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t buffer_id) {
    std::lock_guard<std::mutex> lock(writing_lock);

    if (!att_tracer_record) return ROCPROFILER_STATUS_ERROR;

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
    auto writer_id = att_tracer_record->writer_id;

    std::string fname = outfilepath + "_kernel.txt";
    std::ofstream(fname.c_str()) << name_demangled << " dispatch[" << writer_id << "] GPU["
                                 << att_tracer_record->gpu_id.handle << "]: " << kernel_name_mangled
                                 << '\n';

    // iterate over each shader engine att trace
    header.navi = !att_tracer_record->intercept_list.userdata & 0x1;
    int se_num = att_tracer_record->shader_engine_data_count;
    for (int i = 0; i < se_num; i++) {
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
      if (header.enable)
        out.write((const char*)&header, sizeof(header.raw));
      out.write(data_buffer_ptr, se_att_trace->buffer_size);
    }

    if (isa_mode == rocprofiler::ISA_MODE_DUMP_NONE)
      return 0;

    uint64_t kernel_addr = att_tracer_record->intercept_list.userdata >> 1;

    std::ofstream isafile(outfilepath + "_isa.s");
    if (!isafile.is_open()) {
      std::cerr << "Could not open ISA file: " << outfilepath << "_isa.s" << std::endl;
      return ROCPROFILER_STATUS_ERROR;
    }
    isafile << "<Kernel> " << kernel_name_mangled << '\n';

    for (size_t i = 0; i < att_tracer_record->intercept_list.count; i++) {
      const rocprofiler_intercepted_codeobj_t& symbol =
          att_tracer_record->intercept_list.symbols[i];

      if (!CheckAddrMatches(kernel_addr, symbol.base_address, symbol.mem_size)) continue;

      std::unique_ptr<CodeObjectBinary> binary;
      std::unique_ptr<code_object_decoder_t> decoder;

      if (symbol.data && symbol.data_size) {
        decoder = std::make_unique<code_object_decoder_t>(symbol.data, symbol.data_size);
      } else if (std::string(symbol.filepath).find("file://") != std::string::npos) {
        binary = std::make_unique<CodeObjectBinary>(symbol.filepath);
        decoder =
            std::make_unique<code_object_decoder_t>(binary->buffer.data(), binary->buffer.size());
      } else {
        continue;
      }

      if (isa_mode == rocprofiler::ISA_MODE_DUMP_KERNEL)
        decoder->disassemble_single_kernel(kernel_addr-symbol.base_address);
      else
        decoder->disassemble_kernels();

      for (auto& instance : decoder->instructions) {
        uint64_t addr = instance.address + symbol.base_address;

        if (decoder->m_symbol_map.find(instance.address) != decoder->m_symbol_map.end())
          isafile << "; Begin " << decoder->m_symbol_map[instance.address].name << '\n';
        if (instance.cpp_reference) isafile << "; " << instance.cpp_reference << '\n';
        isafile << instance.instruction << " // " << std::hex << addr << '\n';
      }
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

}  // namespace

ROCPROFILER_EXPORT int rocprofiler_plugin_initialize(uint32_t rocprofiler_major_version,
                                                     uint32_t rocprofiler_minor_version,
                                                     void* data) {
  if (rocprofiler_major_version != ROCPROFILER_VERSION_MAJOR ||
      rocprofiler_minor_version < ROCPROFILER_VERSION_MINOR)
    return ROCPROFILER_STATUS_ERROR;

  if (att_plugin != nullptr) return ROCPROFILER_STATUS_ERROR;

  att_plugin = new att_plugin_t(data);
  if (att_plugin->IsValid()) return ROCPROFILER_STATUS_SUCCESS;

  // The plugin failed to initialied, destroy it and return an error.
  delete att_plugin;
  att_plugin = nullptr;
  return ROCPROFILER_STATUS_ERROR;
}

ROCPROFILER_EXPORT void rocprofiler_plugin_finalize() {
  if (!att_plugin) return;
  delete att_plugin;
  att_plugin = nullptr;
}

ROCPROFILER_EXPORT int rocprofiler_plugin_write_buffer_records(
    const rocprofiler_record_header_t* begin, const rocprofiler_record_header_t* end,
    rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t buffer_id) {
  if (!att_plugin || !att_plugin->IsValid()) return ROCPROFILER_STATUS_ERROR;
  return att_plugin->WriteBufferRecords(begin, end, session_id, buffer_id);
}

ROCPROFILER_EXPORT int rocprofiler_plugin_write_record(rocprofiler_record_tracer_t record) {
  if (!att_plugin || !att_plugin->IsValid()) return ROCPROFILER_STATUS_ERROR;
  if (record.header.id.handle == 0) return ROCPROFILER_STATUS_SUCCESS;
  return ROCPROFILER_STATUS_SUCCESS;
}
