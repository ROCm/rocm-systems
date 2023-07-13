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

#ifndef SAMPLES_PCSAMPLER_CODE_PRINTING_SAMPLE_CODE_PRINTING_HPP_
#define SAMPLES_PCSAMPLER_CODE_PRINTING_SAMPLE_CODE_PRINTING_HPP_

#include <map>
#include <optional>
#include <string>
#include <vector>

#include <amd-dbgapi/amd-dbgapi.h>

namespace amd::debug_agent {

class code_object_t {
  struct symbol_info_t {
    const std::string m_name;
    amd_dbgapi_global_address_t m_value;
    amd_dbgapi_size_t m_size;
  };

  using symbol_map_t = std::optional<
      std::map<amd_dbgapi_global_address_t, std::pair<std::string, amd_dbgapi_size_t>>>;

 public:
  void load_symbol_map();
  void load_debug_info();

  std::optional<symbol_info_t> find_symbol(amd_dbgapi_global_address_t address);

  code_object_t(amd_dbgapi_code_object_id_t code_object_id);
  code_object_t(code_object_t&& rhs);

  ~code_object_t();

  void open();
  bool is_open() const { return m_fd.has_value(); }

  amd_dbgapi_global_address_t load_address() const { return m_load_address; }
  amd_dbgapi_size_t mem_size() const { return m_mem_size; }
  // FIXME(?): extra function not in rocr-debug-agent
  uint32_t elf_amdgpu_machine() const { return m_elf_amdgpu_machine; }

  void disassemble_around(amd_dbgapi_architecture_id_t architecture_id,
                          amd_dbgapi_global_address_t pc);

  void disassemble_kernel(amd_dbgapi_architecture_id_t architecture_id,
                          amd_dbgapi_global_address_t start_addr, bool const print_src = false);

  bool save(const std::string& directory) const;

  amd_dbgapi_global_address_t m_load_address{0};
  amd_dbgapi_size_t m_mem_size{0};
  std::optional<int> m_fd;

  std::optional<std::map<amd_dbgapi_global_address_t, std::pair<std::string, size_t>>>
      m_line_number_map;

  std::optional<std::map<amd_dbgapi_global_address_t, amd_dbgapi_global_address_t>> m_pc_ranges_map;

  symbol_map_t m_symbol_map;
  std::string m_uri;
  amd_dbgapi_code_object_id_t const m_code_object_id;
  // FIXME(?): extra field not in rocr-debug-agent
  uint32_t m_elf_amdgpu_machine{0};
};

}  // namespace amd::debug_agent

enum struct disassembly_mode { AROUND, KERNEL };

std::tuple<amd_dbgapi_process_id_t,
           std::map<amd_dbgapi_global_address_t, amd::debug_agent::code_object_t>>
init_disassembly();

void disassemble(
    disassembly_mode const mode, amd_dbgapi_process_id_t const process_id,
    std::map<amd_dbgapi_global_address_t, amd::debug_agent::code_object_t>& code_object_map,
    uint64_t const addr);

void print_pc_context(
    amd_dbgapi_process_id_t const process_id,
    std::map<amd_dbgapi_global_address_t, amd::debug_agent::code_object_t>& code_object_map,
    amd_dbgapi_global_address_t const pc);

#endif  // SAMPLES_PCSAMPLER_CODE_PRINTING_SAMPLE_CODE_PRINTING_HPP_
