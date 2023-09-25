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

#include "code_printing.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <hsa/amd_hsa_elf.h>
#include "../utils.h"
#include <cxxabi.h>
#include <elfutils/libdw.h>
#include <sys/mman.h>

code_object_decoder_t::code_object_decoder_t(const char* codeobj_data, uint64_t codeobj_size) {
  buffer = std::vector<char>{};
  buffer.resize(codeobj_size);
  std::memcpy(buffer.data(), codeobj_data, codeobj_size);

  m_fd = -1;
#if defined(_GNU_SOURCE) && defined(MFD_ALLOW_SEALING) && defined(MFD_CLOEXEC)
  m_fd = ::memfd_create(m_uri.c_str(), MFD_ALLOW_SEALING | MFD_CLOEXEC);
#endif
  if (m_fd == -1) // If fail, attempt under /tmp
    m_fd = ::open("/tmp", O_TMPFILE | O_RDWR, 0666);

  if (m_fd == -1) {
    printf("could not create a temporary file for code object\n");
    return;
  }

  if (size_t size = ::write(m_fd, buffer.data(), buffer.size()); size != buffer.size()) {
    printf("could not write to the temporary file\n");
    return;
  }
  ::lseek(m_fd, 0, SEEK_SET);
  fsync(m_fd);

  m_line_number_map = {};

  std::unique_ptr<Dwarf, void (*)(Dwarf*)> dbg(dwarf_begin(m_fd, DWARF_C_READ),
                                               [](Dwarf* dbg) { dwarf_end(dbg); });

  /*if (!dbg) {
    rocprofiler::warning("Error opening Dwarf!\n");
    return;
  } */

  if (dbg) {
    Dwarf_Off cu_offset{0}, next_offset;
    size_t header_size;

    while (!dwarf_nextcu(dbg.get(), cu_offset, &next_offset, &header_size, nullptr, nullptr,
                         nullptr)) {
      Dwarf_Die die;
      if (!dwarf_offdie(dbg.get(), cu_offset + header_size, &die)) continue;

      Dwarf_Lines* lines;
      size_t line_count;
      if (dwarf_getsrclines(&die, &lines, &line_count)) continue;

      for (size_t i = 0; i < line_count; ++i) {
        Dwarf_Addr addr;
        int line_number;

        if (Dwarf_Line* line = dwarf_onesrcline(lines, i))
          if (!dwarf_lineaddr(line, &addr) && !dwarf_lineno(line, &line_number) && line_number) {
            m_line_number_map.emplace(
                addr, std::make_pair(dwarf_linesrc(line, nullptr, nullptr), line_number));
          }
      }
      cu_offset = next_offset;
    }
    // load_symbol_map();
  }
  disassemble_kernels();
}


code_object_decoder_t::~code_object_decoder_t() {
  if (m_fd) ::close(m_fd);
}

std::optional<code_object_decoder_t::symbol_info_t> code_object_decoder_t::find_symbol(
    uint64_t address) {
  /* Load the symbol table.  */
  if (auto it = m_symbol_map.upper_bound(address); it != m_symbol_map.begin()) {
    if (auto&& [symbol_value, symbol] = *std::prev(it); address < (symbol_value + symbol.second)) {
      std::string symbol_name = symbol.first;

      if (int status; auto* demangled_name =
                          abi::__cxa_demangle(symbol_name.c_str(), nullptr, nullptr, &status)) {
        symbol_name = demangled_name;
        free(demangled_name);
      }
      return symbol_info_t{std::move(symbol_name), symbol_value, symbol.second};
    }
  }
  return {};
}

void code_object_decoder_t::disassemble_kernel(uint64_t addr) {
  auto symbol = find_symbol(addr);

  if (!symbol) {
    std::cerr << "No symbol found at address 0x" << std::hex << addr << std::endl;
    return;
  }

  std::cout << "Dumping ISA for " << symbol->m_name << std::endl;

  uint64_t end_addr = addr + symbol->m_size;
  while (addr < end_addr) {
    char* cpp_line = nullptr;
    auto it = m_line_number_map.find(addr);
    if (it != m_line_number_map.end()) {
      const std::string& file_name = it->second.first;
      size_t line_number = it->second.second;

      std::string cpp = file_name + ':' + std::to_string(line_number);
      cpp_line = (char*)calloc(cpp.size() + 4, sizeof(char));
      std::memcpy(cpp_line, cpp.data(), cpp.size() * sizeof(char));
    }

    size_t size = disassembly->ReadInstruction(addr, cpp_line);
    addr += size;
  }
}

void code_object_decoder_t::disassemble_kernels() {
  disassembly = std::make_unique<DisassemblyInstance>(*this);
  m_symbol_map = disassembly->GetKernelMap();

  for (auto& [k, v] : m_symbol_map) disassemble_kernel(k);
}
