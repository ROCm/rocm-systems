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

#include <atomic>

#define C_API_BEGIN try {

#define C_API_END(returndata)                       \
} catch (std::exception& e)                         \
{                                                   \
  std::cerr << "Error: " << e.what() << std::endl;  \
  return returndata;                                \
}                                                   \
catch (std::string& s)                              \
{                                                   \
  std::cerr << "Error: " << s << std::endl;         \
  return returndata;                                \
}                                                   \
catch (...)                                         \
{                                                   \
  return returndata;                                \
}

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
        Dwarf_Line* line = dwarf_onesrcline(lines, i);
        if (!line) continue;

        if (!dwarf_lineaddr(line, &addr) && !dwarf_lineno(line, &line_number) && line_number)
          m_line_number_map[addr] = {dwarf_linesrc(line, nullptr, nullptr), line_number};
      }
      cu_offset = next_offset;
    }
    // load_symbol_map();
  }

  try {
    disassembly = std::make_unique<DisassemblyInstance>(*this); // Can throw
  } catch(std::exception& e) {
    return;
  }
  try {
    m_symbol_map = disassembly->GetKernelMap(); // Can throw
  } catch(std::exception& e) {
    return;
  }

  //disassemble_kernels();
}


code_object_decoder_t::~code_object_decoder_t() {
  if (m_fd) ::close(m_fd);
}

std::optional<SymbolInfo> code_object_decoder_t::find_symbol(uint64_t vaddr) {
  /* Load the symbol table.  */
  auto it = m_symbol_map.upper_bound(vaddr);
  if (it == m_symbol_map.begin())
    return std::nullopt;

  auto&& [symbol_vaddr, symbol] = *std::prev(it);
  if (vaddr >= symbol_vaddr + symbol.mem_size)
    return std::nullopt;

  std::string symbol_name = symbol.name;

  int status = 0;
  auto* demangled_name = abi::__cxa_demangle(symbol_name.c_str(), nullptr, nullptr, &status);
  if (status == 0 && demangled_name)
  {
    symbol_name = demangled_name;
    free(demangled_name);
  }
  return SymbolInfo{symbol_name, symbol.faddr, symbol.mem_size};
}

std::pair<instruction_instance_t, size_t>
code_object_decoder_t::disassemble_instruction(uint64_t faddr, uint64_t vaddr)
{
  if (!disassembly)
    throw std::exception();

  char* cpp_line = nullptr;
  auto it = m_line_number_map.find(vaddr);
  if (it != m_line_number_map.end()) {
    const std::string& file_name = it->second.first;
    size_t line_number = it->second.second;

    std::string cpp = file_name + ':' + std::to_string(line_number);
    cpp_line = (char*)calloc(cpp.size() + 4, sizeof(char));
    std::memcpy(cpp_line, cpp.data(), cpp.size() * sizeof(char));
  }
  size_t size = disassembly->ReadInstruction(faddr, vaddr, cpp_line);
  return {disassembly->last_instruction, size};
}

void code_object_decoder_t::disassemble_kernel(uint64_t faddr, uint64_t vaddr)
{
  if (!disassembly) return;
  auto symbol = find_symbol(vaddr);

  if (!symbol)
  {
    std::cerr << "No symbol found at address 0x" << std::hex << faddr << std::endl;
    return;
  }

  std::cout << "Dumping ISA for " << symbol->name << std::endl;

  uint64_t end_addr = faddr + symbol->mem_size;
  while (faddr < end_addr)
  {
    size_t size;
    instruction_instance_t inst;
    std::tie(inst, size) = this->disassemble_instruction(faddr, vaddr);
    instructions.push_back(inst);
    faddr += size;
    vaddr += size;
  }
}

void code_object_decoder_t::disassemble_kernels() {
  for (auto& [vaddr, v] : m_symbol_map) disassemble_kernel(v.faddr, vaddr);
}

void code_object_decoder_t::disassemble_single_kernel(uint64_t kaddr) {
  for (auto& [vaddr, v] : m_symbol_map)
    if (kaddr >= vaddr && kaddr < vaddr + v.mem_size)
      disassemble_kernel(v.faddr, vaddr);
}

CodeobjService::CodeobjService(const char* filepath, uint64_t load_base): load_base(load_base)
{
  if (!filepath) throw "Empty filepath.";

  std::string_view fpath(filepath);

  if (fpath.rfind(".out") + 4 == fpath.size())
  {
    std::ifstream file(filepath, std::ios::in | std::ios::binary);

    if (!file.is_open())
      throw "Invalid filename " + std::string(filepath);

    std::vector<char> buffer;
    file.seekg(0, file.end);
    buffer.resize(file.tellg());
    file.seekg(0, file.beg);
    file.read(buffer.data(), buffer.size());

    decoder = std::make_unique<code_object_decoder_t>(buffer.data(), buffer.size());
  }
  else
  {
    std::unique_ptr<CodeObjectBinary> binary = std::make_unique<CodeObjectBinary>(filepath);
    decoder = std::make_unique<code_object_decoder_t>(binary->buffer.data(), binary->buffer.size());
  }
}

bool CodeobjService::decode_single(uint64_t vaddr, uint64_t faddr)
{
  if (!decoder->disassembly) return false;

  try
  {
    decoded_map[vaddr] = decoder->disassemble_instruction(faddr, vaddr-load_base);
  }
  catch(std::exception& e)
  {
    return false;
  }
  return true;
}

std::pair<instruction_instance_t, size_t>& CodeobjService::getDecoded(uint64_t addr)
{
  if (decoded_map.find(addr) != decoded_map.end())
    return decoded_map[addr];

  std::optional<uint64_t> faddr{};

  if (!bNotElfFILE)
  {
    faddr = DisassemblyInstance::va2fo(decoder->buffer.data(), addr-load_base);
    if (!faddr)
      bNotElfFILE = true;
  }

  if (bNotElfFILE && decoder->buffer.size() > 0x100) {
    uint64_t f_offset = *reinterpret_cast<uint32_t*>(decoder->buffer.data()+0xb8);
    uint64_t v_offset = *reinterpret_cast<uint32_t*>(decoder->buffer.data()+0xc8);

    faddr = addr+f_offset-load_base-v_offset;
  }

  if (!faddr || !decode_single(addr, *faddr))
  {
    std::cerr << "Invalid addr: " << std::hex << addr << std::dec << std::endl;
    throw std::exception();
  }

  return decoded_map[addr];
}

std::unordered_map<uint64_t, std::unique_ptr<CodeobjService>> services{};
std::atomic<uint64_t> shandles{1};

#define PUBLIC_API __attribute__((visibility("default")))

extern "C"
{
  PUBLIC_API uint64_t createService(const char* filename, uint64_t load_base)
  {
    C_API_BEGIN

    uint64_t handle = shandles.fetch_add(1);
    services[handle] = std::make_unique<CodeobjService>(filename, load_base);
    return handle;

    C_API_END(0)
  }
  PUBLIC_API int deleteService(uint64_t handle)
  {
    return services.erase(handle);
  }
  PUBLIC_API const char* getInstruction(uint64_t handle, uint64_t addr)
  {
    C_API_BEGIN

    return services.at(handle)->getInstruction(addr);

    C_API_END(nullptr)
  }
  PUBLIC_API const char* getCppref(uint64_t handle, uint64_t addr)
  {
    C_API_BEGIN
    
    return services.at(handle)->getCppref(addr);

    C_API_END(nullptr)
  }
  PUBLIC_API size_t getInstSize(uint64_t handle, uint64_t addr)
  {
    C_API_BEGIN

    return services.at(handle)->getSize(addr);

    C_API_END(0)
  }
  PUBLIC_API const char* getSymbolName(uint64_t addr)
  {
    C_API_BEGIN
  
    for (auto& [handle, service] : services)
    {
      if (!service->inrange(addr)) continue;
      return service->getSymbolName(addr);
    }
    return nullptr;

    C_API_END(nullptr)
  }
}