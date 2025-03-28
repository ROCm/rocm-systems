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

#define C_API_END(returndata)                                         \
} catch (std::exception& e)                                           \
{                                                                     \
  std::string s = e.what();                                           \
  if (s.find("memory protocol not supported!") == std::string::npos)  \
    std::cerr << "Codeobj API lookup: " << e.what() << std::endl;     \
  return returndata;                                                  \
}                                                                     \
catch (std::string& s)                                                \
{                                                                     \
  if (s.find("memory protocol not supported!") == std::string::npos)  \
    std::cerr << "Codeobj API lookup: " << s << std::endl;            \
  return returndata;                                                  \
}                                                                     \
catch (...)                                                           \
{                                                                     \
  return returndata;                                                  \
}

CodeObjDecoderComponent::CodeObjDecoderComponent(
  const char* codeobj_data,
  uint64_t codeobj_size,
  uint64_t gpu_id
) {
  if (
      codeobj_size <= 4 ||
      codeobj_data[0] != ELFMAG0 ||
      codeobj_data[1] != ELFMAG1 ||
      codeobj_data[2] != ELFMAG2 ||
      codeobj_data[3] != ELFMAG3
  )
    throw std::invalid_argument("Invalid ELF file");

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

  if (size_t size = ::write(m_fd, codeobj_data, codeobj_size); size != codeobj_size) {
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

    std::map<uint64_t, std::string> used_addrs;

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

        if (line && !dwarf_lineaddr(line, &addr) && !dwarf_lineno(line, &line_number) && line_number)
        {
          std::string src = dwarf_linesrc(line, nullptr, nullptr);
          auto dwarf_line = src + ':' + std::to_string(line_number);

          if (used_addrs.find(addr) != used_addrs.end())
          {
            used_addrs.at(addr) += ' ' + dwarf_line;
            continue;
          }

          used_addrs.emplace(addr, std::move(dwarf_line));
        }
      }
      cu_offset = next_offset;
    }

    auto it = used_addrs.begin();
    if(it != used_addrs.end())
    {
      while(std::next(it) != used_addrs.end())
      {
        uint64_t delta   = std::next(it)->first - it->first;
        auto     segment = address_range_t{it->first, delta, 0};
        m_line_number_map.emplace(segment, std::move(it->second));
        it++;
      }
      auto segment = address_range_t{it->first, codeobj_size - it->first, 0};
      m_line_number_map.emplace(segment, std::move(it->second));
    }
  }

  // Can throw
  disassembly = std::make_unique<DisassemblyInstance>(codeobj_data, codeobj_size, gpu_id);
  try {
    m_symbol_map = disassembly->GetKernelMap(); // Can throw
  } catch(...) {}

  //disassemble_kernels();
}

CodeObjDecoderComponent::~CodeObjDecoderComponent() {
  if (m_fd) ::close(m_fd);
}

std::optional<SymbolInfo> CodeObjDecoderComponent::find_symbol(uint64_t vaddr) {
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
CodeObjDecoderComponent::disassemble_instruction(uint64_t faddr, uint64_t vaddr)
{
  if (!disassembly)
    throw std::exception();

  const char* cpp_line = nullptr;

  auto it = m_line_number_map.find({vaddr, 0, 0});
  if(it != m_line_number_map.end()) cpp_line = it->second.data();

  size_t size = disassembly->ReadInstruction(faddr, vaddr, cpp_line);
  return {disassembly->last_instruction, size};
}

void CodeObjDecoderComponent::disassemble_kernel(uint64_t faddr, uint64_t vaddr)
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

void CodeObjDecoderComponent::disassemble_kernels() {
  for (auto& [vaddr, v] : m_symbol_map) disassemble_kernel(v.faddr, vaddr);
}

void CodeObjDecoderComponent::disassemble_single_kernel(uint64_t kaddr) {
  for (auto& [vaddr, v] : m_symbol_map)
    if (kaddr >= vaddr && kaddr < vaddr + v.mem_size)
      disassemble_kernel(v.faddr, vaddr);
}

CodeobjDecoder::CodeobjDecoder(
  const char* filepath,
  uint64_t loadbase,
  uint64_t mem_size,
  uint64_t gpu_id
): loadbase(loadbase), load_end(loadbase + mem_size)
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

    decoder = std::make_unique<CodeObjDecoderComponent>(buffer.data(), buffer.size(), gpu_id);
  }
  else
  {
    std::unique_ptr<CodeObjectBinary> binary = std::make_unique<CodeObjectBinary>(filepath);
    auto& buffer = binary->buffer;
    decoder = std::make_unique<CodeObjDecoderComponent>(buffer.data(), buffer.size(), gpu_id);
  }

  auto elf_segments = decoder->disassembly->getSegments();
}

bool CodeobjDecoder::add_to_map(uint64_t faddr, uint64_t vaddr, uint64_t voffset)
{
  try
  {
    decoded_map[vaddr] = decoder->disassemble_instruction(faddr, voffset);
  }
  catch(std::exception& e)
  {
    return false;
  }
  return true;
}

bool CodeobjDecoder::decode_single_at_offset(uint64_t vaddr, uint64_t voffset)
{
  auto faddr = decoder->disassembly->va2fo(voffset);
  if (!faddr)
    return false;

  return add_to_map(*faddr, vaddr, voffset);
}

bool CodeobjDecoder::decode_single(uint64_t vaddr)
{
  if (!decoder || vaddr < loadbase) return false;
  return decode_single_at_offset(vaddr, vaddr-loadbase);
}

std::pair<instruction_instance_t, size_t>& CodeobjDecoder::getDecoded(uint64_t addr)
{
  if (decoded_map.find(addr) != decoded_map.end())
    return decoded_map[addr];

  if (!decode_single(addr))
  {
    std::cerr << "Invalid addr: " << std::hex << addr << std::dec << std::endl;
    throw std::exception();
  }

  return decoded_map[addr];
}

#define PUBLIC_API __attribute__((visibility("default")))

CodeobjTableTranslation table;

extern "C"
{
  PUBLIC_API int addDecoder(
    const char* filename,
    uint32_t id,
    uint64_t loadbase,
    uint64_t memsize,
    uint64_t gpu_id
  ) {
    C_API_BEGIN

    table.addDecoder(filename, id, loadbase, memsize, gpu_id);
    return 0;

    C_API_END(1)
  }
  PUBLIC_API int removeDecoder(uint32_t id, uint64_t loadbase)
  {
    return table.removeDecoder(id, loadbase) != false;
  }
  PUBLIC_API instruction_info_t getInstructionFromAddr(uint64_t vaddr)
  {
    static instruction_info_t default_info{nullptr, nullptr, 0};
    C_API_BEGIN

    return table.get(vaddr);

    C_API_END(default_info)
  }
  PUBLIC_API instruction_info_t getInstructionFromID(uint32_t id, uint64_t offset)
  {
    static instruction_info_t default_info{nullptr, nullptr, 0};
    C_API_BEGIN

    return table.get(id, offset);

    C_API_END(default_info)
  }
  PUBLIC_API const char* getSymbolName(uint64_t addr)
  {
    C_API_BEGIN
  
    return table.getSymbolName(addr);

    C_API_END(nullptr)
  }
}
