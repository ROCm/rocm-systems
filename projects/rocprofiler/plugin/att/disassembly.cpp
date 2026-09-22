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

#if !defined(_GNU_SOURCE) || !defined(_XOPEN_SOURCE)
#define _XOPEN_SOURCE 700
#endif

#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <elf.h>
#include <cxxabi.h>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <elfutils/libdw.h>
#include "../utils.h"
#include "code_printing.hpp"
#include <hsa/amd_hsa_elf.h>

#define THROW_COMGR(call)                                                                          \
if (amd_comgr_status_s status = call) {                                                            \
  const char* reason = "";                                                                         \
  amd_comgr_status_string(status, &reason);                                                        \
  std::cerr << __FILE__ << ':' << __LINE__ << " code: "                                            \
            << status << " failed: " << reason << std::endl;                                       \
  throw std::exception();                                                                          \
}

#define RETURN_COMGR(call)                                                                         \
if (amd_comgr_status_s status = call) {                                                            \
  const char* reason = "";                                                                         \
  amd_comgr_status_string(status, &reason);                                                        \
  std::cerr << __FILE__ << ':' << __LINE__ << " code: "                                            \
            << status << " failed: " << reason << std::endl;                                       \
  return AMD_COMGR_STATUS_ERROR;                                                                   \
}

std::unordered_map<uint64_t, std::string> DisassemblyInstance::agent_isa_name{};

CodeObjectBinary::CodeObjectBinary(const std::string& uri) : m_uri(uri) {
  const std::string protocol_delim{"://"};

  size_t protocol_end = m_uri.find(protocol_delim);
  std::string protocol = m_uri.substr(0, protocol_end);
  protocol_end += protocol_delim.length();

  std::transform(protocol.begin(), protocol.end(), protocol.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  std::string path;
  size_t path_end = m_uri.find_first_of("#?", protocol_end);
  if (path_end != std::string::npos) {
    path = m_uri.substr(protocol_end, path_end++ - protocol_end);
  } else {
    path = m_uri.substr(protocol_end);
  }

  /* %-decode the string.  */
  std::string decoded_path;
  decoded_path.reserve(path.length());
  for (size_t i = 0; i < path.length(); ++i)
  {
    if (path[i] == '%' && std::isxdigit(path[i + 1]) && std::isxdigit(path[i + 2])) {
      decoded_path += std::stoi(path.substr(i + 1, 2), 0, 16);
      i += 2;
    } else {
      decoded_path += path[i];
    }
  }

  /* Tokenize the query/fragment.  */
  std::vector<std::string> tokens;
  size_t pos, last = path_end;
  while ((pos = m_uri.find('&', last)) != std::string::npos) {
    tokens.emplace_back(m_uri.substr(last, pos - last));
    last = pos + 1;
  }
  if (last != std::string::npos) {
    tokens.emplace_back(m_uri.substr(last));
  }

  /* Create a tag-value map from the tokenized query/fragment.  */
  std::unordered_map<std::string, std::string> params;
  std::for_each(tokens.begin(), tokens.end(), [&](std::string& token) {
    size_t delim = token.find('=');
    if (delim != std::string::npos) {
      params.emplace(token.substr(0, delim), token.substr(delim + 1));
    }
  });

  buffer = std::vector<char>{};
  size_t offset{0}, size{0};

  if (auto offset_it = params.find("offset"); offset_it != params.end()) {
    offset = std::stoul(offset_it->second, nullptr, 0);
  }

  if (auto size_it = params.find("size"); size_it != params.end()) {
    if (!(size = std::stoul(size_it->second, nullptr, 0))) return;
  }

  if (protocol != "file") throw protocol + " protocol not supported!";

  std::ifstream file(decoded_path, std::ios::in | std::ios::binary);
  if (!file || !file.is_open()) throw "could not open " +  decoded_path;

  if (!size) {
    file.ignore(std::numeric_limits<std::streamsize>::max());
    size_t bytes = file.gcount();
    file.clear();

    if (bytes < offset)
      throw "invalid uri " + decoded_path + " (file size < offset)";

    size = bytes - offset;
  }

  file.seekg(offset, std::ios_base::beg);
  buffer.resize(size);
  file.read(&buffer[0], size);
}

DisassemblyInstance::DisassemblyInstance(
  const char* codeobj_data,
  uint64_t codeobj_size,
  uint64_t gpu_id
)
{
  if (
      codeobj_size <= 4 ||
      codeobj_data[0] != ELFMAG0 ||
      codeobj_data[1] != ELFMAG1 ||
      codeobj_data[2] != ELFMAG2 ||
      codeobj_data[3] != ELFMAG3
  )
    throw std::invalid_argument("Invalid ELF file");

  buffer = std::vector<char>(codeobj_size, 0);
  std::memcpy(buffer.data(), codeobj_data, codeobj_size);

  THROW_COMGR(amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &data));
  THROW_COMGR(amd_comgr_set_data(data, buffer.size(), buffer.data()));

  std::string input_isa{};
  if (agent_isa_name.find(gpu_id) == agent_isa_name.end())
  {
    size_t isa_size = 128;
    input_isa.resize(isa_size);
    THROW_COMGR(amd_comgr_get_data_isa_name(data, &isa_size, input_isa.data()));
    agent_isa_name[gpu_id] = input_isa;
  }
  else
  {
    input_isa = agent_isa_name.at(gpu_id);
  }

  THROW_COMGR(amd_comgr_create_disassembly_info(
      input_isa.data(),
      &DisassemblyInstance::memory_callback, &DisassemblyInstance::inst_callback,
      [](uint64_t address, void* user_data) {}, &info));

}

static bool IsKernelType(amd_comgr_symbol_type_t type)
{
  if (type == AMD_COMGR_SYMBOL_TYPE_FUNC)
    return true;
#ifdef AMD_COMGR_SYMBOL_TYPE_AMDGPU_HSA_KERNEL // To be deprecated
  if (type == AMD_COMGR_SYMBOL_TYPE_AMDGPU_HSA_KERNEL)
    return true;
#endif
  return false;
}

amd_comgr_status_t DisassemblyInstance::symbol_callback(amd_comgr_symbol_t symbol,
                                                        void* user_data) {
  amd_comgr_symbol_type_t type;
  RETURN_COMGR(amd_comgr_symbol_get_info(symbol, AMD_COMGR_SYMBOL_INFO_TYPE, &type));

  if (!IsKernelType(type))
    return AMD_COMGR_STATUS_SUCCESS;

  uint64_t vaddr;
  uint64_t mem_size;
  uint64_t name_size;
  RETURN_COMGR(amd_comgr_symbol_get_info(symbol, AMD_COMGR_SYMBOL_INFO_VALUE, &vaddr));
  RETURN_COMGR(amd_comgr_symbol_get_info(symbol, AMD_COMGR_SYMBOL_INFO_SIZE, &mem_size));
  RETURN_COMGR(amd_comgr_symbol_get_info(symbol, AMD_COMGR_SYMBOL_INFO_NAME_LENGTH, &name_size));

  std::string name;
  name.resize(name_size);

  RETURN_COMGR(amd_comgr_symbol_get_info(symbol, AMD_COMGR_SYMBOL_INFO_NAME, name.data()));

  DisassemblyInstance& instance = *static_cast<DisassemblyInstance*>(user_data);
  std::optional<uint64_t> faddr = instance.va2fo(vaddr);

  if (faddr)
    instance.symbol_map[vaddr] = {name, *faddr, mem_size};
  return AMD_COMGR_STATUS_SUCCESS;
}

std::map<uint64_t, SymbolInfo>& DisassemblyInstance::GetKernelMap() {
  symbol_map = {};
  THROW_COMGR(amd_comgr_iterate_symbols(data, &DisassemblyInstance::symbol_callback, this));
    
  return symbol_map;
}

DisassemblyInstance::~DisassemblyInstance() {
  amd_comgr_release_data(data);
  amd_comgr_destroy_disassembly_info(info);
}

uint64_t DisassemblyInstance::ReadInstruction(uint64_t faddr, uint64_t vaddr, const char* cpp_line)
{
  uint64_t size_read;
  uint64_t addr_in_buffer = reinterpret_cast<uint64_t>(buffer.data()) + faddr;

  THROW_COMGR(amd_comgr_disassemble_instruction(info, addr_in_buffer, (void*)this, &size_read));

  last_instruction.address = vaddr;
  last_instruction.cpp_reference = cpp_line;
  return size_read;
}

uint64_t DisassemblyInstance::memory_callback(uint64_t from, char* to, uint64_t size,
                                              void* user_data) {
  DisassemblyInstance& instance = *static_cast<DisassemblyInstance*>(user_data);
  int64_t copysize = reinterpret_cast<int64_t>(instance.buffer.data())
                      + instance.buffer.size() - static_cast<int64_t>(from);
  copysize = std::min<int64_t>(size, copysize);
  std::memcpy(to, (char*)from, copysize);
  return copysize;
}

void DisassemblyInstance::inst_callback(const char* instruction, void* user_data) {
  DisassemblyInstance& instance = *static_cast<DisassemblyInstance*>(user_data);
  instance.last_instruction.instruction = strdup(instruction);
}

#define CHECK_VA2FO(x, msg) if (!(x)) {                                \
  std::cerr << __FILE__ << ' ' << __LINE__ << ' ' << msg << std::endl; \
  return std::nullopt;                                                 \
}

// mem - input argument, start of the elf
// va  - input argument, virtual address
// return file offset, if found
std::optional<uint64_t> DisassemblyInstance::va2fo(uint64_t va)
{
  CHECK_VA2FO(buffer.size(), "buffer is not large enough");

  uint8_t *e_ident = (uint8_t*)buffer.data();
  CHECK_VA2FO(e_ident, "e_ident is nullptr");

  CHECK_VA2FO(
    e_ident[EI_MAG0] == ELFMAG0 ||
    e_ident[EI_MAG1] == ELFMAG1 ||
    e_ident[EI_MAG2] == ELFMAG2 ||
    e_ident[EI_MAG3] == ELFMAG3, "unexpected ei_mag");

  CHECK_VA2FO(e_ident[EI_CLASS] == ELFCLASS64, "unexpected ei_class");
  CHECK_VA2FO(e_ident[EI_DATA] == ELFDATA2LSB, "unexpected ei_data");
  CHECK_VA2FO(e_ident[EI_VERSION] == EV_CURRENT, "unexpected ei_version");
  CHECK_VA2FO(e_ident[EI_OSABI] == 64, "unexpected ei_osabi"); // ELFOSABI_AMDGPU_HSA

  CHECK_VA2FO(
    e_ident[EI_ABIVERSION] == 2 || // ELFABIVERSION_AMDGPU_HSA_V4
    e_ident[EI_ABIVERSION] == 3 || // ELFABIVERSION_AMDGPU_HSA_V5
    e_ident[EI_ABIVERSION] == 4, "unexpected ei_abiversion"); // ELFABIVERSION_AMDGPU_HSA_V6

  Elf64_Ehdr *ehdr = (Elf64_Ehdr*)buffer.data();
  CHECK_VA2FO(ehdr, "ehdr is nullptr");
  CHECK_VA2FO(ehdr->e_type == ET_DYN, "unexpected e_type");
  CHECK_VA2FO(ehdr->e_machine == ELF::EM_AMDGPU, "unexpected e_machine");

  CHECK_VA2FO(buffer.size() > sizeof(Elf64_Ehdr), "buffer is not large enough");
  CHECK_VA2FO(ehdr->e_phoff != 0, "unexpected e_phoff");

  CHECK_VA2FO(buffer.size() > ehdr->e_phoff + sizeof(Elf64_Phdr), "buffer is not large enough");

  Elf64_Phdr *phdr = (Elf64_Phdr*)((uint8_t*)buffer.data() + ehdr->e_phoff);
  CHECK_VA2FO(phdr, "phdr is nullptr");

  for (uint16_t i = 0; i < ehdr->e_phnum; ++i)
  {
    if (phdr[i].p_type != PT_LOAD)
      continue;
    if (va < phdr[i].p_vaddr || va >= (phdr[i].p_vaddr + phdr[i].p_memsz))
      continue;

    return va + phdr[i].p_offset - phdr[i].p_vaddr;
  }
  return std::nullopt;
}

#undef CHECK_VA2FO
#define CHECK_VA2FO(x, msg) if (!(x)) {                                \
  std::cerr << __FILE__ << ' ' << __LINE__ << ' ' << msg << std::endl; \
  return {};                                                 \
}

std::vector<std::pair<uint64_t, uint64_t>> DisassemblyInstance::getSegments()
{
  CHECK_VA2FO(buffer.size(), "buffer is not large enough");

  uint8_t *e_ident = (uint8_t*)buffer.data();
  CHECK_VA2FO(e_ident, "e_ident is nullptr");

  CHECK_VA2FO(
    e_ident[EI_MAG0] == ELFMAG0 ||
    e_ident[EI_MAG1] == ELFMAG1 ||
    e_ident[EI_MAG2] == ELFMAG2 ||
    e_ident[EI_MAG3] == ELFMAG3, "unexpected ei_mag");

  CHECK_VA2FO(e_ident[EI_CLASS] == ELFCLASS64, "unexpected ei_class");
  CHECK_VA2FO(e_ident[EI_DATA] == ELFDATA2LSB, "unexpected ei_data");
  CHECK_VA2FO(e_ident[EI_VERSION] == EV_CURRENT, "unexpected ei_version");
  CHECK_VA2FO(e_ident[EI_OSABI] == 64, "unexpected ei_osabi"); // ELFOSABI_AMDGPU_HSA

  CHECK_VA2FO(
    e_ident[EI_ABIVERSION] == 2 || // ELFABIVERSION_AMDGPU_HSA_V4
    e_ident[EI_ABIVERSION] == 3 || // ELFABIVERSION_AMDGPU_HSA_V5
    e_ident[EI_ABIVERSION] == 4, "unexpected ei_abiversion"); // ELFABIVERSION_AMDGPU_HSA_V6

  Elf64_Ehdr *ehdr = (Elf64_Ehdr*)buffer.data();
  CHECK_VA2FO(ehdr, "ehdr is nullptr");
  CHECK_VA2FO(ehdr->e_type == ET_DYN, "unexpected e_type");
  CHECK_VA2FO(ehdr->e_machine == ELF::EM_AMDGPU, "unexpected e_machine");

  CHECK_VA2FO(buffer.size() > sizeof(Elf64_Ehdr), "buffer is not large enough");
  CHECK_VA2FO(ehdr->e_phoff != 0, "unexpected e_phoff");

  CHECK_VA2FO(buffer.size() > ehdr->e_phoff + sizeof(Elf64_Phdr), "buffer is not large enough");

  Elf64_Phdr *phdr = (Elf64_Phdr*)((uint8_t*)buffer.data() + ehdr->e_phoff);
  CHECK_VA2FO(phdr, "phdr is nullptr");

  std::vector<std::pair<uint64_t, uint64_t>> segments;
  for (Elf64_Half i = 0; i < ehdr->e_phnum; ++i)
  {
    if (phdr[i].p_type != PT_LOAD)
      continue;

    segments.push_back({phdr[i].p_vaddr - phdr[i].p_offset, phdr[i].p_memsz});
  }

  return segments;
}
