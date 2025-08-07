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

#pragma once

#include "rocprofiler.h"

#include <map>
#include <optional>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

#include "disassembly.hpp"
#include "segment.hpp"

struct DSourceLine
{
  uint64_t vaddr;
  uint64_t size;
  std::string str;
  uint64_t begin() const { return vaddr; }
  bool inrange(uint64_t addr) const { return addr >= vaddr && addr < vaddr+size; }
};

class CodeObjDecoderComponent
{
public:
  std::optional<SymbolInfo> find_symbol(uint64_t address);

  CodeObjDecoderComponent(const char* codeobj_data, uint64_t codeobj_size, uint64_t gpu_id);
  ~CodeObjDecoderComponent();

  std::pair<instruction_instance_t, size_t>
  disassemble_instruction(uint64_t faddr, uint64_t vaddr);
  void disassemble_kernel(uint64_t faddr, uint64_t vaddr);
  void disassemble_single_kernel(uint64_t kaddr);
  void disassemble_kernels();

  int m_fd;

  std::map<address_range_t, std::string> m_line_number_map{};
  std::map<uint64_t, SymbolInfo> m_symbol_map{};

  std::string m_uri;
  std::vector<instruction_instance_t> instructions{};
  std::unique_ptr<DisassemblyInstance> disassembly{};
};

typedef struct {
  const char* inst;
  const char* cpp;
  size_t size;
} instruction_info_t;

class CodeobjDecoder
{
public:
  CodeobjDecoder(const char* filepath, uint64_t loadbase, uint64_t memsize, uint64_t gpu_id);

  bool decode_single(uint64_t vaddr);
  bool decode_single_at_offset(uint64_t vaddr, uint64_t voffset);
  bool add_to_map(uint64_t faddr, uint64_t vaddr, uint64_t voffset);

  std::pair<instruction_instance_t, size_t>& getDecoded(uint64_t addr);
  const char* getInstruction(uint64_t addr) { return getDecoded(addr).first.instruction; }
  const char* getCppref(uint64_t addr) { return getDecoded(addr).first.cpp_reference; }
  size_t getSize(uint64_t addr) { return getDecoded(addr).second; }
  instruction_info_t get(uint64_t addr) {
    auto& inst = getDecoded(addr);
    return {inst.first.instruction, inst.first.cpp_reference, inst.second};
  }

  uint64_t begin() const { return loadbase; };
  uint64_t end() const { return load_end; }
  uint64_t size() const { return load_end-loadbase; }
  bool inrange(uint64_t addr) const { return addr >= begin() && addr < end(); }

  const char* getSymbolName(uint64_t addr) const {
    if (!decoder) return nullptr;

    auto it = decoder->m_symbol_map.find(addr-loadbase);
    if (it != decoder->m_symbol_map.end())
      return it->second.name.data();

    return nullptr;
  }
  std::vector<std::pair<uint64_t, uint64_t>> elf_segments{};

private:
  const uint64_t loadbase;
  uint64_t load_end = 0;

  std::unordered_map<uint64_t, std::pair<instruction_instance_t, size_t>> decoded_map;
  std::unique_ptr<CodeObjDecoderComponent> decoder{nullptr};
};

/**
 * @brief Maps ID and offsets into instructions
*/
class CodeobjList
{
public:
  CodeobjList() = default;

  virtual void addDecoder(
    const char* filepath,
    uint32_t id,
    uint64_t loadbase,
    uint64_t memsize,
    uint64_t gpu_id
  )
  {
    decoders[id] = std::make_shared<CodeobjDecoder>(filepath, loadbase, memsize, gpu_id);
  }

  virtual bool removeDecoder(uint32_t id)
  {
    return decoders.erase(id) != 0;
  }

  instruction_info_t get(uint32_t id, uint64_t offset)
  {
    auto& decoder = decoders.at(id);
    auto& inst = decoder->getDecoded(decoder->begin() + offset);
    return {inst.first.instruction, inst.first.cpp_reference, inst.second};
  }

  const char* getSymbolName(uint32_t id, uint64_t offset)
  {
    auto& decoder = decoders.at(id);
    uint64_t vaddr = decoder->begin() + offset;
    if (decoder->inrange(vaddr))
      return decoder->getSymbolName(vaddr);
    return nullptr;
  }

protected:
  std::unordered_map<uint32_t, std::shared_ptr<CodeobjDecoder>> decoders{};
};

/**
 * @brief Translates virtual addresses to elf file offsets
*/
class CodeobjTableTranslation : protected CodeobjList
{
  using Super = CodeobjList;
public:
  CodeobjTableTranslation() = default;

  virtual void addDecoder(
    const char* filepath,
    uint32_t id,
    uint64_t loadbase,
    uint64_t memsize,
    uint64_t gpu_id
  ) override
  {
    this->Super::addDecoder(filepath, id, loadbase, memsize, gpu_id);
    auto ptr = decoders.at(id);
    table.insert(address_range_t{ptr->begin(), static_cast<uint32_t>(ptr->size()), id});
  }

  virtual bool removeDecoder(uint32_t id, uint64_t loadbase)
  {
    return table.remove(loadbase) && this->Super::removeDecoder(id);
  }

  instruction_info_t get(uint64_t vaddr)
  {
    auto addr_range = table.find_codeobj_in_range(vaddr);
    return get(addr_range.id, vaddr - addr_range.addr);
  }
  instruction_info_t get(uint32_t id, uint64_t offset) { return this->Super::get(id, offset); }

  const char* getSymbolName(uint64_t vaddr)
  {
    for (auto& [_, decoder] : decoders)
    {
      if (!decoder->inrange(vaddr)) continue;
      return decoder->getSymbolName(vaddr);
    }
    return nullptr;
  }

private:
  CodeobjTableTranslator table;
};
