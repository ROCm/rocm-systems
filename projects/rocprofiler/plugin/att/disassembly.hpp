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

#include <amd_comgr/amd_comgr.h>
#include <string>
#include <vector>
#include <memory>
#include <limits>

typedef struct {
  const char* instruction;
  const char* cpp_reference;
  uint64_t address;
} instruction_instance_t;

class CodeObjectBinary {
 public:
  CodeObjectBinary(const std::string& uri);
  std::string m_uri;
  std::vector<char> buffer;
};

struct SymbolInfo
{
  std::string name;
  uint64_t faddr;
  uint64_t mem_size;
};

class DisassemblyInstance {
 public:
  DisassemblyInstance(
    const char* codeobj_data,
    uint64_t codeobj_size,
    uint64_t gpu_id
  );
  ~DisassemblyInstance();

  uint64_t ReadInstruction(uint64_t faddr, uint64_t vaddr, const char* cpp_line);
  std::map<uint64_t, SymbolInfo>& GetKernelMap();

  static uint64_t memory_callback(uint64_t from, char* to, uint64_t size, void* user_data);
  static void inst_callback(const char* instruction, void* user_data);
  static amd_comgr_status_t symbol_callback(amd_comgr_symbol_t symbol, void* user_data);
  // Per-gpu_id isa_name
  static std::unordered_map<uint64_t, std::string> agent_isa_name;

  std::optional<uint64_t> va2fo(uint64_t va);
  std::vector<std::pair<uint64_t, uint64_t>> getSegments();

  std::vector<char> buffer;
  instruction_instance_t last_instruction;
  amd_comgr_disassembly_info_t info;
  amd_comgr_data_t data;
  std::map<uint64_t, SymbolInfo> symbol_map;
};
