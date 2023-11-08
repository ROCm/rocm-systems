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

#include "disassembly.hpp"

class code_object_decoder_t {
 public:
  // void load_symbol_map();
  std::optional<SymbolInfo> find_symbol(uint64_t address);

  code_object_decoder_t(const char* codeobj_data, uint64_t codeobj_size);
  ~code_object_decoder_t();

  void disassemble_kernel(uint64_t faddr, uint64_t vaddr);
  void disassemble_single_kernel(uint64_t kaddr);
  void disassemble_kernels();

  int m_fd;

  std::map<uint64_t, std::pair<std::string, size_t>> m_line_number_map;
  std::map<uint64_t, SymbolInfo> m_symbol_map;

  std::string m_uri;
  std::vector<char> buffer;
  std::vector<instruction_instance_t> instructions;
  std::unique_ptr<DisassemblyInstance> disassembly;
};
