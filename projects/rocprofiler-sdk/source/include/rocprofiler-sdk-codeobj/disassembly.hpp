// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <amd_comgr/amd_comgr.h>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace rocprofiler
{
namespace codeobj
{
namespace disassembly
{
class CodeObjectBinary
{
public:
    CodeObjectBinary(const std::string& uri);
    std::string       m_uri;
    std::vector<char> buffer;
};

struct SymbolInfo
{
    std::string name{};
    uint64_t    faddr    = 0;
    uint64_t    vaddr    = 0;
    uint64_t    mem_size = 0;
};

class DisassemblyInstance
{
public:
    DisassemblyInstance(const char* codeobj_data, uint64_t codeobj_size);
    ~DisassemblyInstance();

    std::pair<std::string, size_t>  ReadInstruction(uint64_t faddr);
    std::map<uint64_t, SymbolInfo>& GetKernelMap();

    static uint64_t memory_callback(uint64_t from, char* to, uint64_t size, void* user_data);
    static void     inst_callback(const char* instruction, void* user_data);
    static amd_comgr_status_t symbol_callback(amd_comgr_symbol_t symbol, void* user_data);

    std::optional<uint64_t>                    va2fo(uint64_t va);
    std::vector<std::pair<uint64_t, uint64_t>> getSegments();

    std::vector<char>              buffer;
    std::string                    last_instruction;
    amd_comgr_disassembly_info_t   info;
    amd_comgr_data_t               data;
    std::map<uint64_t, SymbolInfo> symbol_map;
};

}  // namespace disassembly
}  // namespace codeobj
}  // namespace rocprofiler
