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

#include "lib/rocprofiler-sdk-codeobj/code_printing.hpp"

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

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cxxabi.h>
#include <elfutils/libdw.h>
#include <hsa/amd_hsa_elf.h>
#include <sys/mman.h>

#include <atomic>

#define C_API_BEGIN                                                                                \
    try                                                                                            \
    {
#define C_API_END(returndata)                                                                      \
    }                                                                                              \
    catch(std::exception & e)                                                                      \
    {                                                                                              \
        std::string s = e.what();                                                                  \
        if(s.find("memory protocol not supported!") == std::string::npos)                          \
            std::cerr << "Codeobj API lookup: " << e.what() << std::endl;                          \
        return returndata;                                                                         \
    }                                                                                              \
    catch(std::string & s)                                                                         \
    {                                                                                              \
        if(s.find("memory protocol not supported!") == std::string::npos)                          \
            std::cerr << "Codeobj API lookup: " << s << std::endl;                                 \
        return returndata;                                                                         \
    }                                                                                              \
    catch(...) { return returndata; }

CodeobjDecoderComponent::CodeobjDecoderComponent(const char* codeobj_data, uint64_t codeobj_size)
{
    m_fd = -1;
#if defined(_GNU_SOURCE) && defined(MFD_ALLOW_SEALING) && defined(MFD_CLOEXEC)
    m_fd = ::memfd_create(m_uri.c_str(), MFD_ALLOW_SEALING | MFD_CLOEXEC);
#endif
    if(m_fd == -1)  // If fail, attempt under /tmp
        m_fd = ::open("/tmp", O_TMPFILE | O_RDWR, 0666);

    if(m_fd == -1)
    {
        printf("could not create a temporary file for code object\n");
        return;
    }

    if(size_t size = ::write(m_fd, codeobj_data, codeobj_size); size != codeobj_size)
    {
        printf("could not write to the temporary file\n");
        return;
    }
    ::lseek(m_fd, 0, SEEK_SET);
    fsync(m_fd);

    m_line_number_map = {};

    std::unique_ptr<Dwarf, void (*)(Dwarf*)> dbg(dwarf_begin(m_fd, DWARF_C_READ),
                                                 [](Dwarf* _dbg) { dwarf_end(_dbg); });

    /*if (!dbg) {
        rocprofiler::warning("Error opening Dwarf!\n");
        return;
    } */

    if(dbg)
    {
        Dwarf_Off cu_offset{0}, next_offset;
        size_t    header_size;

        std::unordered_set<uint64_t> used_addrs;

        while(!dwarf_nextcu(
            dbg.get(), cu_offset, &next_offset, &header_size, nullptr, nullptr, nullptr))
        {
            Dwarf_Die die;
            if(!dwarf_offdie(dbg.get(), cu_offset + header_size, &die)) continue;

            Dwarf_Lines* lines;
            size_t       line_count;
            if(dwarf_getsrclines(&die, &lines, &line_count)) continue;

            for(size_t i = 0; i < line_count; ++i)
            {
                Dwarf_Addr  addr;
                int         line_number;
                Dwarf_Line* line = dwarf_onesrcline(lines, i);

                if(line && !dwarf_lineaddr(line, &addr) && !dwarf_lineno(line, &line_number) &&
                   line_number)
                {
                    std::string src        = dwarf_linesrc(line, nullptr, nullptr);
                    auto        dwarf_line = src + ':' + std::to_string(line_number);

                    if(used_addrs.find(addr) != used_addrs.end())
                    {
                        size_t pos = m_line_number_map.lower_bound(addr);
                        m_line_number_map.data()[pos].str += ' ' + dwarf_line;
                        continue;
                    }

                    used_addrs.insert(addr);
                    m_line_number_map.insert(DSourceLine{addr, 0, std::move(dwarf_line)});
                }
            }
            cu_offset = next_offset;
        }
    }

    // Can throw
    disassembly = std::make_unique<DisassemblyInstance>(codeobj_data, codeobj_size);
    if(m_line_number_map.size())
    {
        size_t total_size = 0;
        for(size_t i = 0; i < m_line_number_map.size() - 1; i++)
        {
            size_t s = m_line_number_map.get(i + 1).vaddr - m_line_number_map.get(i).vaddr;
            m_line_number_map.data()[i].size = s;
            total_size += s;
        }
        m_line_number_map.back().size = std::max(total_size, codeobj_size) - total_size;
    }
    try
    {
        m_symbol_map = disassembly->GetKernelMap();  // Can throw
    } catch(...)
    {}

    // disassemble_kernels();
}

CodeobjDecoderComponent::~CodeobjDecoderComponent()
{
    if(m_fd) ::close(m_fd);
}

std::shared_ptr<Instruction>
CodeobjDecoderComponent::disassemble_instruction(uint64_t faddr, uint64_t vaddr)
{
    if(!disassembly) throw std::exception();

    const char* cpp_line = nullptr;

    try
    {
        const DSourceLine& it = m_line_number_map.find_obj(vaddr);
        cpp_line              = it.str.data();
    } catch(...)
    {}

    auto pair   = disassembly->ReadInstruction(faddr);
    auto inst   = std::make_shared<Instruction>(std::move(pair.first), pair.second);
    inst->faddr = faddr;
    inst->vaddr = vaddr;

    if(cpp_line) inst->comment = cpp_line;
    return inst;
}

LoadedCodeobjDecoder::LoadedCodeobjDecoder(const char* filepath,
                                           uint64_t    _load_addr,
                                           uint64_t    mem_size)
: load_addr(_load_addr)
, load_end(load_addr + mem_size)
{
    if(!filepath) throw "Empty filepath.";

    std::string_view fpath(filepath);

    if(fpath.rfind(".out") + 4 == fpath.size())
    {
        std::ifstream file(filepath, std::ios::in | std::ios::binary);

        if(!file.is_open()) throw "Invalid filename " + std::string(filepath);

        std::vector<char> buffer;
        file.seekg(0, file.end);
        buffer.resize(file.tellg());
        file.seekg(0, file.beg);
        file.read(buffer.data(), buffer.size());

        decoder = std::make_unique<CodeobjDecoderComponent>(buffer.data(), buffer.size());
    }
    else
    {
        std::unique_ptr<CodeObjectBinary> binary = std::make_unique<CodeObjectBinary>(filepath);
        auto&                             buffer = binary->buffer;
        decoder = std::make_unique<CodeobjDecoderComponent>(buffer.data(), buffer.size());
    }

    elf_segments = decoder->disassembly->getSegments();
}

LoadedCodeobjDecoder::LoadedCodeobjDecoder(const void* data,
                                           size_t      size,
                                           uint64_t    _load_addr,
                                           uint64_t    mem_size)
: load_addr(_load_addr)
, load_end(load_addr + mem_size)
{
    decoder = std::make_unique<CodeobjDecoderComponent>(reinterpret_cast<const char*>(data), size);
    elf_segments = decoder->disassembly->getSegments();
}

std::shared_ptr<Instruction>
LoadedCodeobjDecoder::add_to_map(uint64_t ld_addr)
{
    if(!decoder || ld_addr < load_addr) throw std::out_of_range("Addr not in decoder");

    uint64_t voffset = ld_addr - load_addr;
    auto     faddr   = decoder->disassembly->va2fo(voffset);
    if(!faddr) throw std::out_of_range("Could not find file offset");

    auto shared          = decoder->disassemble_instruction(*faddr, voffset);
    shared->ld_addr      = ld_addr;
    decoded_map[ld_addr] = shared;
    return shared;
}

std::shared_ptr<Instruction>
LoadedCodeobjDecoder::get(uint64_t addr)
{
    if(decoded_map.find(addr) != decoded_map.end()) return decoded_map[addr];

    try
    {
        return add_to_map(addr);
    } catch(std::exception& e)
    {
        std::cerr << e.what() << " at addr " << std::hex << addr << std::dec << std::endl;
    }
    throw std::out_of_range("Invalid address");
    return nullptr;
}

#define PUBLIC_API __attribute__((visibility("default")))
