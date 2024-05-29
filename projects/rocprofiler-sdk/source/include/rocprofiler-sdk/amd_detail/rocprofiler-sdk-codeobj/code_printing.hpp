// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <elfutils/libdw.h>
#include <hsa/amd_hsa_elf.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "disassembly.hpp"
#include "segment.hpp"

namespace rocprofiler
{
namespace codeobj
{
namespace disassembly
{
struct Instruction
{
    Instruction() = default;
    Instruction(std::string&& _inst, size_t _size)
    : inst(std::move(_inst))
    , size(_size)
    {}
    std::string inst;
    std::string comment;
    uint64_t    faddr;
    uint64_t    vaddr;
    uint64_t    ld_addr;
    size_t      size;
};

struct DSourceLine
{
    uint64_t    vaddr;
    uint64_t    size;
    std::string str;
    uint64_t    begin() const { return vaddr; }
    bool        inrange(uint64_t addr) const { return addr >= vaddr && addr < vaddr + size; }
};

class CodeobjDecoderComponent
{
public:
    CodeobjDecoderComponent(const void* codeobj_data, uint64_t codeobj_size)
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

        if(size_t size = ::write(m_fd, (const char*) codeobj_data, codeobj_size);
           size != codeobj_size)
        {
            printf("could not write to the temporary file\n");
            return;
        }
        ::lseek(m_fd, 0, SEEK_SET);
        fsync(m_fd);

        m_line_number_map = {};

        std::unique_ptr<Dwarf, void (*)(Dwarf*)> dbg(dwarf_begin(m_fd, DWARF_C_READ),
                                                     [](Dwarf* _dbg) { dwarf_end(_dbg); });

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
        disassembly =
            std::make_unique<DisassemblyInstance>((const char*) codeobj_data, codeobj_size);
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
    ~CodeobjDecoderComponent()
    {
        if(m_fd) ::close(m_fd);
    }

    std::optional<uint64_t> va2fo(uint64_t vaddr)
    {
        if(disassembly) return disassembly->va2fo(vaddr);
        return {};
    };

    std::shared_ptr<Instruction> disassemble_instruction(uint64_t faddr, uint64_t vaddr)
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
    int m_fd;

    cached_ordered_vector<DSourceLine> m_line_number_map;
    std::map<uint64_t, SymbolInfo>     m_symbol_map{};

    std::string                               m_uri;
    std::vector<std::shared_ptr<Instruction>> instructions{};
    std::unique_ptr<DisassemblyInstance>      disassembly{};
};

class LoadedCodeobjDecoder
{
public:
    LoadedCodeobjDecoder(const char* filepath, uint64_t _load_addr, uint64_t _memsize)
    : load_addr(_load_addr)
    , load_end(_load_addr + _memsize)
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
    }
    LoadedCodeobjDecoder(const void* data, uint64_t size, uint64_t _load_addr, size_t _memsize)
    : load_addr(_load_addr)
    , load_end(load_addr + _memsize)
    {
        decoder =
            std::make_unique<CodeobjDecoderComponent>(reinterpret_cast<const char*>(data), size);
    }
    std::shared_ptr<Instruction> add_to_map(uint64_t ld_addr)
    {
        if(!decoder || ld_addr < load_addr) throw std::out_of_range("Addr not in decoder");

        uint64_t voffset = ld_addr - load_addr;
        auto     faddr   = decoder->va2fo(voffset);
        if(!faddr) throw std::out_of_range("Could not find file offset");

        auto shared          = decoder->disassemble_instruction(*faddr, voffset);
        shared->ld_addr      = ld_addr;
        decoded_map[ld_addr] = shared;
        return shared;
    }

    std::shared_ptr<Instruction> get(uint64_t addr)
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
    uint64_t begin() const { return load_addr; };
    uint64_t end() const { return load_end; }
    uint64_t size() const { return load_end - load_addr; }
    bool     inrange(uint64_t addr) const { return addr >= begin() && addr < end(); }

    const char* getSymbolName(uint64_t addr) const
    {
        if(!decoder) return nullptr;

        auto it = decoder->m_symbol_map.find(addr - load_addr);
        if(it != decoder->m_symbol_map.end()) return it->second.name.data();

        return nullptr;
    }

    std::map<uint64_t, SymbolInfo>& getSymbolMap() const
    {
        if(!decoder) throw std::exception();
        return decoder->m_symbol_map;
    }
    const uint64_t load_addr;

private:
    uint64_t load_end = 0;

    std::unordered_map<uint64_t, std::shared_ptr<Instruction>> decoded_map;
    std::unique_ptr<CodeobjDecoderComponent>                   decoder{nullptr};
};

/**
 * @brief Maps ID and offsets into instructions
 */
class CodeobjMap
{
public:
    CodeobjMap() = default;

    virtual void addDecoder(const char*         filepath,
                            codeobj_marker_id_t id,
                            uint64_t            load_addr,
                            uint64_t            memsize)
    {
        decoders[id] = std::make_shared<LoadedCodeobjDecoder>(filepath, load_addr, memsize);
    }

    virtual void addDecoder(const void*         data,
                            size_t              memory_size,
                            codeobj_marker_id_t id,
                            uint64_t            load_addr,
                            uint64_t            memsize)
    {
        decoders[id] =
            std::make_shared<LoadedCodeobjDecoder>(data, memory_size, load_addr, memsize);
    }

    virtual bool removeDecoderbyId(codeobj_marker_id_t id) { return decoders.erase(id) != 0; }

    std::shared_ptr<Instruction> get(codeobj_marker_id_t id, uint64_t offset)
    {
        auto& decoder = decoders.at(id);
        return decoder->get(decoder->begin() + offset);
    }

    const char* getSymbolName(codeobj_marker_id_t id, uint64_t offset)
    {
        auto&    decoder = decoders.at(id);
        uint64_t vaddr   = decoder->begin() + offset;
        if(decoder->inrange(vaddr)) return decoder->getSymbolName(vaddr);
        return nullptr;
    }

protected:
    std::unordered_map<codeobj_marker_id_t, std::shared_ptr<LoadedCodeobjDecoder>> decoders{};
};

/**
 * @brief Translates virtual addresses to elf file offsets
 */
class CodeobjAddressTranslate : public CodeobjMap
{
    using Super = CodeobjMap;

public:
    CodeobjAddressTranslate() = default;

    virtual void addDecoder(const char*         filepath,
                            codeobj_marker_id_t id,
                            uint64_t            load_addr,
                            uint64_t            memsize) override
    {
        this->Super::addDecoder(filepath, id, load_addr, memsize);
        auto ptr = decoders.at(id);
        table.insert({ptr->begin(), ptr->size(), id, 0});
    }

    virtual void addDecoder(const void*         data,
                            size_t              memory_size,
                            codeobj_marker_id_t id,
                            uint64_t            load_addr,
                            uint64_t            memsize) override
    {
        this->Super::addDecoder(data, memory_size, id, load_addr, memsize);
        auto ptr = decoders.at(id);
        table.insert({ptr->begin(), ptr->size(), id, 0});
    }

    virtual bool removeDecoder(codeobj_marker_id_t id, uint64_t load_addr)
    {
        return table.remove(load_addr) && this->Super::removeDecoderbyId(id);
    }

    std::shared_ptr<Instruction> get(uint64_t vaddr)
    {
        auto& addr_range = table.find_codeobj_in_range(vaddr);
        return this->Super::get(addr_range.id, vaddr - addr_range.vbegin);
    }

    std::shared_ptr<Instruction> get(codeobj_marker_id_t id, uint64_t offset)
    {
        if(id == 0)
            return get(offset);
        else
            return this->Super::get(id, offset);
    }

    const char* getSymbolName(uint64_t vaddr)
    {
        for(auto& [_, decoder] : decoders)
        {
            if(!decoder->inrange(vaddr)) continue;
            return decoder->getSymbolName(vaddr);
        }
        return nullptr;
    }

    void getSymbolMap(const std::shared_ptr<LoadedCodeobjDecoder>& dec,
                      std::unordered_map<uint64_t, SymbolInfo>&    symbols) const
    {
        try
        {
            auto& smap = dec->getSymbolMap();
            for(auto& [vaddr, sym] : smap)
                symbols[vaddr + dec->load_addr] = sym;
        } catch(std::exception& e)
        {
            return;
        };
    }

    std::unordered_map<uint64_t, SymbolInfo> getSymbolMap() const
    {
        std::unordered_map<uint64_t, SymbolInfo> symbols;

        for(auto& [_, dec] : decoders)
            this->getSymbolMap(dec, symbols);

        return symbols;
    }

    std::unordered_map<uint64_t, SymbolInfo> getSymbolMap(codeobj_marker_id_t id) const
    {
        std::unordered_map<uint64_t, SymbolInfo> symbols;

        auto it = decoders.find(id);
        if(it == decoders.end()) return symbols;

        this->getSymbolMap(it->second, symbols);
        return symbols;
    }

private:
    CodeobjTableTranslator table;
};

}  // namespace disassembly
}  // namespace codeobj
}  // namespace rocprofiler
