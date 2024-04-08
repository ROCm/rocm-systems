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

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "disassembly.hpp"
#include "segment.hpp"

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
    CodeobjDecoderComponent(const char* codeobj_data, uint64_t codeobj_size);
    ~CodeobjDecoderComponent();

    std::shared_ptr<Instruction> disassemble_instruction(uint64_t faddr, uint64_t vaddr);
    int                          m_fd;

    cached_ordered_vector<DSourceLine> m_line_number_map;
    std::map<uint64_t, SymbolInfo>     m_symbol_map{};

    std::string                               m_uri;
    std::vector<std::shared_ptr<Instruction>> instructions{};
    std::unique_ptr<DisassemblyInstance>      disassembly{};
};

class LoadedCodeobjDecoder
{
public:
    LoadedCodeobjDecoder(const char* filepath, uint64_t load_addr, uint64_t memsize);
    LoadedCodeobjDecoder(const void* data, uint64_t size, uint64_t load_addr, size_t memsize);
    std::shared_ptr<Instruction> add_to_map(uint64_t ld_addr);

    std::shared_ptr<Instruction> get(uint64_t addr);
    uint64_t                     begin() const { return load_addr; };
    uint64_t                     end() const { return load_end; }
    uint64_t                     size() const { return load_end - load_addr; }
    bool inrange(uint64_t addr) const { return addr >= begin() && addr < end(); }

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
    std::vector<std::pair<uint64_t, uint64_t>> elf_segments{};
    const uint64_t                             load_addr;

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
class CodeobjAddressTranslate : protected CodeobjMap
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
