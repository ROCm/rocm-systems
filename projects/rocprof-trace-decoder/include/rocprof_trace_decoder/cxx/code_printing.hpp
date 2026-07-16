// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "disassembly.hpp"
#include "funcmap.hpp"
#include "segment.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocprof_trace_decoder
{
namespace codeobj
{

struct Instruction
{
    Instruction() = default;
    Instruction(std::string&& _inst, size_t _size) : inst(std::move(_inst)), size(_size) {}
    std::string inst{};
    std::string comment{};
    uint64_t faddr{0};
    uint64_t vaddr{0};
    size_t size{0};
    uint64_t ld_addr{0};            // Instruction load address, if from loaded codeobj
    code_object_id_t codeobj_id{0}; // Instruction code object load id, if from loaded codeobj

    static constexpr std::string_view separator = " -> ";
};

class CodeobjDecoderComponent
{
public:
    CodeobjDecoderComponent(const char* codeobj_data, uint64_t codeobj_size)
    {
        // Can throw
        disassembly = std::make_unique<DisassemblyInstance>(codeobj_data, codeobj_size);
        try
        {
            m_symbol_map = disassembly->GetKernelMap(); // Can throw
        }
        catch (...)
        {}

        // Best-effort parse of `.sqtt_funcmap` (emitted by the
        // sqtt_instrumentation LLVM pass). Absence is the common case for
        // non-instrumented binaries and produces no diagnostic. Any failure
        // here must not break disassembly.
        try
        {
            auto section_bytes =
                extract_elf_section(codeobj_data, codeobj_size, ".sqtt_funcmap", m_funcmap.diagnostics);
            if (section_bytes)
            {
                Funcmap parsed = parse_funcmap_section(*section_bytes);
                // Carry forward extraction-time diagnostics into the funcmap.
                if (!m_funcmap.diagnostics.empty())
                {
                    parsed.diagnostics.insert(
                        parsed.diagnostics.begin(), m_funcmap.diagnostics.begin(), m_funcmap.diagnostics.end()
                    );
                }
                m_funcmap = std::move(parsed);

                // Resolve vaddrs for F/K rows by name via the FUNC symbol map.
                std::unordered_map<std::string, uint64_t> name_to_vaddr;
                name_to_vaddr.reserve(m_symbol_map.size());
                for (const auto& [vaddr, sym] : m_symbol_map) name_to_vaddr.emplace(sym.name, vaddr);

                for (auto& entry_ptr : m_funcmap.entries)
                {
                    if (!entry_ptr) continue;
                    if (entry_ptr->kind != FuncmapEntryKind::Function && entry_ptr->kind != FuncmapEntryKind::Kernel)
                        continue;
                    auto it = name_to_vaddr.find(entry_ptr->name);
                    if (it == name_to_vaddr.end()) continue;

                    // FuncmapEntry is shared as `const`; rewrite via a fresh node
                    // and update by_id in lockstep so consumers see consistent state.
                    auto updated = std::make_shared<FuncmapEntry>(*entry_ptr);
                    updated->vaddr = it->second;
                    if (updated->kind != FuncmapEntryKind::Kernel)
                    {
                        auto bid = m_funcmap.by_id.find(updated->id);
                        if (bid != m_funcmap.by_id.end() && bid->second == entry_ptr) bid->second = updated;
                    }
                    entry_ptr = std::move(updated);
                }
            }
        }
        catch (...)
        {}
    }
    ~CodeobjDecoderComponent() = default;

    std::optional<uint64_t> va2fo(uint64_t vaddr) const
    {
        if (disassembly) return disassembly->va2fo(vaddr);
        return std::nullopt;
    };

    std::unique_ptr<Instruction> disassemble_instruction(uint64_t faddr, uint64_t vaddr)
    {
        if (!disassembly) throw std::exception();

        auto pair = disassembly->ReadInstruction(faddr);
        auto inst = std::make_unique<Instruction>(std::move(pair.first), pair.second);
        inst->faddr = faddr;
        inst->vaddr = vaddr;

        return inst;
    }

    const Funcmap& getFuncmap() const { return m_funcmap; }

    std::map<uint64_t, SymbolInfo> m_symbol_map{};
    Funcmap m_funcmap{};
    std::vector<std::shared_ptr<Instruction>> instructions{};
    std::unique_ptr<DisassemblyInstance> disassembly{};
};

class LoadedCodeobjDecoder
{
public:
    LoadedCodeobjDecoder(const char* filepath, uint64_t _load_addr, uint64_t _memsize) :
    load_addr(_load_addr), load_end(_load_addr + _memsize)
    {
        if (!filepath) throw std::runtime_error("Empty filepath.");

        std::string_view fpath(filepath);

        if (fpath.rfind(".out") + 4 == fpath.size())
        {
            std::ifstream file(filepath, std::ios::in | std::ios::binary);

            if (!file.is_open()) throw std::runtime_error("Invalid file " + std::string(filepath));

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
            auto& buffer = binary->buffer;
            decoder = std::make_unique<CodeobjDecoderComponent>(buffer.data(), buffer.size());
        }
    }
    LoadedCodeobjDecoder(const void* data, uint64_t size, uint64_t _load_addr, size_t _memsize) :
    load_addr(_load_addr), load_end(load_addr + _memsize)
    {
        decoder = std::make_unique<CodeobjDecoderComponent>(static_cast<const char*>(data), size);
    }
    std::unique_ptr<Instruction> get(uint64_t ld_addr)
    {
        if (!decoder || ld_addr <= load_addr) return nullptr;

        uint64_t voffset = ld_addr - load_addr;
        auto faddr = decoder->va2fo(voffset);
        if (!faddr) return nullptr;

        auto unique = decoder->disassemble_instruction(*faddr, voffset);
        if (unique == nullptr || unique->size == 0) return nullptr;
        unique->ld_addr = ld_addr;
        return unique;
    }

    uint64_t begin() const { return load_addr; };
    uint64_t end() const { return load_end; }
    uint64_t size() const { return load_end - load_addr; }
    bool inrange(uint64_t addr) const { return addr >= begin() && addr < end(); }

    const char* getSymbolName(uint64_t addr) const
    {
        if (!decoder) return nullptr;

        auto it = decoder->m_symbol_map.find(addr - load_addr);
        if (it != decoder->m_symbol_map.end()) return it->second.name.data();

        return nullptr;
    }

    std::map<uint64_t, SymbolInfo>& getSymbolMap() const
    {
        if (!decoder) throw std::exception();
        return decoder->m_symbol_map;
    }

    const Funcmap& getFuncmap() const
    {
        if (!decoder) throw std::exception();
        return decoder->getFuncmap();
    }
    const uint64_t load_addr;

private:
    uint64_t load_end{0};

    std::unique_ptr<CodeobjDecoderComponent> decoder{nullptr};
};

/**
 * @brief Maps ID and offsets into instructions
 */
class CodeobjMap
{
public:
    CodeobjMap() = default;
    virtual ~CodeobjMap() = default;

    virtual void addDecoder(const char* filepath, code_object_id_t id, uint64_t load_addr, uint64_t memsize)
    {
        decoders[id] = std::make_shared<LoadedCodeobjDecoder>(filepath, load_addr, memsize);
    }

    virtual void addDecoder(
        const void* data, size_t memory_size, code_object_id_t id, uint64_t load_addr, uint64_t memsize
    )
    {
        decoders[id] = std::make_shared<LoadedCodeobjDecoder>(data, memory_size, load_addr, memsize);
    }

    virtual void addDecoder(code_object_id_t id, std::shared_ptr<LoadedCodeobjDecoder>& decoder)
    {
        decoders[id] = decoder;
    }

    virtual bool removeDecoderbyId(code_object_id_t id) { return decoders.erase(id) != 0; }

    std::unique_ptr<Instruction> get(code_object_id_t id, uint64_t offset)
    {
        try
        {
            auto& decoder = decoders.at(id);
            auto inst = decoder->get(decoder->begin() + offset);
            if (inst != nullptr) inst->codeobj_id = id;
            return inst;
        }
        catch (std::out_of_range&)
        {}
        return nullptr;
    }

    const char* getSymbolName(code_object_id_t id, uint64_t offset)
    {
        try
        {
            auto& decoder = decoders.at(id);
            return decoder->getSymbolName(decoder->begin() + offset);
        }
        catch (std::out_of_range&)
        {}
        return nullptr;
    }

    // Lookup a marker by ID within a specific code-object. Returns nullptr if
    // either the codeobj or the ID is unknown — callers should treat absence
    // as "no funcmap info" rather than an error.
    Funcmap::EntryPtr getMarker(code_object_id_t id, uint32_t marker_id) const
    {
        auto it = decoders.find(id);
        if (it == decoders.end()) return nullptr;
        try
        {
            return it->second->getFuncmap().find(marker_id);
        }
        catch (...)
        {
            return nullptr;
        }
    }

    // Throws std::out_of_range when `id` is unknown (matches the lookup
    // pattern used by getSymbolName via decoders.at(id)).
    const Funcmap& getFuncmap(code_object_id_t id) const { return decoders.at(id)->getFuncmap(); }

protected:
    std::unordered_map<code_object_id_t, std::shared_ptr<LoadedCodeobjDecoder>> decoders{};
};

/**
 * @brief Translates virtual addresses to elf file offsets
 */
class CodeobjAddressTranslate : public CodeobjMap
{
    using Super = CodeobjMap;

public:
    CodeobjAddressTranslate() = default;
    ~CodeobjAddressTranslate() override = default;

    void addDecoder(const char* filepath, code_object_id_t id, uint64_t load_addr, uint64_t memsize) override
    {
        this->Super::addDecoder(filepath, id, load_addr, memsize);
        auto ptr = decoders.at(id);
        table.insert({ptr->begin(), ptr->size(), id});
    }

    void addDecoder(const void* data, size_t memory_size, code_object_id_t id, uint64_t load_addr, uint64_t memsize)
        override
    {
        this->Super::addDecoder(data, memory_size, id, load_addr, memsize);
        auto ptr = decoders.at(id);
        table.insert({ptr->begin(), ptr->size(), id});
    }

    void addDecoder(code_object_id_t id, std::shared_ptr<LoadedCodeobjDecoder>& decoder) override
    {
        this->Super::addDecoder(id, decoder);
        auto ptr = decoders.at(id);
        table.insert({ptr->begin(), ptr->size(), id});
    }

    bool removeDecoder(code_object_id_t id, uint64_t load_addr)
    {
        return table.remove(load_addr) && this->Super::removeDecoderbyId(id);
    }

    bool removeDecoder(code_object_id_t id)
    {
        uint64_t addr = 0;
        if (decoders.find(id) != decoders.end()) addr = decoders.at(id)->begin();

        return removeDecoder(id, addr);
    }

    std::unique_ptr<Instruction> get(uint64_t vaddr)
    {
        address_range_t addr_range;
        if (!table.find_codeobj_in_range(vaddr, addr_range)) return nullptr;
        return this->Super::get(addr_range.id, vaddr - addr_range.addr);
    }

    std::unique_ptr<Instruction> get(code_object_id_t id, uint64_t offset)
    {
        if (id == 0 && decoders.find(id) == decoders.end())
            return get(offset);
        else
            return this->Super::get(id, offset);
    }

    const char* getSymbolName(uint64_t vaddr)
    {
        for (auto& [_, decoder] : decoders)
        {
            if (!decoder->inrange(vaddr)) continue;
            return decoder->getSymbolName(vaddr);
        }
        return nullptr;
    }

    std::map<uint64_t, SymbolInfo> getSymbolMap() const
    {
        std::map<uint64_t, SymbolInfo> symbols;

        for (const auto& [id, dec] : decoders)
        {
            auto& smap = dec->getSymbolMap();
            for (auto& [vaddr, sym] : smap) symbols[vaddr + dec->load_addr] = sym;
        }

        return symbols;
    }

    std::map<uint64_t, SymbolInfo> getSymbolMap(code_object_id_t id) const
    {
        if (decoders.find(id) == decoders.end()) return {};

        try
        {
            return decoders.at(id)->getSymbolMap();
        }
        catch (...)
        {
            return {};
        }
    }

    // Marker IDs are scoped per-codeobj. If multiple loaded code-objects
    // happen to share an ID, all hits are returned — there is no defensible
    // tiebreaker at this layer; the caller should disambiguate (e.g. by
    // correlating to the active kernel's code-object).
    std::vector<std::pair<code_object_id_t, Funcmap::EntryPtr>> findMarkerAny(uint32_t marker_id) const
    {
        std::vector<std::pair<code_object_id_t, Funcmap::EntryPtr>> out;
        for (const auto& [id, dec] : decoders)
        {
            try
            {
                if (auto entry = dec->getFuncmap().find(marker_id)) out.emplace_back(id, std::move(entry));
            }
            catch (...)
            {}
        }
        return out;
    }

    // Returns the unique non-zero `W:` value across all loaded code-objects,
    // or 0 if none reported a wave size or the values disagree (in which
    // case a diagnostic is also emitted to std::cerr — wave size is
    // hardware-uniform per trace, so disagreement is a real misconfiguration).
    uint32_t getWaveSize() const
    {
        uint32_t agreed = 0;
        for (const auto& [id, dec] : decoders)
        {
            uint32_t w = 0;
            try
            {
                w = dec->getFuncmap().wave_size;
            }
            catch (...)
            {
                continue;
            }
            if (w == 0) continue;
            if (agreed == 0) { agreed = w; }
            else if (agreed != w)
            {
                std::cerr << "rocprof-trace-decoder: .sqtt_funcmap wave size disagreement (" << agreed << " vs " << w
                          << " from codeobj id " << id << ")\n";
                return 0;
            }
        }
        return agreed;
    }

private:
    CodeobjTableTranslator table{};
};

} // namespace codeobj
} // namespace rocprof_trace_decoder
