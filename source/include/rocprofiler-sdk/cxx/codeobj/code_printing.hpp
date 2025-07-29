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
#include "segment.hpp"

#include <dwarf.h>
#include <elfutils/libdw.h>
#include <hsa/amd_hsa_elf.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace sdk
{
namespace codeobj
{
namespace disassembly
{
using marker_id_t = segment::marker_id_t;

struct Instruction
{
    Instruction() = default;
    Instruction(std::string&& _inst, size_t _size)
    : inst(std::move(_inst))
    , size(_size)
    {}
    std::string inst{};
    std::string comment{};
    uint64_t    faddr{0};
    uint64_t    vaddr{0};
    size_t      size{0};
    uint64_t    ld_addr{0};     // Instruction load address, if from loaded codeobj
    marker_id_t codeobj_id{0};  // Instruction code object load id, if from loaded codeobj

    static constexpr std::string_view separator = " -> ";
};

class CodeobjDecoderComponent
{
    struct ProtectedFd
    {
        ProtectedFd(std::string_view uri)
        {
#if defined(_GNU_SOURCE) && defined(MFD_ALLOW_SEALING) && defined(MFD_CLOEXEC)
            m_fd = ::memfd_create(uri.data(), MFD_ALLOW_SEALING | MFD_CLOEXEC);
#endif
            if(m_fd == -1) m_fd = ::open("/tmp", O_TMPFILE | O_RDWR, 0666);
            if(m_fd == -1) throw std::runtime_error("Could not create a file for codeobj!");
        }
        ~ProtectedFd()
        {
            if(m_fd != -1) ::close(m_fd);
        }
        int m_fd{-1};
    };

public:
    CodeobjDecoderComponent(const char* codeobj_data, uint64_t codeobj_size)
    {
        ProtectedFd prot("");
        if(::write(prot.m_fd, codeobj_data, codeobj_size) != static_cast<int64_t>(codeobj_size))
            throw std::runtime_error("Could not write to temporary file!");

        ::lseek(prot.m_fd, 0, SEEK_SET);
        fsync(prot.m_fd);

        m_line_number_map = {};

        std::unique_ptr<Dwarf, void (*)(Dwarf*)> dbg(dwarf_begin(prot.m_fd, DWARF_C_READ),
                                                     [](Dwarf* _dbg) { dwarf_end(_dbg); });

        if(dbg)
        {
            Dwarf_Off cu_offset{0}, next_offset;
            size_t    header_size;

            std::map<uint64_t, std::string> line_addrs;

            while(
                dwarf_nextcu(
                    dbg.get(), cu_offset, &next_offset, &header_size, nullptr, nullptr, nullptr) ==
                0)
            {
                Dwarf_Die die;
                if(!dwarf_offdie(dbg.get(), cu_offset + header_size, &die))
                {
                    cu_offset = next_offset;
                    continue;
                }

                Dwarf_Lines* lines;
                size_t       line_count;
                if(dwarf_getsrclines(&die, &lines, &line_count) != 0)
                {
                    cu_offset = next_offset;
                    continue;
                }

                for(size_t i = 0; i < line_count; ++i)
                {
                    Dwarf_Addr  addr;
                    int         line_number;
                    Dwarf_Line* line = dwarf_onesrcline(lines, i);

                    if(line && dwarf_lineaddr(line, &addr) == 0 &&
                       dwarf_lineno(line, &line_number) == 0 && line_number != 0)
                    {
                        std::string src             = dwarf_linesrc(line, nullptr, nullptr);
                        auto        dwarf_line      = src + ':' + std::to_string(line_number);
                        auto        call_stack_info = extractInlinedCallStackInfo(dbg.get(), addr);

                        size_t capacity = dwarf_line.size() +
                                          Instruction::separator.size() * call_stack_info.size();
                        for(const auto& call : call_stack_info)
                            capacity += call.size();

                        dwarf_line.reserve(capacity);
                        for(const auto& call : call_stack_info)
                        {
                            dwarf_line += Instruction::separator;
                            dwarf_line += call;
                        }
                        line_addrs[addr] = std::move(dwarf_line);
                    }
                }
                cu_offset = next_offset;
            }

            auto it = line_addrs.begin();
            if(it != line_addrs.end())
            {
                while(std::next(it) != line_addrs.end())
                {
                    uint64_t delta   = std::next(it)->first - it->first;
                    auto     segment = segment::address_range_t{it->first, delta, 0};
                    m_line_number_map.emplace(segment, std::move(it->second));
                    it++;
                }
                auto segment = segment::address_range_t{it->first, codeobj_size - it->first, 0};
                m_line_number_map.emplace(segment, std::move(it->second));
            }
        }

        // Can throw
        disassembly = std::make_unique<DisassemblyInstance>(codeobj_data, codeobj_size);
        try
        {
            m_symbol_map = disassembly->GetKernelMap();  // Can throw
        } catch(...)
        {}
    }
    ~CodeobjDecoderComponent() = default;

    std::optional<uint64_t> va2fo(uint64_t vaddr) const
    {
        if(disassembly) return disassembly->va2fo(vaddr);
        return std::nullopt;
    };

    std::unique_ptr<Instruction> disassemble_instruction(uint64_t faddr, uint64_t vaddr)
    {
        if(!disassembly) throw std::exception();

        auto pair   = disassembly->ReadInstruction(faddr);
        auto inst   = std::make_unique<Instruction>(std::move(pair.first), pair.second);
        inst->faddr = faddr;
        inst->vaddr = vaddr;

        auto it = m_line_number_map.find({vaddr, 0, 0});
        if(it != m_line_number_map.end()) inst->comment = it->second;

        return inst;
    }

    std::map<uint64_t, SymbolInfo>            m_symbol_map{};
    std::vector<std::shared_ptr<Instruction>> instructions{};
    std::unique_ptr<DisassemblyInstance>      disassembly{};

    std::map<segment::address_range_t, std::string> m_line_number_map{};

private:
    /**
     * @brief Extracts inlined function call stack information for a given address
     *
     * This function searches through DWARF debug information to find all inlined functions
     * that contain the specified address, building a complete call stack from the outermost
     * function down to the innermost inlined function.
     *
     * @param dbg DWARF debug information handle
     * @param addr The address to analyze for inlined function information
     * @return Vector of strings representing the call stack, formatted as "filename:line"
     *         The stack is ordered from caller to callee (outermost to innermost)
     */
    static std::vector<std::string> extractInlinedCallStackInfo(Dwarf* dbg, Dwarf_Addr addr);

    /**
     * @brief Checks if a DWARF Debug Information Entry (DIE) contains a specific address
     *
     * This function recursively searches through a DIE and its children to determine
     * if any of them contain the specified address within their address ranges.
     * Used as an optimization to quickly determine if a compilation unit or function
     * contains the target address before doing expensive traversal.
     *
     * @param die Pointer to the DWARF DIE to check
     * @param addr The address to search for
     * @return true if the DIE or any of its children contain the address, false otherwise
     */
    static bool checkDIEContainsAddress(Dwarf_Die* die, Dwarf_Addr addr);

    /**
     * @brief Recursively traverses all DWARF DIEs to find inlined functions at a specific address
     *
     * This function performs a depth-first traversal of the DWARF debug information tree,
     * checking each DIE for inlined function information that covers the specified address.
     * It processes both the current DIE and all its children (including siblings at each level)
     * to ensure comprehensive coverage of all possible inlined function contexts.
     *
     * The traversal is necessary because inlined functions can be nested (function A inlines
     * function B which inlines function C) and multiple inlined functions can exist at the
     * same scope level as siblings in the DWARF tree.
     *
     * @param die Pointer to the current DWARF DIE to examine
     * @param addr The address to search for inlined function information
     * @param call_stack Reference to vector that accumulates the call stack information
     */
    static void traverseAllDIEs(Dwarf_Die*                die,
                                Dwarf_Addr                addr,
                                std::vector<std::string>& call_stack);

    /**
     * @brief Examines a specific DWARF DIE for inlined function information at an address
     *
     * This function checks if a given DIE represents an inlined subroutine that contains
     * the specified address. If it does, it extracts the call site information (filename
     * and line number where the function was inlined) and adds it to the call stack.
     *
     * The function specifically looks for DW_TAG_inlined_subroutine DIEs and validates
     * that the address falls within the DIE's address range (either contiguous via
     * low_pc/high_pc or non-contiguous via ranges attribute). It then extracts the
     * call site information using DW_AT_call_file and DW_AT_call_line attributes.
     *
     * @param die Pointer to the DWARF DIE to examine
     * @param addr The address to check against the DIE's address ranges
     * @param call_stack Reference to vector where call site info will be added
     *
     * @note Only processes DW_TAG_inlined_subroutine DIEs. Regular functions
     *       (DW_TAG_subprogram) are ignored since this function specifically
     *       extracts inlined function call information.
     */
    static void checkDIEForInlinedFunction(Dwarf_Die*                die,
                                           Dwarf_Addr                addr,
                                           std::vector<std::string>& call_stack);
};

class LoadedCodeobjDecoder
{
public:
    LoadedCodeobjDecoder(const char* filepath, uint64_t _load_addr, uint64_t _memsize)
    : load_addr(_load_addr)
    , load_end(_load_addr + _memsize)
    {
        if(!filepath) throw std::runtime_error("Empty filepath.");

        std::string_view fpath(filepath);

        if(fpath.rfind(".out") + 4 == fpath.size())
        {
            std::ifstream file(filepath, std::ios::in | std::ios::binary);

            if(!file.is_open()) throw std::runtime_error("Invalid file " + std::string(filepath));

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
        decoder = std::make_unique<CodeobjDecoderComponent>(static_cast<const char*>(data), size);
    }
    std::unique_ptr<Instruction> get(uint64_t ld_addr)
    {
        if(!decoder || ld_addr < load_addr) return nullptr;

        uint64_t voffset = ld_addr - load_addr;
        auto     faddr   = decoder->va2fo(voffset);
        if(!faddr) return nullptr;

        auto unique = decoder->disassemble_instruction(*faddr, voffset);
        if(unique == nullptr || unique->size == 0) return nullptr;
        unique->ld_addr = ld_addr;
        return unique;
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
    uint64_t load_end{0};

    std::unique_ptr<CodeobjDecoderComponent> decoder{nullptr};
};

/**
 * @brief Maps ID and offsets into instructions
 */
class CodeobjMap
{
public:
    CodeobjMap()          = default;
    virtual ~CodeobjMap() = default;

    virtual void addDecoder(const char* filepath,
                            marker_id_t id,
                            uint64_t    load_addr,
                            uint64_t    memsize)
    {
        decoders[id] = std::make_shared<LoadedCodeobjDecoder>(filepath, load_addr, memsize);
    }

    virtual void addDecoder(const void* data,
                            size_t      memory_size,
                            marker_id_t id,
                            uint64_t    load_addr,
                            uint64_t    memsize)
    {
        decoders[id] =
            std::make_shared<LoadedCodeobjDecoder>(data, memory_size, load_addr, memsize);
    }

    virtual bool removeDecoderbyId(marker_id_t id) { return decoders.erase(id) != 0; }

    std::unique_ptr<Instruction> get(marker_id_t id, uint64_t offset)
    {
        try
        {
            auto& decoder = decoders.at(id);
            auto  inst    = decoder->get(decoder->begin() + offset);
            if(inst != nullptr) inst->codeobj_id = id;
            return inst;
        } catch(std::out_of_range&)
        {}
        return nullptr;
    }

    const char* getSymbolName(marker_id_t id, uint64_t offset)
    {
        try
        {
            auto&    decoder = decoders.at(id);
            uint64_t vaddr   = decoder->begin() + offset;
            if(decoder->inrange(vaddr)) return decoder->getSymbolName(vaddr);
        } catch(std::out_of_range&)
        {}
        return nullptr;
    }

protected:
    std::unordered_map<marker_id_t, std::shared_ptr<LoadedCodeobjDecoder>> decoders{};
};

/**
 * @brief Translates virtual addresses to elf file offsets
 */
class CodeobjAddressTranslate : public CodeobjMap
{
    using Super = CodeobjMap;

public:
    CodeobjAddressTranslate()           = default;
    ~CodeobjAddressTranslate() override = default;

    void addDecoder(const char* filepath,
                    marker_id_t id,
                    uint64_t    load_addr,
                    uint64_t    memsize) override
    {
        this->Super::addDecoder(filepath, id, load_addr, memsize);
        auto ptr = decoders.at(id);
        table.insert({ptr->begin(), ptr->size(), id});
    }

    void addDecoder(const void* data,
                    size_t      memory_size,
                    marker_id_t id,
                    uint64_t    load_addr,
                    uint64_t    memsize) override
    {
        this->Super::addDecoder(data, memory_size, id, load_addr, memsize);
        auto ptr = decoders.at(id);
        table.insert({ptr->begin(), ptr->size(), id});
    }

    bool removeDecoder(marker_id_t id, uint64_t load_addr)
    {
        return table.remove(load_addr) && this->Super::removeDecoderbyId(id);
    }

    bool removeDecoder(marker_id_t id)
    {
        uint64_t addr = 0;
        if(decoders.find(id) != decoders.end()) addr = decoders.at(id)->begin();

        return removeDecoder(id, addr);
    }

    std::unique_ptr<Instruction> get(uint64_t vaddr)
    {
        auto addr_range = table.find_codeobj_in_range(vaddr);
        return this->Super::get(addr_range.id, vaddr - addr_range.addr);
    }

    std::unique_ptr<Instruction> get(marker_id_t id, uint64_t offset)
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

    std::map<uint64_t, SymbolInfo> getSymbolMap() const
    {
        std::map<uint64_t, SymbolInfo> symbols;

        for(const auto& [_, dec] : decoders)
        {
            auto& smap = dec->getSymbolMap();
            for(auto& [vaddr, sym] : smap)
                symbols[vaddr + dec->load_addr] = sym;
        }

        return symbols;
    }

    std::map<uint64_t, SymbolInfo> getSymbolMap(marker_id_t id) const
    {
        if(decoders.find(id) == decoders.end()) return {};

        try
        {
            return decoders.at(id)->getSymbolMap();
        } catch(...)
        {
            return {};
        }
    }

private:
    segment::CodeobjTableTranslator table{};
};

inline std::vector<std::string>
CodeobjDecoderComponent::extractInlinedCallStackInfo(Dwarf* dbg, Dwarf_Addr addr)
{
    std::vector<std::string> call_stack{};

    // Iterate through all compilation units to find the one containing our address
    Dwarf_Off cu_offset{};
    Dwarf_Off next_offset{};
    size_t    header_size{};

    while(dwarf_nextcu(dbg, cu_offset, &next_offset, &header_size, nullptr, nullptr, nullptr) == 0)
    {
        Dwarf_Die cu_die{};
        if(!dwarf_offdie(dbg, cu_offset + header_size, &cu_die))
        {
            cu_offset = next_offset;
            continue;
        }

        bool cu_contains_addr = false;

        // Try to get low_pc and high_pc from CU
        // If no simple range, check if any child DIE contains this address
        Dwarf_Addr low_pc{};
        Dwarf_Addr high_pc{};
        if(dwarf_lowpc(&cu_die, &low_pc) == 0 && dwarf_highpc(&cu_die, &high_pc) == 0)
            cu_contains_addr = (addr >= low_pc && addr < high_pc);
        else
            cu_contains_addr = checkDIEContainsAddress(&cu_die, addr);

        if(cu_contains_addr)
        {
            traverseAllDIEs(&cu_die, addr, call_stack);
            break;
        }

        cu_offset = next_offset;
    }

    // Reverse the call stack to show from caller to callee
    std::reverse(call_stack.begin(), call_stack.end());

    return call_stack;
}

inline bool
CodeobjDecoderComponent::checkDIEContainsAddress(Dwarf_Die* die, Dwarf_Addr addr)
{
    if(die == nullptr) return false;
    // Check current DIE's address range
    Dwarf_Addr low_pc{};
    Dwarf_Addr high_pc{};
    if(dwarf_lowpc(die, &low_pc) == 0 && dwarf_highpc(die, &high_pc) == 0)
    {
        if(addr >= low_pc && addr < high_pc) return true;
    }
    else
    {
        // Check ranges attribute for non-contiguous ranges
        Dwarf_Addr base{};
        ptrdiff_t  offset = 0;
        while((offset = dwarf_ranges(die, offset, &base, &low_pc, &high_pc)) > 0)
            if(addr >= low_pc && addr < high_pc) return true;
    }

    // Check children recursively
    Dwarf_Die child{};
    if(dwarf_child(die, &child) == 0)
        if(checkDIEContainsAddress(&child, addr)) return true;

    return false;
}

inline void
CodeobjDecoderComponent::traverseAllDIEs(Dwarf_Die*                die,
                                         Dwarf_Addr                addr,
                                         std::vector<std::string>& call_stack)
{
    if(die == nullptr) return;
    // Check current DIE for inlined function information
    checkDIEForInlinedFunction(die, addr, call_stack);

    // Traverse children recursively (depth-first)
    Dwarf_Die child{};
    if(dwarf_child(die, &child) == 0)
    {
        // Check all children AND their siblings at this level
        // This is crucial because inlined functions can appear as siblings
        // when multiple functions are inlined at the same scope level
        do
        {
            traverseAllDIEs(&child, addr, call_stack);
        } while(dwarf_siblingof(&child, &child) == 0);
    }
}

inline void
CodeobjDecoderComponent::checkDIEForInlinedFunction(Dwarf_Die*                die,
                                                    Dwarf_Addr                addr,
                                                    std::vector<std::string>& call_stack)
{
    // Only process inlined subroutines - these are functions that were
    // expanded inline at compile time and have call site information
    if(die == nullptr || dwarf_tag(die) != DW_TAG_inlined_subroutine) return;

    Dwarf_Addr low_pc{};
    Dwarf_Addr high_pc{};
    bool       has_range{false};

    // Check if this inlined subroutine covers the target address
    // First try simple contiguous range (low_pc to high_pc)

    if(dwarf_lowpc(die, &low_pc) == 0 && dwarf_highpc(die, &high_pc) == 0)
    {
        // Simple contiguous range - check if address falls within
        has_range = (addr >= low_pc && addr < high_pc);
    }
    else
    {
        // Function may have non-contiguous ranges (optimized code)
        // Check all address ranges associated with this DIE
        Dwarf_Addr base{};
        ptrdiff_t  offset{};

        while((offset = dwarf_ranges(die, offset, &base, &low_pc, &high_pc)) > 0)
        {
            if(addr >= low_pc && addr < high_pc)
            {
                has_range = true;
                break;
            }
        }
    }

    // If address doesn't fall within this inlined function, skip it
    if(!has_range) return;

    // Extract call site information - where this function was inlined
    Dwarf_Attribute call_file_attr{};
    Dwarf_Attribute call_line_attr{};
    Dwarf_Word      call_file{};
    Dwarf_Word      call_line{};

    // Get the file and line number where this function was called/inlined

    if(!dwarf_attr(die, DW_AT_call_file, &call_file_attr) ||
       !dwarf_attr(die, DW_AT_call_line, &call_line_attr) ||
       dwarf_formudata(&call_file_attr, &call_file) != 0 ||
       dwarf_formudata(&call_line_attr, &call_line) != 0)
        return;  // No call site information available

    // Get the compilation unit to resolve file names
    Dwarf_Die cu_die{};
    if(!dwarf_diecu(die, &cu_die, nullptr, nullptr)) return;

    // Get the source files table for this compilation unit
    Dwarf_Files* files{};
    size_t       nfiles{};
    if(dwarf_getsrcfiles(&cu_die, &files, &nfiles) == 0 && call_file < nfiles)
        if(const char* filename = dwarf_filesrc(files, call_file, nullptr, nullptr))
            // Add "filename:line" to call stack showing where this function was inlined
            call_stack.push_back(std::string(filename) + ":" + std::to_string(call_line));
}

}  // namespace disassembly
}  // namespace codeobj
}  // namespace sdk
}  // namespace rocprofiler
