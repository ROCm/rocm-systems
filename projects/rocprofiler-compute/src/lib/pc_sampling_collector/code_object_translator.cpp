// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "code_object_translator.h"

#include "gsl_assert.h"
#include "rocprofiler-sdk/cxx/codeobj/code_printing.hpp"

using namespace rocprofiler_compute_tool;

code_object_translator_impl_t::code_object_translator_impl_t()
    : m_translator(std::make_unique<rocprofiler::sdk::codeobj::disassembly::CodeobjAddressTranslate>())
{
}

code_object_translator_impl_t::~code_object_translator_impl_t() = default;

void code_object_translator_impl_t::add_code_object(const char* filepath,
                                                    size_t      id,
                                                    uint64_t    load_addr,
                                                    uint64_t    mem_size)
{
    m_translator->addDecoder(filepath, id, load_addr, mem_size);
    m_obj_id_to_load_addr[id] = load_addr;
    m_obj_id_to_load_size[id] = mem_size;
    m_obj_ids.push_back(id);
}

void code_object_translator_impl_t::add_code_object(uint64_t memory_base,
                                                    size_t   memory_size,
                                                    size_t   id,
                                                    uint64_t load_base,
                                                    uint64_t load_size)
{
    m_translator->addDecoder(reinterpret_cast<void*>(memory_base), memory_size, id, load_base, load_size);
    m_obj_id_to_load_addr[id] = load_base;
    m_obj_id_to_load_size[id] = load_size;
    m_obj_ids.push_back(id);
}

const std::vector<size_t>& code_object_translator_impl_t::get_code_object_ids() const
{
    return m_obj_ids;
}

uint64_t code_object_translator_impl_t::get_load_base(size_t object_id) const
{
    return m_obj_id_to_load_addr.at(object_id);
}

uint64_t code_object_translator_impl_t::get_load_size(size_t object_id) const
{
    return m_obj_id_to_load_size.at(object_id);
}

std::vector<symbol_t> code_object_translator_impl_t::get_symbols(size_t object_id) const
{
    Expects(m_obj_id_to_load_addr.find(object_id) != m_obj_id_to_load_addr.end());
    const auto&           symbols      = m_translator->getSymbolMap(object_id);
    const auto&           load_address = m_obj_id_to_load_addr.at(object_id);
    std::vector<symbol_t> symbol_map;
    for (const auto& [virtual_address, symbol_info] : symbols)
    {
        Expects(virtual_address == symbol_info.vaddr);
        symbol_t sym{};
        sym.name = symbol_info.name;
        // PC samples carry the loaded-basis offset (vaddr - load_base). Key the
        // symbol on the same basis so analyze joins to it, not the ELF file
        // offset (faddr), which differs from the loaded layout.
        sym.code_object_offset = symbol_info.vaddr;
        sym.virtual_address    = symbol_info.vaddr + load_address;
        sym.size               = symbol_info.mem_size;
        symbol_map.push_back(sym);
    }
    return symbol_map;
}

instruction_t code_object_translator_impl_t::get_instruction(size_t object_id, uint64_t virtual_address) const
{
    // The underlying COMGR disassembler throws on bytes it cannot decode (e.g.
    // data/padding inside the loaded range). The full-range walk in write() must
    // tolerate that: return an empty (size 0) instruction so the caller can skip
    // and advance rather than aborting serialization.
    try
    {
        const auto& inst = m_translator->get(virtual_address);
        if (inst)
        {
            // inst->vaddr is the loaded-basis offset (global vaddr - load_base),
            // which matches a PC sample's code_object_offset. Use it as the join
            // key rather than inst->faddr (the ELF file offset), so analyze
            // attributes the right instruction to each sampled PC.
            return {inst->inst, inst->comment, virtual_address, inst->vaddr, inst->size};
        }
    }
    catch (const std::exception&)
    {
        // fall through to the empty instruction below
    }
    return {};
}
