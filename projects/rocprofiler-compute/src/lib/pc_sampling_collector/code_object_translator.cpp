// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "code_object_translator.h"

#include "gsl_assert.h"
#include "rocprofiler-sdk/cxx/codeobj/code_printing.hpp"


using namespace rocm_compute;

code_object_translator_impl_t::code_object_translator_impl_t()
    : m_translator(std::make_unique<rocprofiler::sdk::codeobj::disassembly::CodeobjAddressTranslate>())
{
}

code_object_translator_impl_t::~code_object_translator_impl_t() = default;

void code_object_translator_impl_t::add_code_object(const char* filepath, size_t id, uint64_t load_addr, uint64_t mem_size)
{
    std::cout << "Adding code object from file: " << filepath << ", id: " << id << ", load_addr: " << std::hex << load_addr
              << ", mem_size: " << mem_size << std::dec << std::endl;
    m_translator->addDecoder(filepath, id, load_addr, mem_size);
    const auto& symbols = m_translator->getSymbolMap(id);
    for (const auto& [virtual_address, symbol_info] : symbols)
    {
        std::cout << std::hex << "get_symbols: id=" << id
                  << ", virtual_address=0x" << virtual_address
                  << ", symbol_info.name=" << symbol_info.name
                  << ", symbol_info.vaddr=" << std::hex << symbol_info.vaddr
                  << ", symbol_info.faddr=" << std::hex << symbol_info.faddr
                  << ", symbol_info.mem_size=" << std::hex << symbol_info.mem_size
                  << std::dec << std::endl;
    }
    m_code_object_ids.push_back(id);
}

void code_object_translator_impl_t::add_code_object(uint64_t memory_base,
                                                    size_t   memory_size,
                                                    size_t   id,
                                                    uint64_t load_base,
                                                    uint64_t load_size)
{
    std::cout << "Adding code object from memory: base: " << std::hex << memory_base << ", size: " << memory_size
              << ", id: " << id << ", load_base: " << load_base << ", load_size: " << load_size << std::dec
              << std::endl;
    m_translator->addDecoder(reinterpret_cast<void*>(memory_base), memory_size, id, load_base, load_size);
    const auto& symbols = m_translator->getSymbolMap();
    for (const auto& [virtual_address, symbol_info] : symbols)
    {
        std::cout << std::hex << "get_symbols: id=" << id
                  << ", virtual_address=0x" << virtual_address
                  << ", symbol_info.name=" << symbol_info.name
                  << ", symbol_info.vaddr=" << std::hex << symbol_info.vaddr
                  << ", symbol_info.faddr=" << std::hex << symbol_info.faddr
                  << ", symbol_info.mem_size=" << std::hex << symbol_info.mem_size
                  << std::dec << std::endl;
    }
    m_code_object_ids.push_back(id);
}

const std::vector<size_t>& code_object_translator_impl_t::get_code_object_ids() const
{
    return m_code_object_ids;
}

std::vector<symbol_t> code_object_translator_impl_t::get_symbols(size_t object_id) const
{
    const auto& symbols = m_translator->getSymbolMap(object_id);
    std::vector<symbol_t> symbol_map;
    for (const auto& [virtual_address, symbol_info] : symbols)
    {
        Expects(virtual_address == symbol_info.vaddr);
        symbol_t sym{};
        sym.name            = symbol_info.name;
        sym.code_object_offset     = symbol_info.faddr;
        sym.virtual_address = symbol_info.vaddr;
        sym.size        = symbol_info.mem_size;
        symbol_map.push_back(sym);
    }
    return symbol_map;
}

instruction_t code_object_translator_impl_t::get_instruction(size_t   object_id,
                                                             uint64_t virtual_address) const
{
    const auto& inst = m_translator->get(object_id, virtual_address);
    return {inst->inst, inst->comment, inst->size};
}