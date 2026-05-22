// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "code_object_translator.h"

#include "gsl_assert.h"
#include "rocprofiler-sdk/cxx/codeobj/code_printing.hpp"

#include <mutex>

using namespace rocm_compute;

code_object_translator_impl_t::code_object_translator_impl_t()
    : m_translator(std::make_unique<rocprofiler::sdk::codeobj::disassembly::CodeobjAddressTranslate>())
{
}

code_object_translator_impl_t::~code_object_translator_impl_t() = default;

void code_object_translator_impl_t::add_file_code_object(const char* filepath,
                                                         size_t      id,
                                                         uint64_t    load_addr,
                                                         uint64_t    load_size)
{
    auto lock = std::unique_lock{m_mutex};
    m_translator->addDecoder(filepath, id, load_addr, load_size);
    m_obj_id_to_load_addr[id] = load_addr;
    m_obj_ids.push_back(id);
}

void code_object_translator_impl_t::add_memory_code_object(uint64_t memory_base,
                                                           size_t   memory_size,
                                                           size_t   id,
                                                           uint64_t load_base,
                                                           uint64_t load_size)
{
    auto lock = std::unique_lock{m_mutex};
    m_translator->addDecoder(reinterpret_cast<void*>(memory_base), memory_size, id, load_base, load_size);
    m_obj_id_to_load_addr[id] = load_base;
    m_obj_ids.push_back(id);
}

std::vector<size_t> code_object_translator_impl_t::get_code_object_ids() const
{
    auto lock = std::shared_lock{m_mutex};
    return m_obj_ids;
}

std::vector<symbol_t> code_object_translator_impl_t::get_symbols(size_t object_id) const
{
    auto lock = std::shared_lock{m_mutex};
    Expects(m_obj_id_to_load_addr.find(object_id) != m_obj_id_to_load_addr.end());
    const auto&           symbols      = m_translator->getSymbolMap(object_id);
    const auto&           load_address = m_obj_id_to_load_addr.at(object_id);
    std::vector<symbol_t> symbol_map;
    for (const auto& [virtual_address, symbol_info] : symbols)
    {
        Expects(virtual_address == symbol_info.vaddr);
        symbol_t sym{};
        sym.name               = symbol_info.name;
        sym.code_object_offset = symbol_info.faddr;
        sym.virtual_address    = symbol_info.vaddr + load_address;
        sym.size               = symbol_info.mem_size;
        symbol_map.push_back(sym);
    }
    return symbol_map;
}

instruction_t code_object_translator_impl_t::get_instruction(size_t /*object_id*/,
                                                             uint64_t virtual_address) const
{
    auto        lock = std::shared_lock{m_mutex};
    const auto& inst = m_translator->get(virtual_address);
    if (inst)
    {
        return {inst->inst, inst->comment, virtual_address, inst->faddr, inst->size};
    }
    return {};
}
