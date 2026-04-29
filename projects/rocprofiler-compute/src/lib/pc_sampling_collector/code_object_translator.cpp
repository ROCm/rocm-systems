// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "code_object_translator.h"
#include "rocprofiler-sdk/cxx/codeobj/code_printing.hpp"


using namespace rocm_compute;

code_object_translator_impl_t::code_object_translator_impl_t()
    : m_translator(std::make_unique<rocprofiler::sdk::codeobj::disassembly::CodeobjAddressTranslate>())
{
}

code_object_translator_impl_t::~code_object_translator_impl_t() = default;

void code_object_translator_impl_t::add_code_object(const char* filepath, size_t id, uint64_t load_addr, uint64_t mem_size)
{
    m_translator->addDecoder(filepath, id, load_addr, mem_size);
    m_code_object_ids.push_back(id);
}

void code_object_translator_impl_t::add_code_object(uint64_t memory_base,
                                                    size_t   memory_size,
                                                    size_t   id,
                                                    uint64_t load_base,
                                                    uint64_t load_size)
{
    m_translator->addDecoder(reinterpret_cast<void*>(memory_base), memory_size, id, load_base, load_size);
    m_code_object_ids.push_back(id);
}

std::vector<size_t> code_object_translator_impl_t::get_code_object_ids() const
{
    return m_code_object_ids;
}

void code_object_translator_impl_t::get_symbol_map(size_t object_id) const
{
    const auto& symbols = m_translator->getSymbolMap(object_id);
}
