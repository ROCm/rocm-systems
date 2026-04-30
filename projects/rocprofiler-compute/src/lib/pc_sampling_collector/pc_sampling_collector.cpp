// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "pc_sampling_collector.h"

#include "gsl_assert.h"

#include <ios>
#include <iostream>

using namespace rocm_compute;

pc_sampling_collector_t::ptr pc_sampling_collector_t::create()
{
    return std::make_unique<pc_sampling_collector_impl_t>(
        std::make_shared<code_object_translator_impl_t>());
}

pc_sampling_collector_impl_t::pc_sampling_collector_impl_t(
    const std::shared_ptr<code_object_translator_t>& translator)
    : m_translator(translator)
{
}

void pc_sampling_collector_impl_t::on_code_object_load(
    const rocprofiler_callback_tracing_code_object_load_data_t& info)
{
    if (info.storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE)
    {
        m_translator->add_code_object(info.uri, info.code_object_id, info.load_delta, info.load_size);
    }
    else if (info.storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY)
    {
        m_translator->add_code_object(info.memory_base,
                                      info.memory_size,
                                      info.code_object_id,
                                      info.load_delta,
                                      info.load_size);
    }
}

void pc_sampling_collector_impl_t::write(code_object_writer_t& writer)
{
    for (const auto& id : m_translator->get_code_object_ids())
    {
        writer.start_code_obj(id);
        const auto& symbols = m_translator->get_symbols(id);
        for (const auto& sym : symbols)
        {
            std::clog << "[pc_sampling_collector] write: object_id=" << id
                      << ", symbol.name=" << sym.name
                      << ", symbol.vaddr=" << std::hex << sym.virtual_address
                      << ", symbol.code_object_offset=" << std::hex << sym.code_object_offset
                      << ", symbol.size=" << std::dec << sym.size << "\n";
            writer.start_symbol(sym);
            uint64_t       pc  = sym.virtual_address;
            const uint64_t end = sym.virtual_address + sym.size;
            while (pc < end)
            {
                const auto& inst = m_translator->get_instruction(id, pc);
                if (inst.size == 0)
                {
                    // If instruction size is 0, it means the translator failed to decode instruction at this address.
                    // To avoid infinite loop, break out of the loop in this case. This may cause some instructions at the end of the symbol to be missing in the output, but it's better than hanging indefinitely.
                    std::clog << "[pc_sampling_collector] Warning: Failed to decode instruction at virtual address "
                              << std::hex << pc << " in symbol " << sym.name << " (object_id=" << id
                              << "). Stopping further decoding of this symbol.\n";
                    break;
                }
                writer.write_instruction(inst);
                pc += inst.size;
            }
            writer.end_symbol();
        }
        writer.end_code_obj();
    }
}

