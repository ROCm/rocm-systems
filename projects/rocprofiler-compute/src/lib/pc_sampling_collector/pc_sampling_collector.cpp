// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "pc_sampling_collector.h"

#include "gsl_assert.h"

#include <ios>
#include <iostream>

using namespace rocprofiler_compute_tool;

pc_sampling_collector_t::ptr pc_sampling_collector_t::create()
{
    return std::make_shared<pc_sampling_collector_impl_t>(
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
        m_translator->add_code_object(info.uri, info.code_object_id, info.load_base, info.load_size);
    }
    else if (info.storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY)
    {
        m_translator->add_code_object(info.memory_base,
                                      info.memory_size,
                                      info.code_object_id,
                                      info.load_base,
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
            writer.start_symbol(sym);
            uint64_t       pc  = sym.virtual_address;
            const uint64_t end = sym.virtual_address + sym.size;
            while (pc < end)
            {
                const auto& inst = m_translator->get_instruction(id, pc);
                Expects(inst.size);
                writer.write_instruction(inst);
                pc += inst.size;
            }
            writer.end_symbol();
        }
        writer.end_code_obj();
    }
}
