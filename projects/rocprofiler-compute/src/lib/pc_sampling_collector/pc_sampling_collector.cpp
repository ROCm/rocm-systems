// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "pc_sampling_collector.h"

#include "gsl_assert.h"
#include "source_snapshot.h"

#include <ios>
#include <iostream>
#include <unordered_set>

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

void pc_sampling_collector_impl_t::record_source_path(const std::string& comment)
{
    if (auto path = parse_source_path(comment))
    {
        m_source_paths.insert(*path);
    }
}

void pc_sampling_collector_impl_t::write(code_object_writer_t& writer)
{
    // Reset the harvest so a re-run reflects the current translator state.
    m_source_paths.clear();

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
                record_source_path(inst.comment);
                writer.write_instruction(inst);
                pc += inst.size;
            }
            writer.end_symbol();
        }
        writer.end_code_obj();
    }

    m_source_paths_collected = true;
}

std::vector<std::string> pc_sampling_collector_impl_t::collect_source_paths()
{
    // write() harvests source paths during its disassembly walk, so finalize()
    // can reuse them instead of paying for a second full traversal. When write()
    // has not run, walk once here so the method stays callable independently.
    if (m_source_paths_collected)
    {
        return std::vector<std::string>(m_source_paths.begin(), m_source_paths.end());
    }

    for (const auto& id : m_translator->get_code_object_ids())
    {
        const auto& symbols = m_translator->get_symbols(id);
        for (const auto& sym : symbols)
        {
            uint64_t       pc  = sym.virtual_address;
            const uint64_t end = sym.virtual_address + sym.size;
            while (pc < end)
            {
                const auto& inst = m_translator->get_instruction(id, pc);
                if (inst.size == 0)
                {
                    // Best-effort: avoid an infinite loop without aborting the run.
                    break;
                }
                record_source_path(inst.comment);
                pc += inst.size;
            }
        }
    }

    m_source_paths_collected = true;
    return std::vector<std::string>(m_source_paths.begin(), m_source_paths.end());
}
