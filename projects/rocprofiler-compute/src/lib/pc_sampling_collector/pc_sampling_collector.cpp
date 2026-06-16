// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "pc_sampling_collector.h"

#include "gsl_assert.h"
#include "pair_hash.h"
#include "source_snapshot.h"

#include <unistd.h>

#include <ios>
#include <iostream>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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

        // Disassemble the whole loaded code-object range, not just named symbols.
        // PC samples can land on any instruction, including code that has no ELF
        // symbol; analyze joins each sample's code_object_offset to an entry here,
        // so every loaded instruction must be present. Group them under one span
        // symbol -- the consumer flattens symbols and reads only the instruction
        // list, keyed by the loaded-basis code_obj_offset.
        const uint64_t base = m_translator->get_load_base(id);

        symbol_t span{};
        span.name               = "<code object>";
        span.virtual_address    = base;
        span.code_object_offset = 0;
        span.size               = m_translator->get_load_size(id);
        writer.start_symbol(span);

        for_each_instruction_in(id,
                                [&writer](const instruction_t& inst)
                                { writer.write_instruction(inst); });

        writer.end_symbol();
        writer.end_code_obj();
    }
}

void pc_sampling_collector_impl_t::append_sample(const pc_sample_record_t& record)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_samples.push_back(record);
}

void pc_sampling_collector_impl_t::add_kernel_symbol(uint64_t           code_object_id,
                                                     const std::string& formatted_kernel_name,
                                                     uint64_t           kernel_id)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_kernel_symbols.push_back(kernel_symbol_entry_t{code_object_id, formatted_kernel_name, kernel_id});
}

void pc_sampling_collector_impl_t::add_agent(const agent_record_t& agent)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_agents.push_back(agent);
}

void pc_sampling_collector_impl_t::append_kernel_dispatch(const kernel_dispatch_record_t& record)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_kernel_dispatches.push_back(record);
}

instruction_t pc_sampling_collector_impl_t::resolve_instruction(uint64_t code_object_id,
                                                                uint64_t code_object_offset)
{
    // PC sample offsets are relative to the code object's load base, but
    // get_instruction keys on a global virtual address. Translate before lookup.
    try
    {
        const uint64_t vaddr = code_object_offset + m_translator->get_load_base(code_object_id);
        return m_translator->get_instruction(code_object_id, vaddr);
    }
    catch (const std::out_of_range&)
    {
        return instruction_t{};
    }
}

void pc_sampling_collector_impl_t::write_samples(pc_sample_writer_t& writer)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    writer.begin();
    // Many samples hit the same PC, so cache the string-table index per location
    // to avoid re-disassembling and re-inserting each one.
    std::unordered_map<std::pair<uint64_t, uint64_t>, size_t, pair_hash_t> idx_by_location;
    for (const auto& sample : m_samples)
    {
        const auto location = std::make_pair(sample.pc.code_object_id, sample.pc.code_object_offset);
        size_t idx = 0;
        if (const auto it = idx_by_location.find(location); it != idx_by_location.end())
        {
            idx = it->second;
        }
        else
        {
            const instruction_t inst = resolve_instruction(sample.pc.code_object_id,
                                                           sample.pc.code_object_offset);
            idx                      = m_string_table.insert(inst.name, inst.comment);
            idx_by_location.emplace(location, idx);
        }

        switch (sample.kind)
        {
        case pc_sample_kind_t::Stochastic:
            writer.append_stochastic(sample, idx);
            break;
        case pc_sample_kind_t::HostTrap:
            writer.append_host_trap(sample, idx);
            break;
        }
    }

    writer.set_strings(m_string_table);
    writer.set_kernel_symbols(m_kernel_symbols);
    writer.set_agents(m_agents);
    writer.set_kernel_dispatches(m_kernel_dispatches);
    writer.set_metadata(static_cast<int>(getpid()));
}

size_t pc_sampling_collector_impl_t::snapshot_sources(const std::filesystem::path& output_root)
{
    const std::shared_ptr<source_snapshot_t> snapshotter = source_snapshot_t::create();

    // A large kernel can annotate hundreds of thousands of instructions with the
    // same handful of source files, so dedup at collection time rather than
    // accumulating one string per instruction.
    std::set<std::string> unique_refs;
    for_each_instruction(
        [&unique_refs, &snapshotter](uint64_t /*id*/, const instruction_t& inst)
        {
            if (const auto ref = snapshotter->parse_ref(inst.comment))
            {
                unique_refs.insert(*ref);
            }
        });

    const std::vector<std::string> refs(unique_refs.begin(), unique_refs.end());
    return snapshotter->snapshot(refs, output_root);
}
