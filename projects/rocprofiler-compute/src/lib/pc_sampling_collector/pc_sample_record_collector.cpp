// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "pc_sample_record_collector.h"

#include <iostream>
#include <set>

using namespace rocprofiler_compute_tool;

pc_sample_record_collector_t::ptr pc_sample_record_collector_t::create(
    std::shared_ptr<code_object_translator_t> translator)
{
    return std::make_shared<pc_sample_record_collector_impl_t>(std::move(translator));
}

pc_sample_record_collector_impl_t::pc_sample_record_collector_impl_t(
    std::shared_ptr<code_object_translator_t> translator)
    : m_translator(std::move(translator))
{
}

void pc_sample_record_collector_impl_t::on_code_object_load(
    const rocprofiler_callback_tracing_code_object_load_data_t& info)
{
    load_code_object(*m_translator, info);
}

void pc_sample_record_collector_impl_t::add_record(const pc_sampling_record_t& rec)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_records.push_back(rec);
}

void pc_sample_record_collector_impl_t::add_records(const std::vector<pc_sampling_record_t>& recs)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_records.reserve(m_records.size() + recs.size());
    m_records.insert(m_records.end(), recs.begin(), recs.end());
}

void pc_sample_record_collector_impl_t::write(ps_file_writer_t& writer)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    const auto&            ids = m_translator->get_code_object_ids();
    const std::set<size_t> known_ids(ids.begin(), ids.end());

    for (const auto& rec : m_records)
    {
        if (known_ids.find(rec.code_object_id) == known_ids.end())
        {
            std::clog << "Dropping PC sample for unknown code object id " << rec.code_object_id
                      << std::endl;
            continue;
        }
        const auto key = std::make_pair(rec.code_object_id, rec.code_object_offset);
        if (m_inst_index.find(key) == m_inst_index.end())
        {
            const int idx = static_cast<int>(m_instructions.size());
            const uint64_t va = m_translator->get_load_base(rec.code_object_id) + rec.code_object_offset;
            const auto inst = m_translator->get_instruction(rec.code_object_id, va);
            m_instructions.push_back(inst.name);
            m_comments.push_back(inst.comment);
            m_inst_index[key] = idx;
        }
    }

    writer.set_instruction_strings(m_instructions, m_comments);

    for (const auto& id : ids)
    {
        for (const auto& sym : m_translator->get_symbols(id))
        {
            writer.add_kernel_symbol(id, sym.name);
        }
    }

    for (const auto& rec : m_records)
    {
        const auto key = std::make_pair(rec.code_object_id, rec.code_object_offset);
        const auto it  = m_inst_index.find(key);
        if (it == m_inst_index.end())
        {
            continue;
        }
        const int idx = it->second;
        if (rec.is_stochastic)
        {
            writer.add_stochastic_sample(idx, rec);
        }
        else
        {
            writer.add_host_trap_sample(idx, rec);
        }
    }
}
