// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "mocks.h"

using namespace rocprofiler_compute_tool;

void mock_code_object_translator_t::add_code_object(const char* filepath,
                                                    size_t      id,
                                                    uint64_t    load_addr,
                                                    uint64_t    load_size)
{
    m_code_object_ids.push_back(id);
    m_file_code_obj_info.push_back({filepath, id, load_addr, load_size});
}

void mock_code_object_translator_t::add_code_object(uint64_t memory_base,
                                                    size_t   memory_size,
                                                    size_t   id,
                                                    uint64_t load_base,
                                                    uint64_t load_size)
{
    m_code_object_ids.push_back(id);
    m_mem_code_obj_info.push_back({memory_base, memory_size, id, load_base, load_size});
}

const std::vector<size_t>& mock_code_object_translator_t::get_code_object_ids() const
{
    return m_code_object_ids;
}

std::vector<symbol_t> mock_code_object_translator_t::get_symbols(size_t object_id) const
{
    if (const auto item = m_symbols_per_obj.find(object_id); item != m_symbols_per_obj.end())
    {
        return item->second;
    }
    return {};
}

instruction_t mock_code_object_translator_t::get_instruction(size_t, uint64_t) const
{
    return m_instruction;
}

uint64_t mock_code_object_translator_t::get_load_base(size_t object_id) const
{
    if (const auto item = m_load_base_per_obj.find(object_id); item != m_load_base_per_obj.end())
    {
        return item->second;
    }
    return 0;
}

void mock_code_object_translator_t::set_load_base(size_t object_id, uint64_t load_base)
{
    m_load_base_per_obj[object_id] = load_base;
}

void mock_code_object_translator_t::add_symbols(size_t object_id,
                                                const std::vector<rocprofiler_compute_tool::symbol_t>& symbols)
{
    m_symbols_per_obj[object_id] = symbols;
}

void mock_code_object_translator_t::add_instruction(const rocprofiler_compute_tool::instruction_t& instruction)
{
    m_instruction = instruction;
}

const std::vector<mock_code_object_translator_t::mem_code_object_info_t>&
    mock_code_object_translator_t::get_mem_code_object_info() const
{
    return m_mem_code_obj_info;
}

const std::vector<mock_code_object_translator_t::file_code_object_info_t>&
    mock_code_object_translator_t::get_file_code_object_info() const
{
    return m_file_code_obj_info;
}

void mock_code_object_writer_t::start_code_obj(size_t obj_id)
{
    m_started_code_obj_ids.push_back(obj_id);
}

void mock_code_object_writer_t::end_code_obj()
{
    ++m_ended_code_obj_count;
}

void mock_code_object_writer_t::start_symbol(const symbol_t& symbol)
{
    m_symbol_descriptions.push_back(symbol);
}

void mock_code_object_writer_t::end_symbol()
{
    ++m_end_symbol_count;
}

void mock_code_object_writer_t::write_instruction(const instruction_t& inst)
{
    m_instructions.push_back(inst);
}

std::string mock_code_object_writer_t::get_result()
{
    return {};
}

void mock_code_object_writer_t::flush(const std::filesystem::path& string) {}

const std::vector<size_t>& mock_code_object_writer_t::get_start_code_obj_ids() const
{
    return m_started_code_obj_ids;
}

uint32_t mock_code_object_writer_t::get_end_code_obj_count() const
{
    return m_ended_code_obj_count;
}

const std::vector<symbol_t>& mock_code_object_writer_t::get_symbol_descriptions() const
{
    return m_symbol_descriptions;
}

const std::vector<instruction_t>& mock_code_object_writer_t::get_instruction_descriptions() const
{
    return m_instructions;
}

uint32_t mock_code_object_writer_t::get_end_symbol_count() const
{
    return m_end_symbol_count;
}

void mock_ps_file_writer_t::add_stochastic_sample(int inst_index, const pc_sampling_record_t& rec)
{
    m_stochastic_samples.emplace_back(inst_index, rec);
}

void mock_ps_file_writer_t::add_host_trap_sample(int inst_index, const pc_sampling_record_t& rec)
{
    m_host_trap_samples.emplace_back(inst_index, rec);
}

void mock_ps_file_writer_t::set_instruction_strings(const std::vector<std::string>& instructions,
                                                    const std::vector<std::string>& comments)
{
    m_instructions = instructions;
    m_comments     = comments;
}

void mock_ps_file_writer_t::add_kernel_symbol(uint64_t           code_object_id,
                                              const std::string& formatted_kernel_name)
{
    m_kernel_symbols.emplace_back(code_object_id, formatted_kernel_name);
}

std::string mock_ps_file_writer_t::get_result()
{
    return {};
}

void mock_ps_file_writer_t::flush(const std::filesystem::path& output_file_path) {}

const std::vector<std::pair<int, pc_sampling_record_t>>& mock_ps_file_writer_t::get_stochastic_samples() const
{
    return m_stochastic_samples;
}

const std::vector<std::pair<int, pc_sampling_record_t>>& mock_ps_file_writer_t::get_host_trap_samples() const
{
    return m_host_trap_samples;
}

const std::vector<std::string>& mock_ps_file_writer_t::get_instruction_strings() const
{
    return m_instructions;
}

const std::vector<std::string>& mock_ps_file_writer_t::get_comment_strings() const
{
    return m_comments;
}

const std::vector<std::pair<uint64_t, std::string>>& mock_ps_file_writer_t::get_kernel_symbols() const
{
    return m_kernel_symbols;
}
