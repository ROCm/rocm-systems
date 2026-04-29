// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "mocks.h"

using namespace rocm_compute;

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

const std::vector<size_t>& mock_code_object_translator_t::get_code_object_ids() const
{
    return m_code_object_ids;
}

std::vector<obj_symbol_t> mock_code_object_translator_t::get_symbol_map(size_t object_id) const
{
    return {};
}

instruction_t mock_code_object_translator_t::get_instruction(size_t object_id, uint64_t virtual_address) const
{
    return {};
}

void mock_pc_samples_writer_t::start_code_obj_desc(const obj_symbol_t& desc)
{
    m_obj_descriptions.push_back(desc);
}

void mock_pc_samples_writer_t::end_code_obj_desc()
{
    ++m_end_code_obj_desc_count;
}

std::string mock_pc_samples_writer_t::get_result()
{
    return {};
}

const std::vector<obj_symbol_t>& mock_pc_samples_writer_t::get_obj_descriptions() const
{
    return m_obj_descriptions;
}

uint32_t mock_pc_samples_writer_t::get_end_code_obj_desc_count() const
{
    return m_end_code_obj_desc_count;
}
