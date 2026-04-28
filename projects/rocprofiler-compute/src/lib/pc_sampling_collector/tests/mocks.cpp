// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "mocks.h"

void mock_code_object_translator_t::add_code_object(const char* filepath,
                                                    size_t      id,
                                                    uint64_t    load_addr,
                                                    uint64_t    load_size)
{
    m_file_code_obj_info.push_back({filepath, id, load_addr, load_size});
}

void mock_code_object_translator_t::add_code_object(uint64_t memory_base,
                                                    size_t   memory_size,
                                                    size_t   id,
                                                    uint64_t load_base,
                                                    uint64_t load_size)
{
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