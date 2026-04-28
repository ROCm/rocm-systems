// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "mocks.h"

void mock_code_object_translator_t::add_code_object(const char* filepath,
                                                    size_t      id,
                                                    uint64_t    load_addr,
                                                    uint64_t    mem_size)
{
    m_file_code_obj_info.push_back({filepath, id, load_addr, mem_size});
}

void mock_code_object_translator_t::add_code_object(const void* data,
                                                    size_t      memory_size,
                                                    size_t      id,
                                                    uint64_t    load_addr,
                                                    uint64_t    mem_size)
{
    m_mem_code_obj_info.push_back({const_cast<void*>(data), memory_size, id, load_addr, mem_size});
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