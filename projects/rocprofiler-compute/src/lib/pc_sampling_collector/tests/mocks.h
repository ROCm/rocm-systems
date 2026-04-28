// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "code_object_translator.h"

#include <string>
#include <vector>

class mock_code_object_translator_t : public rocm_compute::code_object_translator_t
{
public:
    struct file_code_object_info_t
    {
        std::string filepath;
        size_t      id = 0;
        uint64_t    load_addr = 0;
        uint64_t    mem_size= 0;
    };

    struct mem_code_object_info_t
    {
        void*    data = nullptr;
        size_t   memory_size = 0;
        size_t   id = 0;
        uint64_t load_addr = 0;
        uint64_t mem_size = 0;
    };

    void add_code_object(const char* filepath, size_t id, uint64_t load_addr, uint64_t mem_size) override;
    void add_code_object(const void* data,
                         size_t      memory_size,
                         size_t      id,
                         uint64_t    load_addr,
                         uint64_t    mem_size) override;

    const std::vector<mem_code_object_info_t>& get_mem_code_object_info() const;
    const std::vector<file_code_object_info_t>& get_file_code_object_info() const;

private:
    std::vector<mem_code_object_info_t> m_mem_code_obj_info;
    std::vector<file_code_object_info_t> m_file_code_obj_info;
};
