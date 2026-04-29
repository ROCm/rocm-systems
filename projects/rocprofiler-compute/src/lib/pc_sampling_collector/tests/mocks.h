// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "code_object_translator.h"
#include "pc_samples_writer.h"

#include <string>
#include <vector>

class mock_code_object_translator_t : public rocm_compute::code_object_translator_t
{
public:
    struct file_code_object_info_t
    {
        std::string filepath;
        size_t      id        = 0;
        uint64_t    load_base = 0;
        uint64_t    load_size = 0;
    };

    struct mem_code_object_info_t
    {
        uint64_t memory_base = 0;
        size_t   memory_size = 0;
        size_t   id          = 0;
        uint64_t load_base   = 0;
        uint64_t load_size   = 0;
    };

    void add_code_object(const char* filepath, size_t id, uint64_t load_addr, uint64_t load_size) override;
    void add_code_object(uint64_t memory_base,
                         size_t   memory_size,
                         size_t   id,
                         uint64_t load_base,
                         uint64_t load_size) override;

    const std::vector<mem_code_object_info_t>&  get_mem_code_object_info() const;
    const std::vector<file_code_object_info_t>& get_file_code_object_info() const;
    std::vector<size_t>                         get_code_object_ids() const override;

private:
    std::vector<mem_code_object_info_t>  m_mem_code_obj_info;
    std::vector<file_code_object_info_t> m_file_code_obj_info;
    std::vector<size_t>                  m_code_object_ids;
};

class mock_pc_samples_writer_t : public rocm_compute::pc_samples_writer_t
{
public:
    void        write() override;
    std::string get_result() override;

    uint32_t get_write_count();
private:
    uint32_t m_write_count = 0;
};
