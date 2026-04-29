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
    const std::vector<size_t>&                  get_code_object_ids() const override;
    std::vector<rocm_compute::obj_symbol_t>     get_symbol_map(size_t object_id) const override;
    rocm_compute::instruction_t get_instruction(size_t   object_id,
                                                uint64_t virtual_address) const override;

private:
    std::vector<mem_code_object_info_t>  m_mem_code_obj_info;
    std::vector<file_code_object_info_t> m_file_code_obj_info;
    std::vector<size_t>                  m_code_object_ids;
};

class mock_pc_samples_writer_t : public rocm_compute::pc_samples_writer_t
{
public:
    void        start_code_obj_desc(const rocm_compute::obj_symbol_t& desc) override;
    void        end_code_obj_desc() override;
    std::string get_result() override;

    const std::vector<rocm_compute::obj_symbol_t>& get_obj_descriptions() const;
    uint32_t get_end_code_obj_desc_count() const;
private:
    std::vector<rocm_compute::obj_symbol_t> m_obj_descriptions;
    uint32_t m_end_code_obj_desc_count = 0;
};
