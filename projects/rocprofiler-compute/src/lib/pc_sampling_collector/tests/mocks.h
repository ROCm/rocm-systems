// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "code_object_translator.h"
#include "code_object_writer.h"

#include <string>
#include <unordered_map>
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

    const std::vector<size_t>&          get_code_object_ids() const override;
    std::vector<rocm_compute::symbol_t> get_symbols(size_t object_id) const override;
    rocm_compute::instruction_t get_instruction(size_t object_id, uint64_t virtual_address) const override;

    void add_symbols(size_t object_id, const std::vector<rocm_compute::symbol_t>& symbols);
    void add_instruction(const rocm_compute::instruction_t& instruction);

    const std::vector<mem_code_object_info_t>&  get_mem_code_object_info() const;
    const std::vector<file_code_object_info_t>& get_file_code_object_info() const;

private:
    std::vector<mem_code_object_info_t>                             m_mem_code_obj_info;
    std::vector<file_code_object_info_t>                            m_file_code_obj_info;
    std::vector<size_t>                                             m_code_object_ids;
    std::unordered_map<size_t, std::vector<rocm_compute::symbol_t>> m_symbols_per_obj;
    rocm_compute::instruction_t m_instruction = {"", "", 0, 0, 1};
};

class mock_code_object_writer_t : public rocm_compute::code_object_writer_t
{
public:
    void        start_code_obj(size_t obj_id) override;
    void        end_code_obj_desc(size_t obj_id) override;
    void        start_symbol(const rocm_compute::symbol_t& symbol) override;
    void        end_symbol() override;
    void        write_instruction(const rocm_compute::instruction_t& inst) override;
    std::string get_result() override;

    const std::vector<size_t>&                      get_started_code_obj_ids() const;
    const std::vector<size_t>&                      get_ended_code_obj_desc_ids() const;
    const std::vector<rocm_compute::symbol_t>&      get_symbol_descriptions() const;
    const std::vector<rocm_compute::instruction_t>& get_instruction_descriptions() const;
    uint32_t                                        get_end_symbol_count() const;

private:
    std::vector<size_t>                      m_started_code_obj_ids;
    std::vector<size_t>                      m_ended_code_obj_desc_ids;
    std::vector<rocm_compute::symbol_t>      m_symbol_descriptions;
    std::vector<rocm_compute::instruction_t> m_instructions;
    uint32_t                                 m_end_symbol_count = 0;
};
