// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "code_object_translator.h"
#include "code_object_writer.h"
#include "pc_sampling_record.h"
#include "ps_file_writer.h"

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

class mock_code_object_translator_t : public rocprofiler_compute_tool::code_object_translator_t
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

    const std::vector<size_t>&                      get_code_object_ids() const override;
    std::vector<rocprofiler_compute_tool::symbol_t> get_symbols(size_t object_id) const override;
    rocprofiler_compute_tool::instruction_t get_instruction(size_t object_id,
                                                            uint64_t virtual_address) const override;
    uint64_t get_load_base(size_t object_id) const override;

    void add_symbols(size_t object_id, const std::vector<rocprofiler_compute_tool::symbol_t>& symbols);
    void add_instruction(const rocprofiler_compute_tool::instruction_t& instruction);
    void set_load_base(size_t object_id, uint64_t load_base);

    const std::vector<mem_code_object_info_t>&  get_mem_code_object_info() const;
    const std::vector<file_code_object_info_t>& get_file_code_object_info() const;

private:
    std::vector<mem_code_object_info_t>  m_mem_code_obj_info;
    std::vector<file_code_object_info_t> m_file_code_obj_info;
    std::vector<size_t>                  m_code_object_ids;
    std::unordered_map<size_t, std::vector<rocprofiler_compute_tool::symbol_t>> m_symbols_per_obj;
    rocprofiler_compute_tool::instruction_t m_instruction = {"", "", 0, 0, 1};
    std::unordered_map<size_t, uint64_t>    m_load_base_per_obj;
};

class mock_code_object_writer_t : public rocprofiler_compute_tool::code_object_writer_t
{
public:
    void        start_code_obj(size_t obj_id) override;
    void        end_code_obj() override;
    void        start_symbol(const rocprofiler_compute_tool::symbol_t& symbol) override;
    void        end_symbol() override;
    void        write_instruction(const rocprofiler_compute_tool::instruction_t& inst) override;
    std::string get_result() override;
    void        flush(const std::filesystem::path& string) override;

    const std::vector<size_t>&                             get_start_code_obj_ids() const;
    uint32_t                                               get_end_code_obj_count() const;
    const std::vector<rocprofiler_compute_tool::symbol_t>& get_symbol_descriptions() const;
    const std::vector<rocprofiler_compute_tool::instruction_t>& get_instruction_descriptions() const;
    uint32_t get_end_symbol_count() const;

private:
    std::vector<size_t>                                  m_started_code_obj_ids;
    uint32_t                                             m_ended_code_obj_count = 0;
    std::vector<rocprofiler_compute_tool::symbol_t>      m_symbol_descriptions;
    std::vector<rocprofiler_compute_tool::instruction_t> m_instructions;
    uint32_t                                             m_end_symbol_count = 0;
};

class mock_ps_file_writer_t : public rocprofiler_compute_tool::ps_file_writer_t
{
public:
    void add_stochastic_sample(int                                                   inst_index,
                               const rocprofiler_compute_tool::pc_sampling_record_t& rec) override;
    void add_host_trap_sample(int                                                   inst_index,
                              const rocprofiler_compute_tool::pc_sampling_record_t& rec) override;
    void set_instruction_strings(const std::vector<std::string>& instructions,
                                 const std::vector<std::string>& comments) override;
    void add_kernel_symbol(uint64_t code_object_id, const std::string& formatted_kernel_name) override;
    std::string get_result() override;
    void        flush(const std::filesystem::path& output_file_path) override;

    const std::vector<std::pair<int, rocprofiler_compute_tool::pc_sampling_record_t>>& get_stochastic_samples() const;
    const std::vector<std::pair<int, rocprofiler_compute_tool::pc_sampling_record_t>>& get_host_trap_samples() const;
    const std::vector<std::string>&                      get_instruction_strings() const;
    const std::vector<std::string>&                      get_comment_strings() const;
    const std::vector<std::pair<uint64_t, std::string>>& get_kernel_symbols() const;

private:
    std::vector<std::pair<int, rocprofiler_compute_tool::pc_sampling_record_t>> m_stochastic_samples;
    std::vector<std::pair<int, rocprofiler_compute_tool::pc_sampling_record_t>> m_host_trap_samples;
    std::vector<std::string>                                                    m_instructions;
    std::vector<std::string>                                                    m_comments;
    std::vector<std::pair<uint64_t, std::string>>                               m_kernel_symbols;
};
