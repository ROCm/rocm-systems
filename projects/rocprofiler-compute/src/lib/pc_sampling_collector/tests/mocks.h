// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "code_object_translator.h"
#include "code_object_writer.h"
#include "filesystem_wrapper.h"
#include "sdk_wrapper.h"

#include <filesystem>
#include <map>
#include <string>
#include <system_error>
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

    void add_symbols(size_t object_id, const std::vector<rocprofiler_compute_tool::symbol_t>& symbols);
    void add_instruction(const rocprofiler_compute_tool::instruction_t& instruction);
    void set_instruction(size_t                                         object_id,
                         uint64_t                                       virtual_address,
                         const rocprofiler_compute_tool::instruction_t& instruction);

    const std::vector<mem_code_object_info_t>&      get_mem_code_object_info() const;
    const std::vector<file_code_object_info_t>&     get_file_code_object_info() const;
    const std::vector<std::pair<size_t, uint64_t>>& get_instruction_requests() const;

private:
    std::vector<mem_code_object_info_t>  m_mem_code_obj_info;
    std::vector<file_code_object_info_t> m_file_code_obj_info;
    std::vector<size_t>                  m_code_object_ids;
    std::unordered_map<size_t, std::vector<rocprofiler_compute_tool::symbol_t>> m_symbols_per_obj;
    std::map<std::pair<size_t, uint64_t>, rocprofiler_compute_tool::instruction_t> m_instructions;
    rocprofiler_compute_tool::instruction_t          m_instruction = {"", "", 0, 0, 1};
    mutable std::vector<std::pair<size_t, uint64_t>> m_instruction_requests;
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
    bool        empty() const override;

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

class mock_sdk_wrapper_t : public rocprofiler_compute_tool::sdk_wrapper_t
{
public:
    std::string_view source_frame_separator() const override;

    void set_source_frame_separator(std::string source_frame_separator);

private:
    std::string m_source_frame_separator = " -> ";
};

class mock_filesystem_wrapper_t : public rocprofiler_compute_tool::filesystem_wrapper_t
{
public:
    struct status_response_t
    {
        std::filesystem::file_status status;
        std::error_code              error;
    };

    struct absolute_response_t
    {
        std::filesystem::path result;
        std::error_code       error;
    };

    struct weakly_canonical_response_t
    {
        std::filesystem::path result;
        std::error_code       error;
    };

    struct copy_file_call_t
    {
        std::filesystem::path         source;
        std::filesystem::path         destination;
        std::filesystem::copy_options options = std::filesystem::copy_options::none;
    };

    std::filesystem::path absolute(const std::filesystem::path& path, std::error_code& error) override;
    std::filesystem::file_status status(const std::filesystem::path& path, std::error_code& error) override;
    bool create_directories(const std::filesystem::path& path, std::error_code& error) override;
    bool copy_file(const std::filesystem::path&  source,
                   const std::filesystem::path&  destination,
                   std::filesystem::copy_options options,
                   std::error_code&              error) override;
    bool exists(const std::filesystem::file_status& status) override;
    bool is_regular_file(const std::filesystem::file_status& status) override;
    bool has_parent_path(const std::filesystem::path& path) override;
    std::filesystem::path parent_path(const std::filesystem::path& path) override;
    std::filesystem::path weakly_canonical(const std::filesystem::path& path,
                                           std::error_code&             error) override;
    std::filesystem::path relative_path(const std::filesystem::path& path) override;

    void set_absolute(const std::filesystem::path& path, const std::filesystem::path& result);
    void set_absolute_error(const std::filesystem::path& path, std::error_code error);
    void set_status(const std::filesystem::path& path, std::filesystem::file_status status);
    void set_status_error(const std::filesystem::path& path, std::error_code error);
    void set_weakly_canonical(const std::filesystem::path& path, const std::filesystem::path& result);
    void set_weakly_canonical_error(const std::filesystem::path& path, std::error_code error);
    void set_create_directories_error(std::error_code error);
    void set_copy_file_error(std::error_code error);

    const std::vector<std::filesystem::path>& get_absolute_calls() const;
    const std::vector<std::filesystem::path>& get_status_calls() const;
    const std::vector<std::filesystem::path>& get_has_parent_path_calls() const;
    const std::vector<std::filesystem::path>& get_weakly_canonical_calls() const;
    const std::vector<std::filesystem::path>& get_relative_path_calls() const;
    const std::vector<std::filesystem::path>& get_create_directories_calls() const;
    const std::vector<copy_file_call_t>&      get_copy_file_calls() const;

private:
    std::map<std::filesystem::path, absolute_response_t>         m_absolute_responses;
    std::map<std::filesystem::path, status_response_t>           m_status_responses;
    std::map<std::filesystem::path, weakly_canonical_response_t> m_weakly_canonical_responses;
    std::error_code                                              m_create_directories_error;
    std::error_code                                              m_copy_file_error;
    std::vector<std::filesystem::path>                           m_absolute_calls;
    std::vector<std::filesystem::path>                           m_status_calls;
    std::vector<std::filesystem::path>                           m_has_parent_path_calls;
    std::vector<std::filesystem::path>                           m_weakly_canonical_calls;
    std::vector<std::filesystem::path>                           m_relative_path_calls;
    std::vector<std::filesystem::path>                           m_create_directories_calls;
    std::vector<copy_file_call_t>                                m_copy_file_calls;
};
