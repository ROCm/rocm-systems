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

instruction_t mock_code_object_translator_t::get_instruction(size_t object_id, uint64_t virtual_address) const
{
    m_instruction_requests.emplace_back(object_id, virtual_address);
    if (const auto item = m_instructions.find({object_id, virtual_address}); item != m_instructions.end())
    {
        return item->second;
    }
    return m_instruction;
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

void mock_code_object_translator_t::set_instruction(size_t   object_id,
                                                    uint64_t virtual_address,
                                                    const rocprofiler_compute_tool::instruction_t& instruction)
{
    m_instructions[{object_id, virtual_address}] = instruction;
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

const std::vector<std::pair<size_t, uint64_t>>& mock_code_object_translator_t::get_instruction_requests() const
{
    return m_instruction_requests;
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

bool mock_code_object_writer_t::empty() const
{
    return m_started_code_obj_ids.empty();
}

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

std::string_view mock_sdk_wrapper_t::source_frame_separator() const
{
    return m_source_frame_separator;
}

void mock_sdk_wrapper_t::set_source_frame_separator(std::string source_frame_separator)
{
    m_source_frame_separator = std::move(source_frame_separator);
}

std::filesystem::path mock_filesystem_wrapper_t::absolute(const std::filesystem::path& path,
                                                          std::error_code&             error)
{
    m_absolute_calls.push_back(path);
    if (const auto item = m_absolute_responses.find(path); item != m_absolute_responses.end())
    {
        error = item->second.error;
        return item->second.result;
    }

    error.clear();
    return path;
}

std::filesystem::file_status mock_filesystem_wrapper_t::status(const std::filesystem::path& path,
                                                               std::error_code&             error)
{
    m_status_calls.push_back(path);
    if (const auto item = m_status_responses.find(path); item != m_status_responses.end())
    {
        error = item->second.error;
        return item->second.status;
    }

    error.clear();
    return std::filesystem::file_status{std::filesystem::file_type::not_found};
}

bool mock_filesystem_wrapper_t::create_directories(const std::filesystem::path& path,
                                                   std::error_code&             error)
{
    m_create_directories_calls.push_back(path);
    error = m_create_directories_error;
    return !error;
}

bool mock_filesystem_wrapper_t::copy_file(const std::filesystem::path&  source,
                                          const std::filesystem::path&  destination,
                                          std::filesystem::copy_options options,
                                          std::error_code&              error)
{
    m_copy_file_calls.push_back({source, destination, options});
    error = m_copy_file_error;
    return !error;
}

bool mock_filesystem_wrapper_t::exists(const std::filesystem::file_status& status)
{
    return std::filesystem::exists(status);
}

bool mock_filesystem_wrapper_t::is_regular_file(const std::filesystem::file_status& status)
{
    return std::filesystem::is_regular_file(status);
}

bool mock_filesystem_wrapper_t::has_parent_path(const std::filesystem::path& path)
{
    m_has_parent_path_calls.push_back(path);
    return path.has_parent_path();
}

std::filesystem::path mock_filesystem_wrapper_t::parent_path(const std::filesystem::path& path)
{
    return path.parent_path();
}

std::filesystem::path mock_filesystem_wrapper_t::weakly_canonical(const std::filesystem::path& path,
                                                                  std::error_code& error)
{
    m_weakly_canonical_calls.push_back(path);
    if (const auto item = m_weakly_canonical_responses.find(path);
        item != m_weakly_canonical_responses.end())
    {
        error = item->second.error;
        return item->second.result;
    }

    return std::filesystem::weakly_canonical(path, error);
}

std::filesystem::path mock_filesystem_wrapper_t::relative_path(const std::filesystem::path& path)
{
    m_relative_path_calls.push_back(path);
    return path.relative_path();
}

void mock_filesystem_wrapper_t::set_absolute(const std::filesystem::path& path,
                                             const std::filesystem::path& result)
{
    m_absolute_responses[path] = {result, std::error_code{}};
}

void mock_filesystem_wrapper_t::set_absolute_error(const std::filesystem::path& path, std::error_code error)
{
    m_absolute_responses[path] = {std::filesystem::path{}, error};
}

void mock_filesystem_wrapper_t::set_status(const std::filesystem::path& path,
                                           std::filesystem::file_status status)
{
    m_status_responses[path] = {status, std::error_code{}};
}

void mock_filesystem_wrapper_t::set_status_error(const std::filesystem::path& path, std::error_code error)
{
    m_status_responses[path] = {std::filesystem::file_status{}, error};
}

void mock_filesystem_wrapper_t::set_weakly_canonical(const std::filesystem::path& path,
                                                     const std::filesystem::path& result)
{
    m_weakly_canonical_responses[path] = {result, std::error_code{}};
}

void mock_filesystem_wrapper_t::set_weakly_canonical_error(const std::filesystem::path& path,
                                                           std::error_code              error)
{
    m_weakly_canonical_responses[path] = {std::filesystem::path{}, error};
}

void mock_filesystem_wrapper_t::set_create_directories_error(std::error_code error)
{
    m_create_directories_error = error;
}

void mock_filesystem_wrapper_t::set_copy_file_error(std::error_code error)
{
    m_copy_file_error = error;
}

const std::vector<std::filesystem::path>& mock_filesystem_wrapper_t::get_absolute_calls() const
{
    return m_absolute_calls;
}

const std::vector<std::filesystem::path>& mock_filesystem_wrapper_t::get_status_calls() const
{
    return m_status_calls;
}

const std::vector<std::filesystem::path>& mock_filesystem_wrapper_t::get_has_parent_path_calls() const
{
    return m_has_parent_path_calls;
}

const std::vector<std::filesystem::path>& mock_filesystem_wrapper_t::get_weakly_canonical_calls() const
{
    return m_weakly_canonical_calls;
}

const std::vector<std::filesystem::path>& mock_filesystem_wrapper_t::get_relative_path_calls() const
{
    return m_relative_path_calls;
}

const std::vector<std::filesystem::path>& mock_filesystem_wrapper_t::get_create_directories_calls() const
{
    return m_create_directories_calls;
}

const std::vector<mock_filesystem_wrapper_t::copy_file_call_t>& mock_filesystem_wrapper_t::get_copy_file_calls() const
{
    return m_copy_file_calls;
}
