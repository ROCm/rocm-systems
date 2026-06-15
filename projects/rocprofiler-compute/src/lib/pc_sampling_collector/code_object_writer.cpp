// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "code_object_writer.h"

#include "file_writer.h"
#include "gsl_assert.h"

#include <iostream>

using namespace rocprofiler_compute_tool;

std::shared_ptr<code_object_writer_t> code_object_writer_t::create()
{
    return std::make_shared<code_object_writer_json_t>();
}

void code_object_writer_json_t::start_code_obj(size_t obj_id)
{
    Expects(m_code_object_closure_count == 0);
    ++m_code_object_closure_count;

    m_current_obj_id = obj_id;
}

void code_object_writer_json_t::end_code_obj()
{
    Expects(m_code_object_closure_count != 0);
    Expects(m_symbol_closure_count == 0);
    --m_code_object_closure_count;

    m_code_objects.push_back(nlohmann::json::object({
        {"id", m_current_obj_id},
        {"symbols", std::move(m_symbols)},
    }));
}

void code_object_writer_json_t::start_symbol(const symbol_t& symbol)
{
    Expects(m_code_object_closure_count != 0);
    Expects(m_symbol_closure_count == 0);
    ++m_symbol_closure_count;

    m_current_symbol = symbol;
}

void code_object_writer_json_t::end_symbol()
{
    Expects(m_symbol_closure_count != 0);
    --m_symbol_closure_count;

    m_symbols.push_back(nlohmann::json::object({
        {"name", m_current_symbol.name},
        {"code_object_offset", m_current_symbol.code_object_offset},
        {"virtual_address", m_current_symbol.virtual_address},
        {"size", m_current_symbol.size},
        {"instructions", std::move(m_instructions)},
    }));
}

void code_object_writer_json_t::write_instruction(const instruction_t& inst)
{
    Expects(m_code_object_closure_count != 0);
    Expects(m_symbol_closure_count != 0);

    m_instructions.push_back(nlohmann::json::object({
        {"name", inst.name},
        {"comment", inst.comment},
        {"virtual_address", inst.virtual_address},
        {"code_obj_offset", inst.code_obj_offset},
        {"size", inst.size},
    }));
}

std::string code_object_writer_json_t::get_result()
{
    Expects(m_code_object_closure_count == 0);
    Expects(m_symbol_closure_count == 0);

    return nlohmann::json{{"code_objects", std::move(m_code_objects)}}.dump();
}

void code_object_writer_json_t::flush(const std::filesystem::path& output_file_path)
{
    file_writer_t::create()->write(output_file_path, get_result());
    std::clog << "[rocprofiler-compute] [" << __FUNCTION__
              << "] Code object data has been written to: " << output_file_path << "\n";
}
