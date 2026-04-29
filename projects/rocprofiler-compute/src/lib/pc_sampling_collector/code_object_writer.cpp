// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "code_object_writer.h"

#include "nlohmann/json.hpp"

using namespace rocm_compute;

void code_object_writer_json_t::start_code_obj(size_t obj_id)
{
    if (m_code_object_closure_count != 0)
    {
        throw std::runtime_error("Code object description is not properly closed");
    }
    ++m_code_object_closure_count;
    m_code_objects.push_back(code_object_entry_t{obj_id, {}});
}

void code_object_writer_json_t::end_code_obj()
{
    if (m_code_object_closure_count == 0)
    {
        throw std::runtime_error("Code object description is not properly opened");
    }
    if (m_symbol_closure_count != 0)
    {
        throw std::runtime_error("Symbol description is not properly closed");
    }
    --m_code_object_closure_count;
}

void code_object_writer_json_t::start_symbol(const symbol_t& symbol)
{
    if (m_code_object_closure_count == 0)
    {
        throw std::runtime_error("Symbol must be opened inside a code object");
    }
    if (m_symbol_closure_count != 0)
    {
        throw std::runtime_error("Symbol description is not properly closed");
    }
    ++m_symbol_closure_count;
    m_code_objects.back().symbols.push_back(symbol);
}

void code_object_writer_json_t::end_symbol()
{
    if (m_symbol_closure_count == 0)
    {
        throw std::runtime_error("Symbol description is not properly opened");
    }
    --m_symbol_closure_count;
}

void code_object_writer_json_t::write_instruction(const instruction_t& inst) {}

std::string code_object_writer_json_t::get_result()
{
    if (m_code_object_closure_count != 0)
    {
        throw std::runtime_error("Code object description is not properly closed");
    }
    if (m_symbol_closure_count != 0)
    {
        throw std::runtime_error("Symbol description is not properly closed");
    }

    auto code_objects = nlohmann::json::array();
    for (const auto& entry : m_code_objects)
    {
        auto symbols = nlohmann::json::array();
        for (const auto& s : entry.symbols)
        {
            symbols.push_back(nlohmann::json::object({
                {"name", s.name},
                {"code_object_offset", s.code_object_offset},
                {"virtual_address", s.virtual_address},
                {"size", s.size},
            }));
        }
        code_objects.push_back(nlohmann::json::object({
            {"id", entry.id},
            {"symbols", std::move(symbols)},
        }));
    }
    return nlohmann::json{{"code_objects", std::move(code_objects)}}.dump();
}
