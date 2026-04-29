// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "code_object_writer.h"

#include "nlohmann/json.hpp"

using namespace rocm_compute;

void code_object_writer_json_t::start_code_obj(size_t obj_id)
{
    ++m_code_object_closure_count;
    m_code_object_ids.push_back(obj_id);
}

void code_object_writer_json_t::end_code_obj()
{
    --m_code_object_closure_count;
}

void code_object_writer_json_t::start_symbol(const symbol_t& symbol) {}

void code_object_writer_json_t::end_symbol() {}

void code_object_writer_json_t::write_instruction(const instruction_t& inst) {}

std::string code_object_writer_json_t::get_result()
{
    if (m_code_object_closure_count != 0)
    {
        throw std::runtime_error("Code object description is not properly closed");
    }

    auto code_objects = nlohmann::json::array();
    for (auto id : m_code_object_ids)
    {
        code_objects.push_back(nlohmann::json::object({{"id", id}}));
    }
    return nlohmann::json{{"code_objects", std::move(code_objects)}}.dump();
}
