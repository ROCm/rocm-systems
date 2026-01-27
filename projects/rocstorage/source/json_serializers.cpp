// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "json_serializers.hpp"

#include "common/string_conversions.hpp"

#include <nlohmann/json.hpp>

namespace rocstorage::json_serializers
{

json_t
to_json(const writer_api::address_range_info_t& addr_range)
{
    json_t j;
    j["address_base"] = addr_range.address_base;
    j["address_low"]  = addr_range.address_low;
    j["address_high"] = addr_range.address_high;
    if(is_valid_cstring(addr_range.extdata))
    {
        j["extdata"] = addr_range.extdata;
    }
    return j;
}

json_t
to_json(const writer_api::program_counter_info_t& pc_info)
{
    json_t j;
    if(is_valid_cstring(pc_info.function))
    {
        j["function"] = pc_info.function;
    }
    if(is_valid_cstring(pc_info.filename))
    {
        j["filename"] = pc_info.filename;
    }
    if(pc_info.line_number.has_value())
    {
        j["line_number"] = pc_info.line_number.value();
    }
    if(is_valid_cstring(pc_info.extdata))
    {
        j["extdata"] = pc_info.extdata;
    }
    return j;
}

json_t
to_json(const writer_api::stack_frame_t& frame)
{
    json_t j;
    if(frame.program_counter.has_value())
    {
        j["program_counter"] = to_json(frame.program_counter.value());
    }
    if(frame.address_range.has_value())
    {
        j["address_range"] = to_json(frame.address_range.value());
    }
    if(is_valid_cstring(frame.extdata))
    {
        j["extdata"] = frame.extdata;
    }
    return j;
}

std::string
serialize_call_stack(const writer_api::call_stack_t& call_stack)
{
    if(call_stack.empty())
    {
        return "{}";
    }

    json_t j;
    json_t frames = json_t::array();

    for(const auto& frame : call_stack)
    {
        frames.push_back(to_json(frame));
    }

    j["frames"] = frames;
    return j.dump();
}

json_t
to_json(const writer_api::source_code_info_t& source_code)
{
    json_t j;
    if(source_code.filename.has_value() && is_valid_cstring(source_code.filename.value()))
    {
        j["filename"] = source_code.filename.value();
    }
    if(source_code.starting_line_number.has_value())
    {
        j["starting_line_number"] = source_code.starting_line_number.value();
    }
    if(!source_code.source_code_lines.empty())
    {
        json_t lines = json_t::array();
        for(const auto* line : source_code.source_code_lines)
        {
            if(line != nullptr)
            {
                lines.push_back(line);
            }
        }
        if(!lines.empty())
        {
            j["source_code_lines"] = lines;
        }
    }
    if(!source_code.assembly_instruction_lines.empty())
    {
        json_t asm_lines = json_t::array();
        for(const auto* line : source_code.assembly_instruction_lines)
        {
            if(line != nullptr)
            {
                asm_lines.push_back(line);
            }
        }
        if(!asm_lines.empty())
        {
            j["assembly_instruction_lines"] = asm_lines;
        }
    }
    if(is_valid_cstring(source_code.extdata))
    {
        j["extdata"] = source_code.extdata;
    }
    return j;
}

json_t
to_json(const writer_api::line_info_entry_t& line_info)
{
    json_t j;
    if(line_info.source_code.has_value())
    {
        j["source_code"] = to_json(line_info.source_code.value());
    }
    if(line_info.program_counter.has_value())
    {
        j["program_counter"] = to_json(line_info.program_counter.value());
    }
    if(line_info.address_range.has_value())
    {
        j["address_range"] = to_json(line_info.address_range.value());
    }
    return j;
}

std::string
serialize_source_context(const writer_api::source_context_list_t& line_info_list)
{
    if(line_info_list.empty())
    {
        return "{}";
    }

    json_t j;
    json_t entries = json_t::array();

    for(const auto& entry : line_info_list)
    {
        entries.push_back(to_json(entry));
    }

    j["entries"] = entries;
    return j.dump();
}

}  // namespace rocstorage::json_serializers
