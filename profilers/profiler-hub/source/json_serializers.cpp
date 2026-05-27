// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "json_serializers.hpp"

#include "common/string_conversions.hpp"

#include <nlohmann/json.hpp>
#include <string>

namespace profiler_hub::json_serializers
{

json_t
to_json(const profiler_hub::shared_types::address_range_info_t& addr_range)
{
    json_t j;
    j["address_base"] = addr_range.address_base;
    j["address_low"]  = addr_range.address_low;
    j["address_high"] = addr_range.address_high;
    if(is_valid_string_view(addr_range.extdata))
    {
        j["extdata"] = std::string(addr_range.extdata);
    }
    return j;
}

json_t
to_json(const profiler_hub::shared_types::program_counter_info_t& pc_info)
{
    json_t j;
    if(is_valid_optional_string_view(pc_info.function))
    {
        j["function"] = std::string(pc_info.function.value());
    }
    if(is_valid_optional_string_view(pc_info.filename))
    {
        j["filename"] = std::string(pc_info.filename.value());
    }
    if(pc_info.line_number.has_value())
    {
        j["line_number"] = pc_info.line_number.value();
    }
    if(is_valid_string_view(pc_info.extdata))
    {
        j["extdata"] = std::string(pc_info.extdata);
    }
    return j;
}

json_t
to_json(const profiler_hub::shared_types::stack_frame_t& frame)
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
    if(is_valid_string_view(frame.extdata))
    {
        j["extdata"] = std::string(frame.extdata);
    }
    return j;
}

std::string
serialize_call_stack(const profiler_hub::shared_types::call_stack_t& call_stack)
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
to_json(const profiler_hub::shared_types::source_code_info_t& source_code)
{
    json_t j;
    if(is_valid_optional_string_view(source_code.filename))
    {
        j["filename"] = std::string(source_code.filename.value());
    }
    if(source_code.starting_line_number.has_value())
    {
        j["starting_line_number"] = source_code.starting_line_number.value();
    }
    if(!source_code.source_code_lines.empty())
    {
        json_t lines = json_t::array();
        for(const auto& line : source_code.source_code_lines)
        {
            if(!line.empty())
            {
                lines.push_back(std::string(line));
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
        for(const auto& line : source_code.assembly_instruction_lines)
        {
            if(!line.empty())
            {
                asm_lines.push_back(std::string(line));
            }
        }
        if(!asm_lines.empty())
        {
            j["assembly_instruction_lines"] = asm_lines;
        }
    }
    if(is_valid_string_view(source_code.extdata))
    {
        j["extdata"] = std::string(source_code.extdata);
    }
    return j;
}

json_t
to_json(const profiler_hub::shared_types::line_info_entry_t& line_info)
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
serialize_source_context(
    const profiler_hub::shared_types::source_context_list_t& line_info_list)
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

namespace
{

reader_types::address_range_info_t
parse_address_range(const json_t& j)
{
    reader_types::address_range_info_t addr;
    addr.address_base = j.value("address_base", size_t{ 0 });
    addr.address_low  = j.value("address_low", size_t{ 0 });
    addr.address_high = j.value("address_high", size_t{ 0 });
    if(j.contains("extdata") && j["extdata"].is_string())
    {
        addr.extdata = j["extdata"].get<std::string>();
    }
    return addr;
}

reader_types::program_counter_info_t
parse_program_counter(const json_t& j)
{
    reader_types::program_counter_info_t pc;
    if(j.contains("function") && j["function"].is_string())
    {
        pc.function = j["function"].get<std::string>();
    }
    if(j.contains("filename") && j["filename"].is_string())
    {
        pc.filename = j["filename"].get<std::string>();
    }
    if(j.contains("line_number") && j["line_number"].is_number())
    {
        pc.line_number = j["line_number"].get<size_t>();
    }
    if(j.contains("extdata") && j["extdata"].is_string())
    {
        pc.extdata = j["extdata"].get<std::string>();
    }
    return pc;
}

reader_types::stack_frame_t
parse_stack_frame(const json_t& j)
{
    reader_types::stack_frame_t frame;
    if(j.contains("program_counter") && j["program_counter"].is_object())
    {
        frame.program_counter = parse_program_counter(j["program_counter"]);
    }
    if(j.contains("address_range") && j["address_range"].is_object())
    {
        frame.address_range = parse_address_range(j["address_range"]);
    }
    if(j.contains("extdata") && j["extdata"].is_string())
    {
        frame.extdata = j["extdata"].get<std::string>();
    }
    return frame;
}

reader_types::source_code_info_t
parse_source_code(const json_t& j)
{
    reader_types::source_code_info_t sc;
    if(j.contains("filename") && j["filename"].is_string())
    {
        sc.filename = j["filename"].get<std::string>();
    }
    if(j.contains("starting_line_number") && j["starting_line_number"].is_number())
    {
        sc.starting_line_number = j["starting_line_number"].get<size_t>();
    }
    if(j.contains("source_code_lines") && j["source_code_lines"].is_array())
    {
        for(const auto& line : j["source_code_lines"])
        {
            if(line.is_string())
            {
                sc.source_code_lines.push_back(line.get<std::string>());
            }
        }
    }
    if(j.contains("assembly_instruction_lines") &&
       j["assembly_instruction_lines"].is_array())
    {
        for(const auto& line : j["assembly_instruction_lines"])
        {
            if(line.is_string())
            {
                sc.assembly_instruction_lines.push_back(line.get<std::string>());
            }
        }
    }
    if(j.contains("extdata") && j["extdata"].is_string())
    {
        sc.extdata = j["extdata"].get<std::string>();
    }
    return sc;
}

reader_types::line_info_entry_t
parse_line_info_entry(const json_t& j)
{
    reader_types::line_info_entry_t entry;
    if(j.contains("source_code") && j["source_code"].is_object())
    {
        entry.source_code = parse_source_code(j["source_code"]);
    }
    if(j.contains("program_counter") && j["program_counter"].is_object())
    {
        entry.program_counter = parse_program_counter(j["program_counter"]);
    }
    if(j.contains("address_range") && j["address_range"].is_object())
    {
        entry.address_range = parse_address_range(j["address_range"]);
    }
    return entry;
}

}  // namespace

reader_types::call_stack_t
deserialize_call_stack(const std::string& json)
{
    reader_types::call_stack_t result;

    if(json.empty() || json == "{}")
    {
        return result;
    }

    try
    {
        auto j = json_t::parse(json);
        if(!j.contains("frames") || !j["frames"].is_array())
        {
            return result;
        }

        for(const auto& frame_json : j["frames"])
        {
            if(frame_json.is_object())
            {
                result.push_back(parse_stack_frame(frame_json));
            }
        }
    } catch(const json_t::exception&)
    {
        return {};
    }

    return result;
}

reader_types::source_context_list_t
deserialize_source_context(const std::string& json)
{
    reader_types::source_context_list_t result;

    if(json.empty() || json == "{}")
    {
        return result;
    }

    try
    {
        auto j = json_t::parse(json);
        if(!j.contains("entries") || !j["entries"].is_array())
        {
            return result;
        }

        for(const auto& entry_json : j["entries"])
        {
            if(entry_json.is_object())
            {
                result.push_back(parse_line_info_entry(entry_json));
            }
        }
    } catch(const json_t::exception&)
    {
        return {};
    }

    return result;
}

}  // namespace profiler_hub::json_serializers
