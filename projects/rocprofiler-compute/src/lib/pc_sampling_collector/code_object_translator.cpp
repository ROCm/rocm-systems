// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "code_object_translator.h"

#include "gsl_assert.h"
#include "rocprofiler-sdk/cxx/codeobj/code_printing.hpp"

#include <algorithm>
#include <cctype>

using namespace rocprofiler_compute_tool;

namespace
{
constexpr std::string_view source_frame_separator = " -> ";

bool is_source_line_token(std::string_view token)
{
    if (token == "?")
        return true;

    return !token.empty() &&
           std::all_of(token.cbegin(),
                       token.cend(),
                       [](char ch) { return std::isdigit(static_cast<unsigned char>(ch)) != 0; });
}

std::string_view path_from_source_frame(std::string_view frame)
{
    const auto separator_position = frame.rfind(':');
    if (separator_position == std::string_view::npos)
        return frame;

    const auto line_token = frame.substr(separator_position + 1);
    if (!is_source_line_token(line_token))
        return frame;

    return frame.substr(0, separator_position);
}
}  // namespace

code_object_translator_impl_t::code_object_translator_impl_t()
    : m_translator(std::make_unique<rocprofiler::sdk::codeobj::disassembly::CodeobjAddressTranslate>())
{
}

code_object_translator_impl_t::~code_object_translator_impl_t() = default;

void code_object_translator_impl_t::add_code_object(const char* filepath,
                                                    size_t      id,
                                                    uint64_t    load_addr,
                                                    uint64_t    mem_size)
{
    m_translator->addDecoder(filepath, id, load_addr, mem_size);
    m_obj_id_to_load_addr[id] = load_addr;
    m_obj_ids.push_back(id);
}

void code_object_translator_impl_t::add_code_object(uint64_t memory_base,
                                                    size_t   memory_size,
                                                    size_t   id,
                                                    uint64_t load_base,
                                                    uint64_t load_size)
{
    m_translator->addDecoder(reinterpret_cast<void*>(memory_base), memory_size, id, load_base, load_size);
    m_obj_id_to_load_addr[id] = load_base;
    m_obj_ids.push_back(id);
}

const std::vector<size_t>& code_object_translator_impl_t::get_code_object_ids() const
{
    return m_obj_ids;
}

std::vector<symbol_t> code_object_translator_impl_t::get_symbols(size_t object_id) const
{
    Expects(m_obj_id_to_load_addr.find(object_id) != m_obj_id_to_load_addr.end());
    const auto&           symbols      = m_translator->getSymbolMap(object_id);
    const auto&           load_address = m_obj_id_to_load_addr.at(object_id);
    std::vector<symbol_t> symbol_map;
    for (const auto& [virtual_address, symbol_info] : symbols)
    {
        Expects(virtual_address == symbol_info.vaddr);
        symbol_t sym{};
        sym.name               = symbol_info.name;
        sym.code_object_offset = symbol_info.faddr;
        sym.virtual_address    = symbol_info.vaddr + load_address;
        sym.size               = symbol_info.mem_size;
        symbol_map.push_back(sym);
    }
    return symbol_map;
}

instruction_t code_object_translator_impl_t::get_instruction(size_t object_id, uint64_t virtual_address) const
{
    const auto& inst = m_translator->get(virtual_address);
    if (inst)
    {
        return {inst->inst, inst->comment, virtual_address, inst->faddr, inst->size};
    }
    std::clog << "Could not get instruction for object id " << object_id << " at virtual address "
              << virtual_address << std::endl;
    return {};
}

std::set<std::filesystem::path> code_object_translator_impl_t::get_source_paths() const
{
    std::set<std::filesystem::path> source_paths;

    for (const auto& id : get_code_object_ids())
    {
        const auto symbols = get_symbols(id);
        for (const auto& sym : symbols)
        {
            uint64_t       pc  = sym.virtual_address;
            const uint64_t end = sym.virtual_address + sym.size;
            while (pc < end)
            {
                const auto inst = get_instruction(id, pc);
                Expects(inst.size);
                const auto paths = source_paths_from_comment(inst.comment);
                source_paths.insert(paths.cbegin(), paths.cend());
                pc += inst.size;
            }
        }
    }

    return source_paths;
}

std::vector<std::filesystem::path> code_object_translator_impl_t::source_paths_from_comment(std::string_view comment)
{
    std::vector<std::filesystem::path> source_paths;
    size_t                             frame_start = 0;

    while (frame_start <= comment.size())
    {
        const auto separator_position = comment.find(source_frame_separator, frame_start);
        const auto frame_end = separator_position == std::string_view::npos ? comment.size()
                                                                            : separator_position;
        const auto frame     = comment.substr(frame_start, frame_end - frame_start);
        const auto path      = path_from_source_frame(frame);
        if (!path.empty())
        {
            source_paths.emplace_back(std::string{path});
        }

        if (separator_position == std::string_view::npos)
            break;

        frame_start = separator_position + source_frame_separator.size();
    }

    return source_paths;
}
