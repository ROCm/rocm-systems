// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "pc_sampling_collector.h"

#include "gsl_assert.h"
#include "sdk_wrapper.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <set>
#include <string_view>
#include <utility>

using namespace rocprofiler_compute_tool;

bool pc_sampling_collector_impl_t::is_source_line_token(std::string_view token)
{
    if (token == "?")
        return true;

    return !token.empty() &&
           std::all_of(token.cbegin(),
                       token.cend(),
                       [](char ch) { return std::isdigit(static_cast<unsigned char>(ch)) != 0; });
}

std::string_view pc_sampling_collector_impl_t::source_frame_separator() const
{
    return m_sdk_wrapper->source_frame_separator();
}

std::string_view pc_sampling_collector_impl_t::path_from_source_frame(std::string_view frame)
{
    const auto separator_position = frame.rfind(':');
    if (separator_position == std::string_view::npos)
        return frame;

    const auto line_token = frame.substr(separator_position + 1);
    if (!is_source_line_token(line_token))
        return frame;

    return frame.substr(0, separator_position);
}

void pc_sampling_collector_impl_t::collect_source_paths_from_comment(std::string_view comment,
                                                                     std::set<std::filesystem::path>& source_paths) const
{
    size_t     frame_start = 0;
    const auto separator   = source_frame_separator();

    while (frame_start <= comment.size())
    {
        const auto separator_position = comment.find(separator, frame_start);
        const auto frame_end = separator_position == std::string_view::npos ? comment.size()
                                                                            : separator_position;
        const auto frame     = comment.substr(frame_start, frame_end - frame_start);
        const auto path      = path_from_source_frame(frame);
        if (!path.empty())
        {
            source_paths.emplace(path.begin(), path.end());
        }

        if (separator_position == std::string_view::npos)
            break;

        frame_start = separator_position + separator.size();
    }
}

pc_sampling_collector_t::ptr pc_sampling_collector_t::create()
{
    return std::make_shared<pc_sampling_collector_impl_t>(code_object_translator_t::create(),
                                                          sdk_wrapper_t::create());
}

pc_sampling_collector_impl_t::pc_sampling_collector_impl_t(code_object_translator_t::ptr translator)
    : pc_sampling_collector_impl_t(std::move(translator), sdk_wrapper_t::create())
{
}

pc_sampling_collector_impl_t::pc_sampling_collector_impl_t(code_object_translator_t::ptr translator,
                                                           sdk_wrapper_t::ptr sdk_wrapper)
    : m_translator(std::move(translator))
    , m_sdk_wrapper(std::move(sdk_wrapper))
{
}

void pc_sampling_collector_impl_t::on_code_object_load(
    const rocprofiler_callback_tracing_code_object_load_data_t& info)
{
    if (info.storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE)
    {
        m_translator->add_code_object(info.uri, info.code_object_id, info.load_base, info.load_size);
    }
    else if (info.storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY)
    {
        m_translator->add_code_object(info.memory_base,
                                      info.memory_size,
                                      info.code_object_id,
                                      info.load_base,
                                      info.load_size);
    }
}

void pc_sampling_collector_impl_t::on_kernel_symbol_register(uint64_t         kernel_id,
                                                             uint64_t         code_object_id,
                                                             std::string_view symbol_name)
{
    m_kernel_symbols.insert_or_assign(kernel_id,
                                      KernelSymbolKey{code_object_id, std::string{symbol_name}});
}

bool pc_sampling_collector_impl_t::has_code_objects() const
{
    return !m_translator->get_code_object_ids().empty();
}

void pc_sampling_collector_impl_t::write_symbol(code_object_writer_t& writer,
                                                uint64_t              code_object_id,
                                                const symbol_t&       symbol)
{
    writer.start_symbol(symbol);
    uint64_t       pc  = symbol.virtual_address;
    const uint64_t end = symbol.virtual_address + symbol.size;
    while (pc < end)
    {
        const auto& instruction = m_translator->get_instruction(code_object_id, pc);
        Expects(instruction.size);
        writer.write_instruction(instruction);
        collect_source_paths_from_comment(instruction.comment, m_source_paths);
        pc += instruction.size;
    }
    writer.end_symbol();
}

void pc_sampling_collector_impl_t::finalize(code_object_writer_t&     writer,
                                            const std::set<uint64_t>& dispatched_kernel_ids)
{
    std::set<std::pair<uint64_t, std::string>> dispatched_symbols;
    for (const auto kernel_id : dispatched_kernel_ids)
    {
        const auto found = m_kernel_symbols.find(kernel_id);
        if (found == m_kernel_symbols.end())
            continue;

        const auto& key = found->second;
        if (key.code_object_id != ROCPROFILER_CODE_OBJECT_ID_NONE && !key.symbol_name.empty())
            dispatched_symbols.emplace(key.code_object_id, key.symbol_name);
    }

    for (const auto id : m_translator->get_code_object_ids())
    {
        auto symbols = m_translator->get_symbols(id);
        symbols.erase(std::remove_if(symbols.begin(),
                                     symbols.end(),
                                     [&](const auto& symbol)
                                     { return dispatched_symbols.count({id, symbol.name}) == 0; }),
                      symbols.end());
        if (symbols.empty())
            continue;

        writer.start_code_obj(id);
        for (const auto& symbol : symbols)
            write_symbol(writer, id, symbol);
        writer.end_code_obj();
    }
}

const std::set<std::filesystem::path>& pc_sampling_collector_impl_t::get_source_paths() const
{
    return m_source_paths;
}
