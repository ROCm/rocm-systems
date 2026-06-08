// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "pc_record_writer.h"

#include "gsl_assert.h"

#include <fstream>
#include <iostream>
#include <utility>

using namespace rocprofiler_compute_tool;

namespace
{
// Resolve the index into the instruction/comment string table for a sample,
// matching on (code_object_id, code_object_offset). Falls back to the record's
// own inst_index (default -1) when the offset is not present in the table.
int resolve_inst_index(const pc_sample_strings_t& strings, const pc_sample_record_t& record)
{
    auto it = strings.offset_to_index.find({record.code_object_id, record.code_object_offset});
    if (it != strings.offset_to_index.end())
        return it->second;
    return record.inst_index;
}

// Build the element shared by both sample kinds. Stochastic samples additionally
// carry snapshot.stall_reason and wave_issued; host_trap samples omit them.
nlohmann::json make_record_element(const pc_sample_strings_t& strings,
                                   const pc_sample_record_t&  record,
                                   bool                       include_stochastic_fields)
{
    auto inner = nlohmann::json::object({
        {"pc",
         nlohmann::json::object({
             {"code_object_id", record.code_object_id},
             {"code_object_offset", record.code_object_offset},
         })},
        {"dispatch_id", record.dispatch_id},
        {"inst_type", record.inst_type},
    });
    if (include_stochastic_fields)
    {
        inner["snapshot"]    = nlohmann::json::object({{"stall_reason", record.stall_reason}});
        inner["wave_issued"] = record.wave_issued;
    }
    return nlohmann::json::object({
        {"record", std::move(inner)},
        {"inst_index", resolve_inst_index(strings, record)},
    });
}

nlohmann::json make_stochastic_element(const pc_sample_strings_t& strings, const pc_sample_record_t& record)
{
    return make_record_element(strings, record, /*include_stochastic_fields=*/true);
}

nlohmann::json make_host_trap_element(const pc_sample_strings_t& strings, const pc_sample_record_t& record)
{
    return make_record_element(strings, record, /*include_stochastic_fields=*/false);
}
}  // namespace

void pc_record_writer_t::write(const pc_record_store_t&               store,
                               const pc_sample_strings_t&             strings,
                               const std::vector<pc_kernel_symbol_t>& kernel_symbols,
                               const nlohmann::json::array_t&         code_objects)
{
    nlohmann::json::array_t stochastic_records = nlohmann::json::array();
    for (const auto& record : store.stochastic())
        stochastic_records.push_back(make_stochastic_element(strings, record));

    nlohmann::json::array_t host_trap_records = nlohmann::json::array();
    for (const auto& record : store.host_trap())
        host_trap_records.push_back(make_host_trap_element(strings, record));

    nlohmann::json::array_t symbols = nlohmann::json::array();
    for (const auto& sym : kernel_symbols)
    {
        symbols.push_back(nlohmann::json::object({
            {"code_object_id", sym.code_object_id},
            {"formatted_kernel_name", sym.formatted_kernel_name},
        }));
    }

    nlohmann::json entry = nlohmann::json::object({
        {"strings",
         nlohmann::json::object({
             {"pc_sample_instructions", strings.pc_sample_instructions},
             {"pc_sample_comments", strings.pc_sample_comments},
         })},
        {"kernel_symbols", std::move(symbols)},
        {"code_objects", code_objects},
        {"buffer_records",
         nlohmann::json::object({
             {"pc_sample_stochastic", std::move(stochastic_records)},
             {"pc_sample_host_trap", std::move(host_trap_records)},
         })},
    });

    nlohmann::json::array_t tool_array = nlohmann::json::array();
    tool_array.push_back(std::move(entry));

    m_document = nlohmann::json::object({
        {"rocprofiler-sdk-tool", std::move(tool_array)},
    });
}

std::string pc_record_writer_t::get_result() const
{
    return m_document.dump();
}

void pc_record_writer_t::flush(const std::filesystem::path& output_file_path) const
{
    Expects(!output_file_path.empty());
    create_parent_dir(output_file_path);

    std::ofstream out_file(output_file_path, std::ios::out);
    if (!out_file.is_open())
    {
        std::cerr << "Failed to open output file: " << output_file_path << "\n";
        return;
    }
    out_file << get_result();
    std::clog << "[rocprofiler-compute] [" << __FUNCTION__
              << "] PC sampling records have been written to: " << output_file_path << "\n";
}

void pc_record_writer_t::create_parent_dir(const std::filesystem::path& output_file_path)
{
    if (!output_file_path.has_parent_path())
        return;

    std::error_code error;
    std::filesystem::create_directories(output_file_path.parent_path(), error);
    if (error)
    {
        throw std::runtime_error("Failed to create output directory: " + output_file_path.string() +
                                 ", error: " + error.message());
    }
}
