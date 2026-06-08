// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include "nlohmann/json.hpp"
#include "pc_record_store.h"

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace rocprofiler_compute_tool
{
// Disassembled instruction strings to embed in the ps_file results. The two
// vectors are index-parallel: entry i of pc_sample_comments is the comment for
// the instruction whose disassembly text is entry i of pc_sample_instructions.
// offset_to_index maps a (code_object_id, code_object_offset) pair to the index
// of the attributed instruction so per-sample inst_index can be resolved.
struct pc_sample_strings_t
{
    std::vector<std::string>                     pc_sample_instructions{};
    std::vector<std::string>                     pc_sample_comments{};
    std::map<std::pair<uint64_t, uint64_t>, int> offset_to_index{};
};

struct pc_kernel_symbol_t
{
    uint64_t    code_object_id = 0;
    std::string formatted_kernel_name{};
};

// nlohmann-json writer for the PC sampling ps_file_results.json, mirroring the
// style of code_object_writer_json_t (build into nlohmann::json, flush to path).
class pc_record_writer_t
{
public:
    void write(const pc_record_store_t&               store,
               const pc_sample_strings_t&             strings,
               const std::vector<pc_kernel_symbol_t>& kernel_symbols,
               const nlohmann::json::array_t&         code_objects);

    std::string get_result() const;
    void        flush(const std::filesystem::path& output_file_path) const;

private:
    static void create_parent_dir(const std::filesystem::path& output_file_path);

    nlohmann::json m_document{};
};
}  // namespace rocprofiler_compute_tool
