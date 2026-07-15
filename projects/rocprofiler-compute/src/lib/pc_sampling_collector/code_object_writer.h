// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "code_object_translator.h"
#include "nlohmann/json.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace rocprofiler_compute_tool
{
class code_object_writer_t
{
public:
    virtual ~code_object_writer_t()                                  = default;
    virtual void        start_code_obj(size_t obj_id)                = 0;
    virtual void        end_code_obj()                               = 0;
    virtual void        start_symbol(const symbol_t& symbol)         = 0;
    virtual void        end_symbol()                                 = 0;
    virtual void        write_instruction(const instruction_t& inst) = 0;
    virtual std::string get_result()                                 = 0;
    virtual void        flush(const std::filesystem::path& string)   = 0;
};

class code_object_writer_json_t : public code_object_writer_t
{
public:
    void        start_code_obj(size_t obj_id) override;
    void        end_code_obj() override;
    void        start_symbol(const symbol_t& symbol) override;
    void        end_symbol() override;
    void        write_instruction(const instruction_t& inst) override;
    std::string get_result() override;
    void        flush(const std::filesystem::path& output_file_path) override;
    bool        empty() const;

private:
    static void create_parent_dir(const std::filesystem::path& output_file_path);

    int32_t m_code_object_closure_count = 0;
    int32_t m_symbol_closure_count      = 0;

    size_t                  m_current_obj_id = 0;
    symbol_t                m_current_symbol{};
    nlohmann::json::array_t m_code_objects = nlohmann::json::array();
    nlohmann::json::array_t m_symbols      = nlohmann::json::array();
    nlohmann::json::array_t m_instructions = nlohmann::json::array();
};
}  // namespace rocprofiler_compute_tool
