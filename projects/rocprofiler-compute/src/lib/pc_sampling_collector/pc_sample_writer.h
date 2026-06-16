// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include "pc_sample_types.h"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace rocprofiler_compute_tool
{
class pc_sample_writer_t
{
public:
    static std::shared_ptr<pc_sample_writer_t> create();

    virtual ~pc_sample_writer_t() = default;
    virtual void begin()          = 0;
    // inst_index is the interned (instruction, comment) index for this sample.
    virtual void append_stochastic(const pc_sample_record_t& r, size_t inst_index)              = 0;
    virtual void append_host_trap(const pc_sample_record_t& r, size_t inst_index)               = 0;
    virtual void set_strings(const pc_string_table_t& table)                                    = 0;
    virtual void set_kernel_symbols(const std::vector<kernel_symbol_entry_t>& syms)             = 0;
    virtual void set_agents(const std::vector<agent_record_t>& agents)                          = 0;
    virtual void set_kernel_dispatches(const std::vector<kernel_dispatch_record_t>& dispatches) = 0;
    virtual void set_metadata(int pid)                                                          = 0;
    virtual std::string get_result()                                                            = 0;
    virtual void        flush(const std::filesystem::path& output_file_path)                    = 0;
};

class pc_sample_writer_json_t : public pc_sample_writer_t
{
public:
    void begin() override;
    void append_stochastic(const pc_sample_record_t& r, size_t inst_index) override;
    void append_host_trap(const pc_sample_record_t& r, size_t inst_index) override;
    void set_strings(const pc_string_table_t& table) override;
    void set_kernel_symbols(const std::vector<kernel_symbol_entry_t>& syms) override;
    void set_agents(const std::vector<agent_record_t>& agents) override;
    void set_kernel_dispatches(const std::vector<kernel_dispatch_record_t>& dispatches) override;
    void set_metadata(int pid) override;
    std::string get_result() override;
    void        flush(const std::filesystem::path& output_file_path) override;

private:
    // A decoded sample paired with its interned (instruction, comment) index.
    struct sample_entry_t
    {
        pc_sample_record_t record;
        size_t             inst_index = 0;
    };

    int                                   m_pid = 0;
    std::vector<sample_entry_t>           m_stochastic;
    std::vector<sample_entry_t>           m_host_trap;
    std::vector<std::string>              m_instructions;
    std::vector<std::string>              m_comments;
    std::vector<kernel_symbol_entry_t>    m_kernel_symbols;
    std::vector<agent_record_t>           m_agents;
    std::vector<kernel_dispatch_record_t> m_kernel_dispatches;
};
}  // namespace rocprofiler_compute_tool
