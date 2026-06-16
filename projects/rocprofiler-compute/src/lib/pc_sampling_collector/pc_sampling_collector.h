// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "code_object_translator.h"
#include "code_object_writer.h"
#include "gsl_assert.h"
#include "pc_sample_writer.h"

#include <rocprofiler-sdk/rocprofiler.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace rocprofiler_compute_tool
{

enum class PcSamplingMode : uint8_t
{
    Disabled,
    Stochastic,
    HostTrap
};

class pc_sampling_collector_t
{
public:
    using ptr = std::shared_ptr<pc_sampling_collector_t>;
    static ptr create();

    virtual ~pc_sampling_collector_t() = default;
    virtual void on_code_object_load(const rocprofiler_callback_tracing_code_object_load_data_t& info) = 0;
    virtual void write(code_object_writer_t& writer) = 0;

    virtual void   append_sample(const pc_sample_record_t& record)                = 0;
    virtual void   add_kernel_symbol(uint64_t           code_object_id,
                                     const std::string& formatted_kernel_name,
                                     uint64_t           kernel_id)                = 0;
    virtual void   add_agent(const agent_record_t& agent)                         = 0;
    virtual void   append_kernel_dispatch(const kernel_dispatch_record_t& record) = 0;
    virtual void   write_samples(pc_sample_writer_t& writer)                      = 0;
    virtual size_t snapshot_sources(const std::filesystem::path& output_root)     = 0;
};

class pc_sampling_collector_impl_t : public pc_sampling_collector_t
{
public:
    pc_sampling_collector_impl_t(const std::shared_ptr<code_object_translator_t>& translator);
    void on_code_object_load(const rocprofiler_callback_tracing_code_object_load_data_t& info) override;
    void write(code_object_writer_t& writer) override;

    void   append_sample(const pc_sample_record_t& record) override;
    void   add_kernel_symbol(uint64_t           code_object_id,
                             const std::string& formatted_kernel_name,
                             uint64_t           kernel_id) override;
    void   add_agent(const agent_record_t& agent) override;
    void   append_kernel_dispatch(const kernel_dispatch_record_t& record) override;
    void   write_samples(pc_sample_writer_t& writer) override;
    size_t snapshot_sources(const std::filesystem::path& output_root) override;

private:
    instruction_t resolve_instruction(uint64_t code_object_id, uint64_t code_object_offset);

    template<typename Fn>
    void for_each_instruction(Fn&& fn);

    std::shared_ptr<code_object_translator_t> m_translator;

    std::mutex                            m_mutex;
    std::vector<pc_sample_record_t>       m_samples;
    std::vector<kernel_symbol_entry_t>    m_kernel_symbols;
    std::vector<agent_record_t>           m_agents;
    std::vector<kernel_dispatch_record_t> m_kernel_dispatches;
    pc_string_table_t                     m_string_table;
};

template<typename Fn>
void pc_sampling_collector_impl_t::for_each_instruction(Fn&& fn)
{
    for (const auto& id : m_translator->get_code_object_ids())
    {
        // Walk the whole loaded range so callers (e.g. source snapshotting) see
        // every instruction, including code outside named symbols.
        const uint64_t base      = m_translator->get_load_base(id);
        const uint64_t range_end = base + m_translator->get_load_size(id);

        uint64_t pc = base;
        while (pc < range_end)
        {
            const auto inst = m_translator->get_instruction(id, pc);
            if (inst.size == 0)
            {
                pc += sizeof(uint32_t);
                continue;
            }
            fn(id, inst);
            pc += inst.size;
        }
    }
}
}  // namespace rocprofiler_compute_tool
