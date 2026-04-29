// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "pc_sampling_collector.h"

using namespace rocm_compute;

pc_sampling_collector_t::ptr pc_sampling_collector_t::create()
{
    return std::make_unique<pc_sampling_collector_impl_t>(
        std::make_shared<code_object_translator_impl_t>());
}

pc_sampling_collector_impl_t::pc_sampling_collector_impl_t(
    const std::shared_ptr<code_object_translator_t>& translator)
    : m_translator(translator)
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

void pc_sampling_collector_impl_t::write(pc_samples_writer_t& writer)
{
    for (const auto& id : m_translator->get_code_object_ids())
    {
        const auto& symbols = m_translator->get_symbol_map(id);
        // auto symbols = translator.getSymbolMap(id);  // vaddr -> { name, mem_size }
        // for (auto& [vaddr, sym] : symbols)
        //{
        //     uint64_t pc = vaddr, end = vaddr + sym.mem_size;
        //     while (pc < end)
        //     {
        //         auto inst = translator.get(id, pc);
        //         std::cout << std::hex << pc << ": " << inst->inst << "\n";
        //         pc += inst->size;
        //     }
        // }
        writer.write();
    }
}
