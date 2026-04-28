// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "pc_sampling_collector.h"


using namespace pc_sampling_collector;

pc_sampling_collector_impl_t::pc_sampling_collector_impl_t() {}

void pc_sampling_collector_impl_t::on_code_object_load(
    const rocprofiler_callback_tracing_code_object_load_data_t& info)
{
    //CodeobjAddressTranslate translator;
    //if (info.storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE)
    //{
    //    translator.addDecoder(info.uri, info.code_object_id, info.load_base, info.load_size);
    //}
    //else if (info.storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY)
    //{
    //    translator.addDecoder(reinterpret_cast<const void*>(info.memory_base),
    //                          info.memory_size,
    //                          info.code_object_id,
    //                          info.load_base,
    //                          info.load_size);
    //}
    //auto symbols = translator.getSymbolMap();  // vaddr -> { name, mem_size }
    //for (auto& [vaddr, sym] : symbols)
    //{
    //    uint64_t pc = vaddr, end = vaddr + sym.mem_size;
    //    while (pc < end)
    //    {
    //        auto inst = translator.get(pc);
    //        std::cout << std::hex << pc << ": " << inst->inst << "\n";
    //        pc += inst->size;
    //    }
    //}
}