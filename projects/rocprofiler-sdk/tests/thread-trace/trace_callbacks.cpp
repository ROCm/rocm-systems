// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// undefine NDEBUG so asserts are implemented
#ifdef NDEBUG
#    undef NDEBUG
#endif

#include <rocprofiler-sdk/amd_detail/thread_trace.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/cxx/codeobj/code_printing.hpp>
#include "common.hpp"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>

namespace ATTTest
{
namespace Callbacks
{
using code_obj_load_data_t = rocprofiler_callback_tracing_code_object_load_data_t;
using kernel_symbol_data_t = rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;
using CodeobjAddressTranslate = rocprofiler::sdk::codeobj::disassembly::CodeobjAddressTranslate;
using Instruction             = rocprofiler::sdk::codeobj::disassembly::Instruction;

CodeobjAddressTranslate* codeobjTranslate = nullptr;

struct trace_data_t
{
    int64_t   id;
    uint8_t*  data;
    uint64_t  size;
    ToolData* tool;
};

void
tool_codeobj_tracing_callback(rocprofiler_callback_tracing_record_t record,
                              rocprofiler_user_data_t* /* user_data */,
                              void* callback_data)
{
    C_API_BEGIN
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT) return;
    if(record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD) return;

    assert(callback_data && "Shader callback passed null!");
    ToolData& tool = *reinterpret_cast<ToolData*>(callback_data);

    if(record.operation == ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
    {
        std::unique_lock<std::mutex> lg(tool.isa_map_mut);
        auto*                        data = static_cast<kernel_symbol_data_t*>(record.payload);
        tool.kernel_id_to_kernel_name.emplace(data->kernel_id, data->kernel_name);
    }

    if(record.operation != ROCPROFILER_CODE_OBJECT_LOAD) return;

    auto* data = static_cast<code_obj_load_data_t*>(record.payload);
    if(!data || !data->uri) return;

    std::unique_lock<std::mutex> lg(tool.isa_map_mut);

    if(std::string_view(data->uri).find("file:///") == 0)
    {
        codeobjTranslate->addDecoder(
            data->uri, data->code_object_id, data->load_delta, data->load_size);
    }
    else
    {
        codeobjTranslate->addDecoder(reinterpret_cast<const void*>(data->memory_base),
                                     data->memory_size,
                                     data->code_object_id,
                                     data->load_delta,
                                     data->load_size);
    }

    C_API_END
}

void
get_trace_data(rocprofiler_att_parser_data_type_t type, void* att_data, void* userdata)
{
    C_API_BEGIN
    assert(userdata && "ISA callback passed null!");
    trace_data_t& trace_data = *reinterpret_cast<trace_data_t*>(userdata);
    assert(trace_data.tool && "ISA callback passed null!");
    ToolData& tool = *reinterpret_cast<ToolData*>(trace_data.tool);

    std::unique_lock<std::mutex> lk(tool.isa_map_mut);

    if(type == ROCPROFILER_ATT_PARSER_DATA_TYPE_OCCUPANCY)
    {
        const auto& ev = *reinterpret_cast<const rocprofiler_att_data_type_occupancy_t*>(att_data);
        tool.wave_start_locations.insert({ev.offset, ev.marker_id});
        if(ev.enabled)
            tool.waves_started.fetch_add(1);
        else
            tool.waves_ended.fetch_add(1);
    }

    if(type != ROCPROFILER_ATT_PARSER_DATA_TYPE_ISA) return;

    auto& event = *reinterpret_cast<rocprofiler_att_data_type_isa_t*>(att_data);

    pcInfo pc{event.offset, event.marker_id};
    auto   it = tool.isa_map.find(pc);
    if(it == tool.isa_map.end())
    {
        auto ptr = std::make_unique<TrackedIsa>();
        try
        {
            auto unique_inst = codeobjTranslate->get(pc.marker_id, pc.addr);
            if(unique_inst == nullptr) return;
            ptr->inst = unique_inst->inst;
        } catch(...)
        {
            return;
        }
        it = tool.isa_map.emplace(pc, std::move(ptr)).first;
    }

    it->second->hitcount.fetch_add(event.hitcount, std::memory_order_relaxed);
    it->second->latency.fetch_add(event.latency, std::memory_order_relaxed);
    C_API_END
}

uint64_t
copy_trace_data(int* seid, uint8_t** buffer, uint64_t* buffer_size, void* userdata)
{
    trace_data_t& data = *reinterpret_cast<trace_data_t*>(userdata);
    *seid              = data.id;
    *buffer_size       = data.size;
    *buffer            = data.data;
    data.size          = 0;
    return *buffer_size;
}

rocprofiler_status_t
isa_callback(char*     isa_instruction,
             uint64_t* isa_memory_size,
             uint64_t* isa_size,
             uint64_t  marker_id,
             uint64_t  offset,
             void*     userdata)
{
    C_API_BEGIN
    assert(userdata && "ISA callback passed null!");
    trace_data_t& trace_data = *reinterpret_cast<trace_data_t*>(userdata);
    assert(trace_data.tool && "ISA callback passed null!");
    ToolData& tool = *reinterpret_cast<ToolData*>(trace_data.tool);

    std::unique_ptr<Instruction> instruction;

    try
    {
        std::unique_lock<std::mutex> unique_lock(tool.isa_map_mut);
        instruction = codeobjTranslate->get(marker_id, offset);
    } catch(...)
    {
        return ROCPROFILER_STATUS_ERROR;
    }

    if(!instruction.get()) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    {
        size_t tmp_isa_size = *isa_size;
        *isa_size           = instruction->inst.size();

        if(*isa_size > tmp_isa_size) return ROCPROFILER_STATUS_ERROR_OUT_OF_RESOURCES;
    }

    memcpy(isa_instruction, instruction->inst.data(), *isa_size);
    *isa_memory_size = instruction->size;

    auto ptr  = std::make_unique<TrackedIsa>();
    ptr->inst = instruction->inst;
    tool.isa_map.emplace(pcInfo{offset, marker_id}, std::move(ptr));
    return ROCPROFILER_STATUS_SUCCESS;
    C_API_END
    return ROCPROFILER_STATUS_ERROR;
}

void
shader_data_callback(int64_t                 se_id,
                     void*                   se_data,
                     size_t                  data_size,
                     rocprofiler_user_data_t userdata)
{
    C_API_BEGIN
    assert(userdata.ptr && "Shader callback passed null!");
    ToolData& tool = *reinterpret_cast<ToolData*>(userdata.ptr);

    trace_data_t data{.id = se_id, .data = (uint8_t*) se_data, .size = data_size, .tool = &tool};
    auto status = rocprofiler_att_parse_data(copy_trace_data, get_trace_data, isa_callback, &data);
    if(status != ROCPROFILER_STATUS_SUCCESS)
        std::cerr << "shader_data_callback failed with status " << status << std::endl;
    C_API_END
}

void
callbacks_init()
{
    codeobjTranslate = new CodeobjAddressTranslate();
}
void
callbacks_fini()
{
    delete codeobjTranslate;
}

}  // namespace Callbacks
}  // namespace ATTTest
