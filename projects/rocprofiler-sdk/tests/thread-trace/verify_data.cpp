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

/**
 * @file samples/code_object_isa_decode/client.cpp
 *
 * @brief Example rocprofiler client (tool)
 */

#include <rocprofiler-sdk/amd_detail/thread_trace.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/amd_detail/rocprofiler-sdk-codeobj/code_printing.hpp>

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

#define WAVE_RATIO_TOLERANCE 0.05

#define ROCPROFILER_CALL(result, msg)                                                              \
    {                                                                                              \
        rocprofiler_status_t CHECKSTATUS = result;                                                 \
        if(CHECKSTATUS != ROCPROFILER_STATUS_SUCCESS)                                              \
        {                                                                                          \
            std::string status_msg = rocprofiler_get_status_string(CHECKSTATUS);                   \
            std::cerr << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg            \
                      << " failed with error code " << CHECKSTATUS << ": " << status_msg           \
                      << std::endl;                                                                \
            std::stringstream errmsg{};                                                            \
            errmsg << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg " failure ("  \
                   << status_msg << ")";                                                           \
            throw std::runtime_error(errmsg.str());                                                \
        }                                                                                          \
    }

#define C_API_BEGIN                                                                                \
    try                                                                                            \
    {
#define C_API_END                                                                                  \
    }                                                                                              \
    catch(std::exception & e)                                                                      \
    {                                                                                              \
        std::cerr << "Error in " << __FILE__ << ':' << __LINE__ << ' ' << e.what() << std::endl;   \
    }                                                                                              \
    catch(...) { std::cerr << "Error in " << __FILE__ << ':' << __LINE__ << std::endl; }

namespace thread_trace_test_client
{
using code_obj_load_data_t = rocprofiler_callback_tracing_code_object_load_data_t;
using kernel_symbol_data_t = rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;
using Instruction          = rocprofiler::codeobj::disassembly::Instruction;
using CodeobjAddressTranslate = rocprofiler::codeobj::disassembly::CodeobjAddressTranslate;

std::mutex               isa_map_mut;
rocprofiler_client_id_t* client_id = nullptr;

struct isa_map_elem_t
{
    std::atomic<size_t>          hitcount{0};
    std::atomic<size_t>          latency{0};
    std::shared_ptr<Instruction> code_line{nullptr};
};

struct pcinfo_t
{
    uint64_t marker_id;
    uint64_t addr;
};

bool
operator==(const pcinfo_t& a, const pcinfo_t& b)
{
    return a.addr == b.addr && a.marker_id == b.marker_id;
};

bool
operator<(const pcinfo_t& a, const pcinfo_t& b)
{
    if(a.marker_id == b.marker_id) return a.addr < b.addr;
    return a.marker_id < b.marker_id;
};

struct ToolData
{
    std::unordered_map<uint64_t, std::string>           kernel_object_to_kernel_name = {};
    CodeobjAddressTranslate                             codeobjTranslate;
    std::map<pcinfo_t, std::unique_ptr<isa_map_elem_t>> isa_map;
    std::atomic<int>                                    waves_started = 0;
    std::atomic<int>                                    waves_ended   = 0;
};

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
        std::unique_lock<std::mutex> lg(isa_map_mut);
        auto*                        data = static_cast<kernel_symbol_data_t*>(record.payload);
        tool.kernel_object_to_kernel_name.emplace(data->kernel_object, data->kernel_name);
    }

    if(record.operation != ROCPROFILER_CODE_OBJECT_LOAD) return;

    auto* data = static_cast<code_obj_load_data_t*>(record.payload);
    if(!data || !data->uri) return;

    std::unique_lock<std::mutex> lg(isa_map_mut);

    if(std::string_view(data->uri).find("file:///") == 0)
    {
        tool.codeobjTranslate.addDecoder(data->uri, 0, data->load_delta, data->load_size);
    }
    else
    {
        tool.codeobjTranslate.addDecoder(reinterpret_cast<const void*>(data->memory_base),
                                         data->memory_size,
                                         data->code_object_id,
                                         data->load_delta,
                                         data->load_size);
    }

    C_API_END
}

rocprofiler_att_control_flags_t
dispatch_callback(rocprofiler_queue_id_t /* queue_id  */,
                  const rocprofiler_agent_t* /* agent  */,
                  rocprofiler_correlation_id_t /* correlation_id  */,
                  const hsa_kernel_dispatch_packet_t* dispatch_packet,
                  uint64_t /* kernel_id */,
                  void* userdata)
{
    C_API_BEGIN
    assert(userdata && "Dispatch callback passed null!");
    ToolData& tool = *reinterpret_cast<ToolData*>(userdata);

    static std::atomic<int> call_id{0};
    static std::string_view desired_func_name = "branching_kernel";

    try
    {
        auto& kernel_name = tool.kernel_object_to_kernel_name.at(dispatch_packet->kernel_object);
        if(kernel_name.find(desired_func_name) == std::string::npos)
            return ROCPROFILER_ATT_CONTROL_NONE;

        if(call_id.fetch_add(1) == 0) return ROCPROFILER_ATT_CONTROL_START_AND_STOP;
    } catch(...)
    {
        std::cerr << "Could not find kernel object: " << dispatch_packet->kernel_object
                  << std::endl;
    }

    C_API_END
    return ROCPROFILER_ATT_CONTROL_NONE;
}

void
get_trace_data(rocprofiler_att_parser_data_type_t type, void* att_data, void* userdata)
{
    C_API_BEGIN
    assert(userdata && "ISA callback passed null!");
    trace_data_t& trace_data = *reinterpret_cast<trace_data_t*>(userdata);
    assert(trace_data.tool && "ISA callback passed null!");
    ToolData& tool = *reinterpret_cast<ToolData*>(trace_data.tool);

    if(type == ROCPROFILER_ATT_PARSER_DATA_TYPE_OCCUPANCY)
    {
        const auto& ev = reinterpret_cast<const rocprofiler_att_data_type_occupancy_t*>(att_data);
        if(ev->enabled)
            tool.waves_started.fetch_add(1);
        else
            tool.waves_ended.fetch_add(1);
    }

    if(type != ROCPROFILER_ATT_PARSER_DATA_TYPE_ISA) return;

    std::unique_lock<std::mutex> lk(isa_map_mut);
    auto& event = *reinterpret_cast<rocprofiler_att_data_type_isa_t*>(att_data);

    pcinfo_t pc{event.marker_id, event.offset};
    auto     it = tool.isa_map.find(pc);
    if(it == tool.isa_map.end())
    {
        auto ptr = std::make_unique<isa_map_elem_t>();
        try
        {
            ptr->code_line = tool.codeobjTranslate.get(pc.marker_id, pc.addr);
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

    std::shared_ptr<Instruction> instruction;

    {
        std::unique_lock<std::mutex> unique_lock(isa_map_mut);
        instruction = tool.codeobjTranslate.get(marker_id, offset);
    }

    if(!instruction.get()) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    {
        size_t tmp_isa_size = *isa_size;
        *isa_size           = instruction->inst.size();

        if(*isa_size > tmp_isa_size) return ROCPROFILER_STATUS_ERROR_OUT_OF_RESOURCES;
    }

    memcpy(isa_instruction, instruction->inst.data(), *isa_size);
    *isa_memory_size = instruction->size;

    auto ptr       = std::make_unique<isa_map_elem_t>();
    ptr->code_line = std::move(instruction);
    tool.isa_map.emplace(pcinfo_t{marker_id, offset}, std::move(ptr));
    C_API_END
    return ROCPROFILER_STATUS_SUCCESS;
}

void
shader_data_callback(int64_t se_id, void* se_data, size_t data_size, void* userdata)
{
    C_API_BEGIN
    assert(userdata && "Shader callback passed null!");
    ToolData& tool = *reinterpret_cast<ToolData*>(userdata);

    trace_data_t data{.id = se_id, .data = (uint8_t*) se_data, .size = data_size, .tool = &tool};
    auto status = rocprofiler_att_parse_data(copy_trace_data, get_trace_data, isa_callback, &data);
    if(status != ROCPROFILER_STATUS_SUCCESS)
        std::cerr << "shader_data_callback failed with status " << status << std::endl;
    C_API_END
}

int
tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data)
{
    static rocprofiler_context_id_t client_ctx = {};
    (void) fini_func;
    ROCPROFILER_CALL(rocprofiler_create_context(&client_ctx), "context creation");

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(client_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                       nullptr,
                                                       0,
                                                       tool_codeobj_tracing_callback,
                                                       tool_data),
        "code object tracing service configure");

    ROCPROFILER_CALL(
        rocprofiler_configure_thread_trace_service(
            client_ctx, nullptr, 0, dispatch_callback, shader_data_callback, tool_data),
        "thread trace service configure");

    int valid_ctx = 0;
    ROCPROFILER_CALL(rocprofiler_context_is_valid(client_ctx, &valid_ctx),
                     "context validity check");
    if(valid_ctx == 0)
    {
        // notify rocprofiler that initialization failed
        // and all the contexts, buffers, etc. created
        // should be ignored
        return -1;
    }

    ROCPROFILER_CALL(rocprofiler_start_context(client_ctx), "context start");

    // no errors
    return 0;
}

void
tool_fini(void* tool_data)
{
    assert(tool_data && "tool_fini callback passed null!");
    ToolData& tool = *reinterpret_cast<ToolData*>(tool_data);

    std::unique_lock<std::mutex> isa_lk(isa_map_mut);

    // Find largest instruction
    size_t max_inst_size = 0;
    for(auto& [addr, lines] : tool.isa_map)
        if(lines.get()) max_inst_size = std::max(max_inst_size, lines->code_line->inst.size());

    assert(max_inst_size > 0);

    size_t total_hit    = 0;
    size_t total_cycles = 0;

    for(auto& [addr, line] : tool.isa_map)
    {
        total_hit += line->hitcount.load(std::memory_order_relaxed);
        total_cycles += line->latency.load(std::memory_order_relaxed);
    }

    assert(total_cycles > 0);
    assert(total_hit > 0);

    double wave_started     = (double) tool.waves_started.load();
    double wave_event_ratio = wave_started / (wave_started + (double) tool.waves_ended.load());
    assert(wave_event_ratio > 0.5 - WAVE_RATIO_TOLERANCE);
    assert(wave_event_ratio < 0.5 + WAVE_RATIO_TOLERANCE);
}

}  // namespace thread_trace_test_client

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t /* version */,
                      const char* /* runtime_version */,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
    // only activate if main tool
    if(priority > 0) return nullptr;

    // set the client name
    id->name = "Adv_Thread_Trace_Sample";

    // store client info
    thread_trace_test_client::client_id = id;

    auto* data = new thread_trace_test_client::ToolData{};

    // create configure data
    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &thread_trace_test_client::tool_init,
                                            &thread_trace_test_client::tool_fini,
                                            reinterpret_cast<void*>(data)};

    // return pointer to configure data
    return &cfg;
}
