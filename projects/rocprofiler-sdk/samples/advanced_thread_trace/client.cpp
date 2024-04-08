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

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include "lib/rocprofiler-sdk/aql/aql_profile_v2.h"

#include "code_object_track.hpp"
#include "common/defines.hpp"
#include "common/filesystem.hpp"
#include "lib/rocprofiler-sdk-codeobj/code_printing.hpp"

#include <cxxabi.h>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <regex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#define TARGET_CU   1
#define SIMD_SELECT 0x3
#define BUFFER_SIZE 0x6000000
#define NUM_SE      2
constexpr bool COPY_MEMORY_CODEOBJ = false;

template <>
struct std::hash<pcinfo_t>
{
    uint64_t operator()(const pcinfo_t& info) const
    {
        return info.addr ^ (info.marker_id << 32ul) ^ (info.marker_id >> 32ul);
    }
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

namespace client
{
using code_obj_load_data_t = rocprofiler_callback_tracing_code_object_load_data_t;
using kernel_symbol_data_t = rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;

rocprofiler_client_id_t* client_id  = nullptr;
rocprofiler_context_id_t client_ctx = {};

struct isa_map_elem_t
{
    std::atomic<size_t>          hitcount{0};
    std::atomic<size_t>          latency{0};
    std::shared_ptr<Instruction> code_line{nullptr};
};

struct ToolData
{
    std::shared_mutex                                   isa_map_mut;
    std::mutex                                          output_mut;
    CodeobjAddressTranslate                             codeobjTranslate;
    std::map<pcinfo_t, std::unique_ptr<isa_map_elem_t>> isa_map;
    std::unordered_map<uint64_t, SymbolInfo>            kernels_in_codeobj           = {};
    std::unordered_map<uint64_t, std::string>           kernel_object_to_kernel_name = {};
    std::stringstream                                   output;

    std::stringstream printKernel(uint64_t vaddr)
    {
        std::stringstream ss;
        try
        {
            ss << '\n' << std::hex;
            SymbolInfo& info = kernels_in_codeobj.at(vaddr);

            ss << std::hex << "Found: " << info.name << " at addr: 0x" << vaddr << " with offset 0x"
               << info.faddr << " vaddr 0x" << info.vaddr << std::dec << '\n';
        } catch(std::exception& e)
        {
            ss << e.what() << '\n';
        }
        return ss;
    }
};

struct source_location
{
    std::string function = {};
    std::string file     = {};
    uint32_t    line     = 0;
    std::string context  = {};
};

struct trace_data_t
{
    int64_t   id;
    uint8_t*  data;
    uint64_t  size;
    ToolData* tool;
};

std::atomic<int> TRACE_DATA_ID{-1};
std::atomic<int> KERNEL_ADDR_ID{-1};
std::atomic<int> OCCUPANCY_ID{-1};

void
tool_codeobj_tracing_callback(rocprofiler_callback_tracing_record_t record,
                              rocprofiler_user_data_t*              user_data,
                              void*                                 callback_data)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT) return;
    if(record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD) return;

    assert(callback_data && "Shader callback passed null!");
    ToolData& tool = *reinterpret_cast<ToolData*>(callback_data);

    if(record.operation == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
    {
        std::unique_lock<std::shared_mutex> lg(tool.isa_map_mut);
        auto* data = static_cast<kernel_symbol_data_t*>(record.payload);
        tool.kernel_object_to_kernel_name.emplace(data->kernel_object, data->kernel_name);
    }

    if(record.operation != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT_LOAD) return;

    std::unique_lock<std::shared_mutex> lg(tool.isa_map_mut);
    auto*                               data = static_cast<code_obj_load_data_t*>(record.payload);

    if(std::string_view(data->uri).find("file:///") == 0)
    {
        tool.codeobjTranslate.addDecoder(
            data->uri, data->code_object_id, data->load_delta, data->load_size);
        auto symbolmap = tool.codeobjTranslate.getSymbolMap(data->code_object_id);
        for(auto& [vaddr, symbol] : symbolmap)
            tool.kernels_in_codeobj[vaddr] = symbol;
    }
    else if(COPY_MEMORY_CODEOBJ)
    {
        tool.codeobjTranslate.addDecoder(reinterpret_cast<const void*>(data->memory_base),
                                         data->memory_size,
                                         data->code_object_id,
                                         data->load_delta,
                                         data->load_size);
        auto symbolmap = tool.codeobjTranslate.getSymbolMap(data->code_object_id);
        for(auto& [vaddr, symbol] : symbolmap)
            tool.kernels_in_codeobj[vaddr] = symbol;
    }

    (void) user_data;
    (void) callback_data;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"

rocprofiler_att_control_flags_t
dispatch_callback(rocprofiler_queue_id_t              queue_id,
                  const rocprofiler_agent_t*          agent,
                  rocprofiler_correlation_id_t        correlation_id,
                  const hsa_kernel_dispatch_packet_t* dispatch_packet,
                  uint64_t                            kernel_id,
                  void*                               userdata)
{
    assert(userdata && "Dispatch callback passed null!");
    ToolData& tool = *reinterpret_cast<ToolData*>(userdata);

    std::shared_lock<std::shared_mutex> lg(tool.isa_map_mut);

    constexpr int           desired_call_iteration = 1;
    static std::atomic<int> call_id{0};
    static std::string_view desired_func_name = "transposeLdsSwapInplace";

    try
    {
        auto& kernel_name = tool.kernel_object_to_kernel_name.at(dispatch_packet->kernel_object);
        if(kernel_name.find(desired_func_name) == std::string::npos)
            return ROCPROFILER_ATT_CONTROL_NONE;

        if(call_id.fetch_add(1) == desired_call_iteration)
            return ROCPROFILER_ATT_CONTROL_START_AND_STOP;
    } catch(...)
    {
        std::cerr << "Could not find kernel object: " << dispatch_packet->kernel_object
                  << std::endl;
    }

    return ROCPROFILER_ATT_CONTROL_NONE;
}

void
iterate_trace_types(int id, const char* metadata, void*)
{
    if(std::string_view(metadata).find("occupancy") == 0)
        OCCUPANCY_ID.store(id);
    else if(std::string_view(metadata).find("kernel_ids_addr") == 0)
        KERNEL_ADDR_ID.store(id);
    else if(std::string_view(metadata).find("tracedata") == 0)
        TRACE_DATA_ID.store(id);
}

hsa_status_t
get_trace_data(int      trace_type_id,
               int      correlation_id,
               void*    trace_events,
               uint64_t trace_size,
               void*    userdata)
{
    assert(userdata && "ISA callback passed null!");
    trace_data_t& trace_data = *reinterpret_cast<trace_data_t*>(userdata);
    assert(trace_data.tool && "ISA callback passed null!");
    ToolData& tool = *reinterpret_cast<ToolData*>(trace_data.tool);

    std::stringstream                   ss;
    std::shared_lock<std::shared_mutex> shared_lock(tool.isa_map_mut);

    if(trace_type_id == OCCUPANCY_ID)
    {
        ss << "Num waves: " << trace_size / 2 << '\n';
        // auto* occ = reinterpret_cast<att_occupancy_info_t*>(trace_events);
    }
    else if(trace_type_id == KERNEL_ADDR_ID)
    {
        ss << "Num KRN events: " << trace_size << std::hex << '\n';
        auto* kaddr = reinterpret_cast<pcinfo_t*>(trace_events);
        for(size_t i = 0; i < trace_size; i++)
            if(kaddr[i].addr != 0)
            {
                ss << " - ADDR: " << kaddr[i].addr << ' ' << tool.printKernel(kaddr[i].addr).str()
                   << '\n';
            }
        ss << std::dec;
    }
    else if(trace_type_id == TRACE_DATA_ID)
    {
        ss << "Trace Length: " << trace_size << '\n';
        auto* tracedata = reinterpret_cast<att_trace_event_t*>(trace_events);

        for(size_t i = 0; i < trace_size; i++)
        {
            pcinfo_t pc = tracedata[i].pc;
            auto     it = tool.isa_map.find(pc);
            if(it == tool.isa_map.end())
            {
                shared_lock.unlock();
                {
                    std::unique_lock<std::shared_mutex> unique_lock(tool.isa_map_mut);
                    auto                                ptr = std::make_unique<isa_map_elem_t>();
                    ptr->code_line = tool.codeobjTranslate.get(pc.marker_id, pc.addr);
                    it             = tool.isa_map.emplace(pc, std::move(ptr)).first;
                }
                shared_lock.lock();
            }

            it->second->hitcount.fetch_add(tracedata[i].hitcount, std::memory_order_relaxed);
            it->second->latency.fetch_add(tracedata[i].latency, std::memory_order_relaxed);
        }
    }

    std::unique_lock<std::mutex> lk(tool.output_mut);
    tool.output << ss.str();

    return HSA_STATUS_SUCCESS;
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

hsa_status_t
isa_callback(char*     isa_instruction,
             char*     source_reference,
             uint64_t* isa_memory_size,
             uint64_t* isa_size,
             uint64_t* source_size,
             uint64_t  marker_id,
             uint64_t  offset,
             void*     userdata)
{
    assert(userdata && "ISA callback passed null!");
    trace_data_t& trace_data = *reinterpret_cast<trace_data_t*>(userdata);
    assert(trace_data.tool && "ISA callback passed null!");
    ToolData& tool = *reinterpret_cast<ToolData*>(trace_data.tool);

    std::shared_ptr<Instruction> instruction;

    {
        std::unique_lock<std::shared_mutex> unique_lock(tool.isa_map_mut);
        instruction = tool.codeobjTranslate.get(marker_id, offset);
    }

    if(!instruction.get()) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

    {
        size_t tmp_isa_size    = *isa_size;
        size_t tmp_source_size = *source_size;
        *isa_size              = instruction->inst.size();
        *source_size           = instruction->comment.size();

        if(*isa_size > tmp_isa_size || *source_size > tmp_source_size)
            return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
    }

    memcpy(isa_instruction, instruction->inst.data(), *isa_size);
    memcpy(source_reference, instruction->comment.data(), *source_size);
    *isa_memory_size = instruction->size;

    auto ptr       = std::make_unique<isa_map_elem_t>();
    ptr->code_line = std::move(instruction);
    tool.isa_map.emplace(pcinfo_t{offset, marker_id}, std::move(ptr));

    return HSA_STATUS_SUCCESS;
}

void
shader_data_callback(int64_t     se_id,
                     int64_t     data_type_id,
                     const char* data_type_name,
                     void*       se_data,
                     size_t      data_size,
                     void*       userdata)
{
    assert(userdata && "Shader callback passed null!");
    ToolData& tool = *reinterpret_cast<ToolData*>(userdata);

    {
        std::unique_lock<std::mutex> lk(tool.output_mut);
        tool.output << "SE ID: " << se_id << " with size " << data_size << std::hex << '\n';
    }
    trace_data_t data{.id = se_id, .data = (uint8_t*) se_data, .size = data_size, .tool = &tool};
    auto status = aqlprofile_att_parse_data(copy_trace_data, get_trace_data, isa_callback, &data);
    (void) status;
}

#pragma GCC diagnostic pop

int
tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data)
{
    (void) fini_func;
    aqlprofile_att_parser_iterate_event_list(iterate_trace_types, nullptr);
    ROCPROFILER_CALL(rocprofiler_create_context(&client_ctx), "context creation");

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(client_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                       nullptr,
                                                       0,
                                                       tool_codeobj_tracing_callback,
                                                       tool_data),
        "code object tracing service configure");

    rocprofiler_att_parameters_t parameters{};
    parameters.target_cu   = TARGET_CU;
    parameters.simd_select = SIMD_SELECT;
    parameters.buffer_size = BUFFER_SIZE;

    std::vector<int> shaders;
    for(size_t i = 0; i < NUM_SE; i++)
        shaders.push_back(2 * i);  // use shader engines 0, 2

    parameters.shader_ids = shaders.data();
    parameters.shader_num = shaders.size();

    ROCPROFILER_CALL(
        rocprofiler_configure_thread_trace_service(
            client_ctx, parameters, dispatch_callback, shader_data_callback, tool_data),
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

    std::unique_lock<std::shared_mutex> isa_lk(tool.isa_map_mut);
    std::unique_lock<std::mutex>        out_lk(tool.output_mut);

    // Find largest instruction
    size_t max_inst_size = 0;
    for(auto& [addr, lines] : tool.isa_map)
        if(lines.get()) max_inst_size = std::max(max_inst_size, lines->code_line->inst.size());

    std::string empty_space;
    empty_space.resize(max_inst_size, ' ');

    size_t vmc_latency    = 0;
    size_t lgk_latency    = 0;
    size_t scalar_latency = 0;
    size_t vector_latency = 0;
    size_t other_latency  = 0;

    size_t scalar_exec = 0;
    size_t vector_exec = 0;
    size_t other_exec  = 0;

    for(auto& [addr, line] : tool.isa_map)
        if(line.get())
        {
            size_t hitcount  = line->hitcount.load(std::memory_order_relaxed);
            size_t latency   = line->latency.load(std::memory_order_relaxed);
            auto&  code_line = line->code_line->inst;

            tool.output << std::hex << "0x" << addr.addr << std::dec << ' ' << code_line
                        << empty_space.substr(0, max_inst_size - code_line.size())
                        << " Hit: " << hitcount << " - Latency: " << latency << '\n';

            if(code_line.find("s_waitcnt") == 0)
            {
                other_exec += hitcount;
                if(code_line.find("lgkmcnt") != std::string::npos)
                    lgk_latency += latency;
                else
                    vmc_latency += latency;
            }
            else if(code_line.find("v_") == 0)
            {
                vector_exec += hitcount;
                vector_latency += latency;
            }
            else if(code_line.find("s_") == 0)
            {
                scalar_exec += hitcount;
                scalar_latency += latency;
            }
            else
            {
                other_exec += hitcount;
                other_latency += latency;
            }
        }

    size_t total_exec     = vector_exec + scalar_exec + other_exec;
    size_t memory_latency = vmc_latency + lgk_latency;
    size_t total_latency  = memory_latency + vector_latency + scalar_latency + other_latency;
    float  vmc_fraction   = 100 * vmc_latency / float(total_latency);
    float  lgk_fraction   = 100 * lgk_latency / float(total_latency);

    tool.output << "Total executed instructions: " << total_exec << '\n'
                << "Total executed vector instructions: " << vector_exec << " with average "
                << vector_latency / float(vector_exec) << " cycles.\n"
                << "Total executed scalar instructions: " << scalar_exec << " with average "
                << scalar_latency / float(scalar_exec) << " cycles.\n"
                << "Vector memory ops occupied: " << vmc_fraction << "% of cycles.\n"
                << "Scalar and LDS memory ops occupied: " << lgk_fraction << "% of cycles.\n";

    std::cout << tool.output.str();
}

void
setup()
{
    if(int status = 0;
       rocprofiler_is_initialized(&status) == ROCPROFILER_STATUS_SUCCESS && status == 0)
    {
        ROCPROFILER_CALL(rocprofiler_force_configure(&rocprofiler_configure),
                         "force configuration");
    }
}

// force configuration when library is loaded
bool cfg_on_load = (client::setup(), true);
}  // namespace client

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
    // only activate if main tool
    if(priority > 0) return nullptr;

    // set the client name
    id->name = "Adv_Thread_Trace_Sample";

    // store client info
    client::client_id = id;

    // compute major/minor/patch version info
    uint32_t major = version / 10000;
    uint32_t minor = (version % 10000) / 100;
    uint32_t patch = version % 100;

    // generate info string
    auto info = std::stringstream{};
    info << id->name << " is using rocprofiler-sdk v" << major << "." << minor << "." << patch
         << " (" << runtime_version << ")";

    std::clog << info.str() << std::endl;

    auto* data = new client::ToolData{};

    // create configure data
    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &client::tool_init,
                                            &client::tool_fini,
                                            reinterpret_cast<void*>(data)};

    // return pointer to configure data
    return &cfg;
}
