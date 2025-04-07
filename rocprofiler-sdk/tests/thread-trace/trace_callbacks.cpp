// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "trace_callbacks.hpp"
#include <rocprofiler-sdk/cxx/codeobj/code_printing.hpp>

#ifdef ENABLE_ATT_FILES
#    include <nlohmann/json.hpp>
#endif

#include <unistd.h>
#include <cassert>
#include <fstream>

namespace Callbacks
{
using code_obj_load_data_t = rocprofiler_callback_tracing_code_object_load_data_t;
using kernel_symbol_data_t = rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;

void
tool_codeobj_tracing_callback(rocprofiler_callback_tracing_record_t record,
                              rocprofiler_user_data_t* /* user_data */,
                              void* userdata)
{
    C_API_BEGIN
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT) return;
    if(record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD) return;

    assert(userdata && "Dispatch callback passed null!");
    auto& tool = *reinterpret_cast<Callbacks::ToolData*>(userdata);

    if(record.operation == ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
    {
        auto* data = static_cast<kernel_symbol_data_t*>(record.payload);
        tool.kernel_id_to_kernel_name.emplace(data->kernel_id, data->kernel_name);
    }

    if(record.operation != ROCPROFILER_CODE_OBJECT_LOAD) return;

    auto* data = static_cast<code_obj_load_data_t*>(record.payload);

    static std::atomic<int> filecnt{0};
    std::string             name = "codeobj_" + std::to_string(filecnt.fetch_add(1)) + ".out";

#ifdef ENABLE_ATT_FILES
    if(std::string_view(data->uri).find("file:///") == 0)
    {
        rocprofiler::sdk::codeobj::disassembly::CodeObjectBinary binary(data->uri);

        std::ofstream file(tool.out_dir + name, std::ios::binary);
        assert(file.is_open() && "Could not open codeobj file for writing");
        file.write((char*) binary.buffer.data(), binary.buffer.size());
    }
    else
    {
        std::ofstream file(tool.out_dir + name, std::ios::binary);
        file.write((char*) data->memory_base, data->memory_size);
    }
#endif

    auto _lk = std::unique_lock{tool.mut};
    tool.codeobjs.push_back(
        {data->load_delta, data->load_size, data->code_object_id, name, data->uri});

    C_API_END
}

void
shader_data_callback(rocprofiler_agent_id_t  agent,
                     int64_t                 se_id,
                     void*                   se_data,
                     size_t                  data_size,
                     rocprofiler_user_data_t userdata)
{
    C_API_BEGIN

    assert(userdata.ptr && "Dispatch callback passed null!");
    auto& tool = *reinterpret_cast<Callbacks::ToolData*>(userdata.ptr);

    std::string name = "agent_" + std::to_string(agent.handle) + "_shader_engine_" +
                       std::to_string(se_id) + "_" + std::to_string(agent.handle) + ".att";

#ifdef ENABLE_ATT_FILES
    {
        std::ofstream file(tool.out_dir + name, std::ios::binary);
        assert(file.is_open() && "Could not open ATT file for writing");
        file.write((char*) se_data, data_size);
    }
#endif

    assert(se_data);
    assert(data_size);

    auto _lk = std::unique_lock{tool.mut};
    tool.att_files.push_back(name);

    C_API_END
}

void
finalize_json(void* userdata)
{
    assert(userdata && "Dispatch callback passed null!");

    auto& tool = *reinterpret_cast<Callbacks::ToolData*>(userdata);
    auto  _lk  = std::unique_lock{tool.mut};
    assert(!tool.att_files.empty());

#ifdef ENABLE_ATT_FILES
    nlohmann::json att_json;
    for(auto& file : tool.att_files)
        att_json.push_back(file);

    nlohmann::json codeobj_json;
    nlohmann::json snapshot_json;
    for(auto& file : tool.codeobjs)
    {
        nlohmann::json codeobj;
        codeobj["code_object_id"] = file.id;
        codeobj["load_delta"]     = file.addr;
        codeobj["load_size"]      = file.size;
        codeobj["uri"]            = file.uri;
        codeobj["filename"]       = file.filename;
        codeobj_json.push_back(codeobj);

        nlohmann::json pair_json;
        pair_json["key"]   = file.id;
        pair_json["value"] = file.filename;
        snapshot_json.push_back(pair_json);
    }
    nlohmann::json tool_json;
    tool_json["strings"]["att_files"]                  = att_json;
    tool_json["code_objects"]                          = codeobj_json;
    tool_json["strings"]["code_object_snapshot_files"] = snapshot_json;

    nlohmann::json array;
    array.push_back(tool_json);

    nlohmann::json sdk_json;
    sdk_json["rocprofiler-sdk-tool"] = array;

    std::ofstream json_file(tool.out_dir + (std::to_string(getpid()) + "_results.json"));
    assert(json_file.is_open() && "Could not open json file for writing!");
    json_file << sdk_json;
#endif
}

}  // namespace Callbacks
