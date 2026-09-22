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

#define OUTPUT_OFSTREAM "code_obj_isa_decode.log"

/**
 * @file samples/code_object_isa_decode/client.cpp
 *
 * @brief Example rocprofiler client (tool)
 */

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/cxx/codeobj/code_printing.hpp>

#include "common/defines.hpp"
#include "common/filesystem.hpp"

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
#include <vector>

constexpr bool COPY_MEMORY_CODEOBJ = true;

namespace client
{
std::ostream&
output_stream()
{
    static std::ofstream file(OUTPUT_OFSTREAM);

    static bool file_is_open_check = [&]() {
        if(!file.is_open())
            std::cout << "Could not open log file: " << OUTPUT_OFSTREAM << ", writing to stdout\n";
        else
            std::cout << "Writing code-object-isa-decode log to: " << OUTPUT_OFSTREAM << std::endl;
        return file.is_open();
    }();

    if(!file_is_open_check) return std::cout;
    return file;
};

namespace
{
using code_obj_load_data_t = rocprofiler_callback_tracing_code_object_load_data_t;
using kernel_symbol_data_t = rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;
using kernel_symbol_map_t  = std::unordered_map<std::string, std::pair<uint64_t, size_t>>;

using Instruction             = rocprofiler::sdk::codeobj::disassembly::Instruction;
using CodeobjAddressTranslate = rocprofiler::sdk::codeobj::disassembly::CodeobjAddressTranslate;

rocprofiler_client_id_t*      client_id          = nullptr;
rocprofiler_client_finalize_t client_fini_func   = nullptr;
rocprofiler_context_id_t      client_ctx         = {0};
kernel_symbol_map_t           registered_kernels = {};

CodeobjAddressTranslate codeobjTranslate;

void
tool_codeobj_tracing_callback(rocprofiler_callback_tracing_record_t record,
                              rocprofiler_user_data_t*              user_data,
                              void*                                 callback_data)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT) return;
    if(record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD) return;

    if(record.operation == ROCPROFILER_CODE_OBJECT_LOAD)
    {
        auto* data = static_cast<code_obj_load_data_t*>(record.payload);

        if(std::string_view(data->uri).find("file:///") == 0)
        {
            codeobjTranslate.addDecoder(
                data->uri, data->code_object_id, data->load_delta, data->load_size);
        }
        else if(COPY_MEMORY_CODEOBJ)
        {
            codeobjTranslate.addDecoder(reinterpret_cast<const void*>(data->memory_base),
                                        data->memory_size,
                                        data->code_object_id,
                                        data->load_delta,
                                        data->load_size);
        }
        else
        {
            return;
        }

        auto symbolmap = codeobjTranslate.getSymbolMap();
        for(auto& [vaddr, symbol] : symbolmap)
            registered_kernels.insert({symbol.name, {vaddr, vaddr + symbol.mem_size}});
    }
    else if(record.operation == ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
    {
        output_stream() << std::hex;
        auto* data        = static_cast<kernel_symbol_data_t*>(record.payload);
        auto  kernel_name = std::regex_replace(data->kernel_name, std::regex{"(\\.kd)$"}, "");

        if(registered_kernels.find(kernel_name) == registered_kernels.end())
        {
            output_stream() << "Not Found: " << kernel_name << " in codeobj." << std::endl;
            return;
        }

        auto& begin_end = registered_kernels.at(kernel_name);

        output_stream() << std::hex << "Found: " << kernel_name << " at addr: 0x" << begin_end.first
                        << std::dec << ". Printing first 64 bytes:" << std::endl;

        std::unordered_set<std::string> references{};

        int num_waitcnts = 0;
        int num_scalar   = 0;
        int num_vector   = 0;
        int num_other    = 0;

        size_t vaddr = begin_end.first;
        while(vaddr < begin_end.second)
        {
            auto inst = codeobjTranslate.get(vaddr);
            assert(inst != nullptr);
            if(inst->comment.size())
            {
                std::string_view source = inst->comment;
                if(source.rfind('/') < source.size()) source = source.substr(source.rfind('/'));
                if(vaddr < begin_end.first + 64) output_stream() << '\t' << inst->inst << '\n';

                if(source.rfind(':') < source.size()) source = source.substr(0, source.rfind(':'));

                references.insert(std::string(source));
            }
            if(inst->inst.find("v_") == 0)
                num_vector++;
            else if(inst->inst.find("s_waitcnt") == 0)
                num_waitcnts++;
            else if(inst->inst.find("s_") == 0)
                num_scalar++;
            else
                num_other++;

            vaddr += inst->size;
        }

        output_stream() << "  --- Num Scalar: " << num_scalar
                        << "\n  --- Num Vector: " << num_vector
                        << "\n  --- Num Waitcnts: " << num_waitcnts
                        << "\n  --- Other instructions: " << num_other
                        << "\nKernel has source references to: " << std::endl;
        for(auto& ref : references)
            output_stream() << '\t' << ref << std::endl;
    }

    (void) user_data;
    (void) callback_data;
}

int
tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data)
{
    client_fini_func = fini_func;

    ROCPROFILER_CALL(rocprofiler_create_context(&client_ctx), "context creation");

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(client_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                       nullptr,
                                                       0,
                                                       tool_codeobj_tracing_callback,
                                                       tool_data),
        "code object tracing service configure");

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
tool_fini(void* /* tool_data */)
{}

}  // namespace

}  // namespace client

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
    // set the client name
    id->name = "ExampleTool";

    // store client info
    client::client_id = id;

    // compute major/minor/patch version info
    uint32_t major = version / 10000;
    uint32_t minor = (version % 10000) / 100;
    uint32_t patch = version % 100;

    // generate info string
    auto info = std::stringstream{};
    info << id->name << " (priority=" << priority << ") is using rocprofiler-sdk v" << major << "."
         << minor << "." << patch << " (" << runtime_version << ")";

    std::clog << info.str() << std::endl;

    // create configure data
    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &client::tool_init,
                                            &client::tool_fini,
                                            nullptr};

    // return pointer to configure data
    return &cfg;
}
