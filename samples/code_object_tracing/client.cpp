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

/**
 * @file samples/code_object_tracing/client.cpp
 *
 * @brief Example rocprofiler client (tool)
 */

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "common/defines.hpp"
#include "common/filesystem.hpp"
#include "common/secure_filename.hpp"

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

namespace client
{
namespace
{
struct source_location
{
    std::string function = {};
    std::string file     = {};
    uint32_t    line     = 0;
    std::string context  = {};
};

using call_stack_t         = std::vector<source_location>;
using code_obj_load_data_t = rocprofiler_callback_tracing_code_object_load_data_t;
using kernel_symbol_data_t = rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;
using kernel_symbol_map_t  = std::unordered_map<rocprofiler_kernel_id_t, kernel_symbol_data_t>;

rocprofiler_client_id_t*      client_id        = nullptr;
rocprofiler_client_finalize_t client_fini_func = nullptr;
rocprofiler_context_id_t      client_ctx       = {0};
kernel_symbol_map_t*          client_kernels   = nullptr;

std::string
cxa_demangle(std::string_view _mangled_name, int* _status)
{
    constexpr size_t buffer_len = 4096;
    // return the mangled since there is no buffer
    if(_mangled_name.empty())
    {
        *_status = -2;
        return std::string{};
    }

    auto _demangled_name = std::string{_mangled_name};

    // PARAMETERS to __cxa_demangle
    //  mangled_name:
    //      A NULL-terminated character string containing the name to be demangled.
    //  buffer:
    //      A region of memory, allocated with malloc, of *length bytes, into which the
    //      demangled name is stored. If output_buffer is not long enough, it is expanded
    //      using realloc. output_buffer may instead be NULL; in that case, the demangled
    //      name is placed in a region of memory allocated with malloc.
    //  _buflen:
    //      If length is non-NULL, the length of the buffer containing the demangled name
    //      is placed in *length.
    //  status:
    //      *status is set to one of the following values
    size_t _demang_len = 0;
    char*  _demang = abi::__cxa_demangle(_demangled_name.c_str(), nullptr, &_demang_len, _status);
    switch(*_status)
    {
        //  0 : The demangling operation succeeded.
        // -1 : A memory allocation failure occurred.
        // -2 : mangled_name is not a valid name under the C++ ABI mangling rules.
        // -3 : One of the arguments is invalid.
        case 0:
        {
            if(_demang) _demangled_name = std::string{_demang};
            break;
        }
        case -1:
        {
            char _msg[buffer_len];
            ::memset(_msg, '\0', buffer_len * sizeof(char));
            ::snprintf(_msg,
                       buffer_len,
                       "memory allocation failure occurred demangling %s",
                       _demangled_name.c_str());
            ::perror(_msg);
            break;
        }
        case -2: break;
        case -3:
        {
            char _msg[buffer_len];
            ::memset(_msg, '\0', buffer_len * sizeof(char));
            ::snprintf(_msg,
                       buffer_len,
                       "Invalid argument in: (\"%s\", nullptr, nullptr, %p)",
                       _demangled_name.c_str(),
                       (void*) _status);
            ::perror(_msg);
            break;
        }
        default: break;
    };

    // if it "demangled" but the length is zero, set the status to -2
    if(_demang_len == 0 && *_status == 0) *_status = -2;

    // free allocated buffer
    ::free(_demang);
    return _demangled_name;
}

void
print_call_stack(const call_stack_t& _call_stack)
{
    // Use secure filename validation from common utilities
    auto ofname = common::secure::get_safe_output_filename("code_object_trace.log");

    std::ostream* ofs     = nullptr;
    auto          cleanup = std::function<void(std::ostream*&)>{};

    if(ofname == "stdout")
        ofs = &std::cout;
    else if(ofname == "stderr")
        ofs = &std::cerr;
    else
    {
        ofs = new std::ofstream{ofname};
        if(ofs && *ofs)
            cleanup = [](std::ostream*& _os) { delete _os; };
        else
        {
            std::cerr << "Error outputting to " << ofname << ". Redirecting to stderr...\n";
            ofname = "stderr";
            ofs    = &std::cerr;
        }
    }

    std::cout << "Outputting collected data to " << ofname << "...\n" << std::flush;

    size_t n = 0;
    for(const auto& itr : _call_stack)
    {
        *ofs << std::left << std::setw(2) << ++n << "/" << std::setw(2) << _call_stack.size()
             << " [" << common::fs::path{itr.file}.filename() << ":" << itr.line << "] "
             << std::setw(20) << itr.function;
        if(!itr.context.empty()) *ofs << " :: " << itr.context;
        *ofs << "\n";
    }

    *ofs << std::flush;

    if(cleanup) cleanup(ofs);
}

template <typename Tp>
std::string
as_hex(Tp _v, size_t _width = 16)
{
    auto _ss = std::stringstream{};
    _ss.fill('0');
    _ss << "0x" << std::hex << std::setw(_width) << _v;
    return _ss.str();
}

void
tool_tracing_callback(rocprofiler_callback_tracing_record_t record,
                      rocprofiler_user_data_t*              user_data,
                      void*                                 callback_data)
{
    if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
       record.operation == ROCPROFILER_CODE_OBJECT_LOAD)
    {
        auto* data         = static_cast<code_obj_load_data_t*>(record.payload);
        auto* call_stack_v = static_cast<call_stack_t*>(callback_data);
        auto  info         = std::stringstream{};

        if(record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD)
        {
            info << "code object load :: ";
        }
        else if(record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
        {
            info << "code object unload :: ";
        }

        info << "code_object_id=" << data->code_object_id
             << ", rocp_agent=" << data->rocp_agent.handle << ", uri=" << data->uri
             << ", load_base=" << as_hex(data->load_base) << ", load_size=" << data->load_size
             << ", load_delta=" << as_hex(data->load_delta);
        if(data->storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE)
            info << ", storage_file_descr=" << data->storage_file;
        else if(data->storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY)
            info << ", storage_memory_base=" << as_hex(data->memory_base)
                 << ", storage_memory_size=" << data->memory_size;

        call_stack_v->emplace_back(source_location{__FUNCTION__, __FILE__, __LINE__, info.str()});
    }
    if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
       record.operation == ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
    {
        auto* data         = static_cast<kernel_symbol_data_t*>(record.payload);
        auto* call_stack_v = static_cast<call_stack_t*>(callback_data);
        auto  info         = std::stringstream{};

        if(record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD)
        {
            info << "kernel symbol load :: ";
            client_kernels->emplace(data->kernel_id, *data);
        }
        else if(record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
        {
            info << "kernel symbol unload :: ";
            client_kernels->erase(data->kernel_id);
        }

        auto kernel_name     = std::regex_replace(data->kernel_name, std::regex{"(\\.kd)$"}, "");
        int  demangle_status = 0;
        kernel_name          = cxa_demangle(kernel_name, &demangle_status);

        info << "code_object_id=" << data->code_object_id << ", kernel_id=" << data->kernel_id
             << ", kernel_object=" << as_hex(data->kernel_object)
             << ", kernarg_segment_size=" << data->kernarg_segment_size
             << ", kernarg_segment_alignment=" << data->kernarg_segment_alignment
             << ", group_segment_size=" << data->group_segment_size
             << ", private_segment_size=" << data->private_segment_size
             << ", kernel_name=" << kernel_name;

        call_stack_v->emplace_back(source_location{__FUNCTION__, __FILE__, __LINE__, info.str()});
    }

    (void) user_data;
}

int
tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data)
{
    client_kernels = new kernel_symbol_map_t{};

    assert(tool_data != nullptr);

    auto* call_stack_v = static_cast<call_stack_t*>(tool_data);

    call_stack_v->emplace_back(source_location{__FUNCTION__, __FILE__, __LINE__, ""});

    client_fini_func = fini_func;

    ROCPROFILER_CALL(rocprofiler_create_context(&client_ctx), "context creation");

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(client_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                       nullptr,
                                                       0,
                                                       tool_tracing_callback,
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
tool_fini(void* tool_data)
{
    assert(tool_data != nullptr);

    auto* _call_stack = static_cast<call_stack_t*>(tool_data);
    _call_stack->emplace_back(source_location{__FUNCTION__, __FILE__, __LINE__, ""});

    print_call_stack(*_call_stack);

    delete _call_stack;
    delete client_kernels;
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
}  // namespace

// force configuration when library is loaded
bool cfg_on_load = (client::setup(), true);
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

    auto* client_tool_data = new std::vector<client::source_location>{};

    client_tool_data->emplace_back(
        client::source_location{__FUNCTION__, __FILE__, __LINE__, info.str()});

    // create configure data
    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &client::tool_init,
                                            &client::tool_fini,
                                            static_cast<void*>(client_tool_data)};

    // return pointer to configure data
    return &cfg;
}
