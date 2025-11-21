// MIT License
//
// Copyright (c) 2024-2025 ROCm Developer Tools
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

// undefine NDEBUG so asserts are implemented
#ifdef NDEBUG
#    undef NDEBUG
#endif

/**
 * @file samples/pc_sampling_library/client.cpp
 *
 * @brief Example rocprofiler client (tool)
 */

#include "address_translation.hpp"
#include "client.hpp"
#include "pcs.hpp"
#include "utils.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <cxxabi.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <regex>
#include <string>
#include <string_view>
#include <vector>

namespace client
{
namespace codeobj
{
#define CODEOBJ_DEBUG 0

constexpr bool COPY_MEMORY_CODEOBJ = true;

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
codeobj_tracing_callback(rocprofiler_callback_tracing_record_t record,
                         rocprofiler_user_data_t* /*user_data*/,
                         void* /*callback_data*/)
{
    std::stringstream info;

    info << "-----------------------------\n";
    if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
       record.operation == ROCPROFILER_CODE_OBJECT_LOAD)
    {
        auto* data =
            static_cast<rocprofiler_callback_tracing_code_object_load_data_t*>(record.payload);

        if(record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD)
        {
            auto& global_mut = address_translation::get_global_mutex();
            {
                auto lock = std::unique_lock{global_mut};

                auto& translator = client::address_translation::get_address_translator();
                // register code object inside the decoder
                if(std::string_view(data->uri).find("file:///") == 0)
                {
                    translator.addDecoder(
                        data->uri, data->code_object_id, data->load_delta, data->load_size);
                }
                else if(COPY_MEMORY_CODEOBJ)
                {
                    translator.addDecoder(reinterpret_cast<const void*>(data->memory_base),
                                          data->memory_size,
                                          data->code_object_id,
                                          data->load_delta,
                                          data->load_size);
                }
                else
                {
                    return;
                }

                // extract symbols from code object
                auto& kernel_object_map = client::address_translation::get_kernel_object_map();
                auto  symbolmap         = translator.getSymbolMap(data->code_object_id);
                for(auto& [vaddr, symbol] : symbolmap)
                {
                    kernel_object_map.add_kernel(
                        data->code_object_id, symbol.name, vaddr, vaddr + symbol.mem_size);
                }
            }

            info << "code object load :: ";
        }
        else if(record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
        {
            // Ensure all PC samples of the unloaded code object are decoded,
            // prior to removing the decoder.
            client::sync();
            auto& global_mut = address_translation::get_global_mutex();
            {
                auto  lock       = std::unique_lock{global_mut};
                auto& translator = client::address_translation::get_address_translator();
                translator.removeDecoder(data->code_object_id, data->load_delta);
            }

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

        info << std::endl;
    }
    if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
       record.operation == ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
    {
        auto* data =
            static_cast<rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t*>(
                record.payload);

        if(record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD)
        {
            info << "kernel symbol load :: ";
        }
        else if(record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
        {
            info << "kernel symbol unload :: ";
            // client_kernels.erase(data->kernel_id);
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

        info << std::endl;
    }

    *utils::get_output_stream() << info.str() << std::endl;
}

}  // namespace codeobj
}  // namespace client
