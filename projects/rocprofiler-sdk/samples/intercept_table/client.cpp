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
 * @file samples/intercept_table/client.cpp
 *
 * @brief Example rocprofiler client (tool)
 */

#include "client.hpp"

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "common/defines.hpp"
#include "common/filesystem.hpp"

#include <hip/amd_detail/hip_api_trace.hpp>

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
#include <ratio>
#include <string>
#include <string_view>
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

using call_stack_t          = std::vector<source_location>;
using callback_kind_names_t = std::map<rocprofiler_callback_tracing_kind_t, const char*>;
using callback_kind_operation_names_t =
    std::map<rocprofiler_callback_tracing_kind_t, std::map<uint32_t, const char*>>;
using wrap_count_t = std::pair<source_location, size_t>;

rocprofiler_client_id_t* client_id        = nullptr;
auto*                    client_wrap_data = new std::map<size_t, wrap_count_t>{};
size_t                   func_width       = 0;

void
print_call_stack(const call_stack_t& _call_stack)
{
    auto ofname = std::string{"intercept_table.log"};
    if(auto* eofname = getenv("ROCPROFILER_SAMPLE_OUTPUT_FILE")) ofname = eofname;

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

    const size_t _func_width = std::min<size_t>(func_width, 60);
    size_t       n           = 0;
    *ofs << std::left;
    for(const auto& itr : _call_stack)
    {
        *ofs << std::left << std::setw(2) << ++n << "/" << std::setw(2) << _call_stack.size()
             << " [" << common::fs::path{itr.file}.filename() << ":" << itr.line << "] "
             << std::setw(_func_width) << itr.function;
        if(!itr.context.empty()) *ofs << " :: " << itr.context;
        *ofs << "\n";
    }

    *ofs << std::flush;

    if(cleanup) cleanup(ofs);
}

void
tool_fini(void* tool_data)
{
    assert(tool_data != nullptr);

    auto* _call_stack = static_cast<call_stack_t*>(tool_data);

    size_t wrapped_count = 0;
    for(const auto& itr : *client_wrap_data)
    {
        auto src_loc = itr.second.first;
        src_loc.context += "call_count = " + std::to_string(itr.second.second);
        _call_stack->emplace_back(std::move(src_loc));
        wrapped_count += itr.second.second;
    }

    _call_stack->emplace_back(source_location{__FUNCTION__, __FILE__, __LINE__, ""});

    print_call_stack(*_call_stack);

    delete _call_stack;
    delete client_wrap_data;

    if(wrapped_count == 0)
    {
        throw std::runtime_error{"intercept_table sample did not wrap HIP runtime API table"};
    }
}

template <size_t Idx, typename RetT, typename... Args>
RetT (*underlying_function)(Args...) = nullptr;

template <size_t Idx, typename RetT, typename... Args>
RetT
get_wrapper_function(Args... args)
{
    if(client_wrap_data)
    {
        if(client_wrap_data->at(Idx).second == 0)
            std::clog << "First invocation of wrapped function: '"
                      << client_wrap_data->at(Idx).first.function << "'...\n"
                      << std::flush;

        client_wrap_data->at(Idx).second += 1;
    }

    if(underlying_function<Idx, RetT, Args...>)
        return underlying_function<Idx, RetT, Args...>(args...);
    if constexpr(!std::is_void<RetT>::value) return RetT{};
}

template <size_t Idx, typename RetT, typename... Args>
auto
generate_wrapper(const char* name, uint32_t line, RetT (*func)(Args...))
{
    func_width = std::max(func_width, std::string_view{name}.length());
    client_wrap_data->emplace(Idx, wrap_count_t{source_location{name, __FILE__, line, ""}, 0});

    underlying_function<Idx, RetT, Args...> = func;
    return &get_wrapper_function<Idx, RetT, Args...>;
}

#define GENERATE_WRAPPER(TABLE, FUNC)                                                              \
    TABLE->FUNC##_fn = generate_wrapper<__COUNTER__>(#FUNC, __LINE__, TABLE->FUNC##_fn)

void
api_registration_callback(rocprofiler_intercept_table_t type,
                          uint64_t                      lib_version,
                          uint64_t                      lib_instance,
                          void**                        tables,
                          uint64_t                      num_tables,
                          void*                         user_data)
{
    if(type != ROCPROFILER_HIP_RUNTIME_TABLE)
        throw std::runtime_error{"unexpected library type: " +
                                 std::to_string(static_cast<int>(type))};
    if(lib_instance != 0) throw std::runtime_error{"multiple instances of HIP runtime library"};
    if(num_tables != 1)
        throw std::runtime_error{"expected only one table of type HipDispatchTable"};

    auto* call_stack = static_cast<std::vector<client::source_location>*>(user_data);

    uint32_t major = lib_version / 10000;
    uint32_t minor = (lib_version % 10000) / 100;
    uint32_t patch = lib_version % 100;

    auto info = std::stringstream{};
    info << client_id->name << " is using HIP runtime v" << major << "." << minor << "." << patch;

    std::clog << info.str() << "\n" << std::flush;

    call_stack->emplace_back(client::source_location{__FUNCTION__, __FILE__, __LINE__, info.str()});

    auto* hip_api_table = static_cast<HipDispatchTable*>(tables[0]);

    // common API functions
    GENERATE_WRAPPER(hip_api_table, hipGetDeviceCount);
    GENERATE_WRAPPER(hip_api_table, hipSetDevice);
    GENERATE_WRAPPER(hip_api_table, hipStreamCreate);
    GENERATE_WRAPPER(hip_api_table, hipStreamDestroy);
    GENERATE_WRAPPER(hip_api_table, hipStreamSynchronize);
    GENERATE_WRAPPER(hip_api_table, hipDeviceSynchronize);
    GENERATE_WRAPPER(hip_api_table, hipDeviceReset);
    GENERATE_WRAPPER(hip_api_table, hipGetErrorString);
    // kernel launch
    GENERATE_WRAPPER(hip_api_table, hipExtLaunchKernel);
    GENERATE_WRAPPER(hip_api_table, hipExtLaunchMultiKernelMultiDevice);
    GENERATE_WRAPPER(hip_api_table, hipGraphLaunch);
    GENERATE_WRAPPER(hip_api_table, hipLaunchByPtr);
    GENERATE_WRAPPER(hip_api_table, hipLaunchCooperativeKernel);
    GENERATE_WRAPPER(hip_api_table, hipLaunchCooperativeKernelMultiDevice);
    GENERATE_WRAPPER(hip_api_table, hipLaunchHostFunc);
    GENERATE_WRAPPER(hip_api_table, hipLaunchKernel);
    GENERATE_WRAPPER(hip_api_table, hipModuleLaunchCooperativeKernel);
    GENERATE_WRAPPER(hip_api_table, hipModuleLaunchCooperativeKernelMultiDevice);
    GENERATE_WRAPPER(hip_api_table, hipModuleLaunchKernel);
    GENERATE_WRAPPER(hip_api_table, hipExtModuleLaunchKernel);
    GENERATE_WRAPPER(hip_api_table, hipHccModuleLaunchKernel);
    // memcpy + memset
    GENERATE_WRAPPER(hip_api_table, hipMemcpy);
    GENERATE_WRAPPER(hip_api_table, hipMemcpyAsync);
    GENERATE_WRAPPER(hip_api_table, hipMemset);
    GENERATE_WRAPPER(hip_api_table, hipMemsetAsync);
}
}  // namespace

void
setup()
{}

void
shutdown()
{}
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

    // demonstration of alternative way to get the version info
    {
        auto version_info = std::array<uint32_t, 3>{};
        ROCPROFILER_CALL(
            rocprofiler_get_version(&version_info.at(0), &version_info.at(1), &version_info.at(2)),
            "failed to get version info");

        if(std::array<uint32_t, 3>{major, minor, patch} != version_info)
        {
            throw std::runtime_error{"version info mismatch"};
        }
    }

    // data passed around all the callbacks
    auto* client_tool_data = new std::vector<client::source_location>{};

    // add first entry
    client_tool_data->emplace_back(
        client::source_location{__FUNCTION__, __FILE__, __LINE__, info.str()});

    ROCPROFILER_CALL(
        rocprofiler_at_intercept_table_registration(client::api_registration_callback,
                                                    ROCPROFILER_HIP_RUNTIME_TABLE,
                                                    static_cast<void*>(client_tool_data)),
        "runtime api registration");

    // create configure data
    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            nullptr,
                                            &client::tool_fini,
                                            static_cast<void*>(client_tool_data)};

    // return pointer to configure data
    return &cfg;
}
