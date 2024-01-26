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
 * @file samples/intercept_table/client.cpp
 *
 * @brief Example rocprofiler client (tool)
 */

#include "client.hpp"

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "common/defines.hpp"
#include "common/filesystem.hpp"

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
#include "common/defines.hpp"
#include "common/filesystem.hpp"

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

rocprofiler_client_id_t*       client_id        = nullptr;
std::map<size_t, wrap_count_t> client_wrap_data = {};

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

    size_t n = 0;
    *ofs << std::left;
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

void
tool_fini(void* tool_data)
{
    assert(tool_data != nullptr);

    auto* _call_stack = static_cast<call_stack_t*>(tool_data);

    size_t wrapped_count = 0;
    for(const auto& itr : client_wrap_data)
    {
        auto src_loc = itr.second.first;
        src_loc.context += "call_count=" + std::to_string(itr.second.second);
        _call_stack->emplace_back(std::move(src_loc));
        wrapped_count += itr.second.second;
    }

    _call_stack->emplace_back(source_location{__FUNCTION__, __FILE__, __LINE__, ""});

    print_call_stack(*_call_stack);

    delete _call_stack;

    if(wrapped_count == 0)
    {
        throw std::runtime_error{"intercept_table sample did not wrap HSA API table"};
    }
}

template <size_t Idx, typename RetT, typename... Args>
auto
generate_wrapper(const char* name, RetT (*func)(Args...))
{
    using functor_type = RetT (*)(Args...);

    client_wrap_data.emplace(Idx, wrap_count_t{source_location{name, __FILE__, __LINE__, ""}, 0});

    static functor_type underlying_func = func;
    static functor_type wrapped_func    = [](Args... args) -> RetT {
        client_wrap_data.at(Idx).second += 1;
        if(underlying_func) return underlying_func(args...);
        if constexpr(!std::is_void<RetT>::value) return RetT{};
    };

    return wrapped_func;
}

#define GENERATE_WRAPPER(TABLE, FUNC)                                                              \
    TABLE->FUNC##_fn = generate_wrapper<__COUNTER__>(#FUNC, TABLE->FUNC##_fn)

void
api_registration_callback(rocprofiler_intercept_table_t type,
                          uint64_t                      lib_version,
                          uint64_t                      lib_instance,
                          void**                        tables,
                          uint64_t                      num_tables,
                          void*                         user_data)
{
    if(type != ROCPROFILER_HSA_TABLE)
        throw std::runtime_error{"unexpected library type: " +
                                 std::to_string(static_cast<int>(type))};
    if(lib_instance != 0) throw std::runtime_error{"multiple instances of HSA runtime library"};
    if(num_tables != 1) throw std::runtime_error{"expected only one table of type HsaApiTable"};

    auto* call_stack = static_cast<std::vector<client::source_location>*>(user_data);

    uint32_t major = lib_version / 10000;
    uint32_t minor = (lib_version % 10000) / 100;
    uint32_t patch = lib_version % 100;

    auto info = std::stringstream{};
    info << client_id->name << " is using HSA runtime v" << major << "." << minor << "." << patch;

    std::clog << info.str() << "\n" << std::flush;

    call_stack->emplace_back(client::source_location{__FUNCTION__, __FILE__, __LINE__, info.str()});

    auto* hsa_api_table = static_cast<HsaApiTable*>(tables[0]);
    GENERATE_WRAPPER(hsa_api_table->core_, hsa_agent_get_info);
    GENERATE_WRAPPER(hsa_api_table->core_, hsa_agent_iterate_isas);
    GENERATE_WRAPPER(hsa_api_table->core_, hsa_code_object_reader_create_from_memory);
    GENERATE_WRAPPER(hsa_api_table->core_, hsa_executable_create_alt);
    GENERATE_WRAPPER(hsa_api_table->core_, hsa_executable_freeze);
    GENERATE_WRAPPER(hsa_api_table->core_, hsa_executable_get_symbol_by_name);
    GENERATE_WRAPPER(hsa_api_table->core_, hsa_executable_iterate_symbols);
    GENERATE_WRAPPER(hsa_api_table->core_, hsa_executable_load_agent_code_object);
    GENERATE_WRAPPER(hsa_api_table->core_, hsa_executable_symbol_get_info);
    GENERATE_WRAPPER(hsa_api_table->core_, hsa_isa_get_info_alt);
    GENERATE_WRAPPER(hsa_api_table->core_, hsa_iterate_agents);
    GENERATE_WRAPPER(hsa_api_table->core_, hsa_queue_add_write_index_screlease);
    GENERATE_WRAPPER(hsa_api_table->core_, hsa_queue_create);
    GENERATE_WRAPPER(hsa_api_table->core_, hsa_queue_load_read_index_relaxed);
    GENERATE_WRAPPER(hsa_api_table->core_, hsa_queue_load_read_index_scacquire);
    GENERATE_WRAPPER(hsa_api_table->core_, hsa_signal_create);
    GENERATE_WRAPPER(hsa_api_table->core_, hsa_signal_destroy);
    GENERATE_WRAPPER(hsa_api_table->core_, hsa_signal_load_relaxed);
    GENERATE_WRAPPER(hsa_api_table->core_, hsa_signal_silent_store_relaxed);
    GENERATE_WRAPPER(hsa_api_table->core_, hsa_signal_store_screlease);
    GENERATE_WRAPPER(hsa_api_table->core_, hsa_signal_wait_scacquire);
    GENERATE_WRAPPER(hsa_api_table->core_, hsa_system_get_info);
    GENERATE_WRAPPER(hsa_api_table->core_, hsa_system_get_major_extension_table);
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
    // only activate if main tool
    if(priority > 0) return nullptr;

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
    info << id->name << " is using rocprofiler-sdk v" << major << "." << minor << "." << patch
         << " (" << runtime_version << ")";

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
                                                    ROCPROFILER_HSA_TABLE,
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
