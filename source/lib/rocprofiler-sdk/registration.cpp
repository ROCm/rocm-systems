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

#define _GNU_SOURCE 1

#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/common/elf_utils.hpp"
#include "lib/common/environment.hpp"
#include "lib/common/filesystem.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/static_tl_object.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/code_object/code_object.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/context/correlation_id.hpp"
#include "lib/rocprofiler-sdk/hip/hip.hpp"
#include "lib/rocprofiler-sdk/hip/stream.hpp"
#include "lib/rocprofiler-sdk/hsa/async_copy.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/memory_allocation.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/hsa/scratch_memory.hpp"
#include "lib/rocprofiler-sdk/intercept_table.hpp"
#include "lib/rocprofiler-sdk/internal_threading.hpp"
#include "lib/rocprofiler-sdk/kfd/kfd.hpp"
#include "lib/rocprofiler-sdk/marker/marker.hpp"
#include "lib/rocprofiler-sdk/ompt.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/code_object.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/service.hpp"
#include "lib/rocprofiler-sdk/rccl/rccl.hpp"
#include "lib/rocprofiler-sdk/rocdecode/rocdecode.hpp"
#include "lib/rocprofiler-sdk/rocjpeg/rocjpeg.hpp"
#include "lib/rocprofiler-sdk/runtime_initialization.hpp"

#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/ext_version.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hip.h>
#include <rocprofiler-sdk/hsa.h>
#include <rocprofiler-sdk/marker.h>
#include <rocprofiler-sdk/ompt.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/version.h>

#include <fmt/format.h>

#include <dlfcn.h>
#include <link.h>
#include <unistd.h>
#include <atomic>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <unordered_set>
#include <vector>

extern "C" {
#pragma weak rocprofiler_configure

extern rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t*);

#if defined(CODECOV) && CODECOV > 0
extern void
__gcov_dump(void);
#endif
}

namespace rocprofiler
{
namespace registration
{
namespace
{
namespace fs = ::rocprofiler::common::filesystem;

bool
resolved_exists(std::string_view fname)
{
    if(fs::is_symlink(fname))
    {
        // NOTE: Use of ROCP_CI_LOG(WARNING) causes segfault. Likely bc glog is not fully
        // initialized
        auto _errc      = std::error_code{};
        auto _symlinked = fs::read_symlink(fname, _errc);
        if(_errc && _symlinked.empty())
        {
            ROCP_WARNING << fmt::format("Symbolic link '{}' returned error code {} :: {}",
                                        fname,
                                        _errc.value(),
                                        _errc.message());
            return false;
        }
        else if(_errc && !_symlinked.empty())
        {
            ROCP_WARNING << fmt::format("Symbolic link '{}' -> '{}' returned error code {} :: {}",
                                        fname,
                                        _symlinked.string(),
                                        _errc.value(),
                                        _errc.message());
            return false;
        }

        if(_symlinked.is_relative()) _symlinked = fs::path{fname}.parent_path() / _symlinked;

        ROCP_TRACE << fmt::format("Symbolic link:\n\t{}\n\t\t-> {}", fname, _symlinked.string());

        if(!fs::exists(_symlinked))
        {
            ROCP_WARNING << fmt::format("{} is broken symbolic link", fname);
            return false;
        }

        return resolved_exists(fs::absolute(_symlinked).string());
    }

    return fs::exists(fname);
}

// invoke all rocprofiler_configure symbols
bool
invoke_client_configures();

// invoke initialize functions returned from rocprofiler_configure
bool
invoke_client_initializers();

// invoke finalize functions returned from rocprofiler_configure
bool
invoke_client_finalizers();

// explicitly invoke the finalize function of a specific client
void invoke_client_finalizer(rocprofiler_client_id_t);

auto*
get_status()
{
    static auto*& _v =
        common::static_object<std::pair<std::atomic<int>, std::atomic<int>>>::construct(0, 0);
    return _v;
}

auto&
get_invoked_configures()
{
    static auto _v = std::unordered_set<rocprofiler_configure_func_t>{};
    return _v;
}

auto&
get_forced_configure()
{
    static rocprofiler_configure_func_t _v = nullptr;
    return _v;
}

std::vector<std::string>
get_link_map()
{
    auto  chain  = std::vector<std::string>{};
    void* handle = dlopen(nullptr, RTLD_LAZY | RTLD_NOLOAD);

    if(handle)
    {
        struct link_map* link_map_v = nullptr;
        dlinfo(handle, RTLD_DI_LINKMAP, &link_map_v);
        struct link_map* next_link = link_map_v->l_next;
        while(next_link)
        {
            if(next_link->l_name != nullptr && !std::string_view{next_link->l_name}.empty())
            {
                chain.emplace_back(next_link->l_name);
            }
            next_link = next_link->l_next;
        }
    }

    return chain;
}

struct client_library
{
    client_library() = default;
    ~client_library() { delete configure_result; }

    client_library(const client_library&)     = delete;
    client_library(client_library&&) noexcept = default;

    client_library& operator=(const client_library&) = delete;
    client_library& operator=(client_library&&) noexcept = delete;

    std::string                          name               = {};
    void*                                dlhandle           = nullptr;
    decltype(::rocprofiler_configure)*   configure_func     = nullptr;
    rocprofiler_tool_configure_result_t* configure_result   = nullptr;
    rocprofiler_client_id_t              internal_client_id = {};
    rocprofiler_client_id_t              mutable_client_id  = {};
};

using client_library_vec_t = std::vector<std::optional<client_library>>;

client_library_vec_t
find_clients()
{
    auto data            = client_library_vec_t{};
    auto priority_offset = get_client_offset();

    auto is_unique_configure_func = [&data](auto* _cfg_func) {
        for(const auto& itr : data)
        {
            if(itr && itr->configure_func && itr->configure_func == _cfg_func) return false;
        }
        return true;
    };

    auto emplace_client = [&data, priority_offset](
                              std::string_view _name,
                              void*            _dlhandle,
                              auto*            _cfg_func) -> std::optional<client_library>& {
        constexpr auto client_id_size = sizeof(rocprofiler_client_id_t);
        uint32_t       _prio          = priority_offset + data.size();
        return data.emplace_back(
            client_library{std::string{_name},
                           _dlhandle,
                           _cfg_func,
                           nullptr,
                           rocprofiler_client_id_t{client_id_size, nullptr, _prio},
                           rocprofiler_client_id_t{client_id_size, nullptr, _prio}});
    };

    auto rocprofiler_configure_dlsym = [](auto _handle) {
        decltype(::rocprofiler_configure)* _sym = nullptr;
        *(void**) (&_sym)                       = dlsym(_handle, "rocprofiler_configure");
        return _sym;
    };

    if(get_forced_configure() && is_unique_configure_func(get_forced_configure()))
    {
        ROCP_INFO << "adding forced configure";
        emplace_client("(forced)", nullptr, get_forced_configure());
    }

    auto get_env_libs = []() {
        auto       val       = common::get_env("ROCP_TOOL_LIBRARIES", std::string{});
        auto       val_arr   = std::vector<std::string>{};
        size_t     pos       = 0;
        const auto delimiter = std::string_view{":"};
        auto       token     = std::string{};

        if(val.empty())
        {
            // do nothing
        }
        else if(val.find(delimiter) == std::string::npos)
        {
            val_arr.emplace_back(val);
        }
        else
        {
            while((pos = val.find(delimiter)) != std::string::npos)
            {
                token = val.substr(0, pos);
                if(!token.empty()) val_arr.emplace_back(token);
                val.erase(0, pos + delimiter.length());
            }
        }
        return val_arr;
    };

    auto env = get_env_libs();

    if(!env.empty())
    {
        for(const auto& itr : env)
        {
            ROCP_INFO << "[ROCP_TOOL_LIBRARIES] searching " << itr << " for rocprofiler_configure";

            if(fs::exists(itr) && resolved_exists(itr))
            {
                auto elfinfo = common::elf_utils::read(itr);
                if(!elfinfo.has_symbol([](std::string_view symname) {
                       return (symname == "rocprofiler_configure");
                   }))
                {
                    ROCP_CI_LOG(WARNING) << fmt::format(
                        "[ROCP_TOOL_LIBRARIES] rocprofiler-sdk tool library '{}' did not "
                        "contain rocprofiler_configure symbol (search method: ELF parsing). "
                        "Attempting dlopen anyway since the library was explicitly listed in "
                        "ROCP_TOOL_LIBRARIES",
                        itr);
                }
            }

            void* handle = dlopen(itr.c_str(), RTLD_NOLOAD | RTLD_LAZY);

            if(!handle)
            {
                ROCP_INFO << "[ROCP_TOOL_LIBRARIES] '" << itr
                          << "' is not already loaded, doing a local lazy dlopen...";
                handle = dlopen(itr.c_str(), RTLD_LOCAL | RTLD_LAZY);
            }

            if(!handle)
            {
                ROCP_FATAL << "[ROCP_TOOL_LIBRARIES] error dlopening '" << itr << "'";
            }

            for(const auto& ditr : data)
            {
                if(ditr->dlhandle && ditr->dlhandle == handle)
                {
                    handle = nullptr;
                    break;
                }
            }

            if(handle)
            {
                auto _sym = rocprofiler_configure_dlsym(handle);
                // FATAL bc they explicitly said this was a tool library
                ROCP_CI_LOG_IF(WARNING, !_sym)
                    << "[ROCP_TOOL_LIBRARIES] rocprofiler-sdk tool library '" << itr
                    << "' did not contain rocprofiler_configure symbol (search method: dlsym)";
                if(_sym && is_unique_configure_func(_sym)) emplace_client(itr, handle, _sym);
            }
        }
    }

    if(rocprofiler_configure && is_unique_configure_func(rocprofiler_configure))
        emplace_client("unknown", nullptr, rocprofiler_configure);

    auto _default_configure = rocprofiler_configure_dlsym(RTLD_DEFAULT);
    auto _next_configure    = rocprofiler_configure_dlsym(RTLD_NEXT);

    if(_default_configure && is_unique_configure_func(_default_configure))
        emplace_client("(RTLD_DEFAULT)", nullptr, _default_configure);

    if(_next_configure && is_unique_configure_func(_next_configure))
        emplace_client("(RTLD_NEXT)", nullptr, _next_configure);

    // if there are two "rocprofiler_configures", we need to trigger a search of all the shared
    // libraries
    if(_default_configure)
    {
        for(const auto& itr : get_link_map())
        {
            ROCP_INFO << "searching " << itr << " for rocprofiler_configure";

            if(fs::exists(itr) && resolved_exists(itr))
            {
                auto elfinfo = common::elf_utils::read(itr);
                if(!elfinfo.has_symbol([](std::string_view symname) {
                       return (symname == "rocprofiler_configure");
                   }))
                {
                    ROCP_INFO << fmt::format(
                        "Shared library '{}' did not contain the 'rocprofiler_configure' symbol "
                        "(search method: ELF parsing) required by rocprofiler-sdk for tools",
                        itr);
                    continue;
                }
            }
            else
            {
                ROCP_INFO << fmt::format(
                    "Shared library '{}' either does not exist or is a broken symbolic link", itr);
                continue;
            }

            ROCP_INFO << "dlopening " << itr << " for rocprofiler_configure";

            void* handle = dlopen(itr.c_str(), RTLD_LAZY | RTLD_NOLOAD);
            ROCP_ERROR_IF(handle == nullptr) << "error dlopening " << itr;

            auto* _sym = rocprofiler_configure_dlsym(handle);

            // symbol not found
            if(!_sym)
            {
                ROCP_INFO << "|_" << itr << " did not contain rocprofiler_configure symbol";
                continue;
            }

            // skip the configure function that was forced
            if(_sym == get_forced_configure())
            {
                data.front()->name                    = itr;
                data.front()->dlhandle                = handle;
                data.front()->internal_client_id.name = "(forced)";
                continue;
            }

            if(_sym == &rocprofiler_configure && data.size() == 1)
            {
                data.front()->name                    = itr;
                data.front()->dlhandle                = handle;
                data.front()->internal_client_id.name = "default";
            }
            else if(is_unique_configure_func(_sym))
            {
                auto& entry                    = emplace_client(itr, handle, _sym);
                entry->internal_client_id.name = entry->name.c_str();
            }
        }
    }

    ROCP_INFO << __FUNCTION__ << " found " << data.size() << " clients";

    return data;
}

client_library_vec_t*
get_clients()
{
    static auto*& _v = common::static_object<client_library_vec_t>::construct(find_clients());
    return _v;
}

uint64_t
get_num_clients()
{
    uint64_t val = 0;
    if(get_clients())
    {
        for(auto& itr : *get_clients())
        {
            if(itr && itr->configure_result != nullptr) val += 1;
        }
    }
    return val;
}

using mutex_t       = std::mutex;
using scoped_lock_t = std::unique_lock<mutex_t>;

mutex_t&
get_registration_mutex()
{
    static auto _v = mutex_t{};
    return _v;
}

bool
invoke_client_configures()
{
    if(get_init_status() > 0) return false;

    auto _lk = scoped_lock_t{get_registration_mutex()};

    ROCP_INFO << __FUNCTION__;

    if(!get_clients()) return false;

    for(auto& itr : *get_clients())
    {
        if(!itr) continue;

        if(!itr->configure_func)
        {
            ROCP_ERROR << "rocprofiler::registration::invoke_client_configures() attempted to "
                          "invoke configure function from "
                       << itr->name << " that had no configuration function";
            continue;
        }

        if(get_invoked_configures().find(itr->configure_func) != get_invoked_configures().end())
        {
            ROCP_ERROR << "rocprofiler::registration::invoke_client_configures() attempted to "
                          "invoke configure function from "
                       << itr->name << " (addr="
                       << fmt::format("{:#018x}", reinterpret_cast<uint64_t>(itr->configure_func))
                       << ") more than once";
            continue;
        }
        else
        {
            ROCP_INFO << "rocprofiler::registration::invoke_client_configures() invoking configure "
                         "function from "
                      << itr->name << " (addr="
                      << fmt::format("{:#018x}", reinterpret_cast<uint64_t>(itr->configure_func))
                      << ")";
        }

        auto* _result = itr->configure_func(ROCPROFILER_VERSION,
                                            ROCPROFILER_VERSION_STRING,
                                            itr->internal_client_id.handle - get_client_offset(),
                                            &itr->mutable_client_id);

        if(_result)
        {
            itr->configure_result = new rocprofiler_tool_configure_result_t{*_result};
        }
        else
        {
            context::deactivate_client_contexts(itr->internal_client_id);
            context::deregister_client_contexts(itr->internal_client_id);
        }

        get_invoked_configures().emplace(itr->configure_func);
    }

    return true;
}

bool
invoke_client_initializers()
{
    if(get_init_status() > 0) return false;

    auto _lk = scoped_lock_t{get_registration_mutex()};

    ROCP_INFO << __FUNCTION__;

    if(!get_clients()) return false;

    // if there is only one client, just fully finalize
    rocprofiler_client_finalize_t client_fini_func =
        (get_clients()->size() == 1) ? [](rocprofiler_client_id_t) -> void { finalize(); }
                                     : &invoke_client_finalizer;

    for(auto& itr : *get_clients())
    {
        if(itr && itr->configure_result && itr->configure_result->initialize)
        {
            context::push_client(itr->internal_client_id.handle);
            itr->configure_result->initialize(client_fini_func, itr->configure_result->tool_data);
            context::pop_client(itr->internal_client_id.handle);
            // set to nullptr so initialize only gets called once
            itr->configure_result->initialize = nullptr;
        }
    }

    return true;
}

bool
invoke_client_finalizers()
{
    // NOTE: this function is expected to only be invoked from the finalize function (which sets the
    // fini status)

    if(get_init_status() < 1 || get_fini_status() > 0) return false;

    if(get_clients())
    {
        for(auto& itr : *get_clients())
        {
            if(itr) invoke_client_finalizer(itr->internal_client_id);
        }
    }

    return true;
}

void
invoke_client_finalizer(rocprofiler_client_id_t client_id)
{
    ROCP_INFO << __FUNCTION__ << "(client_id=" << client_id.handle << ")";

    auto _lk = scoped_lock_t{get_registration_mutex()};

    if(!get_clients()) return;

    for(auto& itr : *get_clients())
    {
        if(itr && itr->internal_client_id.handle == client_id.handle &&
           itr->mutable_client_id.handle == client_id.handle)
        {
            context::stop_client_contexts(itr->internal_client_id);
            if(itr->configure_result && itr->configure_result->finalize)
            {
                // set to nullptr so finalize only gets called once
                rocprofiler_tool_finalize_t _finalize_func = nullptr;
                std::swap(_finalize_func, itr->configure_result->finalize);

                hsa::async_copy_sync();
                hsa::queue_controller_sync();
                pc_sampling::service_sync();

                auto _fini_status = get_fini_status();
                if(_fini_status == 0) set_fini_status(-1);
                _finalize_func(itr->configure_result->tool_data);
                if(_fini_status == 0) set_fini_status(_fini_status);
            }
            context::deactivate_client_contexts(itr->internal_client_id);
            itr.reset();
        }
    }
}
}  // namespace

void
init_logging()
{
    common::init_logging("ROCPROFILER");
}

// ensure that logging is always initialized when library is loaded
bool init_logging_at_load = (init_logging(), true);

uint32_t
get_client_offset()
{
    static uint32_t _v = []() {
        auto gen = std::mt19937{std::random_device{}()};
        auto rng = std::uniform_int_distribution<uint32_t>{
            std::numeric_limits<uint8_t>::max(),
            std::numeric_limits<uint32_t>::max() - std::numeric_limits<uint8_t>::max()};
        return rng(gen);
    }();
    return _v;
}

int
get_init_status()
{
    return (get_status()) ? get_status()->first.load(std::memory_order_acquire) : 1;
}

int
get_fini_status()
{
    return (get_status()) ? get_status()->second.load(std::memory_order_acquire) : 1;
}

void
set_init_status(int v)
{
    if(get_status()) get_status()->first.store(v, std::memory_order_release);
}

void
set_fini_status(int v)
{
    if(get_status()) get_status()->second.store(v, std::memory_order_release);
}

void
initialize()
{
    ROCP_INFO << "rocprofiler initialize called...";

    if(get_init_status() != 0)
    {
        ROCP_INFO << "rocprofiler initialize ignored...";
        return;
    }

    static auto _once = std::once_flag{};
    std::call_once(_once, []() {
        ROCP_INFO << "rocprofiler initialize started...";
        // initialization is in process
        set_init_status(-1);
        std::atexit([]() {
            finalize();
            common::destroy_static_tl_objects();
            common::destroy_static_objects();
        });
        init_logging();
        invoke_client_configures();
        invoke_client_initializers();
        if(get_num_clients() > 0) internal_threading::initialize();
        // initialization is no longer available
        set_init_status(1);

        if(get_num_clients() > 0)
        {
            for(const auto& itr : *get_clients())
            {
                if(!itr) continue;
                size_t _client_registered_ctx = 0;
                for(const auto* citr : context::get_registered_contexts())
                {
                    if(citr->client_idx == itr->internal_client_id.handle) ++_client_registered_ctx;
                }

                size_t _client_activated_ctx = 0;
                for(const auto* citr : context::get_active_contexts())
                {
                    if(citr->client_idx == itr->internal_client_id.handle) ++_client_activated_ctx;
                }

                ROCP_INFO << fmt::format("rocprofiler-sdk client '{}' registered {} context(s) and "
                                         "started {} context(s)",
                                         (itr->mutable_client_id.name)
                                             ? std::string_view{itr->mutable_client_id.name}
                                             : std::string_view{"unspecified"},
                                         _client_registered_ctx,
                                         _client_activated_ctx);
            }
        }
    });
}

void
finalize()
{
#if defined(CODECOV) && CODECOV > 0
    if(get_fini_status() > 0) __gcov_dump();
#endif

    if(get_fini_status() != 0)
    {
        ROCP_INFO << "ignoring finalization request (value=" << get_fini_status() << ")";
        return;
    }

    static auto _sync = std::atomic_flag{};
    if(_sync.test_and_set())
    {
        ROCP_INFO << "ignoring finalization request [already finalized] (value="
                  << get_fini_status() << ")";
        return;
    }
    // above returns true for all invocations after the first one

    ROCP_INFO << "finalizing rocprofiler (value=" << get_fini_status() << ")";

    static auto _once = std::once_flag{};
    std::call_once(_once, []() {
        auto num_clients = get_num_clients();
        set_fini_status(-1);
        hsa::async_copy_fini();
        counters::device_counting_service_finalize();
        hsa::queue_controller_fini();
        thread_trace::finalize();
        ompt::finalize_ompt();
        kfd::finalize();
#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
        // WARNING: this must precede `code_object::finalize()`
        pc_sampling::code_object::finalize();
        // WARNING: this must follows queue_controller_fini.
        pc_sampling::service_fini();
#endif
        code_object::finalize();
        context::correlation_id_finalize();
        if(get_init_status() > 0)
        {
            invoke_client_finalizers();
        }
        if(num_clients > 0) internal_threading::finalize();
        set_fini_status(1);
    });

#if defined(CODECOV) && CODECOV > 0
    __gcov_dump();
#endif
}
}  // namespace registration
}  // namespace rocprofiler

extern "C" {
rocprofiler_status_t
rocprofiler_is_initialized(int* status)
{
    *status = rocprofiler::registration::get_init_status();
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_is_finalized(int* status)
{
    *status = rocprofiler::registration::get_fini_status();
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_force_configure(rocprofiler_configure_func_t configure_func)
{
    rocprofiler::registration::init_logging();

    ROCP_INFO << "forcing rocprofiler configuration";

    auto& forced_config = rocprofiler::registration::get_forced_configure();

    // init status may be -1 (currently initializing) or 1 (already initialized).
    // if either case, we want to ignore this function call but if this is
    if(rocprofiler::registration::get_init_status() != 0)
        return ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED;

    // if another tool forced configure, the init status should be 1, but
    // let's just make sure that the forced configure function is a nullptr
    if(forced_config) return ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED;

    setenv("ROCPROFILER_REGISTER_FORCE_LOAD", "1", 1);
    forced_config = configure_func;
    rocprofiler::registration::initialize();

    return ROCPROFILER_STATUS_SUCCESS;
}

int
rocprofiler_set_api_table(const char* name,
                          uint64_t    lib_version,
                          uint64_t    lib_instance,
                          void**      tables,
                          uint64_t    num_tables)
{
    // implementation has a call once
    rocprofiler::registration::init_logging();

    ROCP_INFO << __FUNCTION__ << "(\"" << name << "\", " << lib_version << ", " << lib_instance
              << ", ..., " << num_tables << ")";

    static auto _once = std::once_flag{};
    std::call_once(_once, rocprofiler::registration::initialize);

    // pass to ROCTx init
    ROCP_ERROR_IF(num_tables == 0) << "rocprofiler expected " << name
                                   << " library to pass at least one table, not " << num_tables;
    ROCP_ERROR_IF(tables == nullptr) << "rocprofiler expected pointer to array of tables from "
                                     << name << " library, not a nullptr";

    if(std::string_view{name} == "hip")
    {
        // pass to hip init
        ROCP_ERROR_IF(num_tables > 1) << "rocprofiler expected HIP library to pass 1 API table for "
                                      << name << ", not " << num_tables;

        auto* hip_runtime_api_table = static_cast<HipDispatchTable*>(*tables);

        // any internal modifications to the HipDispatchTable need to be done before we make the
        // copy or else those modifications will be lost when HIP API tracing is enabled
        // because the HIP API tracing invokes the function pointers from the copy below
        rocprofiler::hip::copy_table(hip_runtime_api_table, lib_instance);

        // install rocprofiler API wrappers
        rocprofiler::hip::update_table(hip_runtime_api_table);

        // Tracing notifications the runtime has initialized
        rocprofiler::runtime_init::initialize(
            ROCPROFILER_RUNTIME_INITIALIZATION_HIP, lib_version, lib_instance);

        // install HIP stream deduction wrappers
        rocprofiler::hip::stream::update_table(hip_runtime_api_table);

        // allow tools to install API wrappers
        rocprofiler::intercept_table::notify_intercept_table_registration(
            ROCPROFILER_HIP_RUNTIME_TABLE,
            lib_version,
            lib_instance,
            std::make_tuple(hip_runtime_api_table));
    }
    else if(std::string_view{name} == "hip_compiler")
    {
        // pass to hip init
        ROCP_ERROR_IF(num_tables > 1) << "rocprofiler expected HIP library to pass 1 API table for "
                                      << name << ", not " << num_tables;

        auto* hip_compiler_api_table = static_cast<HipCompilerDispatchTable*>(*tables);

        // any internal modifications to the HipCompilerDispatchTable need to be done before we make
        // the copy or else those modifications will be lost when HIP API tracing is enabled because
        // the HIP API tracing invokes the function pointers from the copy below
        rocprofiler::hip::copy_table(hip_compiler_api_table, lib_instance);

        rocprofiler::code_object::initialize(hip_compiler_api_table);

        // install rocprofiler API wrappers
        rocprofiler::hip::update_table(hip_compiler_api_table);

        // allow tools to install API wrappers
        rocprofiler::intercept_table::notify_intercept_table_registration(
            ROCPROFILER_HIP_COMPILER_TABLE,
            lib_version,
            lib_instance,
            std::make_tuple(hip_compiler_api_table));
    }
    else if(std::string_view{name} == "hsa")
    {
        // this is a slight hack due to a hsa-runtime bug with rocprofiler-register which
        // causes it to register the API table twice when HSA_TOOL_LIB is set to this
        // rocprofiler library. Fixed in Gerrit review 961592.
        setenv("HSA_TOOLS_ROCPROFILER_V1_TOOLS", "0", 0);

        // pass to hsa init
        ROCP_ERROR_IF(num_tables > 1)
            << "rocprofiler expected HSA library to pass 1 API table, not " << num_tables;

        auto* hsa_api_table = static_cast<HsaApiTable*>(*tables);

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
        auto hsa_api_table_size = hsa_api_table->version.minor_id;
        auto runtime_pc_sampling_table =
            (offsetof(::HsaApiTable, pc_sampling_ext_) < hsa_api_table_size);
#endif

        // store a reference of the HsaApiTable implementations for invoking these functions
        // without going through tracing wrappers
        rocprofiler::hsa::copy_table(hsa_api_table->core_, lib_instance);
        rocprofiler::hsa::copy_table(hsa_api_table->amd_ext_, lib_instance);
        rocprofiler::hsa::copy_table(hsa_api_table->image_ext_, lib_instance);
        rocprofiler::hsa::copy_table(hsa_api_table->finalizer_ext_, lib_instance);
        rocprofiler::hsa::copy_table(hsa_api_table->tools_, lib_instance);
#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
        if(runtime_pc_sampling_table)
            rocprofiler::hsa::copy_table(hsa_api_table->pc_sampling_ext_, lib_instance);
#endif

        // need to construct agent mappings before initializing the queue controller
        rocprofiler::agent::construct_agent_cache(hsa_api_table);
        rocprofiler::thread_trace::initialize(hsa_api_table);
        rocprofiler::hsa::queue_controller_init(hsa_api_table);
        // Process agent ctx's that were started prior to HSA init
        rocprofiler::counters::device_counting_service_hsa_registration();

        rocprofiler::hsa::async_copy_init(hsa_api_table, lib_instance);
        rocprofiler::hsa::memory_allocation_init(hsa_api_table->core_, lib_instance);
        rocprofiler::hsa::memory_allocation_init(hsa_api_table->amd_ext_, lib_instance);
        rocprofiler::code_object::initialize(hsa_api_table);
#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
        if(runtime_pc_sampling_table)
            rocprofiler::pc_sampling::code_object::initialize(hsa_api_table);
#endif
        rocprofiler::thread_trace::code_object::initialize(hsa_api_table);

        // install rocprofiler API wrappers
        rocprofiler::hsa::update_table(hsa_api_table->core_, lib_instance);
        rocprofiler::hsa::update_table(hsa_api_table->amd_ext_, lib_instance);
        rocprofiler::hsa::update_table(hsa_api_table->image_ext_, lib_instance);
        rocprofiler::hsa::update_table(hsa_api_table->finalizer_ext_, lib_instance);
        rocprofiler::hsa::update_table(hsa_api_table->tools_, lib_instance);

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
        // Initialize PC sampling service if configured
        if(runtime_pc_sampling_table)
            rocprofiler::pc_sampling::post_hsa_init_start_active_service();
#endif

        // Tracing notifications the runtime has initialized
        rocprofiler::runtime_init::initialize(
            ROCPROFILER_RUNTIME_INITIALIZATION_HSA, lib_version, lib_instance);

        // allow tools to install API wrappers
        rocprofiler::intercept_table::notify_intercept_table_registration(
            ROCPROFILER_HSA_TABLE, lib_version, lib_instance, std::make_tuple(hsa_api_table));
    }
    else if(std::string_view{name} == "roctx")
    {
        // pass to ROCTx init
        ROCP_FATAL_IF(num_tables < 3)
            << "rocprofiler expected ROCTX library to pass 3 API tables, not " << num_tables;
        ROCP_ERROR_IF(num_tables > 3)
            << "rocprofiler expected ROCTX library to pass 3 API tables, not " << num_tables;

        auto* roctx_core = static_cast<roctxCoreApiTable_t*>(tables[0]);
        auto* roctx_ctrl = static_cast<roctxControlApiTable_t*>(tables[1]);
        auto* roctx_name = static_cast<roctxNameApiTable_t*>(tables[2]);

        // any internal modifications to the roctxApiTable_t need to be done before we make
        // the copy or else those modifications will be lost when ROCTx tracing is enabled because
        // the ROCTx tracing invokes the function pointers from the copy below
        rocprofiler::marker::copy_table(roctx_core, lib_instance);
        rocprofiler::marker::copy_table(roctx_ctrl, lib_instance);
        rocprofiler::marker::copy_table(roctx_name, lib_instance);

        // install rocprofiler API wrappers
        rocprofiler::marker::update_table(roctx_core);
        rocprofiler::marker::update_table(roctx_ctrl);
        rocprofiler::marker::update_table(roctx_name);

        // Tracing notifications the runtime has initialized
        rocprofiler::runtime_init::initialize(
            ROCPROFILER_RUNTIME_INITIALIZATION_MARKER, lib_version, lib_instance);

        // allow tools to install API wrappers
        rocprofiler::intercept_table::notify_intercept_table_registration(
            ROCPROFILER_MARKER_CORE_TABLE, lib_version, lib_instance, std::make_tuple(roctx_core));

        rocprofiler::intercept_table::notify_intercept_table_registration(
            ROCPROFILER_MARKER_CONTROL_TABLE,
            lib_version,
            lib_instance,
            std::make_tuple(roctx_ctrl));

        rocprofiler::intercept_table::notify_intercept_table_registration(
            ROCPROFILER_MARKER_NAME_TABLE, lib_version, lib_instance, std::make_tuple(roctx_name));
    }
    else if(std::string_view{name} == "rccl")
    {
        // pass to rccl init
        ROCP_ERROR_IF(num_tables > 1)
            << "rocprofiler expected RCCL library to pass 1 API table, not " << num_tables;

        auto* rccl_api = static_cast<rcclApiFuncTable*>(tables[0]);

        // any internal modifications to the rcclApiFuncTable need to be done before we make the
        // copy or else those modifications will be lost when RCCL API tracing is enabled
        // because the RCCL API tracing invokes the function pointers from the copy below
        rocprofiler::rccl::copy_table(rccl_api, lib_instance);

        // install rocprofiler API wrappers
        rocprofiler::rccl::update_table(rccl_api);

        // Tracing notifications the runtime has initialized
        rocprofiler::runtime_init::initialize(
            ROCPROFILER_RUNTIME_INITIALIZATION_RCCL, lib_version, lib_instance);

        // allow tools to install API wrappers
        rocprofiler::intercept_table::notify_intercept_table_registration(
            ROCPROFILER_RCCL_TABLE, lib_version, lib_instance, std::make_tuple(rccl_api));
    }
    else if(std::string_view{name} == "rocdecode")
    {
        // pass to rocdecode init
        ROCP_ERROR_IF(num_tables > 1)
            << "rocprofiler expected ROCDecode library to pass 1 API table, not " << num_tables;

        auto* rocdecode_api = static_cast<RocDecodeDispatchTable*>(tables[0]);

        // any internal modifications to the rocdecodeApiFuncTable need to be done before we make
        // the copy or else those modifications will be lost when ROCDecode API tracing is enabled
        // because the ROCDecode API tracing invokes the function pointers from the copy below
        rocprofiler::rocdecode::copy_table(rocdecode_api, lib_instance);

        // install rocprofiler API wrappers
        rocprofiler::rocdecode::update_table(rocdecode_api);

        // Tracing notifications the runtime has initialized
        rocprofiler::runtime_init::initialize(
            ROCPROFILER_RUNTIME_INITIALIZATION_ROCDECODE, lib_version, lib_instance);

        // allow tools to install API wrappers
        rocprofiler::intercept_table::notify_intercept_table_registration(
            ROCPROFILER_ROCDECODE_TABLE, lib_version, lib_instance, std::make_tuple(rocdecode_api));
    }
    else if(std::string_view{name} == "rocjpeg")
    {
        ROCP_ERROR_IF(num_tables > 1)
            << "rocprofiler expected rocJPEG library to pass 1 API table, not " << num_tables;

        auto* rocjpeg_api = static_cast<RocJpegDispatchTable*>(tables[0]);

        // any internal modifications to the rocjpegApiFuncTable need to be done before we make
        // the copy or else those modifications will be lost when rocJPEG API tracing is enabled
        // because the rocJPEG API tracing invokes the function pointers from the copy below
        rocprofiler::rocjpeg::copy_table(rocjpeg_api, lib_instance);

        // install rocprofiler API wrappers
        rocprofiler::rocjpeg::update_table(rocjpeg_api);

        // Tracing notifications the runtime has initialized
        rocprofiler::runtime_init::initialize(
            ROCPROFILER_RUNTIME_INITIALIZATION_ROCJPEG, lib_version, lib_instance);

        // allow tools to install API wrappers
        rocprofiler::intercept_table::notify_intercept_table_registration(
            ROCPROFILER_ROCJPEG_TABLE, lib_version, lib_instance, std::make_tuple(rocjpeg_api));
    }
    else
    {
        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
    }

    (void) lib_version;
    (void) lib_instance;
    (void) tables;
    (void) num_tables;

    return 0;
}
}
