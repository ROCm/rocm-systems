// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#define TIMEMORY_KOKKOSP_POSTFIX OMNITRACE_PUBLIC_API

#include "api.hpp"
#include "library/components/category_region.hpp"
#include "library/components/fwd.hpp"
#include "library/config.hpp"
#include "library/debug.hpp"
#include "library/defines.hpp"
#include "library/perfetto.hpp"
#include "library/runtime.hpp"

#include <timemory/api/kokkosp.hpp>
#include <timemory/backends/process.hpp>
#include <timemory/hash/types.hpp>
#include <timemory/utility/procfs/maps.hpp>

#include <cstdlib>
#include <sstream>
#include <string>

namespace kokkosp  = ::tim::kokkosp;
namespace category = ::tim::category;
namespace comp     = ::omnitrace::component;

using kokkosp_region = comp::local_category_region<category::kokkos>;

//--------------------------------------------------------------------------------------//

namespace tim
{
template <>
inline auto
invoke_preinit<kokkosp::memory_tracker>(long)
{
    kokkosp::memory_tracker::label()       = "kokkos_memory";
    kokkosp::memory_tracker::description() = "Kokkos Memory tracker";
}
}  // namespace tim

//--------------------------------------------------------------------------------------//

namespace
{
std::string kokkos_banner =
    "#---------------------------------------------------------------------------#";

//--------------------------------------------------------------------------------------//

inline void
setup_kernel_logger()
{
    if((tim::settings::debug() && tim::settings::verbose() >= 3) ||
       omnitrace::config::get_use_kokkosp_kernel_logger())
    {
        kokkosp::logger_t::get_initializer() = [](kokkosp::logger_t& _obj) {
            _obj.initialize<kokkosp::kernel_logger>();
        };
    }
}

}  // namespace

namespace
{
bool                     _standalone_initialized = false;
std::vector<std::string> _initialize_arguments   = {};
}  // namespace

//--------------------------------------------------------------------------------------//

extern "C"
{
    struct Kokkos_Tools_ToolSettings
    {
        bool requires_global_fencing;
        bool padding[255];
    };

    void kokkosp_request_tool_settings(const uint32_t,
                                       Kokkos_Tools_ToolSettings*) OMNITRACE_PUBLIC_API;
    void kokkosp_dual_view_sync(const char*, const void* const,
                                bool) OMNITRACE_PUBLIC_API;
    void kokkosp_dual_view_modify(const char*, const void* const,
                                  bool) OMNITRACE_PUBLIC_API;

    void kokkosp_print_help(char*) {}

    void kokkosp_parse_args(int argc, char** argv)
    {
        OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);
        if(!omnitrace::config::settings_are_configured() &&
           omnitrace::get_state() < omnitrace::State::Active)
        {
            _standalone_initialized = true;

            OMNITRACE_BASIC_VERBOSE_F(0, "Parsing arguments...\n");
            std::string _command_line = {};
            for(int i = 0; i < argc; ++i)
            {
                _initialize_arguments.emplace_back(argv[i]);
                _command_line.append(" ").append(argv[i]);
            }
            if(_command_line.length() > 1) _command_line = _command_line.substr(1);
            tim::set_env("OMNITRACE_COMMAND_LINE", _command_line, 0);
        }
    }

    void kokkosp_declare_metadata(const char* key, const char* value)
    {
        OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);
        tim::manager::add_metadata(key, value);
    }

    void kokkosp_request_tool_settings(const uint32_t             _version,
                                       Kokkos_Tools_ToolSettings* _settings)
    {
        if(_version > 0) _settings->requires_global_fencing = false;
    }

    void kokkosp_init_library(const int loadSeq, const uint64_t interfaceVer,
                              const uint32_t devInfoCount, void* deviceInfo)
    {
        OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);
        tim::consume_parameters(devInfoCount, deviceInfo);

        OMNITRACE_BASIC_VERBOSE_F(
            0, "Initializing omnitrace kokkos connector (sequence %d, version: %llu)... ",
            loadSeq, (unsigned long long) interfaceVer);

        if(_standalone_initialized || (!omnitrace::config::settings_are_configured() &&
                                       omnitrace::get_state() < omnitrace::State::Active))
        {
            auto _kokkos_profile_lib =
                tim::get_env<std::string>("KOKKOS_PROFILE_LIBRARY");
            if(_kokkos_profile_lib.find("libomnitrace.so") != std::string::npos)
            {
                auto _maps = tim::procfs::read_maps(tim::process::get_id());
                auto _libs = std::set<std::string>{};
                for(auto& itr : _maps)
                {
                    auto&& _path = itr.pathname;
                    if(!_path.empty() && _path.at(0) != '[') _libs.emplace(_path);
                }
                for(const auto& itr : _libs)
                {
                    if(itr.find("libomnitrace-dl.so") != std::string::npos)
                    {
                        std::stringstream _libs_str{};
                        for(const auto& litr : _libs)
                            _libs_str << "    " << litr << "\n";
                        OMNITRACE_ABORT(
                            "%s was invoked with libomnitrace.so as the "
                            "KOKKOS_PROFILE_LIBRARY.\n"
                            "However, libomnitrace-dl.so has already been loaded by the "
                            "process.\nTo avoid duplicate collections culminating is an "
                            "error, please set KOKKOS_PROFILE_LIBRARY=%s.\nLoaded "
                            "libraries:\n%s",
                            __FUNCTION__, itr.c_str(), _libs_str.str().c_str());
                    }
                }
            }

            OMNITRACE_BASIC_VERBOSE_F(0, "Initializing omnitrace (standalone)... ");
            auto _mode = tim::get_env<std::string>("OMNITRACE_MODE", "trace");
            auto _arg0 = (_initialize_arguments.empty()) ? std::string{ "unknown" }
                                                         : _initialize_arguments.at(0);

            _standalone_initialized = true;
            omnitrace_set_mpi_hidden(false, false);
            omnitrace_init_hidden(_mode.c_str(), false, _arg0.c_str());
            omnitrace_push_trace_hidden("kokkos_main");
        }

        setup_kernel_logger();

        tim::trait::runtime_enabled<kokkosp::memory_tracker>::set(
            omnitrace::config::get_use_timemory());

        if(omnitrace::get_verbose() >= 0)
        {
            fprintf(stderr, "%sDone\n%s", tim::log::color::info(),
                    tim::log::color::end());
        }
    }

    void kokkosp_finalize_library()
    {
        OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);
        if(_standalone_initialized)
        {
            omnitrace_pop_trace_hidden("kokkos_main");
            OMNITRACE_VERBOSE_F(
                0, "Finalizing kokkos omnitrace connector (standalone)...\n");
            omnitrace_finalize_hidden();
        }
        else
        {
            OMNITRACE_VERBOSE_F(0, "Finalizing kokkos omnitrace connector... ");
            kokkosp::cleanup();
            if(omnitrace::get_verbose() >= 0) fprintf(stderr, "Done\n");
        }
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_begin_parallel_for(const char* name, uint32_t devid, uint64_t* kernid)
    {
        OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);
        auto pname =
            (devid > std::numeric_limits<uint16_t>::max())  // junk device number
                ? TIMEMORY_JOIN(" ", "[kokkos]", name)
                : TIMEMORY_JOIN(" ", TIMEMORY_JOIN("", "[kokkos][dev", devid, ']'), name);
        *kernid = kokkosp::get_unique_id();
        kokkosp::logger_t{}.mark(1, __FUNCTION__, name, *kernid);
        kokkosp::create_profiler<kokkosp_region>(pname, *kernid);
        kokkosp::start_profiler<kokkosp_region>(*kernid);
    }

    void kokkosp_end_parallel_for(uint64_t kernid)
    {
        OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::logger_t{}.mark(-1, __FUNCTION__, kernid);
        kokkosp::stop_profiler<kokkosp_region>(kernid);
        kokkosp::destroy_profiler<kokkosp_region>(kernid);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_begin_parallel_reduce(const char* name, uint32_t devid, uint64_t* kernid)
    {
        OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);
        auto pname =
            (devid > std::numeric_limits<uint16_t>::max())  // junk device number
                ? TIMEMORY_JOIN(" ", "[kokkos]", name)
                : TIMEMORY_JOIN(" ", TIMEMORY_JOIN("", "[kokkos][dev", devid, ']'), name);
        *kernid = kokkosp::get_unique_id();
        kokkosp::logger_t{}.mark(1, __FUNCTION__, name, *kernid);
        kokkosp::create_profiler<kokkosp_region>(pname, *kernid);
        kokkosp::start_profiler<kokkosp_region>(*kernid);
    }

    void kokkosp_end_parallel_reduce(uint64_t kernid)
    {
        OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::logger_t{}.mark(-1, __FUNCTION__, kernid);
        kokkosp::stop_profiler<kokkosp_region>(kernid);
        kokkosp::destroy_profiler<kokkosp_region>(kernid);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_begin_parallel_scan(const char* name, uint32_t devid, uint64_t* kernid)
    {
        OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);
        auto pname =
            (devid > std::numeric_limits<uint16_t>::max())  // junk device number
                ? TIMEMORY_JOIN(" ", "[kokkos]", name)
                : TIMEMORY_JOIN(" ", TIMEMORY_JOIN("", "[kokkos][dev", devid, ']'), name);
        *kernid = kokkosp::get_unique_id();
        kokkosp::logger_t{}.mark(1, __FUNCTION__, name, *kernid);
        kokkosp::create_profiler<kokkosp_region>(pname, *kernid);
        kokkosp::start_profiler<kokkosp_region>(*kernid);
    }

    void kokkosp_end_parallel_scan(uint64_t kernid)
    {
        OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::logger_t{}.mark(-1, __FUNCTION__, kernid);
        kokkosp::stop_profiler<kokkosp_region>(kernid);
        kokkosp::destroy_profiler<kokkosp_region>(kernid);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_begin_fence(const char* name, uint32_t devid, uint64_t* kernid)
    {
        OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);
        auto pname =
            (devid > std::numeric_limits<uint16_t>::max())  // junk device number
                ? TIMEMORY_JOIN(" ", "[kokkos]", name)
                : TIMEMORY_JOIN(" ", TIMEMORY_JOIN("", "[kokkos][dev", devid, ']'), name);
        *kernid = kokkosp::get_unique_id();
        kokkosp::logger_t{}.mark(1, __FUNCTION__, name, *kernid);
        kokkosp::create_profiler<kokkosp_region>(pname, *kernid);
        kokkosp::start_profiler<kokkosp_region>(*kernid);
    }

    void kokkosp_end_fence(uint64_t kernid)
    {
        OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::logger_t{}.mark(-1, __FUNCTION__, kernid);
        kokkosp::stop_profiler<kokkosp_region>(kernid);
        kokkosp::destroy_profiler<kokkosp_region>(kernid);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_push_profile_region(const char* name)
    {
        if(omnitrace::get_use_perfetto()) return;  // perfetto doesn't support regions
        OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::logger_t{}.mark(1, __FUNCTION__, name);
        kokkosp::get_profiler_stack<kokkosp_region>().push_back(
            kokkosp::profiler_t<kokkosp_region>(name));
        kokkosp::get_profiler_stack<kokkosp_region>().back().start();
    }

    void kokkosp_pop_profile_region()
    {
        if(omnitrace::get_use_perfetto()) return;  // perfetto doesn't support regions
        OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::logger_t{}.mark(-1, __FUNCTION__);
        if(kokkosp::get_profiler_stack<kokkosp_region>().empty()) return;
        kokkosp::get_profiler_stack<kokkosp_region>().back().stop();
        kokkosp::get_profiler_stack<kokkosp_region>().pop_back();
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_create_profile_section(const char* name, uint32_t* secid)
    {
        if(omnitrace::get_use_perfetto()) return;  // perfetto doesn't support regions
        OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);
        *secid     = kokkosp::get_unique_id();
        auto pname = TIMEMORY_JOIN(" ", "[kokkos]", name);
        kokkosp::create_profiler<kokkosp_region>(pname, *secid);
    }

    void kokkosp_destroy_profile_section(uint32_t secid)
    {
        if(omnitrace::get_use_perfetto()) return;  // perfetto doesn't support regions
        OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::destroy_profiler<kokkosp_region>(secid);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_start_profile_section(uint32_t secid)
    {
        if(omnitrace::get_use_perfetto()) return;  // perfetto doesn't support regions
        OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::logger_t{}.mark(1, __FUNCTION__, secid);
        kokkosp::start_profiler<kokkosp_region>(secid);
    }

    void kokkosp_stop_profile_section(uint32_t secid)
    {
        if(omnitrace::get_use_perfetto()) return;  // perfetto doesn't support regions
        OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::logger_t{}.mark(-1, __FUNCTION__, secid);
        kokkosp::start_profiler<kokkosp_region>(secid);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_allocate_data(const SpaceHandle space, const char* label,
                               const void* const ptr, const uint64_t size)
    {
        OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::logger_t{}.mark(0, __FUNCTION__, space.name, label,
                                 TIMEMORY_JOIN("", '[', ptr, ']'), size);
        kokkosp::profiler_alloc_t<>{ TIMEMORY_JOIN(" ", "[kokkos][allocate]", space.name,
                                                   label) }
            .store(std::plus<int64_t>{}, size);
    }

    void kokkosp_deallocate_data(const SpaceHandle space, const char* label,
                                 const void* const ptr, const uint64_t size)
    {
        OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::logger_t{}.mark(0, __FUNCTION__, space.name, label,
                                 TIMEMORY_JOIN("", '[', ptr, ']'), size);
        kokkosp::profiler_alloc_t<>{ TIMEMORY_JOIN(" ", "[kokkos][deallocate]",
                                                   space.name, label) }
            .store(std::plus<int64_t>{}, size);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_begin_deep_copy(SpaceHandle dst_handle, const char* dst_name,
                                 const void* dst_ptr, SpaceHandle src_handle,
                                 const char* src_name, const void* src_ptr, uint64_t size)
    {
        OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::logger_t{}.mark(1, __FUNCTION__, dst_handle.name, dst_name,
                                 TIMEMORY_JOIN("", '[', dst_ptr, ']'), src_handle.name,
                                 src_name, TIMEMORY_JOIN("", '[', src_ptr, ']'), size);

        auto name = TIMEMORY_JOIN(" ", "[kokkos][deep_copy]",
                                  TIMEMORY_JOIN('=', dst_handle.name, dst_name),
                                  TIMEMORY_JOIN('=', src_handle.name, src_name));

        auto& _data = kokkosp::get_profiler_stack<kokkosp_region>();
        _data.emplace_back(name);
        _data.back().audit(dst_handle, dst_name, dst_ptr, src_handle, src_name, src_ptr,
                           size);
        _data.back().start();
        _data.back().store(std::plus<int64_t>{}, size);
    }

    void kokkosp_end_deep_copy()
    {
        OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::logger_t{}.mark(-1, __FUNCTION__);
        auto& _data = kokkosp::get_profiler_stack<kokkosp_region>();
        if(_data.empty()) return;
        _data.back().store(std::minus<int64_t>{}, 0);
        _data.back().stop();
        _data.pop_back();
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_profile_event(const char* name)
    {
        OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::profiler_t<kokkosp_region>{}.mark(name);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_dual_view_sync(const char* label, const void* const, bool is_device)
    {
        OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);
        if(omnitrace::config::get_use_perfetto())
        {
            auto _name = tim::get_hash_identifier_fast(
                tim::add_hash_id(TIMEMORY_JOIN(" ", "[kokkos][dual_view_sync]", label)));
            TRACE_EVENT_INSTANT("user", ::perfetto::StaticString{ _name.data() },
                                "target", (is_device) ? "device" : "host");
        }
    }

    void kokkosp_dual_view_modify(const char* label, const void* const, bool is_device)
    {
        OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);
        if(omnitrace::config::get_use_perfetto())
        {
            auto _name = tim::get_hash_identifier_fast(tim::add_hash_id(
                TIMEMORY_JOIN(" ", "[kokkos][dual_view_modify]", label)));
            TRACE_EVENT_INSTANT("user", ::perfetto::StaticString{ _name.data() },
                                "target", (is_device) ? "device" : "host");
        }
    }

    //----------------------------------------------------------------------------------//
}

TIMEMORY_INITIALIZE_STORAGE(kokkosp::memory_tracker)
