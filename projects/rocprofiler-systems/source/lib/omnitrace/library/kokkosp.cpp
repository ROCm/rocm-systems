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

#include "library/defines.hpp"

#define TIMEMORY_KOKKOSP_POSTFIX OMNITRACE_PUBLIC_API

#include "library/api.hpp"
#include "library/components/user_region.hpp"
#include "library/config.hpp"
#include "library/debug.hpp"

#include <timemory/api/kokkosp.hpp>

namespace kokkosp = tim::kokkosp;

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
    void kokkosp_print_help(char*) {}

    void kokkosp_parse_args(int argc, char** argv)
    {
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
        tim::manager::add_metadata(key, value);
    }

    void kokkosp_init_library(const int loadSeq, const uint64_t interfaceVer,
                              const uint32_t devInfoCount, void* deviceInfo)
    {
        tim::consume_parameters(devInfoCount, deviceInfo);

        if(_standalone_initialized || (!omnitrace::config::settings_are_configured() &&
                                       omnitrace::get_state() < omnitrace::State::Active))
        {
            OMNITRACE_BASIC_VERBOSE_F(0, "Initializing omnitrace...\n");
            auto _mode = tim::get_env<std::string>("OMNITRACE_MODE", "trace");
            auto _arg0 = (_initialize_arguments.empty()) ? std::string{ "unknown" }
                                                         : _initialize_arguments.at(0);

            _standalone_initialized = true;
            omnitrace_set_mpi_hidden(false, false);
            omnitrace_init_hidden(_mode.c_str(), false, _arg0.c_str());
            omnitrace_push_trace("kokkos_main");
        }

        OMNITRACE_VERBOSE_F(0,
                            "Initializing connector (sequence is %d, version: %llu)...",
                            loadSeq, (unsigned long long) interfaceVer);

        setup_kernel_logger();

        tim::trait::runtime_enabled<kokkosp::memory_tracker>::set(
            omnitrace::config::get_use_timemory());

        if(omnitrace::get_verbose() >= 0) fprintf(stderr, "Done\n");
    }

    void kokkosp_finalize_library()
    {
        if(_standalone_initialized)
        {
            omnitrace_pop_trace("kokkos_main");
            OMNITRACE_VERBOSE_F(0, "Finalizing connector (standalone)...\n");
            omnitrace_finalize_hidden();
        }
        else
        {
            OMNITRACE_VERBOSE_F(0, "Finalizing connector... ");
            kokkosp::cleanup();
            if(omnitrace::get_verbose() >= 0) fprintf(stderr, "Done\n");
        }
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_begin_parallel_for(const char* name, uint32_t devid, uint64_t* kernid)
    {
        auto pname =
            (devid > std::numeric_limits<uint16_t>::max())  // junk device number
                ? TIMEMORY_JOIN(" ", "[kokkos]", name)
                : TIMEMORY_JOIN(" ", TIMEMORY_JOIN("", "[kokkos][dev", devid, ']'), name);
        *kernid = kokkosp::get_unique_id();
        kokkosp::logger_t{}.mark(1, __FUNCTION__, name, *kernid);
        kokkosp::create_profiler<omnitrace::component::user_region>(pname, *kernid);
        kokkosp::start_profiler<omnitrace::component::user_region>(*kernid);
    }

    void kokkosp_end_parallel_for(uint64_t kernid)
    {
        kokkosp::logger_t{}.mark(-1, __FUNCTION__, kernid);
        kokkosp::stop_profiler<omnitrace::component::user_region>(kernid);
        kokkosp::destroy_profiler<omnitrace::component::user_region>(kernid);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_begin_parallel_reduce(const char* name, uint32_t devid, uint64_t* kernid)
    {
        auto pname =
            (devid > std::numeric_limits<uint16_t>::max())  // junk device number
                ? TIMEMORY_JOIN(" ", "[kokkos]", name)
                : TIMEMORY_JOIN(" ", TIMEMORY_JOIN("", "[kokkos][dev", devid, ']'), name);
        *kernid = kokkosp::get_unique_id();
        kokkosp::logger_t{}.mark(1, __FUNCTION__, name, *kernid);
        kokkosp::create_profiler<omnitrace::component::user_region>(pname, *kernid);
        kokkosp::start_profiler<omnitrace::component::user_region>(*kernid);
    }

    void kokkosp_end_parallel_reduce(uint64_t kernid)
    {
        kokkosp::logger_t{}.mark(-1, __FUNCTION__, kernid);
        kokkosp::stop_profiler<omnitrace::component::user_region>(kernid);
        kokkosp::destroy_profiler<omnitrace::component::user_region>(kernid);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_begin_parallel_scan(const char* name, uint32_t devid, uint64_t* kernid)
    {
        auto pname =
            (devid > std::numeric_limits<uint16_t>::max())  // junk device number
                ? TIMEMORY_JOIN(" ", "[kokkos]", name)
                : TIMEMORY_JOIN(" ", TIMEMORY_JOIN("", "[kokkos][dev", devid, ']'), name);
        *kernid = kokkosp::get_unique_id();
        kokkosp::logger_t{}.mark(1, __FUNCTION__, name, *kernid);
        kokkosp::create_profiler<omnitrace::component::user_region>(pname, *kernid);
        kokkosp::start_profiler<omnitrace::component::user_region>(*kernid);
    }

    void kokkosp_end_parallel_scan(uint64_t kernid)
    {
        kokkosp::logger_t{}.mark(-1, __FUNCTION__, kernid);
        kokkosp::stop_profiler<omnitrace::component::user_region>(kernid);
        kokkosp::destroy_profiler<omnitrace::component::user_region>(kernid);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_begin_fence(const char* name, uint32_t devid, uint64_t* kernid)
    {
        auto pname =
            (devid > std::numeric_limits<uint16_t>::max())  // junk device number
                ? TIMEMORY_JOIN(" ", "[kokkos]", name)
                : TIMEMORY_JOIN(" ", TIMEMORY_JOIN("", "[kokkos][dev", devid, ']'), name);
        *kernid = kokkosp::get_unique_id();
        kokkosp::logger_t{}.mark(1, __FUNCTION__, name, *kernid);
        kokkosp::create_profiler<omnitrace::component::user_region>(pname, *kernid);
        kokkosp::start_profiler<omnitrace::component::user_region>(*kernid);
    }

    void kokkosp_end_fence(uint64_t kernid)
    {
        kokkosp::logger_t{}.mark(-1, __FUNCTION__, kernid);
        kokkosp::stop_profiler<omnitrace::component::user_region>(kernid);
        kokkosp::destroy_profiler<omnitrace::component::user_region>(kernid);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_push_profile_region(const char* name)
    {
        kokkosp::logger_t{}.mark(1, __FUNCTION__, name);
        kokkosp::get_profiler_stack<omnitrace::component::user_region>().push_back(
            kokkosp::profiler_t<omnitrace::component::user_region>(name));
        kokkosp::get_profiler_stack<omnitrace::component::user_region>().back().start();
    }

    void kokkosp_pop_profile_region()
    {
        kokkosp::logger_t{}.mark(-1, __FUNCTION__);
        if(kokkosp::get_profiler_stack<omnitrace::component::user_region>().empty())
            return;
        kokkosp::get_profiler_stack<omnitrace::component::user_region>().back().stop();
        kokkosp::get_profiler_stack<omnitrace::component::user_region>().pop_back();
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_create_profile_section(const char* name, uint32_t* secid)
    {
        *secid     = kokkosp::get_unique_id();
        auto pname = TIMEMORY_JOIN(" ", "[kokkos]", name);
        kokkosp::create_profiler<omnitrace::component::user_region>(pname, *secid);
    }

    void kokkosp_destroy_profile_section(uint32_t secid)
    {
        kokkosp::destroy_profiler<omnitrace::component::user_region>(secid);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_start_profile_section(uint32_t secid)
    {
        kokkosp::logger_t{}.mark(1, __FUNCTION__, secid);
        kokkosp::start_profiler<omnitrace::component::user_region>(secid);
    }

    void kokkosp_stop_profile_section(uint32_t secid)
    {
        kokkosp::logger_t{}.mark(-1, __FUNCTION__, secid);
        kokkosp::start_profiler<omnitrace::component::user_region>(secid);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_allocate_data(const SpaceHandle space, const char* label,
                               const void* const ptr, const uint64_t size)
    {
        kokkosp::logger_t{}.mark(0, __FUNCTION__, space.name, label,
                                 TIMEMORY_JOIN("", '[', ptr, ']'), size);
        kokkosp::profiler_alloc_t<>{ TIMEMORY_JOIN(" ", "[kokkos][allocate]", space.name,
                                                   label) }
            .store(std::plus<int64_t>{}, size);
    }

    void kokkosp_deallocate_data(const SpaceHandle space, const char* label,
                                 const void* const ptr, const uint64_t size)
    {
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
        kokkosp::logger_t{}.mark(1, __FUNCTION__, dst_handle.name, dst_name,
                                 TIMEMORY_JOIN("", '[', dst_ptr, ']'), src_handle.name,
                                 src_name, TIMEMORY_JOIN("", '[', src_ptr, ']'), size);

        auto name = TIMEMORY_JOIN(" ", "[kokkos][deep_copy]",
                                  TIMEMORY_JOIN('=', dst_handle.name, dst_name),
                                  TIMEMORY_JOIN('=', src_handle.name, src_name));

        auto& _data = kokkosp::get_profiler_stack<omnitrace::component::user_region>();
        _data.emplace_back(name);
        _data.back().audit(dst_handle, dst_name, dst_ptr, src_handle, src_name, src_ptr,
                           size);
        _data.back().start();
        _data.back().store(std::plus<int64_t>{}, size);
    }

    void kokkosp_end_deep_copy()
    {
        kokkosp::logger_t{}.mark(-1, __FUNCTION__);
        auto& _data = kokkosp::get_profiler_stack<omnitrace::component::user_region>();
        if(_data.empty()) return;
        _data.back().store(std::minus<int64_t>{}, 0);
        _data.back().stop();
        _data.pop_back();
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_profile_event(const char* name)
    {
        kokkosp::profiler_t<omnitrace::component::user_region>{}.mark(name);
    }

    //----------------------------------------------------------------------------------//
}

TIMEMORY_INITIALIZE_STORAGE(kokkosp::memory_tracker)
