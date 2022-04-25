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

#include "library/components/omnitrace.hpp"
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

//--------------------------------------------------------------------------------------//

extern "C"
{
    void kokkosp_print_help(char*) {}

    void kokkosp_parse_args(int, char**) {}

    void kokkosp_declare_metadata(const char* key, const char* value)
    {
        tim::manager::add_metadata(key, value);
    }

    void kokkosp_init_library(const int loadSeq, const uint64_t interfaceVer,
                              const uint32_t devInfoCount, void* deviceInfo)
    {
        tim::consume_parameters(devInfoCount, deviceInfo);

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
        OMNITRACE_VERBOSE_F(0, "Finalizing connector... \n");

        kokkosp::cleanup();

        if(omnitrace::get_verbose() >= 0) fprintf(stderr, "Done\n");
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
        kokkosp::create_profiler<omnitrace::component::omnitrace>(pname, *kernid);
        kokkosp::start_profiler<omnitrace::component::omnitrace>(*kernid);
    }

    void kokkosp_end_parallel_for(uint64_t kernid)
    {
        kokkosp::logger_t{}.mark(-1, __FUNCTION__, kernid);
        kokkosp::stop_profiler<omnitrace::component::omnitrace>(kernid);
        kokkosp::destroy_profiler<omnitrace::component::omnitrace>(kernid);
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
        kokkosp::create_profiler<omnitrace::component::omnitrace>(pname, *kernid);
        kokkosp::start_profiler<omnitrace::component::omnitrace>(*kernid);
    }

    void kokkosp_end_parallel_reduce(uint64_t kernid)
    {
        kokkosp::logger_t{}.mark(-1, __FUNCTION__, kernid);
        kokkosp::stop_profiler<omnitrace::component::omnitrace>(kernid);
        kokkosp::destroy_profiler<omnitrace::component::omnitrace>(kernid);
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
        kokkosp::create_profiler<omnitrace::component::omnitrace>(pname, *kernid);
        kokkosp::start_profiler<omnitrace::component::omnitrace>(*kernid);
    }

    void kokkosp_end_parallel_scan(uint64_t kernid)
    {
        kokkosp::logger_t{}.mark(-1, __FUNCTION__, kernid);
        kokkosp::stop_profiler<omnitrace::component::omnitrace>(kernid);
        kokkosp::destroy_profiler<omnitrace::component::omnitrace>(kernid);
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
        kokkosp::create_profiler<omnitrace::component::omnitrace>(pname, *kernid);
        kokkosp::start_profiler<omnitrace::component::omnitrace>(*kernid);
    }

    void kokkosp_end_fence(uint64_t kernid)
    {
        kokkosp::logger_t{}.mark(-1, __FUNCTION__, kernid);
        kokkosp::stop_profiler<omnitrace::component::omnitrace>(kernid);
        kokkosp::destroy_profiler<omnitrace::component::omnitrace>(kernid);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_push_profile_region(const char* name)
    {
        kokkosp::logger_t{}.mark(1, __FUNCTION__, name);
        kokkosp::get_profiler_stack<omnitrace::component::omnitrace>().push_back(
            kokkosp::profiler_t<omnitrace::component::omnitrace>(name));
        kokkosp::get_profiler_stack<omnitrace::component::omnitrace>().back().start();
    }

    void kokkosp_pop_profile_region()
    {
        kokkosp::logger_t{}.mark(-1, __FUNCTION__);
        if(kokkosp::get_profiler_stack<omnitrace::component::omnitrace>().empty()) return;
        kokkosp::get_profiler_stack<omnitrace::component::omnitrace>().back().stop();
        kokkosp::get_profiler_stack<omnitrace::component::omnitrace>().pop_back();
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_create_profile_section(const char* name, uint32_t* secid)
    {
        *secid     = kokkosp::get_unique_id();
        auto pname = TIMEMORY_JOIN(" ", "[kokkos]", name);
        kokkosp::create_profiler<omnitrace::component::omnitrace>(pname, *secid);
    }

    void kokkosp_destroy_profile_section(uint32_t secid)
    {
        kokkosp::destroy_profiler<omnitrace::component::omnitrace>(secid);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_start_profile_section(uint32_t secid)
    {
        kokkosp::logger_t{}.mark(1, __FUNCTION__, secid);
        kokkosp::start_profiler<omnitrace::component::omnitrace>(secid);
    }

    void kokkosp_stop_profile_section(uint32_t secid)
    {
        kokkosp::logger_t{}.mark(-1, __FUNCTION__, secid);
        kokkosp::start_profiler<omnitrace::component::omnitrace>(secid);
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

        auto& _data = kokkosp::get_profiler_stack<omnitrace::component::omnitrace>();
        _data.emplace_back(name);
        _data.back().audit(dst_handle, dst_name, dst_ptr, src_handle, src_name, src_ptr,
                           size);
        _data.back().start();
        _data.back().store(std::plus<int64_t>{}, size);
    }

    void kokkosp_end_deep_copy()
    {
        kokkosp::logger_t{}.mark(-1, __FUNCTION__);
        auto& _data = kokkosp::get_profiler_stack<omnitrace::component::omnitrace>();
        if(_data.empty()) return;
        _data.back().store(std::minus<int64_t>{}, 0);
        _data.back().stop();
        _data.pop_back();
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_profile_event(const char* name)
    {
        kokkosp::profiler_t<omnitrace::component::omnitrace>{}.mark(name);
    }

    //----------------------------------------------------------------------------------//
}

TIMEMORY_INITIALIZE_STORAGE(kokkosp::memory_tracker)
