// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#define TIMEMORY_KOKKOSP_POSTFIX ROCPROFSYS_PUBLIC_API

#include "api.hpp"
#include "core/components/fwd.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "core/defines.hpp"
#include "core/perfetto.hpp"
#include "library/components/category_region.hpp"
#include "library/runtime.hpp"

#include <timemory/api/kokkosp.hpp>
#include <timemory/backends/process.hpp>
#include <timemory/hash/types.hpp>
#include <timemory/mpl/concepts.hpp>
#include <timemory/mpl/type_traits.hpp>
#include <timemory/utility/procfs/maps.hpp>
#include <timemory/utility/types.hpp>

#include <cstdlib>
#include <sstream>
#include <string>

namespace kokkosp  = ::tim::kokkosp;
namespace category = ::tim::category;
namespace comp     = ::rocprofsys::component;

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
       rocprofsys::config::get_use_kokkosp_kernel_logger())
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
bool                     _kp_deep_copy           = false;
size_t                   _name_len_limit         = 0;
std::string              _kp_prefix              = {};
std::vector<std::string> _initialize_arguments   = {};

template <typename Tp>
void
set_invalid_id(Tp* _v)
{
    constexpr bool is32 = std::is_same<Tp, uint32_t>::value;
    constexpr bool is64 = std::is_same<Tp, uint64_t>::value;
    static_assert(is32 || is64, "only support uint32_t or uint64_t");

    *_v = std::numeric_limits<Tp>::max();
}

template <typename Tp>
bool
is_invalid_id(Tp _v)
{
    constexpr bool is32 = std::is_same<Tp, uint32_t>::value;
    constexpr bool is64 = std::is_same<Tp, uint64_t>::value;
    static_assert(is32 || is64, "only support uint32_t or uint64_t");

    return (_v == std::numeric_limits<Tp>::max());
}

template <typename Tp>
auto
strlength(Tp&& _v)
{
    using type = ::tim::concepts::unqualified_type_t<Tp>;
    if constexpr(std::is_same<type, std::string_view>::value ||
                 std::is_same<type, std::string>::value)
        return _v.length();
    else
        return strnlen(_v, std::max<size_t>(_name_len_limit, 1));
}

template <typename Arg, typename... Args>
bool
violates_name_rules(Arg&& _arg, Args&&... _args)
{
    // for causal profiling we only consider callbacks which are explicitly named
    if(rocprofsys::config::get_use_causal() &&
       (std::string_view{ _arg }.find("Kokkos::") == 0 ||
        std::string_view{ _arg }.find("Space::") != std::string_view::npos))
        return true;

    size_t _len =
        (strlength(std::forward<Arg>(_arg)) + ... + strlength(std::forward<Args>(_args)));

    // ignore labels without names
    if(_len == 0)
        return true;
    else if(_name_len_limit == 0)
        return false;

    return (_len >= _name_len_limit);
}
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
                                       Kokkos_Tools_ToolSettings*) ROCPROFSYS_PUBLIC_API;
    void kokkosp_dual_view_sync(const char*, const void* const,
                                bool) ROCPROFSYS_PUBLIC_API;
    void kokkosp_dual_view_modify(const char*, const void* const,
                                  bool) ROCPROFSYS_PUBLIC_API;

    void kokkosp_print_help(char*) {}

    void kokkosp_parse_args(int argc, char** argv)
    {
        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        if(!rocprofsys::config::settings_are_configured() &&
           rocprofsys::get_state() < rocprofsys::State::Active)
        {
            _standalone_initialized = true;

            ROCPROFSYS_BASIC_VERBOSE_F(0, "Parsing arguments...\n");
            std::string _command_line = {};
            for(int i = 0; i < argc; ++i)
            {
                _initialize_arguments.emplace_back(argv[i]);
                _command_line.append(" ").append(argv[i]);
            }
            if(_command_line.length() > 1) _command_line = _command_line.substr(1);
            tim::set_env("ROCPROFSYS_COMMAND_LINE", _command_line, 0);
        }
    }

    void kokkosp_declare_metadata(const char* key, const char* value)
    {
        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
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
        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        tim::consume_parameters(devInfoCount, deviceInfo);

        ROCPROFSYS_BASIC_VERBOSE_F(
            0,
            "Initializing rocprof-sys kokkos connector (sequence %d, version: %llu)... ",
            loadSeq, (unsigned long long) interfaceVer);

        if(_standalone_initialized ||
           (!rocprofsys::config::settings_are_configured() &&
            rocprofsys::get_state() < rocprofsys::State::Active))
        {
            auto _kokkos_profile_lib = tim::get_env<std::string>("KOKKOS_TOOLS_LIBS");
            if(_kokkos_profile_lib.find("librocprof-sys.so") != std::string::npos)
            {
                auto _maps = tim::procfs::read_maps(tim::process::get_id());
                auto _libs = std::set<std::string>{};
                for(auto& itr : _maps)
                {
                    auto&& _path = itr.pathname;
                    if(!_path.empty() && _path.at(0) != '[' &&
                       rocprofsys::filepath::exists(_path))
                        _libs.emplace(_path);
                }
                for(const auto& itr : _libs)
                {
                    if(itr.find("librocprof-sys-dl.so") != std::string::npos)
                    {
                        std::stringstream _libs_str{};
                        for(const auto& litr : _libs)
                            _libs_str << "    " << litr << "\n";
                        ROCPROFSYS_ABORT(
                            "%s was invoked with librocprof-sys.so as the "
                            "KOKKOS_TOOLS_LIBS.\n"
                            "However, librocprof-sys-dl.so has already been loaded by "
                            "the process.\nTo avoid duplicate collections culminating "
                            "is an error, please set KOKKOS_TOOLS_LIBS=%s.\nLoaded "
                            "libraries:\n%s",
                            __FUNCTION__, itr.c_str(), _libs_str.str().c_str());
                    }
                }
            }

            ROCPROFSYS_BASIC_VERBOSE_F(0, "Initializing rocprof-sys (standalone)... ");
            auto _mode = tim::get_env<std::string>("ROCPROFSYS_MODE", "trace");
            auto _arg0 = (_initialize_arguments.empty()) ? std::string{ "unknown" }
                                                         : _initialize_arguments.at(0);

            _standalone_initialized = true;
            rocprofsys_set_mpi_hidden(false, false);
            rocprofsys_init_hidden(_mode.c_str(), false, _arg0.c_str());
            rocprofsys_push_trace_hidden("kokkos_main");
        }

        setup_kernel_logger();

        tim::trait::runtime_enabled<kokkosp::memory_tracker>::set(
            rocprofsys::config::get_use_timemory());

        if(rocprofsys::get_verbose() >= 0)
        {
            fprintf(stderr, "%sDone\n%s", tim::log::color::info(),
                    tim::log::color::end());
        }

        _name_len_limit = rocprofsys::config::get_setting_value<int64_t>(
                              "ROCPROFSYS_KOKKOSP_NAME_LENGTH_MAX")
                              .value_or(_name_len_limit);
        _kp_prefix = rocprofsys::config::get_setting_value<std::string>(
                         "ROCPROFSYS_KOKKOSP_PREFIX")
                         .value_or(_kp_prefix);

        _kp_deep_copy =
            rocprofsys::config::get_setting_value<bool>("ROCPROFSYS_KOKKOSP_DEEP_COPY")
                .value_or(_kp_deep_copy);
    }

    void kokkosp_finalize_library()
    {
        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        if(_standalone_initialized)
        {
            rocprofsys_pop_trace_hidden("kokkos_main");
            ROCPROFSYS_VERBOSE_F(
                0, "Finalizing kokkos rocprof-sys connector (standalone)...\n");
            rocprofsys_finalize_hidden();
        }
        else
        {
            ROCPROFSYS_VERBOSE_F(0, "Finalizing kokkos rocprof-sys connector... ");
            kokkosp::cleanup();
            if(rocprofsys::get_verbose() >= 0) fprintf(stderr, "Done\n");
        }
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_begin_parallel_for(const char* name, uint32_t devid, uint64_t* kernid)
    {
        if(violates_name_rules(name)) return set_invalid_id(kernid);

        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        auto pname = (devid > std::numeric_limits<uint16_t>::max())  // junk device number
                         ? JOIN(" ", _kp_prefix, name, "[for]")
                         : JOIN(" ", _kp_prefix, name, JOIN("", "[for][dev", devid, ']'));
        *kernid    = kokkosp::get_unique_id();
        kokkosp::logger_t{}.mark(1, __FUNCTION__, name, *kernid);
        kokkosp::create_profiler<kokkosp_region>(pname, *kernid);
        kokkosp::start_profiler<kokkosp_region>(*kernid);
    }

    void kokkosp_end_parallel_for(uint64_t kernid)
    {
        if(is_invalid_id(kernid)) return;

        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::logger_t{}.mark(-1, __FUNCTION__, kernid);
        kokkosp::stop_profiler<kokkosp_region>(kernid);
        kokkosp::destroy_profiler<kokkosp_region>(kernid);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_begin_parallel_reduce(const char* name, uint32_t devid, uint64_t* kernid)
    {
        if(violates_name_rules(name)) return set_invalid_id(kernid);

        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        auto pname =
            (devid > std::numeric_limits<uint16_t>::max())  // junk device number
                ? JOIN(" ", _kp_prefix, name, "[reduce]")
                : JOIN(" ", _kp_prefix, name, JOIN("", "[reduce][dev", devid, ']'));
        *kernid = kokkosp::get_unique_id();
        kokkosp::logger_t{}.mark(1, __FUNCTION__, name, *kernid);
        kokkosp::create_profiler<kokkosp_region>(pname, *kernid);
        kokkosp::start_profiler<kokkosp_region>(*kernid);
    }

    void kokkosp_end_parallel_reduce(uint64_t kernid)
    {
        if(is_invalid_id(kernid)) return;

        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::logger_t{}.mark(-1, __FUNCTION__, kernid);
        kokkosp::stop_profiler<kokkosp_region>(kernid);
        kokkosp::destroy_profiler<kokkosp_region>(kernid);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_begin_parallel_scan(const char* name, uint32_t devid, uint64_t* kernid)
    {
        if(violates_name_rules(name)) return set_invalid_id(kernid);

        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        auto pname =
            (devid > std::numeric_limits<uint16_t>::max())  // junk device number
                ? JOIN(" ", _kp_prefix, name, "[scan]")
                : JOIN(" ", _kp_prefix, name, JOIN("", "[scan][dev", devid, ']'));
        *kernid = kokkosp::get_unique_id();
        kokkosp::logger_t{}.mark(1, __FUNCTION__, name, *kernid);
        kokkosp::create_profiler<kokkosp_region>(pname, *kernid);
        kokkosp::start_profiler<kokkosp_region>(*kernid);
    }

    void kokkosp_end_parallel_scan(uint64_t kernid)
    {
        if(is_invalid_id(kernid)) return;

        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::logger_t{}.mark(-1, __FUNCTION__, kernid);
        kokkosp::stop_profiler<kokkosp_region>(kernid);
        kokkosp::destroy_profiler<kokkosp_region>(kernid);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_begin_fence(const char* name, uint32_t devid, uint64_t* kernid)
    {
        if(violates_name_rules(name)) return set_invalid_id(kernid);

        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        auto pname =
            (devid > std::numeric_limits<uint16_t>::max())  // junk device number
                ? JOIN(" ", _kp_prefix, name, "[fence]")
                : JOIN(" ", _kp_prefix, name, JOIN("", "[fence][dev", devid, ']'));
        *kernid = kokkosp::get_unique_id();
        kokkosp::logger_t{}.mark(1, __FUNCTION__, name, *kernid);
        kokkosp::create_profiler<kokkosp_region>(pname, *kernid);
        kokkosp::start_profiler<kokkosp_region>(*kernid);
    }

    void kokkosp_end_fence(uint64_t kernid)
    {
        if(is_invalid_id(kernid)) return;

        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::logger_t{}.mark(-1, __FUNCTION__, kernid);
        kokkosp::stop_profiler<kokkosp_region>(kernid);
        kokkosp::destroy_profiler<kokkosp_region>(kernid);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_push_profile_region(const char* name)
    {
        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::logger_t{}.mark(1, __FUNCTION__, name);
        kokkosp::get_profiler_stack<kokkosp_region>()
            .emplace_back(kokkosp::profiler_t<kokkosp_region>(name))
            .start();
    }

    void kokkosp_pop_profile_region()
    {
        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::logger_t{}.mark(-1, __FUNCTION__);
        if(kokkosp::get_profiler_stack<kokkosp_region>().empty()) return;
        kokkosp::get_profiler_stack<kokkosp_region>().back().stop();
        kokkosp::get_profiler_stack<kokkosp_region>().pop_back();
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_create_profile_section(const char* name, uint32_t* secid)
    {
        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        *secid     = kokkosp::get_unique_id();
        auto pname = std::string{ name };
        kokkosp::create_profiler<kokkosp_region>(name, *secid);
    }

    void kokkosp_destroy_profile_section(uint32_t secid)
    {
        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::destroy_profiler<kokkosp_region>(secid);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_start_profile_section(uint32_t secid)
    {
        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::logger_t{}.mark(1, __FUNCTION__, secid);
        kokkosp::start_profiler<kokkosp_region>(secid);
    }

    void kokkosp_stop_profile_section(uint32_t secid)
    {
        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::logger_t{}.mark(-1, __FUNCTION__, secid);
        kokkosp::stop_profiler<kokkosp_region>(secid);
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_allocate_data(const SpaceHandle space, const char* label,
                               const void* const ptr, const uint64_t size)
    {
        if(violates_name_rules(label)) return;
        if(rocprofsys::config::get_use_causal()) return;

        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::logger_t{}.mark(0, __FUNCTION__, space.name, label,
                                 JOIN("", '[', ptr, ']'), size);
        auto pname =
            JOIN(" ", _kp_prefix, label, JOIN("", '[', space.name, "][allocate]"));
        kokkosp::profiler_alloc_t<>{ pname }.store(std::plus<int64_t>{}, size);
        kokkosp::profiler_t<kokkosp_region>{ pname }.mark();
    }

    void kokkosp_deallocate_data(const SpaceHandle space, const char* label,
                                 const void* const ptr, const uint64_t size)
    {
        if(violates_name_rules(label)) return;
        if(rocprofsys::config::get_use_causal()) return;

        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::logger_t{}.mark(0, __FUNCTION__, space.name, label,
                                 JOIN("", '[', ptr, ']'), size);
        auto pname =
            JOIN(" ", _kp_prefix, label, JOIN("", '[', space.name, "][deallocate]"));
        kokkosp::profiler_alloc_t<>{ pname }.store(std::plus<int64_t>{}, size);
        kokkosp::profiler_t<kokkosp_region>{ pname }.mark();
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_begin_deep_copy(SpaceHandle dst_handle, const char* dst_name,
                                 const void* dst_ptr, SpaceHandle src_handle,
                                 const char* src_name, const void* src_ptr, uint64_t size)
    {
        if(!_kp_deep_copy || rocprofsys::config::get_use_causal()) return;
        if(violates_name_rules(dst_name, src_name)) return;

        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::logger_t{}.mark(1, __FUNCTION__, dst_handle.name, dst_name,
                                 JOIN("", '[', dst_ptr, ']'), src_handle.name, src_name,
                                 JOIN("", '[', src_ptr, ']'), size);

        auto name = JOIN(" ", _kp_prefix, JOIN('=', dst_handle.name, dst_name), "<-",
                         JOIN('=', src_handle.name, src_name), "[deep_copy]");

        auto& _data = kokkosp::get_profiler_stack<kokkosp_region>();
        _data.emplace_back(name);
        _data.back().audit(dst_handle, dst_name, dst_ptr, src_handle, src_name, src_ptr,
                           size);
        _data.back().start();
        _data.back().store(tim::mpl::piecewise_select<kokkosp::memory_tracker>{},
                           std::plus<int64_t>{}, size);
    }

    void kokkosp_end_deep_copy()
    {
        if(!_kp_deep_copy || rocprofsys::config::get_use_causal()) return;

        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        kokkosp::logger_t{}.mark(-1, __FUNCTION__);
        auto& _data = kokkosp::get_profiler_stack<kokkosp_region>();
        if(_data.empty()) return;
        _data.back().store(tim::mpl::piecewise_select<kokkosp::memory_tracker>{},
                           std::minus<int64_t>{}, 0);
        _data.back().stop();
        _data.pop_back();
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_profile_event(const char* name)
    {
        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        auto _name = tim::get_hash_identifier_fast(tim::add_hash_id(name));
        kokkosp::profiler_t<kokkosp_region>{ _name }.mark();
    }

    //----------------------------------------------------------------------------------//

    void kokkosp_dual_view_sync(const char* label, const void* const, bool is_device)
    {
        if(violates_name_rules(label)) return;

        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        if(rocprofsys::config::get_use_perfetto())
        {
            auto _name = tim::get_hash_identifier_fast(
                tim::add_hash_id(JOIN(" ", _kp_prefix, label, "[dual_view_sync]")));
            TRACE_EVENT_INSTANT("user", ::perfetto::StaticString{ _name.data() },
                                "target", (is_device) ? "device" : "host");
        }
        else if(rocprofsys::config::get_use_causal())
        {
            auto _name = tim::get_hash_identifier_fast(tim::add_hash_id(JOIN(
                "", label, " [dual_view_sync][", (is_device) ? "device" : "host", "]")));
            kokkosp::profiler_t<kokkosp_region>{ _name }.mark();
        }
    }

    void kokkosp_dual_view_modify(const char* label, const void* const, bool is_device)
    {
        if(violates_name_rules(label)) return;

        ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);
        if(rocprofsys::config::get_use_perfetto())
        {
            auto _name = tim::get_hash_identifier_fast(
                tim::add_hash_id(JOIN(" ", _kp_prefix, label, "[dual_view_modify]")));
            TRACE_EVENT_INSTANT("user", ::perfetto::StaticString{ _name.data() },
                                "target", (is_device) ? "device" : "host");
        }
        else if(rocprofsys::config::get_use_causal())
        {
            auto _name = tim::get_hash_identifier_fast(
                tim::add_hash_id(JOIN(" ", _kp_prefix, label, "[dual_view_modify][",
                                      (is_device) ? "device" : "host", "]")));
            kokkosp::profiler_t<kokkosp_region>{ _name }.mark();
        }
    }

    //----------------------------------------------------------------------------------//
}

TIMEMORY_INITIALIZE_STORAGE(kokkosp::memory_tracker)
