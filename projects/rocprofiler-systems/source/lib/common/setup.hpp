// MIT License
//
// Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
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

#pragma once

#include "common/defines.h"
#include "common/delimit.hpp"
#include "common/environment.hpp"
#include "common/join.hpp"
#include "common/path.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <ios>
#include <link.h>
#include <linux/limits.h>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>

#if !defined(ROCPROFSYS_SETUP_LOG_NAME)
#    if defined(ROCPROFSYS_COMMON_LIBRARY_NAME)
#        define ROCPROFSYS_SETUP_LOG_NAME "[" ROCPROFSYS_COMMON_LIBRARY_NAME "]"
#    else
#        define ROCPROFSYS_SETUP_LOG_NAME
#    endif
#endif

#if !defined(ROCPROFSYS_SETUP_LOG_START)
#    if defined(ROCPROFSYS_COMMON_LIBRARY_LOG_START)
#        define ROCPROFSYS_SETUP_LOG_START ROCPROFSYS_COMMON_LIBRARY_LOG_START
#    elif defined(TIMEMORY_LOG_COLORS_AVAILABLE)
#        define ROCPROFSYS_SETUP_LOG_START                                               \
            fprintf(stderr, "%s", ::tim::log::color::info());
#    else
#        define ROCPROFSYS_SETUP_LOG_START
#    endif
#endif

#if !defined(ROCPROFSYS_SETUP_LOG_END)
#    if defined(ROCPROFSYS_COMMON_LIBRARY_LOG_END)
#        define ROCPROFSYS_SETUP_LOG_END ROCPROFSYS_COMMON_LIBRARY_LOG_END
#    elif defined(TIMEMORY_LOG_COLORS_AVAILABLE)
#        define ROCPROFSYS_SETUP_LOG_END fprintf(stderr, "%s", ::tim::log::color::end());
#    else
#        define ROCPROFSYS_SETUP_LOG_END
#    endif
#endif

#define ROCPROFSYS_SETUP_LOG(CONDITION, ...)                                             \
    if(CONDITION)                                                                        \
    {                                                                                    \
        fflush(stderr);                                                                  \
        ROCPROFSYS_SETUP_LOG_START                                                       \
        fprintf(stderr, "[rocprof-sys]" ROCPROFSYS_SETUP_LOG_NAME "[%i] ", getpid());    \
        fprintf(stderr, __VA_ARGS__);                                                    \
        ROCPROFSYS_SETUP_LOG_END                                                         \
        fflush(stderr);                                                                  \
    }

namespace rocprofsys
{
inline namespace common
{
inline std::vector<env_config>
get_environ(int _verbose, std::string _search_paths = {},
            std::string _omnilib    = "librocprof-sys.so",
            std::string _omnilib_dl = "librocprof-sys-dl.so")
{
    auto _data            = std::vector<env_config>{};
    auto _omnilib_path    = path::get_origin(_omnilib);
    auto _omnilib_dl_path = path::get_origin(_omnilib_dl);

    if(!_omnilib_path.empty())
    {
        _omnilib      = join('/', _omnilib_path, ::basename(_omnilib.c_str()));
        _search_paths = join(':', _omnilib_path, _search_paths);
    }

    if(!_omnilib_dl_path.empty())
    {
        _omnilib_dl   = join('/', _omnilib_dl_path, ::basename(_omnilib_dl.c_str()));
        _search_paths = join(':', _omnilib_dl_path, _search_paths);
    }

    _omnilib    = common::path::find_path(_omnilib, _verbose, _search_paths);
    _omnilib_dl = common::path::find_path(_omnilib_dl, _verbose, _search_paths);

#if defined(ROCPROFSYS_USE_ROCTRACER) && ROCPROFSYS_USE_ROCTRACER > 0
    _data.emplace_back(env_config{ "HSA_TOOLS_LIB", _omnilib.c_str(), 0 });
#endif

#if defined(ROCPROFSYS_USE_ROCPROFILER) && ROCPROFSYS_USE_ROCPROFILER > 0
#    if ROCPROFSYS_HIP_VERSION >= 50200
#        define ROCPROFILER_METRICS_DIR "lib/rocprofiler"
#    else
#        define ROCPROFILER_METRICS_DIR "rocprofiler/lib"
#    endif
#    if ROCPROFSYS_HIP_VERSION <= 50500
#        define ROCPROFILER_LIBNAME "librocprofiler64.so"
#    else
#        define ROCPROFILER_LIBNAME "librocprofiler64.so.1"
#    endif

    _data.emplace_back(env_config{ "HSA_TOOLS_LIB", _omnilib.c_str(), 0 });
    _data.emplace_back(env_config{ "ROCP_TOOL_LIB", _omnilib.c_str(), 0 });
    _data.emplace_back(env_config{ "ROCPROFILER_LOG", "1", 0 });
    _data.emplace_back(env_config{ "ROCP_HSA_INTERCEPT", "1", 0 });
    _data.emplace_back(env_config{ "HSA_TOOLS_REPORT_LOAD_FAILURE", "1", 0 });

    auto _possible_rocp_metrics = std::vector<std::string>{};
    auto _possible_rocprof_libs = std::vector<std::string>{};
    for(const auto* itr : { "ROCPROFSYS_ROCM_PATH", "ROCM_PATH" })
    {
        if(getenv(itr))
        {
            _possible_rocp_metrics.emplace_back(
                common::join('/', getenv(itr), "lib/rocprofiler"));
            _possible_rocprof_libs.emplace_back(
                common::join('/', getenv(itr), "lib/rocprofiler", ROCPROFILER_LIBNAME));
            _possible_rocp_metrics.emplace_back(
                common::join('/', getenv(itr), "rocprofiler/lib"));
            _possible_rocprof_libs.emplace_back(
                common::join('/', getenv(itr), "rocprofiler/lib", ROCPROFILER_LIBNAME));
        }
    }

    // default path
    _possible_rocp_metrics.emplace_back(
        common::join('/', ROCPROFSYS_DEFAULT_ROCM_PATH, "lib/rocprofiler"));
    _possible_rocp_metrics.emplace_back(
        common::join('/', ROCPROFSYS_DEFAULT_ROCM_PATH, "rocprofiler/lib"));

    auto _realpath_and_unique = [](const auto& _inp_v) {
        auto _out_v = decltype(_inp_v){};
        for(auto& itr : _inp_v)
        {
            if(path::exists(itr)) _out_v.emplace_back(path::realpath(itr));
        }

        _out_v.erase(std::unique(_out_v.begin(), _out_v.end()), _out_v.end());
        return _out_v;
    };

    _possible_rocprof_libs = _realpath_and_unique(_possible_rocprof_libs);

    for(const auto& itr : _possible_rocprof_libs)
    {
        if(path::exists(itr))
        {
            _data.emplace_back(
                env_config{ "ROCPROFSYS_ROCPROFILER_LIBRARY", itr.c_str(), 0 });
            _possible_rocp_metrics.emplace(
                _possible_rocp_metrics.begin(),
                common::join('/', path::dirname(itr), "../../lib/rocprofiler"));
            _possible_rocp_metrics.emplace(_possible_rocp_metrics.begin(),
                                           common::join('/', path::dirname(itr)));
        }
    }

    _possible_rocp_metrics = _realpath_and_unique(_possible_rocp_metrics);

    auto _env_rocp_metrics = get_env("ROCP_METRICS", "");
    if(!_env_rocp_metrics.empty())
    {
        if(!path::exists(_env_rocp_metrics))
            throw std::runtime_error(join("", "Error! ROCP_METRICS file \"",
                                          _env_rocp_metrics, "\" does not exist"));
        _possible_rocp_metrics.clear();
        _possible_rocp_metrics.emplace_back(
            common::join('/', path::dirname(_env_rocp_metrics)));
    }

    auto _found_rocp_metrics = (!_env_rocp_metrics.empty())
                                   ? get_env("ROCPROFSYS_ROCP_METRICS_FORCE_VALID", false)
                                   : false;

    if(!_found_rocp_metrics)
    {
        for(const auto& itr : _possible_rocp_metrics)
        {
            auto _metrics_path = join('/', itr, "metrics.xml");
            if(path::exists(itr) && path::exists(_metrics_path) &&
               path::exists(join('/', itr, "gfx_metrics.xml")))
            {
                _found_rocp_metrics = true;
                _data.emplace_back(
                    env_config{ "ROCP_METRICS", _metrics_path.c_str(), 0 });
                break;
            }
        }
    }

    // handle error
    if(!_found_rocp_metrics)
    {
        auto _msg = std::stringstream{};
        _msg << std::boolalpha;
        if(!_env_rocp_metrics.empty())
        {
            auto _env_rocp_metrics_dir = path::dirname(_env_rocp_metrics);
            auto _rocp_metrics_xml     = join('/', _env_rocp_metrics_dir, "metrics.xml");
            auto _rocp_gfx_metrics_xml =
                join('/', _env_rocp_metrics_dir, "gfx_metrics.xml");
            _msg << "Error! ROCP_METRICS=\"" << _env_rocp_metrics
                 << "\" in the environment but the directory (" << _env_rocp_metrics_dir
                 << ") does not contain "
                    "metrics.xml (found: "
                 << path::exists(_rocp_metrics_xml) << ") and/or gfx_metrics.xml (found: "
                 << path::exists(_rocp_gfx_metrics_xml)
                 << "). To ignore this error, set "
                    "ROCPROFSYS_ROCP_METRICS_FORCE_VALID=true in the environment";
        }
        else
        {
            _msg
                << "Error! ROCP_METRICS not set in environment and rocprof-sys could not "
                   "find a suitable path. Please set ROCP_METRICS=/path/to/metrics.xml "
                   "in the environment. This file is typically located in the same "
                   "folder as the librocprofiler64.so library.\nAdditional note: "
                   "metrics.xml typically contains:\n\t#include "
                   "\"gfx_metrics.xml\"\nMake sure the provided path also contains this "
                   "file.\nExample:\n\texport ROCP_METRICS="
                << ROCPROFSYS_DEFAULT_ROCM_PATH << "/" << ROCPROFILER_METRICS_DIR
                << "/metrics.xml\n";
        }
        throw std::runtime_error(_msg.str());
    }
#endif

#if defined(ROCPROFSYS_USE_OMPT) && ROCPROFSYS_USE_OMPT > 0
    if(get_env("ROCPROFSYS_USE_OMPT", true))
    {
        std::string _omni_omp_libs = _omnilib_dl;
        const char* _omp_libs      = getenv("OMP_TOOL_LIBRARIES");
        int         _override      = 0;
        if(_omp_libs != nullptr &&
           std::string_view{ _omp_libs }.find(_omnilib_dl) == std::string::npos)
        {
            _override      = 1;
            _omni_omp_libs = common::join(':', _omp_libs, _omnilib_dl);
        }
        ROCPROFSYS_SETUP_LOG(_verbose >= 2, "setting OMP_TOOL_LIBRARIES to '%s'\n",
                             _omni_omp_libs.c_str());
        _data.emplace_back(
            env_config{ "OMP_TOOL_LIBRARIES", _omni_omp_libs.c_str(), _override });
    }
#endif

    return _data;
}

inline void
setup_environ(int _verbose, const std::string& _search_paths = {},
              std::string _omnilib    = "librocprof-sys.so",
              std::string _omnilib_dl = "librocprof-sys-dl.so")
{
    auto _data =
        get_environ(_verbose, _search_paths, std::move(_omnilib), std::move(_omnilib_dl));
    for(const auto& itr : _data)
        itr(_verbose >= 3);
}
}  // namespace common
}  // namespace rocprofsys
