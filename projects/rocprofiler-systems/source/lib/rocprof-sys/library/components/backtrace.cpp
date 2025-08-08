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

#include "core/common.hpp"
#include "core/components/fwd.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "core/perfetto.hpp"
#include "core/state.hpp"
#include "library/components/ensure_storage.hpp"
#include "library/ptl.hpp"
#include "library/runtime.hpp"
#include "library/sampling.hpp"

#include <timemory/backends/papi.hpp>
#include <timemory/backends/threading.hpp>
#include <timemory/components/data_tracker/components.hpp>
#include <timemory/components/macros.hpp>
#include <timemory/components/papi/extern.hpp>
#include <timemory/components/papi/papi_array.hpp>
#include <timemory/components/papi/papi_vector.hpp>
#include <timemory/components/rusage/components.hpp>
#include <timemory/components/rusage/types.hpp>
#include <timemory/components/timing/backends.hpp>
#include <timemory/components/trip_count/extern.hpp>
#include <timemory/macros.hpp>
#include <timemory/math.hpp>
#include <timemory/mpl.hpp>
#include <timemory/mpl/quirks.hpp>
#include <timemory/mpl/type_traits.hpp>
#include <timemory/operations.hpp>
#include <timemory/storage.hpp>
#include <timemory/units.hpp>
#include <timemory/utility/backtrace.hpp>
#include <timemory/utility/demangle.hpp>
#include <timemory/utility/types.hpp>
#include <timemory/variadic.hpp>

#include <array>
#include <cstring>
#include <ctime>
#include <initializer_list>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <pthread.h>
#include <signal.h>

namespace rocprofsys
{
namespace component
{
std::vector<backtrace::entry_type>
backtrace::get() const
{
    std::vector<entry_type> _v = {};
    if(size() == 0) return _v;

    {
        static auto _cache = cache_type{ get_sampling_include_inlines() };
        auto_lock_t _lk{ type_mutex<backtrace>() };
        _v = m_data.get(&_cache, false);
    }

    // put the bottom of the call-stack on top
    std::reverse(_v.begin(), _v.end());
    //
    auto _known_excludes =
        std::set<std::string>{ "funlockfile", "killpg", "__restore_rt" };
    // remove some known functions which are by-products of interrupts
    while(!_v.empty() && _known_excludes.find(_v.back().name) != _known_excludes.end())
        _v.pop_back();

    return _v;
}

std::string
backtrace::label()
{
    return "backtrace";
}

std::string
backtrace::description()
{
    return "Records backtrace data";
}

std::vector<backtrace::entry_type>
backtrace::filter_and_patch(const std::vector<entry_type>& _data)
{
    // check whether the call-stack entry should be used. -1 means break, 0 means continue
    auto _use_label = [](std::string_view _lbl) -> short {
        // debugging feature
        bool       _keep_internal = get_sampling_keep_internal();
        const auto _npos          = std::string::npos;
        if(_keep_internal) return 1;
        if(_lbl.find("rocprofsys_main") != _npos) return 0;
        if(_lbl.find("rocprofsys::") != _npos) return 0;
        if(_lbl.find("tim::openmp::") != _npos) return -1;
        if(_lbl.find("tim::") != _npos) return 0;
        if(_lbl.find("DYNINST_") != _npos) return 0;
        if(_lbl.find("rocprofsys_") != _npos) return -1;
        if(_lbl.find("rocprofiler_") != _npos) return -1;
        if(_lbl.find("perfetto::") != _npos) return -1;
        if(_lbl.find("protozero::") == 0) return -1;
        if(_lbl.find("gotcha_") != _npos) return -1;
        return 1;
    };

    static bool _keep_suffix = tim::get_env<bool>(
        "ROCPROFSYS_SAMPLING_KEEP_DYNINST_SUFFIX", get_debug_sampling());

    // in the dyninst binary rewrite runtime, instrumented functions are appended with
    // "_dyninst", i.e. "main" will show up as "main_dyninst" in the backtrace.
    auto _patch_label = [](std::string_view _lbl) -> std::string {
        // debugging feature
        if(_keep_suffix) return std::string{ _lbl };
        const std::string _dyninst{ "_dyninst" };
        auto              _pos = _lbl.find(_dyninst);
        if(_pos == std::string::npos) return std::string{ _lbl };
        return std::string{ _lbl }.replace(_pos, _dyninst.length(), "");
    };

    auto _ret = std::vector<entry_type>{};
    _ret.reserve(_data.size());
    for(const auto& itr : _data)
    {
        auto _name = tim::demangle(_patch_label(itr.name));
        auto _use  = _use_label(_name);
        if(_use == -1) break;
        if(_use == 0) continue;
        auto _v = itr;
        _v.name = _name;
        _ret.emplace_back(_v);
    }

    return _ret;
}

void
backtrace::start()
{}

void
backtrace::stop()
{}

bool
backtrace::empty() const
{
    return (size() == 0);
}

size_t
backtrace::size() const
{
    return m_data.size();
}

void
backtrace::sample(int signo)
{
    if(signo == get_sampling_overflow_signal()) return;

    // on RedHat, the unw_step within get_unw_stack involves a mutex lock
    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);

    using namespace tim::backtrace;
    constexpr bool   with_signal_frame = false;
    constexpr size_t ignore_depth      = 3;
    // ignore depth based on:
    // 1. this frame
    // 2. tim::sampling::sampler<...>::sample(...) [always inline]
    // 3. tim::sampling::sampler<...>::execute(...)
    // 4a. funlockfile       [common but not explicitly in call-stack]
    // 4b. __resume_rt       [common but not explicitly in call-stack]
    // 4c. killpg            [common but not explicitly in call-stack]
    m_data = get_unw_stack<stack_depth, ignore_depth, with_signal_frame>();
}
}  // namespace component
}  // namespace rocprofsys

TIMEMORY_INITIALIZE_STORAGE(rocprofsys::component::backtrace)
