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

#include "callchain.hpp"
#include "binary/analysis.hpp"
#include "core/common.hpp"
#include "core/components/fwd.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "core/perfetto.hpp"
#include "core/state.hpp"
#include "library/components/ensure_storage.hpp"
#include "library/perf.hpp"
#include "library/ptl.hpp"
#include "library/runtime.hpp"
#include "library/sampling.hpp"
#include "library/thread_info.hpp"

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
#include <timemory/unwind/entry.hpp>
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
bool
callchain::record::operator<(const record& rhs) const
{
    return timestamp < rhs.timestamp;
}

std::vector<callchain::ts_entry_vec_t>
callchain::get() const
{
    std::vector<ts_entry_vec_t> _v = {};
    if(size() == 0) return _v;

    _v.reserve(size());
    auto _data = m_data;
    std::sort(_data.begin(), _data.end());
    for(const auto& itr : _data)
    {
        auto _v2 = ts_entry_vec_t{ itr.timestamp, {} };
        for(auto iitr : itr.data)
        {
            auto _entry = binary::lookup_ipaddr_entry<true>(iitr);
            if(_entry) _v2.second.emplace_back(*_entry);
        }

        if(!_v2.second.empty())
        {
            // put the bottom of the call-stack on top
            std::reverse(_v2.second.begin(), _v2.second.end());
            _v.emplace_back(std::move(_v2));
        }
    }

    auto _known_excludes =
        std::set<std::string>{ "funlockfile", "killpg", "__restore_rt" };
    // remove some known functions which are by-products of interrupts
    for(auto& itr : _v)
    {
        while(!itr.second.empty() &&
              _known_excludes.find(itr.second.back().name) != _known_excludes.end())
            itr.second.pop_back();
    }

    std::sort(_v.begin(), _v.end(),
              [](const auto& _lhs, const auto& _rhs) { return _lhs.first < _rhs.first; });

    return _v;
}

std::string
callchain::label()
{
    return "callchain";
}

std::string
callchain::description()
{
    return "Records callchain data";
}

std::vector<callchain::ts_entry_vec_t>
callchain::filter_and_patch(const std::vector<ts_entry_vec_t>& _data)
{
    auto _ret = std::vector<ts_entry_vec_t>{};
    _ret.reserve(_data.size());
    for(const auto& itr : _data)
    {
        auto _v = backtrace::filter_and_patch(itr.second);
        if(!_v.empty()) _ret.emplace_back(ts_entry_vec_t{ itr.first, std::move(_v) });
    }

    return _ret;
}

void
callchain::start()
{}

void
callchain::stop()
{}

bool
callchain::empty() const
{
    return (size() == 0);
}

size_t
callchain::size() const
{
    return m_data.size();
}

void
callchain::sample(int signo)
{
    if(signo != get_sampling_overflow_signal()) return;

    // on RedHat, the unw_step within get_unw_stack involves a mutex lock
    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);

    static thread_local const auto& _tinfo      = thread_info::get();
    auto                            _tid        = _tinfo->index_data->sequent_value;
    auto&                           _perf_event = perf::get_instance(_tid);

    if(!_perf_event) return;

    _perf_event->stop();

    for(auto itr : *_perf_event)
    {
        if(itr.is_sample())
        {
            auto _ip        = itr.get_ip();
            auto _data      = record{};
            _data.timestamp = itr.get_time();
            _data.data.emplace_back(_ip);
            bool _skip_ip = true;
            for(auto ditr : itr.get_callchain())
            {
                // skip the first instance of current IP but allow after that since this
                // might be a recursive call
                if(ditr == _ip && _skip_ip)
                    _skip_ip = false;
                else
                    _data.data.emplace_back(ditr);
                if(_data.data.size() == _data.data.capacity()) break;
            }
            if(!_data.data.empty()) m_data.emplace_back(_data);
        }
    }

    _perf_event->start();
}
}  // namespace component
}  // namespace rocprofsys

TIMEMORY_INITIALIZE_STORAGE(rocprofsys::component::callchain)
