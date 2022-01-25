// Copyright (c) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// with the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
//
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimers in the
// documentation and/or other materials provided with the distribution.
//
// * Neither the names of Advanced Micro Devices, Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this Software without specific prior written permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.

#include "library/sampling.hpp"
#include "library/config.hpp"
#include "library/debug.hpp"
#include "library/ptl.hpp"

#include <timemory/backends/papi.hpp>
#include <timemory/backends/threading.hpp>
#include <timemory/components/data_tracker/components.hpp>
#include <timemory/components/macros.hpp>
#include <timemory/components/papi/extern.hpp>
#include <timemory/components/papi/papi_array.hpp>
#include <timemory/components/papi/papi_vector.hpp>
#include <timemory/components/timing/backends.hpp>
#include <timemory/components/trip_count/extern.hpp>
#include <timemory/macros.hpp>
#include <timemory/math.hpp>
#include <timemory/mpl.hpp>
#include <timemory/mpl/quirks.hpp>
#include <timemory/mpl/type_traits.hpp>
#include <timemory/operations.hpp>
#include <timemory/sampling/allocator.hpp>
#include <timemory/sampling/sampler.hpp>
#include <timemory/storage.hpp>
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
#include <type_traits>

#include <pthread.h>
#include <signal.h>

namespace omnitrace
{
namespace sampling
{
using bundle_t  = tim::lightweight_tuple<backtrace>;
using sampler_t = tim::sampling::sampler<bundle_t, tim::sampling::dynamic>;
}  // namespace sampling
}  // namespace omnitrace

TIMEMORY_DEFINE_CONCRETE_TRAIT(check_signals, omnitrace::sampling::sampler_t,
                               std::false_type)

TIMEMORY_DEFINE_CONCRETE_TRAIT(buffer_size, omnitrace::sampling::sampler_t,
                               TIMEMORY_ESC(std::integral_constant<size_t, 256>))

namespace omnitrace
{
namespace sampling
{
using signal_type_instances     = omnitrace_thread_data<std::set<int>, api::sampling>;
using backtrace_init_instances  = omnitrace_thread_data<backtrace, api::sampling>;
using sampler_running_instances = omnitrace_thread_data<bool, api::sampling>;
using papi_vector_instances     = omnitrace_thread_data<comp::papi_vector, api::sampling>;

namespace
{
std::unique_ptr<comp::papi_vector>&
get_papi_vector(int64_t _tid)
{
    static auto& _v = papi_vector_instances::instances();
    if(_tid == threading::get_id()) papi_vector_instances::construct();
    return _v.at(_tid);
}

std::unique_ptr<backtrace>&
get_backtrace_init(int64_t _tid)
{
    static auto& _v = backtrace_init_instances::instances();
    return _v.at(_tid);
}

std::unique_ptr<bool>&
get_sampler_running(int64_t _tid)
{
    static auto& _v = sampler_running_instances::instances();
    return _v.at(_tid);
}

std::unique_ptr<std::set<int>>&
get_signal_types(int64_t _tid)
{
    static auto& _v = signal_type_instances::instances();
    // on the main thread, use both SIGALRM and SIGPROF.
    // on secondary threads, only use SIGPROF.
    signal_type_instances::construct((_tid == 0) ? std::set<int>{ SIGALRM, SIGPROF }
                                                 : std::set<int>{ SIGPROF });
    return _v.at(_tid);
}

template <typename... Args>
void
thread_sigmask(Args... _args)
{
    auto _err = pthread_sigmask(_args...);
    if(_err != 0)
    {
        errno = _err;
        perror("pthread_sigmask");
        exit(EXIT_FAILURE);
    }
}

template <typename Tp>
sigset_t
get_signal_set(Tp&& _v)
{
    sigset_t _sigset;
    sigemptyset(&_sigset);
    for(auto itr : _v)
        sigaddset(&_sigset, itr);
    return _sigset;
}

template <typename Tp>
std::string
get_signal_names(Tp&& _v)
{
    std::string _sig_names{};
    for(auto&& itr : _v)
        _sig_names += std::get<0>(tim::signal_settings::get_info(
                          static_cast<tim::sys_signal>(itr))) +
                      " ";
    return _sig_names.substr(0, _sig_names.length() - 1);
}
}  // namespace

std::set<int>
setup()
{
    return backtrace::configure(true);
}

std::set<int>
shutdown()
{
    return backtrace::configure(false);
}

void
block_signals(std::set<int> _signals)
{
    if(_signals.empty()) _signals = *get_signal_types(threading::get_id());
    if(_signals.empty())
    {
        OMNITRACE_PRINT("No signals to block...\n");
        return;
    }

    OMNITRACE_DEBUG("Blocking signals [%s] on thread #%lu...\n",
                    get_signal_names(_signals).c_str(), threading::get_id());

    sigset_t _v = get_signal_set(_signals);
    thread_sigmask(SIG_BLOCK, &_v, nullptr);
}

void
unblock_signals(std::set<int> _signals)
{
    if(_signals.empty()) _signals = *get_signal_types(threading::get_id());
    if(_signals.empty())
    {
        OMNITRACE_PRINT("No signals to unblock...\n");
        return;
    }

    OMNITRACE_DEBUG("Unblocking signals [%s] on thread #%lu...\n",
                    get_signal_names(_signals).c_str(), threading::get_id());

    sigset_t _v = get_signal_set(_signals);
    thread_sigmask(SIG_UNBLOCK, &_v, nullptr);
}

std::unique_ptr<sampler_t>&
get_sampler(int64_t _tid)
{
    static auto& _v = sampler_instances::instances();
    return _v.at(_tid);
}
}  // namespace sampling
}  // namespace omnitrace
