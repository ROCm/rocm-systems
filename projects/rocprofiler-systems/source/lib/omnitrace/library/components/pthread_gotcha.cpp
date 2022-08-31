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

#include "library/components/pthread_gotcha.hpp"
#include "library/components/pthread_create_gotcha.hpp"
#include "library/components/pthread_mutex_gotcha.hpp"
#include "library/components/roctracer.hpp"
#include "library/config.hpp"
#include "library/debug.hpp"
#include "library/runtime.hpp"
#include "library/sampling.hpp"
#include "library/thread_data.hpp"
#include "library/utility.hpp"

#include <timemory/backends/threading.hpp>
#include <timemory/utility/macros.hpp>
#include <timemory/utility/types.hpp>

#include <pthread.h>

#include <array>
#include <vector>

namespace tim
{
namespace operation
{
template <>
struct stop<omnitrace::component::pthread_create_gotcha_t>
{
    using type = omnitrace::component::pthread_create_gotcha_t;

    TIMEMORY_DEFAULT_OBJECT(stop)

    template <typename... Args>
    explicit stop(type&, Args&&...)
    {}

    template <typename... Args>
    void operator()(type&, Args&&...)
    {}
};
}  // namespace operation
}  // namespace tim

namespace omnitrace
{
namespace
{
using bundle_t = tim::lightweight_tuple<component::pthread_create_gotcha_t,
                                        component::pthread_mutex_gotcha_t>;

auto&
get_sampling_on_child_threads_history(int64_t _idx = utility::get_thread_index())
{
    static auto _v = utility::get_filled_array<OMNITRACE_MAX_THREADS>(
        []() { return utility::get_reserved_vector<bool>(32); });
    return _v.at(_idx);
}

auto&
get_bundle()
{
    static auto _v = std::unique_ptr<bundle_t>{};
    if(!_v) _v = std::make_unique<bundle_t>("pthread_gotcha");
    return _v;
}

bool is_configured = false;
}  // namespace

//--------------------------------------------------------------------------------------//

void
pthread_gotcha::configure()
{
    if(!is_configured)
    {
        ::omnitrace::component::pthread_create_gotcha::configure();
        ::omnitrace::component::pthread_mutex_gotcha::configure();
        is_configured = true;
    }
}

void
pthread_gotcha::shutdown()
{
    if(is_configured)
    {
        ::omnitrace::component::pthread_mutex_gotcha::shutdown();
        // ::omnitrace::component::pthread_create_gotcha::shutdown();
        is_configured = false;
    }
}

bool
pthread_gotcha::sampling_enabled_on_child_threads()
{
    return sampling_on_child_threads();
}

bool
pthread_gotcha::push_enable_sampling_on_child_threads(bool _v)
{
    bool _last                  = sampling_on_child_threads();
    sampling_on_child_threads() = _v;
    auto& _hist                 = get_sampling_on_child_threads_history();
    _hist.emplace_back(_last);
    return _last;
}

bool
pthread_gotcha::pop_enable_sampling_on_child_threads()
{
    auto& _hist = get_sampling_on_child_threads_history();
    if(!_hist.empty())
    {
        bool _restored = _hist.back();
        _hist.pop_back();
        sampling_on_child_threads() = _restored;
    }
    return sampling_on_child_threads();
}

void
pthread_gotcha::set_sampling_on_all_future_threads(bool _v)
{
    for(size_t i = 0; i < max_supported_threads; ++i)
        get_sampling_on_child_threads_history(i).emplace_back(_v);
}

bool&
pthread_gotcha::sampling_on_child_threads()
{
    static thread_local bool _v = get_sampling_on_child_threads_history().empty()
                                      ? false
                                      : get_sampling_on_child_threads_history().back();
    return _v;
}

void
pthread_gotcha::start()
{
    configure();
    get_bundle()->start();
}

void
pthread_gotcha::stop()
{
    get_bundle()->stop();
}
}  // namespace omnitrace
