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

#include "library/components/pthread_gotcha.hpp"
#include "library/config.hpp"
#include "library/debug.hpp"
#include "library/sampling.hpp"

#include <timemory/sampling/allocator.hpp>
#include <timemory/utility/types.hpp>

#include <pthread.h>

namespace omnitrace
{
namespace sampling
{
std::set<int>
setup();
std::set<int>
shutdown();
}  // namespace sampling

pthread_gotcha::wrapper::wrapper(routine_t _routine, void* _arg, bool _enable_sampling,
                                 promise_t* _p)
: m_enable_sampling{ _enable_sampling }
, m_routine{ _routine }
, m_arg{ _arg }
, m_promise{ _p }
{}

void*
pthread_gotcha::wrapper::operator()() const
{
    std::set<int> _signals{};
    auto&         _enable_sampling = pthread_gotcha::enable_sampling_on_child_threads();
    if(m_enable_sampling && _enable_sampling)
    {
        _enable_sampling = false;
        _signals         = sampling::setup();
        _enable_sampling = true;
        sampling::unblock_signals();
    }

    if(m_promise) m_promise->set_value();

    // execute the original function
    auto* _ret = m_routine(m_arg);

    if(m_enable_sampling && _enable_sampling)
    {
        sampling::block_signals(_signals);
        sampling::shutdown();
    }

    return _ret;
}

void*
pthread_gotcha::wrapper::wrap(void* _arg)
{
    if(_arg == nullptr) return nullptr;

    // convert the argument
    wrapper* _wrapper = static_cast<wrapper*>(_arg);

    // execute the original function
    return (*_wrapper)();
}

void
pthread_gotcha::configure()
{
    pthread_gotcha_t::get_initializer() = []() {
        TIMEMORY_C_GOTCHA(pthread_gotcha_t, 0, pthread_create);
    };
}

bool&
pthread_gotcha::enable_sampling_on_child_threads()
{
    static thread_local bool _v = get_use_sampling();
    return _v;
}

// pthread_create
int
pthread_gotcha::operator()(pthread_t* thread, const pthread_attr_t* attr,
                           void* (*start_routine)(void*), void*     arg) const
{
    auto _enable_sampling = enable_sampling_on_child_threads();

    if(!_enable_sampling)
    {
        auto* _obj = new wrapper(start_routine, arg, _enable_sampling, nullptr);
        // create the thread
        return pthread_create(thread, attr, &wrapper::wrap, static_cast<void*>(_obj));
    }

    // block the signals in entire process
    OMNITRACE_DEBUG("blocking signals...\n");
    tim::sampling::block_signals({ SIGALRM, SIGPROF },
                                 tim::sampling::sigmask_scope::process);

    // promise set by thread when signal handler is configured
    auto  _promise = std::promise<void>{};
    auto  _fut     = _promise.get_future();
    auto* _obj     = new wrapper(start_routine, arg, _enable_sampling, &_promise);

    // create the thread
    auto _ret = pthread_create(thread, attr, &wrapper::wrap, static_cast<void*>(_obj));

    // wait for thread to set promise
    OMNITRACE_DEBUG("waiting for child to signal it is setup...\n");
    _fut.wait();

    // unblock the signals in the entire process
    OMNITRACE_DEBUG("unblocking signals...\n");
    tim::sampling::unblock_signals({ SIGALRM, SIGPROF },
                                   tim::sampling::sigmask_scope::process);

    OMNITRACE_DEBUG("returning success...\n");
    return _ret;
}

}  // namespace omnitrace
