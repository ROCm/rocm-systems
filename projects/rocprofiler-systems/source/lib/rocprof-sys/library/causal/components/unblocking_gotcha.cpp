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

#include "library/causal/components/unblocking_gotcha.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "core/state.hpp"
#include "library/causal/components/causal_gotcha.hpp"
#include "library/causal/delay.hpp"
#include "library/causal/experiment.hpp"
#include "library/causal/sampling.hpp"
#include "library/runtime.hpp"

#include <timemory/components/macros.hpp>
#include <timemory/hash/types.hpp>
#include <timemory/utility/types.hpp>

#include <csignal>
#include <cstdint>
#include <pthread.h>
#include <stdexcept>

#pragma weak pthread_mutex_unlock
#pragma weak pthread_spin_unlock
#pragma weak pthread_cond_signal
#pragma weak pthread_cond_broadcast
#pragma weak pthread_kill
#pragma weak pthread_sigqueue
#pragma weak pthread_barrier_wait
#pragma weak kill

namespace rocprofsys
{
namespace causal
{
namespace component
{
std::string
unblocking_gotcha::label()
{
    return "causal_unblocking_gotcha";
}

std::string
unblocking_gotcha::description()
{
    return "Handles executing all necessary pauses before the thread performs some "
           "blocking function";
}

void
unblocking_gotcha::preinit()
{
    configure();
}

void
unblocking_gotcha::configure()
{
    unblocking_gotcha_t::get_initializer() = []() {
        if(!config::get_use_causal()) return;

        TIMEMORY_C_GOTCHA(unblocking_gotcha_t, 0, pthread_mutex_unlock);
        TIMEMORY_C_GOTCHA(unblocking_gotcha_t, 1, pthread_spin_unlock);
        TIMEMORY_C_GOTCHA(unblocking_gotcha_t, 2, pthread_rwlock_unlock);
        TIMEMORY_C_GOTCHA(unblocking_gotcha_t, 3, pthread_cond_signal);
        TIMEMORY_C_GOTCHA(unblocking_gotcha_t, 4, pthread_cond_broadcast);
        TIMEMORY_C_GOTCHA(unblocking_gotcha_t, 5, pthread_kill);
        TIMEMORY_C_GOTCHA(unblocking_gotcha_t, 6, pthread_sigqueue);
        TIMEMORY_C_GOTCHA(unblocking_gotcha_t, 7, pthread_barrier_wait);
        TIMEMORY_C_GOTCHA(unblocking_gotcha_t, 8, kill);
    };
}

void
unblocking_gotcha::shutdown()
{
    unblocking_gotcha_t::disable();
}

template <size_t Idx, typename Ret, typename... Args>
std::enable_if_t<(Idx < unblocking_gotcha::indexes::kill_idx), Ret>
unblocking_gotcha::operator()(gotcha_index<Idx>, Ret (*_func)(Args...),
                              Args... _args) const noexcept
{
    auto _active = get_thread_state() < ::rocprofsys::ThreadState::Internal;

    if(_active)
    {
        causal::delay::process();

        if constexpr(Idx == pthread_barrier_wait_idx)
        {
            int64_t _delay_value = (_active) ? causal::delay::get_global().load() : 0;

            causal::sampling::block_backtrace_samples();
            auto _ret = (*_func)(_args...);
            causal::sampling::unblock_backtrace_samples();

            causal::delay::postblock(_delay_value);
            return _ret;
        }
    }

    return (*_func)(_args...);
}

int
unblocking_gotcha::operator()(gotcha_index<kill_idx>, int (*_func)(pid_t, int),
                              pid_t _pid, int _sig) const noexcept
{
    auto _active = get_thread_state() < ::rocprofsys::ThreadState::Internal;

    if(_active && _pid == process::get_id()) causal::delay::process();

    causal::sampling::block_backtrace_samples();
    auto _ret = (*_func)(_pid, _sig);
    causal::sampling::unblock_backtrace_samples();

    return _ret;
}
}  // namespace component
}  // namespace causal
}  // namespace rocprofsys

TIMEMORY_INVOKE_PREINIT(rocprofsys::causal::component::unblocking_gotcha)
