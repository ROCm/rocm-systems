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

#include "library/causal/components/blocking_gotcha.hpp"
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

#include <atomic>
#include <csignal>
#include <cstdint>
#include <pthread.h>
#include <stdexcept>
#include <type_traits>

#pragma weak pthread_join
#pragma weak pthread_mutex_lock
#pragma weak pthread_spin_lock
#pragma weak pthread_cond_wait
#pragma weak pthread_rwlock_rdlock
#pragma weak pthread_rwlock_wrlock
#pragma weak pthread_tryjoin_np
#pragma weak pthread_timedjoin_np
#pragma weak pthread_cond_timedwait
#pragma weak pthread_rwlock_timedrdlock
#pragma weak pthread_rwlock_timedwrlock
#pragma weak pthread_mutex_trylock
#pragma weak pthread_spin_trylock
#pragma weak pthread_rwlock_trywrlock
#pragma weak sigwait
#pragma weak sigwaitinfo
#pragma weak sigtimedwait
#pragma weak sigsuspend

namespace rocprofsys
{
namespace causal
{
namespace component
{
std::string
blocking_gotcha::label()
{
    return "causal_blocking_gotcha";
}

std::string
blocking_gotcha::description()
{
    return "Handles executing all necessary pauses before the thread performs some "
           "blocking function";
}

void
blocking_gotcha::preinit()
{
    configure();
}

void
blocking_gotcha::configure()
{
    blocking_gotcha_t::get_initializer() = []() {
        if(!config::get_use_causal()) return;

        // postblock(true)
        //  - pthread_join
        //  - pthread_mutex_lock
        //  - pthread_cond_wait
        //  - pthread_barrier_wait
        //  - pthread_rwlock_rdlock
        //  - pthread_rwlock_wrlock

        TIMEMORY_C_GOTCHA(blocking_gotcha_t, 0, pthread_join);
        TIMEMORY_C_GOTCHA(blocking_gotcha_t, 1, pthread_mutex_lock);
        TIMEMORY_C_GOTCHA(blocking_gotcha_t, 2, pthread_spin_lock);
        TIMEMORY_C_GOTCHA(blocking_gotcha_t, 3, pthread_cond_wait);
        TIMEMORY_C_GOTCHA(blocking_gotcha_t, 4, pthread_rwlock_rdlock);
        TIMEMORY_C_GOTCHA(blocking_gotcha_t, 5, pthread_rwlock_wrlock);

        // postblock(result == 0)
        //  - pthread_tryjoin_np
        //  - pthread_timedjoin_np
        //  - pthread_cond_timedwait
        //  - pthread_rwlock_timedrdlock
        //  - pthread_rwlock_timedwrlock

        TIMEMORY_C_GOTCHA(blocking_gotcha_t, 6, pthread_tryjoin_np);
        TIMEMORY_C_GOTCHA(blocking_gotcha_t, 7, pthread_timedjoin_np);
        TIMEMORY_C_GOTCHA(blocking_gotcha_t, 8, pthread_cond_timedwait);
        TIMEMORY_C_GOTCHA(blocking_gotcha_t, 9, pthread_rwlock_timedrdlock);
        TIMEMORY_C_GOTCHA(blocking_gotcha_t, 10, pthread_rwlock_timedwrlock);

        TIMEMORY_C_GOTCHA(blocking_gotcha_t, 11, pthread_mutex_trylock);
        TIMEMORY_C_GOTCHA(blocking_gotcha_t, 12, pthread_spin_trylock);
        TIMEMORY_C_GOTCHA(blocking_gotcha_t, 13, pthread_rwlock_trywrlock);

        // postblock(...)
        //  - sigwait
        //  - sigwaitinfo
        //  - sigtimedwait
        TIMEMORY_C_GOTCHA(blocking_gotcha_t, 14, sigwait);
        TIMEMORY_C_GOTCHA(blocking_gotcha_t, 15, sigwaitinfo);
        TIMEMORY_C_GOTCHA(blocking_gotcha_t, 16, sigtimedwait);

        // other
        TIMEMORY_C_GOTCHA(blocking_gotcha_t, 17, sigsuspend);
    };
}

void
blocking_gotcha::shutdown()
{
    blocking_gotcha_t::disable();
}

template <size_t Idx, typename Ret, typename... Args>
std::enable_if_t<(Idx <= blocking_gotcha::indexes::maybe_post_block_max_idx), Ret>
blocking_gotcha::operator()(gotcha_index<Idx>, Ret (*_func)(Args...),
                            Args... _args) const noexcept
{
    int64_t _delay_value = causal::delay::get_global().load(std::memory_order_relaxed);

    causal::sampling::block_backtrace_samples();
    auto _ret = (*_func)(_args...);
    causal::sampling::unblock_backtrace_samples();

    if(get_thread_state() < ::rocprofsys::ThreadState::Internal)
    {
        if constexpr(Idx >= always_post_block_min_idx && Idx <= always_post_block_max_idx)
        {
            causal::delay::postblock(_delay_value);
        }
        else if constexpr(Idx >= maybe_post_block_min_idx &&
                          Idx <= maybe_post_block_max_idx)
        {
            if(_ret == 0) causal::delay::postblock(_delay_value);
        }
        else
        {
            static_assert(Idx > maybe_post_block_max_idx, "Error! bad overload");
        }
    }

    return _ret;
}

int
blocking_gotcha::operator()(gotcha_index<sigwait_idx>, int (*)(const sigset_t*, int*),
                            const sigset_t* _set_v, int* _sig) const noexcept
{
    auto _active = get_thread_state() < ::rocprofsys::ThreadState::Internal;

    sigset_t _set = *_set_v;
    causal_gotcha::remove_signals(&_set);
    siginfo_t _info;

    int64_t _delay_value = (_active) ? causal::delay::get_global().load() : 0;

    auto* _data         = blocking_gotcha_t::at(16);
    auto  f_sigwaitinfo = reinterpret_cast<decltype(&sigwaitinfo)>(_data->wrappee);

    causal::sampling::block_backtrace_samples();
    auto _ret = (*f_sigwaitinfo)(&_set, &_info);
    causal::sampling::unblock_backtrace_samples();

    // Woken up by another thread if the call did not fail and this is waking process
    if(_active && _ret != -1 && _info.si_pid == process::get_id())
        causal::delay::postblock(_delay_value);

    if(_ret == -1)
        return errno;  // If there was an error, return the error code
    else
        *_sig = _ret;  // sig is declared as non-null so skip check

    return 0;
}

int
blocking_gotcha::operator()(gotcha_index<sigwaitinfo_idx>,
                            int (*_func)(const sigset_t*, siginfo_t*),
                            const sigset_t* _set_v, siginfo_t* _info_v) const noexcept
{
    auto _active = get_thread_state() < ::rocprofsys::ThreadState::Internal;

    sigset_t _set = *_set_v;
    causal_gotcha::remove_signals(&_set);
    siginfo_t _info;

    int64_t _delay_value = (_active) ? causal::delay::get_global().load() : 0;

    causal::sampling::block_backtrace_samples();
    auto _ret = (*_func)(&_set, &_info);
    causal::sampling::unblock_backtrace_samples();

    // Woken up by another thread if the call did not fail and this is waking process
    if(_active && _ret > 0 && _info.si_pid == process::get_id())
        causal::delay::postblock(_delay_value);

    if(_ret > 0 && _info_v) *_info_v = _info;

    return _ret;
}

int
blocking_gotcha::operator()(gotcha_index<sigtimedwait_idx>,
                            int (*_func)(const sigset_t*, siginfo_t*,
                                         const struct timespec*),
                            const sigset_t* _set_v, siginfo_t* _info_v,
                            const struct timespec* _wait_v) const noexcept
{
    auto _active = get_thread_state() < ::rocprofsys::ThreadState::Internal;

    sigset_t _set = *_set_v;
    causal_gotcha::remove_signals(&_set);
    siginfo_t _info;

    int64_t _delay_value = (_active) ? causal::delay::get_global().load() : 0;

    causal::sampling::block_backtrace_samples();
    auto _ret = (*_func)(&_set, &_info, _wait_v);
    causal::sampling::unblock_backtrace_samples();

    // Woken up by another thread if the call did not fail and this is waking process
    if(_active && _ret > 0 && _info.si_pid == process::get_id())
        causal::delay::postblock(_delay_value);

    if(_ret > 0 && _info_v) *_info_v = _info;

    return _ret;
}

int
blocking_gotcha::operator()(gotcha_index<sigsuspend_idx>, int (*)(const sigset_t*),
                            const sigset_t* _set_v) const noexcept
{
    auto _old_set = sigset_t{};
    int  _sig     = 0;
    ::sigprocmask(SIG_SETMASK, _set_v, &_old_set);
    // sigwait is wrapped so no need to block/unblock signals
    auto _ret = ::sigwait(_set_v, &_sig);
    ::sigprocmask(SIG_SETMASK, &_old_set, nullptr);

    return _ret;
}
}  // namespace component
}  // namespace causal
}  // namespace rocprofsys

TIMEMORY_INVOKE_PREINIT(rocprofsys::causal::component::blocking_gotcha)
