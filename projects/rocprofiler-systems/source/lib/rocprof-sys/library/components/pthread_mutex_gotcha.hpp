// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <timemory/components/base/declaration.hpp>

#include <cerrno>
#include <cstddef>
#include <mutex>
#include <pthread.h>
#include <string>
#include <string_view>

namespace rocprofsys::component
{

template <typename Policy>
struct pthread_mutex_gotcha : tim::component::base<pthread_mutex_gotcha<Policy>, void>
{
    using gotcha_data_t = typename Policy::gotcha_data_t;

    static constexpr size_t gotcha_capacity = 13;

    explicit pthread_mutex_gotcha(const gotcha_data_t& _data)
    : m_data{ &_data }
    {}

    static std::string label() { return "pthread_mutex_gotcha"; }

    static void configure()
    {
        Policy::gotcha_t::get_initializer() = []() {
            if(!Policy::settings_enabled() || Policy::get_use_causal()) return;

            if(Policy::get_trace_locks())
            {
                Policy::gotcha_t::template configure<0, int, pthread_mutex_t*>(
                    "pthread_mutex_lock");
                Policy::gotcha_t::template configure<1, int, pthread_mutex_t*>(
                    "pthread_mutex_unlock");
                Policy::gotcha_t::template configure<2, int, pthread_mutex_t*>(
                    "pthread_mutex_trylock");
            }

            if(Policy::get_trace_rwlocks())
            {
                Policy::gotcha_t::template configure<3, int, pthread_rwlock_t*>(
                    "pthread_rwlock_rdlock");
                Policy::gotcha_t::template configure<4, int, pthread_rwlock_t*>(
                    "pthread_rwlock_wrlock");
                Policy::gotcha_t::template configure<5, int, pthread_rwlock_t*>(
                    "pthread_rwlock_tryrdlock");
                Policy::gotcha_t::template configure<6, int, pthread_rwlock_t*>(
                    "pthread_rwlock_trywrlock");
                Policy::gotcha_t::template configure<7, int, pthread_rwlock_t*>(
                    "pthread_rwlock_unlock");
            }

            if(Policy::get_trace_barriers())
            {
                Policy::gotcha_t::template configure<8, int, pthread_barrier_t*>(
                    "pthread_barrier_wait");
            }

            if(Policy::get_trace_spin_locks())
            {
                Policy::gotcha_t::template configure<9, int, pthread_spinlock_t*>(
                    "pthread_spin_lock");
                Policy::gotcha_t::template configure<10, int, pthread_spinlock_t*>(
                    "pthread_spin_trylock");
                Policy::gotcha_t::template configure<11, int, pthread_spinlock_t*>(
                    "pthread_spin_unlock");
            }

            if(Policy::get_trace_join())
            {
                Policy::gotcha_t::template configure<12, int, pthread_t, void**>(
                    "pthread_join");
            }
        };
    }

    static void shutdown() { Policy::gotcha_t::disable(); }

    static void pause()
    {
        // s_is_paused is the real pause mechanism: pthread_mutex_gotcha_t carries the
        // fast_gotcha trait, so the framework installs fast_func (not wrap) as the PLT
        // wrapper. fast_func never reads _data.ready, making set_ready() a no-op here.
        // s_is_paused is checked first in is_disabled(), suppressing audit calls inside
        // operator() without affecting the underlying pthread function calls.
        s_is_paused.store(true, std::memory_order_relaxed);
        Policy::gotcha_t::set_ready(false);
    }

    static void resume()
    {
        s_is_paused.store(false, std::memory_order_relaxed);
        Policy::gotcha_t::set_ready(true);
    }

    int operator()(int (*_callee)(pthread_mutex_t*), pthread_mutex_t* _mutex) const
    {
        if(Policy::inactive_state() || m_protect)
        {
            return (*_callee)(_mutex);
        }
        return (*this)(reinterpret_cast<uintptr_t>(_mutex), _callee, _mutex);
    }

    int operator()(int (*_callee)(pthread_spinlock_t*), pthread_spinlock_t* _lock) const
    {
        if(Policy::inactive_state() || m_protect)
        {
            return (*_callee)(_lock);
        }
        return (*this)(reinterpret_cast<uintptr_t>(_lock), _callee, _lock);
    }

    int operator()(int (*_callee)(pthread_rwlock_t*), pthread_rwlock_t* _lock) const
    {
        if(Policy::inactive_state() || m_protect)
        {
            return (*_callee)(_lock);
        }
        return (*this)(reinterpret_cast<uintptr_t>(_lock), _callee, _lock);
    }

    int operator()(int (*_callee)(pthread_barrier_t*), pthread_barrier_t* _barrier) const
    {
        if(Policy::inactive_state() || m_protect)
        {
            return (*_callee)(_barrier);
        }
        return (*this)(reinterpret_cast<uintptr_t>(_barrier), _callee, _barrier);
    }

    int operator()(int (*_callee)(pthread_t, void**), pthread_t _thr, void** _tinfo) const
    {
        if(Policy::inactive_state() || m_protect)
        {
            return (*_callee)(_thr, _tinfo);
        }
        return (*this)(static_cast<uintptr_t>(Policy::get_thread_id()), _callee, _thr,
                       _tinfo);
    }

private:
    static bool is_disabled()
    {
        if(s_is_paused.load(std::memory_order_relaxed)) return true;
        return Policy::is_disabled_check();
    }

    template <typename... Args>
    auto operator()(uintptr_t /*addr*/, int (*_callee)(Args...), Args... _args) const
    {
        if(is_disabled() || m_protect)
        {
            if(!_callee)
            {
                return EINVAL;
            }
            return (*_callee)(_args...);
        }

        struct local_dtor
        {
            explicit local_dtor(bool& _v)
            : _protect{ _v }
            {}
            ~local_dtor() { _protect = false; }
            bool& _protect;
        } _dtor{ m_protect = true };

        if(m_data == nullptr)
        {
            throw std::runtime_error("pthread_mutex_gotcha gotcha_data is null.");
        }

        Policy::audit_incoming(std::string_view{ m_data->tool_id }, _args...);
        auto result = (*_callee)(_args...);
        Policy::audit_outgoing(std::string_view{ m_data->tool_id }, result);

        return result;
    }

    mutable bool             m_protect = false;
    const gotcha_data_t*     m_data    = nullptr;
    static std::atomic<bool> s_is_paused;
};

template <typename Policy>
std::atomic<bool> pthread_mutex_gotcha<Policy>::s_is_paused = false;

}  // namespace rocprofsys::component
