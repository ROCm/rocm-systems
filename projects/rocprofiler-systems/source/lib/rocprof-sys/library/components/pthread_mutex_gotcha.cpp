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

#include "library/components/pthread_mutex_gotcha.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "core/utility.hpp"
#include "library/components/category_region.hpp"
#include "library/runtime.hpp"
#include "library/thread_info.hpp"

#include <timemory/backends/threading.hpp>
#include <timemory/utility/signals.hpp>
#include <timemory/utility/types.hpp>

#include <cstdint>
#include <pthread.h>
#include <stdexcept>

namespace rocprofsys
{
namespace component
{
pthread_mutex_gotcha::hash_array_t&
pthread_mutex_gotcha::get_hashes()
{
    // theoretically, this private function will NEVER be called until it
    // is called by a gotcha wrapper, which means the tool ids should be
    // fully populated. If that fails to be the case for some reason,
    // we could see weird results.
    static auto _v = []() {
        const auto& _data = pthread_mutex_gotcha_t::get_gotcha_data();
        auto        _init = hash_array_t{};
        auto        _skip = std::set<size_t>{};
        if(!config::get_trace_thread_locks())
        {
            for(size_t i = 0; i < 3; ++i)
                _skip.emplace(i);
        }
        if(!config::get_trace_thread_rwlocks())
        {
            for(size_t i = 3; i < 8; ++i)
                _skip.emplace(i);
        }
        if(!config::get_trace_thread_spin_locks())
        {
            for(size_t i = 9; i < 12; ++i)
                _skip.emplace(i);
        }
        if(!config::get_trace_thread_barriers()) _skip.emplace(8);
        if(!config::get_trace_thread_join()) _skip.emplace(12);
        for(size_t i = 0; i < gotcha_capacity; ++i)
        {
            auto&& _id = _data.at(i).tool_id;
            if(!_id.empty())
                _init.at(i) = tim::add_hash_id(_id.c_str());
            else
            {
                if(_skip.count(i) > 0) continue;
                ROCPROFSYS_VERBOSE(
                    1,
                    "WARNING!!! pthread_mutex_gotcha tool id at index %zu was empty!\n",
                    i);
            }
            ROCPROFSYS_CI_FAIL(
                _id.empty() || _init.at(i) == 0,
                "pthread_mutex_gotcha tool id at index %zu has no hash value\n", i);
        }
        return _init;
    }();
    return _v;
}

void
pthread_mutex_gotcha::configure()
{
    pthread_mutex_gotcha_t::get_initializer() = []() {
        if(!tim::settings::enabled() || get_use_causal()) return;

        if(config::get_trace_thread_locks())
        {
            pthread_mutex_gotcha_t::configure(
                comp::gotcha_config<0, int, pthread_mutex_t*>{ "pthread_mutex_lock" });

            pthread_mutex_gotcha_t::configure(
                comp::gotcha_config<1, int, pthread_mutex_t*>{ "pthread_mutex_unlock" });

            pthread_mutex_gotcha_t::configure(
                comp::gotcha_config<2, int, pthread_mutex_t*>{ "pthread_mutex_trylock" });
        }

        if(config::get_trace_thread_rwlocks())
        {
            pthread_mutex_gotcha_t::configure(
                comp::gotcha_config<3, int, pthread_rwlock_t*>{
                    "pthread_rwlock_rdlock" });

            pthread_mutex_gotcha_t::configure(
                comp::gotcha_config<4, int, pthread_rwlock_t*>{
                    "pthread_rwlock_wrlock" });

            pthread_mutex_gotcha_t::configure(
                comp::gotcha_config<5, int, pthread_rwlock_t*>{
                    "pthread_rwlock_tryrdlock" });

            pthread_mutex_gotcha_t::configure(
                comp::gotcha_config<6, int, pthread_rwlock_t*>{
                    "pthread_rwlock_trywrlock" });

            pthread_mutex_gotcha_t::configure(
                comp::gotcha_config<7, int, pthread_rwlock_t*>{
                    "pthread_rwlock_unlock" });
        }

        if(config::get_trace_thread_barriers())
        {
            pthread_mutex_gotcha_t::configure(
                comp::gotcha_config<8, int, pthread_barrier_t*>{
                    "pthread_barrier_wait" });
        }

        if(config::get_trace_thread_spin_locks())
        {
            pthread_mutex_gotcha_t::configure(
                comp::gotcha_config<9, int, pthread_spinlock_t*>{ "pthread_spin_lock" });

            pthread_mutex_gotcha_t::configure(
                comp::gotcha_config<10, int, pthread_spinlock_t*>{
                    "pthread_spin_trylock" });

            pthread_mutex_gotcha_t::configure(
                comp::gotcha_config<11, int, pthread_spinlock_t*>{
                    "pthread_spin_unlock" });
        }

        if(config::get_trace_thread_join())
        {
            pthread_mutex_gotcha_t::configure(
                comp::gotcha_config<12, int, pthread_t, void**>{ "pthread_join" });
        }
    };
}

void
pthread_mutex_gotcha::shutdown()
{
    pthread_mutex_gotcha_t::disable();
}

pthread_mutex_gotcha::pthread_mutex_gotcha(const gotcha_data_t& _data)
: m_data{ &_data }
{}

template <typename... Args>
auto
pthread_mutex_gotcha::operator()(uintptr_t&&, int (*_callee)(Args...),
                                 Args... _args) const
{
    using bundle_t = category_region<category::pthread>;

    if(is_disabled() || m_protect)
    {
        if(!_callee)
        {
            if(m_data)
            {
                ROCPROFSYS_PRINT("Warning! nullptr to %s\n", m_data->tool_id.c_str());
            }
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

    bundle_t::audit(std::string_view{ m_data->tool_id }, audit::incoming{}, _args...);
    auto _ret = (*_callee)(_args...);
    bundle_t::audit(std::string_view{ m_data->tool_id }, audit::outgoing{}, _ret);

    return _ret;
}

int
pthread_mutex_gotcha::operator()(int (*_callee)(pthread_mutex_t*),
                                 pthread_mutex_t* _mutex) const
{
    if(get_state() != ::rocprofsys::State::Active || m_protect) return (*_callee)(_mutex);
    return (*this)(reinterpret_cast<uintptr_t>(_mutex), _callee, _mutex);
}

int
pthread_mutex_gotcha::operator()(int (*_callee)(pthread_spinlock_t*),
                                 pthread_spinlock_t* _lock) const
{
    if(get_state() != ::rocprofsys::State::Active || m_protect) return (*_callee)(_lock);
    return (*this)(reinterpret_cast<uintptr_t>(_lock), _callee, _lock);
}

int
pthread_mutex_gotcha::operator()(int (*_callee)(pthread_rwlock_t*),
                                 pthread_rwlock_t* _lock) const
{
    if(get_state() != ::rocprofsys::State::Active || m_protect) return (*_callee)(_lock);
    return (*this)(reinterpret_cast<uintptr_t>(_lock), _callee, _lock);
}

int
pthread_mutex_gotcha::operator()(int (*_callee)(pthread_barrier_t*),
                                 pthread_barrier_t* _barrier) const
{
    if(get_state() != ::rocprofsys::State::Active || m_protect)
        return (*_callee)(_barrier);
    return (*this)(reinterpret_cast<uintptr_t>(_barrier), _callee, _barrier);
}

int
pthread_mutex_gotcha::operator()(int (*_callee)(pthread_t, void**), pthread_t _thr,
                                 void** _tinfo) const
{
    if(get_state() != ::rocprofsys::State::Active || m_protect)
        return (*_callee)(_thr, _tinfo);
    return (*this)(static_cast<uintptr_t>(threading::get_id()), _callee, _thr, _tinfo);
}

bool
pthread_mutex_gotcha::is_disabled()
{
    static thread_local const auto& _info = thread_info::get();
    return (!_info || _info->is_offset || get_state() != ::rocprofsys::State::Active ||
            get_thread_state() != ThreadState::Enabled);
}
}  // namespace component
}  // namespace rocprofsys

namespace tim
{
namespace policy
{
template <size_t N>
pthread_mutex_gotcha&
static_data<pthread_mutex_gotcha, pthread_mutex_gotcha_t>::operator()(
    std::integral_constant<size_t, N>, const component::gotcha_data& _data) const
{
    using thread_data_t =
        rocprofsys::thread_data<pthread_mutex_gotcha, std::integral_constant<size_t, N>>;
    static thread_local auto& _v =
        thread_data_t::instance(rocprofsys::construct_on_thread{}, _data);
    return *_v;
}
}  // namespace policy
}  // namespace tim
