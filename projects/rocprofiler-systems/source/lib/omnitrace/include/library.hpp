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

#pragma once

// this always needs to included first
// clang-format off
#include "library/perfetto.hpp"
// clang-format on

#include "library/timemory.hpp"
#include "library/components/roctracer.hpp"
#include "library/api.hpp"
#include "library/components/fork_gotcha.hpp"
#include "library/components/mpi_gotcha.hpp"
#include "library/api.hpp"
#include "library/common.hpp"
#include "library/state.hpp"
#include "library/config.hpp"
#include "library/thread_data.hpp"
#include "library/ptl.hpp"
#include "library/debug.hpp"
#include "library/critical_trace.hpp"
#include "library/runtime.hpp"

#include <timemory/macros/language.hpp>
#include <timemory/utility/utility.hpp>

#include <mutex>

namespace omnitrace
{
template <critical_trace::Device DevID, critical_trace::Phase PhaseID,
          bool UpdateStack = true>
inline void
add_critical_trace(int64_t _targ_tid, size_t _cpu_cid, size_t _gpu_cid,
                   size_t _parent_cid, int64_t _ts_beg, int64_t _ts_val, size_t _hash,
                   uint16_t _depth, uint16_t _prio = 0)
{
    // clang-format off
    // these are used to create unique type mutexes
    struct critical_insert {};
    struct cpu_cid_stack {};
    // clang-format on

    using tim::type_mutex;
    using auto_lock_t                  = tim::auto_lock_t;
    static constexpr auto num_mutexes  = max_supported_threads;
    static auto           _update_freq = critical_trace::get_update_frequency();

    auto _self_tid = threading::get_id();

    if constexpr(PhaseID != critical_trace::Phase::NONE)
    {
        auto& _self_mtx =
            type_mutex<critical_insert, api::omnitrace, num_mutexes>(_self_tid);

        auto_lock_t _self_lk{ _self_mtx, std::defer_lock };

        // unique lock per thread
        if(!_self_lk.owns_lock()) _self_lk.lock();

        auto& _critical_trace = critical_trace::get(_self_tid);
        _critical_trace->emplace_back(
            critical_trace::entry{ _prio, DevID, PhaseID, _depth, _targ_tid, _cpu_cid,
                                   _gpu_cid, _parent_cid, _ts_beg, _ts_val, _hash });
    }

    if constexpr(UpdateStack)
    {
        auto& _self_mtx = get_cpu_cid_stack_lock(_self_tid);
        auto& _targ_mtx = get_cpu_cid_stack_lock(_targ_tid);

        auto_lock_t _self_lk{ _self_mtx, std::defer_lock };
        auto_lock_t _targ_lk{ _targ_mtx, std::defer_lock };

        // unique lock per thread
        auto _lock = [&_self_lk, &_targ_lk, _self_tid, _targ_tid]() {
            if(!_self_lk.owns_lock() && _self_tid != _targ_tid) _self_lk.lock();
            if(!_targ_lk.owns_lock()) _targ_lk.lock();
        };

        if constexpr(PhaseID == critical_trace::Phase::NONE)
        {
            _lock();
            get_cpu_cid_stack(_targ_tid)->emplace_back(_cpu_cid);
        }
        else if constexpr(PhaseID == critical_trace::Phase::BEGIN)
        {
            _lock();
            get_cpu_cid_stack(_targ_tid)->emplace_back(_cpu_cid);
        }
        else if constexpr(PhaseID == critical_trace::Phase::END)
        {
            _lock();
            get_cpu_cid_stack(_targ_tid)->pop_back();
            if(_gpu_cid == 0 && _cpu_cid % _update_freq == (_update_freq - 1))
                critical_trace::update(_targ_tid);
        }
    }

    tim::consume_parameters(_targ_tid, _cpu_cid, _gpu_cid, _parent_cid, _ts_beg, _ts_val,
                            _hash, _depth, _prio);
}
}  // namespace omnitrace
