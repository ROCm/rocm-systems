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

#include "library/runtime.hpp"
#include "api.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "core/defines.hpp"
#include "core/utility.hpp"
#include "library/thread_data.hpp"
#include "library/thread_info.hpp"

#include <timemory/backends/dmp.hpp>
#include <timemory/backends/mpi.hpp>
#include <timemory/backends/process.hpp>
#include <timemory/backends/threading.hpp>
#include <timemory/components/rusage/backends.hpp>
#include <timemory/environment.hpp>
#include <timemory/process/process.hpp>
#include <timemory/sampling/allocator.hpp>
#include <timemory/settings.hpp>
#include <timemory/settings/types.hpp>
#include <timemory/utility/argparse.hpp>
#include <timemory/utility/declaration.hpp>
#include <timemory/utility/signals.hpp>

#include <array>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <ostream>
#include <string>
#include <unistd.h>

namespace rocprofsys
{
namespace
{
auto root_process_id =
    get_env<pid_t>("ROCPROFSYS_ROOT_PROCESS", process::get_id(), false);

auto&
get_sampling_on_child_threads_history(int64_t _idx = utility::get_thread_index())
{
    static auto _v = utility::get_filled_array<ROCPROFSYS_MAX_THREADS>(
        []() { return utility::get_reserved_vector<bool>(64); });

    if(_idx >= ROCPROFSYS_MAX_THREADS)
    {
        static thread_local auto _tl_v = utility::get_reserved_vector<bool>(128);
        return _tl_v;
    }

    return _v.at(_idx);
}

bool&
sampling_on_child_threads()
{
    static const auto& _thr_info = thread_info::get();
    // if the thread is offset, disable by default
    // if the global state is not active or the thread state is not enabled, disable by
    // default if there is no history, disable by default (first thread) otherwise,
    // inherit the last state
    static thread_local bool _v =
        (_thr_info) ? !_thr_info->is_offset
        : (get_state() != State::Active || get_thread_state() != ThreadState::Enabled)
            ? false
            : (get_sampling_on_child_threads_history().empty()
                   ? false
                   : get_sampling_on_child_threads_history().back());
    return _v;
}
}  // namespace

std::atomic<uint64_t>&
get_cpu_cid()
{
    static std::atomic<uint64_t> _v{ 0 };
    return _v;
}

unique_ptr_t<std::vector<uint64_t>>&
get_cpu_cid_stack(int64_t _tid, int64_t _parent)
{
    struct rocprofsys_cpu_cid_stack
    {};
    using init_data_t   = thread_data<bool, rocprofsys_cpu_cid_stack>;
    using thread_data_t = thread_data<std::vector<uint64_t>, rocprofsys_cpu_cid_stack>;

    auto& _v_tid = thread_data_t::instance(construct_on_thread{ _tid });
    auto& _b_tid = init_data_t::instance(construct_on_thread{ _tid }, false);

    if(_b_tid && !(*_b_tid))
    {
        *_b_tid           = true;
        auto  _parent_tid = _parent;
        auto& _p_tid      = thread_data_t::instance(construct_on_thread{ _parent_tid });
        // if tid != parent and there is not a valid pointer for the provided parent
        // thread id set it to zero since that will always be valid
        if(_tid != _parent_tid && !_p_tid) _parent_tid = 0;
        // copy over the thread ids from the parent if tid != parent
        if(_tid != _parent_tid) *_v_tid = *_p_tid;
    }
    return _v_tid;
}

unique_ptr_t<cpu_cid_parent_map_t>&
get_cpu_cid_parents(int64_t _tid)
{
    struct rocprofsys_cpu_cid_stack
    {};
    using thread_data_t = thread_data<cpu_cid_parent_map_t, rocprofsys_cpu_cid_stack>;
    return thread_data_t::instance(construct_on_thread{ _tid }, cpu_cid_parent_map_t{});
}

std::tuple<uint64_t, uint64_t, uint32_t>
create_cpu_cid_entry(int64_t _tid)
{
    using tim::auto_lock_t;

    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);

    // unique lock for _tid
    auto&       _mtx = get_cpu_cid_stack_lock(_tid);
    auto_lock_t _lk{ _mtx, std::defer_lock };
    if(!_lk.owns_lock()) _lk.lock();

    int64_t _p_idx = (get_cpu_cid_stack(_tid)->empty()) ? 0 : _tid;

    auto&       _p_mtx = get_cpu_cid_stack_lock(_p_idx);
    auto_lock_t _p_lk{ _p_mtx, std::defer_lock };
    if(!_p_lk.owns_lock()) _p_lk.lock();

    auto&& _cid = get_cpu_cid()++;
    // auto&&     _parent_cid = get_cpu_cid_stack(_p_idx)->back();
    uint64_t _parent_cid = 0;
    auto&    cid_stack   = get_cpu_cid_stack(_p_idx);

    if(!cid_stack->empty())
    {
        _parent_cid = cid_stack->back();
    }

    uint32_t&& _depth = get_cpu_cid_stack(_p_idx)->size() - ((_p_idx == _tid) ? 1 : 0);

    get_cpu_cid_parents(_tid)->emplace(_cid, std::make_tuple(_parent_cid, _depth));
    return std::make_tuple(_cid, _parent_cid, _depth);
}

cpu_cid_pair_t
get_cpu_cid_entry(uint64_t _cid, int64_t _tid)
{
    return get_cpu_cid_parents(_tid)->at(_cid);
}

tim::mutex_t&
get_cpu_cid_stack_lock(int64_t _tid)
{
    struct cpu_cid_stack_s
    {};
    return tim::type_mutex<cpu_cid_stack_s, project::rocprofsys, max_supported_threads>(
        _tid);
}

namespace
{
void
setup_gotchas()
{
    static bool _initialized = false;
    if(_initialized) return;
    _initialized = true;

    ROCPROFSYS_BASIC_DEBUG(
        "Configuring gotcha wrapper around fork, MPI_Init, and MPI_Init_thread\n");

    component::mpi_gotcha::configure();
    component::exit_gotcha::configure();
    component::fork_gotcha::configure();
}
}  // namespace

std::unique_ptr<main_bundle_t>&
get_main_bundle()
{
    static auto _v = []() {
        auto _self = RUSAGE_SELF;
        std::swap(_self, tim::get_rusage_type());
        auto _tmp = std::make_unique<main_bundle_t>(
            JOIN('/', "rocprofsys/process", process::get_id()),
            quirk::config<quirk::auto_start>{});
        std::swap(_self, tim::get_rusage_type());
        return _tmp;
    }();
    return _v;
}

std::unique_ptr<init_bundle_t>&
get_init_bundle()
{
    static auto _v = std::make_unique<init_bundle_t>(
        JOIN('/', "rocprofsys/process", process::get_id()));
    return _v;
}

std::unique_ptr<preinit_bundle_t>&
get_preinit_bundle()
{
    static auto _v =
        (setup_gotchas(), std::make_unique<preinit_bundle_t>(
                              JOIN('/', "rocprofsys/process", process::get_id()),
                              quirk::config<quirk::auto_start>{}));
    return _v;
}

bool
sampling_enabled_on_child_threads()
{
    return sampling_on_child_threads();
}

bool
push_enable_sampling_on_child_threads(bool _v)
{
    bool _last                  = sampling_on_child_threads();
    sampling_on_child_threads() = _v;
    auto& _hist                 = get_sampling_on_child_threads_history();
    _hist.emplace_back(_last);
    return _last;
}

bool
pop_enable_sampling_on_child_threads()
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
set_sampling_on_all_future_threads(bool _v)
{
    for(size_t i = 0; i < max_supported_threads; ++i)
        get_sampling_on_child_threads_history(i).emplace_back(_v);
}

pid_t
get_root_process_id()
{
    return root_process_id;
}

bool
is_root_process()
{
    return (root_process_id == process::get_id());
}

bool
is_child_process()
{
    return (root_process_id != process::get_id());
}
}  // namespace rocprofsys
