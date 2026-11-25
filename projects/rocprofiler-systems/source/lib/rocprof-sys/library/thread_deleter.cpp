// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/thread_deleter.hpp"
#include "api.hpp"
#include "core/utility.hpp"
#include "library/components/pthread_create_gotcha.hpp"
#include "library/runtime.hpp"
#include "library/thread_info.hpp"

#include <timemory/backends/threading.hpp>

namespace rocprofsys
{
template struct component_bundle_cache_impl<instrumentation_bundle_t>;

void
thread_deleter<void>::operator()() const
{
    // called after thread info is deleted
    if(!thread_info::exists())
    {
        return;
    }

    const auto& _info = thread_info::get();
    if(_info && _info->index_data)
    {
        auto _tid = _info->index_data->sequent_value;

        auto const is_root_main_thread =
            !is_child_process() && _tid == k_main_thread_index;
        if(is_root_main_thread)
        {
            component::pthread_create_gotcha::shutdown();
        }
        state::thread::set(state::thread::Completed);
        if(state::process::get() < state::process::Finalized && is_root_main_thread)
        {
            rocprofsys_finalize_hidden();
        }
    }
    else
    {
        state::thread::set(state::thread::Completed);
    }
}

template struct thread_deleter<void>;
}  // namespace rocprofsys
