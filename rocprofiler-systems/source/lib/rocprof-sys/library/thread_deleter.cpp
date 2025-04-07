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

#include "library/thread_deleter.hpp"
#include "api.hpp"
#include "core/utility.hpp"
#include "library/components/pthread_create_gotcha.hpp"
#include "library/thread_info.hpp"

#include <timemory/backends/threading.hpp>
#include <timemory/components/timing/backends.hpp>

namespace rocprofsys
{
template struct component_bundle_cache_impl<instrumentation_bundle_t>;

void
thread_deleter<void>::operator()() const
{
    // called after thread info is deleted
    if(!thread_info::exists()) return;

    const auto& _info = thread_info::get();
    if(_info && _info->index_data)
    {
        auto _tid = _info->index_data->sequent_value;

        component::pthread_create_gotcha::shutdown(_tid);
        set_thread_state(ThreadState::Completed);
        if(get_state() < State::Finalized && _tid == 0) rocprofsys_finalize_hidden();
    }
    else
    {
        set_thread_state(ThreadState::Completed);
    }
}

template struct thread_deleter<void>;
}  // namespace rocprofsys
