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

#include "library/api.hpp"
#include "library/common.hpp"
#include "library/config.hpp"
#include "library/defines.hpp"
#include "library/timemory.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <type_traits>

#if !defined(OMNITRACE_MAX_THREADS)
#    define OMNITRACE_MAX_THREADS 1024
#endif

namespace omnitrace
{
// bundle of components used in instrumentation
using instrumentation_bundle_t =
    tim::component_bundle<api::omnitrace, comp::wall_clock*, comp::user_global_bundle*>;

// allocator for instrumentation_bundle_t
using bundle_allocator_t = tim::data::ring_buffer_allocator<instrumentation_bundle_t>;

template <typename Tp>
struct thread_deleter;

// unique ptr type for omnitrace
template <typename Tp>
using unique_ptr_t = std::unique_ptr<Tp, thread_deleter<Tp>>;

static constexpr size_t max_supported_threads = OMNITRACE_MAX_THREADS;

template <typename Tp>
struct thread_deleter
{
    void operator()(Tp* ptr) const
    {
        if(get_state() != State::Finalized) omnitrace_finalize_hidden();
        delete ptr;
    }
};

template <typename Tp, typename Tag = void, size_t MaxThreads = max_supported_threads>
struct thread_data
{
    using instance_array_t  = std::array<unique_ptr_t<Tp>, MaxThreads>;
    using construct_on_init = std::true_type;

    template <typename... Args>
    static void              construct(Args&&...);
    static unique_ptr_t<Tp>& instance();
    static instance_array_t& instances();
    template <typename... Args>
    static unique_ptr_t<Tp>& instance(construct_on_init, Args&&...);
    template <typename... Args>
    static instance_array_t& instances(construct_on_init, Args&&...);

    static constexpr size_t size() { return MaxThreads; }

    decltype(auto) begin() { return instances().begin(); }
    decltype(auto) end() { return instances().end(); }

    decltype(auto) begin() const { return instances().begin(); }
    decltype(auto) end() const { return instances().end(); }
};

template <typename Tp, typename Tag, size_t MaxThreads>
template <typename... Args>
void
thread_data<Tp, Tag, MaxThreads>::construct(Args&&... _args)
{
    // construct outside of lambda to prevent data-race
    static auto&             _instances = instances();
    static thread_local bool _v         = [&_args...]() {
        _instances.at(threading::get_id()) =
            unique_ptr_t<Tp>{ new Tp(std::forward<Args>(_args)...) };
        return true;
    }();
    (void) _v;
}

template <typename Tp, typename Tag, size_t MaxThreads>
unique_ptr_t<Tp>&
thread_data<Tp, Tag, MaxThreads>::instance()
{
    return instances().at(threading::get_id());
}

template <typename Tp, typename Tag, size_t MaxThreads>
typename thread_data<Tp, Tag, MaxThreads>::instance_array_t&
thread_data<Tp, Tag, MaxThreads>::instances()
{
    static auto _v = instance_array_t{};
    return _v;
}

template <typename Tp, typename Tag, size_t MaxThreads>
template <typename... Args>
unique_ptr_t<Tp>&
thread_data<Tp, Tag, MaxThreads>::instance(construct_on_init, Args&&... _args)
{
    construct(std::forward<Args>(_args)...);
    return instances().at(threading::get_id());
}

template <typename Tp, typename Tag, size_t MaxThreads>
template <typename... Args>
typename thread_data<Tp, Tag, MaxThreads>::instance_array_t&
thread_data<Tp, Tag, MaxThreads>::instances(construct_on_init, Args&&... _args)
{
    static auto _v = [&]() {
        auto _internal = instance_array_t{};
        for(size_t i = 0; i < MaxThreads; ++i)
            _internal.at(i) = unique_ptr_t<Tp>{ new Tp(std::forward<Args>(_args)...) };
        return _internal;
    }();
    return _v;
}

//--------------------------------------------------------------------------------------//

// there are currently some strange things that happen with
// vector<instrumentation_bundle_t> so using vector<instrumentation_bundle_t*> and
// timemory's ring_buffer_allocator to create contiguous memory-page aligned instances of
// the bundle
struct instrumentation_bundles
{
    using instance_array_t = std::array<instrumentation_bundles, max_supported_threads>;

    bundle_allocator_t                     allocator{};
    std::vector<instrumentation_bundle_t*> bundles{};

    static instance_array_t& instances();
};
}  // namespace omnitrace
