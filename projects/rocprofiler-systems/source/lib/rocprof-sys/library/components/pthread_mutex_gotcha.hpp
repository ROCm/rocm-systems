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

#pragma once

#include "core/common.hpp"
#include "core/defines.hpp"
#include "core/timemory.hpp"

#include <timemory/components/gotcha/backends.hpp>
#include <timemory/mpl/macros.hpp>

#include <array>
#include <cstddef>
#include <string>

namespace rocprofsys
{
namespace component
{
// this is used to wrap pthread_mutex()
struct pthread_mutex_gotcha : comp::base<pthread_mutex_gotcha, void>
{
    static constexpr size_t gotcha_capacity = 13;
    using hash_array_t                      = std::array<size_t, gotcha_capacity>;
    using gotcha_data_t                     = comp::gotcha_data;

    ROCPROFSYS_DEFAULT_OBJECT(pthread_mutex_gotcha)

    explicit pthread_mutex_gotcha(const gotcha_data_t&);

    // string id for component
    static std::string label() { return "pthread_mutex_gotcha"; }

    // generate the gotcha wrappers
    static void configure();
    static void shutdown();

    int operator()(int (*)(pthread_mutex_t*), pthread_mutex_t*) const;
    int operator()(int (*)(pthread_spinlock_t*), pthread_spinlock_t*) const;
    int operator()(int (*)(pthread_rwlock_t*), pthread_rwlock_t*) const;
    int operator()(int (*)(pthread_barrier_t*), pthread_barrier_t*) const;
    int operator()(int (*)(pthread_t, void**), pthread_t, void**) const;

private:
    static bool          is_disabled();
    static hash_array_t& get_hashes();

    template <typename... Args>
    auto operator()(uintptr_t&&, int (*)(Args...), Args...) const;

    mutable bool         m_protect = false;
    const gotcha_data_t* m_data    = nullptr;
};

using pthread_mutex_gotcha_t = comp::gotcha<pthread_mutex_gotcha::gotcha_capacity,
                                            std::tuple<>, pthread_mutex_gotcha>;
}  // namespace component
}  // namespace rocprofsys

ROCPROFSYS_DEFINE_CONCRETE_TRAIT(fast_gotcha, component::pthread_mutex_gotcha_t,
                                 true_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(static_data, component::pthread_mutex_gotcha_t,
                                 true_type)

namespace tim
{
namespace policy
{
using pthread_mutex_gotcha   = ::rocprofsys::component::pthread_mutex_gotcha;
using pthread_mutex_gotcha_t = ::rocprofsys::component::pthread_mutex_gotcha_t;

template <>
struct static_data<pthread_mutex_gotcha, pthread_mutex_gotcha_t> : std::true_type
{
    template <size_t N>
    pthread_mutex_gotcha& operator()(std::integral_constant<size_t, N>,
                                     const component::gotcha_data& _data) const;
};
}  // namespace policy
}  // namespace tim
