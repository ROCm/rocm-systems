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

#include "core/common.hpp"
#include "core/defines.hpp"
#include "core/timemory.hpp"

#include <timemory/components/gotcha/backends.hpp>
#include <timemory/mpl/macros.hpp>
#include <timemory/utility/types.hpp>

#include <array>
#include <cstddef>
#include <string>

namespace omnitrace
{
namespace causal
{
namespace component
{
struct blocking_gotcha : comp::base<blocking_gotcha, void>
{
    static constexpr size_t gotcha_capacity = 19;

    OMNITRACE_DEFAULT_OBJECT(blocking_gotcha)

    // string id for component
    static std::string label();
    static std::string description();
    static void        preinit();

    // generate the gotcha wrappers
    static void configure();
    static void shutdown();

    template <typename Ret, typename... Args>
    Ret operator()(const comp::gotcha_data&, Ret (*)(Args...), Args...) const noexcept;

    int operator()(const comp::gotcha_data&, int (*)(const sigset_t*, int*),
                   const sigset_t*, int*) const noexcept;

    int operator()(const comp::gotcha_data&, int (*)(const sigset_t*, siginfo_t*),
                   const sigset_t*, siginfo_t*) const noexcept;

    int operator()(const comp::gotcha_data&,
                   int (*)(const sigset_t*, siginfo_t*, const struct timespec*),
                   const sigset_t*, siginfo_t*, const struct timespec*) const noexcept;
};

using blocking_gotcha_t =
    comp::gotcha<blocking_gotcha::gotcha_capacity, tim::type_list<>, blocking_gotcha>;
}  // namespace component
}  // namespace causal
}  // namespace omnitrace

OMNITRACE_DEFINE_CONCRETE_TRAIT(prevent_reentry, causal::component::blocking_gotcha_t,
                                false_type)
OMNITRACE_DEFINE_CONCRETE_TRAIT(static_data, causal::component::blocking_gotcha_t,
                                false_type)
OMNITRACE_DEFINE_CONCRETE_TRAIT(fast_gotcha, causal::component::blocking_gotcha_t,
                                true_type)
