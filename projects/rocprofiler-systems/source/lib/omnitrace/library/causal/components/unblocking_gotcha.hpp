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

#include <array>
#include <cstddef>
#include <string>

namespace omnitrace
{
namespace causal
{
namespace component
{
struct unblocking_gotcha : comp::base<unblocking_gotcha, void>
{
    static constexpr size_t gotcha_capacity = 9;

    OMNITRACE_DEFAULT_OBJECT(unblocking_gotcha)

    // string id for component
    static std::string label();
    static std::string description();
    static void        preinit();

    // generate the gotcha wrappers
    static void configure();
    static void shutdown();

    template <typename Ret, typename... Args>
    Ret operator()(const comp::gotcha_data&, Ret (*)(Args...), Args...) const noexcept;

    int operator()(const comp::gotcha_data&, int (*)(pid_t, int), pid_t,
                   int) const noexcept;
};

using unblocking_gotcha_t =
    comp::gotcha<unblocking_gotcha::gotcha_capacity, tim::type_list<>, unblocking_gotcha>;
}  // namespace component
}  // namespace causal
}  // namespace omnitrace

OMNITRACE_DEFINE_CONCRETE_TRAIT(prevent_reentry, causal::component::unblocking_gotcha_t,
                                false_type)
OMNITRACE_DEFINE_CONCRETE_TRAIT(static_data, causal::component::unblocking_gotcha_t,
                                false_type)
OMNITRACE_DEFINE_CONCRETE_TRAIT(fast_gotcha, causal::component::unblocking_gotcha_t,
                                true_type)
