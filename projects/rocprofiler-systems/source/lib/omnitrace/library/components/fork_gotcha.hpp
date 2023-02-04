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

namespace omnitrace
{
namespace component
{
// this is used to wrap fork()
struct fork_gotcha : comp::base<fork_gotcha, void>
{
    static constexpr size_t gotcha_capacity = 1;

    using gotcha_data_t = comp::gotcha_data;

    OMNITRACE_DEFAULT_OBJECT(fork_gotcha)

    // string id for component
    static std::string label() { return "fork_gotcha"; }

    // generate the gotcha wrappers
    static void configure();

    // this will get called right before fork
    static void audit(const gotcha_data_t& _data, audit::incoming);

    // this will get called right after fork with the return value
    static void audit(const gotcha_data_t& _data, audit::outgoing, pid_t _pid);

    // silence SFINAE disabled for omnitrace::fork_gotcha warnings
    static inline void start() {}
    static inline void stop() {}
};
}  // namespace component

using fork_gotcha_t =
    comp::gotcha<component::fork_gotcha::gotcha_capacity,
                 tim::component_tuple<component::fork_gotcha>, project::omnitrace>;
}  // namespace omnitrace
