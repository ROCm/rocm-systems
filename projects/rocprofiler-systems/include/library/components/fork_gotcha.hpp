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

#include "library/common.hpp"
#include "library/defines.hpp"
#include "library/timemory.hpp"

namespace omnitrace
{
// this is used to wrap fork()
struct fork_gotcha : comp::base<fork_gotcha, void>
{
    using gotcha_data_t = comp::gotcha_data;

    TIMEMORY_DEFAULT_OBJECT(fork_gotcha)

    // string id for component
    static std::string label() { return "fork_gotcha"; }

    // generate the gotcha wrappers
    static void configure();

    // this will get called right before fork
    static void audit(const gotcha_data_t& _data, audit::incoming);

    // this will get called right after fork with the return value
    static void audit(const gotcha_data_t& _data, audit::outgoing, pid_t _pid);
};

using fork_gotcha_t = comp::gotcha<4, tim::component_tuple<fork_gotcha>, api::omnitrace>;
}  // namespace omnitrace
