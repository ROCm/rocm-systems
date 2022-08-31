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

#include <timemory/components/base.hpp>
#include <timemory/components/gotcha/backends.hpp>

#include <cstdint>
#include <cstdlib>

namespace omnitrace
{
namespace component
{
struct exit_gotcha : tim::component::base<exit_gotcha, void>
{
    using gotcha_data  = tim::component::gotcha_data;
    using exit_func_t  = void (*)(int);
    using abort_func_t = void (*)();

    TIMEMORY_DEFAULT_OBJECT(exit_gotcha)

    // string id for component
    static std::string label() { return "exit_gotcha"; }

    // generate the gotcha wrappers
    static void configure();
    static void shutdown();

    static inline void start() {}
    static inline void stop() {}

    // exit
    void operator()(const gotcha_data&, exit_func_t, int) const;
    // abort
    void operator()(const gotcha_data&, abort_func_t) const;
};
}  // namespace component

using exit_gotcha_t = tim::component::gotcha<3, std::tuple<>, component::exit_gotcha>;
}  // namespace omnitrace
