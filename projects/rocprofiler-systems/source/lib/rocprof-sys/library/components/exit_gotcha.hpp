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

#include <timemory/components/base.hpp>
#include <timemory/components/gotcha/backends.hpp>

#include <cstdint>
#include <cstdlib>

namespace rocprofsys
{
namespace component
{
struct exit_gotcha : tim::component::base<exit_gotcha, void>
{
    static constexpr size_t gotcha_capacity = 4;

    using gotcha_data  = tim::component::gotcha_data;
    using exit_func_t  = void (*)(int);
    using abort_func_t = void (*)();

    ROCPROFSYS_DEFAULT_OBJECT(exit_gotcha)

    // string id for component
    static std::string label() { return "exit_gotcha"; }

    // generate the gotcha wrappers
    static void configure();
    static void shutdown();

    static inline void start() {}
    static inline void stop() {}

    // exit / _Exit / quick_exit
    void operator()(const gotcha_data&, exit_func_t, int) const;
    // abort
    void operator()(const gotcha_data&, abort_func_t) const;

    struct exit_info
    {
        bool is_known  = false;
        bool is_quick  = false;
        int  exit_code = EXIT_SUCCESS;
    };
    static exit_info get_exit_info();
};
}  // namespace component

using exit_gotcha_t = tim::component::gotcha<component::exit_gotcha::gotcha_capacity,
                                             std::tuple<>, component::exit_gotcha>;
}  // namespace rocprofsys
