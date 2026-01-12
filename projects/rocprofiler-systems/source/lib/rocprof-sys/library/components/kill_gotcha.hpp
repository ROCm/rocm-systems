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

#include "core/defines.hpp"

#include <timemory/components/base.hpp>
#include <timemory/components/gotcha/backends.hpp>

#include <csignal>
#include <sys/types.h>

namespace rocprofsys
{
namespace component
{
struct kill_gotcha : tim::component::base<kill_gotcha, void>
{
    static constexpr size_t gotcha_capacity = 1;

    using gotcha_data = tim::component::gotcha_data;
    using kill_func_t = int (*)(pid_t, int);

    ROCPROFSYS_DEFAULT_OBJECT(kill_gotcha)

    static std::string label() { return "kill_gotcha"; }

    static void configure();

    static inline void start() {}
    static inline void stop() {}

    int operator()(const gotcha_data&, kill_func_t, pid_t, int) const;
};
}  // namespace component

using kill_gotcha_t = tim::component::gotcha<component::kill_gotcha::gotcha_capacity,
                                             std::tuple<>, component::kill_gotcha>;
}  // namespace rocprofsys
