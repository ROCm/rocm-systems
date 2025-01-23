// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/common/static_object.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <mutex>
#include <stack>

namespace rocprofiler
{
namespace common
{
namespace
{
auto*&
get_static_object_stack()
{
    static auto* _v = new std::stack<static_dtor_func_t>{};
    return _v;
}
}  // namespace

void
destroy_static_objects()
{
    static auto _sync = std::mutex{};
    auto        _lk   = std::unique_lock<std::mutex>{_sync};

    auto*& _stack = get_static_object_stack();
    if(_stack)
    {
        while(!_stack->empty())
        {
            auto& itr = _stack->top();
            if(itr) itr();
            _stack->pop();
        }

        delete _stack;
        _stack = nullptr;
    }
}

void
register_static_dtor(static_dtor_func_t&& _func)
{
    static auto _sync = std::mutex{};
    auto        _lk   = std::unique_lock<std::mutex>{_sync};

    auto*& _stack = get_static_object_stack();
    if(_stack)
    {
        _stack->push(_func);
    }
}
}  // namespace common
}  // namespace rocprofiler
