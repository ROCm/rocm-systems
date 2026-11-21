// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include "lib/common/static_tl_object.hpp"

#include <pthread.h>
#include <stack>

namespace rocprofiler
{
namespace common
{
namespace
{
using static_tl_dtor_stack_t = std::stack<static_dtor_func_t>;

auto*&
get_static_tl_object_stack()
{
    static thread_local static_tl_dtor_stack_t* _v = nullptr;
    return _v;
}

struct static_tl_object_key
{
    pthread_key_t key   = {};
    bool          valid = false;
};

// Thread-exit cleanup runs from a pthread key destructor, not from a thread_local with a
// destructor: the first use of such a thread_local on a thread calls __cxa_thread_atexit, which
// takes glibc's dl_load_lock. That first use can happen inside a HIP or HSA call (or a ROCr
// internal thread) that holds a runtime lock or waits on the GPU, while dlopen holds
// dl_load_lock and runs a library constructor that needs the same runtime.
const static_tl_object_key&
get_static_tl_object_key()
{
    static const auto _v = []() {
        auto _ret = static_tl_object_key{};
        _ret.valid =
            (pthread_key_create(&_ret.key, [](void*) { destroy_static_tl_objects(); }) == 0);
        return _ret;
    }();
    return _v;
}
}  // namespace

void
destroy_static_tl_objects()
{
    auto*& _stack = get_static_tl_object_stack();
    if(!_stack) return;

    while(!_stack->empty())
    {
        auto _func = _stack->top();
        _stack->pop();
        if(_func) _func();
    }

    delete _stack;
    _stack = nullptr;

    if(const auto& _key = get_static_tl_object_key(); _key.valid)
        pthread_setspecific(_key.key, nullptr);
}

void
register_static_tl_dtor(static_dtor_func_t&& _func)
{
    auto*& _stack = get_static_tl_object_stack();
    if(!_stack)
    {
        _stack = new static_tl_dtor_stack_t{};
        // a non-null value is what makes the key's destructor run when this thread exits
        if(const auto& _key = get_static_tl_object_key(); _key.valid)
            pthread_setspecific(_key.key, _stack);
    }

    _stack->push(_func);
}
}  // namespace common
}  // namespace rocprofiler
