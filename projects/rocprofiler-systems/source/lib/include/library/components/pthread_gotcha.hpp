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

#include <future>

namespace omnitrace
{
struct pthread_gotcha : tim::component::base<pthread_gotcha, void>
{
    struct wrapper
    {
        using routine_t = void* (*) (void*);
        using promise_t = std::promise<void>;

        wrapper(routine_t _routine, void* _arg, bool, int64_t, promise_t*);
        void* operator()() const;

        static void* wrap(void* _arg);

    private:
        bool       m_enable_sampling = false;
        int64_t    m_parent_tid      = 0;
        routine_t  m_routine         = nullptr;
        void*      m_arg             = nullptr;
        promise_t* m_promise         = nullptr;
    };

    TIMEMORY_DEFAULT_OBJECT(pthread_gotcha)

    // string id for component
    static std::string label() { return "pthread_gotcha"; }

    // generate the gotcha wrappers
    static void configure();
    static void shutdown();

    // threads can set this to avoid starting sampling on child threads
    static bool& enable_sampling_on_child_threads();

    // pthread_create
    int operator()(pthread_t* thread, const pthread_attr_t* attr,
                   void* (*start_routine)(void*), void*     arg) const;
};

using pthread_gotcha_t = tim::component::gotcha<2, std::tuple<>, pthread_gotcha>;
}  // namespace omnitrace
