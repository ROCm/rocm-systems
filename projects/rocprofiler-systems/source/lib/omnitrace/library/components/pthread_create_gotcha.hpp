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
#include "library/thread_data.hpp"
#include "library/timemory.hpp"

#include <cstdint>
#include <future>

namespace omnitrace
{
struct pthread_create_gotcha : tim::component::base<pthread_create_gotcha, void>
{
    using routine_t = void* (*) (void*);
    using wrappee_t = int (*)(pthread_t*, const pthread_attr_t*, routine_t, void*);

    struct wrapper
    {
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

    TIMEMORY_DEFAULT_OBJECT(pthread_create_gotcha)

    // string id for component
    static std::string label() { return "pthread_create_gotcha"; }

    // generate the gotcha wrappers
    static void configure();
    static void shutdown();
    static void shutdown(int64_t);

    // pthread_create
    int operator()(pthread_t* thread, const pthread_attr_t* attr,
                   void* (*start_routine)(void*), void*     arg) const;

    static auto& get_execution_time(int64_t _tid = threading::get_id());
    static bool  is_valid_execution_time(int64_t _tid, uint64_t _ts);

    void set_data(wrappee_t);

private:
    wrappee_t m_wrappee = &pthread_create;
};

inline auto&
pthread_create_gotcha::get_execution_time(int64_t _tid)
{
    struct omnitrace_thread_exec_time
    {};
    using data_t        = std::pair<uint64_t, uint64_t>;
    using thread_data_t = thread_data<data_t, omnitrace_thread_exec_time>;
    static auto& _v     = thread_data_t::instances(thread_data_t::construct_on_init{});
    return _v.at(_tid);
}

using pthread_create_gotcha_t =
    tim::component::gotcha<2, std::tuple<>, pthread_create_gotcha>;
}  // namespace omnitrace
