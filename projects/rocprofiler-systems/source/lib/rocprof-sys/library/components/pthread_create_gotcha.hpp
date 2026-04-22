// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/common.hpp"
#include "core/defines.hpp"
#include "core/timemory.hpp"
#include "library/thread_data.hpp"

#include <cstdint>
#include <future>

namespace rocprofsys
{
struct pthread_gotcha;

namespace component
{
struct pthread_create_gotcha : tim::component::base<pthread_create_gotcha, void>
{
    static constexpr size_t gotcha_capacity = 1;

    using routine_t       = void* (*) (void*);
    using wrappee_t       = int (*)(pthread_t*, const pthread_attr_t*, routine_t, void*);
    using promise_t       = std::shared_ptr<std::promise<void>>;
    using native_handle_t = std::thread::native_handle_type;

    struct wrapper_config
    {
        bool      enable_causal   = false;
        bool      enable_sampling = false;
        bool      offset          = false;
        int64_t   parent_tid      = 0;
        promise_t promise         = {};
    };

    struct wrapper
    {
        wrapper(routine_t _routine, void* _arg, wrapper_config _cfg);
        void* operator()() const;

        static void* wrap(void* _arg);

    private:
        routine_t      m_routine = nullptr;
        void*          m_arg     = nullptr;
        wrapper_config m_config  = {};
    };

    ROCPROFSYS_DEFAULT_OBJECT(pthread_create_gotcha)

    // string id for component
    static std::string label() { return "pthread_create_gotcha"; }

    // generate the gotcha wrappers
    static void configure();
    static void shutdown();
    static void shutdown(int64_t);

    static void pause();
    static void resume();

    // pthread_create
    int operator()(pthread_t* thread, const pthread_attr_t* attr,
                   void* (*start_routine)(void*), void*     arg) const;

    void set_data(wrappee_t);

private:
    friend struct ::rocprofsys::pthread_gotcha;

    static std::set<native_handle_t> get_native_handles();

    wrappee_t         m_wrappee = &pthread_create;
    static std::mutex s_mutex;
};

using pthread_create_gotcha_t =
    tim::component::gotcha<pthread_create_gotcha::gotcha_capacity, std::tuple<>,
                           pthread_create_gotcha>;
}  // namespace component
}  // namespace rocprofsys
