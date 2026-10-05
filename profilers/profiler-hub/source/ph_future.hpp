#pragma once

#include "common/thread_pool.hpp"

#include <utility>

/**
 * @brief Definition of the opaque ph_future handle, wrapping a submitted
 *        thread_pool task.
 */
struct ph_future
{
    explicit ph_future(profiler_hub::common::thread_pool::task_handle handle)
    : m_handle{ std::move(handle) }
    {}

    profiler_hub::common::thread_pool::task_handle m_handle;
};
