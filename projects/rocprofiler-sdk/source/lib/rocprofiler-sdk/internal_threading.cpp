// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/internal_threading.hpp"
#include "lib/common/container/stable_vector.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/internal_threading.h>

#include <pthread.h>

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace rocprofiler
{
namespace internal_threading
{
namespace
{
using task_group_vec_t     = std::vector<task_group_t*>;
using thread_pool_config_t = PTL::ThreadPool::Config;

auto affinity_functor(intmax_t)
{
    static auto assigned = std::atomic<intmax_t>{0};
    intmax_t    _assign  = assigned++;
    return _assign % std::thread::hardware_concurrency();
}

auto
get_thread_pool_config()
{
    return thread_pool_config_t{.init         = true,
                                .use_tbb      = false,
                                .use_affinity = false,
                                .verbose      = 0,
                                .priority     = 0,
                                .pool_size    = 1,
                                .task_queue   = nullptr,
                                .set_affinity = affinity_functor,
                                .initializer  = []() {},
                                .finalizer    = []() {}};
}
}  // namespace

TaskGroup::TaskGroup()
: parent_type{new thread_pool_t{get_thread_pool_config()}, false}
, m_pool{parent_type::thread_pool()}
{}

TaskGroup::~TaskGroup()
{
    m_pool->destroy_threadpool();
    delete m_pool;
}

void
TaskGroup::exec(std::function<void()>&& _func)
{
    auto lk = std::unique_lock<std::mutex>{m_mutex};
    m_tasks.emplace_back(parent_type::async(std::move(_func)));
}

void
TaskGroup::wait()
{
    auto lk = std::unique_lock<std::mutex>{m_mutex};
    for(auto& itr : m_tasks)
        itr->wait();
    // we hold the handles for the completed tasks to prevent a rare (but possible) data race on the
    // destruction of the shared_ptr
    m_completed_tasks.clear();
    // makes m_tasks empty but delays the destruction of the shared_ptrs until the next wait or the
    // destruction of the task group
    std::swap(m_tasks, m_completed_tasks);
}

void
TaskGroup::join()
{
    wait();
}

namespace
{
template <rocprofiler_runtime_library_t... Idx>
using library_sequence_t     = std::integer_sequence<rocprofiler_runtime_library_t, Idx...>;
using creation_notifier_cb_t = void (*)(rocprofiler_runtime_library_t, void*);

// this is used to loop over the different libraries
constexpr auto creation_notifier_library_seq = library_sequence_t<ROCPROFILER_LIBRARY,
                                                                  ROCPROFILER_HSA_LIBRARY,
                                                                  ROCPROFILER_HIP_LIBRARY,
                                                                  ROCPROFILER_MARKER_LIBRARY,
                                                                  ROCPROFILER_RCCL_LIBRARY,
                                                                  ROCPROFILER_ROCDECODE_LIBRARY,
                                                                  ROCPROFILER_ROCJPEG_LIBRARY>{};

// check that creation_notifier_library_seq is up to date
static_assert((1 << (creation_notifier_library_seq.size() - 1)) == ROCPROFILER_LIBRARY_LAST,
              "Update creation_notifier_library_seq to include new libraries");

// used to distinguish invoking pre vs. post at compile-time
enum class notifier_stage
{
    precreation = 0,
    postcreation,
};

// data structure holding list of callbacks
template <rocprofiler_runtime_library_t LibT>
struct creation_notifier
{
    static constexpr auto value = LibT;

    std::vector<creation_notifier_cb_t> precreate_callbacks  = {};
    std::vector<creation_notifier_cb_t> postcreate_callbacks = {};
    std::vector<void*>                  user_data            = {};
    std::mutex                          mutex                = {};
};

// static accessor for creation_notifier instance
template <rocprofiler_runtime_library_t LibT>
auto&
get_creation_notifier()
{
    static auto _v = creation_notifier<LibT>{};
    return _v;
}

// adds callbacks to creation_notifier instance(s)
template <rocprofiler_runtime_library_t... Idx>
void
update_creation_notifiers(creation_notifier_cb_t pre,
                          creation_notifier_cb_t post,
                          int                    libs,
                          void*                  data,
                          library_sequence_t<Idx...>)
{
    auto update = [pre, post, libs, data](auto& notifier) {
        if(libs == 0 || ((libs & notifier.value) == notifier.value))
        {
            notifier.mutex.lock();
            notifier.precreate_callbacks.emplace_back(pre);
            notifier.postcreate_callbacks.emplace_back(post);
            notifier.user_data.emplace_back(data);
            notifier.mutex.unlock();
        }
    };

    (update(get_creation_notifier<Idx>()), ...);
}

// invokes creation notifiers
template <notifier_stage StageT, rocprofiler_runtime_library_t... Idx>
void
execute_creation_notifiers(rocprofiler_runtime_library_t libs,
                           std::integer_sequence<rocprofiler_runtime_library_t, Idx...>)
{
    auto execute = [libs](auto& notifier) {
        if(((libs & notifier.value) == notifier.value))
        {
            notifier.mutex.lock();
            if constexpr(StageT == notifier_stage::precreation)
            {
                for(size_t i = 0; i < notifier.precreate_callbacks.size(); ++i)
                {
                    auto itr = notifier.precreate_callbacks.at(i);
                    if(itr) itr(notifier.value, notifier.user_data.at(i));
                }
            }
            else if constexpr(StageT == notifier_stage::postcreation)
            {
                for(size_t i = 0; i < notifier.postcreate_callbacks.size(); ++i)
                {
                    auto itr = notifier.postcreate_callbacks.at(i);
                    if(itr) itr(notifier.value, notifier.user_data.at(i));
                }
            }
            notifier.mutex.unlock();
        }
    };

    (execute(get_creation_notifier<Idx>()), ...);
}

auto*&
get_task_groups()
{
    static auto* _v = new task_group_vec_t{};
    return _v;
}

void
create_forked_callback_threads()
{
    if(get_task_groups())
    {
        for(auto& itr : *get_task_groups())
        {
            notify_pre_internal_thread_create(ROCPROFILER_LIBRARY);
            itr = new task_group_t{};
            notify_post_internal_thread_create(ROCPROFILER_LIBRARY);
        }
    }
}
}  // namespace

// initialize the default thread pool
void
initialize()
{
    static auto _once = std::once_flag{};
    std::call_once(_once, []() {
        // Note: create_callback_thread() must occur before atexit
        // registration or else the static objects it is pointing to
        // will be destroyed before finalize is invoked.
        create_callback_thread();
        ::atexit(&registration::finalize);
        // ensure the callback threads are created on the forked process
        ::pthread_atfork(nullptr, nullptr, create_forked_callback_threads);
    });
}

void
finalize()
{
    // PLT::ThreadPool::f_thread_ids() is not destruction order safe
    // if it does become safe, these two calls could be removed.
    if(get_task_groups())
    {
        for(auto& itr : *get_task_groups())
            itr->join();
        for(auto& itr : *get_task_groups())
            delete itr;
        get_task_groups()->clear();
        delete get_task_groups();
        get_task_groups() = nullptr;
    }
}

void
notify_pre_internal_thread_create(rocprofiler_runtime_library_t libs)
{
    execute_creation_notifiers<notifier_stage::precreation>(libs, creation_notifier_library_seq);
}

void
notify_post_internal_thread_create(rocprofiler_runtime_library_t libs)
{
    execute_creation_notifiers<notifier_stage::postcreation>(libs, creation_notifier_library_seq);
}

rocprofiler_callback_thread_t
create_callback_thread()
{
    // notify that rocprofiler library is about to create an inernal thread
    notify_pre_internal_thread_create(ROCPROFILER_LIBRARY);

    // this will be index after emplace_back
    auto idx = CHECK_NOTNULL(get_task_groups())->size();

    // construct the task group to use the newly created thread pool
    get_task_groups()->emplace_back(new task_group_t{});

    // notify that rocprofiler library finished creating an internal thread
    notify_post_internal_thread_create(ROCPROFILER_LIBRARY);

    return rocprofiler_callback_thread_t{idx};
}

// returns the task group for the given callback thread identifier
task_group_t*
get_task_group(rocprofiler_callback_thread_t cb_tid)
{
    if(!get_task_groups() || get_task_groups()->empty()) return nullptr;
    return get_task_groups()->at(cb_tid.handle);
}
}  // namespace internal_threading
}  // namespace rocprofiler

extern "C" {
rocprofiler_status_t
rocprofiler_at_internal_thread_create(rocprofiler_internal_thread_library_cb_t precreate,
                                      rocprofiler_internal_thread_library_cb_t postcreate,
                                      int                                      libs,
                                      void*                                    data)
{
    rocprofiler::internal_threading::update_creation_notifiers(
        precreate,
        postcreate,
        libs,
        data,
        rocprofiler::internal_threading::creation_notifier_library_seq);
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_create_callback_thread(rocprofiler_callback_thread_t* cb_thread_id)
{
    if(rocprofiler::registration::get_init_status() > 0)
        return ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED;

    rocprofiler::internal_threading::initialize();

    auto cb_tid = rocprofiler::internal_threading::create_callback_thread();
    if(cb_tid.handle > 0)
    {
        *cb_thread_id = cb_tid;
        return ROCPROFILER_STATUS_SUCCESS;
    }

    return ROCPROFILER_STATUS_ERROR;
}

rocprofiler_status_t
rocprofiler_assign_callback_thread(rocprofiler_buffer_id_t       buffer_id,
                                   rocprofiler_callback_thread_t cb_thread_id)
{
    if(rocprofiler::registration::get_init_status() > 0)
        return ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED;

    if(!rocprofiler::internal_threading::get_task_groups())
        return ROCPROFILER_STATUS_ERROR_THREAD_NOT_FOUND;

    if(cb_thread_id.handle >= rocprofiler::internal_threading::get_task_groups()->size())
        return ROCPROFILER_STATUS_ERROR_THREAD_NOT_FOUND;

    auto* buff_v = rocprofiler::buffer::get_buffer(buffer_id);
    if(buff_v)
    {
        buff_v->task_group_id = cb_thread_id.handle;
        return ROCPROFILER_STATUS_SUCCESS;
    }

    return ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND;
}
}
