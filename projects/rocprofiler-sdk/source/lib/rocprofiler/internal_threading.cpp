// MIT License
//
// Copyright (c) 2023 ROCm Developer Tools
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

#include <rocprofiler/fwd.h>
#include <rocprofiler/internal_threading.h>
#include <rocprofiler/rocprofiler.h>

#include "lib/common/container/stable_vector.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler/buffer.hpp"
#include "lib/rocprofiler/context/context.hpp"
#include "lib/rocprofiler/internal_threading.hpp"
#include "lib/rocprofiler/registration.hpp"

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace rocprofiler
{
namespace internal_threading
{
namespace
{
template <rocprofiler_internal_thread_library_t... Idx>
using library_sequence_t     = std::integer_sequence<rocprofiler_internal_thread_library_t, Idx...>;
using creation_notifier_cb_t = void (*)(rocprofiler_internal_thread_library_t, void*);
using thread_pool_config_t   = PTL::ThreadPool::Config;

// this is used to loop over the different libraries
constexpr auto creation_notifier_library_seq = library_sequence_t<ROCPROFILER_LIBRARY,
                                                                  ROCPROFILER_HSA_LIBRARY,
                                                                  ROCPROFILER_HIP_LIBRARY,
                                                                  ROCPROFILER_MARKER_LIBRARY>{};

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
template <rocprofiler_internal_thread_library_t LibT>
struct creation_notifier
{
    static constexpr auto value = LibT;

    std::vector<creation_notifier_cb_t> precreate_callbacks  = {};
    std::vector<creation_notifier_cb_t> postcreate_callbacks = {};
    std::vector<void*>                  user_data            = {};
    std::mutex                          mutex                = {};
};

// static accessor for creation_notifier instance
template <rocprofiler_internal_thread_library_t LibT>
auto&
get_creation_notifier()
{
    static auto _v = creation_notifier<LibT>{};
    return _v;
}

// adds callbacks to creation_notifier instance(s)
template <rocprofiler_internal_thread_library_t... Idx>
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
template <notifier_stage StageT, rocprofiler_internal_thread_library_t... Idx>
void
execute_creation_notifiers(rocprofiler_internal_thread_library_t libs,
                           std::integer_sequence<rocprofiler_internal_thread_library_t, Idx...>)
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

// using thread_pool_vec_t = std::vector<std::unique_ptr<thread_pool_t>>;
// using task_group_vec_t  = std::vector<std::unique_ptr<task_group_t>>;

auto&
get_thread_pools()
{
    static auto _v = thread_pool_vec_t{};
    return _v;
}

auto&
get_task_groups()
{
    static auto _v = task_group_vec_t([](auto& data) {
        for(auto& itr : data)
            itr.first->join();
        data.clear();
    });

    return _v;
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
        atexit(&registration::finalize);
    });
}

void
finalize()
{
    // PLT::ThreadPool::f_thread_ids() is not destruction order safe
    // if it does become safe, these two calls could be removed.
    get_task_groups().destroy();
    get_thread_pools().clear();
}

void
notify_pre_internal_thread_create(rocprofiler_internal_thread_library_t libs)
{
    execute_creation_notifiers<notifier_stage::precreation>(libs, creation_notifier_library_seq);
}

void
notify_post_internal_thread_create(rocprofiler_internal_thread_library_t libs)
{
    execute_creation_notifiers<notifier_stage::postcreation>(libs, creation_notifier_library_seq);
}

rocprofiler_callback_thread_t
create_callback_thread()
{
    // notify that rocprofiler library is about to create an inernal thread
    notify_pre_internal_thread_create(ROCPROFILER_LIBRARY);

    // this will be index after emplace_back
    auto idx = get_thread_pools().size();

    auto& thr_pool = get_thread_pools().emplace_back(std::make_shared<thread_pool_cleanup_t>(
        std::make_unique<thread_pool_t>(thread_pool_config_t{.pool_size = 1}),
        [](auto& tp) { tp->destroy_threadpool(); }));

    // construct the task group to use the newly created thread pool
    get_task_groups().get().emplace_back(std::make_unique<task_group_t>(thr_pool->get().get()),
                                         thr_pool);

    // notify that rocprofiler library finished creating an internal thread
    notify_post_internal_thread_create(ROCPROFILER_LIBRARY);

    return rocprofiler_callback_thread_t{idx};
}

// returns the task group for the given callback thread identifier
task_group_t*
get_task_group(rocprofiler_callback_thread_t cb_tid)
{
    return (!get_task_groups().get().empty())
               ? get_task_groups().get().at(cb_tid.handle).first.get()
               : nullptr;
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
    rocprofiler::internal_threading::initialize();

    auto cb_tid = rocprofiler::internal_threading::create_callback_thread();
    if(cb_tid.handle > 0)
    {
        *cb_thread_id = cb_tid;
        return ROCPROFILER_STATUS_SUCCESS;
    }

    return ROCPROFILER_STATUS_ERROR;
}

rocprofiler_status_t ROCPROFILER_API
rocprofiler_assign_callback_thread(rocprofiler_buffer_id_t       buffer_id,
                                   rocprofiler_callback_thread_t cb_thread_id)
{
    if(cb_thread_id.handle >= rocprofiler::internal_threading::get_task_groups().get().size())
        return ROCPROFILER_STATUS_ERROR_THREAD_NOT_FOUND;

    for(auto& bitr : rocprofiler::buffer::get_buffers())
    {
        if(bitr && bitr->buffer_id == buffer_id.handle)
        {
            bitr->task_group_id = cb_thread_id.handle;
            return ROCPROFILER_STATUS_SUCCESS;
        }
    }
    return ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND;
}
}
