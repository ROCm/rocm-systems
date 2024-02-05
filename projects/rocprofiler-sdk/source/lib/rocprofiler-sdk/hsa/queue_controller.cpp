// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/common/static_object.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <glog/logging.h>

namespace rocprofiler
{
namespace hsa
{
namespace
{
// HSA Intercept Functions (create_queue/destroy_queue)
hsa_status_t
create_queue(hsa_agent_t        agent,
             uint32_t           size,
             hsa_queue_type32_t type,
             void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data),
             void*         data,
             uint32_t      private_segment_size,
             uint32_t      group_segment_size,
             hsa_queue_t** queue)
{
    for(const auto& [_, agent_info] : get_queue_controller().get_supported_agents())
    {
        if(agent_info.get_hsa_agent().handle == agent.handle)
        {
            auto new_queue = std::make_unique<Queue>(agent_info,
                                                     size,
                                                     type,
                                                     callback,
                                                     data,
                                                     private_segment_size,
                                                     group_segment_size,
                                                     get_queue_controller().get_core_table(),
                                                     get_queue_controller().get_ext_table(),
                                                     queue);

            get_queue_controller().profiler_serializer_register_ready_signal_handler(
                new_queue->ready_signal, *queue);
            get_queue_controller().add_queue(*queue, std::move(new_queue));

            return HSA_STATUS_SUCCESS;
        }
    }
    LOG(FATAL) << "Could not find agent - " << agent.handle;
    return HSA_STATUS_ERROR_FATAL;
}

hsa_status_t
destroy_queue(hsa_queue_t* hsa_queue)
{
    get_queue_controller().destory_queue(hsa_queue);
    return HSA_STATUS_SUCCESS;
}

constexpr rocprofiler_agent_t default_agent =
    rocprofiler_agent_t{.size = sizeof(rocprofiler_agent_t),
                        .id   = rocprofiler_agent_id_t{std::numeric_limits<uint64_t>::max()}};
}  // namespace

void
QueueController::add_queue(hsa_queue_t* id, std::unique_ptr<Queue> queue)
{
    CHECK(queue);
    _callback_cache.wlock([&](auto& callbacks) {
        _queues.wlock([&](auto& map) {
            const auto agent_id = queue->get_agent().get_rocp_agent()->id.handle;
            map[id]             = std::move(queue);
            for(const auto& [cbid, cb_tuple] : callbacks)
            {
                auto& [agent, qcb, ccb] = cb_tuple;
                if(agent.id.handle == default_agent.id.handle || agent.id.handle == agent_id)
                {
                    map[id]->register_callback(cbid, qcb, ccb);
                }
            }
        });
    });
}

void
QueueController::destory_queue(hsa_queue_t* id)
{
    const auto*                  queue = get_queue_controller().get_queue(*id);
    std::unique_lock<std::mutex> cvlock(queue->cv_mutex);
    profiler_serializer([&](auto& data) {
        /*Deletes the queue to be destructed from the dispatch ready.*/
        data.dispatch_ready.erase(
            std::remove_if(
                data.dispatch_ready.begin(),
                data.dispatch_ready.end(),
                [&](auto& it) {
                    /*Deletes the queue to be destructed from the dispatch ready.*/
                    if(it->get_id().handle == queue->get_id().handle)
                    {
                        if(data.dispatch_queue &&
                           data.dispatch_queue->get_id().handle == queue->get_id().handle)
                        {
                            // insert fatal condition here
                            // ToDO [srnagara]: Need to find a solution rather than abort.
                            LOG(FATAL)
                                << "Queue is being destroyed while kernel launch is still active";
                        }
                        return true;
                    }
                    return false;
                }),
            data.dispatch_ready.end());
        set_queue_state(queue_state::to_destroy, id);
        /*
          This lambda triggers the async ready handler.
          The async ready handler then unregisters itself
          and sets the queue state to done_destroy for which
          the condition variable here is waiting for.
        */
        auto trigger_ready_async_handler = [queue]() {
            get_queue_controller().get_core_table().hsa_signal_store_screlease_fn(
                queue->ready_signal, 0);
        };
        trigger_ready_async_handler();
    });
    queue->cv_ready_signal.wait(
        cvlock, [queue] { return queue->get_state() == queue_state::done_destroy; });
    if(queue->block_signal.handle != 0)
        get_queue_controller().get_core_table().hsa_signal_destroy_fn(queue->block_signal);
    _queues.wlock([&](auto& map) { map.erase(id); });
}

ClientID
QueueController::add_callback(std::optional<rocprofiler_agent_t> agent,
                              Queue::queue_cb_t                  qcb,
                              Queue::completed_cb_t              ccb)
{
    static std::atomic<ClientID> client_id = 1;
    ClientID                     return_id;
    _callback_cache.wlock([&](auto& cb_cache) {
        return_id = client_id;
        if(agent)
        {
            cb_cache[client_id] = std::tuple(*agent, qcb, ccb);
        }
        else
        {
            cb_cache[client_id] = std::tuple(default_agent, qcb, ccb);
        }
        client_id++;

        _queues.wlock([&](auto& map) {
            for(auto& [_, queue] : map)
            {
                if(!agent || queue->get_agent().get_rocp_agent()->id.handle == agent->id.handle)
                {
                    queue->register_callback(return_id, qcb, ccb);
                }
            }
        });
    });
    return return_id;
}

void
QueueController::remove_callback(ClientID id)
{
    _callback_cache.wlock([&](auto& cb_cache) {
        cb_cache.erase(id);
        _queues.wlock([&](auto& map) {
            for(auto& [_, queue] : map)
            {
                queue->remove_callback(id);
            }
        });
    });
}

void
QueueController::init(CoreApiTable& core_table, AmdExtTable& ext_table)
{
    _core_table = core_table;
    _ext_table  = ext_table;

    auto agents = agent::get_agents();

    // Generate supported agents
    for(const auto* itr : agents)
    {
        auto cached_agent = agent::get_agent_cache(itr);
        if(cached_agent && cached_agent->get_rocp_agent()->type == ROCPROFILER_AGENT_TYPE_GPU)
        {
            get_supported_agents().emplace(cached_agent->index(), *cached_agent);
        }
    }

    auto enable_intercepter = false;
    for(const auto& itr : context::get_registered_contexts())
    {
        constexpr auto expected_context_size = 160UL;
        static_assert(
            sizeof(context::context) == expected_context_size,
            "If you added a new field to context struct, make sure there is a check here if it "
            "requires queue interception. Once you have done so, increment expected_context_size");

        if(itr->counter_collection)
        {
            enable_intercepter = true;
            break;
        }
        else if(itr->buffered_tracer)
        {
            if(itr->buffered_tracer->domains(ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH))
            {
                enable_intercepter = true;
                break;
            }
        }
    }

    if(enable_intercepter)
    {
        core_table.hsa_queue_create_fn  = create_queue;
        core_table.hsa_queue_destroy_fn = destroy_queue;
    }
}

const Queue*
QueueController::get_queue(const hsa_queue_t& _hsa_queue) const
{
    return _queues.rlock(
        [](const queue_map_t& _data, const hsa_queue_t& _inp) -> const Queue* {
            for(const auto& itr : _data)
            {
                if(itr.first->id == _inp.id) return itr.second.get();
            }
            return nullptr;
        },
        _hsa_queue);
}

template <typename FuncT>
void
QueueController::profiler_serializer(FuncT&& lambda)
{
    _profiler_serializer.wlock(std::forward<FuncT>(lambda));
}

namespace
{
/*
    Function name:  AsyncSignalReadyHandler
    Argument:    hsa signal value for which the async handler was called
                 and pointer to the data.
    Description: This async handler is invoked when the queue is ready
                 to launch a kernel. It first, resets the queue's ready signal to 1.
                 It then checks if there is any queue which has a kernel currently dispatched.
                 If yes, it pushes the queue to the dispatch ready else
                 it enables the dispatch for the given queue.
    Return :     It returns true since we need this handler to be invoked
                 each time the queue's ready signal (used for entire queue) is set to 0.
                 If we had a separate signal for every dispatch in the queue then we don't
                 need this to be invoked more than once in which case we would return false.
*/
bool
profiler_serializer_ready_signal_handler(hsa_signal_value_t /* signal_value */, void* data)
{
    auto*       hsa_queue = static_cast<hsa_queue_t*>(data);
    const auto* queue     = get_queue_controller().get_queue(*hsa_queue);
    get_queue_controller().profiler_serializer([&](auto& serializer) {
        {
            std::lock_guard<std::mutex> cv_lock(queue->cv_mutex);
            if(queue->get_state() == queue_state::to_destroy)
            {
                get_queue_controller().set_queue_state(queue_state::done_destroy, hsa_queue);
                get_queue_controller().get_core_table().hsa_signal_destroy_fn(queue->ready_signal);
                queue->cv_ready_signal.notify_one();
                return;
            }
        }
        get_queue_controller().get_core_table().hsa_signal_store_screlease_fn(queue->ready_signal,
                                                                              1);
        if(serializer.dispatch_queue == nullptr)
        {
            get_queue_controller().get_core_table().hsa_signal_store_screlease_fn(
                queue->block_signal, 0);
            serializer.dispatch_queue = queue;
        }
        else
        {
            serializer.dispatch_ready.push_back(queue);
        }
    });
    return true;
}
}  // namespace

void
profiler_serializer_kernel_completion_signal(hsa_signal_t queue_block_signal)
{
    get_queue_controller().profiler_serializer([queue_block_signal](auto& serializer) {
        assert(serializer.dispatch_queue != nullptr);
        serializer.dispatch_queue = nullptr;
        get_queue_controller().get_core_table().hsa_signal_store_screlease_fn(queue_block_signal,
                                                                              1);
        if(!serializer.dispatch_ready.empty())
        {
            auto queue = serializer.dispatch_ready.front();
            serializer.dispatch_ready.erase(serializer.dispatch_ready.begin());
            get_queue_controller().get_core_table().hsa_signal_store_screlease_fn(
                queue->block_signal, 0);
            serializer.dispatch_queue = queue;
        }
    });
}

void
QueueController::set_queue_state(enum queue_state state, hsa_queue_t* hsa_queue)
{
    _queues.wlock([&](auto& map) { map[hsa_queue]->set_state(state); });
}

/*
    Function name:  SignalAsyncReadyHandler.
    Argument :      The signal value and pointer to the data to
                    pass to the handler.
    Description :   Registers a asynchronous callback function
                    for the ready signal to be invoked when it goes to zero.
*/
void
QueueController::profiler_serializer_register_ready_signal_handler(const hsa_signal_t& signal,
                                                                   void*               data) const
{
    hsa_status_t status = get_ext_table().hsa_amd_signal_async_handler_fn(
        signal, HSA_SIGNAL_CONDITION_EQ, 0, profiler_serializer_ready_signal_handler, data);
    if(status != HSA_STATUS_SUCCESS) LOG(FATAL) << "hsa_amd_signal_async_handler failed";
}

void
QueueController::iterate_queues(const queue_iterator_cb_t& cb) const
{
    _queues.rlock([&cb](const queue_map_t& _queues_v) {
        for(const auto& itr : _queues_v)
        {
            if(itr.second) cb(itr.second.get());
        }
    });
}

QueueController&
get_queue_controller()
{
    static auto*& controller = common::static_object<QueueController>::construct();
    return *(CHECK_NOTNULL(controller));
}

void
queue_controller_init(HsaApiTable* table)
{
    get_queue_controller().init(*table->core_, *table->amd_ext_);
}

void
queue_controller_fini()
{
    get_queue_controller().iterate_queues([](const Queue* _queue) { _queue->sync(); });
}
}  // namespace hsa
}  // namespace rocprofiler
