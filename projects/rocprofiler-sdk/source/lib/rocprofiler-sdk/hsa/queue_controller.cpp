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
    rocprofiler_agent_t{sizeof(rocprofiler_agent_t),
                        rocprofiler_agent_id_t{std::numeric_limits<uint64_t>::max()}};
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

QueueController&
get_queue_controller()
{
    static QueueController controller;
    return controller;
}

void
queue_controller_init(HsaApiTable* table)
{
    get_queue_controller().init(*table->core_, *table->amd_ext_);
}
}  // namespace hsa
}  // namespace rocprofiler
