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
#include "lib/rocprofiler-sdk/registration.hpp"

#include <rocprofiler-sdk/fwd.h>

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
    auto* controller = CHECK_NOTNULL(get_queue_controller());
    for(const auto& [_, agent_info] : controller->get_supported_agents())
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
                                                     controller->get_core_table(),
                                                     controller->get_ext_table(),
                                                     queue);

            controller->serializer().wlock(
                [&](auto& serializer) { serializer.add_queue(queue, *new_queue); });
            controller->add_queue(*queue, std::move(new_queue));

            return HSA_STATUS_SUCCESS;
        }
    }
    ROCP_FATAL << "Could not find agent - " << agent.handle;
    return HSA_STATUS_ERROR_FATAL;
}

hsa_status_t
destroy_queue(hsa_queue_t* hsa_queue)
{
    if(get_queue_controller()) get_queue_controller()->destroy_queue(hsa_queue);
    return HSA_STATUS_SUCCESS;
}

constexpr rocprofiler_agent_t default_agent =
    rocprofiler_agent_t{.size = sizeof(rocprofiler_agent_t),
                        .id   = rocprofiler_agent_id_t{std::numeric_limits<uint64_t>::max()},
                        .type = ROCPROFILER_AGENT_TYPE_NONE,
                        .cpu_cores_count            = 0,
                        .simd_count                 = 0,
                        .mem_banks_count            = 0,
                        .caches_count               = 0,
                        .io_links_count             = 0,
                        .cpu_core_id_base           = 0,
                        .simd_id_base               = 0,
                        .max_waves_per_simd         = 0,
                        .lds_size_in_kb             = 0,
                        .gds_size_in_kb             = 0,
                        .num_gws                    = 0,
                        .wave_front_size            = 0,
                        .num_xcc                    = 0,
                        .cu_count                   = 0,
                        .array_count                = 0,
                        .num_shader_banks           = 0,
                        .simd_arrays_per_engine     = 0,
                        .cu_per_simd_array          = 0,
                        .simd_per_cu                = 0,
                        .max_slots_scratch_cu       = 0,
                        .gfx_target_version         = 0,
                        .vendor_id                  = 0,
                        .device_id                  = 0,
                        .location_id                = 0,
                        .domain                     = 0,
                        .drm_render_minor           = 0,
                        .num_sdma_engines           = 0,
                        .num_sdma_xgmi_engines      = 0,
                        .num_sdma_queues_per_engine = 0,
                        .num_cp_queues              = 0,
                        .max_engine_clk_ccompute    = 0,
                        .max_engine_clk_fcompute    = 0,
                        .sdma_fw_version            = {},
                        .fw_version                 = {},
                        .capability                 = {},
                        .cu_per_engine              = 0,
                        .max_waves_per_cu           = 0,
                        .family_id                  = 0,
                        .workgroup_max_size         = 0,
                        .grid_max_size              = 0,
                        .local_mem_size             = 0,
                        .hive_id                    = 0,
                        .gpu_id                     = 0,
                        .workgroup_max_dim          = {0, 0, 0},
                        .grid_max_dim               = {0, 0, 0},
                        .mem_banks                  = nullptr,
                        .caches                     = nullptr,
                        .io_links                   = nullptr,
                        .name                       = nullptr,
                        .vendor_name                = nullptr,
                        .product_name               = nullptr,
                        .model_name                 = nullptr,
                        .num_pc_sampling_configs    = 0,
                        .pc_sampling_configs        = nullptr,
                        .node_id                    = 0,
                        .logical_node_id            = 0};
}  // namespace

void
QueueController::add_queue(hsa_queue_t* id, std::unique_ptr<Queue> queue)
{
    for(auto& pre_initialize_fn : pre_initialize)
        pre_initialize_fn(queue->get_agent(), get_core_table(), get_ext_table());

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
QueueController::destroy_queue(hsa_queue_t* id)
{
    if(!id) return;
    _queues.wlock([&](auto& map) {
        for(auto& deinitialize_fn : pre_deinitialize)
            if(map.find(id) != map.end())
                deinitialize_fn(map.at(id)->get_agent(), get_core_table(), get_ext_table());
    });

    const auto* queue = get_queue(*id);

    // return if queue does not exist
    if(!queue) return;

    ROCP_INFO << "destroying queue...";

    queue->sync();
    if(queue->block_signal.handle != 0) get_core_table().hsa_signal_destroy_fn(queue->block_signal);
    _queues.wlock([&](auto& map) { map.erase(id); });

    ROCP_INFO << "queue destroyed";
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
        const auto* cached_agent = agent::get_agent_cache(itr);
        if(cached_agent && cached_agent->get_rocp_agent()->type == ROCPROFILER_AGENT_TYPE_GPU)
        {
            get_supported_agents().emplace(cached_agent->index(), *cached_agent);
        }
    }

    auto enable_intercepter = false;
    for(const auto& itr : context::get_registered_contexts())
    {
        constexpr auto expected_context_size = 192UL;
        static_assert(
            sizeof(context::context) ==
                expected_context_size + sizeof(std::shared_ptr<rocprofiler::ThreadTracer>),
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
        else if(itr->thread_trace)
        {
            enable_intercepter                             = true;
            std::weak_ptr<rocprofiler::ThreadTracer> trace = itr->thread_trace;
            pre_initialize.emplace_back(
                [trace](const AgentCache& cache, const CoreApiTable& core, const AmdExtTable& ext) {
                    if(auto locked = trace.lock()) locked->resource_init(cache, core, ext);
                });
            pre_deinitialize.emplace_back(
                [trace](const AgentCache& cache, const CoreApiTable&, const AmdExtTable&) {
                    if(auto locked = trace.lock()) locked->resource_deinit(cache);
                });
            break;
        }
        else if(itr->callback_tracer)
        {
            if(itr->callback_tracer->domains(ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH))
            {
                enable_intercepter = true;
                break;
            }
        }
    }

    if(enable_intercepter)
    {
        core_table.hsa_queue_create_fn  = hsa::create_queue;
        core_table.hsa_queue_destroy_fn = hsa::destroy_queue;
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

void
QueueController::disable_serialization()
{
    _queues.rlock([](const queue_map_t& _queues_v) {
        if(get_queue_controller())
            get_queue_controller()->serializer().wlock(
                [&](auto& serializer) { serializer.disable(_queues_v); });
    });
}

void
QueueController::enable_serialization()
{
    _queues.rlock([](const queue_map_t& _queues_v) {
        if(get_queue_controller())
            get_queue_controller()->serializer().wlock(
                [&](auto& serializer) { serializer.enable(_queues_v); });
    });
}

void
QueueController::print_debug_signals() const
{
#if !defined(NDEBUG)
    _debug_signals.rlock([&](const auto& signals) {
        for(const auto& [id, signal] : signals)
        {
            ROCP_ERROR << "Signal " << signal.handle << " "
                       << get_core_table().hsa_signal_load_scacquire_fn(signal);
        }
    });
#endif

    _queues.rlock([&](const auto& queues) {
        for(const auto& [_, queue] : queues)
        {
            ROCP_ERROR << "Queue " << queue->get_id().handle << " " << queue->ready_signal.handle
                       << ":" << get_core_table().hsa_signal_load_scacquire_fn(queue->ready_signal)
                       << " " << queue->block_signal.handle << ":"
                       << get_core_table().hsa_signal_load_scacquire_fn(queue->block_signal);
        }
    });
}

void
QueueController::set_queue_state(queue_state state, hsa_queue_t* hsa_queue)
{
    _queues.wlock([&](auto& map) { map[hsa_queue]->set_state(state); });
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

void
QueueController::iterate_callbacks(const callback_iterator_cb_t& cb) const
{
    _callback_cache.rlock([&cb](const auto& map) {
        for(const auto& [cid, tuple] : map)
        {
            cb(cid, tuple);
        }
    });
}

QueueController*
get_queue_controller()
{
    static auto*& controller = common::static_object<QueueController>::construct();
    return controller;
}

void
queue_controller_init(HsaApiTable* table)
{
    CHECK_NOTNULL(get_queue_controller())->init(*table->core_, *table->amd_ext_);
}

void
queue_controller_fini()
{
    if(get_queue_controller())
        get_queue_controller()->iterate_queues([](const Queue* _queue) { _queue->sync(); });
}
}  // namespace hsa
}  // namespace rocprofiler
